"""The figures stage on disk: which chart readings are stored, which are shown, and reading them anew.

Kept apart from :mod:`paperfacts.figures` on purpose: that module's source is hashed into ``figure_key``, and
where a document's readings are stored is not what a vision model was asked. It is orchestration of one
stage, so :mod:`paperfacts.workflow` calls it and marks the stage.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict

from paperfacts.config import Settings
from paperfacts.figures import FigureReading, FigureReadings, read_figures
from paperfacts.keys import figure_key_for
from paperfacts.llm import VisionClient
from paperfacts.models import BACKENDS, Backend, DocumentInput, NormalizedBBox, ParsedArtifact
from paperfacts.pdf import crop_region, png_bytes, render_page
from paperfacts.profile import DomainProfile
from paperfacts.storage import DataLayout

logger = logging.getLogger(__name__)


# ---- What a reader is shown ----------------------------------------------------------------------------

Cell = str | float | int | bool | None


class FiguresView(BaseModel):
    """The readings a document page and a workbook show, with what they should warn about.

    ``stale``: no file exists under the current figure_key, so these were read under older settings (another
    model, prompt or field table). ``orphaned``: figure blocks the readings cite that the current parse no
    longer has; the numbers stand, but clicking one cannot point at the chart.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str
    figure_key: str
    model: str
    stale: bool = False
    orphaned: tuple[str, ...] = ()
    rows: tuple[dict[str, Cell], ...] = ()

    def warning(self) -> str:
        notes = ["stale (older key)"] if self.stale else []
        if self.orphaned:
            notes.append(f"{len(self.orphaned)} cite figure blocks missing from the current parse")
        return ", ".join(notes)


def _x_text(quantity: str | None, value: float | str | None, unit: str | None, on_tick: bool | None) -> str | None:
    if value is None:
        return None
    number = f"{value:g}" if isinstance(value, float) else str(value)
    text = " ".join(part for part in (f"{quantity} =" if quantity else None, number, unit) if part)
    return text + ("（刻度之间，插值）" if on_tick is False else "")


def figure_rows(
    readings: FigureReadings, *, filename: str, stale: bool = False, orphaned: frozenset[str] = frozenset()
) -> tuple[dict[str, Cell], ...]:
    """One display row per reading, keyed like the 图中读数 sheet's columns in :mod:`paperfacts.workbook`."""

    def detail(reading: FigureReading) -> str | None:
        notes = [reading.note] if reading.note else []
        if stale:
            notes.append("旧版本读数（设置已变，尚未重读）")
        if reading.source_id in orphaned:
            notes.append("当前解析里已没有这个图块")
        return "; ".join(notes) or None

    return tuple(
        {
            "document_id": readings.document_id,
            "filename": filename,
            "figure": reading.figure,
            "page": reading.page + 1,
            "source_id": reading.source_id,
            "panel": reading.panel,
            "field": reading.field,
            "series": reading.series,
            "x": _x_text(reading.x_quantity, reading.x_value, reading.x_unit, reading.x_on_tick),
            "value": reading.y,
            "unit": reading.unit,
            "precision": f"±{reading.precision * 100:g}%",
            "value_raw": " ".join(part for part in (f"{reading.y_raw:g}", reading.y_unit_raw) if part),
            "scale": "对数" if reading.scale == "log" else "线性",
            "caption": reading.caption,
            "detail": detail(reading),
        }
        for reading in readings.readings
    )


def figure_artifact(
    document: DocumentInput, settings: Settings, parsed: Mapping[Backend, ParsedArtifact | None] | None = None
) -> ParsedArtifact:
    """Whose figure blocks are cropped: MinerU's, else PaddleOCR-VL's. Both parsers box the same chart, so
    one is enough, and a fixed preference keeps the citations of a document stable run to run. An artifact
    the caller already holds is used as is; otherwise it is read from disk."""
    for backend in BACKENDS:
        artifact = (parsed or {}).get(backend)
        if artifact is not None:
            return artifact
        path = DataLayout(settings.data_root).artifact_path(document.document_id, backend)
        if path.is_file():
            return ParsedArtifact.read(path)
    raise FileNotFoundError(f"no parse artifact for {document.display_filename}; run `paperfacts parse` first")


# Every file stored before profiles existed was read under the TCO field table.
LEGACY_PROFILE = "tco"


class StoredReadings(FigureReadings):
    """Readings as stored: with the profile they were read under, so the stale fallback of another profile
    over the same data root never shows them. Declared here, not in ``figures.py``, whose source is hashed
    into figure_key; None on files stored before the field existed.

    The name is all that is recorded. The profile's figure material is already in ``figure_key``, so a file
    under the current key cannot be of another version of it; and the stale fallback exists for exactly the
    files whose material differs (a field table edited since), so a fingerprint check there would hide every
    reading it is meant to show. Files that still carry a ``profile_fingerprint`` load as before."""

    profile: str | None = None


def stored_figures_path(layout: DataLayout, document_id: str, figure_key: str, profile: str) -> Path | None:
    """The stored readings file of ``profile`` under ``figure_key``, if there is one: its own directory first,
    then, for the TCO profile only, the flat file every reading was stored in before the directories."""
    candidates = [layout.figures_path(document_id, figure_key, profile)]
    if profile == LEGACY_PROFILE:
        candidates.append(layout.legacy_figures_path(document_id, figure_key))
    return next((path for path in candidates if path.is_file()), None)


def _older_files(layout: DataLayout, document_id: str, profile: str, current: frozenset[Path]) -> list[Path]:
    """The profile's readings files under other keys, newest first: its own directory, plus the flat files
    for the TCO profile."""
    directories = [layout.figures_dir(document_id, profile)]
    if profile == LEGACY_PROFILE:
        directories.append(layout.legacy_figures_dir(document_id))
    paths = [path for directory in directories if directory.is_dir() for path in directory.glob("*.json")]
    return sorted((path for path in paths if path not in current), key=lambda path: path.stat().st_mtime, reverse=True)


def migrate_legacy_figures(layout: DataLayout, *, apply: bool) -> list[tuple[Path, Path, str]]:
    """Move every flat ``figures/<figure_key>.json`` into its profile's directory, as (source, destination,
    outcome) per file: the profile the file records, else the TCO profile every flat file was read under, which
    is then stamped on it. A destination that already exists wins (the directory is where every newer reading
    went) and the flat file is left for a person to look at. Without ``apply`` nothing is written.

    Until every data root has been migrated, the TCO profile still reads the flat files as a fallback."""
    moves: list[tuple[Path, Path, str]] = []
    docs = layout.docs_root()
    for doc_dir in sorted(docs.iterdir() if docs.is_dir() else ()):
        flat = layout.legacy_figures_dir(doc_dir.name)
        # Dotfiles are skipped: macOS copies leave "._<name>.json" AppleDouble files beside the real ones.
        for source in sorted(path for path in flat.glob("*.json") if not path.name.startswith(".")):
            readings = _stored_file(source)
            if readings is None:
                moves.append((source, source, "unreadable, left in place"))
                continue
            profile = readings.profile or LEGACY_PROFILE
            destination = layout.figures_path(doc_dir.name, source.stem, profile)
            if destination.exists():
                moves.append((source, destination, "destination exists, left in place"))
                continue
            if apply:
                readings.model_copy(update={"profile": profile}).write(destination)
                source.unlink()
            moves.append((source, destination, "moved" if apply else "would move"))
    return moves


def _belongs(readings: StoredReadings, profile: DomainProfile) -> bool:
    return (readings.profile or LEGACY_PROFILE) == profile.name


def _stored_file(path: Path) -> StoredReadings | None:
    """A stored readings file, or None when there is none or it will not load (it is then read again)."""
    if not path.is_file():
        return None
    try:
        return StoredReadings.read(path)
    except (OSError, ValueError) as exc:
        logger.warning("stored figure readings at %s are unreadable (%s); ignoring them", path, exc)
        return None


def _own_file(path: Path, profile: DomainProfile) -> StoredReadings | None:
    """A stored file only if ``profile`` read it: a flat file of before the per-profile directories may have been
    written by another profile sharing the figure_key."""
    readings = _stored_file(path)
    return readings if readings is not None and _belongs(readings, profile) else None


def _orphaned(readings: FigureReadings, artifact: ParsedArtifact | None) -> frozenset[str]:
    """The figure blocks the readings cite that ``artifact`` does not have at the same place. Without an
    artifact nothing can be checked, and nothing is claimed missing."""
    if artifact is None:
        return frozenset()
    boxes = {block.source_id: block.bbox for block in artifact.blocks if block.type == "figure"}
    cited = [(panel.source_id, None) for panel in readings.panels]
    cited += [(reading.source_id, reading.bbox) for reading in readings.readings]
    return frozenset(
        source_id
        for source_id, bbox in cited
        if source_id not in boxes or (bbox is not None and boxes[source_id] != bbox)
    )


def shown_figures(document_id: str, filename: str, settings: Settings, profile: DomainProfile) -> FiguresView | None:
    """The chart readings to show for a document, whether or not the stage is switched on.

    Switching the stage off stops the asking, not the showing. When nothing is stored under the current
    figure_key (the model, the prompt or the field table moved since), the newest older file of the same
    profile stands in and is marked stale rather than hiding readings that were paid for. Readings citing
    figure blocks the current parse no longer has are marked too, and readings of a field the profile does
    not have are left out.
    """
    layout = DataLayout(settings.data_root)
    key = figure_key_for(settings, profile)
    current = stored_figures_path(layout, document_id, key, profile.name)
    readings, stale = (None if current is None else _own_file(current, profile)), False
    if readings is None:
        keyed = frozenset(
            {layout.figures_path(document_id, key, profile.name), layout.legacy_figures_path(document_id, key)}
        )
        for path in _older_files(layout, document_id, profile.name, keyed):
            readings = _stored_file(path)
            if readings is not None and _belongs(readings, profile):
                stale = True
                break
            readings = None
    if readings is None:
        return None
    readings = readings.model_copy(
        update={"readings": tuple(reading for reading in readings.readings if reading.field in profile.by_name)}
    )
    artifact = None
    if readings.backend is not None and layout.artifact_path(document_id, readings.backend).is_file():
        artifact = ParsedArtifact.read(layout.artifact_path(document_id, readings.backend))
    orphaned = _orphaned(readings, artifact)
    return FiguresView(
        document_id=readings.document_id,
        figure_key=readings.figure_key,
        model=readings.model,
        stale=stale,
        orphaned=tuple(sorted(orphaned)),
        rows=figure_rows(readings, filename=filename, stale=stale, orphaned=orphaned),
    )


def read_document_figures(
    document: DocumentInput,
    settings: Settings,
    profile: DomainProfile,
    client: VisionClient,
    *,
    force: bool = False,
    artifact: ParsedArtifact | None = None,
    stop: threading.Event | None = None,
) -> FigureReadings:
    """Read the charts of one document, or return the stored readings.

    Stored readings are served only when complete and still citing figure blocks of the current parse at
    the same place. Otherwise the charts are read again: answered panels replay from the LLM cache for free,
    a panel whose cached answer was unusable is asked with the cache bypassed, and a failed request is simply
    asked again. ``force`` re-asks every panel.
    """
    key = figure_key_for(settings, profile)
    layout = DataLayout(settings.data_root)
    path = layout.figures_path(document.document_id, key, profile.name)
    artifact = artifact or figure_artifact(document, settings)
    stored = None if force else stored_figures_path(layout, document.document_id, key, profile.name)
    previous = None if stored is None else _own_file(stored, profile)
    if (
        previous is not None
        and previous.complete
        and previous.backend == artifact.backend
        and not _orphaned(previous, artifact)
    ):
        logger.info("figures cache_hit doc=%s", document.document_id[:16])
        return previous
    if not document.pdf_path.is_file():
        raise FileNotFoundError("PDF not available; figure reading crops the charts from it, re-upload to read them")

    # Panels of one figure share a page; rendering it once per page rather than once per panel saves a
    # 200-dpi render (and a turn at the PDFium lock) for every panel after the first.
    pages: dict[int, Image.Image] = {}

    def render(page: int, bbox: NormalizedBBox) -> bytes:
        if page not in pages:
            pages[page] = render_page(document.pdf_path, page, dpi=settings.figures_dpi)
        return png_bytes(crop_region(pages[page], bbox, max_pixels=settings.figures_max_pixels))

    readings = read_figures(
        artifact,
        render,
        client,
        profile,
        figure_key=key,
        max_per_document=settings.figures_max_per_document,
        concurrency=settings.llm_concurrency,
        refresh=force,
        refresh_panels=previous.unreadable() if previous is not None else frozenset(),
        stop=stop,
    )
    StoredReadings(**dict(readings), profile=profile.name).write(path)
    return readings
