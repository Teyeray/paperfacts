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

from paperfacts.config import Settings
from paperfacts.figures import FigureReadings, FiguresView, figure_rows, read_figures
from paperfacts.keys import figure_key_for
from paperfacts.llm import VisionClient
from paperfacts.models import BACKENDS, Backend, DocumentInput, NormalizedBBox, ParsedArtifact
from paperfacts.pdf import crop_region, png_bytes, render_page
from paperfacts.storage import DataLayout

logger = logging.getLogger(__name__)


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


def _stored_file(path: Path) -> FigureReadings | None:
    """A stored readings file, or None when there is none or it will not load (it is then read again)."""
    if not path.is_file():
        return None
    try:
        return FigureReadings.read(path)
    except (OSError, ValueError) as exc:
        logger.warning("stored figure readings at %s are unreadable (%s); ignoring them", path, exc)
        return None


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


def shown_figures(document_id: str, filename: str, settings: Settings) -> FiguresView | None:
    """The chart readings to show for a document, whether or not the stage is switched on.

    Switching the stage off stops the asking, not the showing. When nothing is stored under the current
    figure_key (the model, the prompt or the field table moved since), the newest older file stands in and
    is marked stale rather than hiding readings that were paid for. Readings citing figure blocks the
    current parse no longer has are marked too.
    """
    layout = DataLayout(settings.data_root)
    current = layout.figures_path(document_id, figure_key_for(settings))
    readings, stale = _stored_file(current), False
    if readings is None and current.parent.is_dir():
        older = sorted(
            (path for path in current.parent.glob("*.json") if path != current),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in older:
            readings = _stored_file(path)
            if readings is not None:
                stale = True
                break
    if readings is None:
        return None
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
    key = figure_key_for(settings)
    path = DataLayout(settings.data_root).figures_path(document.document_id, key)
    artifact = artifact or figure_artifact(document, settings)
    previous = None if force else _stored_file(path)
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
        figure_key=key,
        max_per_document=settings.figures_max_per_document,
        concurrency=settings.llm_concurrency,
        refresh=force,
        refresh_panels=previous.unreadable() if previous is not None else frozenset(),
        stop=stop,
    )
    readings.write(path)
    return readings
