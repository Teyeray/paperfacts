"""What is stored for a document, and whether it is still current: the readers of the files the stages write.

Separate from :mod:`paperfacts.workflow`, which writes them, because these are asked by readers that run no
stage -- the web library, "run all", ``deploy.sh --rerun``. A stored result is current only when it was built
under the caller's keys **and** from the parse now on disk: source ids are positional, so a result of an
earlier parse would cite whatever block now holds that ordinal. Not part of any cache key.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from paperfacts.compare import ComparisonReport
from paperfacts.dataset import DatasetPayload
from paperfacts.models import BACKENDS, Backend, ParsedArtifact
from paperfacts.records import LaneExtraction
from paperfacts.storage import DataLayout
from paperfacts.workflow import Stage, StageStatus, stage_names


def stored_comparison(
    layout: DataLayout, document_id: str, extractor_key: str, comparison_key: str
) -> ComparisonReport | None:
    """The stored comparison under these keys, or None when there is none or it compared other parses.

    For readers that hold no lanes (the web library): a report whose recorded artifact hashes differ from
    the artifacts on disk would show citations into blocks the current parse does not have, and would keep
    the document counted as compared, so "run all" would never redo it.
    """
    path = layout.comparison_path(document_id, extractor_key, comparison_key)
    if not path.is_file():
        return None
    report = ComparisonReport.read(path)
    recorded = {report.backend_a: report.artifact_sha256_a, report.backend_b: report.artifact_sha256_b}
    return report if _of_current_parse(layout, document_id, recorded) else None


def stored_dataset(
    layout: DataLayout, document_id: str, extractor_key: str, comparison_key: str
) -> DatasetPayload | None:
    """The stored consolidated table under these keys, or None when there is none or it came from other
    parses -- the same rule as :func:`stored_comparison`, for the same reasons. A file in the wrong shape
    raises ``ValidationError``: the caller decides whether that costs one row or the request."""
    path = layout.dataset_json_path(document_id, extractor_key, comparison_key)
    if not path.is_file():
        return None
    dataset = DatasetPayload.model_validate_json(path.read_text(encoding="utf-8"))
    return dataset if _of_current_parse(layout, document_id, dataset.artifact_sha256) else None


def _of_current_parse(layout: DataLayout, document_id: str, recorded: Mapping[Backend, str | None]) -> bool:
    """Whether every recorded artifact hash is that of the artifact on disk. A hash not recorded, or an
    artifact no longer there, is unknown rather than a mismatch, so files from before hashes were kept
    still read."""
    for backend, sha in recorded.items():
        if sha is None:
            continue
        current = _artifact_hash(layout.artifact_path(document_id, backend))
        if current is not None and current != sha:
            return False
    return True


# (reader, path) -> (stamp, what the reader derived). The library lists every document's stages and checks
# every stored table against its parse on each refresh, and each check means parsing a whole file. The stamp
# includes the inode because every stored file is replaced by a rename, so a rewrite always changes it.
_DERIVED: dict[tuple[str, Path], tuple[tuple[int, int, int], object]] = {}
_DERIVED_LOCK = threading.Lock()


def _by_stamp[T](path: Path, derive: Callable[[Path], T]) -> T | None:
    """``derive(path)``, recomputed only when the file changed; None when there is no file."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    key = (derive.__name__, path)
    with _DERIVED_LOCK:
        cached = _DERIVED.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]  # type: ignore[return-value]
    value = derive(path)
    with _DERIVED_LOCK:
        _DERIVED[key] = (stamp, value)
    return value


def _content_hash(path: Path) -> str:
    return ParsedArtifact.read(path).content_hash()


def _artifact_hash(path: Path) -> str | None:
    return _by_stamp(path, _content_hash)


def _report_hashes(path: Path) -> dict[Backend, str | None]:
    report = ComparisonReport.read(path)
    return {report.backend_a: report.artifact_sha256_a, report.backend_b: report.artifact_sha256_b}


def _dataset_hashes(path: Path) -> dict[Backend, str]:
    return dict(DatasetPayload.model_validate_json(path.read_text(encoding="utf-8")).artifact_sha256)


def _unanswered_fields(path: Path) -> tuple[str, ...]:
    return tuple(question.field for question in LaneExtraction.read(path).failed_questions)


def _current(layout: DataLayout, document_id: str, path: Path, hashes: Callable[[Path], Mapping]) -> bool:
    """Whether the stored file exists and was built from the parse on disk, read cheaply for a listing."""
    try:
        recorded = _by_stamp(path, hashes)
    except (OSError, ValueError):  # pydantic's ValidationError is a ValueError: an unreadable file is not done
        return False
    return recorded is not None and _of_current_parse(layout, document_id, recorded)


def stored_stages(
    layout: DataLayout,
    document_id: str,
    *,
    extractor_key: str,
    comparison_key: str,
    figure_key: str,
    figures_enabled: bool,
) -> tuple[Stage, ...]:
    """How far a stored document got, one entry per :func:`stage_names` stage, read off the file each stage
    writes under these keys. It is the progress to show when no running job describes the document. The
    keys are the caller's, so the library that lists documents and the one that reads them cannot disagree.

    A comparison or table of another parse is pending, by the rule :func:`stored_comparison` and
    :func:`is_finished` apply. A lane with a field question the model never answered validly is failed and
    names the fields: otherwise, after the job is gone, nothing on the page says why the paper stays unfinished.
    """

    def done(path: Path) -> StageStatus:
        return "done" if path.is_file() else "pending"

    def current(path: Path, hashes: Callable[[Path], Mapping]) -> StageStatus:
        return "done" if _current(layout, document_id, path, hashes) else "pending"

    figures = done(layout.figures_path(document_id, figure_key))
    stages: dict[str, Stage] = {
        **{f"parse:{b}": Stage(name=f"parse:{b}", status=done(layout.artifact_path(document_id, b))) for b in BACKENDS},
        # Opt-in: switched off and never read is a skip, not work still to do.
        "figures": Stage(name="figures", status=figures if figures == "done" or figures_enabled else "skipped"),
        **{f"extract:{b}": _extract_stage(layout.extraction_path(document_id, b, extractor_key), b) for b in BACKENDS},
        "compare": Stage(
            name="compare",
            status=current(layout.comparison_path(document_id, extractor_key, comparison_key), _report_hashes),
        ),
        "export": Stage(
            name="export",
            status=current(layout.dataset_json_path(document_id, extractor_key, comparison_key), _dataset_hashes),
        ),
    }
    return tuple(stages[name] for name in stage_names())


def _extract_stage(path: Path, backend: Backend) -> Stage:
    name = f"extract:{backend}"
    try:
        unanswered = _by_stamp(path, _unanswered_fields)
    except (OSError, ValueError):
        unanswered = ()  # the file is there; what it holds is for the lane view to report
    if unanswered is None:
        return Stage(name=name, status="pending")
    if unanswered:
        detail = f"no valid answer to {', '.join(unanswered)}; asked again on the next run"
        return Stage(name=name, status="failed", detail=detail)
    return Stage(name=name, status="done")


def is_finished(layout: DataLayout, document_id: str, *, extractor_key: str, comparison_key: str) -> bool:
    """Whether a run under these keys went all the way. The export is the last stage (see
    :func:`stored_stages`), so a document whose comparison exists but whose export failed is still unfinished,
    and so is one whose run was not kept (``workflow.run_document`` stores no dataset for it). A dataset of
    other parses (:func:`stored_dataset`) or one that cannot be read is unfinished too: running the paper
    again is what replaces it."""
    try:
        return stored_dataset(layout, document_id, extractor_key, comparison_key) is not None
    except (OSError, ValidationError):
        return False
