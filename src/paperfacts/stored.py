"""What is stored for a document, and whether it is still current: the readers of the files the stages write.

Separate from :mod:`paperfacts.workflow`, which writes them, because these are asked by readers that run no
stage -- the web library, "run all", ``deploy.sh --rerun``. A stored result is current only when it was built
under the caller's keys **and** from the parse now on disk: source ids are positional, so a result of an
earlier parse would cite whatever block now holds that ordinal. Not part of any cache key.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from paperfacts.compare import ComparisonReport
from paperfacts.dataset import DatasetPayload
from paperfacts.models import BACKENDS, Backend, ParsedArtifact
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


# path -> (stamp, content hash). The web checks every stored table against its parse on each corpus listing,
# and hashing means parsing a whole artifact; the stamp includes the inode because artifacts are replaced by a
# rename, so a re-parse always changes it.
_ARTIFACT_HASHES: dict[Path, tuple[tuple[int, int, int], str]] = {}
_ARTIFACT_HASHES_LOCK = threading.Lock()


def _artifact_hash(path: Path) -> str | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    with _ARTIFACT_HASHES_LOCK:
        cached = _ARTIFACT_HASHES.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    sha = ParsedArtifact.read(path).content_hash()
    with _ARTIFACT_HASHES_LOCK:
        _ARTIFACT_HASHES[path] = (stamp, sha)
    return sha


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
    keys are the caller's, so the library that lists documents and the one that reads them cannot disagree."""

    def done(path: Path) -> StageStatus:
        return "done" if path.is_file() else "pending"

    figures = done(layout.figures_path(document_id, figure_key))
    status: dict[str, StageStatus] = {
        **{f"parse:{b}": done(layout.artifact_path(document_id, b)) for b in BACKENDS},
        # Opt-in: switched off and never read is a skip, not work still to do.
        "figures": figures if figures == "done" or figures_enabled else "skipped",
        **{f"extract:{b}": done(layout.extraction_path(document_id, b, extractor_key)) for b in BACKENDS},
        "compare": done(layout.comparison_path(document_id, extractor_key, comparison_key)),
        "export": done(layout.dataset_json_path(document_id, extractor_key, comparison_key)),
    }
    return tuple(Stage(name=name, status=status[name]) for name in stage_names())


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
