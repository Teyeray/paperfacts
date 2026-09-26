"""Directory batches and offline re-export, both built on :func:`paperfacts.workflow.run_document`.

A batch is many documents, each run through the single pipeline in :mod:`paperfacts.workflow`; this module only
discovers the PDFs, overlaps them, and checkpoints the one workbook they share. Kept out of ``workflow.py`` so
the pipeline stays the one thing that module does.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, as_completed
from dataclasses import dataclass
from pathlib import Path

from paperfacts.compare import ComparisonReport, compare_lanes
from paperfacts.config import Settings
from paperfacts.dataset import DocumentDataset, consolidate_document, incomplete_reason
from paperfacts.errors import Cancelled, ConfigError, PaperFactsError
from paperfacts.keys import ComparisonOptions, comparison_key, extractor_key_for
from paperfacts.models import BACKENDS, Backend, DocumentInput
from paperfacts.profile import DomainProfile
from paperfacts.readings import shown_figures
from paperfacts.records import LaneExtraction
from paperfacts.storage import DataLayout
from paperfacts.threads import ContextThreadPoolExecutor
from paperfacts.workbook import write_dataset
from paperfacts.workflow import (
    BACKEND_A,
    BACKEND_B,
    StageCallback,
    StageStatus,
    compared_these,
    ignore_stage,
    read_lane,
    run_document,
    store_dataset,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchResult:
    documents: tuple[DocumentDataset, ...]
    failures: tuple[dict[str, str], ...]
    duplicate_count: int
    excel_path: Path


def discover_pdfs(source: Path) -> tuple[Path, ...]:
    """Keep discovery stable across runs, including uppercase PDF suffixes and nested folders."""
    if source.is_file():
        paths = (source,) if source.suffix.lower() == ".pdf" else ()
    elif source.is_dir():
        paths = tuple(sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"))
    else:
        raise FileNotFoundError(source)
    if not paths:
        raise ConfigError(f"no PDF files found in {source}")
    return paths


def export_document(document: DocumentInput, settings: Settings, profile: DomainProfile) -> DocumentDataset:
    """Rebuild a workbook row from current cached extractions without starting a parser or an LLM."""
    layout = DataLayout(settings.data_root)
    key = extractor_key_for(settings, profile)
    options = ComparisonOptions.from_settings(settings, profile)
    report_path = layout.comparison_path(document.document_id, key, comparison_key(options))
    if not report_path.is_file():
        raise FileNotFoundError(f"no current comparison for {document.display_filename}; run `paperfacts run` first")
    report = ComparisonReport.read(report_path)
    lanes: dict[Backend, LaneExtraction] = {}
    for backend in BACKENDS:
        lane = read_lane(layout, document.document_id, backend, key, profile)
        if lane is None:
            raise FileNotFoundError(f"no current {backend} extraction for {document.display_filename}")
        lanes[backend] = lane
    if not compared_these(report, lanes[BACKEND_A], lanes[BACKEND_B]):
        raise FileNotFoundError(f"the comparison of {document.display_filename} predates its parse; run it again")
    # Grounding is rechecked on read, so comparison must use those same refreshed values. Stored too: the web
    # serves the report beside the table, and the two must be the same verdicts. Every entity's stored matching
    # is reused: re-comparing with one would leave every other entity's samples unmatched, as false "missing".
    report = compare_lanes(lanes[BACKEND_A], lanes[BACKEND_B], report.matchings, options)
    reason = incomplete_reason(lanes, report)
    if reason:
        raise FileNotFoundError(f"{document.display_filename}: {reason}; run it again")
    report.write(report_path)
    dataset = consolidate_document(document, lanes, report, options)
    # An offline re-export is how a code-only change reaches the browser, so refresh the web view too.
    store_dataset(layout, dataset)
    return dataset


def run_batch(
    source: Path,
    settings: Settings,
    profile: DomainProfile,
    *,
    output: Path | None = None,
    force: bool = False,
    force_figures: bool = False,
    export_only: bool = False,
    jobs: int = 1,
    on_stage: StageCallback = ignore_stage,
) -> BatchResult:
    """Process unique PDFs, ``jobs`` at a time, and checkpoint the workbook after every attempted document.

    Completed parse/extraction caches make interruption resumable, while expected per-paper errors remain
    visible in the workbook and never stop the other papers. Whatever order the papers finish in, the
    workbook, the failures and the returned datasets are in input order, so a parallel run writes the table
    a serial one would. ``on_stage`` is called under a lock, one call at a time, with each paper's stages
    prefixed by that paper's ``i/n name`` so interleaved progress stays readable.

    Parsing stays one paper per parser (the parse locks in :mod:`paperfacts.parsers`) and model requests
    share one in-flight limit (:mod:`paperfacts.llm`), so ``jobs`` overlaps the model waits of several
    papers without multiplying the load on the GPU or the endpoint.

    When a parallel batch is stopped (Ctrl-C, or an error that is not one paper's own), no new paper starts
    and the running ones stop at their next stage boundary, raising :class:`Cancelled` there; the stage in
    progress finishes first, because a request already paid for is worth caching. Stopped papers are neither
    rows nor failures: the next run picks them up from the caches.
    """
    paths = discover_pdfs(source)
    output = output or DataLayout(settings.data_root).batch_dataset_path(profile.name)
    if output.suffix.lower() != ".xlsx":
        raise ConfigError("Excel output must have the .xlsx extension")
    if force and export_only:
        raise ConfigError("--force cannot be used with offline export")
    if jobs < 1:
        raise ConfigError(f"--jobs must be at least 1, got {jobs}")

    # Two locks, so a paper reporting progress never waits for another paper's workbook checkpoint.
    progress_lock = threading.Lock()
    results_lock = threading.Lock()
    cancel = threading.Event()
    # Keyed by input position, and read back sorted, so completion order never reaches the output.
    datasets: dict[int, DocumentDataset] = {}
    figure_rows: dict[int, tuple[Mapping[str, object], ...]] = {}
    failures: dict[int, dict[str, str]] = {}

    def report(stage: str, status: StageStatus, detail: str) -> None:
        with progress_lock:
            on_stage(stage, status, detail)

    def report_stage(prefix: str) -> StageCallback:
        def mark(stage: str, status: StageStatus, detail: str) -> None:
            # A stage boundary: the one place a running paper can be stopped without abandoning a request.
            if cancel.is_set():
                raise Cancelled("the batch was stopped")
            report(f"{prefix} {stage}", status, detail)

        return mark

    def settle(
        index: int, prefix: str, outcome: DocumentDataset | dict[str, str], rows: Sequence[Mapping[str, object]] = ()
    ) -> None:
        if isinstance(outcome, DocumentDataset):
            # Its rows are written, with the unanswered cells refused, but the paper is not finished.
            report(prefix, "done", f"incomplete: {outcome.incomplete}" if outcome.incomplete else "")
        else:
            report(prefix, "failed", outcome["error"])
        with results_lock:
            if isinstance(outcome, DocumentDataset):
                datasets[index], figure_rows[index] = outcome, tuple(rows)
            else:
                failures[index] = outcome
            # Export failures are fatal: claiming progress without a writable output would be misleading.
            # Written under the lock, so a checkpoint never overwrites a later one.
            write_dataset(
                [datasets[i] for i in sorted(datasets)],
                output,
                profile,
                failures=[failures[i] for i in sorted(failures)],
                figure_rows=[row for i in sorted(figure_rows) for row in figure_rows[i]],
            )

    def failure(document_id: str, path: Path, exc: Exception) -> dict[str, str]:
        logger.exception("batch failed for %s", path.name)
        return {"document_id": document_id, "filename": path.name, "error": str(exc)}

    # Hashing comes first and runs serially, so which copy of a duplicated PDF is processed is decided by
    # input order rather than by whichever thread hashed first.
    queue: list[tuple[int, str, DocumentInput]] = []
    seen: set[str] = set()
    duplicates = 0
    for index, path in enumerate(paths, 1):
        prefix = f"{index}/{len(paths)} {path.name}"
        try:
            document = DocumentInput.from_path(path)
        except (PaperFactsError, OSError, ValueError) as exc:
            settle(index, prefix, failure("", path, exc))
            continue
        if document.document_id in seen:
            duplicates += 1
            report(prefix, "skipped", "duplicate PDF content")
            continue
        seen.add(document.document_id)
        queue.append((index, prefix, document))

    def process(index: int, prefix: str, document: DocumentInput) -> None:
        if cancel.is_set():
            return
        report(prefix, "running", "")
        try:
            if export_only:
                dataset = export_document(document, settings, profile)
                figures = shown_figures(document.document_id, document.display_filename, settings, profile)
            else:
                result = run_document(
                    document,
                    settings,
                    profile,
                    force=force,
                    force_figures=force_figures,
                    on_stage=report_stage(prefix),
                )
                dataset, figures = result.dataset, result.figures
        except Cancelled:
            report(prefix, "skipped", "stopped with the batch; its finished stages are cached")
        except (PaperFactsError, OSError, ValueError) as exc:
            settle(index, prefix, failure(document.document_id, document.pdf_path, exc))
        else:
            settle(index, prefix, dataset, figures.rows if figures is not None else ())

    if jobs == 1 or len(queue) <= 1:
        # On the calling thread, exactly as the serial loop always ran: Ctrl-C interrupts the paper at once.
        for item in queue:
            process(*item)
    else:
        pool = ContextThreadPoolExecutor(max_workers=min(jobs, len(queue)), thread_name_prefix="paperfacts-document")
        futures: list[Future[None]] = []
        try:
            futures += [pool.submit(process, *item) for item in queue]
            # Completion order, so an unexpected error (a workbook that cannot be written) surfaces at once
            # instead of after every paper queued ahead of it.
            for future in as_completed(futures):
                future.result()
        except BaseException:
            cancel.set()
            running = sum(1 for future in futures if future.running())
            report("batch", "failed", f"stopping; waiting for {running} running papers to reach a stage boundary")
            raise
        finally:
            # After an error nothing new starts, and the papers already running are waited for: returning
            # while they still write the workbook would let a checkpoint land after the caller gave up on it,
            # and the interpreter joins these threads at exit anyway.
            pool.shutdown(wait=True, cancel_futures=True)
    return BatchResult(
        tuple(datasets[i] for i in sorted(datasets)),
        tuple(failures[i] for i in sorted(failures)),
        duplicates,
        output,
    )
