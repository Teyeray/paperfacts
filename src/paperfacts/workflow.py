"""Orchestration: parser -> adapter -> disk, then extraction, then comparison.

This is the only place that decides which parser implementation runs: a configured ``*_url`` means an HTTP
service (a GPU server), an empty one means the ``runners/`` script as a subprocess (a workstation).

:func:`run_document` is the single pipeline. The CLI and the web job both call it, so they cannot drift.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from paperfacts.adapters import convert, render_markdown
from paperfacts.compare import ComparisonReport, compare_lanes
from paperfacts.config import Settings
from paperfacts.dataset import DocumentDataset, consolidate_document, write_dataset
from paperfacts.errors import ConfigError, PaperFactsError
from paperfacts.extract import build_extraction_document, extract_lane
from paperfacts.grounding import block_adjacency, ground_lane
from paperfacts.keys import comparison_key, extractor_key_for
from paperfacts.llm import LlmClient, OpenAICompatibleClient
from paperfacts.matching import match_samples
from paperfacts.models import BACKENDS, Backend, DocumentInput, ParsedArtifact
from paperfacts.normalize import normalize_lane
from paperfacts.parsers import MinerUHttpParser, PaddleHttpParser, Parser, SubprocessParser, default_runner_script
from paperfacts.pdf import read_geometry
from paperfacts.records import LaneExtraction
from paperfacts.storage import DataLayout, ensure_identity

logger = logging.getLogger(__name__)

# The comparison is strictly between two lanes; a third parser would need compare_lanes redesigned.
BACKEND_A, BACKEND_B = BACKENDS


# ---- Parsing ----------------------------------------------------------------------------------------------


def _uv_prefix(settings: Settings) -> tuple[str, ...]:
    return (settings.uv_bin, "run", "--locked", "--script")


def _build_mineru(settings: Settings) -> Parser:
    if settings.mineru_url:
        return MinerUHttpParser(settings.mineru_url, timeout_s=settings.http_timeout_s)
    return SubprocessParser(
        "mineru",
        default_runner_script(settings.repo_root, "mineru"),
        command_prefix=_uv_prefix(settings),
        timeout_s=settings.subprocess_timeout_s,
    )


def _build_paddle(settings: Settings) -> Parser:
    if settings.paddle_url:
        return PaddleHttpParser(
            settings.paddle_url, timeout_s=settings.http_timeout_s, render_dpi=settings.paddle_render_dpi
        )
    extra_args: list[str] = ["--dpi", str(settings.paddle_render_dpi)]
    for flag, value in (
        ("--vl-backend", settings.paddle_vl_backend),
        ("--vl-server-url", settings.paddle_vl_server_url),
        ("--vl-model-name", settings.paddle_vl_model_name),
    ):
        if value:
            extra_args += [flag, value]
    return SubprocessParser(
        "paddleocr_vl",
        default_runner_script(settings.repo_root, "paddleocr_vl"),
        command_prefix=_uv_prefix(settings),
        extra_args=tuple(extra_args),
        timeout_s=settings.subprocess_timeout_s,
    )


# A new backend is one builder here plus one adapter in adapters.py, never an if-chain edit.
PARSER_BUILDERS: dict[Backend, Callable[[Settings], Parser]] = {
    "mineru": _build_mineru,
    "paddleocr_vl": _build_paddle,
}


def build_parser(backend: Backend, settings: Settings) -> Parser:
    """The parser implementation the configuration asks for."""
    try:
        builder = PARSER_BUILDERS[backend]
    except KeyError as exc:
        raise ValueError(f"unknown backend: {backend!r}") from exc
    return builder(settings)


@dataclass(frozen=True)
class ParseReport:
    backend: Backend
    backend_version: str | None
    cache_hit: bool
    runtime_s: float
    page_count: int
    block_count: int
    type_counts: dict[str, int]
    artifact_path: Path
    markdown_path: Path


def parse_document(
    document: DocumentInput,
    backend: Backend,
    settings: Settings,
    *,
    force: bool = False,
) -> tuple[ParsedArtifact, ParseReport]:
    """Parse one lane: run the parser (or hit its cache), adapt it, write the Markdown and the artifact."""
    layout = DataLayout(settings.data_root)
    ensure_identity(layout, document)  # written the moment the directory exists; readers only read it
    parser = build_parser(backend, settings)

    clock = time.monotonic()
    raw = parser.parse(document, layout.raw_dir(document.document_id, backend), force=force)
    artifact = convert(raw, document, read_geometry(document.pdf_path))
    runtime_s = time.monotonic() - clock

    markdown_path = layout.markdown_path(document.document_id, backend)
    artifact_path = layout.artifact_path(document.document_id, backend)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(artifact.blocks), encoding="utf-8")
    artifact.write(artifact_path)  # last: its existence means parsed/ is complete

    report = ParseReport(
        backend=backend,
        backend_version=artifact.backend_version,
        cache_hit=raw.cache_hit,
        runtime_s=runtime_s,
        page_count=artifact.page_count,
        block_count=len(artifact.blocks),
        type_counts=artifact.type_counts(),
        artifact_path=artifact_path,
        markdown_path=markdown_path,
    )
    logger.info("parsed %s", report)
    return artifact, report


def load_artifact(document: DocumentInput, backend: Backend, settings: Settings) -> ParsedArtifact:
    path = DataLayout(settings.data_root).artifact_path(document.document_id, backend)
    if not path.is_file():
        raise FileNotFoundError(f"no {backend} artifact at {path}; run `paperfacts parse` first")
    return ParsedArtifact.read(path)


# ---- Extraction and comparison --------------------------------------------------------------------------------


def build_llm_client(settings: Settings) -> OpenAICompatibleClient:
    """One client shared by extraction and matching (connection pool, usage accounting). Close it."""
    return OpenAICompatibleClient(
        settings.llm_base_url,
        settings.require_llm_api_key(),
        settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        cache_dir=DataLayout(settings.data_root).llm_cache_dir(),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        retry_attempts=settings.llm_retry_attempts,
        retry_backoff_s=settings.llm_retry_backoff_s,
    )


def read_lane(
    layout: DataLayout, document_id: str, backend: Backend, key: str, *, artifact: ParsedArtifact | None = None
) -> LaneExtraction | None:
    """A stored lane, re-deriving what is cheap: grounding against the artifact, then normalisation.

    Both are pure functions of the stored record, so they are redone on every read rather than trusted from
    the file: improving a rule costs nothing and never leaves a stale verdict behind. The artifact is read
    from disk unless the caller already holds it; without one the stored grounding verdicts are kept, since
    they cannot be re-checked but are still the best answer.
    """
    path = layout.extraction_path(document_id, backend, key)
    if not path.is_file():
        return None
    lane = LaneExtraction.read(path)
    artifact_path = layout.artifact_path(document_id, backend)
    if artifact is None and artifact_path.is_file():
        artifact = ParsedArtifact.read(artifact_path)
    if artifact is not None:
        lane = ground_lane(lane, build_extraction_document(artifact).blocks, adjacency=block_adjacency(artifact.blocks))
    return normalize_lane(lane)


def extract_document(
    document: DocumentInput,
    backend: Backend,
    settings: Settings,
    client: LlmClient,
    *,
    force: bool = False,
) -> LaneExtraction:
    """Extract one lane. What is stored is the model's own wording; what is returned is normalised.

    Changing the prompt, the model or the schema changes ``extractor_key`` and re-runs the extraction.
    ``force`` bypasses both this cache and the LLM cache, and really re-asks.
    """
    layout = DataLayout(settings.data_root)
    key = extractor_key_for(settings, client.model)
    artifact = load_artifact(document, backend, settings)
    if not force:
        cached = read_lane(layout, document.document_id, backend, key, artifact=artifact)
        if cached is not None:
            logger.info("extraction cache_hit backend=%s doc=%s", backend, document.document_id[:16])
            return cached

    lane = extract_lane(
        artifact,
        client,
        mode=settings.extraction_mode,
        passes=settings.extraction_passes,
        context_tokens=settings.llm_context_tokens,
        candidate_limit=settings.candidate_limit,
        refresh=force,
    )
    lane.write(layout.extraction_path(document.document_id, backend, key))
    return normalize_lane(lane)


def compare_document(
    document: DocumentInput,
    settings: Settings,
    client: LlmClient,
    *,
    force: bool = False,
) -> ComparisonReport:
    """Extract both lanes, match samples with the model, compare fields by rule, store the report.

    The report path carries both keys, so changing a tolerance recomputes the comparison without paying for
    extraction again and cannot serve a stale verdict. ``force`` redoes matching and comparison only;
    extraction has its own cache and its own force.
    """
    layout = DataLayout(settings.data_root)
    path = layout.comparison_path(
        document.document_id,
        extractor_key_for(settings, client.model),
        comparison_key(),
    )
    if path.is_file() and not force:
        logger.info("comparison cache_hit doc=%s", document.document_id[:16])
        return ComparisonReport.read(path)

    lane_a = extract_document(document, BACKEND_A, settings, client)
    lane_b = extract_document(document, BACKEND_B, settings, client)
    matching = match_samples(lane_a, lane_b, client, refresh=force)
    report = compare_lanes(lane_a, lane_b, matching)
    report.write(path)
    logger.info("compared doc=%s counts=%s", document.document_id[:16], report.counts.model_dump())
    return report


# ---- The whole pipeline, shared by the CLI and the web job ------------------------------------------------------

StageStatus = Literal["pending", "running", "done", "failed", "skipped"]
# (stage, status, detail). Stage names are a public contract: the progress bar and the CLI both use them.
StageCallback = Callable[[str, StageStatus, str], None]


def stage_names() -> tuple[str, ...]:
    return (*(f"parse:{b}" for b in BACKENDS), *(f"extract:{b}" for b in BACKENDS), "compare", "export")


@dataclass(frozen=True)
class PipelineResult:
    parse_reports: dict[Backend, ParseReport]
    lanes: dict[Backend, LaneExtraction]
    report: ComparisonReport
    dataset: DocumentDataset
    excel_path: Path


def _ignore_stage(stage: str, status: StageStatus, detail: str) -> None:
    pass


def run_document(
    document: DocumentInput,
    settings: Settings,
    *,
    force: bool = False,
    on_stage: StageCallback = _ignore_stage,
) -> PipelineResult:
    """Run both lanes and automatically export consolidated data. Expensive steps are cached."""
    parse_reports: dict[Backend, ParseReport] = {}
    for backend in BACKENDS:
        on_stage(f"parse:{backend}", "running", "")
        _, parse_report = parse_document(document, backend, settings, force=force)
        parse_reports[backend] = parse_report
        cached = " (cached)" if parse_report.cache_hit else ""
        on_stage(f"parse:{backend}", "done", f"{parse_report.block_count} blocks{cached}")

    lanes: dict[Backend, LaneExtraction] = {}
    with build_llm_client(settings) as client:
        for backend in BACKENDS:
            on_stage(f"extract:{backend}", "running", "")
            lane = extract_document(document, backend, settings, client, force=force)
            lanes[backend] = lane
            ungrounded = len(lane.ungrounded())
            detail = f"{len(lane.samples)} samples" + (f", {ungrounded} ungrounded" if ungrounded else "")
            on_stage(f"extract:{backend}", "done", detail)
        on_stage("compare", "running", "")
        report = compare_document(document, settings, client, force=force)
    counts = report.counts
    on_stage(
        "compare",
        "done",
        f"agree {counts.agree} · conflict {counts.conflict} · ambiguous {counts.ambiguous} · missing {counts.missing}",
    )
    on_stage("export", "running", "")
    dataset = consolidate_document(document, lanes, report)
    excel_path = DataLayout(settings.data_root).dataset_path(document.document_id)
    write_dataset([dataset], excel_path)
    on_stage("export", "done", str(excel_path))
    return PipelineResult(
        parse_reports=parse_reports, lanes=lanes, report=report, dataset=dataset, excel_path=excel_path
    )


# ---- Directory batches and offline re-export -------------------------------------------------------


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


def export_document(document: DocumentInput, settings: Settings) -> DocumentDataset:
    """Rebuild a workbook row from current cached extractions without starting a parser or an LLM."""
    layout = DataLayout(settings.data_root)
    key = extractor_key_for(settings)
    report_path = layout.comparison_path(document.document_id, key, comparison_key())
    if not report_path.is_file():
        raise FileNotFoundError(f"no current comparison for {document.pdf_path.name}; run `paperfacts run` first")
    report = ComparisonReport.read(report_path)
    lanes: dict[Backend, LaneExtraction] = {}
    for backend in BACKENDS:
        lane = read_lane(layout, document.document_id, backend, key)
        if lane is None:
            raise FileNotFoundError(f"no current {backend} extraction for {document.pdf_path.name}")
        lanes[backend] = lane
    # Grounding is rechecked on read, so comparison must use those same refreshed values.
    report = compare_lanes(lanes[BACKEND_A], lanes[BACKEND_B], report.matching)
    return consolidate_document(document, lanes, report)


def run_batch(
    source: Path,
    settings: Settings,
    *,
    output: Path | None = None,
    force: bool = False,
    export_only: bool = False,
    on_stage: StageCallback = _ignore_stage,
) -> BatchResult:
    """Process unique PDFs serially and checkpoint the workbook after every attempted document.

    Local parser models cannot safely share the laptop's memory. Completed parse/extraction caches
    make interruption resumable, while expected per-paper errors remain visible in the workbook.
    """
    paths = discover_pdfs(source)
    output = output or DataLayout(settings.data_root).batch_dataset_path()
    if output.suffix.lower() != ".xlsx":
        raise ConfigError("Excel output must have the .xlsx extension")
    if force and export_only:
        raise ConfigError("--force cannot be used with offline export")
    datasets: list[DocumentDataset] = []
    failures: list[dict[str, str]] = []
    seen: set[str] = set()
    duplicates = 0
    for index, path in enumerate(paths, 1):
        prefix = f"{index}/{len(paths)} {path.name}"
        on_stage(prefix, "running", "")
        document_id = ""
        try:
            document = DocumentInput.from_path(path)
            document_id = document.document_id
            if document_id in seen:
                duplicates += 1
                on_stage(prefix, "skipped", "duplicate PDF content")
                continue
            seen.add(document_id)
            if export_only:
                dataset = export_document(document, settings)
            else:
                result = run_document(document, settings, force=force, on_stage=on_stage)
                dataset = result.dataset
            datasets.append(dataset)
        except (PaperFactsError, OSError, ValueError) as exc:
            logger.exception("batch failed for %s", path.name)
            failures.append({"document_id": document_id, "filename": path.name, "error": str(exc)})
            on_stage(prefix, "failed", str(exc))
        else:
            on_stage(prefix, "done", "")
        # Export failures are fatal: claiming progress without a writable output would be misleading.
        write_dataset(datasets, output, failures=failures)
    return BatchResult(tuple(datasets), tuple(failures), duplicates, output)
