"""Orchestration: parser -> adapter -> disk, then extraction, then comparison.

This module is where "which parser implementation" is decided, and it is the only place that decides it:
a configured ``*_url`` means an HTTP service (a GPU server), an empty one means the ``runners/`` script as
a subprocess (a workstation). Nothing above this layer knows which it got.

:func:`run_document` is the single orchestration path. The CLI and the web job both call it, so the two
cannot drift apart.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from paperfacts.adapters import convert
from paperfacts.config import Settings
from paperfacts.consensus import ComparisonReport, compare_lanes, comparison_key, match_samples
from paperfacts.extraction.document import build_extraction_document
from paperfacts.extraction.extractor import extract_lane, extractor_key
from paperfacts.extraction.grounding import ground_lane
from paperfacts.extraction.llm import LlmClient, OpenAICompatibleClient
from paperfacts.extraction.records import LaneExtraction
from paperfacts.models.artifact import BACKENDS, Backend, DocumentInput, ParsedArtifact
from paperfacts.normalization import normalize_lane
from paperfacts.parsers.base import DocumentParser
from paperfacts.parsers.http_parser import MinerUHttpParser, PaddleHttpParser
from paperfacts.parsers.subprocess_parser import SubprocessParser, default_runner_script
from paperfacts.pdf import read_geometry
from paperfacts.storage.identity import ensure_identity
from paperfacts.storage.paths import DataLayout

logger = logging.getLogger(__name__)


def _uv_prefix(settings: Settings) -> tuple[str, ...]:
    return (settings.uv_bin, "run", "--locked", "--script")


def _build_mineru(settings: Settings) -> DocumentParser:
    if settings.mineru_url:
        return MinerUHttpParser(settings.mineru_url, timeout_s=settings.http_timeout_s)
    return SubprocessParser(
        "mineru",
        default_runner_script(settings.repo_root, "mineru"),
        command_prefix=_uv_prefix(settings),
        timeout_s=settings.subprocess_timeout_s,
    )


def _build_paddle(settings: Settings) -> DocumentParser:
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


# Mirrors adapters.ADAPTERS: a new backend is one builder plus one adapter, never an if-chain edit.
PARSER_BUILDERS: dict[Backend, Callable[[Settings], DocumentParser]] = {
    "mineru": _build_mineru,
    "paddleocr_vl": _build_paddle,
}


def build_parser(backend: Backend, settings: Settings) -> DocumentParser:
    """Pick the parser implementation the configuration asks for."""
    try:
        builder = PARSER_BUILDERS[backend]
    except KeyError as exc:
        raise ValueError(f"unknown backend: {backend!r}") from exc
    return builder(settings)


@dataclass(frozen=True)
class ParseReport:
    """Summary of one parse, for the CLI and for run records."""

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
    """Parse one lane: run the parser (or hit its cache), adapt it, write markdown/sources/artifact."""
    layout = DataLayout(settings.data_root)
    ensure_identity(layout, document)  # write identity the moment the directory exists; readers only read it
    parser = build_parser(backend, settings)

    clock = time.monotonic()
    raw = parser.parse(document, layout.raw_dir(document.document_id, backend), force=force)
    geometry = read_geometry(document.pdf_path)
    artifact = convert(raw, document, geometry)
    runtime_s = time.monotonic() - clock

    layout.parsed_dir(document.document_id).mkdir(parents=True, exist_ok=True)
    markdown_path = layout.markdown_path(document.document_id, backend)
    sources_path = layout.sources_path(document.document_id, backend)
    artifact_path = layout.artifact_path(document.document_id, backend)
    markdown_path.write_text(artifact.markdown, encoding="utf-8")
    sources_path.write_text(
        json.dumps([block.model_dump(mode="json") for block in artifact.blocks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # artifact.json last, same contract as raw/meta.json: its existence means parsed/ is complete.
    artifact.write(artifact_path)

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
    """Read a stored artifact, or say which command produces it."""
    path = DataLayout(settings.data_root).artifact_path(document.document_id, backend)
    if not path.is_file():
        raise FileNotFoundError(f"no {backend} artifact at {path}; run `paperfacts parse` first")
    return ParsedArtifact.read(path)


# ---- Extraction and two-lane alignment ------------------------------------------------------

# The comparison is strictly between two lanes. A third parser would need compare_lanes redesigned, so the
# assumption is stated here rather than hidden inside an unpacking.
BACKEND_A, BACKEND_B = BACKENDS


def build_llm_client(settings: Settings) -> OpenAICompatibleClient:
    """One client shared by extraction and matching (connection pool, usage accounting). Close it."""
    return OpenAICompatibleClient(
        settings.llm_base_url,
        settings.require_llm_api_key(),
        settings.llm_model,
        timeout_s=settings.llm_timeout_s,
        cache_dir=DataLayout(settings.data_root).llm_cache_dir(),
    )


def extract_document(
    document: DocumentInput,
    backend: Backend,
    settings: Settings,
    client: LlmClient,
    *,
    force: bool = False,
) -> LaneExtraction:
    """Extract one lane. What is stored is the model's own wording; what is returned is normalised.

    Normalisation is redone on every read -- it is pure and takes milliseconds -- so changing a conversion
    rule costs nothing. Changing the prompt, the model or the schema changes ``extractor_key`` instead and
    re-runs the extraction. ``force`` bypasses both this cache and the LLM cache, and really re-asks.
    """
    layout = DataLayout(settings.data_root)
    path = layout.extraction_path(
        document.document_id, backend, extractor_key(client.model, passes=settings.extraction_passes)
    )
    artifact = load_artifact(document, backend, settings)
    if path.is_file() and not force:
        logger.info("extraction cache_hit backend=%s doc=%s", backend, document.document_id[:16])
        blocks = build_extraction_document(artifact).blocks
        return normalize_lane(ground_lane(LaneExtraction.read(path), blocks))

    lane = extract_lane(
        artifact,
        client,
        passes=settings.extraction_passes,
        context_tokens=settings.llm_context_tokens,
        refresh=force,
    )
    lane.write(path)
    return normalize_lane(lane)


def compare_document(
    document: DocumentInput,
    settings: Settings,
    client: LlmClient,
    *,
    force: bool = False,
) -> ComparisonReport:
    """Extract both lanes, match samples with the model, compare fields by rule, store the report.

    The report path carries both keys: ``extractor_key`` (prompt, model, schema) and ``comparison_key``
    (tolerances, normalisation rules). Changing a tolerance therefore recomputes the comparison without
    paying for extraction again, and cannot serve a stale verdict either. ``force`` redoes matching and
    comparison only; extraction has its own cache and its own force.
    """
    layout = DataLayout(settings.data_root)
    path = layout.comparison_path(
        document.document_id, extractor_key(client.model, passes=settings.extraction_passes), comparison_key()
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


# ---- The whole pipeline, shared by the CLI and the web job ------------------------------------

StageStatus = Literal["pending", "running", "done", "failed", "skipped"]
# (stage, status, detail). Stage names are a public contract -- the progress bar and the CLI both use them.
StageCallback = Callable[[str, StageStatus, str], None]


def stage_names() -> tuple[str, ...]:
    return (*(f"parse:{b}" for b in BACKENDS), *(f"extract:{b}" for b in BACKENDS), "compare")


@dataclass(frozen=True)
class PipelineResult:
    parse_reports: dict[Backend, ParseReport]
    lanes: dict[Backend, LaneExtraction]
    report: ComparisonReport


def _ignore_stage(stage: str, status: StageStatus, detail: str) -> None:
    pass


def run_document(
    document: DocumentInput,
    settings: Settings,
    *,
    force: bool = False,
    on_stage: StageCallback = _ignore_stage,
) -> PipelineResult:
    """Parse both lanes, extract both, match and compare, reporting each stage. Every step is cached."""
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
    return PipelineResult(parse_reports=parse_reports, lanes=lanes, report=report)
