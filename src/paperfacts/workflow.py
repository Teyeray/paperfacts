"""Orchestration: parser -> adapter -> disk, then figure reading (opt-in), extraction, comparison, export.

This is the only place that decides which parser implementation runs: a configured ``*_url`` means an HTTP
service (a GPU server), an empty one means the ``runners/`` script as a subprocess (a workstation).

:func:`run_document` is the single pipeline. The CLI and the web job both call it, so they cannot drift.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from paperfacts.adapters import convert, render_markdown
from paperfacts.compare import ComparisonReport, compare_lanes
from paperfacts.config import Settings
from paperfacts.dataset import (
    DocumentDataset,
    consolidate_document,
    incomplete_reason,
    write_dataset_json,
)
from paperfacts.errors import Cancelled, LlmOfflineMiss, ParserError
from paperfacts.extract import extract_lane, informative_blocks
from paperfacts.figures import MAX_TOKENS as FIGURE_MAX_TOKENS
from paperfacts.figures import RETRY_ATTEMPTS as FIGURE_RETRY_ATTEMPTS
from paperfacts.figures import TEMPERATURE as FIGURE_TEMPERATURE
from paperfacts.figures import FigureReadings
from paperfacts.grounding import block_adjacency, ground_lane
from paperfacts.keys import ComparisonOptions, ExtractionOptions, comparison_key, extractor_key
from paperfacts.llm import LlmClient, OpenAICompatibleClient
from paperfacts.matching import match_samples
from paperfacts.models import BACKENDS, Backend, DocumentInput, ParsedArtifact
from paperfacts.normalize import normalize_lane
from paperfacts.parsers import MinerUHttpParser, PaddleHttpParser, Parser, SubprocessParser, default_runner_script
from paperfacts.pdf import read_geometry
from paperfacts.profile import DomainProfile, load_profile, profile_path
from paperfacts.readings import FiguresView, figure_artifact, read_document_figures, shown_figures
from paperfacts.records import LaneExtraction
from paperfacts.storage import DataLayout, ensure_identity, write_text_atomic
from paperfacts.threads import ContextThreadPoolExecutor
from paperfacts.workbook import write_dataset

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
    # The raw parser output was gone and the stored artifact stood in for it, so no adapter ran this time.
    from_artifact: bool = False


def _stored_artifact_for_missing_raw(
    parser: Parser,
    document: DocumentInput,
    backend: Backend,
    artifact_path: Path,
    raw_dir: Path,
    *,
    force: bool,
) -> ParsedArtifact | None:
    """The stored artifact when the raw output it came from is gone, otherwise ``None``.

    Raw output is bulky and gets pruned or moved; the artifact is the small file worth keeping. Re-parsing
    a paper costs GPU minutes, so when only the raw output is missing the artifact stands in for it. The
    cost is that adapter changes are not re-applied — hence the warning and ``--force``.
    """
    if force or not artifact_path.is_file() or parser.is_cached(raw_dir):
        return None
    try:
        artifact = ParsedArtifact.read(artifact_path)
    except (OSError, ValueError) as exc:
        logger.warning("stored %s artifact at %s is unreadable (%s); re-parsing", backend, artifact_path, exc)
        return None
    logger.warning(
        "raw parser output missing for backend=%s doc=%s; using the stored artifact — run with --force to re-parse",
        backend,
        document.document_id[:16],
    )
    return artifact


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
    with build_parser(backend, settings) as parser:
        return _parse_with(parser, document, backend, layout, force=force)


def _parse_with(
    parser: Parser, document: DocumentInput, backend: Backend, layout: DataLayout, *, force: bool
) -> tuple[ParsedArtifact, ParseReport]:
    """:func:`parse_document` with the parser it built, which it closes afterwards whatever happens here."""
    markdown_path = layout.markdown_path(document.document_id, backend)
    artifact_path = layout.artifact_path(document.document_id, backend)
    raw_dir = layout.raw_dir(document.document_id, backend)

    stored = _stored_artifact_for_missing_raw(parser, document, backend, artifact_path, raw_dir, force=force)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    if stored is not None:
        artifact, cache_hit, runtime_s = stored, True, 0.0
        if not markdown_path.is_file():
            # The artifact alone is not a complete document directory; re-render rather than leave a hole.
            write_text_atomic(markdown_path, render_markdown(artifact.blocks))
    else:
        if not document.pdf_path.is_file():
            # Only the real parse path needs the file; say so plainly instead of failing inside the parser.
            raise ParserError(backend, "input", "PDF not available; re-upload to re-parse")
        clock = time.monotonic()
        raw = parser.parse(document, raw_dir, force=force)
        artifact = convert(raw, document, read_geometry(document.pdf_path))
        runtime_s = time.monotonic() - clock
        cache_hit = raw.cache_hit
        write_text_atomic(markdown_path, render_markdown(artifact.blocks))
        artifact.write(artifact_path)  # last: its existence means parsed/ is complete

    report = ParseReport(
        backend=backend,
        backend_version=artifact.backend_version,
        cache_hit=cache_hit,
        runtime_s=runtime_s,
        page_count=artifact.page_count,
        block_count=len(artifact.blocks),
        type_counts=artifact.type_counts(),
        artifact_path=artifact_path,
        markdown_path=markdown_path,
        from_artifact=stored is not None,
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
        reasoning_effort=settings.llm_reasoning_effort,
        retry_attempts=settings.llm_retry_attempts,
        retry_backoff_s=settings.llm_retry_backoff_s,
        offline=settings.llm_offline,
    )


def read_lane(
    layout: DataLayout,
    document_id: str,
    backend: Backend,
    key: str,
    profile: DomainProfile,
    *,
    artifact: ParsedArtifact | None = None,
) -> LaneExtraction | None:
    """A stored lane, re-deriving what is cheap: grounding against the artifact, then normalisation.

    Both are pure functions of the stored record, so they are redone on every read rather than trusted from
    the file: improving a rule costs nothing and never leaves a stale verdict behind. The artifact is read
    from disk unless the caller already holds it; without one the stored grounding verdicts are kept, since
    they cannot be re-checked but are still the best answer.

    A lane extracted from a different parse than the current artifact is a miss (``None``): its source ids
    would point at whatever block now has that ordinal. Re-deriving it is cheap whenever the rendered
    prompts are byte-identical, because the LLM cache is keyed by the request.
    """
    path = layout.extraction_path(document_id, backend, key)
    if not path.is_file():
        return None
    lane = LaneExtraction.read(path)
    artifact_path = layout.artifact_path(document_id, backend)
    if artifact is None and artifact_path.is_file():
        artifact = ParsedArtifact.read(artifact_path)
    if artifact is not None and lane.artifact_sha256 not in (None, artifact.content_hash()):
        logger.info("stored %s extraction came from another parse of doc=%s; re-deriving", backend, document_id[:16])
        return None
    if artifact is not None:
        # The same blocks, and so the same neighbours, extraction grounded against: adjacency over every block
        # would put page furniture between two halves of a sentence.
        blocks = informative_blocks(artifact.blocks)
        lane = ground_lane(
            lane, {block.source_id: block.content for block in blocks}, adjacency=block_adjacency(blocks)
        )
    return normalize_lane(lane, profile)


def extract_document(
    document: DocumentInput,
    backend: Backend,
    settings: Settings,
    profile: DomainProfile,
    client: LlmClient,
    *,
    force: bool = False,
    options: ExtractionOptions | None = None,
) -> LaneExtraction:
    """Extract one lane. What is stored is the model's own wording; what is returned is normalised.

    Changing the prompt, the model or the schema changes ``extractor_key`` and re-runs the extraction.
    ``force`` bypasses both this cache and the LLM cache, and really re-asks. A caller extracting both lanes
    passes the one ``options`` it built for the document, so the two lanes cannot be asked differently.
    """
    layout = DataLayout(settings.data_root)
    if options is None:
        options = ExtractionOptions.from_settings(settings, profile, client.model)
    elif options.model != client.model or options.profile != profile:
        # The file would be named after one model or profile while another answered or was asked.
        raise ValueError(
            f"extraction options for model {options.model!r} and profile {options.profile.name!r} "
            f"do not match the client's model {client.model!r} and the profile {profile.name!r}"
        )
    key = extractor_key(options)
    artifact = load_artifact(document, backend, settings)
    if not force:
        cached = read_lane(layout, document.document_id, backend, key, options.profile, artifact=artifact)
        if cached is not None and cached.failed_questions:
            # Only those questions reach the model again: their invalid answers were never cached.
            logger.info(
                "stored %s extraction of doc=%s has %d unanswered questions; asking again",
                backend,
                document.document_id[:16],
                len(cached.failed_questions),
            )
        elif cached is not None:
            logger.info("extraction cache_hit backend=%s doc=%s", backend, document.document_id[:16])
            return cached

    lane = extract_lane(artifact, client, options, concurrency=settings.llm_concurrency, refresh=force).model_copy(
        update={"artifact_sha256": artifact.content_hash()}
    )
    lane.write(layout.extraction_path(document.document_id, backend, key))
    return normalize_lane(lane, options.profile)


def compare_document(
    document: DocumentInput,
    settings: Settings,
    profile: DomainProfile,
    client: LlmClient,
    *,
    force: bool = False,
    lanes: Mapping[Backend, LaneExtraction] | None = None,
    comparison: ComparisonOptions | None = None,
) -> ComparisonReport:
    """Match samples with the model, compare fields by rule, store the report.

    The report path carries both keys, so changing a tolerance recomputes the comparison without paying for
    extraction again and cannot serve a stale verdict. ``force`` redoes matching and comparison only;
    extraction has its own cache and its own force.

    ``lanes`` lets a caller that already holds both extractions hand them over instead of having them
    loaded again; without it the lanes are read through :func:`extract_document`, whose cached path
    rechecks grounding on read. The standalone ``compare`` CLI command relies on that loader. A caller that
    also consolidates passes the one ``comparison`` it built for the document, so the verdicts and the table
    cannot be decided under two options.
    """
    layout = DataLayout(settings.data_root)
    options = ExtractionOptions.from_settings(settings, profile, client.model)
    if comparison is None:
        comparison = ComparisonOptions.from_settings(settings, profile)
    elif comparison.profile != profile:
        raise ValueError(f"comparison options for profile {comparison.profile.name!r}, not {profile.name!r}")
    path = layout.comparison_path(document.document_id, extractor_key(options), comparison_key(comparison))
    if lanes is None:
        lanes = {
            backend: extract_document(document, backend, settings, profile, client, options=options)
            for backend in BACKENDS
        }
    lane_a, lane_b = lanes[BACKEND_A], lanes[BACKEND_B]
    # A stored report is never of an incomplete lane (see below), so with one it would be of other lanes.
    incomplete_lanes = bool(lane_a.failed_questions or lane_b.failed_questions)
    if path.is_file() and not force and not incomplete_lanes:
        cached = ComparisonReport.read(path)
        if compared_these(cached, lane_a, lane_b):
            logger.info("comparison cache_hit doc=%s", document.document_id[:16])
            return cached
        logger.info("stored comparison of doc=%s compared other parses; comparing again", document.document_id[:16])

    matching = match_samples(lane_a, lane_b, client, profile, refresh=force)
    report = compare_lanes(lane_a, lane_b, matching, comparison)
    reason = incomplete_reason(lanes, report)
    if reason:
        # An earlier run's report under these keys goes too: it came from other answers, and kept it would be
        # served to a later run and keep the paper counted as compared.
        logger.warning("%s for doc=%s; the comparison is not stored", reason, document.document_id[:16])
        path.unlink(missing_ok=True)
    else:
        report.write(path)
    logger.info("compared doc=%s counts=%s", document.document_id[:16], report.counts.model_dump())
    return report


def compared_these(report: ComparisonReport, lane_a: LaneExtraction, lane_b: LaneExtraction) -> bool:
    """Whether ``report`` was built from lanes of the same parses as these. A hash missing on either side
    (a file from before it was recorded) is unknown, not a mismatch, so existing stores still read."""
    return all(
        stored is None or current is None or stored == current
        for stored, current in (
            (report.artifact_sha256_a, lane_a.artifact_sha256),
            (report.artifact_sha256_b, lane_b.artifact_sha256),
        )
    )


# ---- Figure reading ------------------------------------------------------------------------------------


def build_vision_client(settings: Settings) -> OpenAICompatibleClient:
    """The figures stage's own client: the LLM's endpoint and key, the vision model, one retry.

    A separate client rather than the extraction one, so the vision model can never answer an extraction
    question by accident and its long timeout never applies to one.
    """
    return OpenAICompatibleClient(
        settings.llm_base_url,
        settings.require_llm_api_key(),
        settings.figures_model,
        timeout_s=settings.figures_timeout_s,
        cache_dir=DataLayout(settings.data_root).llm_cache_dir(),
        temperature=FIGURE_TEMPERATURE,
        max_tokens=FIGURE_MAX_TOKENS,
        reasoning_effort=None,
        retry_attempts=FIGURE_RETRY_ATTEMPTS,
        retry_backoff_s=settings.llm_retry_backoff_s,
        offline=settings.llm_offline,
    )


def _figures_detail(readings: FigureReadings) -> str:
    counts: dict[str, int] = {}
    for panel in readings.panels:
        counts[panel.status] = counts.get(panel.status, 0) + 1
    detail = f"{len(readings.readings)} readings from {len(readings.panels)} panels"
    if counts.get("not_chart"):
        detail += f", {counts['not_chart']} not a chart"
    if counts.get("unreadable"):
        detail += f", {counts['unreadable']} unreadable"
    if counts.get("error"):
        detail += f", {counts['error']} requests failed"
    return detail


def _read_figures_stage(
    document: DocumentInput,
    settings: Settings,
    profile: DomainProfile,
    *,
    force: bool,
    artifact: ParsedArtifact | None,
    stop: threading.Event | None = None,
) -> tuple[StageStatus, str]:
    """The figures stage's work, run beside the extraction lanes. Never raises but for an offline replay
    miss: it is an opt-in extra, and a chart the vision model could not read must not cost the paper its
    extraction. It reports its outcome instead of calling ``on_stage`` itself, so every stage mark still
    comes from the calling thread."""
    try:
        artifact = artifact or figure_artifact(document, settings)
        with build_vision_client(settings) as client:
            readings = read_document_figures(
                document, settings, profile, client, force=force, artifact=artifact, stop=stop
            )
    except Cancelled:
        # Not this stage's failure: it was told to stop because another stage failed. The panels it had
        # read are cached, so the next run picks them up.
        logger.info("figure reading stopped for %s: the rest of the paper failed", document.display_filename)
        return "skipped", "stopped: the rest of the paper failed"
    except LlmOfflineMiss:
        # The one exception to the isolation: a replay that missed must fail the paper, not become a
        # "failed" figures stage beside an exported dataset that looks like a complete replay.
        raise
    except Exception as exc:  # isolation is the point: any failure here is this stage's alone
        logger.exception("figure reading failed for %s", document.display_filename)
        return "failed", f"{type(exc).__name__}: {exc}"[:300]
    return ("done" if readings.complete else "failed"), _figures_detail(readings)


def _figures_mark(status: StageStatus, detail: str, view: FiguresView | None) -> str:
    """The stage detail, plus what the reader should know about the readings actually shown."""
    if status == "skipped":
        detail = "figures.enabled is false" + (f"; {len(view.rows)} stored readings kept" if view is not None else "")
    warning = view.warning() if view is not None else ""
    return f"{detail}; {warning}" if warning else detail


# ---- The whole pipeline, shared by the CLI and the web job ------------------------------------------------------

StageStatus = Literal["pending", "running", "done", "failed", "skipped"]


class Stage(BaseModel):
    """One stage's progress, as a job reports it or as the files on disk show it."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: StageStatus = "pending"
    detail: str = ""


# (stage, status, detail). Stage names are a public contract: the progress bar and the CLI both use them.
StageCallback = Callable[[str, StageStatus, str], None]


def stage_names() -> tuple[str, ...]:
    return (
        *(f"parse:{b}" for b in BACKENDS),
        "figures",
        *(f"extract:{b}" for b in BACKENDS),
        "compare",
        "export",
    )


@dataclass(frozen=True)
class PipelineResult:
    parse_reports: dict[Backend, ParseReport]
    lanes: dict[Backend, LaneExtraction]
    report: ComparisonReport
    dataset: DocumentDataset
    excel_path: Path
    # None when the run's result is not kept (dataset.incomplete says why): such a dataset is not stored.
    dataset_json_path: Path | None
    # The chart readings shown with this paper, when there are any (see shown_figures).
    figures: FiguresView | None = None


def store_dataset(layout: DataLayout, dataset: DocumentDataset) -> Path:
    """The web UI reads the consolidated table from disk, so every path that produces one writes it.

    The dataset carries the keys it was built under, which is what the JSON file is named after: an
    export made with different settings lands beside the old one instead of overwriting it.
    """
    path = layout.dataset_json_path(dataset.document_id, dataset.extractor_key, dataset.comparison_key)
    write_dataset_json(dataset, path)
    return path


def ignore_stage(stage: str, status: StageStatus, detail: str) -> None:
    pass


def run_document(
    document: DocumentInput,
    settings: Settings,
    profile: DomainProfile | None = None,
    *,
    force: bool = False,
    force_figures: bool = False,
    on_stage: StageCallback = ignore_stage,
) -> PipelineResult:
    """Run both lanes and automatically export consolidated data. Expensive steps are cached.

    ``force`` redoes parsing, extraction and comparison; ``force_figures`` re-reads the charts. They are
    separate because each costs minutes of a different model, and wanting one redone rarely means the other.

    Every entry point (the CLI, the web app, a batch) loads its profile once and passes it. ``None`` means the
    profile ``settings`` selects, for a one-off script: the recorded payload generator predates the parameter
    and must keep running unchanged.
    """
    profile = profile or load_profile(profile_path(settings))
    comparison = ComparisonOptions.from_settings(settings, profile)
    parse_reports: dict[Backend, ParseReport] = {}
    parsed: dict[Backend, ParsedArtifact | None] = {}
    outcomes: dict[Backend, tuple[ParsedArtifact, ParseReport]] = {}
    # Written here, once, before two parse threads could both find it missing and race to write it.
    ensure_identity(DataLayout(settings.data_root), document)
    if settings.mineru_url or settings.paddle_url:
        # At least one parser is a service on another machine, so the two lanes parse side by side; each
        # still waits for its own parser's lock. Two runner subprocesses share one lock, so there the lanes
        # stay one after the other and the progress marks say so.
        for backend in BACKENDS:
            on_stage(f"parse:{backend}", "running", "")
        with ContextThreadPoolExecutor(max_workers=len(BACKENDS), thread_name_prefix="paperfacts-parse") as pool:
            outcomes = _every_lane(
                {
                    backend: pool.submit(parse_document, document, backend, settings, force=force)
                    for backend in BACKENDS
                },
                "parse",
                on_stage=on_stage,
                describe=lambda outcome: _parse_detail(outcome[1]),
            )
    for backend in BACKENDS:
        if backend not in outcomes:
            on_stage(f"parse:{backend}", "running", "")
            outcomes[backend] = parse_document(document, backend, settings, force=force)
        parsed[backend], parse_report = outcomes[backend]
        parse_reports[backend] = parse_report
        on_stage(f"parse:{backend}", "done", _parse_detail(parse_report))

    # The figures stage runs beside the two extraction lanes: it waits on a different model for minutes per
    # chart and shares nothing with them but the parse. It is joined before this function returns, whatever
    # happened: a caller that is told the paper is finished (the web queue frees the document then) must not
    # have a thread still writing its readings. When the rest fails, `stop` keeps it from asking for more.
    stop_figures = threading.Event()
    figures_pool = ContextThreadPoolExecutor(max_workers=1, thread_name_prefix="paperfacts-figures")
    figures_future: Future[tuple[StageStatus, str]] | None = None
    if settings.figures_enabled:
        on_stage("figures", "running", "")
        artifact = next((parsed[backend] for backend in BACKENDS if parsed[backend] is not None), None)
        figures_future = figures_pool.submit(
            _read_figures_stage, document, settings, profile, force=force_figures, artifact=artifact, stop=stop_figures
        )
    else:
        figures = shown_figures(document.document_id, document.display_filename, settings, profile)
        on_stage("figures", "skipped", _figures_mark("skipped", "", figures))
    try:
        lanes, report = _extract_and_compare(document, settings, profile, comparison, force=force, on_stage=on_stage)
        if figures_future is not None:
            figures_status, figures_detail = figures_future.result()
            figures = shown_figures(document.document_id, document.display_filename, settings, profile)
            on_stage("figures", figures_status, _figures_mark(figures_status, figures_detail, figures))
    except BaseException:
        # The paper has failed: its charts would be read for nothing. The panels already out finish, and the
        # stage is then marked with its own outcome -- left "running", the job layer would mark it failed
        # with the error of a stage it had nothing to do with.
        stop_figures.set()
        figures_pool.shutdown(wait=True)
        if figures_future is not None and figures_future.done() and figures_future.exception() is None:
            status, detail = figures_future.result()
            _mark_quietly(on_stage, "figures", status, detail)
        raise
    finally:
        figures_pool.shutdown(wait=True)

    on_stage("export", "running", "")
    dataset = consolidate_document(document, lanes, report, comparison)
    layout = DataLayout(settings.data_root)
    excel_path = layout.dataset_path(document.document_id, profile.name)
    write_dataset([dataset], excel_path, profile, figure_rows=figures.rows if figures is not None else ())
    dataset_json_path: Path | None = None
    if dataset.incomplete:
        # The stored dataset is what marks a paper finished (stored.is_finished), so it is kept back for the
        # same reasons as the comparison, and an earlier run's table under these keys is removed with it: left
        # in place it would keep the paper finished. The workbook of this run is still written.
        layout.dataset_json_path(document.document_id, dataset.extractor_key, dataset.comparison_key).unlink(
            missing_ok=True
        )
        on_stage("export", "done", f"{excel_path}; not kept as finished: {dataset.incomplete}")
    else:
        dataset_json_path = store_dataset(layout, dataset)
        on_stage("export", "done", str(excel_path))
    return PipelineResult(
        parse_reports=parse_reports,
        lanes=lanes,
        report=report,
        dataset=dataset,
        excel_path=excel_path,
        dataset_json_path=dataset_json_path,
        figures=figures,
    )


def _parse_detail(report: ParseReport) -> str:
    return f"{report.block_count} blocks" + (" (cached)" if report.cache_hit else "")


def _mark_quietly(on_stage: StageCallback, stage: str, status: StageStatus, detail: str) -> None:
    """A stage mark made while a failure is already on its way out. The callback may raise (a stopped batch
    raises Cancelled from it); that must not replace the failure being reported."""
    try:
        on_stage(stage, status, detail)
    except Exception:
        logger.debug("stage mark %s=%s not delivered while failing", stage, status, exc_info=True)


def _every_lane[T](
    futures: Mapping[Backend, Future[T]],
    what: str,
    *,
    on_stage: StageCallback | None = None,
    describe: Callable[[T], str] = lambda result: "",
) -> dict[Backend, T]:
    """Both lanes' results, in BACKENDS order, or the first lane's failure.

    Every lane's outcome is collected before any of them is acted on, so an exception nobody asked for is
    logged rather than dropped by the garbage collector. A BaseException (a KeyboardInterrupt, say) still
    propagates straight out; leaving the caller's pool then waits for the other lane.

    With ``on_stage``, a failure first marks each lane ``{what}:{backend}`` with its own outcome: done (as
    ``describe`` puts it) or failed with its own error. Otherwise both stay "running", and the job layer marks
    every running stage failed with the first lane's error, a lane that succeeded included.
    """
    results: dict[Backend, T] = {}
    failures: list[tuple[Backend, Exception]] = []
    for backend in BACKENDS:
        try:
            results[backend] = futures[backend].result()
        except Exception as exc:
            failures.append((backend, exc))
    if failures and on_stage is not None:
        errors = dict(failures)
        for backend in BACKENDS:
            if backend in errors:
                error = errors[backend]
                _mark_quietly(on_stage, f"{what}:{backend}", "failed", f"{type(error).__name__}: {error}")
            else:
                _mark_quietly(on_stage, f"{what}:{backend}", "done", describe(results[backend]))
    if failures:
        # In BACKENDS order, so the first lane's failure wins; the rest are explanations.
        for backend, exc in failures[1:]:
            logger.warning("%s lane %s also failed with %s", what, backend, exc)
        raise failures[0][1]
    return results


def _extract_and_compare(
    document: DocumentInput,
    settings: Settings,
    profile: DomainProfile,
    comparison: ComparisonOptions,
    *,
    force: bool,
    on_stage: StageCallback,
) -> tuple[dict[Backend, LaneExtraction], ComparisonReport]:
    """Both extraction lanes, then the comparison: the part of :func:`run_document` that uses the LLM."""
    lanes: dict[Backend, LaneExtraction] = {}
    with build_llm_client(settings) as client:
        # One value for both lanes: whatever decides what a lane is asked cannot differ between them.
        options = ExtractionOptions.from_settings(settings, profile, client.model)
        # The two lanes are independent and both spend their time waiting on the model, so they overlap.
        # Nothing here touches pdf.py: extraction reads the stored artifact JSON and never opens the PDF,
        # so PDFium's process-wide lock is not involved. Both lanes are handed the same `settings` and the
        # same client, so they still ask the same model the same prompts at the same temperature -- only
        # the text differs, which is the measurement. Every on_stage call is made from this thread:
        # results are collected in BACKENDS order, so a caller's callback needs no locking of its own and
        # the stage marks stay in a fixed order. (web/jobs.JobManager would tolerate worker threads anyway
        # -- it replaces the frozen Job under its lock on every transition -- but not every caller is it.)
        for backend in BACKENDS:
            on_stage(f"extract:{backend}", "running", "")
        with ContextThreadPoolExecutor(max_workers=len(BACKENDS), thread_name_prefix="paperfacts-lane") as pool:
            futures: dict[Backend, Future[LaneExtraction]] = {
                backend: pool.submit(
                    extract_document, document, backend, settings, profile, client, force=force, options=options
                )
                for backend in BACKENDS
            }
            extracted = _every_lane(futures, "extract", on_stage=on_stage, describe=_lane_detail)
            for backend, lane in extracted.items():
                lanes[backend] = lane
                on_stage(f"extract:{backend}", "done", _lane_detail(lane))
        on_stage("compare", "running", "")
        report = compare_document(document, settings, profile, client, force=force, lanes=lanes, comparison=comparison)
    counts = report.counts
    detail = (
        f"agree {counts.agree} · conflict {counts.conflict} · ambiguous {counts.ambiguous} · missing {counts.missing}"
    )
    reason = incomplete_reason(lanes, report)
    if reason:
        # Not a failure of the paper: the report was not stored and the next run asks again.
        on_stage("compare", "failed", f"{detail}; {reason}, not stored")
    else:
        on_stage("compare", "done", detail)
    return lanes, report


def _lane_detail(lane: LaneExtraction) -> str:
    ungrounded = len(lane.ungrounded())
    failed = len(lane.failed_questions)
    return (
        f"{len(lane.samples)} samples"
        + (f", {ungrounded} ungrounded" if ungrounded else "")
        + (f", {failed} question{'s' if failed > 1 else ''} unanswered (asked again next run)" if failed else "")
    )


def corpus_workbook(datasets: Sequence[DocumentDataset], settings: Settings, profile: DomainProfile) -> bytes:
    """One workbook for several stored documents, each document's chart readings beside its data."""
    figure_views = (shown_figures(d.document_id, d.filename, settings, profile) for d in datasets)
    rows = [row for view in figure_views if view for row in view.rows]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{profile.name}.xlsx"
        write_dataset(datasets, path, profile, figure_rows=rows)
        return path.read_bytes()
