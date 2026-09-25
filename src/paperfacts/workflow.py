"""Orchestration: parser -> adapter -> disk, then figure reading (opt-in), extraction, comparison, export.

This is the only place that decides which parser implementation runs: a configured ``*_url`` means an HTTP
service (a GPU server), an empty one means the ``runners/`` script as a subprocess (a workstation).

:func:`run_document` is the single pipeline. The CLI and the web job both call it, so they cannot drift.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from paperfacts.adapters import convert, render_markdown
from paperfacts.compare import ComparisonReport, compare_lanes
from paperfacts.config import Settings
from paperfacts.dataset import DocumentDataset, consolidate_document, write_dataset, write_dataset_json
from paperfacts.errors import Cancelled, ConfigError, PaperFactsError, ParserError
from paperfacts.extract import extract_lane, informative_blocks
from paperfacts.figures import MAX_TOKENS as FIGURE_MAX_TOKENS
from paperfacts.figures import RETRY_ATTEMPTS as FIGURE_RETRY_ATTEMPTS
from paperfacts.figures import TEMPERATURE as FIGURE_TEMPERATURE
from paperfacts.figures import FigureReadings, FiguresView, read_figures
from paperfacts.figures import figure_rows as figure_rows_of
from paperfacts.grounding import block_adjacency, ground_lane
from paperfacts.keys import ExtractionOptions, comparison_key, extractor_key, extractor_key_for, figure_key_for
from paperfacts.llm import LlmClient, OpenAICompatibleClient, VisionClient
from paperfacts.matching import match_samples
from paperfacts.models import BACKENDS, Backend, DocumentInput, NormalizedBBox, ParsedArtifact
from paperfacts.normalize import normalize_lane
from paperfacts.parsers import MinerUHttpParser, PaddleHttpParser, Parser, SubprocessParser, default_runner_script
from paperfacts.pdf import crop_region, png_bytes, read_geometry, render_page
from paperfacts.records import LaneExtraction
from paperfacts.storage import DataLayout, ensure_identity
from paperfacts.threads import ContextThreadPoolExecutor

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
    parser = build_parser(backend, settings)

    markdown_path = layout.markdown_path(document.document_id, backend)
    artifact_path = layout.artifact_path(document.document_id, backend)
    raw_dir = layout.raw_dir(document.document_id, backend)

    stored = _stored_artifact_for_missing_raw(parser, document, backend, artifact_path, raw_dir, force=force)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    if stored is not None:
        artifact, cache_hit, runtime_s = stored, True, 0.0
        if not markdown_path.is_file():
            # The artifact alone is not a complete document directory; re-render rather than leave a hole.
            markdown_path.write_text(render_markdown(artifact.blocks), encoding="utf-8")
    else:
        if not document.pdf_path.is_file():
            # Only the real parse path needs the file; say so plainly instead of failing inside the parser.
            raise ParserError(backend, "input", "PDF not available; re-upload to re-parse")
        clock = time.monotonic()
        raw = parser.parse(document, raw_dir, force=force)
        artifact = convert(raw, document, read_geometry(document.pdf_path))
        runtime_s = time.monotonic() - clock
        cache_hit = raw.cache_hit
        markdown_path.write_text(render_markdown(artifact.blocks), encoding="utf-8")
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
        # The same blocks, and so the same neighbours, extraction grounded against: adjacency over every block
        # would put page furniture between two halves of a sentence.
        blocks = informative_blocks(artifact.blocks)
        lane = ground_lane(
            lane, {block.source_id: block.content for block in blocks}, adjacency=block_adjacency(blocks)
        )
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
    options = ExtractionOptions.from_settings(settings, client.model)
    key = extractor_key(options)
    artifact = load_artifact(document, backend, settings)
    if not force:
        cached = read_lane(layout, document.document_id, backend, key, artifact=artifact)
        if cached is not None:
            logger.info("extraction cache_hit backend=%s doc=%s", backend, document.document_id[:16])
            return cached

    lane = extract_lane(artifact, client, options, concurrency=settings.llm_concurrency, refresh=force)
    lane.write(layout.extraction_path(document.document_id, backend, key))
    return normalize_lane(lane)


def compare_document(
    document: DocumentInput,
    settings: Settings,
    client: LlmClient,
    *,
    force: bool = False,
    lanes: Mapping[Backend, LaneExtraction] | None = None,
) -> ComparisonReport:
    """Match samples with the model, compare fields by rule, store the report.

    The report path carries both keys, so changing a tolerance recomputes the comparison without paying for
    extraction again and cannot serve a stale verdict. ``force`` redoes matching and comparison only;
    extraction has its own cache and its own force.

    ``lanes`` lets a caller that already holds both extractions hand them over instead of having them
    loaded again; without it the lanes are read through :func:`extract_document`, whose cached path
    rechecks grounding on read. The standalone ``compare`` CLI command relies on that loader.
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

    if lanes is None:
        lanes = {backend: extract_document(document, backend, settings, client) for backend in BACKENDS}
    lane_a, lane_b = lanes[BACKEND_A], lanes[BACKEND_B]
    matching = match_samples(lane_a, lane_b, client, refresh=force)
    report = compare_lanes(lane_a, lane_b, matching)
    report.write(path)
    logger.info("compared doc=%s counts=%s", document.document_id[:16], report.counts.model_dump())
    return report


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
    )


def _figure_artifact(
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
        rows=figure_rows_of(readings, filename=filename, stale=stale, orphaned=orphaned),
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
    artifact = artifact or _figure_artifact(document, settings)
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
    *,
    force: bool,
    artifact: ParsedArtifact | None,
    stop: threading.Event | None = None,
) -> tuple[StageStatus, str]:
    """The figures stage's work, run beside the extraction lanes. Never raises: it is an opt-in extra, and
    a chart the vision model could not read must not cost the paper its extraction. It reports its outcome
    instead of calling ``on_stage`` itself, so every stage mark still comes from the calling thread."""
    try:
        artifact = artifact or _figure_artifact(document, settings)
        with build_vision_client(settings) as client:
            readings = read_document_figures(document, settings, client, force=force, artifact=artifact, stop=stop)
    except Cancelled as exc:
        logger.info("figure reading stopped for %s: the rest of the paper failed", document.display_filename)
        return "failed", str(exc)
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
    dataset_json_path: Path
    # The chart readings shown with this paper, when there are any (see shown_figures).
    figures: FiguresView | None = None


def _store_dataset(layout: DataLayout, dataset: DocumentDataset) -> Path:
    """The web UI reads the consolidated table from disk, so every path that produces one writes it.

    The dataset carries the keys it was built under, which is what the JSON file is named after: an
    export made with different settings lands beside the old one instead of overwriting it.
    """
    path = layout.dataset_json_path(dataset.document_id, dataset.extractor_key, dataset.comparison_key)
    write_dataset_json(dataset, path)
    return path


def _ignore_stage(stage: str, status: StageStatus, detail: str) -> None:
    pass


def run_document(
    document: DocumentInput,
    settings: Settings,
    *,
    force: bool = False,
    force_figures: bool = False,
    on_stage: StageCallback = _ignore_stage,
) -> PipelineResult:
    """Run both lanes and automatically export consolidated data. Expensive steps are cached.

    ``force`` redoes parsing, extraction and comparison; ``force_figures`` re-reads the charts. They are
    separate because each costs minutes of a different model, and wanting one redone rarely means the other.
    """
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
            )
    for backend in BACKENDS:
        if backend not in outcomes:
            on_stage(f"parse:{backend}", "running", "")
            outcomes[backend] = parse_document(document, backend, settings, force=force)
        parsed[backend], parse_report = outcomes[backend]
        parse_reports[backend] = parse_report
        cached = " (cached)" if parse_report.cache_hit else ""
        on_stage(f"parse:{backend}", "done", f"{parse_report.block_count} blocks{cached}")

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
            _read_figures_stage, document, settings, force=force_figures, artifact=artifact, stop=stop_figures
        )
    else:
        figures = shown_figures(document.document_id, document.display_filename, settings)
        on_stage("figures", "skipped", _figures_mark("skipped", "", figures))
    try:
        lanes, report = _extract_and_compare(document, settings, force=force, on_stage=on_stage)
        if figures_future is not None:
            figures_status, figures_detail = figures_future.result()
            figures = shown_figures(document.document_id, document.display_filename, settings)
            on_stage("figures", figures_status, _figures_mark(figures_status, figures_detail, figures))
    except BaseException:
        # The paper has failed: its charts would be read for nothing. The panels already out finish.
        stop_figures.set()
        raise
    finally:
        figures_pool.shutdown(wait=True)

    on_stage("export", "running", "")
    dataset = consolidate_document(document, lanes, report)
    layout = DataLayout(settings.data_root)
    excel_path = layout.dataset_path(document.document_id)
    write_dataset([dataset], excel_path, figure_rows=figures.rows if figures is not None else ())
    dataset_json_path = _store_dataset(layout, dataset)
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


def _every_lane[T](futures: Mapping[Backend, Future[T]], what: str) -> dict[Backend, T]:
    """Both lanes' results, in BACKENDS order, or the first lane's failure.

    Every lane's outcome is collected before any of them is acted on, so an exception nobody asked for is
    logged rather than dropped by the garbage collector. A BaseException (a KeyboardInterrupt, say) still
    propagates straight out; leaving the caller's pool then waits for the other lane.
    """
    results: dict[Backend, T] = {}
    failures: list[tuple[Backend, Exception]] = []
    for backend in BACKENDS:
        try:
            results[backend] = futures[backend].result()
        except Exception as exc:
            failures.append((backend, exc))
    if failures:
        # In BACKENDS order, so the first lane's failure wins; the rest are explanations.
        for backend, exc in failures[1:]:
            logger.warning("%s lane %s also failed with %s", what, backend, exc)
        raise failures[0][1]
    return results


def _extract_and_compare(
    document: DocumentInput, settings: Settings, *, force: bool, on_stage: StageCallback
) -> tuple[dict[Backend, LaneExtraction], ComparisonReport]:
    """Both extraction lanes, then the comparison: the part of :func:`run_document` that uses the LLM."""
    lanes: dict[Backend, LaneExtraction] = {}
    with build_llm_client(settings) as client:
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
                backend: pool.submit(extract_document, document, backend, settings, client, force=force)
                for backend in BACKENDS
            }
            extracted = _every_lane(futures, "extraction")
            for backend, lane in extracted.items():
                lanes[backend] = lane
                ungrounded = len(lane.ungrounded())
                detail = f"{len(lane.samples)} samples" + (f", {ungrounded} ungrounded" if ungrounded else "")
                on_stage(f"extract:{backend}", "done", detail)
        on_stage("compare", "running", "")
        report = compare_document(document, settings, client, force=force, lanes=lanes)
    counts = report.counts
    on_stage(
        "compare",
        "done",
        f"agree {counts.agree} · conflict {counts.conflict} · ambiguous {counts.ambiguous} · missing {counts.missing}",
    )
    return lanes, report


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
        raise FileNotFoundError(f"no current comparison for {document.display_filename}; run `paperfacts run` first")
    report = ComparisonReport.read(report_path)
    lanes: dict[Backend, LaneExtraction] = {}
    for backend in BACKENDS:
        lane = read_lane(layout, document.document_id, backend, key)
        if lane is None:
            raise FileNotFoundError(f"no current {backend} extraction for {document.display_filename}")
        lanes[backend] = lane
    # Grounding is rechecked on read, so comparison must use those same refreshed values.
    report = compare_lanes(lanes[BACKEND_A], lanes[BACKEND_B], report.matching)
    dataset = consolidate_document(document, lanes, report)
    # An offline re-export is how a code-only change reaches the browser, so refresh the web view too.
    _store_dataset(layout, dataset)
    return dataset


def run_batch(
    source: Path,
    settings: Settings,
    *,
    output: Path | None = None,
    force: bool = False,
    force_figures: bool = False,
    export_only: bool = False,
    jobs: int = 1,
    on_stage: StageCallback = _ignore_stage,
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
    output = output or DataLayout(settings.data_root).batch_dataset_path()
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
            report(prefix, "done", "")
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
                dataset = export_document(document, settings)
                figures = shown_figures(document.document_id, document.display_filename, settings)
            else:
                result = run_document(
                    document,
                    settings,
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
