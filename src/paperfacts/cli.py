"""The ``paperfacts`` command line.

    paperfacts parse   paper.pdf --backend both   # both lanes -> data/docs/<sha>/parsed/
    paperfacts overlay paper.pdf --backend both   # bbox overlays -> data/docs/<sha>/overlays/
    paperfacts run     paper.pdf                  # parse, extract and compare in one go
    paperfacts serve                              # the web interface
    paperfacts prompts --field thickness          # what the model is asked, rendered; no model call

Where the parsers run is decided by the environment (see :mod:`paperfacts.config`): subprocesses from
``runners/`` by default, or long-running services once ``PAPERFACTS_MINERU_URL`` /
``PAPERFACTS_PADDLE_URL`` are set.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import Iterable
from enum import StrEnum
from logging.handlers import BufferingHandler
from pathlib import Path
from typing import Annotated, NoReturn, assert_never

import typer

from paperfacts.batch import run_batch
from paperfacts.config import DEFAULT_REPO_ROOT, ENV_PREFIX, EXTRACTION_MODES, Settings
from paperfacts.errors import ConfigError, PaperFactsError, ParserError
from paperfacts.llm import OFFLINE_MISSES, set_max_in_flight
from paperfacts.models import Backend, DocumentInput
from paperfacts.overlay import render_overlays
from paperfacts.parsers import install_runner_cleanup
from paperfacts.profile import PROFILES_DIRNAME, DomainProfile, load_profile
from paperfacts.prompts import (
    extraction_system_prompt,
    field_system_prompt,
    field_user_prompt,
    inventory_system_prompt,
    matching_system_prompt,
    render_field_table,
)
from paperfacts.report import render_lane, render_report
from paperfacts.storage import DataLayout, write_text_atomic
from paperfacts.workflow import (
    StageStatus,
    build_llm_client,
    compare_document,
    extract_document,
    load_artifact,
    load_run_profile,
    parse_document,
    run_document,
)

app = typer.Typer(
    help="Extract traceable sample-level facts from scientific PDFs using two independent parsers.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    # This process owns the parser subprocesses: make sure Ctrl-C or a kill takes them with it, rather
    # than leaving a multi-gigabyte runner behind.
    install_runner_cleanup()


class BackendOption(StrEnum):
    """Backend choice on the command line; ``paddle`` is shorthand for ``paddleocr_vl``."""

    mineru = "mineru"
    paddle = "paddle"
    both = "both"

    def backends(self) -> tuple[Backend, ...]:
        match self:
            case BackendOption.mineru:
                return ("mineru",)
            case BackendOption.paddle:
                return ("paddleocr_vl",)
            case BackendOption.both:
                # Serial: two model sets do not fit in a laptop's memory at once, and on a server these
                # are HTTP calls where serial is fast enough anyway.
                return ("mineru", "paddleocr_vl")
            case _:
                assert_never(self)  # a new member without a case fails here instead of returning None


class ModeOption(StrEnum):
    """Extraction mode on the command line. Typer needs an enum, so the literals live in two places; the
    check below fails at import if they ever drift apart."""

    document = "document"
    passage = "passage"


if {mode.value for mode in ModeOption} != set(EXTRACTION_MODES):
    raise RuntimeError(f"CLI modes {[m.value for m in ModeOption]} do not match config's {list(EXTRACTION_MODES)}")


PdfArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="the paper")]
SourceArg = Annotated[Path, typer.Argument(exists=True, readable=True, help="PDF or directory (searched recursively)")]
OutputOpt = Annotated[Path | None, typer.Option("--output", "-o", help="Excel workbook (.xlsx)")]
BackendOpt = Annotated[BackendOption, typer.Option("--backend", "-b", help="which parser lane to run")]
DataRootOpt = Annotated[
    Path | None, typer.Option("--data-root", help="data directory; defaults to $PAPERFACTS_DATA_ROOT or ./data")
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="print INFO logs")]
ProfileOpt = Annotated[
    str | None,
    typer.Option(
        "--profile",
        metavar="NAME_OR_PATH",
        help="domain profile: a name under profiles/, or a path to a profile file; "
        "default: profile in config.json or $PAPERFACTS_PROFILE",
    ),
]
PassesOpt = Annotated[
    int | None,
    typer.Option("--passes", min=1, help="extract each lane this many times and keep the majority (costs N calls)"),
]
ModeOpt = Annotated[
    ModeOption | None,
    typer.Option("--mode", help="how to ask the model: the whole paper at once, or one question per field"),
]
FiguresOpt = Annotated[
    bool | None,
    typer.Option(
        "--figures/--no-figures",
        help="read property-vs-condition charts with the vision model (about a minute per chart); "
        "default: figures.enabled in config.json",
    ),
]
ForceFiguresOpt = Annotated[
    bool,
    typer.Option(
        "--force-figures",
        help="read the charts again, ignoring stored readings (vision model, about a minute per chart); "
        "--force alone does not re-read them",
    ),
]
JobsOpt = Annotated[
    int | None,
    typer.Option(
        "--jobs",
        "-j",
        min=1,
        help="papers processed at once; parsing stays one paper per parser, model requests share "
        "llm.max_in_flight (default: web.max_parallel_documents in config.json)",
    ),
]
ForceOpt = Annotated[
    bool, typer.Option("--force", help="ignore caches and redo this step (extraction re-calls the LLM, which costs)")
]
OfflineOpt = Annotated[
    bool,
    typer.Option(
        "--offline",
        help="answer every model request from the LLM cache and fail on a miss instead of sending it; "
        "ends with 'offline misses: N' (same as PAPERFACTS_LLM_OFFLINE=1)",
    ),
]
# Where an offline run writes its misses as JSON. Read here rather than kept in Settings: it is an output of
# one invocation, like --output, and changes nothing the pipeline does.
OFFLINE_REPORT_ENV = "PAPERFACTS_LLM_OFFLINE_REPORT"
# Expected failures (configuration, LLM, parser, missing files) become red text and exit 1; anything else
# keeps its traceback.
REPORTABLE_ERRORS = (PaperFactsError, FileNotFoundError)


def _settings(
    data_root: Path | None,
    passes: int | None = None,
    mode: ModeOption | None = None,
    figures: bool | None = None,
    offline: bool = False,
    force: bool = False,
    profile: str | None = None,
) -> Settings:
    settings = _env_settings()
    changes: dict[str, object] = {}
    if profile is not None:
        changes["profile"] = profile
    if offline:
        changes["llm_offline"] = True
    if figures is not None:
        changes["figures_enabled"] = figures
    if data_root is not None:
        changes["data_root"] = data_root
    if passes is not None:
        changes["extraction_passes"] = passes
    if mode is not None:
        changes["extraction_mode"] = mode.value
    settings = dataclasses.replace(settings, **changes) if changes else settings
    if settings.llm_offline and force:
        # A forced request skips the cache by definition, so an offline forced run is all misses.
        typer.secho("--force re-asks the model; it cannot be combined with offline replay", fg="red", err=True)
        raise typer.Exit(code=2)
    # Every command reads its settings here, once: the one place this process sizes the in-flight limit.
    set_max_in_flight(settings.llm_max_in_flight)
    # And the one place it starts its miss record: a command's summary counts its own misses only.
    OFFLINE_MISSES.clear()
    return settings


def _env_settings() -> Settings:
    """config.json under the environment. A broken configuration is one red line naming the key and the file,
    for every command, not a traceback."""
    try:
        return Settings.from_env()
    except ConfigError as exc:
        _fail("config", exc)


def _profile(settings: Settings) -> DomainProfile:
    """The profile this command runs under, loaded once here and passed to everything it calls."""
    try:
        return load_run_profile(settings)
    except ConfigError as exc:
        _fail("profile", exc)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _fail(name: str, exc: Exception) -> NoReturn:
    typer.secho(f"[{name}] failed: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from exc


def _offline_summary(settings: Settings) -> None:
    """The last line of an offline run, printed whether it finished or failed: a replay proves "zero model
    calls" only by saying how many requests it could not answer, and which (by key, in the report file)."""
    if not settings.llm_offline:
        return
    misses = OFFLINE_MISSES.snapshot()
    typer.echo(f"offline misses: {len(misses)}")
    report = os.environ.get(OFFLINE_REPORT_ENV, "").strip()
    if report:
        write_text_atomic(
            Path(report), json.dumps([dataclasses.asdict(miss) for miss in misses], ensure_ascii=False, indent=2)
        )
        typer.echo(f"offline misses -> {report}")


def _echo_lines(lines: Iterable[str]) -> None:
    for line in lines:
        typer.echo(line)


@app.command()
def parse(
    pdf: PdfArg,
    backend: BackendOpt = BackendOption.both,
    force: Annotated[bool, typer.Option("--force", help="ignore the cache and re-run the parser")] = False,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Parse a PDF into Markdown with provenance markers, a block list and a full artifact."""
    _configure_logging(verbose)
    settings = _settings(data_root)
    document = DocumentInput.from_path(pdf)
    typer.echo(f"document_id={document.document_id[:16]}  {pdf.name}")

    for name in backend.backends():
        try:
            _, report = parse_document(document, name, settings, force=force)
        except ParserError as exc:
            _fail(name, exc)
        cache = "cache" if report.cache_hit else f"{report.runtime_s:.1f}s"
        typer.echo(
            f"[{name}] version={report.backend_version} pages={report.page_count} "
            f"blocks={report.block_count} ({cache}) types={report.type_counts}"
        )
        typer.echo(f"        markdown -> {report.markdown_path}")


@app.command()
def overlay(
    pdf: PdfArg,
    backend: BackendOpt = BackendOption.both,
    dpi: Annotated[int | None, typer.Option(help="overlay rendering DPI; defaults to overlay.dpi")] = None,
    pages: Annotated[str | None, typer.Option(help="only these pages, comma separated, 0-based")] = None,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Draw parsed block boxes back onto page images, to check by eye that they line up. Needs parse."""
    _configure_logging(verbose)
    settings = _settings(data_root)
    document = DocumentInput.from_path(pdf)
    layout = DataLayout(settings.data_root)
    page_list = [int(p) for p in pages.split(",")] if pages else None

    for name in backend.backends():
        try:
            artifact = load_artifact(document, name, settings)
        except FileNotFoundError as exc:
            _fail(name, exc)
        target = layout.overlay_dir(document.document_id, name)
        written = render_overlays(pdf, artifact, target, dpi=dpi or settings.overlay_dpi, pages=page_list)
        typer.echo(f"[{name}] {len(written)} overlays -> {target}")


@app.command()
def extract(
    pdf: PdfArg,
    backend: BackendOpt = BackendOption.both,
    force: ForceOpt = False,
    passes: PassesOpt = None,
    mode: ModeOpt = None,
    profile: ProfileOpt = None,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Extract sample-level records from parsed Markdown with the LLM. Needs parse."""
    _configure_logging(verbose)
    settings = _settings(data_root, passes, mode, force=force, profile=profile)
    domain = _profile(settings)
    document = DocumentInput.from_path(pdf)
    try:
        with build_llm_client(settings) as client:
            for name in backend.backends():
                _echo_lines(render_lane(extract_document(document, name, settings, domain, client, force=force)))
    except REPORTABLE_ERRORS as exc:
        _fail("extract", exc)


@app.command()
def compare(
    pdf: PdfArg,
    force: ForceOpt = False,
    passes: PassesOpt = None,
    mode: ModeOpt = None,
    profile: ProfileOpt = None,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Match samples across the two lanes and compare their fields. Needs parse."""
    _configure_logging(verbose)
    settings = _settings(data_root, passes, mode, force=force, profile=profile)
    domain = _profile(settings)
    document = DocumentInput.from_path(pdf)
    try:
        with build_llm_client(settings) as client:
            report = compare_document(document, settings, domain, client, force=force)
    except REPORTABLE_ERRORS as exc:
        _fail("compare", exc)
    _echo_lines(render_report(report))


@app.command()
def run(
    pdf: PdfArg,
    force: ForceOpt = False,
    passes: PassesOpt = None,
    mode: ModeOpt = None,
    figures: FiguresOpt = None,
    force_figures: ForceFiguresOpt = False,
    offline: OfflineOpt = False,
    profile: ProfileOpt = None,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Parse, extract, compare and automatically save a consolidated Excel workbook."""
    _configure_logging(verbose)
    settings = _settings(data_root, passes, mode, figures, offline, force or force_figures, profile)
    domain = _profile(settings)
    document = DocumentInput.from_path(pdf)
    typer.echo(f"document_id={document.document_id[:16]}  {pdf.name}")

    def on_stage(stage: str, status: StageStatus, detail: str) -> None:
        if status != "running":
            typer.echo(f"[{stage}] {status} {detail}".rstrip())

    try:
        result = run_document(document, settings, domain, force=force, force_figures=force_figures, on_stage=on_stage)
    except REPORTABLE_ERRORS as exc:
        _offline_summary(settings)
        _fail("run", exc)
    for lane in result.lanes.values():
        _echo_lines(render_lane(lane))
    _echo_lines(render_report(result.report))
    if result.figures is not None and result.figures.rows:
        warning = result.figures.warning()
        typer.echo(
            f"figure readings: {len(result.figures.rows)} (approximate; sheet 图中读数)"
            + (f"; {warning}" if warning else "")
        )
    typer.echo(f"Excel -> {result.excel_path}")
    if result.dataset.incomplete:
        typer.echo(f"Incomplete, not kept as finished: {result.dataset.incomplete}; the next run asks again")
        unanswered = [row for row in result.dataset.quality_rows if row.get("decision") == "unanswered"]
        if unanswered:
            # The comparison counts above print these as missing: the failing lane simply has no value there.
            fields = ", ".join(dict.fromkeys(str(row["field"]) for row in unanswered))
            typer.echo(f"{len(unanswered)} cells unanswered ({fields}), counted as missing in the comparison above")
    _offline_summary(settings)


def _batch_summary(
    source: Path,
    settings: Settings,
    output: Path | None,
    *,
    force: bool,
    export_only: bool,
    force_figures: bool = False,
    jobs: int | None = None,
) -> None:
    def on_stage(stage: str, status: StageStatus, detail: str) -> None:
        typer.echo(f"[{stage}] {status} {detail}".rstrip())

    domain = _profile(settings)
    try:
        result = run_batch(
            source,
            settings,
            domain,
            output=output,
            force=force,
            force_figures=force_figures,
            export_only=export_only,
            jobs=jobs or settings.max_parallel_documents,
            on_stage=on_stage,
        )
    except (*REPORTABLE_ERRORS, OSError) as exc:
        _offline_summary(settings)
        _fail("export" if export_only else "batch", exc)
    incomplete = sum(1 for document in result.documents if document.incomplete)
    typer.echo(
        f"Completed: {len(result.documents)} papers"
        + (f" ({incomplete} incomplete, asked again next run)" if incomplete else "")
        + f"; failed: {len(result.failures)}; duplicates skipped: {result.duplicate_count}"
    )
    typer.echo(f"Excel -> {result.excel_path}")
    _offline_summary(settings)
    if result.failures:
        raise typer.Exit(code=1)


@app.command()
def batch(
    source: SourceArg,
    output: OutputOpt = None,
    force: ForceOpt = False,
    passes: PassesOpt = None,
    mode: ModeOpt = None,
    figures: FiguresOpt = None,
    force_figures: ForceFiguresOpt = False,
    jobs: JobsOpt = None,
    offline: OfflineOpt = False,
    profile: ProfileOpt = None,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Recursively process all PDFs and save one paper per row in Excel, with a merged sample sheet."""
    _configure_logging(verbose)
    settings = _settings(data_root, passes, mode, figures, offline, force or force_figures, profile)
    _batch_summary(source, settings, output, force=force, export_only=False, force_figures=force_figures, jobs=jobs)


@app.command()
def export(
    source: SourceArg,
    output: OutputOpt = None,
    passes: PassesOpt = None,
    mode: ModeOpt = None,
    profile: ProfileOpt = None,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Re-export current cached results to Excel, without parser or LLM calls."""
    _configure_logging(verbose)
    settings = _settings(data_root, passes, mode, profile=profile)
    # One paper at a time: an export only reads the caches, so parallel papers would buy nothing.
    _batch_summary(source, settings, output, force=False, export_only=True, jobs=1)


@app.command()
def fields(profile: ProfileOpt = None) -> None:
    """List the field table of the profile a run would load, so an edit to it can be checked at a glance."""
    # Not _settings(): a listing asks no model, so it leaves the in-flight limit and the miss record alone.
    settings = _env_settings()
    if profile is not None:
        settings = dataclasses.replace(settings, profile=profile)
    for spec in _profile(settings).fields:
        unit = spec.canonical_unit or "-"
        tolerance = f"rel={spec.rel_tol:g} abs={spec.abs_tol:g}" if spec.kind == "numeric" else "-"
        hint = f"  condition: {spec.condition_hint}" if spec.condition_hint else ""
        typer.echo(
            f"{spec.name:<26} {spec.group:<8} {spec.kind:<12} {unit:<8} {tolerance:<22} {spec.bare_number}{hint}"
        )
        typer.echo(f"    keywords: {', '.join(spec.keywords) or '-'}")


def _profile_line(domain: DomainProfile) -> str:
    counts = f"{len(domain.paper_fields)} paper + {len(domain.sample_fields)} sample fields"
    return f"{domain.name:<20} {domain.maturity:<10} {counts:<28} {domain.content_hash[:12]}  {domain.title_zh}"


@app.command()
def profiles(
    check: Annotated[
        Path | None,
        typer.Option("--check", metavar="PATH", help="validate this profile file instead of listing profiles/"),
    ] = None,
) -> None:
    """List the profiles under profiles/, or validate one file while writing it (--check)."""
    if check is not None:
        # The loader logs its warnings (a large field table, say); an author checking a file wants them here.
        handler = BufferingHandler(capacity=1000)
        handler.setLevel(logging.WARNING)
        package_logger = logging.getLogger("paperfacts")
        package_logger.addHandler(handler)
        try:
            domain = load_profile(check)
        except ConfigError as exc:
            # Validation names every problem, one per line; each gets its own "error:".
            for line in str(exc).splitlines():
                typer.secho(f"error: {line}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        finally:
            package_logger.removeHandler(handler)
        typer.echo(_profile_line(domain))
        _echo_lines(f"warning: {record.getMessage()}" for record in handler.buffer)
        typer.echo("ok")
        return
    # Not the settings: listing the profiles must work while config.json is broken, which is when one is checked.
    directory = Path(os.environ.get(f"{ENV_PREFIX}REPO_ROOT", "").strip() or DEFAULT_REPO_ROOT) / PROFILES_DIRNAME
    failed = False
    for path in sorted(directory.glob("*.json")):
        try:
            typer.echo(_profile_line(load_profile(path)))
        except ConfigError as exc:
            # One broken file should not hide the others from the listing.
            for line in str(exc).splitlines():
                typer.secho(f"{path.stem:<20} invalid: {line}", fg=typer.colors.RED, err=True)
            failed = True
    if failed:
        raise typer.Exit(code=1)


@app.command()
def prompts(
    profile: ProfileOpt = None,
    field: Annotated[
        str | None, typer.Option("--field", metavar="NAME", help="the per-field question for this field instead")
    ] = None,
) -> None:
    """Print the system prompts a profile renders, exactly as the model gets them, each under the mode that sends
    it. No model is called."""
    settings = _env_settings()
    if profile is not None:
        settings = dataclasses.replace(settings, profile=profile)
    domain = _profile(settings)
    if field is not None:
        spec = domain.by_name.get(field)
        if spec is None:
            typer.secho(f"no field {field!r} in {domain.name}; it has: {', '.join(domain.by_name)}", fg="red", err=True)
            raise typer.Exit(code=1)
        sections = {
            "field system prompt (passage mode)": field_system_prompt(domain),
            f"field line ({field})": render_field_table((spec,)),
            # The question's framing; the two placeholders are what a run fills from the paper.
            f"field user prompt ({field}, passage mode)": field_user_prompt(spec, "<sample list>", "<excerpts>"),
        }
    else:
        # Passage mode (the default) sends the inventory and field prompts and never the extraction prompt;
        # document mode sends only the extraction prompt. Matching is the compare stage's, under either mode.
        sections = {
            "inventory system prompt (passage mode)": inventory_system_prompt(domain),
            "field system prompt (passage mode)": field_system_prompt(domain),
            "extraction system prompt (document mode)": extraction_system_prompt(domain),
            "matching system prompt (compare, both modes)": matching_system_prompt(domain),
        }
    for title, text in sections.items():
        typer.echo(f"===== {title} =====")
        typer.echo(text)
        typer.echo("")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="bind address; defaults to server.host")] = None,
    port: Annotated[int | None, typer.Option(help="port to listen on; defaults to server.port")] = None,
    profile: ProfileOpt = None,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Serve the web interface: upload a PDF, run the pipeline, browse the aligned facts and their sources."""
    import uvicorn

    from paperfacts.web.app import create_app

    _configure_logging(verbose)
    settings = _settings(data_root, profile=profile)
    domain = _profile(settings)
    host = host or settings.server_host
    port = port or settings.server_port
    typer.echo(
        f"PaperFacts UI -> http://{host}:{port}   "
        f"(data_root={settings.data_root}, model={settings.llm_model}, "
        f"profile={domain.name} {domain.content_hash[:12]})"
    )
    try:
        web_app = create_app(settings, profile=domain)
    except ConfigError as exc:
        _fail("serve", exc)
    uvicorn.run(web_app, host=host, port=port, log_level="info" if verbose else "warning")


if __name__ == "__main__":
    app()
