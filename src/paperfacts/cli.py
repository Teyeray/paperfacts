"""The ``paperfacts`` command line.

    paperfacts parse   paper.pdf --backend both   # both lanes -> data/docs/<sha>/parsed/
    paperfacts overlay paper.pdf --backend both   # bbox overlays -> data/docs/<sha>/overlays/
    paperfacts run     paper.pdf                  # parse, extract and compare in one go
    paperfacts serve                              # the web interface

Where the parsers run is decided by the environment (see :mod:`paperfacts.config`): subprocesses from
``runners/`` by default, or long-running services once ``PAPERFACTS_MINERU_URL`` /
``PAPERFACTS_PADDLE_URL`` are set.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn, assert_never

import typer

from paperfacts.config import Settings
from paperfacts.errors import PaperFactsError
from paperfacts.models.artifact import Backend, DocumentInput
from paperfacts.parsers.base import ParserError
from paperfacts.report import render_lane, render_report
from paperfacts.storage.paths import DataLayout
from paperfacts.verification.overlay import DEFAULT_OVERLAY_DPI, render_overlays
from paperfacts.workflow import (
    StageStatus,
    build_llm_client,
    compare_document,
    extract_document,
    load_artifact,
    parse_document,
    run_document,
)

app = typer.Typer(
    help="Extract traceable sample-level facts from scientific PDFs using two independent parsers.",
    no_args_is_help=True,
)


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


PdfArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="the paper")]
BackendOpt = Annotated[BackendOption, typer.Option("--backend", "-b", help="which parser lane to run")]
DataRootOpt = Annotated[
    Path | None, typer.Option("--data-root", help="data directory; defaults to $PAPERFACTS_DATA_ROOT or ./data")
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="print INFO logs")]
PassesOpt = Annotated[
    int | None,
    typer.Option("--passes", min=1, help="extract each lane this many times and keep the majority (costs N calls)"),
]
ForceOpt = Annotated[
    bool, typer.Option("--force", help="ignore caches and redo this step (extraction re-calls the LLM, which costs)")
]
# Expected failures (configuration, LLM, parser, missing files) become red text and exit 1; anything else
# keeps its traceback.
REPORTABLE_ERRORS = (PaperFactsError, FileNotFoundError)


def _settings(data_root: Path | None, passes: int | None = None) -> Settings:
    settings = Settings.from_env()
    changes: dict[str, object] = {}
    if data_root is not None:
        changes["data_root"] = data_root
    if passes is not None:
        changes["extraction_passes"] = passes
    return dataclasses.replace(settings, **changes) if changes else settings


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _fail(name: str, exc: Exception) -> NoReturn:
    typer.secho(f"[{name}] failed: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from exc


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
    dpi: Annotated[int, typer.Option(help="overlay rendering DPI")] = DEFAULT_OVERLAY_DPI,
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
        written = render_overlays(pdf, artifact, target, dpi=dpi, pages=page_list)
        typer.echo(f"[{name}] {len(written)} overlays -> {target}")


@app.command()
def extract(
    pdf: PdfArg,
    backend: BackendOpt = BackendOption.both,
    force: ForceOpt = False,
    passes: PassesOpt = None,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Extract sample-level records from parsed Markdown with the LLM. Needs parse."""
    _configure_logging(verbose)
    settings = _settings(data_root, passes)
    document = DocumentInput.from_path(pdf)
    try:
        with build_llm_client(settings) as client:
            for name in backend.backends():
                _echo_lines(render_lane(extract_document(document, name, settings, client, force=force)))
    except REPORTABLE_ERRORS as exc:
        _fail("extract", exc)


@app.command()
def compare(
    pdf: PdfArg,
    force: ForceOpt = False,
    passes: PassesOpt = None,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Match samples across the two lanes and compare their fields. Needs parse."""
    _configure_logging(verbose)
    settings = _settings(data_root, passes)
    document = DocumentInput.from_path(pdf)
    try:
        with build_llm_client(settings) as client:
            report = compare_document(document, settings, client, force=force)
    except REPORTABLE_ERRORS as exc:
        _fail("compare", exc)
    _echo_lines(render_report(report))


@app.command()
def run(
    pdf: PdfArg,
    force: ForceOpt = False,
    passes: PassesOpt = None,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Parse both lanes, extract both, then match and compare. Every step is cached; --force redoes all."""
    _configure_logging(verbose)
    settings = _settings(data_root, passes)
    document = DocumentInput.from_path(pdf)
    typer.echo(f"document_id={document.document_id[:16]}  {pdf.name}")

    def on_stage(stage: str, status: StageStatus, detail: str) -> None:
        if status != "running":
            typer.echo(f"[{stage}] {status} {detail}".rstrip())

    try:
        result = run_document(document, settings, force=force, on_stage=on_stage)
    except REPORTABLE_ERRORS as exc:
        _fail("run", exc)
    for lane in result.lanes.values():
        _echo_lines(render_lane(lane))
    _echo_lines(render_report(result.report))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="bind address; use 0.0.0.0 to allow other machines")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="port to listen on")] = 8000,
    data_root: DataRootOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Serve the web interface: upload a PDF, run the pipeline, browse the aligned facts and their sources."""
    import uvicorn

    from paperfacts.web.app import create_app

    _configure_logging(verbose)
    settings = _settings(data_root)
    typer.echo(f"PaperFacts UI -> http://{host}:{port}   (data_root={settings.data_root}, model={settings.llm_model})")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info" if verbose else "warning")


if __name__ == "__main__":
    app()
