"""The command line entry point.

The CLI is a thin wrapper over the workflow, so this file only verifies the "thin" part: argument
parsing, backend expansion, error -> exit code, and that the output gives the user the path they need
next. The real parsing is replaced by a monkeypatched fake parser; not one subprocess is started.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from paperfacts.cli import BackendOption, app
from paperfacts.errors import ParserError
from paperfacts.models import Backend, DocumentInput, RawParseOutput
from paperfacts.parsers import Parser
from paperfacts.storage import DataLayout
from support.factories import RawOutputFactory, paddle_page_entry

runner = CliRunner()


class FakeParser(Parser):
    """Copies prepared native output into out_dir, standing in for the real parser."""

    def __init__(self, backend: Backend, source_dir: Path) -> None:
        self.backend = backend
        self.source_dir = source_dir

    def parse(self, document: DocumentInput, out_dir: Path, *, force: bool = False) -> RawParseOutput:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.copytree(self.source_dir, out_dir)
        return RawParseOutput.load(out_dir, self.backend)


@pytest.fixture
def prepared_raw_dirs(
    raw_output_factory: RawOutputFactory,
    mineru_content_list: list[dict[str, Any]],
    paddle_page_wrapped: dict[str, Any],
) -> dict[Backend, Path]:
    mineru = raw_output_factory.mineru(mineru_content_list, dir_name="prepared_mineru")
    paddle = raw_output_factory.paddle(
        [paddle_page_entry(0, paddle_page_wrapped, (1653, 2339))],
        dir_name="prepared_paddle",
    )
    return {"mineru": mineru.out_dir, "paddleocr_vl": paddle.out_dir}


@pytest.fixture
def fake_parsers(monkeypatch, prepared_raw_dirs: dict[Backend, Path]) -> list[Backend]:
    """Replace workflow.build_parser, recording the order each backend was requested in."""
    requested: list[Backend] = []

    def build(backend: Backend, settings) -> FakeParser:
        requested.append(backend)
        return FakeParser(backend, prepared_raw_dirs[backend])

    monkeypatch.setattr("paperfacts.workflow.build_parser", build)
    return requested


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "cli-data"


# ---- Backend expansion ----------------------------------------------------------------


def test_backend_option_expands_to_the_internal_backend_names():
    # "paddle" on the CLI is shorthand for paddleocr_vl; "both" runs the two lanes serially (16 GB of
    # memory cannot hold both model sets at once).
    assert BackendOption.mineru.backends() == ("mineru",)
    assert BackendOption.paddle.backends() == ("paddleocr_vl",)
    assert BackendOption.both.backends() == ("mineru", "paddleocr_vl")


# ---- parse --------------------------------------------------------------------------


def test_parse_reports_the_document_id_and_the_markdown_path(
    two_page_pdf: Path, data_root: Path, fake_parsers: list[Backend], document: DocumentInput
):
    result = runner.invoke(app, ["parse", str(two_page_pdf), "--backend", "mineru", "--data-root", str(data_root)])

    assert result.exit_code == 0, result.output
    assert f"document_id={document.document_id[:16]}" in result.output
    assert "[mineru] version=3.4.5 pages=2 blocks=11" in result.output
    assert str(DataLayout(data_root).markdown_path(document.document_id, "mineru")) in result.output


def test_parse_with_both_runs_each_backend_once_in_order(
    two_page_pdf: Path, data_root: Path, fake_parsers: list[Backend], document: DocumentInput
):
    result = runner.invoke(app, ["parse", str(two_page_pdf), "-b", "both", "--data-root", str(data_root)])

    assert result.exit_code == 0, result.output
    assert fake_parsers == ["mineru", "paddleocr_vl"]
    layout = DataLayout(data_root)
    assert layout.artifact_path(document.document_id, "mineru").is_file()
    assert layout.artifact_path(document.document_id, "paddleocr_vl").is_file()


def test_parse_defaults_to_both_backends(two_page_pdf: Path, data_root: Path, fake_parsers: list[Backend]):
    result = runner.invoke(app, ["parse", str(two_page_pdf), "--data-root", str(data_root)])

    assert result.exit_code == 0, result.output
    assert fake_parsers == ["mineru", "paddleocr_vl"]


def test_parse_honours_the_data_root_option_over_the_environment(
    two_page_pdf: Path, data_root: Path, fake_parsers: list[Backend], monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("PAPERFACTS_DATA_ROOT", str(tmp_path / "from-env"))

    runner.invoke(app, ["parse", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    assert data_root.is_dir()
    assert not (tmp_path / "from-env").exists()


def test_parse_falls_back_to_the_data_root_environment_variable(
    two_page_pdf: Path, fake_parsers: list[Backend], monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("PAPERFACTS_DATA_ROOT", str(tmp_path / "from-env"))

    result = runner.invoke(app, ["parse", str(two_page_pdf), "-b", "mineru"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "from-env").is_dir()


def test_parse_exits_with_code_one_when_the_parser_fails(two_page_pdf: Path, data_root: Path, monkeypatch):
    class ExplodingParser(Parser):
        backend: Backend = "mineru"

        def parse(self, document: DocumentInput, out_dir: Path, *, force: bool = False) -> RawParseOutput:
            raise ParserError("mineru", "run", "exit code 3")

    monkeypatch.setattr("paperfacts.workflow.build_parser", lambda backend, settings: ExplodingParser())

    result = runner.invoke(app, ["parse", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    assert result.exit_code == 1
    assert "failed" in result.output


def test_parse_rejects_a_missing_pdf(tmp_path: Path, data_root: Path):
    # typer's exists=True intercepts this before our code ever runs (exit code 2).
    result = runner.invoke(app, ["parse", str(tmp_path / "nope.pdf"), "--data-root", str(data_root)])

    assert result.exit_code == 2


def test_parse_rejects_an_unknown_backend_value(two_page_pdf: Path, data_root: Path):
    result = runner.invoke(app, ["parse", str(two_page_pdf), "-b", "tesseract", "--data-root", str(data_root)])

    assert result.exit_code == 2


def test_verbose_flag_is_accepted(two_page_pdf: Path, data_root: Path, fake_parsers: list[Backend]):
    result = runner.invoke(
        app, ["parse", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root), "--verbose"]
    )

    assert result.exit_code == 0, result.output


def test_force_flag_is_forwarded_to_the_parser(two_page_pdf: Path, data_root: Path, monkeypatch, prepared_raw_dirs):
    seen: list[bool] = []

    class RecordingParser(FakeParser):
        def parse(self, document: DocumentInput, out_dir: Path, *, force: bool = False) -> RawParseOutput:
            seen.append(force)
            return super().parse(document, out_dir, force=force)

    monkeypatch.setattr(
        "paperfacts.workflow.build_parser",
        lambda backend, settings: RecordingParser(backend, prepared_raw_dirs[backend]),
    )

    runner.invoke(app, ["parse", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root), "--force"])

    assert seen == [True]


# ---- overlay ------------------------------------------------------------------------


def test_overlay_writes_one_png_per_page_with_blocks(
    two_page_pdf: Path, data_root: Path, fake_parsers: list[Backend], document: DocumentInput
):
    runner.invoke(app, ["parse", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    result = runner.invoke(
        app, ["overlay", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root), "--dpi", "72"]
    )

    assert result.exit_code == 0, result.output
    overlay_dir = DataLayout(data_root).overlay_dir(document.document_id, "mineru")
    assert sorted(p.name for p in overlay_dir.iterdir()) == ["page_000.png", "page_001.png"]
    assert "2 overlays" in result.output


def test_overlay_pages_option_limits_the_rendered_pages(
    two_page_pdf: Path, data_root: Path, fake_parsers: list[Backend], document: DocumentInput
):
    runner.invoke(app, ["parse", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    result = runner.invoke(
        app,
        ["overlay", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root), "--dpi", "72", "--pages", "1"],
    )

    assert result.exit_code == 0, result.output
    overlay_dir = DataLayout(data_root).overlay_dir(document.document_id, "mineru")
    assert [p.name for p in overlay_dir.iterdir()] == ["page_001.png"]


def test_overlay_exits_with_code_one_when_the_artifact_is_missing(two_page_pdf: Path, data_root: Path):
    result = runner.invoke(app, ["overlay", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    assert result.exit_code == 1
    assert "paperfacts parse" in result.output


def test_help_lists_both_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "parse" in result.output
    assert "overlay" in result.output


def test_no_arguments_shows_help_instead_of_a_traceback():
    result = runner.invoke(app, [])

    assert result.exit_code != 0
    assert "Usage" in result.output


def test_fields_lists_every_field_of_the_profile(tco_profile):
    # The table lives in the profile; this is the one-glance check that an edit did what it was meant to.
    result = runner.invoke(app, ["fields"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    for spec in tco_profile.fields:
        assert any(line.startswith(spec.name) for line in lines)
        assert any(f"keywords: {', '.join(spec.keywords)}" in line for line in lines)
