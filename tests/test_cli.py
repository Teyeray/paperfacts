"""The command line entry point.

The CLI is a thin wrapper over the workflow, so this file only verifies the "thin" part: argument
parsing, backend expansion, error -> exit code, and that the output gives the user the path they need
next. The real parsing is replaced by a monkeypatched fake parser; not one subprocess is started.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from paperfacts.cli import BackendOption, app
from paperfacts.errors import ConfigError, ParserError
from paperfacts.models import Backend, DocumentInput, RawParseOutput
from paperfacts.parsers import Parser
from paperfacts.storage import DataLayout
from support.factories import RawOutputFactory, paddle_page_entry
from support.profiles import SHIPPED_PROFILE_PATH, make_profile, profile_data

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


def test_fields_takes_the_profile_flag(tmp_path: Path):
    path = write_demo(tmp_path / "demo.json")

    result = runner.invoke(app, ["fields", "--profile", str(path)])

    assert result.exit_code == 0
    names = [line.split()[0] for line in result.output.splitlines() if not line.startswith(" ")]
    assert names == ["precursor_purity", "coating_thickness", "solvent"]


# ---- Authoring tools: profiles, profiles --check, prompts ----------------------------------------------

# Read-only: the recording test_prompt_snapshot.py pins byte for byte, so what `prompts` prints is what is sent.
RECORDED_PROMPTS: dict[str, str] = json.loads(
    (Path(__file__).parent / "fixtures" / "prompts" / "snapshot.json").read_text(encoding="utf-8")
)


def write_demo(path: Path, data: dict[str, Any] | None = None) -> Path:
    path.write_text(json.dumps(data or profile_data(), ensure_ascii=False), encoding="utf-8")
    return path


def prompt_sections(output: str) -> dict[str, str]:
    """``paperfacts prompts`` output split at its ``===== title =====`` lines, each text without the blank line
    that separates it from the next section."""
    sections: dict[str, list[str]] = {}
    current: list[str] = []
    for line in output.splitlines():
        heading = re.fullmatch(r"===== (.+) =====", line)
        if heading:
            current = sections.setdefault(heading.group(1), [])
        else:
            current.append(line)
    return {title: "\n".join(lines[:-1]) for title, lines in sections.items()}


def test_profiles_lists_the_shipped_profile_with_its_counts_and_hash(tco_profile):
    result = runner.invoke(app, ["profiles"])

    assert result.exit_code == 0
    line = next(line for line in result.output.splitlines() if line.startswith("tco "))
    assert tco_profile.maturity in line and tco_profile.title_zh in line
    assert f"{len(tco_profile.paper_fields)} paper + {len(tco_profile.sample_fields)} sample fields" in line
    assert tco_profile.content_hash[:12] in line


def test_profiles_check_accepts_a_valid_file(tmp_path: Path):
    result = runner.invoke(app, ["profiles", "--check", str(write_demo(tmp_path / "demo.json"))])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines[0].startswith("demo ") and lines[-1] == "ok"


def test_profiles_check_prints_the_warnings_of_a_valid_file(tmp_path: Path):
    field = profile_data()["fields"][1]
    data = profile_data({"fields": [field | {"name": f"f{i}"} for i in range(41)]})

    result = runner.invoke(app, ["profiles", "--check", str(write_demo(tmp_path / "demo.json", data))])

    assert result.exit_code == 0
    assert any(line.startswith("warning:") and "41 fields" in line for line in result.output.splitlines())


def test_profiles_check_prints_the_error_and_exits_one(tmp_path: Path):
    # The name must be the file's stem: the one mistake a copied profile is sure to have.
    result = runner.invoke(app, ["profiles", "--check", str(write_demo(tmp_path / "battery.json"))])

    assert result.exit_code == 1
    assert "error:" in result.output and "battery.json" in result.output
    assert "ok" not in result.output.splitlines()


def test_prompts_prints_the_system_prompts_exactly_as_recorded():
    result = runner.invoke(app, ["prompts", "--profile", str(SHIPPED_PROFILE_PATH)])

    assert result.exit_code == 0
    assert prompt_sections(result.output) == {
        "inventory system prompt (passage mode)": RECORDED_PROMPTS["inventory_system"],
        "field system prompt (passage mode)": RECORDED_PROMPTS["field_system"],
        "extraction system prompt (document mode)": RECORDED_PROMPTS["extraction_system"],
        "matching system prompt (compare, both modes)": RECORDED_PROMPTS["matching_system"],
    }


def test_prompts_for_one_field_prints_its_system_prompt_its_line_and_the_question():
    result = runner.invoke(app, ["prompts", "--profile", str(SHIPPED_PROFILE_PATH), "--field", "thickness"])

    assert result.exit_code == 0
    sections = prompt_sections(result.output)
    assert sections["field system prompt (passage mode)"] == RECORDED_PROMPTS["field_system"]
    line = sections["field line (thickness)"]
    assert line.startswith("- `thickness`")
    assert f"Field to extract:\n{line}\n\n" in RECORDED_PROMPTS["field_user:thickness"]
    question = sections["field user prompt (thickness, passage mode)"]
    assert question.startswith(f"Field to extract:\n{line}\n\nSamples this paper reports:\n<sample list>\n\n")
    assert "<excerpts>" in question and question.endswith("Return the JSON object now.")


def test_prompts_for_one_field_use_that_profiles_own_wording():
    # The range sentence names where an implausible number comes from; a battery field must not be told "layer".
    battery = SHIPPED_PROFILE_PATH.with_name("battery_cathode.json")
    result = runner.invoke(app, ["prompts", "--profile", str(battery), "--field", "calcination_temperature"])

    assert result.exit_code == 0
    line = prompt_sections(result.output)["field line (calcination_temperature)"]
    assert "a different electrode component, test condition or quantity" in line
    assert "layer" not in line


def test_prompts_for_an_unknown_field_names_the_fields_there_are():
    result = runner.invoke(app, ["prompts", "--profile", str(SHIPPED_PROFILE_PATH), "--field", "colour"])

    assert result.exit_code == 1
    assert "colour" in result.output and "thickness" in result.output


# ---- A broken configuration --------------------------------------------------------------------------------


@pytest.fixture
def broken_config(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text("{", encoding="utf-8")
    monkeypatch.setenv("PAPERFACTS_CONFIG", str(path))
    return path


@pytest.mark.parametrize(
    "command",
    [
        ["parse", "{pdf}"],
        ["overlay", "{pdf}"],
        ["extract", "{pdf}"],
        ["compare", "{pdf}"],
        ["run", "{pdf}"],
        ["batch", "{dir}"],
        ["export", "{dir}"],
        ["fields"],
        ["prompts"],
        ["serve"],
    ],
)
def test_a_broken_configuration_is_one_red_line_and_exit_one(broken_config: Path, two_page_pdf: Path, command):
    arguments = [part.format(pdf=two_page_pdf, dir=two_page_pdf.parent) for part in command]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 1, result.output
    assert result.output.startswith("[config] failed:") and "not valid JSON" in result.output
    assert len(result.output.strip().splitlines()) == 1
    assert not isinstance(result.exception, ConfigError)


def test_profiles_lists_without_a_valid_configuration(broken_config: Path, tco_profile):
    result = runner.invoke(app, ["profiles"])

    assert result.exit_code == 0
    assert any(line.startswith("tco ") for line in result.output.splitlines())


def test_profiles_check_prints_every_problem_of_a_file(tmp_path: Path):
    changes = {
        "maturity": "beta",
        "fields.0.kind": "number",
        "fields.2.name": "Solvent",
        "retrieval.condition_keywords": 3,
    }

    result = runner.invoke(app, ["profiles", "--check", str(write_demo(tmp_path / "demo.json", profile_data(changes)))])

    errors = [line for line in result.output.splitlines() if line.startswith("error:")]
    assert result.exit_code == 1
    assert len(errors) == 4
    assert all(str(tmp_path / "demo.json") in line for line in errors)
    assert any("maturity" in line for line in errors) and any("condition_keywords" in line for line in errors)
    assert any("'precursor_purity'" in line and "kind" in line for line in errors)
    assert any("'Solvent'" in line for line in errors)


def test_profiles_check_prints_warnings_from_the_whole_package(monkeypatch, tmp_path: Path):
    def load_and_warn(path: Path):
        logging.getLogger("paperfacts.units").warning("a unit warning")
        return make_profile()

    monkeypatch.setattr("paperfacts.cli.load_profile", load_and_warn)

    result = runner.invoke(app, ["profiles", "--check", str(write_demo(tmp_path / "demo.json"))])

    assert result.exit_code == 0
    assert "warning: a unit warning" in result.output.splitlines()
