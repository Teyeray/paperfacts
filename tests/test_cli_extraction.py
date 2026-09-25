"""The ``extract`` / ``compare`` / ``run`` commands.

The CLI is a thin wrapper over the workflow, so this file only verifies the "thin" part: the client's
lifecycle, an expected failure turning into red text + exit code 1, and that the output really contains
what the user needs for their next step (sample ids, fields, counts). Both the LLM and the parser are
monkeypatched away; not one real call or subprocess happens.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from paperfacts.cli import app
from paperfacts.config import Settings
from paperfacts.errors import ParserError
from paperfacts.models import BACKENDS, Backend, DocumentInput, RawParseOutput
from paperfacts.parsers import Parser
from paperfacts.storage import DataLayout
from support.extraction import make_artifact, make_lane
from support.factories import RawOutputFactory, make_block, paddle_page_entry
from support.llm import FakeLlmClient

runner = CliRunner()


def extraction_json(sample_id: str = "A", value: str = "12.5") -> str:
    return json.dumps(
        {
            "target": {"fields": [{"field": "density", "value_raw": "98.5", "unit_raw": "%"}]},
            "samples": [
                {
                    "sample_id": sample_id,
                    "label": "O2 100 sccm",
                    "fields": [{"field": "sheet_resistance", "value_raw": value, "unit_raw": "Ω/sq"}],
                }
            ],
        }
    )


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return tmp_path / "cli-data"


@pytest.fixture
def api_key(monkeypatch) -> None:
    """A fake key, so the CLI does not exit early in build_llm_client for lack of one."""
    monkeypatch.setenv("PAPERFACTS_LLM_API_KEY", "sk-test")


@pytest.fixture
def parsed(data_root: Path, document: DocumentInput) -> None:
    """Write both lane artifacts to disk directly, simulating "already parsed"."""
    layout = DataLayout(data_root)
    for backend in BACKENDS:
        blocks = (make_block(page=0, order=0, backend=backend, document_id=document.document_id, content="Sample A"),)
        artifact = make_artifact(blocks, backend=backend, document_id=document.document_id)
        artifact.write(layout.artifact_path(document.document_id, backend))


def install_fake_llm(monkeypatch, responses: list[str]) -> FakeLlmClient:
    """Replace build_llm_client: ``extract`` / ``compare`` call it directly in cli.py (the name imported
    via ``from ... import``), while ``run`` calls the one in the workflow module through
    ``workflow.run_document`` -- both need patching, or a real call to DeepSeek would go out."""
    client = FakeLlmClient(responses)
    monkeypatch.setattr("paperfacts.cli.build_llm_client", lambda settings: client)
    monkeypatch.setattr("paperfacts.workflow.build_llm_client", lambda settings: client)
    return client


# ---- extract ------------------------------------------------------------------------


def test_extract_prints_the_samples_the_fields_and_the_provenance_summary(
    monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed
):
    client = install_fake_llm(monkeypatch, [extraction_json()])

    result = runner.invoke(
        app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root), "--mode", "document"]
    )

    assert result.exit_code == 0, result.output
    assert "[mineru] samples=1" in result.output
    assert "A  (O2 100 sccm)" in result.output
    assert "sheet_resistance: 12.5 Ω/sq" in result.output
    assert "target.density: 98.5 %" in result.output
    assert client.call_count == 1


def test_extract_shows_the_normalized_value_next_to_the_raw_one(
    monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed
):
    install_fake_llm(monkeypatch, [extraction_json(value="1.2")])

    result = runner.invoke(
        app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root), "--mode", "document"]
    )

    assert "= 1.2 Ω/sq" in result.output


def test_extract_defaults_to_both_backends(monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed):
    client = install_fake_llm(monkeypatch, [extraction_json(), extraction_json()])

    result = runner.invoke(app, ["extract", str(two_page_pdf), "--data-root", str(data_root)])

    assert result.exit_code == 0, result.output
    assert "[mineru]" in result.output and "[paddleocr_vl]" in result.output
    assert client.call_count == 2


def test_extract_closes_the_client_when_it_is_done(monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed):
    # Failing to close the connection pool leaks sockets over a long-running batch job; the CLI uses
    # `with build_llm_client(...)`.
    client = install_fake_llm(monkeypatch, [extraction_json()])

    runner.invoke(app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    assert client.closed is True


def test_extract_forwards_force_to_the_llm(monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed):
    client = install_fake_llm(monkeypatch, [extraction_json(), extraction_json()])

    runner.invoke(app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])
    runner.invoke(app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root), "--force"])

    assert client.refreshes == [False, True]


def test_the_passes_option_reaches_the_settings_used_for_extraction(
    monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed
):
    # --passes has to survive CLI parsing -> _settings() -> the Settings object extract_document actually
    # receives, or asking for self-consistency passes on the command line would silently do nothing.
    monkeypatch.delenv("PAPERFACTS_EXTRACTION_PASSES", raising=False)
    captured: list[Settings] = []

    def fake_extract_document(
        document: DocumentInput, backend: Backend, settings: Settings, client, *, force: bool = False
    ):
        captured.append(settings)
        return make_lane(backend=backend)

    monkeypatch.setattr("paperfacts.cli.extract_document", fake_extract_document)
    install_fake_llm(monkeypatch, [])

    result = runner.invoke(
        app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root), "--passes", "3"]
    )

    assert result.exit_code == 0, result.output
    assert captured[0].extraction_passes == 3


def test_the_passes_option_defaults_to_the_settings_default_when_omitted(
    monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed
):
    monkeypatch.delenv("PAPERFACTS_EXTRACTION_PASSES", raising=False)
    captured: list[Settings] = []

    def fake_extract_document(
        document: DocumentInput, backend: Backend, settings: Settings, client, *, force: bool = False
    ):
        captured.append(settings)
        return make_lane(backend=backend)

    monkeypatch.setattr("paperfacts.cli.extract_document", fake_extract_document)
    install_fake_llm(monkeypatch, [])

    runner.invoke(app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    assert captured[0].extraction_passes == 1


def test_the_mode_option_reaches_the_settings_used_for_extraction(
    monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed
):
    # --mode decides which extractor runs and therefore which key the facts are filed under; if it stopped
    # at the CLI boundary the run would quietly do the other thing.
    monkeypatch.delenv("PAPERFACTS_EXTRACTION_MODE", raising=False)
    captured: list[Settings] = []

    def fake_extract_document(
        document: DocumentInput, backend: Backend, settings: Settings, client, *, force: bool = False
    ):
        captured.append(settings)
        return make_lane(backend=backend)

    monkeypatch.setattr("paperfacts.cli.extract_document", fake_extract_document)
    install_fake_llm(monkeypatch, [])

    result = runner.invoke(
        app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root), "--mode", "passage"]
    )

    assert result.exit_code == 0, result.output
    assert captured[0].extraction_mode == "passage"


def test_the_mode_option_defaults_to_the_settings_default_when_omitted(
    monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed
):
    monkeypatch.delenv("PAPERFACTS_EXTRACTION_MODE", raising=False)
    captured: list[Settings] = []

    def fake_extract_document(
        document: DocumentInput, backend: Backend, settings: Settings, client, *, force: bool = False
    ):
        captured.append(settings)
        return make_lane(backend=backend)

    monkeypatch.setattr("paperfacts.cli.extract_document", fake_extract_document)
    install_fake_llm(monkeypatch, [])

    runner.invoke(app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    assert captured[0].extraction_mode == "passage"


def test_the_mode_option_reaches_compare(monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed):
    monkeypatch.delenv("PAPERFACTS_EXTRACTION_MODE", raising=False)
    captured: list[Settings] = []

    def fake_compare_document(document: DocumentInput, settings: Settings, client, *, force: bool = False):
        captured.append(settings)
        # Stop here rather than build a whole report: the flag has already been observed, and an expected
        # error is the cheapest way back out of the command.
        raise ParserError("mineru", "compare", "stop here")

    monkeypatch.setattr("paperfacts.cli.compare_document", fake_compare_document)
    install_fake_llm(monkeypatch, [])

    runner.invoke(app, ["compare", str(two_page_pdf), "--data-root", str(data_root), "--mode", "passage"])

    assert captured[0].extraction_mode == "passage"


def test_the_mode_option_reaches_run(monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed):
    # `run` goes through workflow.run_document rather than the CLI's own extract_document, so the flag has
    # a second path to survive.
    monkeypatch.delenv("PAPERFACTS_EXTRACTION_MODE", raising=False)
    captured: list[Settings] = []

    def fake_run_document(document: DocumentInput, settings: Settings, **kwargs):
        captured.append(settings)
        raise ParserError("mineru", "run", "stop here, the flag has already been observed")

    monkeypatch.setattr("paperfacts.cli.run_document", fake_run_document)
    install_fake_llm(monkeypatch, [])

    runner.invoke(app, ["run", str(two_page_pdf), "--data-root", str(data_root), "--mode", "passage"])

    assert captured[0].extraction_mode == "passage"


def test_extract_without_a_key_exits_one_and_points_at_the_key_file(
    monkeypatch, two_page_pdf: Path, data_root: Path, tmp_path: Path, parsed
):
    # No key set, and repo_root pointed at an empty directory, so this never reads the real
    # deepseek_api_key that happens to live in the repository root.
    monkeypatch.delenv("PAPERFACTS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("PAPERFACTS_REPO_ROOT", str(tmp_path / "empty-repo"))

    result = runner.invoke(app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    assert result.exit_code == 1
    assert "failed" in result.output
    assert "deepseek_api_key" in result.output


def test_extract_before_parse_exits_one_with_the_next_step(monkeypatch, two_page_pdf: Path, data_root: Path, api_key):
    install_fake_llm(monkeypatch, [])

    result = runner.invoke(app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    assert result.exit_code == 1
    assert "paperfacts parse" in result.output


def test_extract_reports_a_model_that_will_not_produce_valid_json(
    monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed
):
    install_fake_llm(monkeypatch, ["not json", "still not json"])

    result = runner.invoke(app, ["extract", str(two_page_pdf), "-b", "mineru", "--data-root", str(data_root)])

    assert result.exit_code == 1
    assert "twice failed" in result.output


# ---- compare ------------------------------------------------------------------------


def test_compare_prints_the_counts_the_pairs_and_the_field_lines(
    monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed
):
    client = install_fake_llm(monkeypatch, [extraction_json(value="12.5"), extraction_json(value="99")])

    result = runner.invoke(app, ["compare", str(two_page_pdf), "--data-root", str(data_root), "--mode", "document"])

    assert result.exit_code == 0, result.output
    assert "counts:" in result.output
    assert "match A ↔ A" in result.output
    assert "conflict" in result.output
    assert "sheet_resistance" in result.output
    assert client.call_count == 2  # same sample_id on both lanes -> exact match, no matching model call


def test_compare_writes_the_report_where_the_layout_says(
    monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed, document: DocumentInput
):
    install_fake_llm(monkeypatch, [extraction_json(), extraction_json()])

    runner.invoke(app, ["compare", str(two_page_pdf), "--data-root", str(data_root)])

    comparisons = DataLayout(data_root).doc_dir(document.document_id) / "comparisons"
    assert [p.suffix for p in comparisons.iterdir()] == [".json"]


def test_compare_before_parse_exits_one(monkeypatch, two_page_pdf: Path, data_root: Path, api_key):
    install_fake_llm(monkeypatch, [])

    result = runner.invoke(app, ["compare", str(two_page_pdf), "--data-root", str(data_root)])

    assert result.exit_code == 1
    assert "paperfacts parse" in result.output


def test_compare_closes_the_client(monkeypatch, two_page_pdf: Path, data_root: Path, api_key, parsed):
    client = install_fake_llm(monkeypatch, [extraction_json(), extraction_json()])

    runner.invoke(app, ["compare", str(two_page_pdf), "--data-root", str(data_root)])

    assert client.closed is True


# ---- run ----------------------------------------------------------------------------


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
def fake_parsers(
    monkeypatch,
    raw_output_factory: RawOutputFactory,
    mineru_content_list: list[dict[str, Any]],
    paddle_page_wrapped: dict[str, Any],
) -> None:
    prepared: dict[Backend, Path] = {
        "mineru": raw_output_factory.mineru(mineru_content_list, dir_name="prepared_mineru").out_dir,
        "paddleocr_vl": raw_output_factory.paddle(
            [paddle_page_entry(0, paddle_page_wrapped, (1653, 2339))], dir_name="prepared_paddle"
        ).out_dir,
    }
    monkeypatch.setattr(
        "paperfacts.workflow.build_parser", lambda backend, settings: FakeParser(backend, prepared[backend])
    )


def test_run_goes_from_pdf_to_comparison_in_one_command(
    monkeypatch, two_page_pdf: Path, data_root: Path, api_key, fake_parsers, document: DocumentInput
):
    client = install_fake_llm(monkeypatch, [extraction_json(), extraction_json()])

    result = runner.invoke(app, ["run", str(two_page_pdf), "--data-root", str(data_root), "--mode", "document"])

    assert result.exit_code == 0, result.output
    assert f"document_id={document.document_id[:16]}" in result.output
    # Stage lines share the same stage names as the web progress bar (workflow.stage_names); "running" is
    # not printed.
    assert "[parse:mineru] done" in result.output
    assert "[parse:paddleocr_vl] done" in result.output
    assert "[extract:mineru] done 1 samples" in result.output
    assert "[compare] done agree" in result.output
    assert "running" not in result.output
    assert "counts:" in result.output
    assert client.call_count == 2


def test_run_stops_with_code_one_when_a_parser_fails(monkeypatch, two_page_pdf: Path, data_root: Path, api_key):
    class ExplodingParser(Parser):
        def parse(self, document: DocumentInput, out_dir: Path, *, force: bool = False) -> RawParseOutput:
            raise ParserError("mineru", "run", "exit code 3")

    monkeypatch.setattr("paperfacts.workflow.build_parser", lambda backend, settings: ExplodingParser())
    install_fake_llm(monkeypatch, [])

    result = runner.invoke(app, ["run", str(two_page_pdf), "--data-root", str(data_root)])

    assert result.exit_code == 1
    assert "failed" in result.output


def test_run_without_a_key_fails_after_parsing_rather_than_silently(
    monkeypatch, two_page_pdf: Path, data_root: Path, tmp_path: Path, fake_parsers
):
    monkeypatch.delenv("PAPERFACTS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("PAPERFACTS_REPO_ROOT", str(tmp_path / "empty-repo"))

    result = runner.invoke(app, ["run", str(two_page_pdf), "--data-root", str(data_root)])

    assert result.exit_code == 1
    assert "deepseek_api_key" in result.output
    assert "[parse:mineru] done" in result.output  # parsing already finished, so it is not redone for nothing


# ---- Help -------------------------------------------------------------------------


def test_the_help_lists_all_five_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("parse", "overlay", "extract", "compare", "run"):
        assert command in result.output


@pytest.mark.parametrize("command", ["extract", "compare", "run"])
def test_a_missing_pdf_is_rejected_before_any_work(command, tmp_path: Path, data_root: Path):
    result = runner.invoke(app, [command, str(tmp_path / "nope.pdf"), "--data-root", str(data_root)])

    assert result.exit_code == 2


@pytest.mark.parametrize(("flag", "expected"), [("--figures", True), ("--no-figures", False), (None, False)])
def test_the_figures_flag_reaches_run(monkeypatch, two_page_pdf: Path, data_root: Path, flag, expected):
    monkeypatch.delenv("PAPERFACTS_FIGURES_ENABLED", raising=False)
    captured: list[Settings] = []

    def fake_run_document(document: DocumentInput, settings: Settings, **kwargs):
        captured.append(settings)
        raise ParserError("mineru", "run", "stop here, the flag has already been observed")

    monkeypatch.setattr("paperfacts.cli.run_document", fake_run_document)

    runner.invoke(app, ["run", str(two_page_pdf), "--data-root", str(data_root), *([flag] if flag else [])])

    assert captured[0].figures_enabled is expected


@pytest.mark.parametrize(("flag", "expected"), [("--figures", True), ("--no-figures", False)])
def test_the_figures_flag_reaches_batch(monkeypatch, two_page_pdf: Path, data_root: Path, flag, expected):
    captured: list[Settings] = []

    def fake_run_batch(source: Path, settings: Settings, **kwargs):
        captured.append(settings)
        raise ParserError("mineru", "run", "stop here")

    monkeypatch.setattr("paperfacts.cli.run_batch", fake_run_batch)

    runner.invoke(app, ["batch", str(two_page_pdf.parent), "--data-root", str(data_root), flag])

    assert captured[0].figures_enabled is expected


@pytest.mark.parametrize(
    ("flags", "force", "force_figures"), [(["--force"], True, False), (["--force-figures"], False, True)]
)
def test_force_and_force_figures_are_separate(
    monkeypatch, two_page_pdf: Path, data_root: Path, flags, force, force_figures
):
    captured: list[dict] = []

    def fake_run_document(document: DocumentInput, settings: Settings, **kwargs):
        captured.append(kwargs)
        raise ParserError("mineru", "run", "stop here")

    monkeypatch.setattr("paperfacts.cli.run_document", fake_run_document)

    runner.invoke(app, ["run", str(two_page_pdf), "--data-root", str(data_root), *flags])

    assert (captured[0]["force"], captured[0]["force_figures"]) == (force, force_figures)
