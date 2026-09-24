"""The figures stage inside the pipeline: where its input comes from, what it stores, and that nothing it
does -- switched off, failing, or half-failing -- can cost the paper its other stages.

The vision client is :class:`support.vision.FakeVisionClient`; the PDF is a generated blank one, so the
crops are real PNGs of an empty page and the requests never leave the process.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from paperfacts.config import Settings
from paperfacts.errors import ConfigError, LlmError
from paperfacts.figures import FigureReadings
from paperfacts.keys import figure_key_for
from paperfacts.models import Backend, DocumentInput, NormalizedBBox, PageGeometry, ParsedArtifact
from paperfacts.storage import DataLayout
from paperfacts.workflow import read_document_figures, run_document, stored_figures
from support.factories import make_block
from support.vision import NOT_A_CHART, FakeVisionClient, chart_answer
from test_workflow_run import install_fake_pipeline

BOX = NormalizedBBox(x1=0.1, y1=0.1, x2=0.6, y2=0.5)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_root=tmp_path / "data", repo_root=tmp_path, llm_api_key="sk-test", figures_enabled=True)


def store_artifact(document: DocumentInput, settings: Settings, backend: Backend = "mineru") -> None:
    blocks = (
        make_block(
            page=0, order=0, type="figure", content="a.jpg", bbox=BOX, backend=backend, document_id=document.document_id
        ),
        make_block(
            page=0,
            order=1,
            type="caption",
            content="Fig. 2 Sheet resistance of the films versus O2 flow.",
            bbox=BOX,
            backend=backend,
            document_id=document.document_id,
        ),
    )
    artifact = ParsedArtifact(
        document_id=document.document_id,
        backend=backend,
        backend_version="x",
        pages=(PageGeometry(index=0, width_pt=595, height_pt=842), PageGeometry(index=1, width_pt=612, height_pt=792)),
        blocks=blocks,
    )
    artifact.write(DataLayout(settings.data_root).artifact_path(document.document_id, backend))


def figures_file(document: DocumentInput, settings: Settings) -> Path:
    return DataLayout(settings.data_root).figures_path(document.document_id, figure_key_for(settings, "fake-vl"))


# ---- read_document_figures ---------------------------------------------------------------------------


def test_the_charts_are_cropped_from_the_pdf_read_and_stored(document: DocumentInput, settings: Settings):
    store_artifact(document, settings)
    client = FakeVisionClient(chart_answer())

    readings = read_document_figures(document, settings, client)

    assert [call.image_png[:8] for call in client.calls] == [b"\x89PNG\r\n\x1a\n"]
    assert {r.source_id for r in readings.readings} == {"mineru_p0_b0"}
    assert FigureReadings.read(figures_file(document, settings)) == readings


def test_stored_readings_are_served_without_asking_again(document: DocumentInput, settings: Settings):
    store_artifact(document, settings)
    read_document_figures(document, settings, FakeVisionClient(chart_answer()))
    again = FakeVisionClient(chart_answer())

    read_document_figures(document, settings, again)

    assert again.calls == []


def test_a_stored_file_with_a_failed_request_is_read_again(document: DocumentInput, settings: Settings):
    store_artifact(document, settings)
    first = read_document_figures(document, settings, FakeVisionClient(LlmError("HTTP 504")))
    retry = FakeVisionClient(chart_answer())

    second = read_document_figures(document, settings, retry)

    assert not first.complete
    assert len(retry.calls) == 1 and second.complete and second.readings


def test_force_re_asks_the_model(document: DocumentInput, settings: Settings):
    store_artifact(document, settings)
    read_document_figures(document, settings, FakeVisionClient(chart_answer()))
    forced = FakeVisionClient(chart_answer())

    read_document_figures(document, settings, forced, force=True)

    assert [call.refresh for call in forced.calls] == [True]


def test_paddle_figure_blocks_stand_in_when_there_is_no_mineru_artifact(document: DocumentInput, settings: Settings):
    store_artifact(document, settings, backend="paddleocr_vl")

    readings = read_document_figures(document, settings, FakeVisionClient(chart_answer()))

    assert readings.backend == "paddleocr_vl"
    assert {r.source_id for r in readings.readings} == {"paddleocr_vl_p0_b0"}


def test_without_the_pdf_nothing_can_be_cropped(document: DocumentInput, settings: Settings, tmp_path: Path):
    store_artifact(document, settings)
    moved = document.model_copy(update={"pdf_path": tmp_path / "gone.pdf"})

    with pytest.raises(FileNotFoundError, match="PDF not available"):
        read_document_figures(moved, settings, FakeVisionClient(chart_answer()))


def test_without_a_parse_nothing_can_be_selected(document: DocumentInput, settings: Settings):
    with pytest.raises(FileNotFoundError, match="no parse artifact"):
        read_document_figures(document, settings, FakeVisionClient(chart_answer()))


# ---- The stage inside run_document ----------------------------------------------------------------------


def run(monkeypatch, document: DocumentInput, settings: Settings, client: object):
    install_fake_pipeline(monkeypatch)
    monkeypatch.setattr("paperfacts.workflow.build_vision_client", lambda settings: client)
    marks: list[tuple[str, str, str]] = []
    result = run_document(document, settings, on_stage=lambda s, st, d: marks.append((s, st, d)))
    return [mark for mark in marks if mark[0] == "figures"], result, marks


def test_the_stage_reads_between_parse_and_extraction_and_the_rows_reach_the_dataset(
    monkeypatch, document: DocumentInput, settings: Settings
):
    store_artifact(document, settings)
    client = FakeVisionClient(chart_answer())

    figures, result, marks = run(monkeypatch, document, settings, client)

    assert figures == [("figures", "running", ""), ("figures", "done", "2 readings from 1 panels")]
    names = [name for name, status, _ in marks if status == "running"]
    assert names.index("figures") == names.index("parse:paddleocr_vl") + 1
    assert len(result.dataset.figure_rows) == 2
    assert result.figures is not None and client.closed


def test_a_failing_stage_is_marked_failed_and_the_paper_still_finishes(
    monkeypatch, document: DocumentInput, settings: Settings
):
    # No artifact on disk: the stage cannot select anything and raises inside.
    figures, result, marks = run(monkeypatch, document, settings, FakeVisionClient(chart_answer()))

    assert figures[-1][:2] == ("figures", "failed")
    assert "no parse artifact" in figures[-1][2]
    assert marks[-1][:2] == ("export", "done")
    assert result.dataset.figure_rows == () and result.figures is None


def test_a_client_that_cannot_be_built_fails_only_the_stage(monkeypatch, document: DocumentInput, settings: Settings):
    install_fake_pipeline(monkeypatch)

    def no_key(settings: Settings):
        raise ConfigError("no LLM API key")

    monkeypatch.setattr("paperfacts.workflow.build_vision_client", no_key)
    marks: list[tuple[str, str, str]] = []
    run_document(document, settings, on_stage=lambda s, st, d: marks.append((s, st, d)))

    assert ("figures", "failed", "ConfigError: no LLM API key") in marks
    assert marks[-1][:2] == ("export", "done")


def test_a_panel_whose_request_failed_marks_the_stage_failed_but_keeps_what_was_read(
    monkeypatch, document: DocumentInput, settings: Settings
):
    store_artifact(document, settings)

    figures, result, _ = run(monkeypatch, document, settings, FakeVisionClient(LlmError("timeout")))

    assert figures[-1] == ("figures", "failed", "0 readings from 1 panels, 1 requests failed")
    assert result.figures is not None


def test_a_refused_chart_is_a_done_stage_with_nothing_read(monkeypatch, document: DocumentInput, settings: Settings):
    store_artifact(document, settings)

    figures, result, _ = run(monkeypatch, document, settings, FakeVisionClient(NOT_A_CHART))

    assert figures[-1] == ("figures", "done", "0 readings from 1 panels, 1 not a chart")
    assert result.dataset.figure_rows == ()


def test_switched_off_the_stage_asks_nothing_but_stored_readings_still_reach_the_dataset(
    monkeypatch, document: DocumentInput, settings: Settings
):
    store_artifact(document, settings)
    read_document_figures(document, settings, FakeVisionClient(chart_answer(), model=settings.figures_model))
    off = dataclasses.replace(settings, figures_enabled=False)
    client = FakeVisionClient(chart_answer())

    figures, result, _ = run(monkeypatch, document, off, client)

    assert figures == [("figures", "skipped", "figures.enabled is false; 2 stored readings kept")]
    assert client.calls == []
    assert len(result.dataset.figure_rows) == 2
    assert stored_figures(document.document_id, off) is not None


def test_a_corrupt_stored_file_is_read_again(document: DocumentInput, settings: Settings):
    store_artifact(document, settings)
    path = figures_file(document, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ torn", encoding="utf-8")
    client = FakeVisionClient(chart_answer())

    readings = read_document_figures(document, settings, client)

    assert len(client.calls) == 1 and readings.readings


def test_readings_of_another_parse_are_read_again(document: DocumentInput, settings: Settings):
    # Read from PaddleOCR-VL's boxes first; once a MinerU parse exists, those citations belong to no block
    # of the artifact the readings would be shown with.
    store_artifact(document, settings, backend="paddleocr_vl")
    read_document_figures(document, settings, FakeVisionClient(chart_answer()))
    store_artifact(document, settings, backend="mineru")
    client = FakeVisionClient(chart_answer())

    readings = read_document_figures(document, settings, client)

    assert len(client.calls) == 1
    assert readings.backend == "mineru" and {r.source_id for r in readings.readings} == {"mineru_p0_b0"}


def test_the_stored_file_is_named_after_what_the_client_actually_sends(document: DocumentInput, settings: Settings):
    store_artifact(document, settings)

    read_document_figures(document, settings, FakeVisionClient(chart_answer(), temperature=0.5))

    expected = figure_key_for(settings, "fake-vl", temperature=0.5)
    assert DataLayout(settings.data_root).figures_path(document.document_id, expected).is_file()
    assert not figures_file(document, settings).is_file()
