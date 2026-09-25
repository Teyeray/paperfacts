"""``batch --jobs N``: papers run side by side, but the output reads as if they had run one after another.

The fake pipeline from test_workflow_run stands in for parsing and the model. Overlap and completion order
are forced with barriers and events, never with sleeps: a barrier of N opens only when N papers are inside
at once, and "each paper waits for the one after it" makes the last paper finish first.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from openpyxl import load_workbook
from typer.testing import CliRunner

from paperfacts.cli import app
from paperfacts.config import Settings
from paperfacts.errors import ConfigError, ParserError
from paperfacts.models import DocumentInput
from paperfacts.workflow import run_batch, run_document
from support.factories import make_blank_pdf
from support.web import WAIT_TIMEOUT_S
from test_workflow_run import install_fake_pipeline

SHEETS = ("论文数据", "样品数据", "运行记录")


def make_papers(folder: Path, count: int) -> list[Path]:
    # Distinct page sizes make distinct PDFs, so no two share a document id.
    return [make_blank_pdf(folder / f"{chr(ord('a') + i)}.pdf", sizes=[(200 + 10 * i, 200)]) for i in range(count)]


def sheets(path: Path) -> dict[str, list[tuple]]:
    workbook = load_workbook(path)
    try:
        return {name: list(workbook[name].values) for name in SHEETS}
    finally:
        workbook.close()


def test_papers_run_side_by_side(monkeypatch, tmp_path: Path):
    make_papers(tmp_path / "papers", 3)
    install_fake_pipeline(monkeypatch)
    barrier = threading.Barrier(3)

    def overlapping(document: DocumentInput, settings: Settings, **kwargs):
        barrier.wait(timeout=WAIT_TIMEOUT_S)
        return run_document(document, settings, **kwargs)

    monkeypatch.setattr("paperfacts.workflow.run_document", overlapping)
    result = run_batch(tmp_path / "papers", Settings(data_root=tmp_path / "data"), jobs=3)

    assert len(result.documents) == 3
    assert result.failures == ()
    assert not barrier.broken


def test_the_output_is_in_input_order_and_a_failed_paper_costs_only_its_own_row(monkeypatch, tmp_path: Path):
    papers = make_papers(tmp_path / "papers", 4)
    install_fake_pipeline(monkeypatch)
    settings = Settings(data_root=tmp_path / "data")

    def failing_b(document: DocumentInput, settings: Settings, **kwargs):
        if document.pdf_path.name == "b.pdf":
            raise ParserError("mineru", "run", "bad PDF")
        return run_document(document, settings, **kwargs)

    monkeypatch.setattr("paperfacts.workflow.run_document", failing_b)
    serial = run_batch(tmp_path / "papers", settings, output=tmp_path / "serial.xlsx", jobs=1)

    # Now every paper waits for the next one to finish before it does, so they complete in reverse order.
    finished = {path.name: threading.Event() for path in papers}
    after = {papers[i].name: papers[i + 1].name for i in range(len(papers) - 1)}

    def reversed_completion(document: DocumentInput, settings: Settings, **kwargs):
        name = document.pdf_path.name
        try:
            if name in after:
                assert finished[after[name]].wait(timeout=WAIT_TIMEOUT_S)
            return failing_b(document, settings, **kwargs)
        finally:
            finished[name].set()

    monkeypatch.setattr("paperfacts.workflow.run_document", reversed_completion)
    parallel = run_batch(tmp_path / "papers", settings, output=tmp_path / "parallel.xlsx", jobs=4)

    assert [d.filename for d in parallel.documents] == ["a.pdf", "c.pdf", "d.pdf"]
    assert [f["filename"] for f in parallel.failures] == ["b.pdf"]
    assert "bad PDF" in parallel.failures[0]["error"]
    assert parallel.documents == serial.documents
    assert parallel.failures == serial.failures
    assert sheets(parallel.excel_path) == sheets(serial.excel_path)


def test_stage_reports_are_serialised_and_name_their_paper(monkeypatch, tmp_path: Path):
    make_papers(tmp_path / "papers", 3)
    install_fake_pipeline(monkeypatch)
    calls: list[tuple[str, str]] = []
    inside = 0
    overlapped = False
    guard = threading.Lock()

    def on_stage(stage: str, status: str, detail: str) -> None:
        nonlocal inside, overlapped
        with guard:
            inside += 1
            overlapped = overlapped or inside > 1
        calls.append((stage, status))
        with guard:
            inside -= 1

    run_batch(tmp_path / "papers", Settings(data_root=tmp_path / "data"), jobs=3, on_stage=on_stage)

    assert not overlapped
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        own = [stage for stage, _ in calls if name in stage]
        assert own[0].endswith(name)  # the paper's own "running" line comes first
        assert any(stage.endswith(f"{name} compare") for stage in own)
        assert any(stage.endswith(f"{name} parse:mineru") for stage in own)


def test_the_first_copy_of_a_duplicate_is_the_one_processed(monkeypatch, tmp_path: Path):
    source = tmp_path / "papers"
    (original,) = make_papers(source, 1)
    (source / "z_copy.pdf").write_bytes(original.read_bytes())
    spy = install_fake_pipeline(monkeypatch)

    result = run_batch(source, Settings(data_root=tmp_path / "data"), jobs=4)

    assert result.duplicate_count == 1
    assert [d.filename for d in result.documents] == ["a.pdf"]
    assert len(spy.parse) == 2  # one paper, two lanes


def test_a_workbook_that_cannot_be_written_stops_a_parallel_batch(monkeypatch, tmp_path: Path):
    make_papers(tmp_path / "papers", 3)
    install_fake_pipeline(monkeypatch)

    def fail_write(*args, **kwargs):
        raise PermissionError("workbook is locked")

    monkeypatch.setattr("paperfacts.workflow.write_dataset", fail_write)
    with pytest.raises(PermissionError, match="locked"):
        run_batch(tmp_path / "papers", Settings(data_root=tmp_path / "data"), jobs=3)


def test_jobs_below_one_is_refused(tmp_path: Path):
    make_papers(tmp_path / "papers", 1)
    with pytest.raises(ConfigError, match="--jobs"):
        run_batch(tmp_path / "papers", Settings(data_root=tmp_path / "data"), jobs=0)


@pytest.mark.parametrize(("flags", "expected"), [([], 3), (["--jobs", "5"], 5), (["-j", "1"], 1)])
def test_the_cli_takes_jobs_from_the_flag_or_the_parallel_documents_setting(
    monkeypatch, tmp_path: Path, flags: list[str], expected: int
):
    (paper,) = make_papers(tmp_path / "papers", 1)
    captured: list[int] = []

    def fake_run_batch(source: Path, settings: Settings, **kwargs):
        captured.append(kwargs["jobs"])
        raise ParserError("mineru", "run", "stop here")

    monkeypatch.setattr("paperfacts.cli.run_batch", fake_run_batch)
    monkeypatch.setenv("PAPERFACTS_MAX_PARALLEL_DOCUMENTS", "3")

    CliRunner().invoke(app, ["batch", str(paper.parent), "--data-root", str(tmp_path / "data"), *flags])

    assert captured == [expected]
