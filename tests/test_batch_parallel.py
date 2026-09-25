"""``batch --jobs N``: papers run side by side, but the output reads as if they had run one after another.

The fake pipeline from test_workflow_run stands in for parsing and the model. Overlap and completion order
are forced with barriers and events, never with sleeps: a barrier of N opens only when N papers are inside
at once, and "each paper waits for the one after it" makes the last paper finish first.
"""

from __future__ import annotations

import threading
import time
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
    # All three papers are released at once and report straight away, so their first reports contend.
    start_together = threading.Barrier(3)

    def contending(document: DocumentInput, settings: Settings, *, on_stage, **kwargs):
        start_together.wait(timeout=WAIT_TIMEOUT_S)
        on_stage("probe", "running", "")
        return run_document(document, settings, on_stage=on_stage, **kwargs)

    monkeypatch.setattr("paperfacts.workflow.run_document", contending)
    calls: list[tuple[str, str]] = []
    guard = threading.Lock()
    inside = 0
    overlapped = False

    def on_stage(stage: str, status: str, detail: str) -> None:
        nonlocal inside, overlapped
        with guard:
            inside += 1
            overlapped = overlapped or inside > 1
        # Hand the interpreter to the other threads while inside: an unserialised report gets in here.
        for _ in range(50):
            time.sleep(0)
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


def test_a_stopped_batch_stops_its_running_papers_at_the_next_stage(monkeypatch, tmp_path: Path):
    """Ctrl-C and an unexpected error take the same path: nothing new starts, the running papers stop at
    their next stage boundary, and the caller hears how many it is waiting for."""
    make_papers(tmp_path / "papers", 3)
    spy = install_fake_pipeline(monkeypatch)
    all_started = threading.Barrier(3)
    stopping = threading.Event()

    def one_breaks(document: DocumentInput, settings: Settings, **kwargs):
        all_started.wait(timeout=WAIT_TIMEOUT_S)
        if document.pdf_path.name == "c.pdf":
            raise RuntimeError("not a paper's own failure")
        assert stopping.wait(timeout=WAIT_TIMEOUT_S)
        return run_document(document, settings, **kwargs)

    monkeypatch.setattr("paperfacts.workflow.run_document", one_breaks)
    reports: list[tuple[str, str, str]] = []

    def on_stage(stage: str, status: str, detail: str) -> None:
        reports.append((stage, status, detail))
        if stage == "batch":
            stopping.set()

    with pytest.raises(RuntimeError, match="own failure"):
        run_batch(tmp_path / "papers", Settings(data_root=tmp_path / "data"), jobs=3, on_stage=on_stage)

    assert ("batch", "failed", "stopping; waiting for 2 running papers to reach a stage boundary") in reports
    stopped = sorted(stage for stage, status, _ in reports if status == "skipped")
    assert stopped == ["1/3 a.pdf", "2/3 b.pdf"]
    assert spy.parse == []  # both stopped at their first boundary, before any work


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
    # Every paper that was running has finished: none is left writing after the caller gave up.
    assert not [thread for thread in threading.enumerate() if thread.name.startswith("paperfacts-document")]


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


def test_export_reads_the_caches_one_paper_at_a_time(monkeypatch, tmp_path: Path):
    (paper,) = make_papers(tmp_path / "papers", 1)
    captured: list[int] = []

    def fake_run_batch(source: Path, settings: Settings, **kwargs):
        captured.append(kwargs["jobs"])
        raise ParserError("mineru", "run", "stop here")

    monkeypatch.setattr("paperfacts.cli.run_batch", fake_run_batch)
    monkeypatch.setenv("PAPERFACTS_MAX_PARALLEL_DOCUMENTS", "4")

    CliRunner().invoke(app, ["export", str(paper.parent), "--data-root", str(tmp_path / "data")])

    assert captured == [1]
