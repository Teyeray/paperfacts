"""``run_document``: the single orchestration point for two-lane parsing -> two-lane extraction ->
comparison, called by both the ``run`` CLI command and the web job.

Each of the four steps has its own tests (``test_workflow.py`` / ``test_workflow_extraction.py``); here
every one of them is replaced by a fake, so this file watches only the orchestration itself: ordering,
the content of the stage callbacks, whether force is threaded through to every step, and whether the LLM
client is shared and then closed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import paperfacts.workflow as workflow_module
from paperfacts.compare import ComparisonCounts, ComparisonReport
from paperfacts.config import Settings
from paperfacts.errors import Cancelled, ParserError
from paperfacts.matching import SampleMatching
from paperfacts.models import BACKENDS, Backend, DocumentInput
from paperfacts.records import FailedQuestion, LaneExtraction
from paperfacts.storage import DataLayout
from paperfacts.stored import is_finished
from paperfacts.workflow import ParseReport, run_document, stage_names
from support.extraction import make_lane, make_sample
from support.llm import FakeLlmClient


@dataclass
class PipelineSpy:
    """Records how the workflow's four entry points were each called; not one real parse or LLM call
    happens."""

    parse: list[tuple[Backend, bool]] = field(default_factory=list)
    extract: list[tuple[Backend, bool]] = field(default_factory=list)
    compare: list[bool] = field(default_factory=list)
    clients: list[object] = field(default_factory=list)
    client: FakeLlmClient = field(default_factory=lambda: FakeLlmClient([]))
    # The lanes each compare_document call was handed, so a test can prove they are the extracted ones.
    compare_lanes: list[Mapping[Backend, LaneExtraction] | None] = field(default_factory=list)
    # Both lanes record themselves from their own thread, so the recorder needs a lock of its own.
    lock: threading.Lock = field(default_factory=threading.Lock)
    # How many lanes were inside the fake extract at once: 2 proves they really overlapped.
    peak_concurrent_lanes: int = 0
    _in_flight: int = 0


def install_fake_pipeline(
    monkeypatch,
    *,
    block_count: int = 11,
    cache_hit: bool = False,
    sample_count: int = 2,
    counts: ComparisonCounts | None = None,
    matching: SampleMatching | None = None,
) -> PipelineSpy:
    spy = PipelineSpy()
    counts = counts or ComparisonCounts(agree=3, conflict=1, ambiguous=2, missing=4, total=10)

    def fake_parse(document: DocumentInput, backend: Backend, settings: Settings, *, force: bool = False):
        spy.parse.append((backend, force))
        report = ParseReport(
            backend=backend,
            backend_version="3.4.5",
            cache_hit=cache_hit,
            runtime_s=0.1,
            page_count=2,
            block_count=block_count,
            type_counts={"text": block_count},
            artifact_path=Path("artifact.json"),
            markdown_path=Path("doc.md"),
        )
        return None, report

    def fake_extract(
        document: DocumentInput, backend: Backend, settings: Settings, client: object, *, force: bool = False
    ) -> LaneExtraction:
        with spy.lock:
            spy.extract.append((backend, force))
            spy.clients.append(client)
            spy._in_flight += 1
            spy.peak_concurrent_lanes = max(spy.peak_concurrent_lanes, spy._in_flight)
        # Long enough that a sequential implementation could not show two lanes in flight at once.
        time.sleep(0.05)
        with spy.lock:
            spy._in_flight -= 1
        return make_lane(
            backend=backend,
            document_id=document.document_id,
            samples=[make_sample(f"S{i}") for i in range(sample_count)],
        )

    def fake_compare(
        document: DocumentInput,
        settings: Settings,
        client: object,
        *,
        force: bool = False,
        lanes: Mapping[Backend, LaneExtraction] | None = None,
    ) -> ComparisonReport:
        spy.compare.append(force)
        spy.clients.append(client)
        spy.compare_lanes.append(lanes)
        return ComparisonReport(
            document_id=document.document_id,
            extractor_key="0123456789ab",
            comparison_key="ba9876543210",
            backend_a="mineru",
            backend_b="paddleocr_vl",
            matching=matching or SampleMatching(),
            counts=counts,
        )

    monkeypatch.setattr("paperfacts.workflow.parse_document", fake_parse)
    monkeypatch.setattr("paperfacts.workflow.extract_document", fake_extract)
    monkeypatch.setattr("paperfacts.workflow.compare_document", fake_compare)
    monkeypatch.setattr("paperfacts.workflow.build_llm_client", lambda settings: spy.client)
    return spy


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_root=tmp_path / "data", repo_root=tmp_path, llm_api_key="sk-test")


def run(document: DocumentInput, settings: Settings, *, force: bool = False) -> tuple[list[tuple], object]:
    marks: list[tuple[str, str, str]] = []
    result = run_document(document, settings, force=force, on_stage=lambda s, st, d: marks.append((s, st, d)))
    return marks, result


# ---- Stage names are a public contract --------------------------------------------------


def test_the_stage_names_cover_extraction_comparison_and_export():
    assert stage_names() == (
        "parse:mineru",
        "parse:paddleocr_vl",
        "figures",
        "extract:mineru",
        "extract:paddleocr_vl",
        "compare",
        "export",
    )
    assert len(stage_names()) == 2 * len(BACKENDS) + 3


# ---- Orchestration ------------------------------------------------------------------------


def test_the_pipeline_walks_the_six_stages_in_order(monkeypatch, document: DocumentInput, settings: Settings):
    install_fake_pipeline(monkeypatch)

    marks, _ = run(document, settings)

    assert marks == [
        ("parse:mineru", "running", ""),
        ("parse:mineru", "done", "11 blocks"),
        ("parse:paddleocr_vl", "running", ""),
        ("parse:paddleocr_vl", "done", "11 blocks"),
        # Opt-in and off by default, so it is announced as skipped and never runs.
        ("figures", "skipped", "figures.enabled is false"),
        # The two lanes run as a pair, so both are announced before either can finish; their "done"
        # marks are still emitted in BACKENDS order, from the calling thread.
        ("extract:mineru", "running", ""),
        ("extract:paddleocr_vl", "running", ""),
        ("extract:mineru", "done", "2 samples"),
        ("extract:paddleocr_vl", "done", "2 samples"),
        ("compare", "running", ""),
        ("compare", "done", "agree 3 · conflict 1 · ambiguous 2 · missing 4"),
        ("export", "running", ""),
        ("export", "done", str(settings.data_root / "docs" / document.document_id[:16] / "dataset.xlsx")),
    ]
    assert [name for name, status, _ in marks if status in {"running", "skipped"}] == list(stage_names())


def test_a_cached_parse_says_so_in_the_stage_detail(monkeypatch, document: DocumentInput, settings: Settings):
    # The user needs to see "11 blocks (cached)" to know why this step returned instantly.
    install_fake_pipeline(monkeypatch, cache_hit=True)

    marks, _ = run(document, settings)

    assert ("parse:mineru", "done", "11 blocks (cached)") in marks


def test_the_result_carries_every_intermediate_product(monkeypatch, document: DocumentInput, settings: Settings):
    install_fake_pipeline(monkeypatch, sample_count=3)

    _, result = run(document, settings)

    assert set(result.parse_reports) == set(BACKENDS)
    assert {backend: len(lane.samples) for backend, lane in result.lanes.items()} == {b: 3 for b in BACKENDS}
    assert result.report.counts.agree == 3
    assert result.excel_path.is_file()
    # The web UI reads the consolidated table from this file, so the run must leave one behind.
    assert result.dataset_json_path.is_file()
    assert result.dataset_json_path.name == f"{result.dataset.extractor_key}.{result.dataset.comparison_key}.json"
    assert result.dataset.document_id == document.document_id


def test_a_run_whose_matching_failed_is_not_finished(monkeypatch, document: DocumentInput, settings: Settings):
    # compare_document does not store a failed matching so the next run asks again. A stored dataset would
    # count the paper as finished, and "run all" / `deploy.sh --rerun` would then never ask again.
    install_fake_pipeline(monkeypatch, matching=SampleMatching(failed=True, failure="invalid JSON twice"))

    marks, result = run(document, settings)

    layout = DataLayout(settings.data_root)
    keys = {"extractor_key": result.dataset.extractor_key, "comparison_key": result.dataset.comparison_key}
    assert result.excel_path.is_file()  # the CLI run still gets its workbook
    assert result.dataset_json_path is None
    assert not layout.dataset_json_path(document.document_id, **keys).is_file()
    assert is_finished(layout, document.document_id, **keys) is False
    final = {stage: (status, detail) for stage, status, detail in marks}
    assert final["compare"][0] == "failed" and "sample matching failed" in final["compare"][1]


def test_a_run_with_an_unanswered_field_question_is_not_finished(
    monkeypatch, document: DocumentInput, settings: Settings
):
    install_fake_pipeline(monkeypatch)
    fake_extract = workflow_module.extract_document

    def incomplete(document, backend, settings, client, *, force: bool = False):
        lane = fake_extract(document, backend, settings, client, force=force)
        return lane.model_copy(update={"failed_questions": (FailedQuestion(field="thickness", detail="cut off"),)})

    monkeypatch.setattr("paperfacts.workflow.extract_document", incomplete)

    marks, result = run(document, settings)

    assert result.dataset_json_path is None
    final = {stage: (status, detail) for stage, status, detail in marks}
    assert "1 question unanswered" in final["extract:mineru"][1]
    assert "no valid answer" in final["compare"][1]


def test_force_reaches_every_step(monkeypatch, document: DocumentInput, settings: Settings):
    spy = install_fake_pipeline(monkeypatch)

    run(document, settings, force=True)

    assert spy.parse == [("mineru", True), ("paddleocr_vl", True)]
    # Lane order is a race now that they run together, so it is the set that carries the meaning.
    assert sorted(spy.extract) == [("mineru", True), ("paddleocr_vl", True)]
    assert spy.compare == [True]


def test_nothing_is_forced_by_default(monkeypatch, document: DocumentInput, settings: Settings):
    spy = install_fake_pipeline(monkeypatch)

    run(document, settings)

    assert spy.parse == [("mineru", False), ("paddleocr_vl", False)]
    assert sorted(spy.extract) == [("mineru", False), ("paddleocr_vl", False)]
    assert spy.compare == [False]


def test_one_llm_client_is_shared_by_extraction_and_comparison_and_then_closed(
    monkeypatch, document: DocumentInput, settings: Settings
):
    """Both extraction lanes plus matching share one client (connection pool, usage accounting), and it
    must always be closed.

    Opening one per stage, or forgetting to close it, would slowly leak connections on a server over time.
    """
    spy = install_fake_pipeline(monkeypatch)

    run(document, settings)

    assert spy.clients == [spy.client] * 3
    assert spy.client.closed is True


def test_each_failed_lane_is_marked_with_its_own_error_and_the_exception_propagates(
    monkeypatch, document: DocumentInput, settings: Settings
):
    # The exception propagates as-is; it is neither swallowed nor rewritten here.
    install_fake_pipeline(monkeypatch)

    def exploding_extract(document, backend, *args, **kwargs):
        raise RuntimeError(f"{backend}: the LLM did not return JSON")

    monkeypatch.setattr("paperfacts.workflow.extract_document", exploding_extract)
    marks: list[tuple[str, str, str]] = []

    with pytest.raises(RuntimeError, match="mineru: the LLM did not return JSON"):
        run_document(document, settings, on_stage=lambda s, st, d: marks.append((s, st, d)))

    assert [m for m in marks if m[0].startswith("extract:")] == [
        ("extract:mineru", "running", ""),
        ("extract:paddleocr_vl", "running", ""),
        ("extract:mineru", "failed", "RuntimeError: mineru: the LLM did not return JSON"),
        ("extract:paddleocr_vl", "failed", "RuntimeError: paddleocr_vl: the LLM did not return JSON"),
    ]


def test_the_lane_that_succeeded_is_marked_done_when_the_other_fails(
    monkeypatch, document: DocumentInput, settings: Settings
):
    # The review's scenario: the UI said "extract:mineru failed: [paddleocr_vl] ..." for a mineru lane that
    # had succeeded and was cached, because the job layer stamps every stage still running.
    install_fake_pipeline(monkeypatch)

    def one_lane_fails(document, backend, settings, client, *, force: bool = False):
        if backend == "paddleocr_vl":
            raise RuntimeError("paddle lane exploded")
        return make_lane(backend=backend, samples=(make_sample("A"),))

    monkeypatch.setattr("paperfacts.workflow.extract_document", one_lane_fails)
    marks: list[tuple[str, str, str]] = []

    with pytest.raises(RuntimeError, match="paddle lane exploded"):
        run_document(document, settings, on_stage=lambda s, st, d: marks.append((s, st, d)))

    final = {stage: (status, detail) for stage, status, detail in marks}
    assert final["extract:mineru"] == ("done", "1 samples")
    assert final["extract:paddleocr_vl"] == ("failed", "RuntimeError: paddle lane exploded")
    assert not [stage for stage, (status, _) in final.items() if status == "running"]


def test_a_callback_that_raises_while_failing_does_not_mask_the_failure(
    monkeypatch, document: DocumentInput, settings: Settings
):
    # A stopped batch raises Cancelled from its callback; the paper's own error must still be the one raised.
    install_fake_pipeline(monkeypatch)

    def one_lane_fails(document, backend, settings, client, *, force: bool = False):
        raise RuntimeError(f"{backend} lane exploded")

    def refusing(stage: str, status: str, detail: str) -> None:
        if status in {"done", "failed"} and stage.startswith("extract:"):
            raise Cancelled("the batch was stopped")

    monkeypatch.setattr("paperfacts.workflow.extract_document", one_lane_fails)

    with pytest.raises(RuntimeError, match="mineru lane exploded"):
        run_document(document, settings, on_stage=refusing)


def test_the_stage_callback_is_optional(monkeypatch, document: DocumentInput, settings: Settings):
    install_fake_pipeline(monkeypatch)

    result = run_document(document, settings)

    assert result.report.counts.total == 10


# ---- The two lanes overlap ----------------------------------------------------------------


def test_both_lanes_extract_at_the_same_time(monkeypatch, document: DocumentInput, settings: Settings):
    """The lanes are independent and both spend their time waiting on the model, so they must overlap.

    Run one after the other, a paper costs the sum of the two lanes instead of the slower one.
    """
    spy = install_fake_pipeline(monkeypatch)

    marks, result = run(document, settings)

    assert spy.peak_concurrent_lanes == 2
    assert set(result.lanes) == set(BACKENDS)
    for backend in BACKENDS:
        assert (f"extract:{backend}", "done", "2 samples") in marks


def test_a_lane_failure_still_surfaces_as_itself(monkeypatch, document: DocumentInput, settings: Settings):
    """Running the lanes together must not wrap, swallow or reorder the error one of them raises."""
    install_fake_pipeline(monkeypatch)

    def failing_extract(document, backend, settings, client, *, force: bool = False):
        if backend == BACKENDS[1]:
            raise RuntimeError("paddle lane exploded")
        return make_lane(backend=backend, samples=(make_sample("A"),))

    monkeypatch.setattr("paperfacts.workflow.extract_document", failing_extract)

    with pytest.raises(RuntimeError, match="paddle lane exploded"):
        run_document(document, settings)


def test_the_client_is_closed_even_when_a_lane_fails(monkeypatch, document: DocumentInput, settings: Settings):
    """The other lane is awaited on the way out, so the shared client is never closed underneath it."""
    spy = install_fake_pipeline(monkeypatch)

    def failing_extract(document, backend, settings, client, *, force: bool = False):
        raise RuntimeError("both lanes exploded")

    monkeypatch.setattr("paperfacts.workflow.extract_document", failing_extract)

    with pytest.raises(RuntimeError, match="both lanes exploded"):
        run_document(document, settings)

    assert spy.client.closed is True


def test_the_report_is_built_from_the_lanes_that_were_extracted(
    monkeypatch, document: DocumentInput, settings: Settings
):
    """The comparison must not re-load what the run already holds: one extraction per backend per run,
    and the very objects it produced are the ones compare_document is handed."""
    spy = install_fake_pipeline(monkeypatch)

    result = run_document(document, settings)

    assert [backend for backend, _ in spy.extract] == list(BACKENDS)  # exactly one extraction per lane
    assert spy.compare_lanes == [result.lanes]
    for backend in BACKENDS:
        assert spy.compare_lanes[0][backend] is result.lanes[backend]


def test_the_first_lane_in_backends_order_wins_when_both_fail(
    monkeypatch, document: DocumentInput, settings: Settings, caplog
):
    """Both lanes are collected before either is acted on: the first in BACKENDS order is raised and the
    other one's exception is logged rather than dropped."""
    install_fake_pipeline(monkeypatch)

    def failing_extract(document, backend, settings, client, *, force: bool = False):
        raise RuntimeError(f"{backend} lane exploded")

    monkeypatch.setattr("paperfacts.workflow.extract_document", failing_extract)

    with caplog.at_level(logging.WARNING, logger="paperfacts.workflow"):
        with pytest.raises(RuntimeError, match=f"{BACKENDS[0]} lane exploded"):
            run_document(document, settings)

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert f"{BACKENDS[1]} lane exploded" in logged
    assert f"{BACKENDS[0]} lane exploded" not in logged  # the one that was raised is not also logged


def test_a_surviving_lane_is_not_reported_as_a_failure(
    monkeypatch, document: DocumentInput, settings: Settings, caplog
):
    """Only the lane that raised is explained; the one that succeeded has nothing to say."""
    install_fake_pipeline(monkeypatch)

    def failing_extract(document, backend, settings, client, *, force: bool = False):
        if backend == BACKENDS[0]:
            raise RuntimeError("mineru lane exploded")
        return make_lane(backend=backend, samples=(make_sample("A"),))

    monkeypatch.setattr("paperfacts.workflow.extract_document", failing_extract)

    with caplog.at_level(logging.WARNING, logger="paperfacts.workflow"):
        with pytest.raises(RuntimeError, match="mineru lane exploded"):
            run_document(document, settings)

    assert [record for record in caplog.records if "also failed" in record.getMessage()] == []


# ---- the two parses of one paper ---------------------------------------------------------------------


def _parses_recorded(monkeypatch, barrier: threading.Barrier | None = None):
    """Replace the fake parse with one that records overlap, optionally meeting the other lane at a barrier."""
    fake_parse = workflow_module.parse_document
    state = {"inside": 0, "peak": 0}
    lock = threading.Lock()

    def parse(document, backend, settings, *, force=False):
        with lock:
            state["inside"] += 1
            state["peak"] = max(state["peak"], state["inside"])
        try:
            if barrier is not None:
                barrier.wait(timeout=5.0)  # BrokenBarrierError unless the other lane is parsing too
            return fake_parse(document, backend, settings, force=force)
        finally:
            with lock:
                state["inside"] -= 1

    monkeypatch.setattr("paperfacts.workflow.parse_document", parse)
    return state


def test_with_a_parser_service_the_two_lanes_parse_side_by_side(monkeypatch, two_page_pdf: Path, tmp_path: Path):
    install_fake_pipeline(monkeypatch)
    barrier = threading.Barrier(2)
    _parses_recorded(monkeypatch, barrier)
    settings = Settings(data_root=tmp_path / "data", mineru_url="http://gpu:8002", paddle_url="http://gpu:8080")
    marks: list[tuple[str, str]] = []

    run_document(DocumentInput.from_path(two_page_pdf), settings, on_stage=lambda s, st, d: marks.append((s, st)))

    assert not barrier.broken
    parse_marks = [mark for mark in marks if mark[0].startswith("parse:")]
    assert parse_marks == [
        ("parse:mineru", "running"),
        ("parse:paddleocr_vl", "running"),
        ("parse:mineru", "done"),
        ("parse:paddleocr_vl", "done"),
    ]


def test_with_two_runner_subprocesses_the_lanes_parse_one_after_the_other(
    monkeypatch, two_page_pdf: Path, tmp_path: Path
):
    install_fake_pipeline(monkeypatch)
    state = _parses_recorded(monkeypatch)
    marks: list[tuple[str, str]] = []

    run_document(
        DocumentInput.from_path(two_page_pdf),
        Settings(data_root=tmp_path / "data"),
        on_stage=lambda s, st, d: marks.append((s, st)),
    )

    assert state["peak"] == 1
    assert [mark for mark in marks if mark[0].startswith("parse:")] == [
        ("parse:mineru", "running"),
        ("parse:mineru", "done"),
        ("parse:paddleocr_vl", "running"),
        ("parse:paddleocr_vl", "done"),
    ]


def test_a_failed_parse_in_one_lane_fails_the_paper_after_both_have_ended(
    monkeypatch, two_page_pdf: Path, tmp_path: Path
):
    install_fake_pipeline(monkeypatch)
    fake_parse = workflow_module.parse_document
    ended: list[str] = []

    def parse(document, backend, settings, *, force=False):
        try:
            if backend == "mineru":
                raise ParserError("mineru", "http", "service down")
            return fake_parse(document, backend, settings, force=force)
        finally:
            ended.append(backend)

    monkeypatch.setattr("paperfacts.workflow.parse_document", parse)
    settings = Settings(data_root=tmp_path / "data", mineru_url="http://gpu:8002", paddle_url="http://gpu:8080")

    marks: list[tuple[str, str]] = []

    with pytest.raises(ParserError, match="service down"):
        run_document(DocumentInput.from_path(two_page_pdf), settings, on_stage=lambda s, st, d: marks.append((s, st)))

    assert sorted(ended) == ["mineru", "paddleocr_vl"]
    assert [mark for mark in marks if mark[1] != "running"] == [
        ("parse:mineru", "failed"),
        ("parse:paddleocr_vl", "done"),
    ]
