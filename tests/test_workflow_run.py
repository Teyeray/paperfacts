"""``run_document``: the single orchestration point for two-lane parsing -> two-lane extraction ->
comparison, called by both the ``run`` CLI command and the web job.

Each of the four steps has its own tests (``test_workflow.py`` / ``test_workflow_extraction.py``); here
every one of them is replaced by a fake, so this file watches only the orchestration itself: ordering,
the content of the stage callbacks, whether force is threaded through to every step, and whether the LLM
client is shared and then closed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from paperfacts.compare import ComparisonCounts, ComparisonReport
from paperfacts.config import Settings
from paperfacts.matching import SampleMatching
from paperfacts.models import BACKENDS, Backend, DocumentInput
from paperfacts.records import LaneExtraction
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
        document: DocumentInput, settings: Settings, client: object, *, force: bool = False
    ) -> ComparisonReport:
        spy.compare.append(force)
        spy.clients.append(client)
        return ComparisonReport(
            document_id=document.document_id,
            extractor_key="0123456789ab",
            comparison_key="ba9876543210",
            backend_a="mineru",
            backend_b="paddleocr_vl",
            matching=SampleMatching(),
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
        "extract:mineru",
        "extract:paddleocr_vl",
        "compare",
        "export",
    )
    assert len(stage_names()) == 2 * len(BACKENDS) + 2


# ---- Orchestration ------------------------------------------------------------------------


def test_the_pipeline_walks_the_six_stages_in_order(monkeypatch, document: DocumentInput, settings: Settings):
    install_fake_pipeline(monkeypatch)

    marks, _ = run(document, settings)

    assert marks == [
        ("parse:mineru", "running", ""),
        ("parse:mineru", "done", "11 blocks"),
        ("parse:paddleocr_vl", "running", ""),
        ("parse:paddleocr_vl", "done", "11 blocks"),
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
    assert [name for name, status, _ in marks if status == "running"] == list(stage_names())


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


def test_the_running_stage_is_the_last_mark_when_a_step_blows_up(
    monkeypatch, document: DocumentInput, settings: Settings
):
    # The exception propagates as-is (the job layer marks the currently running stage failed on the
    # strength of that); it is neither swallowed nor rewritten here.
    install_fake_pipeline(monkeypatch)

    def exploding_extract(*args, **kwargs):
        raise RuntimeError("the LLM did not return JSON")

    monkeypatch.setattr("paperfacts.workflow.extract_document", exploding_extract)
    marks: list[tuple[str, str, str]] = []

    with pytest.raises(RuntimeError, match="not return JSON"):
        run_document(document, settings, on_stage=lambda s, st, d: marks.append((s, st, d)))

    # Both lanes were announced and neither finished, so the job layer marks both failed -- which is
    # right: the run stopped there.
    assert [m for m in marks if m[0].startswith("extract:")] == [
        ("extract:mineru", "running", ""),
        ("extract:paddleocr_vl", "running", ""),
    ]
    assert marks[-1][1] == "running"


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
