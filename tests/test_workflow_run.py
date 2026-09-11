"""``run_document``: the single orchestration point for two-lane parsing -> two-lane extraction ->
comparison, called by both the ``run`` CLI command and the web job.

Each of the four steps has its own tests (``test_workflow.py`` / ``test_workflow_extraction.py``); here
every one of them is replaced by a fake, so this file watches only the orchestration itself: ordering,
the content of the stage callbacks, whether force is threaded through to every step, and whether the LLM
client is shared and then closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from paperfacts.config import Settings
from paperfacts.consensus import ComparisonCounts, ComparisonReport
from paperfacts.consensus.matching import SampleMatching
from paperfacts.extraction.records import LaneExtraction
from paperfacts.models import BACKENDS, Backend, DocumentInput
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
        spy.extract.append((backend, force))
        spy.clients.append(client)
        return make_lane(backend=backend, samples=[make_sample(f"S{i}") for i in range(sample_count)])

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


def test_the_stage_names_cover_both_parses_both_extractions_and_the_comparison():
    assert stage_names() == (
        "parse:mineru",
        "parse:paddleocr_vl",
        "extract:mineru",
        "extract:paddleocr_vl",
        "compare",
    )
    assert len(stage_names()) == 2 * len(BACKENDS) + 1


# ---- Orchestration ------------------------------------------------------------------------


def test_the_pipeline_walks_the_five_stages_in_order(monkeypatch, document: DocumentInput, settings: Settings):
    install_fake_pipeline(monkeypatch)

    marks, _ = run(document, settings)

    assert marks == [
        ("parse:mineru", "running", ""),
        ("parse:mineru", "done", "11 blocks"),
        ("parse:paddleocr_vl", "running", ""),
        ("parse:paddleocr_vl", "done", "11 blocks"),
        ("extract:mineru", "running", ""),
        ("extract:mineru", "done", "2 samples"),
        ("extract:paddleocr_vl", "running", ""),
        ("extract:paddleocr_vl", "done", "2 samples"),
        ("compare", "running", ""),
        ("compare", "done", "agree 3 · conflict 1 · ambiguous 2 · missing 4"),
    ]
    assert [name for name, _, _ in marks[::2]] == list(stage_names())


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


def test_force_reaches_every_step(monkeypatch, document: DocumentInput, settings: Settings):
    spy = install_fake_pipeline(monkeypatch)

    run(document, settings, force=True)

    assert spy.parse == [("mineru", True), ("paddleocr_vl", True)]
    assert spy.extract == [("mineru", True), ("paddleocr_vl", True)]
    assert spy.compare == [True]


def test_nothing_is_forced_by_default(monkeypatch, document: DocumentInput, settings: Settings):
    spy = install_fake_pipeline(monkeypatch)

    run(document, settings)

    assert spy.parse == [("mineru", False), ("paddleocr_vl", False)]
    assert spy.extract == [("mineru", False), ("paddleocr_vl", False)]
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

    assert marks[-1] == ("extract:mineru", "running", "")


def test_the_stage_callback_is_optional(monkeypatch, document: DocumentInput, settings: Settings):
    install_fake_pipeline(monkeypatch)

    result = run_document(document, settings)

    assert result.report.counts.total == 10
