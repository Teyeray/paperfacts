"""Every pool in the pipeline runs its tasks in a copy of the caller's context.

The pools are all :class:`paperfacts.threads.ContextThreadPoolExecutor`; the first cases pin that class,
the rest check that each pool site in the pipeline really is one.

With several web jobs at once, a context variable is how a log record finds its job (web/jobs.py): a task
that loses the context loses its records from the job panel, silently. One case per pool site -- the lanes
and the figures stage in workflow.py, the field questions in extract.py, the chart panels in figures.py --
each reading a variable set only on the calling thread.
"""

from __future__ import annotations

import contextvars
import dataclasses
from pathlib import Path

from paperfacts import keys, workflow
from paperfacts.config import Settings
from paperfacts.extract import extract_lane
from paperfacts.figures import read_figures
from paperfacts.models import DocumentInput
from paperfacts.profile import default_profile
from paperfacts.prompts import inventory_system_prompt
from paperfacts.threads import ContextThreadPoolExecutor
from support.extraction import lane_options, make_artifact
from support.llm import FakeLlmClient
from support.vision import FakeVisionClient, chart_answer
from test_extract_passages import make_blocks, responder
from test_figures import artifact, cap, fig
from test_workflow_run import install_fake_pipeline

CALLER: contextvars.ContextVar[str | None] = contextvars.ContextVar("caller", default=None)


def test_submit_and_map_run_every_task_in_the_submitters_context():
    CALLER.set("outer")
    with ContextThreadPoolExecutor(max_workers=2) as pool:
        submitted = pool.submit(CALLER.get).result()
        mapped = list(pool.map(lambda _: CALLER.get(), range(4)))

    assert submitted == "outer"
    assert mapped == ["outer"] * 4


def test_a_nested_pool_carries_the_context_of_the_task_that_opened_it():
    def inner(label: str) -> list[str | None]:
        CALLER.set(label)  # set on the outer pool's thread, in that task's own copy
        with ContextThreadPoolExecutor(max_workers=2) as pool:
            return list(pool.map(lambda _: CALLER.get(), range(2)))

    CALLER.set("outer")
    with ContextThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(inner, ["a", "b"]))

    assert results == [["a", "a"], ["b", "b"]]
    assert CALLER.get() == "outer"  # a task's change stays in its copy


def test_the_pool_module_is_not_cache_key_material():
    """threads.py decides when work runs, never what is asked; hashing it would rename every stored
    extraction for a scheduling change."""
    assert "threads.py" not in Path(keys.__file__).read_text(encoding="utf-8")


def test_the_field_questions_run_in_the_callers_context():
    seen: list[str | None] = []
    answer = responder()

    def recording(system: str, user: str) -> str:
        if system != inventory_system_prompt(default_profile()):
            seen.append(CALLER.get())
        return answer(system, user)

    CALLER.set("job-1")
    extract_lane(
        make_artifact(make_blocks()),
        FakeLlmClient(recording),
        lane_options(mode="passage"),
        concurrency=4,
    )

    assert seen and set(seen) == {"job-1"}


def test_the_chart_panels_run_in_the_callers_context():
    seen: list[str | None] = []

    def recording(user: str, image: bytes):
        seen.append(CALLER.get())
        return chart_answer()

    CALLER.set("job-2")
    read_figures(
        artifact(fig(0, 0), fig(0, 1), cap(0, 2, "Fig. 3 Sheet resistance")),
        lambda page, bbox: b"png",
        FakeVisionClient(recording),
        figure_key="k",
        max_per_document=12,
        concurrency=2,
    )

    assert seen == ["job-2", "job-2"]


def test_the_lanes_and_the_figures_stage_run_in_the_callers_context(monkeypatch, two_page_pdf, tmp_path):
    install_fake_pipeline(monkeypatch)
    seen: list[tuple[str, str | None]] = []
    fake_extract = workflow.extract_document  # the fake pipeline's, installed above

    def recording_extract(document, backend, settings, client, *, force=False):
        seen.append((backend, CALLER.get()))
        return fake_extract(document, backend, settings, client, force=force)

    def recording_figures(document, settings, *, force, artifact, stop):
        seen.append(("figures", CALLER.get()))
        return "done", ""

    monkeypatch.setattr("paperfacts.workflow.extract_document", recording_extract)
    monkeypatch.setattr("paperfacts.workflow._read_figures_stage", recording_figures)
    settings = dataclasses.replace(Settings(data_root=tmp_path / "data"), figures_enabled=True)

    CALLER.set("job-3")
    workflow.run_document(DocumentInput.from_path(two_page_pdf), settings)

    assert sorted(seen) == [("figures", "job-3"), ("mineru", "job-3"), ("paddleocr_vl", "job-3")]
