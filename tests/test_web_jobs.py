"""Background jobs: a few worker threads run the whole pipeline, one document each, and the web frontend
polls for status and logs.

This is the only threaded layer in the codebase, so the tests here follow two hard rules:

1. **Timing is staged with** :class:`threading.Event` (and barriers), and waiting for a result polls
   with a timeout (see :mod:`support.web`); any ``sleep(0.1)`` would only be "not flaky on this
   machine, for now".
2. **A failure must leave a trace**: if the worker thread's exception isn't caught and doesn't show
   up in the snapshot, the job died silently and the web page would show "running" forever.

During a job, the ``paperfacts`` logger is temporarily raised to INFO — the default root logger is
WARNING, and an INFO record gets dropped **before** it reaches any handler; it must be restored
afterwards, and with several jobs running only when the last one ends. Each of these has its own case.
"""

from __future__ import annotations

import contextvars
import itertools
import logging
import threading

import pytest
from pydantic import ValidationError

from paperfacts.web.jobs import MAX_LOG_LINES, Job, JobManager, StageStatus
from support.web import RecordingRunner, wait_for_status, wait_until

STAGES: tuple[str, ...] = ("parse", "extract", "compare")
PAPERFACTS_LOGGER = "paperfacts"


def manager_for(runner: RecordingRunner, stages: tuple[str, ...] = STAGES) -> JobManager:
    return JobManager(runner, stages)


def run_to_completion(runner: RecordingRunner, *, document_id: str = "doc-1", force: bool = False) -> tuple[Job, Job]:
    """Submit a job and wait for it to end; returns ``(the object from submit, the finished snapshot)``."""
    manager = manager_for(runner)
    submitted = manager.submit(document_id, force=force)
    return submitted, wait_for_status(manager, submitted.job_id, "done", "failed")


# ---- submit --------------------------------------------------------------------------


def test_a_submitted_job_starts_out_queued_with_every_stage_pending():
    # there's only one worker thread: pin the first job on the gate, so the second one is guaranteed to still be queued.
    gate = threading.Event()
    runner = RecordingRunner(gate=gate)
    manager = manager_for(runner)
    try:
        first = manager.submit("doc-1")
        assert runner.entered.wait(timeout=2.0)

        queued = manager.submit("doc-2")

        assert queued.status == "queued"
        assert queued.started_at is None and queued.finished_at is None
        assert [stage.status for stage in queued.stages] == ["pending"] * len(STAGES)
        assert [stage.name for stage in queued.stages] == list(STAGES)
    finally:
        gate.set()
    wait_for_status(manager, first.job_id, "done")


def test_every_job_gets_its_own_id():
    runner = RecordingRunner()
    manager = manager_for(runner)

    ids = {manager.submit(f"doc-{i}").job_id for i in range(5)}

    assert len(ids) == 5
    wait_until(lambda: runner.call_count == 5, what="all five jobs to finish")


def test_submitting_a_document_that_is_already_active_returns_the_same_job():
    """Idempotent: double-clicking "reprocess" shouldn't queue two jobs and pay for the LLM call twice."""
    gate = threading.Event()
    runner = RecordingRunner(gate=gate)
    manager = manager_for(runner)
    try:
        first = manager.submit("doc-1")
        assert runner.entered.wait(timeout=2.0)

        again = manager.submit("doc-1")

        assert again.job_id == first.job_id
    finally:
        gate.set()
    wait_for_status(manager, first.job_id, "done")
    assert runner.call_count == 1


def test_a_forced_resubmission_is_a_new_job_even_while_a_cached_run_is_active():
    # upgrading from "use cache" to "force rerun" is a genuinely new request; it must not merge into the running one.
    gate = threading.Event()
    runner = RecordingRunner(gate=gate)
    manager = manager_for(runner)
    try:
        first = manager.submit("doc-1")
        forced = manager.submit("doc-1", force=True)
        assert forced.job_id != first.job_id
    finally:
        gate.set()
    wait_until(lambda: runner.call_count == 2, what="both jobs to finish")
    assert runner.forces == [False, True]


def test_a_plain_resubmission_reuses_an_active_forced_run():
    gate = threading.Event()
    runner = RecordingRunner(gate=gate)
    manager = manager_for(runner)
    try:
        forced = manager.submit("doc-1", force=True)
        plain = manager.submit("doc-1")
        assert plain.job_id == forced.job_id
    finally:
        gate.set()
    wait_for_status(manager, forced.job_id, "done")
    assert runner.call_count == 1


def test_a_finished_document_can_be_submitted_again():
    runner = RecordingRunner()
    manager = manager_for(runner)
    first = manager.submit("doc-1")
    wait_for_status(manager, first.job_id, "done")

    second = manager.submit("doc-1")

    assert second.job_id != first.job_id
    wait_for_status(manager, second.job_id, "done")
    assert runner.call_count == 2


def test_the_runner_sees_the_document_id_and_the_force_flag():
    runner = RecordingRunner()

    _, job = run_to_completion(runner, document_id="doc-42", force=True)

    assert runner.call_count == 1
    assert runner.document_ids == ["doc-42"]
    assert runner.forces == [
        True
    ]  # force isn't threaded through anywhere else; the web checkbox is otherwise decorative
    assert job.force is True


def test_a_finished_job_is_done_and_carries_both_timestamps():
    _, job = run_to_completion(RecordingRunner())

    assert job.status == "done"
    assert job.started_at is not None
    assert job.finished_at is not None
    assert job.error is None


# ---- failure ----------------------------------------------------------------------------


def test_a_failing_runner_marks_the_job_failed_with_the_exception_type():
    _, job = run_to_completion(RecordingRunner(error=RuntimeError("parser exited with code 3")))

    assert job.status == "failed"
    assert job.error == "RuntimeError: parser exited with code 3"
    assert job.finished_at is not None


def test_the_stage_that_was_running_when_it_blew_up_is_marked_failed():
    def body(job: Job, mark) -> None:
        mark("parse", "done", "11 blocks")
        mark("extract", "running", "")

    _, job = run_to_completion(RecordingRunner(body=body, error=ValueError("LLM did not return JSON")))

    statuses = {stage.name: stage.status for stage in job.stages}
    assert statuses == {"parse": "done", "extract": "failed", "compare": "pending"}
    assert next(s for s in job.stages if s.name == "extract").detail == job.error


def test_the_traceback_is_kept_so_the_failure_can_be_diagnosed_from_the_page():
    _, job = run_to_completion(RecordingRunner(error=RuntimeError("boom")))

    joined = "\n".join(job.log)
    assert "Traceback" in joined
    assert "RuntimeError: boom" in joined
    assert f"job {job.job_id} failed" in joined  # failure goes through the logger too, matched by job_id


def test_log_records_from_the_pipeline_pools_reach_the_job_log_and_request_threads_do_not():
    # run_document's pools run every task in a copy of the submitting thread's context; an HTTP request
    # thread starts with an empty one.
    def body(job: Job, mark) -> None:
        def say(message: str) -> None:
            logging.getLogger("paperfacts.extract").warning(message)

        context = contextvars.copy_context()
        pool = threading.Thread(target=context.run, args=(say, "from a lane pool"), name="paperfacts-lane_0")
        other = threading.Thread(target=say, args=("from a request thread",), name="AnyIO worker thread")
        for thread in (pool, other):
            thread.start()
            thread.join()

    _, job = run_to_completion(RecordingRunner(body=body))

    joined = "\n".join(job.log)
    assert "from a lane pool" in joined
    assert "from a request thread" not in joined


def test_a_base_exception_in_the_runner_still_ends_the_job_as_failed():
    # KeyboardInterrupt / SystemExit aren't Exception; without this catch the job would show "running" forever.
    def body(job: Job, mark) -> None:
        mark("parse", "running", "")

    _, job = run_to_completion(RecordingRunner(body=body, error=KeyboardInterrupt()))

    assert job.status == "failed"
    assert job.error == "interrupted"
    assert job.finished_at is not None
    assert job.stages[0].status == "failed"


def test_a_finished_snapshot_is_never_half_written():
    """The terminal state (status / error / stages / finished_at) is written all at once inside one
    lock: any snapshot is either still running, or completely finished."""
    runner = RecordingRunner(body=lambda job, mark: mark("compare", "running", ""))
    manager = manager_for(runner)
    job = manager.submit("doc-1")
    seen: list[Job] = []

    def consistent() -> bool:
        snapshot = manager.get(job.job_id)
        seen.append(snapshot)
        return snapshot.status == "done"

    wait_until(consistent, what="the job to finish")

    for snapshot in seen:
        if snapshot.status in ("done", "failed"):
            assert snapshot.finished_at is not None
            assert all(stage.status != "running" for stage in snapshot.stages) or snapshot.status == "done"
        else:
            assert snapshot.finished_at is None


def test_a_failure_does_not_kill_the_worker_thread():
    # a single failure must not take down the pool's one worker, or every later job would queue forever.
    runner = RecordingRunner(error=RuntimeError("boom"))
    manager = manager_for(runner)
    failed = manager.submit("doc-1")
    wait_for_status(manager, failed.job_id, "failed")

    runner.error = None
    following = manager.submit("doc-2")

    assert wait_for_status(manager, following.job_id, "done").status == "done"


# ---- mark ----------------------------------------------------------------------------


def test_mark_updates_only_the_named_stage():
    def body(job: Job, mark) -> None:
        mark("extract", "running", "mineru")

    _, job = run_to_completion(RecordingRunner(body=body))

    assert {s.name: (s.status, s.detail) for s in job.stages} == {
        "parse": ("pending", ""),
        "extract": ("running", "mineru"),
        "compare": ("pending", ""),
    }


def test_marking_an_unknown_stage_changes_nothing():
    def body(job: Job, mark) -> None:
        mark("upload", "done", "not a known stage")

    _, job = run_to_completion(RecordingRunner(body=body))

    assert [stage.status for stage in job.stages] == ["pending"] * len(STAGES)
    assert job.status == "done"


@pytest.mark.parametrize("status", ["pending", "running", "done", "failed", "skipped"])
def test_every_stage_status_can_be_recorded(status: StageStatus):
    def body(job: Job, mark) -> None:
        mark("parse", status, "")

    _, job = run_to_completion(RecordingRunner(body=body))

    assert job.stages[0].status == status


# ---- queries ----------------------------------------------------------------------------


def test_an_unknown_job_id_is_none():
    assert manager_for(RecordingRunner()).get("nope") is None


def test_for_document_only_returns_that_documents_jobs():
    runner = RecordingRunner()
    manager = manager_for(runner)
    mine = manager.submit("doc-1")
    manager.submit("doc-2")
    wait_for_status(
        manager, mine.job_id, "done"
    )  # while still running, a resubmission would be merged (see idempotency tests)
    also_mine = manager.submit("doc-1")
    wait_until(lambda: runner.call_count == 3, what="all three jobs to finish")

    assert [job.job_id for job in manager.for_document("doc-1")] == [mine.job_id, also_mine.job_id]
    assert manager.for_document("doc-3") == []


def test_for_document_sorts_by_creation_time_not_by_insertion_order(monkeypatch):
    # make the clock run backwards: if the implementation just returned insertion order, this would fail.
    clock = itertools.count(start=30, step=-1)
    monkeypatch.setattr("paperfacts.web.jobs._now", lambda: f"2026-01-01T00:00:{next(clock):02d}+00:00")
    runner = RecordingRunner()
    manager = manager_for(runner)

    earlier_submission = manager.submit("doc-1")
    wait_for_status(manager, earlier_submission.job_id, "done")
    later_submission = manager.submit("doc-1")
    wait_for_status(manager, later_submission.job_id, "done")

    assert [job.job_id for job in manager.for_document("doc-1")] == [
        later_submission.job_id,
        earlier_submission.job_id,
    ]


# ---- snapshots ------------------------------------------------------------------------------


def test_jobs_handed_out_are_frozen():
    """What's handed out is a frozen snapshot: neither later changes from the worker thread nor
    from a request thread can mutate it.

    A mutable object shared across threads would show up as "the web page occasionally shows
    self-contradictory progress".
    """
    runner = RecordingRunner()
    manager = manager_for(runner)
    job = manager.submit("doc-1")

    with pytest.raises(ValidationError):
        job.status = "done"
    with pytest.raises(ValidationError):
        job.stages[0].status = "failed"
    assert isinstance(job.stages, tuple) and isinstance(job.log, tuple)
    wait_for_status(manager, job.job_id, "done")


def test_a_snapshot_taken_while_running_does_not_change_afterwards():
    gate = threading.Event()
    runner = RecordingRunner(gate=gate, body=lambda job, mark: mark("parse", "done", "11 blocks"))
    manager = manager_for(runner)
    job = manager.submit("doc-1")
    running = wait_for_status(manager, job.job_id, "running")
    assert running.stages[0].status == "pending"

    gate.set()
    finished = wait_for_status(manager, job.job_id, "done")

    assert running.status == "running" and running.finished_at is None
    assert finished.stages[0].detail == "11 blocks"


# ---- logging ----------------------------------------------------------------------------


def test_the_package_logger_is_raised_to_info_during_the_job_and_restored_afterwards():
    """When serve runs without -v the process is at WARNING; without raising to INFO during the
    job, progress logs would never reach the panel."""
    logger = logging.getLogger(PAPERFACTS_LOGGER)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:

        def body(job: Job, mark) -> None:
            logging.getLogger("paperfacts.test").info("progress during the job")

        _, job = run_to_completion(RecordingRunner(body=body))

        assert any("progress during the job" in line for line in job.log)
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(previous)


def test_the_job_log_collects_paperfacts_records_from_the_worker_thread():
    def body(job: Job, mark) -> None:
        logging.getLogger("paperfacts.test").info("mineru parsing done")

    _, job = run_to_completion(RecordingRunner(body=body))

    assert len(job.log) == 1
    assert "mineru parsing done" in job.log[0]
    assert "INFO" in job.log[0] and "paperfacts.test" in job.log[0]


def test_records_from_other_threads_do_not_enter_the_job_log():
    """The handler is attached to the global logger, so it must filter by thread.

    Otherwise, an HTTP request thread's (or even another job's) log records would bleed into this
    job's log, and "this paper's processing history" on the page would be a lie.
    """
    gate = threading.Event()
    runner = RecordingRunner(
        gate=gate, body=lambda job, mark: logging.getLogger("paperfacts.test").info("from the worker thread")
    )
    manager = manager_for(runner)
    job = manager.submit("doc-1")
    assert runner.entered.wait(timeout=2.0)

    logging.getLogger("paperfacts.test").info("from the request thread")
    gate.set()
    finished = wait_for_status(manager, job.job_id, "done")

    assert [line for line in finished.log if "from the request thread" in line] == []
    assert any("from the worker thread" in line for line in finished.log)


def test_records_from_other_libraries_are_ignored():
    def body(job: Job, mark) -> None:
        logging.getLogger("httpx").info("HTTP Request: POST /v1/chat/completions")
        logging.getLogger("paperfacts.test").info("the one that should remain")

    _, job = run_to_completion(RecordingRunner(body=body))

    assert [line.split(": ", 1)[-1] for line in job.log] == ["the one that should remain"]


def test_the_log_keeps_only_the_most_recent_lines(monkeypatch):
    # a single paper's processing log shouldn't exhaust memory; truncation keeps the most recent tail.
    monkeypatch.setattr("paperfacts.web.jobs.MAX_LOG_LINES", 4)

    def body(job: Job, mark) -> None:
        for index in range(10):
            logging.getLogger("paperfacts.test").info("msg-%02d", index)

    _, job = run_to_completion(RecordingRunner(body=body))

    assert len(job.log) == 4
    assert "msg-06" in job.log[0]
    assert "msg-09" in job.log[-1]


def test_the_default_log_budget_is_generous_enough_for_a_whole_run():
    assert MAX_LOG_LINES >= 1000


def test_the_handler_is_removed_when_the_job_finishes():
    """Failing to remove the handler would add one more listener to the global logger per job run,
    slowing logging down over time."""
    logger = logging.getLogger(PAPERFACTS_LOGGER)
    before = list(logger.handlers)

    _, job = run_to_completion(RecordingRunner(error=RuntimeError("boom")))

    assert list(logger.handlers) == before
    logging.getLogger("paperfacts.test").info("log after the job finished")
    assert not any("log after the job finished" in line for line in job.log)


# ---- bounded history ------------------------------------------------------------------------------------


def test_only_the_most_recent_finished_jobs_are_kept(monkeypatch):
    # Job status lives in memory for the life of the process; without a bound every submission since
    # start-up would ride along on every /api/jobs request.
    monkeypatch.setattr("paperfacts.web.jobs.MAX_FINISHED_JOBS", 3)
    runner = RecordingRunner()
    manager = manager_for(runner)

    submitted = []
    for index in range(5):
        job = manager.submit(f"doc-{index}")
        wait_for_status(manager, job.job_id, "done")
        submitted.append(job.job_id)

    assert {job.job_id for job in manager.all_jobs()} == set(submitted[-3:])
    assert manager.get(submitted[0]) is None


def test_an_active_job_is_never_pruned(monkeypatch):
    monkeypatch.setattr("paperfacts.web.jobs.MAX_FINISHED_JOBS", 1)
    gate = threading.Event()

    def body(job: Job, mark) -> None:
        if job.document_id == "slow-doc":
            gate.wait(timeout=5)

    runner = RecordingRunner(body=body)
    manager = JobManager(runner, STAGES, workers=2)
    try:
        slow = manager.submit("slow-doc")
        wait_until(runner.entered.is_set, what="the slow job to start")
        quick = [manager.submit(f"doc-{index}") for index in range(3)]
        # one worker is held by the slow job, so the quick ones run one after another, in order
        wait_for_status(manager, quick[-1].job_id, "done")

        # the oldest submission is still running, so it stays; of the finished ones only the newest is kept
        assert {job.job_id for job in manager.all_jobs()} == {slow.job_id, quick[-1].job_id}
    finally:
        gate.set()
    wait_for_status(manager, slow.job_id, "done")
