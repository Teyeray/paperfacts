"""Several workers: different documents at once, one document never twice at once, queue order kept.

The single-worker invariants live in test_web_jobs.py and still hold here: the job is a frozen value
replaced whole under the lock, and a document submitted twice while active gets the same job. What more
workers add is overlap, so every case stages it with barriers and events -- a barrier of N opens only when
N bodies are inside at the same moment -- and waits with timeouts, never with sleeps.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from paperfacts.web.jobs import Job, JobManager
from support.web import WAIT_TIMEOUT_S, RecordingRunner, wait_for_status, wait_until

STAGES: tuple[str, ...] = ("parse", "extract", "compare")


class Overlap:
    """A job body that counts the bodies running per document and in total, then waits for ``release``."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.running: dict[str, int] = {}
        self.peak_per_document: dict[str, int] = {}
        self.peak_total = 0
        self.started: list[tuple[str, bool]] = []
        self._lock = threading.Lock()

    def __call__(self, job: Job, mark) -> None:
        with self._lock:
            self.running[job.document_id] = self.running.get(job.document_id, 0) + 1
            self.peak_per_document[job.document_id] = max(
                self.peak_per_document.get(job.document_id, 0), self.running[job.document_id]
            )
            self.peak_total = max(self.peak_total, sum(self.running.values()))
            self.started.append((job.document_id, job.force))
        try:
            if not self.release.wait(timeout=WAIT_TIMEOUT_S):
                raise AssertionError("the test never released the jobs")
        finally:
            with self._lock:
                self.running[job.document_id] -= 1

    def total(self) -> int:
        with self._lock:
            return sum(self.running.values())


def test_different_documents_run_at_the_same_time():
    barrier = threading.Barrier(3)
    runner = RecordingRunner(body=lambda job, mark: barrier.wait(timeout=WAIT_TIMEOUT_S))
    manager = JobManager(runner, STAGES, workers=3)

    jobs = [manager.submit(f"doc-{i}") for i in range(3)]

    for job in jobs:
        assert wait_for_status(manager, job.job_id, "done", "failed").status == "done"
    assert not barrier.broken


def test_no_more_documents_run_than_there_are_workers():
    body = Overlap()
    manager = JobManager(body, STAGES, workers=2)
    jobs = [manager.submit(f"doc-{i}") for i in range(5)]
    wait_until(lambda: body.total() == 2, what="both workers to be busy")
    assert [manager.get(job.job_id).status for job in jobs].count("queued") == 3

    body.release.set()
    for job in jobs:
        wait_for_status(manager, job.job_id, "done")
    assert body.peak_total == 2


def test_a_forced_rerun_of_a_running_document_waits_for_it_while_other_documents_go_ahead():
    """The force upgrade is the one way a document gets two active jobs; the second must not start until the
    first has finished, or two workers would write the same document directory at once."""
    body = Overlap()
    manager = JobManager(body, STAGES, workers=3)
    first = manager.submit("doc-1")
    wait_until(lambda: body.total() == 1, what="the first job to start")
    forced = manager.submit("doc-1", force=True)
    other = manager.submit("doc-2")

    # doc-2 was queued after the forced job, so once it runs the forced one has been looked at and passed
    # over; it is still queued although a worker is free.
    wait_until(lambda: body.total() == 2, what="the other document to start")
    assert manager.get(forced.job_id).status == "queued"
    assert manager.get(other.job_id).status == "running"

    body.release.set()
    for job in (first, forced, other):
        wait_for_status(manager, job.job_id, "done")
    assert body.peak_per_document == {"doc-1": 1, "doc-2": 1}
    assert [started for started in body.started if started[0] == "doc-1"] == [("doc-1", False), ("doc-1", True)]


def test_jobs_start_in_the_order_they_were_submitted():
    gate = threading.Event()
    runner = RecordingRunner(gate=gate)
    manager = JobManager(runner, STAGES, workers=1)
    first = manager.submit("doc-0")
    assert runner.entered.wait(timeout=WAIT_TIMEOUT_S)
    rest = [manager.submit(f"doc-{i}") for i in range(1, 5)]

    gate.set()
    for job in (first, *rest):
        wait_for_status(manager, job.job_id, "done")
    assert runner.document_ids == [f"doc-{i}" for i in range(5)]


def test_simultaneous_submissions_of_one_document_share_one_job():
    body = Overlap()
    manager = JobManager(body, STAGES, workers=4)
    barrier = threading.Barrier(8)

    def submit() -> str:
        barrier.wait(timeout=WAIT_TIMEOUT_S)
        return manager.submit("doc-1").job_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = {future.result(timeout=WAIT_TIMEOUT_S) for future in [pool.submit(submit) for _ in range(8)]}

    body.release.set()
    (job_id,) = ids
    wait_for_status(manager, job_id, "done")
    assert body.started == [("doc-1", False)]


def test_stage_marks_from_concurrent_jobs_land_on_their_own_job():
    barrier = threading.Barrier(2)

    def body(job: Job, mark) -> None:
        barrier.wait(timeout=WAIT_TIMEOUT_S)
        for stage in STAGES:
            mark(stage, "done", job.document_id)

    manager = JobManager(RecordingRunner(body=body), STAGES, workers=2)
    jobs = [manager.submit(f"doc-{i}") for i in range(2)]

    for job in jobs:
        finished = wait_for_status(manager, job.job_id, "done")
        assert [(stage.status, stage.detail) for stage in finished.stages] == [("done", job.document_id)] * 3


def test_the_logs_of_concurrent_jobs_do_not_mix():
    """Each job's log is "this paper's processing history": a record from another paper's lane is a lie."""
    barrier = threading.Barrier(2)

    def body(job: Job, mark) -> None:
        barrier.wait(timeout=WAIT_TIMEOUT_S)
        log = logging.getLogger("paperfacts.test")
        log.warning("worker of %s", job.document_id)
        pool_thread = threading.Thread(
            target=contextvars.copy_context().run, args=(log.warning, "pool of %s", job.document_id)
        )
        pool_thread.start()
        pool_thread.join()
        barrier.wait(timeout=WAIT_TIMEOUT_S)  # both have logged before either finishes

    manager = JobManager(RecordingRunner(body=body), STAGES, workers=2)
    jobs = [manager.submit(f"doc-{i}") for i in range(2)]

    for job in jobs:
        finished = wait_for_status(manager, job.job_id, "done")
        messages = [line.split(": ", 1)[-1] for line in finished.log]
        assert messages == [f"worker of {job.document_id}", f"pool of {job.document_id}"]


def test_the_package_logger_stays_at_info_until_the_last_running_job_ends():
    logger = logging.getLogger("paperfacts")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    first_done = threading.Event()
    both_in = threading.Barrier(2)
    try:

        def body(job: Job, mark) -> None:
            both_in.wait(timeout=WAIT_TIMEOUT_S)
            if job.document_id == "doc-late":
                assert first_done.wait(timeout=WAIT_TIMEOUT_S)
                logging.getLogger("paperfacts.test").info("after the other job ended")

        manager = JobManager(RecordingRunner(body=body), STAGES, workers=2)
        early = manager.submit("doc-early")
        late = manager.submit("doc-late")
        wait_for_status(manager, early.job_id, "done")
        first_done.set()
        finished = wait_for_status(manager, late.job_id, "done")

        assert any("after the other job ended" in line for line in finished.log)
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(previous)


def test_shutdown_drops_queued_jobs_and_lets_running_ones_finish():
    body = Overlap()
    manager = JobManager(body, STAGES, workers=2)
    running = [manager.submit(f"doc-{i}") for i in range(2)]
    wait_until(lambda: body.total() == 2, what="both workers to be busy")
    queued = manager.submit("doc-queued")

    manager.shutdown()
    body.release.set()

    for job in running:
        assert wait_for_status(manager, job.job_id, "done").status == "done"
    assert manager.get(queued.job_id).status == "queued"
    assert [document for document, _ in body.started] == ["doc-0", "doc-1"]


@pytest.mark.parametrize("workers", [0, -1])
def test_a_manager_needs_at_least_one_worker(workers: int):
    with pytest.raises(ValueError, match="at least one worker"):
        JobManager(RecordingRunner(), STAGES, workers=workers)
