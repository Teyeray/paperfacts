"""Background jobs: a few worker threads run "parse -> extract -> compare", one document each, and the
web frontend polls for status and logs.

``web.max_parallel_documents`` workers, never two on the same document. Parsing still runs one paper per
parser at a time (the parse locks in :mod:`paperfacts.parsers`) and every model request takes a slot of
the process-wide in-flight limit (:mod:`paperfacts.llm`), so extra workers overlap the long model waits
rather than multiplying load on the GPU or the endpoint. Job status lives only in memory -- it describes
"what this process is doing right now", while the artifacts written to disk are the persistent truth.

Shared state has exactly one shape: a frozen :class:`Job`, kept in a dict guarded by a lock. Every
state change swaps in a whole new value inside the lock (``model_copy``). A request thread always
gets a complete, self-consistent snapshot from one instant -- never a half-written one where
``status`` is already "done" but ``finished_at`` is still empty. The queue and the set of documents being
worked on live under the same lock, so taking a job off the queue and marking it running is one step.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from paperfacts.workflow import StageStatus

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "failed"]
ACTIVE: frozenset[str] = frozenset({"queued", "running"})
MAX_LOG_LINES = 2000
MAX_TRACEBACK_CHARS = 2000


class Stage(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: StageStatus = "pending"
    detail: str = ""


class Job(BaseModel):
    """A snapshot of one background job. ``stages`` is ordered by execution order; the frontend
    draws progress from it."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    document_id: str
    force: bool = False
    status: JobStatus = "queued"
    stages: tuple[Stage, ...] = ()
    log: tuple[str, ...] = ()
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


# Job body: receives a job snapshot (only document_id / force are used) and a mark(stage, status, detail) callback
JobRunner = Callable[[Job, Callable[[str, StageStatus, str], None]], None]


class JobManager:
    def __init__(self, runner: JobRunner, stage_names: tuple[str, ...], *, workers: int = 1) -> None:
        if workers < 1:
            raise ValueError(f"a job manager needs at least one worker, got {workers}")
        self._runner = runner
        self._stage_names = stage_names
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="paperfacts-job")
        self._jobs: dict[str, Job] = {}
        # Job ids waiting for a worker, oldest first, and the documents a worker holds right now.
        self._queue: deque[str] = deque()
        self._busy: set[str] = set()
        self._closed = False
        self._lock = threading.Lock()

    def submit(self, document_id: str, *, force: bool = False) -> Job:
        """Idempotent: if the same document is already queued or running, reuse that job (double-
        clicking "reprocess" shouldn't pay for the LLM call twice); a new job only starts when
        upgrading from "use cache" to "force rerun", and then only after the running one has finished."""
        with self._lock:
            for job in self._jobs.values():
                if job.document_id == document_id and job.status in ACTIVE and job.force >= force:
                    return job
            job = Job(
                job_id=uuid.uuid4().hex[:12],
                document_id=document_id,
                force=force,
                created_at=_now(),
                stages=tuple(Stage(name=name) for name in self._stage_names),
            )
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
        # One turn of a worker per submission. The turn takes the oldest job it may run, which is not
        # necessarily this one; a worker that finishes looks again, so a job passed over because its
        # document was busy is picked up the moment that document is free.
        self._executor.submit(self._work)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> list[Job]:
        """Every job this process knows about, newest first. The dict is small (one entry per
        submission since start-up), so the frontend can learn which documents are busy with one
        request instead of one per row."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def for_document(self, document_id: str) -> list[Job]:
        with self._lock:
            return sorted((j for j in self._jobs.values() if j.document_id == document_id), key=lambda j: j.created_at)

    def shutdown(self) -> None:
        """Stop accepting new jobs; queued ones are dropped, running ones end with the process
        (not awaited -- a single paper's pipeline can take minutes)."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ---- state changes: swap in a whole new value inside the lock, every time ----

    def _mark(self, job_id: str, stage: str, status: StageStatus, detail: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            stages = tuple(
                s.model_copy(update={"status": status, "detail": detail}) if s.name == stage else s for s in job.stages
            )
            self._jobs[job_id] = job.model_copy(update={"stages": stages})

    def _append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(update={"log": (*job.log, line)[-MAX_LOG_LINES:]})

    def _finish(self, job_id: str, status: JobStatus, error: str | None, *, extra_log: str | None = None) -> None:
        """Write the terminal state all at once: status, error, the stage that was running, log,
        and finish time, all inside the same lock."""
        with self._lock:
            job = self._jobs[job_id]
            stages = job.stages
            if status == "failed":  # whichever stage was running is the one that failed
                stages = tuple(
                    s.model_copy(update={"status": "failed", "detail": error or ""}) if s.status == "running" else s
                    for s in stages
                )
            log = (*job.log, extra_log)[-MAX_LOG_LINES:] if extra_log else job.log
            self._jobs[job_id] = job.model_copy(
                update={"status": status, "error": error, "stages": stages, "log": log, "finished_at": _now()}
            )

    # ---- worker threads ----------------------------------------------------------------------

    def _work(self) -> None:
        """Run queued jobs until none is runnable: none left, or every one left is for a busy document."""
        while True:
            with self._lock:
                job = None if self._closed else self._take()
            if job is None:
                return
            try:
                self._run(job)
            finally:
                with self._lock:
                    self._busy.discard(job.document_id)

    def _take(self) -> Job | None:
        """Under the lock: the oldest queued job whose document no worker holds, now marked running.

        Skipping a busy document rather than waiting on it keeps the other workers busy; running it would
        let two workers write the same document directory at once.
        """
        for job_id in self._queue:
            job = self._jobs[job_id]
            if job.document_id not in self._busy:
                self._queue.remove(job_id)
                self._busy.add(job.document_id)
                job = job.model_copy(update={"status": "running", "started_at": _now()})
                self._jobs[job_id] = job
                return job
        return None

    def _run(self, job: Job) -> None:
        job_id = job.job_id
        handler = _JobLogHandler(self, job_id)
        token = _CURRENT_JOB.set(job_id)
        _raise_package_level()
        logging.getLogger(PACKAGE_LOGGER).addHandler(handler)

        def mark(stage: str, status: StageStatus, detail: str = "") -> None:
            self._mark(job_id, stage, status, detail)

        try:
            self._runner(job, mark)
        except Exception as exc:  # any failure must reach the snapshot; the thread must not die silently
            error = f"{type(exc).__name__}: {exc}"
            self._finish(job_id, "failed", error, extra_log=traceback.format_exc()[-MAX_TRACEBACK_CHARS:])
            logger.error("job %s failed: %s", job_id, error)
        except BaseException:  # KeyboardInterrupt / SystemExit: record it and re-raise, never leave the job "running"
            self._finish(job_id, "failed", "interrupted")
            raise
        else:
            self._finish(job_id, "done", None)
        finally:
            logging.getLogger(PACKAGE_LOGGER).removeHandler(handler)
            _restore_package_level()
            _CURRENT_JOB.reset(token)


PACKAGE_LOGGER = "paperfacts"
# The job whose work the current thread is doing. The worker sets it; the pipeline's pools copy their
# caller's context into every task they run (workflow, extract, figures), so a record from a lane or a
# field question carries the job it belongs to even with several jobs running at once. A request thread
# has none.
_CURRENT_JOB: contextvars.ContextVar[str | None] = contextvars.ContextVar("paperfacts_job", default=None)

# The process may default to WARNING (serve without -v); while any job runs, the package logger is raised to
# INFO so progress reaches the panel, and it is restored when the last running job ends. Counted, because
# with several workers the first job to finish must not lower it under the others.
_level_lock = threading.Lock()
_level_holders = 0
_saved_level = logging.NOTSET


def _raise_package_level() -> None:
    global _level_holders, _saved_level
    package = logging.getLogger(PACKAGE_LOGGER)
    with _level_lock:
        if _level_holders == 0:
            _saved_level = package.level
            if not package.isEnabledFor(logging.INFO):
                package.setLevel(logging.INFO)
        _level_holders += 1


def _restore_package_level() -> None:
    global _level_holders
    with _level_lock:
        _level_holders -= 1
        if _level_holders == 0:
            logging.getLogger(PACKAGE_LOGGER).setLevel(_saved_level)


class _JobLogHandler(logging.Handler):
    """Collect this job's paperfacts.* log records into job.log: those emitted while doing this job's work,
    on its worker or on the pipeline's pools, so HTTP request threads and other jobs never leak in."""

    def __init__(self, manager: JobManager, job_id: str) -> None:
        super().__init__(level=logging.INFO)
        self._manager = manager
        self._job_id = job_id
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        # A handler runs on the thread that logged, so the context read here is the emitter's.
        if _CURRENT_JOB.get() == self._job_id:
            self._manager._append_log(self._job_id, self.format(record))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
