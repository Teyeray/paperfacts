"""Background jobs: a single worker thread runs "parse -> extract -> compare" serially, and the
web frontend polls for status and logs.

Serial execution is deliberate: running both parser models at once blows out memory on a dev
machine; on the server the parsers are HTTP services, so serial is fast enough there too. Job
status lives only in memory — it describes "what this process is doing right now", while the
artifacts written to disk are the persistent truth.

Shared state has exactly one shape: a frozen :class:`Job`, kept in a dict guarded by a lock. Every
state change swaps in a whole new value inside the lock (``model_copy``). A request thread always
gets a complete, self-consistent snapshot from one instant — never a half-written one where
``status`` is already "done" but ``finished_at`` is still empty.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Literal

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
    def __init__(self, runner: JobRunner, stage_names: tuple[str, ...]) -> None:
        self._runner = runner
        self._stage_names = stage_names
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paperfacts-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, document_id: str, *, force: bool = False) -> Job:
        """Idempotent: if the same document is already queued or running, reuse that job (double-
        clicking "reprocess" shouldn't pay for the LLM call twice); a new job only starts when
        upgrading from "use cache" to "force rerun"."""
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
        self._executor.submit(self._run, job.job_id)
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
        """Stop accepting new jobs; queued ones are dropped, a running one ends with the process
        (not awaited — a single paper's pipeline can take minutes)."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ---- state changes: swap in a whole new value inside the lock, every time ----

    def _update(self, job_id: str, **changes: Any) -> Job:
        with self._lock:
            job = self._jobs[job_id].model_copy(update=changes)
            self._jobs[job_id] = job
            return job

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

    # ---- worker thread -----------------------------------------------------------------------

    def _run(self, job_id: str) -> None:
        handler = _JobLogHandler(self, job_id)
        # the process may default to WARNING (when serve runs without -v); raise it to INFO for
        # the duration of the job so progress logs reach the panel, then restore it afterwards
        package_logger = logging.getLogger("paperfacts")
        previous_level = package_logger.level
        if not package_logger.isEnabledFor(logging.INFO):
            package_logger.setLevel(logging.INFO)
        package_logger.addHandler(handler)
        job = self._update(job_id, status="running", started_at=_now())

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
            package_logger.removeHandler(handler)
            package_logger.setLevel(previous_level)


# The pipeline fans out onto pools named with this prefix (workflow.run_document, extract._extract_passages).
# One job runs at a time, so a record from such a thread can only belong to the current job.
PIPELINE_THREAD_PREFIX = "paperfacts-"


class _JobLogHandler(logging.Handler):
    """Collect this job's paperfacts.* log records into job.log: those from the worker thread itself and
    those from the pipeline's own pools, so HTTP request-thread logs never leak in."""

    def __init__(self, manager: JobManager, job_id: str) -> None:
        super().__init__(level=logging.INFO)
        self._manager = manager
        self._job_id = job_id
        self._thread = threading.get_ident()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self._thread or (record.threadName or "").startswith(PIPELINE_THREAD_PREFIX):
            self._manager._append_log(self._job_id, self.format(record))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
