"""Shared fixtures for the web-layer tests: a controllable fake job body, a helper to poll for a
job's status, and helpers to seed a document directory with artifacts.

Background jobs are the only threaded part of the whole codebase, so this module holds two hard
lines:

1. **Timing is controlled explicitly with** :class:`threading.Event`, never guessed at with
   ``sleep``;
2. **Waiting for a result always polls status with a timeout**, so a test never fails sporadically
   because the machine was slow, nor hangs forever because a job got stuck.

Every ``seed_*`` helper that plants artifacts goes through
:class:`~paperfacts.storage.paths.DataLayout`: a wrong path should show up as "the library can't
see it", not get papered over by a second, hand-written path living in the test.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from paperfacts.consensus import ComparisonCounts, ComparisonReport
from paperfacts.consensus.matching import SampleMatching
from paperfacts.extraction.records import LaneExtraction, SampleRecord
from paperfacts.models import Backend, DocumentInput, ParsedArtifact, SourceBlock
from paperfacts.storage.identity import DocumentIdentity, ensure_identity
from paperfacts.web.documents import Library
from paperfacts.web.jobs import Job, JobManager, JobStatus, StageStatus
from support.extraction import make_artifact, make_lane
from support.factories import DOC_ID

# The fake job body normally finishes in milliseconds; this ceiling only guards against a hung
# test, it is not "how long we expect to wait".
WAIT_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.005

# The document seeded into the document directory: its full sha256, and the 16-char directory name derived from it.
DOC_SHA = DOC_ID
DOC_KEY = DOC_ID[:16]

Mark = Callable[[str, StageStatus, str], None]


# ---- fake job body ------------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerCall:
    """The arguments observable from one call to the job body (all sourced from Job)."""

    document_id: str
    force: bool


class RecordingRunner:
    """A :data:`~paperfacts.web.jobs.JobRunner` that records every call.

    ``gate``: the job body blocks here until the test calls ``set()`` — used to pin down
    intermediate states like "queued" or "running";
    ``body``: the action to actually perform (e.g. calling ``mark`` or writing to the log);
    ``error``: an exception raised after ``body`` runs, used to exercise the failure path.
    """

    def __init__(
        self,
        *,
        gate: threading.Event | None = None,
        body: Callable[[Job, Mark], None] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.calls: list[RunnerCall] = []
        self.gate = gate
        self.body = body
        self.error = error
        self.entered = threading.Event()  # the job body actually started running (not just submitted)

    def __call__(self, job: Job, mark: Mark) -> None:
        self.calls.append(RunnerCall(document_id=job.document_id, force=job.force))
        self.entered.set()
        if self.gate is not None and not self.gate.wait(timeout=WAIT_TIMEOUT_S):
            raise AssertionError("job body timed out waiting for the gate: the test forgot to call set()")
        if self.body is not None:
            self.body(job, mark)
        if self.error is not None:
            raise self.error

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def forces(self) -> list[bool]:
        return [call.force for call in self.calls]

    @property
    def document_ids(self) -> list[str]:
        return [call.document_id for call in self.calls]


# ---- waiting ---------------------------------------------------------------------------


def wait_until(predicate: Callable[[], bool], *, what: str, timeout: float = WAIT_TIMEOUT_S) -> None:
    """Poll until ``predicate`` is true; on timeout, report what it was waiting for instead of
    leaving the test to hang silently."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(f"timed out waiting for {what!r} ({timeout}s)")


def wait_for_status(manager: JobManager, job_id: str, *statuses: JobStatus, timeout: float = WAIT_TIMEOUT_S) -> Job:
    """Wait until the job reaches one of the given statuses, and return that snapshot."""

    def reached() -> bool:
        job = manager.get(job_id)
        return job is not None and job.status in statuses

    wait_until(reached, what=f"job {job_id} to become {'/'.join(statuses)}", timeout=timeout)
    job = manager.get(job_id)
    assert job is not None
    return job


# ---- seeding a document directory with artifacts ---------------------------------------------


def seed_artifact(
    library: Library,
    backend: Backend,
    *,
    document_sha: str = DOC_SHA,
    blocks: Sequence[SourceBlock] | None = None,
) -> ParsedArtifact:
    """Write a ``parsed/<backend>.artifact.json`` (the on-disk evidence that this lane was parsed)."""
    artifact = make_artifact(blocks, backend=backend, document_id=document_sha)
    artifact.write(library.layout.artifact_path(document_sha, backend))
    return artifact


def seed_extraction(
    library: Library,
    backend: Backend,
    *,
    document_sha: str = DOC_SHA,
    samples: Iterable[SampleRecord] = (),
    extractor_key: str | None = None,
) -> LaneExtraction:
    """Write a ``facts/<backend>.<extractor_key>.json``; ``extractor_key`` defaults to the
    library's current one."""
    key = library.extractor_key if extractor_key is None else extractor_key
    lane = make_lane(backend=backend, samples=samples, document_id=document_sha, extractor_key=key)
    lane.write(library.layout.extraction_path(document_sha, backend, key))
    return lane


def seed_report(
    library: Library,
    *,
    document_sha: str = DOC_SHA,
    counts: ComparisonCounts | None = None,
    extractor_key: str | None = None,
    comparison_key: str | None = None,
) -> ComparisonReport:
    """Write a ``comparisons/<extractor_key>.<comparison_key>.json``."""
    key = library.extractor_key if extractor_key is None else extractor_key
    cmp_key = library.comparison_key if comparison_key is None else comparison_key
    report = ComparisonReport(
        document_id=document_sha,
        extractor_key=key,
        comparison_key=cmp_key,
        backend_a="mineru",
        backend_b="paddleocr_vl",
        matching=SampleMatching(),
        counts=counts or ComparisonCounts(),
    )
    report.write(library.layout.comparison_path(document_sha, key, cmp_key))
    return report


def seed_cli_document(library: Library, pdf: Path, *, document_sha: str = DOC_SHA) -> DocumentIdentity:
    """Seed the identity (``identity.json``) left behind by a document processed via the **CLI**:
    it has an original path and no source.pdf.

    This goes through the same ``ensure_identity`` step that ``parse_document`` uses, so the shape
    of the identity file can't drift from the real pipeline.
    """
    document = DocumentInput(document_id=document_sha, pdf_path=pdf, sha256=document_sha)
    return ensure_identity(library.layout, document)
