"""FastAPI application: the document library, upload-then-process, artifact and page-image reads,
and the static frontend.

Endpoints (all under ``/api``, JSON)::

    GET  /api/health
    GET  /api/documents                            document list (stage reached, counts)
    POST /api/documents  (multipart file, ?force)  upload a PDF and queue it -> {document, job}
    POST /api/documents/{id}/run?force=            reprocess an existing document (reuses the
                                                    running job if the same document is active)
    GET  /api/documents/{id}                       single document summary
    GET  /api/documents/{id}/report                ComparisonReport
    GET  /api/documents/{id}/extraction/{backend}  normalized LaneExtraction
    GET  /api/documents/{id}/artifact/{backend}    ParsedArtifact (blocks + Markdown + page geometry)
    GET  /api/documents/{id}/pages/{page}.png?dpi= rendered page image (cached)
    GET  /api/documents/{id}/jobs                  this document's job list
    GET  /api/jobs/{job_id}                        job snapshot (stages, log)

All business logic lives in :mod:`paperfacts.workflow` (the job body is ``run_document``); this
module only does the HTTP mapping.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from paperfacts.config import Settings
from paperfacts.consensus import ComparisonReport
from paperfacts.extraction.records import LaneExtraction
from paperfacts.models import Backend, ParsedArtifact
from paperfacts.storage.paths import document_key
from paperfacts.web.documents import DocumentSummary, Library
from paperfacts.web.jobs import Job, JobManager, JobRunner
from paperfacts.workflow import run_document, stage_names

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1 << 20
DEFAULT_PAGE_DPI = 110
MIN_PAGE_DPI, MAX_PAGE_DPI = 50, 220


class UploadAccepted(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: DocumentSummary
    job: Job


def pipeline_runner(settings: Settings, library: Library) -> JobRunner:
    """Job body: hand the document to workflow.run_document; the stage callback is just mark."""
    return lambda job, mark: run_document(library.document(job.document_id), settings, force=job.force, on_stage=mark)


def create_app(settings: Settings | None = None, *, jobs: JobManager | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    library = Library(settings)
    manager = jobs or JobManager(pipeline_runner(settings, library), stage_names())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown()  # stop accepting new jobs on shutdown; a job already running ends with the process

    app = FastAPI(title="PaperFacts", version="0.1.0", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.library = library
    app.state.jobs = manager

    def require_document(document_id: str) -> DocumentSummary:
        try:
            exists = library.exists(document_id)
        except KeyError as exc:  # malformed id shape: always 404 to callers, never leak the internal convention
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not exists:
            raise HTTPException(status_code=404, detail=f"No document {document_id}")
        return library.summary(document_id)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": settings.llm_model}

    @app.get("/api/documents")
    def list_documents() -> list[DocumentSummary]:
        return library.list()

    @app.post("/api/documents", status_code=202)
    async def upload_document(
        file: Annotated[UploadFile, File()], force: Annotated[bool, Query()] = False
    ) -> UploadAccepted:
        if (file.size or 0) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        data = await _read_limited(file, MAX_UPLOAD_BYTES)
        if not data.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted (missing %PDF header)")
        # writing to disk is sync IO; offload to the threadpool so a multi-hundred-MB write can't stall the event loop
        document = await run_in_threadpool(library.register_upload, file.filename or "upload.pdf", data)
        key = document_key(document.document_id)
        job = manager.submit(key, force=force)
        return UploadAccepted(document=library.summary(key), job=job)

    @app.post("/api/documents/{document_id}/run", status_code=202)
    def run_existing(document_id: str, force: Annotated[bool, Query()] = False) -> Job:
        require_document(document_id)
        if library.pdf_path(document_id) is None:
            raise HTTPException(
                status_code=409, detail="This document has no available PDF to reprocess; please re-upload"
            )
        return manager.submit(document_id, force=force)

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str) -> DocumentSummary:
        return require_document(document_id)

    @app.get("/api/documents/{document_id}/report")
    def get_report(document_id: str) -> ComparisonReport:
        require_document(document_id)
        report = library.report(document_id)
        if report is None:
            raise HTTPException(status_code=404, detail="No comparison report yet")
        return report

    @app.get("/api/documents/{document_id}/extraction/{backend}")
    def get_extraction(document_id: str, backend: Backend) -> LaneExtraction:
        require_document(document_id)
        lane = library.extraction(document_id, backend)
        if lane is None:
            raise HTTPException(status_code=404, detail=f"No extraction results yet for {backend}")
        return lane

    @app.get("/api/documents/{document_id}/artifact/{backend}")
    def get_artifact(document_id: str, backend: Backend) -> ParsedArtifact:
        require_document(document_id)
        artifact = library.artifact(document_id, backend)
        if artifact is None:
            raise HTTPException(status_code=404, detail=f"No parsed artifact yet for {backend}")
        return artifact

    @app.get("/api/documents/{document_id}/pages/{page}.png")
    def get_page_image(
        document_id: str,
        page: int,
        dpi: Annotated[int, Query(ge=MIN_PAGE_DPI, le=MAX_PAGE_DPI)] = DEFAULT_PAGE_DPI,
    ) -> FileResponse:
        require_document(document_id)
        try:
            path = library.page_image(document_id, page, dpi=dpi)
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/png")

    @app.get("/api/documents/{document_id}/jobs")
    def list_jobs(document_id: str) -> list[Job]:
        require_document(document_id)
        return manager.for_document(document_id)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Job:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No job {job_id}")
        return job

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Read the upload body in chunks, raising 413 as soon as the limit is exceeded instead of
    buffering the whole file into memory before checking."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"File exceeds {limit // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)
