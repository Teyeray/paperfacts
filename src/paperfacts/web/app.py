"""FastAPI application: the document library, upload-then-process, artifact and page-image reads,
and the static frontend.

Endpoints (all under ``/api``, JSON)::

    GET  /api/health
    GET  /api/documents                            document list (stage reached, counts)
    GET  /api/dataset                              corpus results table (one paper_row per document)
    GET  /api/dataset.xlsx                         the whole library as one Excel workbook
    POST /api/documents  (multipart file, ?force)  upload a PDF and queue it -> {document, job}
    POST /api/documents/run-all?force=             queue every unfinished document (or all, with
                                                    force) -> {submitted, skipped}
    POST /api/documents/{id}/run?force=            reprocess an existing document (reuses the
                                                    running job if the same document is active)
    GET  /api/documents/{id}                       single document summary
    GET  /api/documents/{id}/report                ComparisonReport
    GET  /api/documents/{id}/extraction/{backend}  normalized LaneExtraction
    GET  /api/documents/{id}/artifact/{backend}    ParsedArtifact (blocks + Markdown + page geometry)
    GET  /api/documents/{id}/dataset               consolidated per-sample table (rows + field list)
    GET  /api/documents/{id}/dataset.xlsx          the same data as the Excel workbook
    GET  /api/documents/{id}/figures               values read off the paper's charts (opt-in stage)
    GET  /api/documents/{id}/pages/{page}.png?dpi= rendered page image (cached)
    GET  /api/documents/{id}/jobs                  this document's job list
    GET  /api/jobs                                 every job of this process, newest first
    GET  /api/jobs/{job_id}                        job snapshot (stages, log)

All business logic lives in :mod:`paperfacts.workflow` (the job body is ``run_document``); this
module only does the HTTP mapping.
"""

from __future__ import annotations

import base64
import binascii
import logging
import secrets
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from paperfacts.compare import ComparisonReport
from paperfacts.config import Settings
from paperfacts.dataset import DatasetPayload, write_dataset
from paperfacts.figures import FiguresView
from paperfacts.models import Backend, ParsedArtifact
from paperfacts.parsers import install_runner_cleanup
from paperfacts.records import LaneExtraction
from paperfacts.storage import document_key
from paperfacts.web.documents import CorpusPayload, DocumentSummary, Library
from paperfacts.web.jobs import Job, JobManager, JobRunner
from paperfacts.workflow import run_document, shown_figures, stage_names

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_CHUNK_BYTES = 1 << 20
EXCEL_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class UploadAccepted(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: DocumentSummary
    job: Job


class SkippedDocument(BaseModel):
    """One document the bulk run did not queue, and why."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    name: str
    reason: str


class RunAllAccepted(BaseModel):
    model_config = ConfigDict(frozen=True)

    submitted: list[Job]
    skipped: list[SkippedDocument]


def pipeline_runner(settings: Settings, library: Library) -> JobRunner:
    """Job body: hand the document to workflow.run_document; the stage callback is just mark."""
    return lambda job, mark: run_document(library.document(job.document_id), settings, force=job.force, on_stage=mark)


def login_accepted(header: str | None, settings: Settings) -> bool:
    """Is this ``Authorization`` header the configured HTTP Basic login?

    ``compare_digest`` rather than ``==``: a wrong password should take the same time to reject whoever
    guesses it, so the check does not hand out the password's length or its matching prefix.
    """
    scheme, _, credentials = (header or "").partition(" ")
    if scheme.lower() != "basic" or not settings.web_password:
        return False
    try:
        decoded = base64.b64decode(credentials, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):  # not base64, or not even UTF-8: not our login
        return False
    username, separator, password = decoded.partition(":")
    if not separator:
        return False
    return secrets.compare_digest(username, settings.web_username) and secrets.compare_digest(
        password, settings.web_password
    )


def create_app(settings: Settings | None = None, *, jobs: JobManager | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    library = Library(settings)
    manager = jobs or JobManager(pipeline_runner(settings, library), stage_names())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Jobs run on a worker thread, where signal handlers cannot be installed: arm the runner cleanup
        # here, on the main thread, so a SIGTERM to the server takes the parser subprocesses with it.
        install_runner_cleanup()
        yield
        manager.shutdown()  # stop accepting new jobs on shutdown; a job already running ends with the process

    app = FastAPI(title="PaperFacts", version="0.1.0", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.library = library
    app.state.jobs = manager

    @app.middleware("http")
    async def require_login(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """The password gate, over every route -- the static frontend and ``/api`` alike.

        The app is published through a tunnel on this machine, so this is all that stands between the
        internet and a library that can upload PDFs and spend LLM tokens. Off unless a password is
        configured, which is what a laptop wants. A 401 carrying ``WWW-Authenticate`` is what makes a
        browser ask for the login instead of showing the app.
        """
        if not settings.web_password or login_accepted(request.headers.get("authorization"), settings):
            return await call_next(request)
        return PlainTextResponse(
            "PaperFacts: login required",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="PaperFacts", charset="UTF-8"'},
        )

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

    @app.get("/api/dataset")
    def get_corpus() -> CorpusPayload:
        """The home view's table: every document that has a dataset under the current keys. Reading N small
        JSON files is cheap enough that a cache would only be a way to serve a stale table."""
        return library.corpus()

    @app.get("/api/dataset.xlsx")
    def get_corpus_excel() -> Response:
        """The same corpus, rebuilt into one workbook. It is built on demand rather than read from disk:
        the per-document workbooks are not key-stamped, so only the datasets are a trustworthy source."""
        datasets = library.corpus_datasets()
        if not datasets:
            raise HTTPException(status_code=404, detail="No consolidated dataset yet")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paperfacts.xlsx"
            figure_views = (shown_figures(d.document_id, d.filename, settings) for d in datasets)
            write_dataset(datasets, path, figure_rows=[row for view in figure_views if view for row in view.rows])
            content = path.read_bytes()
        return Response(
            content=content,
            media_type=EXCEL_MEDIA_TYPE,
            headers={"content-disposition": 'attachment; filename="paperfacts-corpus.xlsx"'},
        )

    @app.post("/api/documents", status_code=202)
    async def upload_document(
        file: Annotated[UploadFile, File()], force: Annotated[bool, Query()] = False
    ) -> UploadAccepted:
        if (file.size or 0) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_bytes // (1024 * 1024)} MB")
        data = await _read_limited(file, settings.max_upload_bytes)
        if not data.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted (missing %PDF header)")
        # writing to disk is sync IO; offload to the threadpool so a multi-hundred-MB write can't stall the event loop
        document = await run_in_threadpool(library.register_upload, file.filename or "upload.pdf", data)
        key = document_key(document.document_id)
        job = manager.submit(key, force=force)
        return UploadAccepted(document=library.summary(key), job=job)

    # Registration order matters: FastAPI matches in order, so this literal route must stay above the
    # ``/api/documents/{document_id}`` routes, or "run-all" is read as a document id and answered with 404.
    @app.post("/api/documents/run-all", status_code=202)
    def run_all(force: Annotated[bool, Query()] = False) -> RunAllAccepted:
        """Queue every document that is not finished yet (or every document at all, with force).

        Same submission path as ``run_existing``, once per document in library order: a document
        already queued or running simply gets its existing job back, so pressing the button twice
        costs nothing.
        """
        submitted: list[Job] = []
        skipped: list[SkippedDocument] = []
        for summary in library.list():
            if not library.runnable(summary.document_id):
                skipped.append(
                    SkippedDocument(
                        document_id=summary.document_id, name=summary.name, reason="No PDF and no cached parse"
                    )
                )
                continue
            if summary.compared and not force:
                skipped.append(
                    SkippedDocument(
                        document_id=summary.document_id, name=summary.name, reason="Already processed under these keys"
                    )
                )
                continue
            submitted.append(manager.submit(summary.document_id, force=force))
        logger.info("run-all force=%s submitted=%d skipped=%d", force, len(submitted), len(skipped))
        return RunAllAccepted(submitted=submitted, skipped=skipped)

    @app.post("/api/documents/{document_id}/run", status_code=202)
    def run_existing(document_id: str, force: Annotated[bool, Query()] = False) -> Job:
        require_document(document_id)
        if not library.runnable(document_id):
            # A stored parse for both lanes is enough: extraction, comparison and export never open the PDF.
            raise HTTPException(
                status_code=409, detail="No PDF and no cached parse for this document; re-upload it to process it"
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

    @app.get("/api/documents/{document_id}/dataset")
    def get_dataset(document_id: str) -> DatasetPayload:
        require_document(document_id)
        dataset = library.dataset(document_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="No consolidated dataset yet")
        return dataset

    @app.get("/api/documents/{document_id}/figures")
    def get_figures(document_id: str) -> FiguresView:
        """The chart readings, read straight from their own file: they are never part of the dataset."""
        summary = require_document(document_id)
        figures = shown_figures(summary.document_id, summary.name, settings)
        if figures is None:
            raise HTTPException(status_code=404, detail="No figure readings")
        return figures

    @app.get("/api/documents/{document_id}/dataset.xlsx")
    def get_dataset_excel(document_id: str) -> FileResponse:
        summary = require_document(document_id)
        path = library.dataset_excel(document_id)
        if path is None:
            raise HTTPException(status_code=404, detail="No Excel export yet")
        return FileResponse(
            path,
            media_type=EXCEL_MEDIA_TYPE,
            # The upload name is user input; the id is the safe, stable download name.
            filename=f"paperfacts-{summary.document_id}.xlsx",
        )

    @app.get("/api/documents/{document_id}/pages/{page}.png")
    def get_page_image(
        document_id: str,
        page: int,
        dpi: Annotated[int, Query()] = settings.page_dpi,
    ) -> FileResponse:
        # Checked here rather than in the annotation: the bounds come from the settings this app was built
        # with, and `from __future__ import annotations` would leave FastAPI a string it cannot resolve.
        if not settings.page_dpi_min <= dpi <= settings.page_dpi_max:
            raise HTTPException(
                status_code=422, detail=f"dpi must be between {settings.page_dpi_min} and {settings.page_dpi_max}"
            )
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

    @app.get("/api/jobs")
    def list_all_jobs() -> list[Job]:
        """Every job of this process, newest first: one request tells the library list which
        documents are busy."""
        return manager.all_jobs()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> Job:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No job {job_id}")
        return job

    app.mount("/", _RevalidatedStaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


class _RevalidatedStaticFiles(StaticFiles):
    """Static files that browsers must revalidate on every load.

    There is no build step and no hashed filenames, so after a deploy the only thing standing between a
    user and last week's modules is the browser's heuristic freshness on a Last-Modified header -- which
    was observed serving a stale index.html for minutes. ``no-cache`` still allows caching; it only forces
    the conditional request, which the ETag answers with a 304.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["cache-control"] = "no-cache"
        return response


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
