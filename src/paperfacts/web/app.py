"""FastAPI application: the document library, upload-then-process, artifact and page-image reads,
and the static frontend.

Endpoints (all under ``/api``, JSON)::

    GET  /api/health
    GET  /api/profile                              the served profile's title, UI copy, groups and fields
    GET  /api/documents                            document list (stage reached, counts)
    GET  /api/dataset                              corpus results table (paper_row + every sample row per document)
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
    GET  /api/documents/{id}/jobs                  this document's job list (with stages and logs)
    GET  /api/jobs                                 the jobs this process holds, newest first, without logs
    GET  /api/jobs/{job_id}                        job snapshot (stages, log)

All business logic lives in :mod:`paperfacts.workflow` (the job body is ``run_document``); this
module only does the HTTP mapping, and guards its edge: the login, a same-origin check on every request
that changes something, frame and sniffing headers, and an upload size counted as the body arrives.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from starlette.datastructures import UploadFile
from starlette.types import Message

from paperfacts.compare import ComparisonReport
from paperfacts.config import Settings
from paperfacts.dataset import DatasetPayload
from paperfacts.errors import ConfigError
from paperfacts.llm import set_max_in_flight
from paperfacts.models import Backend, ParsedArtifact
from paperfacts.parsers import install_runner_cleanup
from paperfacts.profile import DomainProfile
from paperfacts.profile_loader import IDENTIFIER, loaded_file_sha256, profile_path
from paperfacts.readings import FiguresView, shown_figures
from paperfacts.records import LaneExtraction
from paperfacts.storage import document_key
from paperfacts.web.documents import CorpusPayload, DocumentSummary, Library
from paperfacts.web.jobs import Job, JobBrief, JobManager, JobRunner
from paperfacts.workflow import StageCallback, corpus_workbook, load_run_profile, run_document, stage_names

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_CHUNK_BYTES = 1 << 20
# Room for the multipart boundary and part headers around the one file an upload carries.
UPLOAD_OVERHEAD_BYTES = 64 * 1024
EXCEL_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# On every response, the login prompt included: nothing here is meant to be framed (the run buttons could
# otherwise be clickjacked), and no response should be sniffed into a type it was not served as.
SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
}


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


def same_origin(request: Request) -> bool:
    """Did this request come from a page of this app, as far as the browser says?

    A browser attaches cached Basic credentials to a cross-site form POST, so without this check any page
    the operator visits could queue a forced rerun of the whole library. Browsers state where a request
    came from (``Sec-Fetch-Site``, ``Origin``, ``Referer``); a request that states nothing is not from a
    browser, and a client that is not a browser cannot be tricked into sending the operator's login.

    ``Sec-Fetch-Site`` is the browser's own verdict, which page script cannot set, so when it is there it
    decides alone (``none`` is the user typing the URL or following a bookmark). Comparing hosts instead
    would refuse the real UI behind any proxy that rewrites ``Host``. Only a browser that sends no Fetch
    Metadata falls back to the host comparison, and only the host: behind the tunnel the page is https
    while this server sees plain http.
    """
    site = request.headers.get("sec-fetch-site")
    if site is not None:
        return site in {"same-origin", "none"}
    source = request.headers.get("origin") or request.headers.get("referer")
    if source is None:
        return True
    if source == "null":  # a sandboxed frame or a local file: never this app's own page
        return False
    hosts = {request.headers.get("host"), (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()}
    return urlsplit(source).netloc in hosts - {None, ""}


def profile_origin(settings: Settings, profile: DomainProfile) -> Path:
    """The path ``profile`` is named by, unresolved: the settings' own when it leads to the file the profile was
    loaded from, so a symlink retargeted later is followed again; otherwise that file itself."""
    named = profile_path(settings)
    return named if named.resolve() == profile.source else profile.source


def profile_file_changed(profile: DomainProfile, origin: Path) -> bool:
    """Whether the file ``origin`` leads to now holds other bytes than ``profile`` was loaded from. Every byte
    counts, display text included: a server keeps showing the text it started with. Raises OSError when the file
    cannot be read; a profile built in memory has no file, and never changed."""
    loaded = loaded_file_sha256(profile)
    if loaded is None:
        return False
    return hashlib.sha256(origin.resolve().read_bytes()).hexdigest() != loaded


def profile_view(profile: DomainProfile) -> dict[str, Any]:
    """What the page needs to name things the profile's way: its title, its copy and its groups and fields.
    Display text only; the prompts, units and retrieval stay on the server."""
    return {
        "name": profile.name,
        "title_zh": profile.title_zh,
        "maturity": profile.maturity,
        "description_zh": profile.description_zh,
        "ui": dataclasses.asdict(profile.ui),
        "groups": [
            {"name": group.name, "level": group.level, "label_zh": group.label_zh, "entity": group.entity}
            for group in profile.groups
        ],
        # The kinds of sample, the primary first; a profile without entity types has the one implicit entity.
        "entities": [
            {
                "name": entity.name,
                "label_zh": entity.label_zh,
                "fields": [spec.name for spec in profile.entity_fields(entity)],
            }
            for entity in profile.entities
        ],
        "fields": [
            {
                "name": spec.name,
                "label": spec.label,
                "group": spec.group,
                "level": spec.level,
                "unit": spec.canonical_unit,
                "entity": spec.entity,
            }
            for spec in profile.fields
        ],
        "field_count": {"paper": len(profile.paper_fields), "sample": len(profile.sample_fields)},
    }


def pipeline_runner(settings: Settings, profile: DomainProfile, library: Library) -> JobRunner:
    """Job body: hand the document to workflow.run_document; the stage callback is just mark.

    The profile file is checked first: after an edit on disk the server would still run the old profile and
    store its results under keys the edited file no longer names, so the job is refused until a restart."""

    origin = profile_origin(settings, profile)

    def run(job: Job, mark: StageCallback) -> None:
        try:
            changed = profile_file_changed(profile, origin)
        except OSError as exc:
            # Deleted, locked, or caught mid-save: not known to have changed, and not safe to run under either.
            # The job's error reaches the browser, so it names the file only; the log has the path and the error
            # (an OSError's message carries the absolute path too).
            logger.error("cannot read the profile file %s (%s)", origin, exc)
            raise ConfigError(
                f"无法读取领域配置文件 {origin.name}，请检查后重启服务器 "
                f"(cannot read the profile file {origin.name}: {type(exc).__name__})"
            ) from exc
        if changed:
            logger.error("profile %s changed on disk (%s); restart the server", profile.name, origin)
            raise ConfigError(f"领域配置 {profile.source.name} 在磁盘上已改动，请重启服务器 (profile changed on disk)")
        run_document(library.document(job.document_id), settings, profile, force=job.force, on_stage=mark)

    return run


def login_accepted(header: str | None, settings: Settings) -> bool:
    """Is this ``Authorization`` header the configured HTTP Basic login?

    ``compare_digest`` rather than ``==``: a wrong password should take the same time to reject whoever
    guesses it, so the check does not hand out the password's length or its matching prefix. It compares
    UTF-8 bytes: given two ``str``, it raises on any non-ASCII character, which turned a Chinese password
    into a 500 for everyone, the right login included.
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
    # Both halves are always compared, so a right username does not answer faster than a wrong one.
    user_ok = secrets.compare_digest(username.encode("utf-8"), settings.web_username.encode("utf-8"))
    password_ok = secrets.compare_digest(password.encode("utf-8"), settings.web_password.encode("utf-8"))
    return user_ok and password_ok


def create_app(
    settings: Settings | None = None, *, profile: DomainProfile | None = None, jobs: JobManager | None = None
) -> FastAPI:
    """The app over one data root under one profile. Without ``profile``, the one ``settings`` selects is loaded
    here, once: the library's keys and every job run under that same value."""
    settings = settings or Settings.from_env()
    if settings.llm_offline:
        # Replay is a proof run over a batch; a server under it would fail every upload, and its misses
        # would pile up in one process-wide record that no job reports.
        raise ConfigError("offline replay is for `run` and `batch`; unset PAPERFACTS_LLM_OFFLINE / llm.offline")
    set_max_in_flight(settings.llm_max_in_flight)
    profile = profile or load_run_profile(settings)
    if not IDENTIFIER.fullmatch(profile.name):
        # The name goes unquoted into a Content-Disposition header; the loader enforces this, a profile built in
        # memory need not have been through it.
        raise ConfigError(f"profile name {profile.name!r} must match {IDENTIFIER.pattern}")
    logger.info("serving profile %s (%s)", profile.name, profile.content_hash[:12])
    library = Library(settings, profile)
    manager = jobs or JobManager(
        pipeline_runner(settings, profile, library), stage_names(), workers=settings.max_parallel_documents
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Jobs run on a worker thread, where signal handlers cannot be installed: arm the runner cleanup
        # here, on the main thread, so a SIGTERM to the server takes the parser subprocesses with it.
        install_runner_cleanup()
        yield
        # Stop accepting jobs, drop the queued ones, and let the running ones finish before the process exits.
        manager.shutdown(wait=True)

    app = FastAPI(title="PaperFacts", version="0.1.0", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.library = library
    app.state.jobs = manager

    @app.middleware("http")
    async def guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """The edge of the app, over every route -- the static frontend and ``/api`` alike.

        The app is published through a tunnel on this machine, so this is all that stands between the
        internet and a library that can upload PDFs and spend LLM tokens. The password gate is off unless a
        password is configured, which is what a laptop wants; a 401 carrying ``WWW-Authenticate`` is what
        makes a browser ask for the login instead of showing the app. The origin check applies either way.
        """
        response = _refusal(request)
        if response is None:
            response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    def _refusal(request: Request) -> Response | None:
        if settings.web_password and not login_accepted(request.headers.get("authorization"), settings):
            return PlainTextResponse(
                "PaperFacts: login required",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="PaperFacts", charset="UTF-8"'},
            )
        if request.method not in SAFE_METHODS and not same_origin(request):
            logger.warning("refused a cross-origin %s %s", request.method, request.url.path)
            return JSONResponse({"detail": "Cross-origin request refused"}, status_code=403)
        return None

    def require_document(document_id: str) -> None:
        # A malformed id is as absent as an unknown one, and the detail says only that: the id convention
        # is not the caller's business, and str(KeyError) would arrive wrapped in a second pair of quotes.
        try:
            exists = library.exists(document_id)
        except KeyError:
            exists = False
        if not exists:
            raise HTTPException(status_code=404, detail=f"No document {document_id}")

    def submit(document_id: str, *, force: bool) -> Job:
        try:
            return manager.submit(document_id, force=force)
        except RuntimeError as exc:  # the manager has shut down: the server is on its way out
            raise HTTPException(status_code=503, detail="The server is shutting down; try again shortly") from exc

    origin = profile_origin(settings, profile)

    @app.get("/api/health")
    def health() -> dict[str, str | bool]:
        try:
            changed = profile_file_changed(profile, origin)
        except OSError:
            changed = True  # a file that cannot be read is no longer the one being served
        return {
            "status": "ok",
            "model": settings.llm_model,
            "profile": profile.name,
            "profile_hash": profile.content_hash[:12],
            "profile_on_disk_changed": changed,
        }

    @app.get("/api/profile")
    def get_profile() -> dict[str, Any]:
        return profile_view(profile)

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
        return Response(
            content=corpus_workbook(datasets, settings, library.profile),
            media_type=EXCEL_MEDIA_TYPE,
            # create_app refused a name outside IDENTIFIER, so it needs no quoting in the header.
            headers={"content-disposition": f'attachment; filename="{profile.name}-corpus.xlsx"'},
        )

    @app.post(
        "/api/documents",
        status_code=202,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {"file": {"type": "string", "format": "binary"}},
                        }
                    }
                },
            }
        },
    )
    async def upload_document(request: Request, force: Annotated[bool, Query()] = False) -> UploadAccepted:
        """One PDF per request. The form is parsed here rather than through a ``File()`` parameter because
        only this call can cap the parts: left to the default, one request may carry a thousand files.

        The size is checked before the body is parsed, because Starlette spools a multipart file to disk
        without any limit. A declared length over the limit is refused unread; the bytes that actually
        arrive are counted too, so a chunked body (which declares nothing) or one that lies is refused the
        moment it passes the limit.
        """
        limit = settings.max_upload_bytes + UPLOAD_OVERHEAD_BYTES
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            raise HTTPException(status_code=413, detail=_too_large(settings.max_upload_bytes))
        async with _capped(request, limit, settings.max_upload_bytes).form(max_files=1, max_fields=0) as form:
            file = form.get("file")
            if not isinstance(file, UploadFile):
                raise HTTPException(status_code=422, detail="Expected one PDF in the multipart field 'file'")
            if (file.size or 0) > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail=_too_large(settings.max_upload_bytes))
            data = await _read_limited(file, settings.max_upload_bytes)
            filename = file.filename or "upload.pdf"
        if not data.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted (missing %PDF header)")
        # writing to disk is sync IO; offload to the threadpool so a multi-hundred-MB write can't stall the event loop
        document = await run_in_threadpool(library.register_upload, filename, data)
        key = document_key(document.document_id)
        job = submit(key, force=force)
        return UploadAccepted(document=library.summary(key), job=job)

    # Registration order matters: FastAPI matches in order, so this literal route must stay above the
    # ``/api/documents/{document_id}`` routes, or "run-all" is read as a document id and answered with 404.
    @app.post("/api/documents/run-all", status_code=202)
    def run_all(force: Annotated[bool, Query()] = False) -> RunAllAccepted:
        """Queue every document that is not finished yet (or every document at all, with force).

        Same submission path as ``run_existing``, once per document in library order: a document
        already queued or running simply gets its existing job back, so pressing the button twice
        costs nothing. Finished means exported under the current keys (:func:`stored.is_finished`).
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
            if not force and library.finished(summary.document_id):
                skipped.append(
                    SkippedDocument(
                        document_id=summary.document_id, name=summary.name, reason="Already processed under these keys"
                    )
                )
                continue
            submitted.append(submit(summary.document_id, force=force))
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
        return submit(document_id, force=force)

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str) -> DocumentSummary:
        require_document(document_id)
        return library.summary(document_id)

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
        """The chart readings, read straight from their own file: they are never part of the dataset.

        Most papers have none, which is a normal state rather than a missing resource: an empty list, not a
        404 that every document page would log as a failed request.
        """
        require_document(document_id)
        identity = library.identity(document_id)
        figures = shown_figures(document_id, identity.name if identity else document_id, settings, library.profile)
        return figures or FiguresView(document_id=document_id, figure_key=library.figure_key, model="")

    @app.get("/api/documents/{document_id}/dataset.xlsx")
    def get_dataset_excel(document_id: str) -> FileResponse:
        require_document(document_id)
        path = library.dataset_excel(document_id)
        if path is None:
            raise HTTPException(status_code=404, detail="No Excel export yet")
        return FileResponse(
            path,
            media_type=EXCEL_MEDIA_TYPE,
            # The upload name is user input; the id is the safe, stable download name.
            filename=f"{profile.name}-{document_id}.xlsx",
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
    def list_all_jobs() -> list[JobBrief]:
        """The jobs this process holds, newest first and without their logs: one small request tells the
        library list which documents are busy. A job's stages and log are on ``/api/jobs/{job_id}``."""
        return manager.briefs()

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


def _capped(request: Request, limit: int, max_upload_bytes: int) -> Request:
    """The same request, whose body raises 413 once more than ``limit`` bytes of it have arrived."""
    received = 0

    async def receive() -> Message:
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(status_code=413, detail=_too_large(max_upload_bytes))
        return message

    return Request(request.scope, receive)


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    """Read the upload body in chunks, raising 413 as soon as the limit is exceeded instead of
    buffering the whole file into memory before checking."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=_too_large(limit))
        chunks.append(chunk)
    return b"".join(chunks)


def _too_large(limit: int) -> str:
    return f"File exceeds {limit // (1024 * 1024)} MB"
