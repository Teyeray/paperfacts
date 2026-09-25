"""The HTTP edge: what every response carries, which requests are refused before any route runs, and the
shapes the frontend draws progress from.

The app is reachable through a public tunnel with the operator's Basic login cached in their browser, so
a page elsewhere can make that browser send requests here. Those requests are what the origin check is
for; the size check keeps an upload from filling the disk before the route ever sees it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paperfacts.config import Settings
from paperfacts.models import BACKENDS
from paperfacts.web.app import create_app
from paperfacts.web.documents import Library
from paperfacts.web.jobs import JobManager
from paperfacts.workflow import stage_names
from support.factories import make_blank_pdf
from support.web import DOC_KEY, RecordingRunner, seed_artifact, seed_report, wait_for_status

FRAME_HEADERS = {
    "x-frame-options": "DENY",
    "content-security-policy": "frame-ancestors 'none'",
    "x-content-type-options": "nosniff",
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_root=tmp_path / "data", repo_root=tmp_path, llm_api_key="sk-test", llm_model="fake-model")


@pytest.fixture
def library(settings: Settings) -> Library:
    return Library(settings)


@pytest.fixture
def runner() -> RecordingRunner:
    return RecordingRunner()


@pytest.fixture
def jobs(runner: RecordingRunner) -> JobManager:
    return JobManager(runner, stage_names())


@pytest.fixture
def client(settings: Settings, jobs: JobManager) -> Iterator[TestClient]:
    with TestClient(create_app(settings, jobs=jobs)) as test_client:
        yield test_client


@pytest.fixture
def pdf_bytes(tmp_path: Path) -> bytes:
    return make_blank_pdf(tmp_path / "upload.pdf").read_bytes()


def pdf_part(data: bytes, name: str = "paper.pdf") -> tuple[str, bytes, str]:
    return (name, data, "application/pdf")


# ---- headers on every response ------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/app.js", "/api/health", "/api/documents", "/nope.js"])
def test_every_response_refuses_to_be_framed_or_sniffed(client: TestClient, path: str):
    headers = client.get(path).headers

    for name, value in FRAME_HEADERS.items():
        assert headers[name] == value, (path, name)


def test_the_login_prompt_carries_the_same_headers(settings: Settings):
    with TestClient(create_app(dataclasses.replace(settings, web_password="pw"))) as client:
        response = client.get("/")

    assert response.status_code == 401
    assert response.headers["x-frame-options"] == "DENY"


# ---- the origin check ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
        {"Referer": "https://evil.example/page"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site", "Origin": "http://testserver"},
    ],
)
def test_a_cross_origin_request_that_changes_something_is_refused(
    client: TestClient, runner: RecordingRunner, headers: dict[str, str]
):
    response = client.post("/api/documents/run-all?force=true", headers=headers)

    assert response.status_code == 403
    assert response.json()["detail"] == "Cross-origin request refused"


def test_a_cross_origin_upload_never_reaches_the_library(client: TestClient, library: Library, pdf_bytes: bytes):
    response = client.post(
        "/api/documents", files={"file": pdf_part(pdf_bytes)}, headers={"Origin": "https://evil.example"}
    )

    assert response.status_code == 403
    assert library.list() == []


@pytest.mark.parametrize(
    "headers",
    [
        {},  # not a browser: curl, a script, the tests
        {"Origin": "http://testserver"},
        {"Origin": "https://testserver", "Sec-Fetch-Site": "same-origin"},  # https at the tunnel, http here
        {"Referer": "http://testserver/#/doc/0123456789abcdef"},
        {"Origin": "https://paperfacts.example.org", "X-Forwarded-Host": "paperfacts.example.org"},
    ],
)
def test_a_same_origin_request_goes_through(client: TestClient, headers: dict[str, str]):
    assert client.post("/api/documents/run-all", headers=headers).status_code == 202


def test_reading_is_never_refused_for_its_origin(client: TestClient):
    # a cross-origin read cannot see the answer anyway, and images and links are GETs
    assert client.get("/api/documents", headers={"Origin": "https://evil.example"}).status_code == 200


# ---- upload limits, before the body is read ----------------------------------------------------------


def test_an_upload_declaring_more_than_the_limit_is_refused_unread(settings: Settings, pdf_bytes: bytes):
    small = dataclasses.replace(settings, max_upload_bytes=1024)
    with TestClient(create_app(small)) as client:
        response = client.post(
            "/api/documents",
            content=b"x" * 16,
            headers={"Content-Type": "multipart/form-data; boundary=x", "Content-Length": str(10 * 1024 * 1024)},
        )

    assert response.status_code == 413
    assert "MB" in response.json()["detail"]


def test_an_upload_without_a_declared_length_is_refused(client: TestClient):
    def chunks() -> Iterator[bytes]:
        yield b"--x\r\n"

    response = client.post(
        "/api/documents", content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=x"}
    )

    assert response.status_code == 411


def test_one_upload_carries_one_file(client: TestClient, library: Library, pdf_bytes: bytes):
    response = client.post(
        "/api/documents", files=[("file", pdf_part(pdf_bytes)), ("file", pdf_part(pdf_bytes, "again.pdf"))]
    )

    assert response.status_code == 400
    assert library.list() == []


def test_an_upload_with_the_file_under_another_name_is_a_validation_error(client: TestClient, pdf_bytes: bytes):
    response = client.post("/api/documents", files={"document": pdf_part(pdf_bytes)})

    # any other part is refused by the parser before the route could look for "file"
    assert response.status_code in (400, 422)


def test_the_upload_route_still_documents_its_file_field(client: TestClient):
    schema = client.get("/openapi.json").json()

    content = schema["paths"]["/api/documents"]["post"]["requestBody"]["content"]
    assert "file" in content["multipart/form-data"]["schema"]["properties"]


# ---- error bodies ------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/api/documents/zz", "/api/documents/zz/report", "/api/documents/zz/figures"])
def test_a_malformed_id_is_simply_not_found(client: TestClient, path: str):
    response = client.get(path)

    assert response.status_code == 404
    # not str(KeyError): no second pair of quotes, and nothing about what a valid id looks like
    assert response.json()["detail"] == "No document zz"


def test_submitting_after_shutdown_is_unavailable_not_a_crash(settings: Settings, jobs: JobManager):
    seed_artifact(Library(settings), "mineru")
    seed_artifact(Library(settings), "paddleocr_vl")
    with TestClient(create_app(settings, jobs=jobs)) as client:
        jobs.shutdown()

        response = client.post(f"/api/documents/{DOC_KEY}/run")

    assert response.status_code == 503


# ---- the jobs list ---------------------------------------------------------------------------------


def test_the_list_of_every_job_leaves_the_logs_out(client: TestClient, jobs: JobManager, pdf_bytes: bytes):
    job_id = client.post("/api/documents", files={"file": pdf_part(pdf_bytes)}).json()["job"]["job_id"]
    wait_for_status(jobs, job_id, "done")

    [brief] = client.get("/api/jobs").json()
    full = client.get(f"/api/jobs/{job_id}").json()

    assert brief["job_id"] == job_id and brief["status"] == "done"
    assert "log" not in brief and "stages" not in brief
    assert "log" in full and [stage["name"] for stage in full["stages"]] == list(stage_names())


# ---- progress on the summary ------------------------------------------------------------------------


def test_the_summary_lists_every_stage_in_pipeline_order(client: TestClient, library: Library):
    seed_artifact(library, "mineru")
    seed_report(library)

    stages = client.get(f"/api/documents/{DOC_KEY}").json()["stages"]

    assert [stage["name"] for stage in stages] == list(stage_names())
    status = {stage["name"]: stage["status"] for stage in stages}
    assert status["parse:mineru"] == "done" and status["parse:paddleocr_vl"] == "pending"
    assert status["compare"] == "done" and status["export"] == "pending"
    # figures is opt-in and off here: never read is a skip, not work left to do
    assert status["figures"] == "skipped"


def test_figures_switched_on_but_not_read_is_pending(settings: Settings, library: Library):
    seed_artifact(library, "mineru")

    summary = Library(dataclasses.replace(settings, figures_enabled=True)).summary(DOC_KEY)

    assert {stage.name: stage.status for stage in summary.stages}["figures"] == "pending"


def test_a_document_with_one_parse_and_no_pdf_cannot_be_rerun(client: TestClient, library: Library):
    seed_artifact(library, "mineru")

    assert client.get(f"/api/documents/{DOC_KEY}").json()["runnable"] is False


def test_a_document_parsed_elsewhere_can_be_rerun_without_its_pdf(client: TestClient, library: Library):
    for backend in BACKENDS:
        seed_artifact(library, backend)

    body = client.get(f"/api/documents/{DOC_KEY}").json()

    assert body["pdf_available"] is False
    assert body["runnable"] is True


def test_an_uploaded_document_can_be_rerun(client: TestClient, pdf_bytes: bytes):
    document = client.post("/api/documents", files={"file": pdf_part(pdf_bytes)}).json()["document"]

    assert document["runnable"] is True
    assert [stage["name"] for stage in document["stages"]] == list(stage_names())
