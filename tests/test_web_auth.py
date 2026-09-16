"""The password gate: the only thing between a public tunnel and the LLM budget.

The app is published on the internet here, so the failure that matters is not "the login is annoying" but
"a stranger's PDF upload spends our tokens". These tests pin the gate itself: with a password configured
every route is behind it, and on a laptop without one nothing changes.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paperfacts.config import Settings
from paperfacts.web.app import create_app, login_accepted

USERNAME = "paperfacts"
PASSWORD = "s3cret-pw"


def settings_for(tmp_path: Path, *, password: str | None = PASSWORD) -> Settings:
    return Settings(
        data_root=tmp_path / "data",
        repo_root=tmp_path,
        llm_api_key="sk-test",
        llm_model="fake-model",
        web_username=USERNAME,
        web_password=password,
    )


def basic(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(settings_for(tmp_path))) as test_client:
        yield test_client


def test_a_request_without_credentials_is_refused(client: TestClient):
    response = client.get("/")

    assert response.status_code == 401
    # without this header a browser shows an empty page instead of asking for the password
    assert response.headers["www-authenticate"].startswith("Basic")


def test_the_api_is_behind_the_same_gate(client: TestClient):
    # the frontend is not the only way in: /api can upload, run the pipeline and read every document
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/health").status_code == 401
    assert client.post("/api/documents").status_code == 401


def test_the_configured_login_gets_in(client: TestClient):
    response = client.get("/", headers=basic(USERNAME, PASSWORD))

    assert response.status_code == 200
    assert "PaperFacts" in response.text


def test_the_api_answers_the_configured_login(client: TestClient):
    response = client.get("/api/health", headers=basic(USERNAME, PASSWORD))

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "fake-model"}


@pytest.mark.parametrize(
    ("username", "password"),
    [(USERNAME, "wrong"), (USERNAME, PASSWORD + "x"), ("someone-else", PASSWORD), ("", "")],
)
def test_anything_but_the_exact_login_is_refused(client: TestClient, username: str, password: str):
    assert client.get("/", headers=basic(username, password)).status_code == 401


@pytest.mark.parametrize(
    "header",
    [
        "Bearer sk-test",
        "Basic",
        "Basic !!!not-base64!!!",
        f"Basic {base64.b64encode(b'no-colon-in-here').decode()}",
    ],
)
def test_a_header_that_is_not_a_basic_login_is_refused(client: TestClient, header: str):
    """Each of these used to be a way to crash the check instead of answering it: the app must reject
    them, not return 500."""
    response = client.get("/", headers={"Authorization": header})

    assert response.status_code == 401


def test_a_laptop_without_a_password_is_open(tmp_path: Path):
    with TestClient(create_app(settings_for(tmp_path, password=None))) as open_client:
        assert open_client.get("/").status_code == 200


def test_the_password_is_not_in_the_settings_repr(tmp_path: Path):
    # Settings objects get logged; a password in a log line is a password in a log file
    assert PASSWORD not in repr(settings_for(tmp_path))


def test_login_accepted_accepts_exactly_the_configured_header(tmp_path: Path):
    settings = settings_for(tmp_path)

    assert login_accepted(basic(USERNAME, PASSWORD)["Authorization"], settings)
    assert not login_accepted(None, settings)
    assert not login_accepted(basic(USERNAME, "wrong")["Authorization"], settings)
