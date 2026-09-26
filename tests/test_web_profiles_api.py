"""One server, several profiles: ``?profile=`` on every route whose answer depends on one, the default when it is
absent, the read-only profile routes, and jobs that never run two at once on one document whatever their profiles.
"""

from __future__ import annotations

import dataclasses
import shutil
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paperfacts.config import Settings
from paperfacts.errors import ConfigError
from paperfacts.fields import FieldSpec
from paperfacts.profile_view import prompt_sections
from paperfacts.web.app import create_app, pipeline_runner
from paperfacts.web.jobs import Job, JobManager
from paperfacts.web.registry import ProfileRegistry
from paperfacts.workflow import stage_names
from support.factories import make_blank_pdf
from support.profiles import SHIPPED_PROFILE_PATH
from support.web import DOC_KEY, RecordingRunner, corpus_payload, seed_dataset, wait_for_status, wait_until


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    profiles = tmp_path / "repo" / "profiles"
    profiles.mkdir(parents=True)
    for name in ("tco", "catalysis"):
        shutil.copy(SHIPPED_PROFILE_PATH.parent / f"{name}.json", profiles / f"{name}.json")
    (profiles / "broken.json").write_text("{", encoding="utf-8")
    return tmp_path / "repo"


@pytest.fixture
def settings(repo: Path) -> Settings:
    return Settings(data_root=repo.parent / "data", repo_root=repo, profile="tco", llm_model="fake-model")


@pytest.fixture
def runner() -> RecordingRunner:
    return RecordingRunner()


@pytest.fixture
def client(settings: Settings, runner: RecordingRunner) -> Iterator[TestClient]:
    with TestClient(create_app(settings, jobs=JobManager(runner, stage_names()))) as test_client:
        yield test_client


def _registry(client: TestClient) -> ProfileRegistry:
    return client.app.state.profiles


def _seed_both(client: TestClient) -> None:
    """One document finished under both served profiles, each with its own dataset."""
    registry = _registry(client)
    seed_dataset(registry.get("tco").library, DOC_KEY, corpus_payload(DOC_KEY, name="tco.pdf"))
    seed_dataset(registry.get("catalysis").library, DOC_KEY, corpus_payload(DOC_KEY, name="catalysis.pdf"))


# ---- the profile list and definitions -------------------------------------------------------------------------


def test_the_profile_list_names_the_default_every_served_profile_and_the_invalid_ones(client: TestClient):
    _seed_both(client)

    body = client.get("/api/profiles").json()

    assert body["default"] == "tco"
    assert [entry["name"] for entry in body["profiles"]] == ["tco", "catalysis"]
    catalysis = body["profiles"][1]
    assert catalysis["runnable"] is True and catalysis["not_runnable"] is None
    assert catalysis["on_disk_changed"] is False
    assert catalysis["finished_documents"] == 1
    assert len(catalysis["entities"]) == 2 and all(entity["field_count"] > 0 for entity in catalysis["entities"])
    assert set(catalysis["field_count"]) == {"paper", "sample"}
    assert [(entry["name"], entry["file"]) for entry in body["invalid"]] == [("broken", "broken.json")]
    assert body["invalid"][0]["errors"] and "repo" not in body["invalid"][0]["errors"][0]


def test_a_definition_carries_every_field_attribute(client: TestClient):
    _seed_both(client)

    body = client.get("/api/profiles/catalysis").json()

    assert body["name"] == "catalysis"
    assert set(body["fields"][0]) == {attribute.name for attribute in dataclasses.fields(FieldSpec)}
    assert [entity["name"] for entity in body["entities"]] == ["catalyst", "test"]
    assert body["finished_documents"] == [DOC_KEY]
    assert (body["runnable"], body["on_disk_changed"]) == (True, False)
    assert {"declared", "ignored_suffixes", "known"} <= set(body["units"])
    assert {"condition_keywords", "condition_unit_pattern"} == set(body["retrieval"])


@pytest.mark.parametrize("field", [None, "catalyst"])
def test_the_prompts_are_the_sections_the_cli_prints(client: TestClient, field: str | None):
    query = "" if field is None else f"?field={field}"
    profile = _registry(client).get("catalysis").profile

    body = client.get(f"/api/profiles/catalysis/prompts{query}").json()

    assert body == {"sections": [{"title": title, "text": text} for title, text in prompt_sections(profile, field)]}


def test_an_unknown_field_is_not_found_and_the_detail_lists_the_fields(client: TestClient):
    response = client.get("/api/profiles/tco/prompts?field=colour")

    assert response.status_code == 404
    assert "thickness" in response.json()["detail"]


def test_an_internal_keyerror_for_a_known_field_is_not_swallowed_into_a_404(client: TestClient, monkeypatch):
    """The route checks the field against ``by_name`` itself and 404s only on that; any other ``KeyError`` a
    known field's own rendering raises is a bug, not "no such field", and must not be mistaken for one."""

    def boom(profile, field):
        raise KeyError("unrelated internal error")

    monkeypatch.setattr("paperfacts.web.app.prompt_sections", boom)

    with pytest.raises(KeyError, match="unrelated internal error"):
        client.get("/api/profiles/tco/prompts?field=thickness")


# ---- ?profile= on the per-profile routes ----------------------------------------------------------------------


GET_ROUTES = [
    "/api/profile",
    "/api/documents",
    "/api/dataset",
    f"/api/documents/{DOC_KEY}",
    f"/api/documents/{DOC_KEY}/report",
    f"/api/documents/{DOC_KEY}/extraction/mineru",
    f"/api/documents/{DOC_KEY}/dataset",
    f"/api/documents/{DOC_KEY}/figures",
]
# Every route whose answer depends on a profile: the JSON reads, the downloads and the writes.
PER_PROFILE_ROUTES = [
    *(("GET", route) for route in GET_ROUTES),
    ("GET", "/api/dataset.xlsx"),
    ("GET", f"/api/documents/{DOC_KEY}/dataset.xlsx"),
    ("POST", "/api/documents"),
    ("POST", "/api/documents/run-all"),
    ("POST", f"/api/documents/{DOC_KEY}/run"),
]


def _call(client: TestClient, method: str, route: str, profile: str | None, pdf: bytes):
    params = {} if profile is None else {"profile": profile}
    if method == "GET":
        return client.get(route, params=params)
    files = {"file": ("p.pdf", pdf, "application/pdf")} if route == "/api/documents" else None
    return client.post(route, params=params, files=files)


@pytest.fixture
def pdf(tmp_path: Path) -> bytes:
    return make_blank_pdf(tmp_path / "paper.pdf").read_bytes()


# What each route answers once ``_seed_both`` has planted a dataset alone (no report, extraction or per-document
# workbook): the exact status, not merely "not a 500", so a route that quietly started 500ing on real input
# would still be caught even though it is under 500.
EXPECTED_STATUS: dict[tuple[str, str], int] = {
    ("GET", "/api/profile"): 200,
    ("GET", "/api/documents"): 200,
    ("GET", "/api/dataset"): 200,
    ("GET", f"/api/documents/{DOC_KEY}"): 200,
    ("GET", f"/api/documents/{DOC_KEY}/report"): 404,  # no report seeded
    ("GET", f"/api/documents/{DOC_KEY}/extraction/mineru"): 404,  # no extraction seeded
    ("GET", f"/api/documents/{DOC_KEY}/dataset"): 200,
    ("GET", f"/api/documents/{DOC_KEY}/figures"): 200,  # no charts read is a normal empty view, not a 404
    ("GET", "/api/dataset.xlsx"): 200,
    ("GET", f"/api/documents/{DOC_KEY}/dataset.xlsx"): 404,  # no per-document workbook was ever written
    ("POST", "/api/documents"): 202,
    ("POST", "/api/documents/run-all"): 202,
    ("POST", f"/api/documents/{DOC_KEY}/run"): 409,  # no PDF and no cached parse to rerun from
}


@pytest.mark.parametrize(("method", "route"), PER_PROFILE_ROUTES)
@pytest.mark.parametrize(("profile", "echoed"), [(None, "tco"), ("tco", "tco"), ("catalysis", "catalysis")])
def test_every_per_profile_route_says_which_profile_answered(
    client: TestClient, pdf: bytes, method: str, route: str, profile: str | None, echoed: str
):
    """A caller that forgot ``?profile=`` is answered under the default, and the header says so."""
    _seed_both(client)

    response = _call(client, method, route, profile, pdf)

    assert response.status_code == EXPECTED_STATUS[method, route]
    assert response.headers["X-PaperFacts-Profile"] == echoed


@pytest.mark.parametrize(("method", "route"), PER_PROFILE_ROUTES)
@pytest.mark.parametrize(("profile", "status"), [("nosuch", 404), ("../x", 422), ("TCO", 422), ("broken", 409)])
def test_every_per_profile_route_refuses_a_profile_not_served(
    client: TestClient, runner: RecordingRunner, pdf: bytes, method: str, route: str, profile: str, status: int
):
    _seed_both(client)

    response = _call(client, method, route, profile, pdf)

    assert response.status_code == status
    assert "X-PaperFacts-Profile" not in response.headers
    assert runner.call_count == 0


@pytest.mark.parametrize(("profile", "status"), [("nosuch", 404), ("TCO", 422), ("broken", 409)])
def test_the_profile_routes_refuse_a_profile_not_served(client: TestClient, profile: str, status: int):
    assert client.get(f"/api/profiles/{profile}").status_code == status
    assert client.get(f"/api/profiles/{profile}/prompts").status_code == status


def test_a_profile_free_route_names_no_profile(client: TestClient):
    assert "X-PaperFacts-Profile" not in client.get("/api/jobs").headers
    assert "X-PaperFacts-Profile" not in client.get("/api/health").headers


@pytest.mark.parametrize("route", GET_ROUTES)
def test_no_profile_is_the_default(client: TestClient, route: str):
    _seed_both(client)

    absent = client.get(route)
    explicit = client.get(route, params={"profile": "tco"})

    assert (absent.status_code, absent.json()) == (explicit.status_code, explicit.json())


def test_each_profile_reads_its_own_dataset_and_both_are_done(client: TestClient):
    _seed_both(client)

    default = client.get(f"/api/documents/{DOC_KEY}/dataset").json()
    other = client.get(f"/api/documents/{DOC_KEY}/dataset?profile=catalysis").json()
    summary = client.get(f"/api/documents/{DOC_KEY}?profile=catalysis").json()

    assert (default["filename"], other["filename"]) == ("tco.pdf", "catalysis.pdf")
    assert summary["profiles_done"] == ["tco", "catalysis"]
    assert client.get("/api/documents").json()[0]["profiles_done"] == ["tco", "catalysis"]
    assert client.get("/api/profile?profile=catalysis").json()["name"] == "catalysis"


def test_the_corpus_workbook_is_named_after_the_profile(client: TestClient):
    _seed_both(client)

    response = client.get("/api/dataset.xlsx?profile=catalysis")

    assert response.status_code == 200
    assert 'filename="catalysis-corpus.xlsx"' in response.headers["content-disposition"]


def test_two_profiles_of_identical_content_share_results_but_not_the_workbook(settings: Settings, tco_profile):
    """Content twins share keys (the name is display), so both are done and both read the dataset; the
    per-document workbook is named after the profile a run was made under, so the twin's is missing until a run
    under it writes one."""
    twin = dataclasses.replace(tco_profile, name="tco_twin")
    app = create_app(settings, profile=tco_profile, profiles=(twin,), jobs=JobManager(RecordingRunner(), stage_names()))
    with TestClient(app) as client:
        library = _registry(client).get("tco").library
        seed_dataset(library, DOC_KEY, corpus_payload(DOC_KEY))
        workbook = library.layout.dataset_path(DOC_KEY, "tco")
        workbook.parent.mkdir(parents=True, exist_ok=True)
        workbook.write_bytes(b"xlsx")

        summary = client.get(f"/api/documents/{DOC_KEY}?profile=tco_twin").json()
        dataset = client.get(f"/api/documents/{DOC_KEY}/dataset?profile=tco_twin")
        workbooks = [
            client.get(f"/api/documents/{DOC_KEY}/dataset.xlsx", params={"profile": name}).status_code
            for name in ("tco", "tco_twin")
        ]

    assert summary["profiles_done"] == ["tco", "tco_twin"]
    assert dataset.status_code == 200
    assert workbooks == [200, 404]


def test_a_profile_the_mode_cannot_ask_is_listed_and_described_but_has_no_library(
    repo: Path, runner: RecordingRunner, pdf: bytes
):
    """Its keys cannot be computed under document mode, so every read and write under it is 409; what it is
    stays visible."""
    settings = Settings(
        data_root=repo.parent / "data", repo_root=repo, profile="tco", llm_model="fake", extraction_mode="document"
    )
    with TestClient(create_app(settings, jobs=JobManager(runner, stage_names()))) as client:
        seed_dataset(_registry(client).get("tco").library, DOC_KEY, corpus_payload(DOC_KEY))
        responses = {route: _call(client, method, route, "catalysis", pdf) for method, route in PER_PROFILE_ROUTES}
        listed = client.get("/api/profiles").json()["profiles"][1]
        described = client.get("/api/profiles/catalysis")
        prompts = client.get("/api/profiles/catalysis/prompts")
        view = client.get("/api/profile?profile=catalysis")
        done = client.get(f"/api/documents/{DOC_KEY}").json()["profiles_done"]

    refused = {route: response.status_code for route, response in responses.items() if route != "/api/profile"}
    assert set(refused.values()) == {409}
    detail = responses["/api/documents"].json()["detail"]
    assert "passage" in detail and "catalysis.json" in detail and str(repo) not in detail
    assert (view.status_code, described.status_code) == (200, 200)
    # A profile's prompts are read-only text over its definition; not being runnable is a job-time concern.
    assert prompts.status_code == 200 and prompts.json()["sections"]
    assert described.json()["finished_documents"] == []
    assert (listed["runnable"], listed["not_runnable"] is not None, listed["finished_documents"]) == (False, True, 0)
    assert done == ["tco"]
    assert runner.call_count == 0


# ---- jobs ------------------------------------------------------------------------------------------------------


def test_jobs_dedupe_per_profile_and_never_overlap_on_one_document(settings: Settings, tmp_path: Path):
    active: list[str] = []
    overlap: list[bool] = []
    gate = threading.Event()

    def body(job: Job, mark) -> None:
        active.append(job.profile)
        overlap.append(len(active) > 1)
        if job.profile == "tco":
            gate.wait(timeout=5)
        active.remove(job.profile)

    runner = RecordingRunner(body=body)
    manager = JobManager(runner, stage_names(), workers=2)
    pdf = make_blank_pdf(tmp_path / "paper.pdf").read_bytes()
    with TestClient(create_app(settings, jobs=manager)) as client:
        uploaded = client.post("/api/documents", files={"file": ("p.pdf", pdf, "application/pdf")}).json()
        document_id, first = uploaded["document"]["document_id"], uploaded["job"]
        wait_until(lambda: runner.call_count == 1, what="the tco job to start")
        again = client.post(f"/api/documents/{document_id}/run").json()
        other = client.post(f"/api/documents/{document_id}/run?profile=catalysis").json()
        other_again = client.post(f"/api/documents/{document_id}/run?profile=catalysis").json()
        assert manager.get(other["job_id"]).status == "queued"  # the document is busy under tco
        gate.set()
        wait_for_status(manager, other["job_id"], "done")
        briefs = client.get("/api/jobs").json()

    assert (first["profile"], again["job_id"]) == ("tco", first["job_id"])
    assert (other["profile"], other_again["job_id"]) == ("catalysis", other["job_id"])
    assert other["job_id"] != first["job_id"]
    assert [call.profile for call in runner.calls] == ["tco", "catalysis"]
    assert not any(overlap)
    assert {brief["profile"] for brief in briefs} == {"tco", "catalysis"}


def test_the_runner_runs_each_job_under_its_own_profile_and_checks_its_own_file(
    monkeypatch, settings: Settings, repo: Path, tmp_path: Path
):
    registry = ProfileRegistry.build(settings)
    library = registry.default.library
    document_id = library.register_upload("p.pdf", make_blank_pdf(tmp_path / "p.pdf").read_bytes()).document_id[:16]
    handed: list[str] = []
    monkeypatch.setattr(
        "paperfacts.web.app.run_document", lambda document, s, profile, **kwargs: handed.append(profile.name)
    )
    run = pipeline_runner(settings, registry)

    def job(profile: str) -> Job:
        return Job(job_id="j", document_id=document_id, profile=profile, created_at="2026-01-01T00:00:00+00:00")

    run(job("catalysis"), lambda *args: None)
    catalysis = repo / "profiles" / "catalysis.json"
    catalysis.write_bytes(catalysis.read_bytes() + b"\n")
    with pytest.raises(ConfigError, match="changed on disk"):
        run(job("catalysis"), lambda *args: None)
    run(job("tco"), lambda *args: None)

    assert handed == ["catalysis", "tco"]


def test_the_runner_refuses_a_job_for_a_profile_the_mode_cannot_ask(repo: Path):
    """``served.library`` is ``None`` for a profile the configured mode cannot ask; the job body must refuse
    it with the same reason the browser is shown, and never reach ``run_document``."""
    settings = Settings(
        data_root=repo.parent / "data", repo_root=repo, profile="tco", llm_model="fake", extraction_mode="document"
    )
    registry = ProfileRegistry.build(settings)
    assert registry.get("catalysis").library is None  # the branch under test
    run = pipeline_runner(settings, registry)
    job = Job(job_id="j", document_id="0123456789abcdef", profile="catalysis", created_at="2026-01-01T00:00:00+00:00")

    with pytest.raises(ConfigError, match="passage"):
        run(job, lambda *args: None)
