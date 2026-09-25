"""FastAPI application: the document library, upload-then-process, artifact and page-image reads,
and the static frontend.

This layer is the HTTP <-> :mod:`paperfacts.web.documents` / :mod:`paperfacts.web.jobs` mapping,
so the tests here watch "is the mapping right", not the business outcome:

- **Status codes are the contract**: the frontend decides what to show the user based on
  404 / 409 / 413 / 422 — if everything came back 500, the page would only ever say "something
  went wrong";
- **No real parser / LLM call ever happens**: the job body is swapped for
  :class:`support.web.RecordingRunner`; ``pipeline_runner`` just hands the job to
  ``workflow.run_document``, and here we only verify what it handed over — the pipeline's own
  behavior is covered in ``test_workflow_run.py``.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paperfacts.compare import ComparisonCounts, ComparisonReport
from paperfacts.config import Settings
from paperfacts.figures import FigureReading, FigureReadings
from paperfacts.keys import figure_key_for
from paperfacts.models import BACKENDS, Backend, NormalizedBBox, ParsedArtifact
from paperfacts.records import LaneExtraction
from paperfacts.web.app import create_app, pipeline_runner
from paperfacts.web.documents import Library
from paperfacts.web.jobs import Job, JobManager
from paperfacts.workflow import stage_names
from support.extraction import make_field, make_sample
from support.factories import make_blank_pdf
from support.web import (
    DOC_KEY,
    DOC_SHA,
    RecordingRunner,
    seed_artifact,
    seed_extraction,
    seed_report,
    wait_for_status,
    wait_until,
)

UNKNOWN_ID = "0123456789abcdef"
MALFORMED_ID = "not-a-document"


# ---- fixtures --------------------------------------------------------------------------


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


def upload(client: TestClient, data: bytes, *, name: str = "paper.pdf", query: str = "") -> dict:
    response = client.post(f"/api/documents{query}", files={"file": (name, data, "application/pdf")})
    assert response.status_code == 202, response.text
    return response.json()


@pytest.fixture
def uploaded(client: TestClient, pdf_bytes: bytes) -> str:
    """Upload a real (blank) PDF and return its 16-character document_id."""
    return upload(client, pdf_bytes)["document"]["document_id"]


@pytest.fixture
def parsed_only(library: Library) -> str:
    """A document processed via the CLI whose original PDF is no longer on this machine: it has
    artifacts but can't be reprocessed."""
    seed_artifact(library, "mineru")
    return DOC_KEY


@pytest.fixture
def parsed_both_lanes(library: Library) -> str:
    """A document processed on another machine: no PDF here, but both lanes' artifacts are stored, so
    extraction, comparison and export can still run."""
    for backend in BACKENDS:
        seed_artifact(library, backend)
    return DOC_KEY


# ---- health / listing ---------------------------------------------------------------------


def test_health_reports_the_model_that_will_be_used(client: TestClient):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "fake-model"}


def test_the_document_list_is_empty_before_anything_is_uploaded(client: TestClient):
    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == []


def test_the_document_list_shows_what_has_been_uploaded(client: TestClient, uploaded: str):
    body = client.get("/api/documents").json()

    assert [row["document_id"] for row in body] == [uploaded]
    assert body[0]["name"] == "paper.pdf"
    assert body[0]["pdf_available"] is True


# ---- upload ------------------------------------------------------------------------------


def test_a_file_that_is_not_a_pdf_is_rejected(client: TestClient):
    response = client.post("/api/documents", files={"file": ("notes.txt", b"just text", "application/pdf")})

    assert response.status_code == 400
    assert "%PDF" in response.json()["detail"]


def test_a_file_larger_than_the_limit_is_rejected(settings: Settings, pdf_bytes: bytes):
    # The limit is configuration, so the test sets it the way a deployment would rather than patching.
    with TestClient(create_app(dataclasses.replace(settings, max_upload_bytes=16))) as client:
        response = client.post("/api/documents", files={"file": ("big.pdf", pdf_bytes, "application/pdf")})

    assert response.status_code == 413


def test_a_request_without_a_file_is_a_validation_error(client: TestClient):
    assert client.post("/api/documents").status_code == 422


def test_an_upload_is_stored_and_queued(
    client: TestClient, library: Library, pdf_bytes: bytes, runner: RecordingRunner
):
    body = upload(client, pdf_bytes)

    document_id = body["document"]["document_id"]
    assert len(document_id) == 16  # the public id is the directory name, not the 64-char sha
    assert library.layout.source_pdf(document_id).read_bytes() == pdf_bytes
    assert body["job"]["document_id"] == document_id
    assert body["job"]["job_id"]
    assert [stage["name"] for stage in body["job"]["stages"]] == list(stage_names())
    wait_until(lambda: runner.call_count == 1, what="the job to run")
    assert runner.document_ids == [document_id]
    assert runner.forces == [False]


def test_the_force_flag_reaches_the_job(client: TestClient, pdf_bytes: bytes, runner: RecordingRunner):
    upload(client, pdf_bytes, query="?force=true")

    wait_until(lambda: runner.call_count == 1, what="the job to run")
    assert runner.forces == [True]


def test_uploading_the_same_pdf_twice_reuses_the_document(client: TestClient, jobs: JobManager, pdf_bytes: bytes):
    first = upload(client, pdf_bytes, name="paper.pdf")
    wait_for_status(jobs, first["job"]["job_id"], "done")

    second = upload(client, pdf_bytes, name="paper-copy.pdf")

    assert second["document"]["document_id"] == first["document"]["document_id"]
    assert second["job"]["job_id"] != first["job"]["job_id"]
    assert len(client.get("/api/documents").json()) == 1


def test_uploading_again_while_the_first_run_is_still_active_joins_that_run(settings: Settings, pdf_bytes: bytes):
    # the user impatiently drags the same PDF in a second time: don't queue a second job, return the one already running
    gate = threading.Event()
    runner = RecordingRunner(gate=gate)
    manager = JobManager(runner, stage_names())
    try:
        with TestClient(create_app(settings, jobs=manager)) as client:
            first = upload(client, pdf_bytes)
            assert runner.entered.wait(timeout=2.0)

            second = upload(client, pdf_bytes)

            assert second["job"]["job_id"] == first["job"]["job_id"]
            gate.set()  # leaving the app waits for the running job, so it must be able to finish
    finally:
        gate.set()
    wait_for_status(manager, first["job"]["job_id"], "done")
    assert runner.call_count == 1


# ---- reprocessing ---------------------------------------------------------------------------


def test_rerunning_an_unknown_document_is_not_found(client: TestClient):
    assert client.post(f"/api/documents/{UNKNOWN_ID}/run").status_code == 404


def test_rerunning_a_malformed_id_is_not_found(client: TestClient):
    assert client.post(f"/api/documents/{MALFORMED_ID}/run").status_code == 404


def test_rerunning_a_document_whose_pdf_is_gone_is_a_conflict(client: TestClient, parsed_only: str):
    """Artifacts exist but the PDF doesn't (e.g. the CLI ran on another machine): that's a 409, not a 404.

    A 404 would make the frontend think the document doesn't exist and drop it from the list, even
    though the artifacts are still viewable.
    """
    response = client.post(f"/api/documents/{parsed_only}/run")

    assert response.status_code == 409
    assert "re-upload" in response.json()["detail"]


def test_rerunning_a_document_with_both_parses_and_no_pdf_is_accepted(
    client: TestClient, parsed_both_lanes: str, runner: RecordingRunner
):
    """Nothing after parsing reads the PDF, so a stored parse for both lanes is enough to re-run."""
    response = client.post(f"/api/documents/{parsed_both_lanes}/run")

    assert response.status_code == 202
    assert response.json()["document_id"] == parsed_both_lanes
    wait_until(lambda: runner.call_count == 1, what="the rerun job to start")


def test_rerunning_a_document_queues_a_new_job(client: TestClient, uploaded: str, runner: RecordingRunner):
    response = client.post(f"/api/documents/{uploaded}/run?force=true")

    assert response.status_code == 202
    assert response.json()["document_id"] == uploaded
    wait_until(lambda: runner.call_count == 2, what="one job each for upload and rerun")
    assert runner.forces == [False, True]


# ---- single document summary and artifacts -----------------------------------------------------


def test_an_unknown_document_is_not_found(client: TestClient):
    assert client.get(f"/api/documents/{UNKNOWN_ID}").status_code == 404


def test_a_malformed_document_id_is_not_found(client: TestClient):
    # a KeyError (malformed id shape) must also become a 404, not leak out as a 500
    response = client.get(f"/api/documents/{MALFORMED_ID}")

    assert response.status_code == 404
    assert MALFORMED_ID in response.json()["detail"]


def test_a_known_document_returns_its_summary(client: TestClient, uploaded: str):
    body = client.get(f"/api/documents/{uploaded}").json()

    assert body["document_id"] == uploaded
    assert body["name"] == "paper.pdf"
    assert body["parsed"] == {"mineru": False, "paddleocr_vl": False}


def test_the_report_is_not_found_before_the_comparison_ran(client: TestClient, parsed_only: str):
    response = client.get(f"/api/documents/{parsed_only}/report")

    assert response.status_code == 404
    assert "comparison report" in response.json()["detail"]


def test_the_report_is_returned_once_it_is_on_disk(client: TestClient, library: Library, parsed_only: str):
    seed_report(library, counts=ComparisonCounts(agree=5, conflict=1, total=6))

    body = client.get(f"/api/documents/{parsed_only}/report").json()

    assert ComparisonReport.model_validate(body).counts.agree == 5


def test_the_report_of_an_unknown_document_is_not_found(client: TestClient):
    assert client.get(f"/api/documents/{UNKNOWN_ID}/report").status_code == 404


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_extraction_is_not_found_before_it_ran(client: TestClient, parsed_only: str, backend: Backend):
    response = client.get(f"/api/documents/{parsed_only}/extraction/{backend}")

    assert response.status_code == 404
    assert backend in response.json()["detail"]


def test_the_extraction_is_returned_normalised(client: TestClient, library: Library, parsed_only: str):
    seed_extraction(
        library, "mineru", samples=[make_sample("A", [make_field("sheet_resistance", "12.5", unit_raw="Ω/sq")])]
    )

    body = client.get(f"/api/documents/{parsed_only}/extraction/mineru").json()

    lane = LaneExtraction.model_validate(body)
    assert lane.sample("A").get("sheet_resistance").value == 12.5


def test_an_unknown_backend_is_a_validation_error(client: TestClient, parsed_only: str):
    assert client.get(f"/api/documents/{parsed_only}/extraction/tesseract").status_code == 422
    assert client.get(f"/api/documents/{parsed_only}/artifact/tesseract").status_code == 422


def test_the_artifact_is_not_found_before_parsing(client: TestClient, parsed_only: str):
    response = client.get(f"/api/documents/{parsed_only}/artifact/paddleocr_vl")

    assert response.status_code == 404
    assert "parsed artifact" in response.json()["detail"]


def test_the_artifact_carries_the_blocks_and_the_page_geometry(client: TestClient, parsed_only: str):
    body = client.get(f"/api/documents/{parsed_only}/artifact/mineru").json()

    artifact = ParsedArtifact.model_validate(body)
    assert artifact.backend == "mineru"
    assert artifact.blocks and artifact.pages
    assert artifact.blocks[0].bbox.x1 >= 0.0  # the normalized bbox used for provenance goes to the frontend as-is


# ---- the consolidated dataset ------------------------------------------------------------------


def seed_dataset(library: Library, document_id: str, payload: dict) -> Path:
    """Write the consolidated table under the library's current keys, the way the export stage does."""
    path = library.layout.dataset_json_path(document_id, library.extractor_key, library.comparison_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_the_dataset_is_not_found_before_the_export_ran(client: TestClient, parsed_only: str):
    response = client.get(f"/api/documents/{parsed_only}/dataset")

    assert response.status_code == 404
    assert "dataset" in response.json()["detail"]


def test_the_dataset_of_an_unknown_document_is_not_found(client: TestClient):
    assert client.get(f"/api/documents/{UNKNOWN_ID}/dataset").status_code == 404


def test_the_dataset_is_returned_once_it_is_on_disk(client: TestClient, library: Library, parsed_only: str):
    payload = {"document_id": parsed_only, "sample_rows": [{"sample_id": "A", "thickness": 300}]}
    seed_dataset(library, parsed_only, payload)

    body = client.get(f"/api/documents/{parsed_only}/dataset").json()

    # The endpoint publishes the full DatasetPayload: what was written is there, the rest at its default.
    assert body["document_id"] == parsed_only
    assert body["sample_rows"] == payload["sample_rows"]
    assert body["paper_row"] == {}
    assert body["quality_rows"] == []
    assert body["fields"] == []


def seed_figures(library: Library, settings: Settings) -> None:
    key = figure_key_for(settings)
    reading = FigureReading(
        source_id="mineru_p0_b9",
        page=0,
        bbox=NormalizedBBox(x1=0.1, y1=0.1, x2=0.5, y2=0.4),
        figure="Fig. 3",
        caption="Fig. 3 Sheet resistance",
        panel=1,
        field="sheet_resistance",
        y_raw=25.0,
        y_unit_raw="Ω/sq",
        y=25.0,
        unit="Ω/sq",
        precision=0.1,
    )
    FigureReadings(
        document_id=DOC_SHA, figure_key=key, model="qwen3.7-plus", backend="mineru", readings=(reading,)
    ).write(library.layout.figures_path(DOC_SHA, key))


def test_the_figure_readings_are_not_found_before_the_stage_ran(client: TestClient, parsed_only: str):
    assert client.get(f"/api/documents/{parsed_only}/figures").status_code == 404


def test_the_figure_readings_are_served_from_their_own_file(
    client: TestClient, library: Library, settings: Settings, parsed_only: str
):
    seed_figures(library, settings)

    body = client.get(f"/api/documents/{parsed_only}/figures").json()

    assert body["stale"] is False
    [row] = body["rows"]
    assert row["value"] == 25.0 and row["precision"] == "±10%" and row["figure"] == "Fig. 3"
    # The seeded parse has no block mineru_p0_b9, so the reading cannot be located on the page.
    assert body["orphaned"] == ["mineru_p0_b9"]


def test_a_dataset_written_under_other_keys_is_not_served(client: TestClient, library: Library, parsed_only: str):
    # The same rule as the comparison report: a table built by a different model or field table is stale.
    stale = library.layout.dataset_json_path(parsed_only, "other-extractor", "other-comparison")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("{}", encoding="utf-8")

    assert client.get(f"/api/documents/{parsed_only}/dataset").status_code == 404


def test_the_excel_export_is_not_found_before_it_is_written(client: TestClient, parsed_only: str):
    response = client.get(f"/api/documents/{parsed_only}/dataset.xlsx")

    assert response.status_code == 404
    assert "Excel" in response.json()["detail"]


def test_the_excel_export_is_served_as_a_download(client: TestClient, library: Library, parsed_only: str):
    path = library.layout.dataset_path(parsed_only)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK\x03\x04 workbook")

    response = client.get(f"/api/documents/{parsed_only}/dataset.xlsx")

    assert response.status_code == 200
    assert response.content == b"PK\x03\x04 workbook"
    assert response.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert "attachment" in response.headers["content-disposition"]
    assert f'filename="paperfacts-{parsed_only}.xlsx"' in response.headers["content-disposition"]


# ---- the corpus table --------------------------------------------------------------------------


def corpus_payload(document_id: str, *, name: str = "paper.pdf", samples: int = 2) -> dict:
    """A dataset payload shaped like ``DatasetPayload``, trimmed to what the corpus reads."""
    return {
        "document_id": document_id,
        "filename": name,
        "fields": [{"name": "thickness", "label": "厚度", "unit": "nm", "scope": "sample", "description": "膜厚"}],
        "paper_row": {"sample_id": "S1", "available_fields": 1, "agree_fields": 1, "thickness": 300},
        "sample_rows": [{"sample_id": f"S{i}", "thickness": 300} for i in range(1, samples + 1)],
        "quality_rows": [],
    }


def test_the_corpus_is_empty_until_a_document_has_a_dataset(client: TestClient, parsed_only: str):
    body = client.get("/api/dataset").json()

    assert body == {"fields": [], "rows": []}


def test_the_corpus_carries_one_row_per_document_with_a_dataset(
    client: TestClient, library: Library, parsed_only: str, uploaded: str
):
    # `uploaded` is a second document, deliberately left without a dataset: it must simply be absent.
    seed_dataset(library, parsed_only, corpus_payload(parsed_only))

    body = client.get("/api/dataset").json()

    assert [row["document_id"] for row in body["rows"]] == [parsed_only]
    assert body["fields"] == [
        {"name": "thickness", "label": "厚度", "unit": "nm", "scope": "sample", "description": "膜厚"}
    ]
    row = body["rows"][0]
    assert row["paper_row"]["thickness"] == 300
    assert row["sample_count"] == 2
    # Every sample travels with the paper, so the home table can expand it without another request.
    assert [sample["sample_id"] for sample in row["sample_rows"]] == ["S1", "S2"]
    assert row["name"]
    assert "fields" not in row  # the field list travels once, at the top level


def test_a_dataset_under_a_stale_key_is_left_out_of_the_corpus(client: TestClient, library: Library, parsed_only: str):
    stale = library.layout.dataset_json_path(parsed_only, "other-extractor", "other-comparison")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps(corpus_payload(parsed_only)), encoding="utf-8")

    assert client.get("/api/dataset").json() == {"fields": [], "rows": []}


def test_a_stale_dataset_does_not_displace_the_current_one(client: TestClient, library: Library, parsed_only: str):
    # Both keys present at once: the row must come from the current key, not from whichever file sorts first.
    stale = library.layout.dataset_json_path(parsed_only, "other-extractor", "other-comparison")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps(corpus_payload(parsed_only, samples=9)), encoding="utf-8")
    seed_dataset(library, parsed_only, corpus_payload(parsed_only, samples=2))

    rows = client.get("/api/dataset").json()["rows"]

    assert [row["sample_count"] for row in rows] == [2]


def test_only_the_current_key_document_appears_when_another_is_stale(
    client: TestClient, library: Library, parsed_only: str, uploaded: str
):
    # Two different documents, one mined under the current keys and one left behind by an older run:
    # the corpus is the current table, so only the first is a row.
    stale = library.layout.dataset_json_path(parsed_only, "other-extractor", "other-comparison")
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps(corpus_payload(parsed_only)), encoding="utf-8")
    seed_dataset(library, uploaded, corpus_payload(uploaded))

    rows = client.get("/api/dataset").json()["rows"]

    assert [row["document_id"] for row in rows] == [uploaded]


def test_a_corrupt_dataset_json_is_skipped_in_the_corpus(
    client: TestClient, library: Library, parsed_only: str, uploaded: str
):
    # One unusable file costs exactly its own row: the corpus is library-wide, so it must not 500 as a whole.
    seed_dataset(library, uploaded, corpus_payload(uploaded))
    corrupt = library.layout.dataset_json_path(parsed_only, library.extractor_key, library.comparison_key)
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{not json", encoding="utf-8")

    body = client.get("/api/dataset").json()

    assert [row["document_id"] for row in body["rows"]] == [uploaded]
    assert client.get("/api/dataset.xlsx").status_code == 200


def test_a_dataset_in_the_wrong_shape_is_skipped_in_the_corpus(client: TestClient, library: Library, parsed_only: str):
    # Valid JSON, wrong shape: still one dropped row rather than a broken endpoint.
    path = library.layout.dataset_json_path(parsed_only, library.extractor_key, library.comparison_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert client.get("/api/dataset").json() == {"fields": [], "rows": []}
    assert client.get("/api/dataset.xlsx").status_code == 404


def test_the_dataset_endpoints_publish_their_schemas(client: TestClient):
    """The dataset crosses the HTTP boundary as a declared model, so the browser's contract is in the
    schema rather than only in the code that happens to build the dict."""
    schema = client.get("/openapi.json").json()

    assert {"DatasetPayload", "FieldColumn", "CorpusPayload", "CorpusRow"} <= set(schema["components"]["schemas"])

    def response_ref(path: str) -> str:
        return schema["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]

    assert response_ref("/api/dataset").endswith("/CorpusPayload")
    assert response_ref("/api/documents/{document_id}/dataset").endswith("/DatasetPayload")


def test_the_corpus_workbook_is_not_found_while_nothing_is_mined(client: TestClient, parsed_only: str):
    assert client.get("/api/dataset.xlsx").status_code == 404


def test_the_corpus_workbook_is_rebuilt_from_the_datasets(client: TestClient, library: Library, parsed_only: str):
    seed_dataset(library, parsed_only, corpus_payload(parsed_only))

    response = client.get("/api/dataset.xlsx")

    assert response.status_code == 200
    assert response.content.startswith(b"PK")  # a real xlsx (a zip), built on demand rather than read from disk
    assert response.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert 'filename="paperfacts-corpus.xlsx"' in response.headers["content-disposition"]


# ---- page images -----------------------------------------------------------------------------


def test_a_page_is_rendered_as_png(client: TestClient, uploaded: str):
    response = client.get(f"/api/documents/{uploaded}/pages/0.png?dpi=50")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_the_second_request_serves_the_file_that_is_already_on_disk(
    client: TestClient, library: Library, uploaded: str
):
    """Page-image rendering is the slowest operation in this layer; a caching bug would only show
    up as "paging feels a bit slow"."""
    client.get(f"/api/documents/{uploaded}/pages/1.png?dpi=50")
    cached = library.layout.page_cache_dir(uploaded, 50) / "page_001.png"
    assert cached.is_file()
    before = cached.stat().st_mtime_ns

    second = client.get(f"/api/documents/{uploaded}/pages/1.png?dpi=50")

    assert second.status_code == 200
    assert cached.stat().st_mtime_ns == before


@pytest.mark.parametrize("dpi", [1, Settings().page_dpi_min - 1, Settings().page_dpi_max + 1, 10_000])
def test_a_dpi_outside_the_sane_range_is_a_validation_error(client: TestClient, uploaded: str, dpi: int):
    # dpi comes straight from the URL: without a bound, dpi=10000 is a request that can take down the server
    assert client.get(f"/api/documents/{uploaded}/pages/0.png?dpi={dpi}").status_code == 422


@pytest.mark.parametrize("dpi", [Settings().page_dpi_min, Settings().page_dpi_max])
def test_the_dpi_bounds_themselves_are_allowed(client: TestClient, library: Library, uploaded: str, dpi: int):
    response = client.get(f"/api/documents/{uploaded}/pages/0.png?dpi={dpi}")

    assert response.status_code == 200
    assert (library.layout.page_cache_dir(uploaded, dpi) / "page_000.png").is_file()


@pytest.mark.parametrize("page", [2, 99, -1])
def test_a_page_outside_the_document_is_not_found(client: TestClient, uploaded: str, page: int):
    response = client.get(f"/api/documents/{uploaded}/pages/{page}.png?dpi=50")

    assert response.status_code == 404
    assert str(page) in response.json()["detail"]


def test_a_page_number_that_is_not_a_number_is_a_validation_error(client: TestClient, uploaded: str):
    assert client.get(f"/api/documents/{uploaded}/pages/first.png").status_code == 422


def test_pages_cannot_be_rendered_without_a_pdf(client: TestClient, parsed_only: str):
    response = client.get(f"/api/documents/{parsed_only}/pages/0.png")

    assert response.status_code == 404
    assert "PDF" in response.json()["detail"]


def test_pages_of_an_unknown_document_are_not_found(client: TestClient):
    assert client.get(f"/api/documents/{UNKNOWN_ID}/pages/0.png").status_code == 404


# ---- jobs -------------------------------------------------------------------------------


def test_a_documents_jobs_are_listed(client: TestClient, jobs: JobManager, uploaded: str):
    wait_for_status(
        jobs, jobs.for_document(uploaded)[0].job_id, "done"
    )  # run would merge into this job while it's still active
    client.post(f"/api/documents/{uploaded}/run")

    body = client.get(f"/api/documents/{uploaded}/jobs").json()

    assert len(body) == 2
    assert {row["document_id"] for row in body} == {uploaded}


def test_the_jobs_of_an_unknown_document_are_not_found(client: TestClient):
    assert client.get(f"/api/documents/{UNKNOWN_ID}/jobs").status_code == 404


def test_a_job_can_be_polled_by_id(client: TestClient, uploaded: str, runner: RecordingRunner):
    job_id = client.get(f"/api/documents/{uploaded}/jobs").json()[0]["job_id"]
    wait_until(lambda: runner.call_count == 1, what="the job to run")

    body = client.get(f"/api/jobs/{job_id}").json()

    assert body["job_id"] == job_id
    assert body["document_id"] == uploaded


def test_an_unknown_job_is_not_found(client: TestClient):
    response = client.get("/api/jobs/deadbeef")

    assert response.status_code == 404
    assert "deadbeef" in response.json()["detail"]


# ---- running the whole library ----------------------------------------------------------------


@pytest.fixture
def second_pdf_bytes(tmp_path: Path) -> bytes:
    """A second, different PDF: a different page size is enough to give it another sha256."""
    return make_blank_pdf(tmp_path / "other.pdf", [(200.0, 300.0)]).read_bytes()


@pytest.fixture
def two_idle_documents(
    client: TestClient, jobs: JobManager, runner: RecordingRunner, pdf_bytes: bytes, second_pdf_bytes: bytes
) -> list[str]:
    """Two uploaded documents whose upload jobs have finished, so a bulk run starts from a quiet
    manager rather than merging into whatever is still active."""
    ids = [
        upload(client, pdf_bytes)["document"]["document_id"],
        upload(client, second_pdf_bytes, name="other.pdf")["document"]["document_id"],
    ]
    for document_id in ids:
        wait_for_status(jobs, jobs.for_document(document_id)[0].job_id, "done", "failed")
    return ids


def mark_compared(library: Library, document_id: str) -> None:
    identity = library.identity(document_id)
    assert identity is not None
    seed_report(library, document_sha=identity.sha256)


def test_running_everything_skips_what_is_already_compared(
    client: TestClient, library: Library, two_idle_documents: list[str]
):
    done, todo = two_idle_documents
    mark_compared(library, done)

    response = client.post("/api/documents/run-all")

    assert response.status_code == 202
    body = response.json()
    assert [job["document_id"] for job in body["submitted"]] == [todo]
    assert [row["document_id"] for row in body["skipped"]] == [done]
    assert body["skipped"][0]["reason"]


def test_forcing_a_run_of_everything_queues_the_finished_document_too(
    client: TestClient, library: Library, two_idle_documents: list[str]
):
    mark_compared(library, two_idle_documents[0])

    body = client.post("/api/documents/run-all?force=true").json()

    assert {job["document_id"] for job in body["submitted"]} == set(two_idle_documents)
    assert body["skipped"] == []
    assert all(job["force"] for job in body["submitted"])


def test_a_document_without_a_pdf_is_skipped_with_a_reason(client: TestClient, parsed_only: str):
    """The same situation ``run`` answers with a 409: a bulk run cannot fail over one such document,
    so it reports it instead."""
    body = client.post("/api/documents/run-all").json()

    assert body["submitted"] == []
    assert [row["document_id"] for row in body["skipped"]] == [parsed_only]
    assert "PDF" in body["skipped"][0]["reason"]


def test_a_document_with_both_parses_and_no_pdf_is_submitted_by_a_bulk_run(client: TestClient, parsed_both_lanes: str):
    body = client.post("/api/documents/run-all").json()

    assert [job["document_id"] for job in body["submitted"]] == [parsed_both_lanes]
    assert body["skipped"] == []


def test_running_everything_twice_while_the_jobs_are_active_reuses_them(
    settings: Settings, pdf_bytes: bytes, second_pdf_bytes: bytes
):
    # the button is pressed twice: the second press must not pay for a second pass over the library
    gate = threading.Event()
    runner = RecordingRunner(gate=gate)
    manager = JobManager(runner, stage_names())
    try:
        with TestClient(create_app(settings, jobs=manager)) as client:
            upload(client, pdf_bytes)
            upload(client, second_pdf_bytes, name="other.pdf")

            first = client.post("/api/documents/run-all").json()
            second = client.post("/api/documents/run-all").json()

            assert len(first["submitted"]) == 2
            assert [job["job_id"] for job in second["submitted"]] == [job["job_id"] for job in first["submitted"]]
            gate.set()  # leaving the app waits for the running job, so it must be able to finish
    finally:
        gate.set()


def test_every_job_of_the_process_is_listed_newest_first(client: TestClient, two_idle_documents: list[str]):
    body = client.get("/api/jobs").json()

    assert {job["document_id"] for job in body} == set(two_idle_documents)
    assert [job["created_at"] for job in body] == sorted((job["created_at"] for job in body), reverse=True)


# ---- static frontend ----------------------------------------------------------------------


def test_static_files_must_be_revalidated_by_the_browser(client: TestClient):
    # No build step, no hashed names: a deploy changes app.js in place, so the browser has to ask again.
    for path in ("/", "/app.js", "/app.css"):
        assert client.get(path).headers["cache-control"] == "no-cache", path


def test_the_index_page_is_served_at_the_root(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert "PaperFacts" in response.text


@pytest.mark.parametrize(
    "asset",
    ["/app.css"]
    + [
        f"/{m}.js"
        for m in (
            "app",
            "state",
            "api",
            "html",
            "router",
            "library",
            "document",
            "facts",
            "figures",
            "samples",
            "job",
            "viewer",
        )
    ],
)
def test_the_front_end_assets_are_served(client: TestClient, asset: str):
    assert client.get(asset).status_code == 200


def test_an_unknown_path_is_not_found(client: TestClient):
    assert client.get("/nope.js").status_code == 404


def test_the_job_worker_is_shut_down_with_the_app(settings: Settings, runner: RecordingRunner):
    # stop accepting new jobs on exit; queued ones are dropped and running ones finish before the app is gone
    manager = JobManager(runner, stage_names())
    with TestClient(create_app(settings, jobs=manager)):
        pass

    with pytest.raises(RuntimeError):
        manager.submit("0123456789abcdef")


def test_an_app_can_be_built_from_the_environment_alone(monkeypatch, tmp_path: Path):
    # this is the path the serve command takes: without an explicit settings object, it reads from the environment
    monkeypatch.setenv("PAPERFACTS_DATA_ROOT", str(tmp_path / "from-env"))

    app = create_app()

    assert app.state.settings.data_root == tmp_path / "from-env"
    assert isinstance(app.state.library, Library)
    assert isinstance(app.state.jobs, JobManager)


# ---- pipeline_runner ---------------------------------------------------------------------


@pytest.fixture
def registered(library: Library, pdf_bytes: bytes) -> str:
    return library.register_upload("paper.pdf", pdf_bytes).document_id[:16]


def test_the_pipeline_runner_hands_the_job_to_run_document(
    monkeypatch, settings: Settings, library: Library, registered
):
    """The job body doesn't orchestrate the pipeline itself: it hands the library's DocumentInput,
    the job's force flag, and the mark callback straight through to workflow, unchanged."""
    received: dict = {}

    def fake_run_document(document, settings_seen, *, force, on_stage):
        received.update(document=document, settings=settings_seen, force=force, on_stage=on_stage)

    monkeypatch.setattr("paperfacts.web.app.run_document", fake_run_document)
    job = Job(job_id="job-1", document_id=registered, force=True, created_at="2026-01-01T00:00:00+00:00")

    def mark(stage: str, status: str, detail: str = "") -> None:
        pass

    pipeline_runner(settings, library)(job, mark)

    assert received == {
        "document": library.document(registered),
        "settings": settings,
        "force": True,
        "on_stage": mark,
    }


def test_the_pipeline_fails_loudly_when_the_pdf_is_gone(monkeypatch, settings: Settings, library: Library):
    calls: list[object] = []
    monkeypatch.setattr("paperfacts.web.app.run_document", lambda *args, **kwargs: calls.append(args))
    seed_artifact(library, "mineru")
    job = Job(job_id="job-1", document_id=DOC_KEY, force=False, created_at="2026-01-01T00:00:00+00:00")

    with pytest.raises(FileNotFoundError):
        pipeline_runner(settings, library)(job, lambda stage, status, detail="": None)
    assert calls == []
