"""The consolidated results over HTTP: one document's dataset and workbook, its chart readings, and the
library-wide corpus table and workbook.

Like ``test_web_app.py``, these watch the HTTP mapping -- which file is served under which keys, and what
is left out -- not how a dataset is built (``test_dataset.py``).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paperfacts.config import Settings
from paperfacts.dataset import field_columns
from paperfacts.figures import FigureReading, FigureReadings
from paperfacts.keys import figure_key_for
from paperfacts.models import NormalizedBBox
from paperfacts.profile import DomainProfile
from paperfacts.web.app import create_app
from paperfacts.web.documents import Library
from paperfacts.web.jobs import JobManager
from paperfacts.workflow import stage_names
from support.factories import make_blank_pdf
from support.profiles import SHIPPED_PROFILE_PATH, shipped_profile
from support.web import (
    DOC_KEY,
    DOC_SHA,
    RecordingRunner,
    corpus_payload,
    seed_artifact,
    seed_dataset,
)

UNKNOWN_ID = "0123456789abcdef"


# ---- fixtures --------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path / "data",
        repo_root=tmp_path,
        profile=str(SHIPPED_PROFILE_PATH),
        llm_api_key="sk-test",
        llm_model="fake-model",
    )


@pytest.fixture
def library(settings: Settings, tco_profile: DomainProfile) -> Library:
    return Library(settings, tco_profile)


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


# ---- the consolidated dataset ------------------------------------------------------------------


def test_the_dataset_is_not_found_before_the_export_ran(client: TestClient, parsed_only: str):
    response = client.get(f"/api/documents/{parsed_only}/dataset")

    assert response.status_code == 404
    assert "dataset" in response.json()["detail"]


def test_the_dataset_of_an_unknown_document_is_not_found(client: TestClient):
    assert client.get(f"/api/documents/{UNKNOWN_ID}/dataset").status_code == 404


def test_the_dataset_is_returned_once_it_is_on_disk(
    client: TestClient, library: Library, parsed_only: str, tco_profile: DomainProfile
):
    payload = {"document_id": parsed_only, "sample_rows": [{"sample_id": "A", "thickness": 300}]}
    seed_dataset(library, parsed_only, payload)

    body = client.get(f"/api/documents/{parsed_only}/dataset").json()

    # The endpoint publishes the full DatasetPayload: what was written is there, the rest at its default.
    assert body["document_id"] == parsed_only
    assert body["sample_rows"] == payload["sample_rows"]
    assert body["paper_row"] == {}
    assert body["quality_rows"] == []
    # The columns come from the profile, whatever the file stored with the table.
    assert body["fields"] == [column.model_dump() for column in field_columns(tco_profile)]


def test_a_label_edit_shows_without_a_rerun(settings: Settings, library: Library, parsed_only: str):
    # A label is display text: it re-keys nothing, so the stored table is still the current one and must be
    # shown under the edited header rather than the one it was written with.
    seed_dataset(library, parsed_only, {"document_id": parsed_only, "fields": [{"name": "x", "scope": "sample"}]})
    profile = library.profile
    first = profile.fields[0]
    edited = dataclasses.replace(
        profile, fields=(dataclasses.replace(first, label=first.label + "（新）"), *profile.fields[1:])
    )
    relabelled = Library(settings, edited)

    dataset = relabelled.dataset(parsed_only)

    assert (relabelled.extractor_key, relabelled.comparison_key) == (library.extractor_key, library.comparison_key)
    assert dataset is not None
    assert dataset.fields == field_columns(edited)
    assert dataset.fields[0].label == first.label + "（新）"


def seed_figures(library: Library, settings: Settings) -> None:
    key = figure_key_for(settings, shipped_profile())
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


def test_a_document_without_figure_readings_has_an_empty_list(client: TestClient, parsed_only: str):
    # Most papers have no chart readings; a 404 here was a red console error on every document page.
    response = client.get(f"/api/documents/{parsed_only}/figures")

    assert response.status_code == 200
    assert response.json()["rows"] == []


def test_the_figure_readings_of_an_unknown_document_are_not_found(client: TestClient):
    assert client.get(f"/api/documents/{UNKNOWN_ID}/figures").status_code == 404


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
    path = library.layout.dataset_path(parsed_only, library.profile.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK\x03\x04 workbook")

    response = client.get(f"/api/documents/{parsed_only}/dataset.xlsx")

    assert response.status_code == 200
    assert response.content == b"PK\x03\x04 workbook"
    assert response.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert "attachment" in response.headers["content-disposition"]
    assert f'filename="paperfacts-{parsed_only}.xlsx"' in response.headers["content-disposition"]


# ---- the corpus table --------------------------------------------------------------------------


def test_the_corpus_is_empty_until_a_document_has_a_dataset(client: TestClient, parsed_only: str):
    body = client.get("/api/dataset").json()

    assert body == {"fields": [], "rows": []}


def test_the_corpus_carries_one_row_per_document_with_a_dataset(
    client: TestClient, library: Library, parsed_only: str, uploaded: str, tco_profile: DomainProfile
):
    # `uploaded` is a second document, deliberately left without a dataset: it must simply be absent.
    seed_dataset(library, parsed_only, corpus_payload(parsed_only))

    body = client.get("/api/dataset").json()

    assert [row["document_id"] for row in body["rows"]] == [parsed_only]
    # The profile's columns, not the one column the stored payload carried.
    assert body["fields"] == [column.model_dump() for column in field_columns(tco_profile)]
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
