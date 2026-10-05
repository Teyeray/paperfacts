"""Supplementary information attached at upload: merged after the main text into one document, identified by
its parts' hashes (``storage.parts_sha256``), and found again by its identity rather than by hashing the merged
file (``storage.document_for_path``)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from paperfacts.cli import app as cli_app
from paperfacts.config import Settings
from paperfacts.models import DocumentInput
from paperfacts.pdf import read_geometry
from paperfacts.profile import DomainProfile
from paperfacts.storage import (
    DataLayout,
    document_for_path,
    document_key,
    parts_sha256,
    read_identity,
    write_bytes_if_absent,
)
from paperfacts.web.app import create_app
from paperfacts.web.documents import Library
from paperfacts.web.jobs import JobManager
from paperfacts.workflow import stage_names
from support.factories import make_blank_pdf
from support.profiles import SHIPPED_PROFILE_PATH
from support.web import RecordingRunner
from test_workflow_run import install_fake_pipeline

# ---- the id ------------------------------------------------------------------------------------------


def test_the_parts_id_is_pinned_to_its_encoding():
    # sha256(b"paperfacts-parts-v1\n" + "\n".join(hex_shas).encode()): changing it would re-identify every
    # document uploaded with SI, so the literal is held here.
    assert parts_sha256(["a" * 64, "b" * 64]) == "8000aacb711c7d5c79b31cd67ae0467569584ef36877e30e359c0103ab99d2e3"


def test_the_parts_id_depends_on_their_order():
    assert parts_sha256(["a" * 64, "b" * 64]) != parts_sha256(["b" * 64, "a" * 64])


def test_a_parts_id_never_equals_the_hash_of_the_main_part_alone():
    assert parts_sha256(["a" * 64]) != "a" * 64


def test_a_document_has_at_least_one_part():
    with pytest.raises(ValueError):
        parts_sha256([])


def test_writing_if_absent_keeps_the_first_bytes(tmp_path: Path):
    target = tmp_path / "doc" / "source.pdf"

    assert write_bytes_if_absent(target, b"first") is True
    assert write_bytes_if_absent(target, b"second") is False

    assert target.read_bytes() == b"first"
    assert [p.name for p in target.parent.iterdir()] == ["source.pdf"]  # no temp file left behind


# ---- the upload route --------------------------------------------------------------------------------


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
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings, jobs=JobManager(RecordingRunner(), stage_names()))) as test_client:
        yield test_client


@pytest.fixture
def main_pdf(tmp_path: Path) -> bytes:
    return make_blank_pdf(tmp_path / "main.pdf", [(300.0, 400.0), (301.0, 400.0), (302.0, 400.0)]).read_bytes()


@pytest.fixture
def si_pdfs(tmp_path: Path) -> list[bytes]:
    return [
        make_blank_pdf(tmp_path / "si1.pdf", [(500.0, 600.0)]).read_bytes(),
        make_blank_pdf(tmp_path / "si2.pdf", [(510.0, 600.0)]).read_bytes(),
    ]


def post(client: TestClient, main: bytes, si: list[tuple[str, bytes]], *, name: str = "paper.pdf"):
    files = [("file", (name, main, "application/pdf"))]
    files += [("si", (si_name, data, "application/pdf")) for si_name, data in si]
    return client.post("/api/documents", files=files)


def test_an_upload_with_two_si_parts_is_one_merged_document(
    client: TestClient, library: Library, main_pdf: bytes, si_pdfs: list[bytes]
):
    response = post(client, main_pdf, [("S1.pdf", si_pdfs[0]), ("S2.pdf", si_pdfs[1])])

    assert response.status_code == 202, response.text
    shas = [hashlib.sha256(data).hexdigest() for data in (main_pdf, *si_pdfs)]
    expected = parts_sha256(shas)
    body = response.json()
    assert body["document"]["document_id"] == document_key(expected)
    assert body["document"]["name"] == "paper.pdf"
    assert body["duplicate_of"] is None
    identity = read_identity(library.layout, expected)
    assert identity is not None and identity.sha256 == expected
    assert [(p.name, p.sha256, p.first_page, p.pages) for p in identity.parts or ()] == [
        ("paper.pdf", shas[0], 0, 3),
        ("S1.pdf", shas[1], 3, 1),
        ("S2.pdf", shas[2], 4, 1),
    ]
    assert body["document"]["parts"] == [p.model_dump() for p in identity.parts or ()]
    geometry = read_geometry(library.layout.source_pdf(expected))
    assert [page.width_pt for page in geometry.pages] == [300.0, 301.0, 302.0, 500.0, 510.0]


def test_the_same_upload_twice_is_the_same_document_and_keeps_the_first_bytes(
    client: TestClient, library: Library, main_pdf: bytes, si_pdfs: list[bytes]
):
    first = post(client, main_pdf, [("S1.pdf", si_pdfs[0])]).json()
    key = first["document"]["document_id"]
    stored = library.layout.source_pdf(key).read_bytes()

    second = post(client, main_pdf, [("S1.pdf", si_pdfs[0])], name="renamed.pdf").json()

    assert second["document"]["document_id"] == key
    assert second["document"]["name"] == "paper.pdf"  # the identity is written once
    assert library.layout.source_pdf(key).read_bytes() == stored
    assert library.document_ids() == [key]


def test_the_si_order_is_part_of_the_id(client: TestClient, main_pdf: bytes, si_pdfs: list[bytes]):
    one = post(client, main_pdf, [("S1.pdf", si_pdfs[0]), ("S2.pdf", si_pdfs[1])]).json()
    other = post(client, main_pdf, [("S2.pdf", si_pdfs[1]), ("S1.pdf", si_pdfs[0])]).json()

    assert one["document"]["document_id"] != other["document"]["document_id"]
    assert [p["first_page"] for p in other["document"]["parts"]] == [0, 3, 4]
    assert [p["name"] for p in other["document"]["parts"]] == ["paper.pdf", "S2.pdf", "S1.pdf"]


def test_a_single_file_upload_keeps_the_hash_of_its_bytes(client: TestClient, library: Library, main_pdf: bytes):
    body = post(client, main_pdf, []).json()

    sha = hashlib.sha256(main_pdf).hexdigest()
    assert body["document"]["document_id"] == document_key(sha)
    assert body["document"]["parts"] is None
    assert library.layout.source_pdf(sha).read_bytes() == main_pdf
    assert "parts" not in json.loads(library.layout.identity_path(sha).read_text())


def test_the_size_limit_is_on_all_the_parts_together(settings: Settings, main_pdf: bytes, si_pdfs: list[bytes]):
    # Each part fits on its own; the three together do not.
    limit = max(len(main_pdf), *map(len, si_pdfs)) + 16
    small = dataclasses.replace(settings, max_upload_bytes=limit)
    with TestClient(create_app(small)) as client:
        alone = post(client, main_pdf, [])
        response = post(client, main_pdf, [("S1.pdf", si_pdfs[0]), ("S2.pdf", si_pdfs[1])])

    assert alone.status_code == 202
    assert response.status_code == 413


def test_an_si_part_that_is_not_a_pdf_is_refused_by_name(client: TestClient, library: Library, main_pdf: bytes):
    response = post(client, main_pdf, [("notes.docx", b"PK\x03\x04 not a pdf")])

    assert response.status_code == 400
    assert "notes.docx" in response.json()["detail"]
    assert library.document_ids() == []


def test_an_si_part_pdfium_cannot_open_is_refused_by_name(client: TestClient, library: Library, main_pdf: bytes):
    response = post(client, main_pdf, [("broken.pdf", b"%PDF-1.4 not really a pdf")])

    assert response.status_code == 400
    assert "broken.pdf" in response.json()["detail"]
    assert library.document_ids() == []


def test_an_si_part_whose_pages_cannot_be_imported_is_refused_by_name(
    client: TestClient, library: Library, main_pdf: bytes, si_pdfs: list[bytes], monkeypatch
):
    real_import = pdfium.PdfDocument.import_pages

    def import_pages(self, pdf, *args, **kwargs):
        if len(pdf) == 1:  # the SI part (one page); the main text has three
            raise pdfium.PdfiumError("Failed to import pages.")
        return real_import(self, pdf, *args, **kwargs)

    monkeypatch.setattr(pdfium.PdfDocument, "import_pages", import_pages)
    response = post(client, main_pdf, [("odd.pdf", si_pdfs[0])])

    assert response.status_code == 400
    assert "odd.pdf" in response.json()["detail"]
    assert library.document_ids() == []


def test_a_merge_that_cannot_be_saved_is_refused_as_the_upload(
    client: TestClient, library: Library, main_pdf: bytes, si_pdfs: list[bytes], monkeypatch
):
    def save(self, *args, **kwargs):
        raise pdfium.PdfiumError("Failed to save document.")

    monkeypatch.setattr(pdfium.PdfDocument, "save", save)
    response = post(client, main_pdf, [("si.pdf", si_pdfs[0])])

    assert response.status_code == 400
    assert response.json()["detail"].startswith("The uploaded PDFs could not be merged into one")
    assert library.document_ids() == []


@pytest.mark.parametrize("repeat", ["main", "si"])
def test_a_part_attached_twice_is_refused(
    client: TestClient, library: Library, main_pdf: bytes, si_pdfs: list[bytes], repeat: str
):
    si = [("S1.pdf", si_pdfs[0]), ("again.pdf", main_pdf if repeat == "main" else si_pdfs[0])]

    response = post(client, main_pdf, si)

    assert response.status_code == 400
    assert "again.pdf" in response.json()["detail"]
    assert library.document_ids() == []


def test_more_than_four_si_parts_are_refused(client: TestClient, main_pdf: bytes, tmp_path: Path):
    si = [(f"S{i}.pdf", make_blank_pdf(tmp_path / f"s{i}.pdf", [(400.0 + i, 500.0)]).read_bytes()) for i in range(5)]

    assert post(client, main_pdf, si).status_code == 400


def test_two_main_files_are_still_refused(client: TestClient, main_pdf: bytes, si_pdfs: list[bytes]):
    files = [("file", ("a.pdf", main_pdf, "application/pdf")), ("file", ("b.pdf", si_pdfs[0], "application/pdf"))]

    assert client.post("/api/documents", files=files).status_code == 400


def test_the_main_text_alone_already_in_the_library_is_named(client: TestClient, main_pdf: bytes, si_pdfs: list[bytes]):
    alone = post(client, main_pdf, [], name="main only.pdf").json()

    merged = post(client, main_pdf, [("S1.pdf", si_pdfs[0])]).json()

    assert merged["duplicate_of"] == {
        "document_id": alone["document"]["document_id"],
        "name": "main only.pdf",
        "has_si": False,
    }


def test_the_main_text_uploaded_alone_after_its_merged_document_is_named(
    client: TestClient, main_pdf: bytes, si_pdfs: list[bytes]
):
    merged = post(client, main_pdf, [("S1.pdf", si_pdfs[0])]).json()

    alone = post(client, main_pdf, [], name="main only.pdf").json()
    again = post(client, main_pdf, [], name="main only.pdf").json()

    expected = {"document_id": merged["document"]["document_id"], "name": "paper.pdf", "has_si": True}
    assert alone["duplicate_of"] == expected
    assert again["duplicate_of"] == expected


def test_an_ordinary_upload_has_no_duplicate(client: TestClient, main_pdf: bytes, si_pdfs: list[bytes]):
    post(client, si_pdfs[0], [])

    assert post(client, main_pdf, []).json()["duplicate_of"] is None


def test_the_upload_route_documents_its_si_field(client: TestClient):
    schema = client.get("/openapi.json").json()

    properties = schema["paths"]["/api/documents"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"][
        "properties"
    ]
    assert properties["si"]["type"] == "array"
    assert properties["si"]["items"]["format"] == "binary"
    assert properties["si"]["maxItems"] == 4


# ---- resolving a stored source.pdf -------------------------------------------------------------------


@pytest.fixture
def merged(library: Library, main_pdf: bytes, si_pdfs: list[bytes]) -> DocumentInput:
    return library.register_upload("paper.pdf", main_pdf, [("S1.pdf", si_pdfs[0])])


def test_a_stored_source_pdf_is_its_document_not_a_new_hash(library: Library, merged: DocumentInput):
    found = document_for_path(library.layout, library.layout.source_pdf(merged.document_id))

    assert found.document_id == merged.document_id
    assert found.display_name == "paper.pdf"
    assert found.document_id != hashlib.sha256(found.pdf_path.read_bytes()).hexdigest()


def test_any_other_path_is_hashed(library: Library, tmp_path: Path, main_pdf: bytes):
    path = tmp_path / "elsewhere" / "source.pdf"
    path.parent.mkdir()
    path.write_bytes(main_pdf)

    assert document_for_path(library.layout, path).document_id == hashlib.sha256(main_pdf).hexdigest()


def test_a_source_pdf_without_an_identity_is_hashed(tmp_path: Path, main_pdf: bytes):
    layout = DataLayout(tmp_path / "data")
    path = layout.source_pdf("0123456789abcdef")
    path.parent.mkdir(parents=True)
    path.write_bytes(main_pdf)

    assert document_for_path(layout, path).document_id == hashlib.sha256(main_pdf).hexdigest()


def test_a_batch_over_the_stored_documents_mints_no_new_document(
    monkeypatch, settings: Settings, library: Library, merged: DocumentInput
):
    install_fake_pipeline(monkeypatch)
    before = sorted(p.name for p in library.layout.docs_root().iterdir())

    result = CliRunner().invoke(
        cli_app,
        ["batch", str(library.layout.docs_root()), "--offline", "--data-root", str(settings.data_root)],
    )

    assert result.exit_code == 0, result.output
    assert "Completed: 1 papers" in result.output
    assert sorted(p.name for p in library.layout.docs_root().iterdir()) == before == [merged.document_id[:16]]


def test_a_cli_command_on_a_stored_source_pdf_names_its_document(
    monkeypatch, settings: Settings, library: Library, merged: DocumentInput
):
    from paperfacts import workflow

    spy = install_fake_pipeline(monkeypatch)
    # The CLI imported its own binding; patch that call site as well to keep this test parser-free.
    monkeypatch.setattr("paperfacts.cli.parse_document", workflow.parse_document)
    pdf_path = library.layout.source_pdf(merged.document_id)

    result = CliRunner().invoke(cli_app, ["parse", str(pdf_path), "--data-root", str(settings.data_root)])

    assert result.exit_code == 0, result.output
    assert spy.parse == [("mineru", False), ("paddleocr_vl", False)]
    assert f"document_id={merged.document_id[:16]}" in result.output
    assert library.document_ids() == [merged.document_id[:16]]
