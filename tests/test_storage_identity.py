"""Document identity ``identity.json``: the single record of the full sha256, display name, and source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperfacts.models import META_FILENAME, DocumentInput
from paperfacts.storage.identity import (
    LEGACY_UPLOAD_META,
    DocumentIdentity,
    ensure_identity,
    mark_uploaded,
    read_identity,
)
from paperfacts.storage.paths import DataLayout
from support.factories import DOC_ID, RawOutputFactory


@pytest.fixture
def layout(tmp_path: Path) -> DataLayout:
    return DataLayout(tmp_path / "data")


@pytest.fixture
def document(tmp_path: Path) -> DocumentInput:
    return DocumentInput(document_id=DOC_ID, pdf_path=tmp_path / "papers" / "ito.pdf", sha256=DOC_ID)


def test_there_is_no_identity_before_anything_was_written(layout: DataLayout):
    assert read_identity(layout, DOC_ID) is None


def test_a_cli_document_records_its_pdf_name_and_original_path(layout: DataLayout, document: DocumentInput):
    identity = ensure_identity(layout, document)

    assert identity.sha256 == DOC_ID
    assert identity.name == "ito.pdf"
    assert identity.source_path == str(document.pdf_path)
    assert identity.uploaded is False
    assert identity.created_at.startswith("20")


def test_an_uploaded_document_records_the_given_name_and_no_path(layout: DataLayout, document: DocumentInput):
    identity = ensure_identity(layout, document, name="Sputtered ITO.pdf", uploaded=True)

    assert identity.name == "Sputtered ITO.pdf"
    assert identity.source_path is None
    assert identity.uploaded is True


def test_the_identity_round_trips_through_the_file(layout: DataLayout, document: DocumentInput):
    written = ensure_identity(layout, document)

    assert layout.identity_path(DOC_ID).is_file()
    assert read_identity(layout, DOC_ID) == written


def test_ensure_identity_is_idempotent_and_keeps_the_first_record(layout: DataLayout, document: DocumentInput):
    # Uploaded via the web, then processed via CLI (or vice versa): the identity written first is
    # not overwritten by whichever comes second.
    first = ensure_identity(layout, document, name="first.pdf", uploaded=True)

    second = ensure_identity(layout, document, name="second.pdf")

    assert second == first
    assert read_identity(layout, DOC_ID).name == "first.pdf"


def test_the_identity_is_addressable_by_the_full_sha_and_by_the_directory_name(
    layout: DataLayout, document: DocumentInput
):
    # The web layer only has the 16-char directory name; both forms must resolve to the same file.
    ensure_identity(layout, document)

    assert read_identity(layout, DOC_ID[:16]) == read_identity(layout, DOC_ID)
    assert layout.identity_path(DOC_ID[:16]) == layout.identity_path(DOC_ID)


def test_the_identity_model_is_frozen_and_validates_the_sha_length():
    with pytest.raises(ValueError):
        DocumentIdentity(sha256="abc", name="x.pdf", created_at="2026-01-01T00:00:00+00:00")
    identity = DocumentIdentity(sha256=DOC_ID, name="x.pdf", created_at="2026-01-01T00:00:00+00:00")
    with pytest.raises(ValueError):
        identity.name = "y.pdf"


# ---- mark_uploaded ----------------------------------------------------------------------


def test_a_cli_identity_is_upgraded_when_the_same_pdf_is_uploaded(layout: DataLayout, document: DocumentInput):
    cli = ensure_identity(layout, document)

    upgraded = mark_uploaded(layout, cli, name="Uploaded name.pdf")

    assert upgraded.uploaded is True
    assert upgraded.name == "Uploaded name.pdf"
    assert upgraded.created_at == cli.created_at  # the timestamp of first seeing this paper is unchanged
    assert read_identity(layout, DOC_ID) == upgraded


def test_an_uploaded_identity_is_left_alone(layout: DataLayout, document: DocumentInput):
    uploaded = ensure_identity(layout, document, name="first.pdf", uploaded=True)
    before = layout.identity_path(DOC_ID).stat().st_mtime_ns

    assert mark_uploaded(layout, uploaded, name="second.pdf") == uploaded
    assert layout.identity_path(DOC_ID).stat().st_mtime_ns == before


# ---- one-time recovery for legacy directories --------------------------------------------


def test_a_legacy_upload_directory_recovers_its_identity_from_source_json(layout: DataLayout):
    legacy = layout.doc_dir(DOC_ID) / LEGACY_UPLOAD_META
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps({"name": "old-upload.pdf", "sha256": DOC_ID, "uploaded_at": "2026-09-01T00:00:00+00:00"}),
        encoding="utf-8",
    )

    identity = read_identity(layout, DOC_ID)

    assert identity == DocumentIdentity(
        sha256=DOC_ID, name="old-upload.pdf", uploaded=True, created_at="2026-09-01T00:00:00+00:00"
    )
    assert layout.identity_path(DOC_ID).is_file()  # written back, so later reads skip the recovery path


def test_a_legacy_cli_directory_recovers_its_identity_from_the_parser_meta(
    layout: DataLayout, raw_output_factory: RawOutputFactory, mineru_content_list, document: DocumentInput
):
    raw = raw_output_factory.mineru(mineru_content_list, dir_name="prepared")
    target = layout.raw_dir(document.document_id, "mineru")
    target.mkdir(parents=True)
    (target / META_FILENAME).write_bytes((raw.out_dir / META_FILENAME).read_bytes())

    identity = read_identity(layout, document.document_id)

    assert identity is not None
    assert identity.sha256 == document.sha256
    assert identity.name == document.pdf_path.name
    assert identity.source_path == str(document.pdf_path)
    assert identity.uploaded is False


def test_an_unreadable_legacy_file_is_skipped_not_fatal(layout: DataLayout, caplog):
    legacy = layout.doc_dir(DOC_ID) / LEGACY_UPLOAD_META
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{not json", encoding="utf-8")

    assert read_identity(layout, DOC_ID) is None
    assert LEGACY_UPLOAD_META in caplog.text


def test_a_directory_with_no_trace_of_its_origin_has_no_identity(layout: DataLayout):
    (layout.doc_dir(DOC_ID) / "parsed").mkdir(parents=True)

    assert read_identity(layout, DOC_ID) is None
    assert not layout.identity_path(DOC_ID).exists()
