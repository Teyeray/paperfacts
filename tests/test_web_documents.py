"""Document library: what's under ``data/docs/``, how far each document got, and where uploaded
PDFs live.

This layer has no business logic — everything it answers is "what's on disk" — so every test case
starts by seeding artifacts at the paths :class:`~paperfacts.storage.DataLayout` computes,
then asks the library whether it can see them. Two easy-to-silently-break things each get their
own dedicated test case here:

- Artifact paths carry ``extractor_key`` / ``comparison_key``, so switching models or tolerances
  should read as "not processed yet", not pass off a stale result as a fresh one;
- The public document_id is the 16-character directory name, while ``DocumentInput`` needs the
  full 64-character sha256 — the only source for the full sha is ``identity.json``, so a directory
  without an identity can't be reprocessed, and the directory name can't stand in for the sha.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from paperfacts.compare import ComparisonCounts
from paperfacts.config import Settings
from paperfacts.dataset import DocumentDataset, write_dataset_json
from paperfacts.keys import extractor_key
from paperfacts.models import BACKENDS, DocumentInput
from paperfacts.storage import write_text_atomic
from paperfacts.web.documents import Library
from support.extraction import make_field, make_sample
from support.factories import make_block
from support.web import (
    DOC_KEY,
    DOC_SHA,
    seed_artifact,
    seed_cli_document,
    seed_extraction,
    seed_report,
    seed_validation,
)

PDF_BYTES = b"%PDF-1.7\n% fake but well-formed enough for the upload path\n"
OTHER_PDF_BYTES = b"%PDF-1.7\n% a different document\n"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_root=tmp_path / "data", repo_root=tmp_path, llm_api_key="sk-test")


@pytest.fixture
def library(settings: Settings) -> Library:
    return Library(settings)


def sha_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def key_of(data: bytes) -> str:
    return sha_of(data)[:16]


# ---- empty library and id validation -------------------------------------------------------------------


def test_a_data_root_that_does_not_exist_yet_lists_nothing(library: Library):
    # data/ doesn't exist yet on first launch; the list page should be empty, not a 500.
    assert library.list() == []


def test_a_docs_root_without_any_document_lists_nothing(library: Library):
    library.layout.docs_root().mkdir(parents=True)

    assert library.list() == []


def test_files_lying_around_in_the_docs_root_are_not_documents(library: Library):
    library.layout.docs_root().mkdir(parents=True)
    (library.layout.docs_root() / "README.txt").write_text("not a document", encoding="utf-8")

    assert library.list() == []


def test_a_directory_that_is_not_a_document_key_is_skipped_with_a_warning(library: Library, caplog):
    # a hand-made scratch directory must not turn the whole listing into a 500, but it also
    # shouldn't be silently treated as if it weren't there — leave a warning behind.
    (library.layout.docs_root() / "scratch-notes").mkdir(parents=True)
    library.register_upload("paper.pdf", PDF_BYTES)

    listed = library.list()

    assert [s.document_id for s in listed] == [key_of(PDF_BYTES)]
    assert "scratch-notes" in caplog.text


@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        "abc",  # too short
        "a" * 15,
        "a" * 17,  # too long
        "0123456789abcdeg",  # g is not hex
        "0123456789ABCDEF",  # uppercase: directory names are lowercase sha
        "../../../etc/pas",  # right length, wrong characters — also the first line of defense against path traversal
    ],
)
def test_a_malformed_document_id_is_rejected(library: Library, bad_id: str):
    with pytest.raises(KeyError):
        library.summary(bad_id)


def test_a_well_formed_id_that_has_never_been_seen_summarises_as_untouched(library: Library):
    summary = library.summary("0123456789abcdef")

    assert summary.name == "0123456789abcdef"
    assert summary.pdf_available is False
    assert summary.parsed == {"mineru": False, "paddleocr_vl": False}
    assert summary.extracted == {"mineru": False, "paddleocr_vl": False}
    assert summary.compared is False
    assert summary.counts is None
    assert summary.uploaded_at is None


# ---- register_upload ------------------------------------------------------------------


def test_an_upload_is_stored_under_the_sha_of_its_content(library: Library):
    document = library.register_upload("paper.pdf", PDF_BYTES)

    assert document.document_id == sha_of(PDF_BYTES)
    assert len(document.document_id) == 64
    assert document.sha256 == document.document_id
    assert document.pdf_path == library.layout.source_pdf(document.document_id)
    assert document.pdf_path.read_bytes() == PDF_BYTES


def test_an_upload_writes_an_identity_with_the_name_the_sha_and_the_time(library: Library):
    document = library.register_upload("Sputtered ITO.pdf", PDF_BYTES)

    identity = library.identity(document.document_id[:16])
    assert identity is not None
    assert identity.name == "Sputtered ITO.pdf"
    assert identity.sha256 == sha_of(PDF_BYTES)
    assert identity.uploaded is True
    assert identity.source_path is None  # an uploaded document only ever has a source.pdf, no "original path"
    assert identity.created_at.startswith("20")  # ISO-8601; the exact instant isn't worth asserting on


def test_only_the_basename_of_the_uploaded_filename_is_kept(library: Library):
    # a browser can put any string into filename; it's for display only and must never touch a path.
    document = library.register_upload("../../../etc/passwd", PDF_BYTES)

    assert library.identity(document.document_id[:16]).name == "passwd"


def test_an_empty_filename_falls_back_to_the_document_id(library: Library):
    document = library.register_upload("", PDF_BYTES)

    assert library.identity(document.document_id[:16]).name == f"{document.document_id[:16]}.pdf"


def test_uploading_the_same_bytes_again_rewrites_nothing(library: Library):
    """Re-uploading the same content is idempotent: the file isn't rewritten, and the identity
    (including the first-seen time and name) doesn't change.

    Otherwise, clicking upload twice would overwrite "when this paper was first seen", and a large
    PDF would get rewritten for no reason.
    """
    first = library.register_upload("paper.pdf", PDF_BYTES)
    pdf_path = library.layout.source_pdf(first.document_id)
    identity_path = library.layout.identity_path(first.document_id)
    pdf_stat, identity_stat = pdf_path.stat(), identity_path.stat()

    second = library.register_upload("renamed.pdf", PDF_BYTES)

    assert second == first
    assert pdf_path.stat().st_mtime_ns == pdf_stat.st_mtime_ns
    assert identity_path.stat().st_mtime_ns == identity_stat.st_mtime_ns
    assert library.identity(first.document_id[:16]).name == "paper.pdf"


def test_a_truncated_source_pdf_is_repaired_by_uploading_again(library: Library):
    # killed mid-write last time: the file exists but is incomplete; the idempotency check looks at size, not existence.
    pdf_path = library.layout.source_pdf(sha_of(PDF_BYTES))
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(PDF_BYTES[:5])

    library.register_upload("paper.pdf", PDF_BYTES)

    assert pdf_path.read_bytes() == PDF_BYTES


def test_the_identity_is_written_before_the_pdf(library: Library, monkeypatch):
    """Identity lands first, the PDF second: dying partway through never leaves a directory with a
    source.pdf but no discoverable full sha.

    The other way around, that directory could never be reprocessed (document() needs the full
    sha) and would need a manual cleanup.
    """

    def explode(path: Path, data: bytes) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("paperfacts.web.documents.write_bytes_atomic", explode)
    with pytest.raises(OSError):
        library.register_upload("paper.pdf", PDF_BYTES)

    assert library.identity(key_of(PDF_BYTES)) is not None
    assert not library.layout.source_pdf(sha_of(PDF_BYTES)).exists()


def test_two_different_pdfs_get_two_directories(library: Library):
    one = library.register_upload("a.pdf", PDF_BYTES)
    two = library.register_upload("b.pdf", OTHER_PDF_BYTES)

    assert one.document_id != two.document_id
    assert {s.document_id for s in library.list()} == {one.document_id[:16], two.document_id[:16]}


# ---- summary: the three booleans are judged from artifacts on disk ----


def test_a_fresh_upload_has_a_pdf_and_nothing_else(library: Library):
    document = library.register_upload("paper.pdf", PDF_BYTES)

    summary = library.summary(document.document_id[:16])

    assert summary.pdf_available is True
    assert summary.parsed == {"mineru": False, "paddleocr_vl": False}
    assert summary.compared is False
    assert summary.uploaded_at is not None


def test_parsed_is_per_backend(library: Library):
    seed_artifact(library, "mineru")

    summary = library.summary(DOC_KEY)

    assert summary.parsed == {"mineru": True, "paddleocr_vl": False}


def test_extracted_is_per_backend(library: Library):
    seed_artifact(library, "mineru")
    seed_extraction(library, "mineru")

    summary = library.summary(DOC_KEY)

    assert summary.extracted == {"mineru": True, "paddleocr_vl": False}


def test_an_extraction_from_another_model_does_not_count_as_extracted(library: Library):
    """The extraction artifact's path carries an ``extractor_key`` (a fingerprint of the model +
    prompt + field schema).

    If switching models still showed "already extracted", the user would be judging the old
    model's results without knowing it.
    """
    seed_extraction(library, "mineru", extractor_key=extractor_key("some-other-model"))

    assert library.summary(DOC_KEY).extracted["mineru"] is False
    assert library.extraction(DOC_KEY, "mineru") is None


def test_compared_and_counts_come_from_the_report(library: Library):
    seed_report(library, counts=ComparisonCounts(agree=7, conflict=2, ambiguous=1, missing=3, total=13))

    summary = library.summary(DOC_KEY)

    assert summary.compared is True
    assert summary.counts is not None
    assert (summary.counts.agree, summary.counts.conflict, summary.counts.total) == (7, 2, 13)


def test_a_report_written_under_another_comparison_key_does_not_count(library: Library):
    # comparison_key fingerprints the tolerance + normalization rules: change it and an old report must be recomputed.
    seed_report(library, comparison_key="0123456789ab")

    assert library.summary(DOC_KEY).compared is False
    assert library.report(DOC_KEY) is None


# ---- name and uploaded_at both come from identity ----------------------------------------------------


def test_the_name_of_an_uploaded_document_is_the_uploaded_filename(library: Library):
    document = library.register_upload("Sputtered ITO.pdf", PDF_BYTES)

    assert library.summary(document.document_id[:16]).name == "Sputtered ITO.pdf"


def test_a_cli_document_is_named_after_its_pdf_and_has_no_upload_time(library: Library, tmp_path: Path):
    seed_cli_document(library, tmp_path / "papers" / "ito-films.pdf")

    summary = library.summary(DOC_KEY)

    assert summary.name == "ito-films.pdf"
    assert summary.uploaded_at is None


def test_the_name_falls_back_to_the_id_when_there_is_no_identity(library: Library):
    seed_artifact(library, "mineru")  # artifacts exist, but no identity file (e.g. a manually copied directory)

    assert library.summary(DOC_KEY).name == DOC_KEY


# ---- pdf_path and document ---------------------------------------------------------------


def test_the_uploaded_pdf_wins_over_the_path_recorded_by_the_cli(library: Library, two_page_pdf: Path):
    # first processed by the CLI (identity records the original path), then later uploaded via the
    # web from the same content: from now on source.pdf wins, and identity is upgraded to "uploaded".
    data = two_page_pdf.read_bytes()
    seed_cli_document(library, two_page_pdf, document_sha=sha_of(data))

    library.register_upload("paper.pdf", data)

    assert library.pdf_path(key_of(data)) == library.layout.source_pdf(key_of(data))
    identity = library.identity(key_of(data))
    assert identity.uploaded is True
    assert identity.name == "paper.pdf"
    assert library.summary(key_of(data)).uploaded_at == identity.created_at


def test_a_cli_document_falls_back_to_the_original_path(library: Library, two_page_pdf: Path):
    seed_cli_document(library, two_page_pdf)

    assert library.pdf_path(DOC_KEY) == two_page_pdf
    assert library.summary(DOC_KEY).pdf_available is True


def test_there_is_no_pdf_when_the_original_file_is_gone(library: Library, tmp_path: Path):
    # processed by the CLI elsewhere: artifacts exist, the PDF doesn't. The web can view but not reprocess.
    seed_cli_document(library, tmp_path / "elsewhere" / "gone.pdf")
    seed_artifact(library, "mineru")

    assert library.pdf_path(DOC_KEY) is None
    assert library.summary(DOC_KEY).pdf_available is False


def test_asking_for_a_document_without_a_pdf_says_so(library: Library, tmp_path: Path):
    seed_cli_document(library, tmp_path / "gone.pdf")

    with pytest.raises(FileNotFoundError, match="has no available PDF"):
        library.document(DOC_KEY)


def test_only_one_parsed_lane_is_still_not_enough_without_a_pdf(library: Library, tmp_path: Path):
    seed_cli_document(library, tmp_path / "gone.pdf")
    seed_artifact(library, "mineru")

    assert library.runnable(DOC_KEY) is False
    with pytest.raises(FileNotFoundError, match="has no available PDF"):
        library.document(DOC_KEY)


def test_both_parsed_lanes_make_a_document_runnable_without_its_pdf(library: Library, tmp_path: Path):
    """Processed on another machine: the path points where this document's PDF would live, never at an
    invented one, and the export still keeps the filename because that comes from the display name."""
    original = tmp_path / "elsewhere" / "ito-films.pdf"
    seed_cli_document(library, original)
    for backend in BACKENDS:
        seed_artifact(library, backend)

    assert library.has_cached_parse(DOC_KEY) is True
    assert library.runnable(DOC_KEY) is True
    document = library.document(DOC_KEY)
    assert document.sha256 == DOC_SHA
    assert document.pdf_path == library.layout.source_pdf(DOC_SHA)
    assert document.pdf_path.exists() is False
    assert document.display_filename == original.name


def test_asking_for_a_document_without_an_identity_says_so(library: Library):
    # only a source.pdf, no identity: better to raise than let a 16-character directory name stand in for the sha256.
    pdf = library.layout.source_pdf(DOC_SHA)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(PDF_BYTES)

    with pytest.raises(FileNotFoundError, match="identity"):
        library.document(DOC_KEY)


def test_the_document_of_an_upload_is_what_register_upload_returned(library: Library):
    uploaded = library.register_upload("paper.pdf", PDF_BYTES)

    document = library.document(uploaded.document_id[:16])

    assert document == uploaded
    assert isinstance(document, DocumentInput)


def test_the_document_of_a_cli_run_carries_the_full_sha_and_the_original_path(library: Library, two_page_pdf: Path):
    seed_cli_document(library, two_page_pdf)

    document = library.document(DOC_KEY)

    assert document.document_id == DOC_SHA
    assert document.sha256 == DOC_SHA
    assert document.pdf_path == two_page_pdf


# ---- exists / page_image --------------------------------------------------------------


def test_exists_answers_whether_the_directory_is_there(library: Library):
    assert library.exists(DOC_KEY) is False
    seed_artifact(library, "mineru")
    assert library.exists(DOC_KEY) is True


def test_exists_rejects_a_malformed_id_instead_of_probing_the_filesystem(library: Library):
    with pytest.raises(KeyError):
        library.exists("../../../etc/pas")


def test_a_page_image_is_rendered_and_cached_under_the_dpi(library: Library, two_page_pdf: Path):
    seed_cli_document(library, two_page_pdf)

    path = library.page_image(DOC_KEY, 1, dpi=50)

    assert path == library.layout.page_cache_dir(DOC_KEY, 50) / "page_001.png"
    assert path.read_bytes().startswith(b"\x89PNG")


def test_a_page_image_needs_a_pdf(library: Library):
    seed_artifact(library, "mineru")

    with pytest.raises(FileNotFoundError, match="PDF"):
        library.page_image(DOC_KEY, 0, dpi=50)


@pytest.mark.parametrize("page", [2, -1])
def test_a_page_outside_the_document_is_an_index_error(library: Library, two_page_pdf: Path, page: int):
    seed_cli_document(library, two_page_pdf)

    with pytest.raises(IndexError):
        library.page_image(DOC_KEY, page, dpi=50)


# ---- reading artifacts back ---------------------------------------------------------------------


def test_nothing_is_returned_before_anything_has_been_produced(library: Library):
    assert library.report(DOC_KEY) is None
    assert library.extraction(DOC_KEY, "mineru") is None
    assert library.artifact(DOC_KEY, "mineru") is None


def test_the_artifact_is_read_back_as_written(library: Library):
    written = seed_artifact(library, "paddleocr_vl")

    assert library.artifact(DOC_KEY, "paddleocr_vl") == written


def test_the_report_is_read_back_as_written(library: Library):
    written = seed_report(library, counts=ComparisonCounts(agree=1, total=1))

    assert library.report(DOC_KEY) == written


def test_the_extraction_is_normalised_on_the_way_out(library: Library):
    """What's on disk is the LLM's raw-text-level result; what comes back out must be normalized.

    The web page displays ``value``/``unit`` directly; forgetting to normalize would only show up
    as "the table is missing a column", with no error at all — hence this dedicated test.
    """
    sample = make_sample("A", [make_field("sheet_resistance", "12.5", unit_raw="Ω/sq")])
    seed_extraction(library, "mineru", samples=[sample])

    lane = library.extraction(DOC_KEY, "mineru")

    assert lane is not None
    field = lane.sample("A").get("sheet_resistance")
    assert (field.value, field.unit) == (12.5, "Ω/sq")


# ---- list ------------------------------------------------------------------------------


def test_the_most_recently_uploaded_document_comes_first(library: Library):
    for index, (data, created_at) in enumerate(
        [(PDF_BYTES, "2026-01-01T00:00:00+00:00"), (OTHER_PDF_BYTES, "2026-03-01T00:00:00+00:00")]
    ):
        document = library.register_upload(f"paper-{index}.pdf", data)
        identity = library.identity(document.document_id[:16]).model_copy(update={"created_at": created_at})
        write_text_atomic(library.layout.identity_path(document.document_id), identity.model_dump_json())

    assert [s.name for s in library.list()] == ["paper-1.pdf", "paper-0.pdf"]


def test_documents_without_an_upload_time_sort_last(library: Library, two_page_pdf: Path):
    seed_cli_document(library, two_page_pdf)  # CLI-processed, no uploaded_at
    uploaded = library.register_upload("uploaded.pdf", PDF_BYTES)

    listed = library.list()

    assert [s.document_id for s in listed] == [uploaded.document_id[:16], DOC_KEY]
    assert listed[-1].uploaded_at is None


def test_the_extraction_is_regrounded_on_the_way_out(library: Library):
    """Grounding is re-derived on read, exactly as normalisation is.

    The stored verdicts date from whenever the file was written. Serving them unchecked is how the browser
    ends up disagreeing with the command line about which values are verified.
    """
    block = make_block(content="a sheet resistance of 12.5 Ohm/sq was measured")
    seed_artifact(library, "mineru", blocks=[block])
    stale = make_field("sheet_resistance", "12.5", unit_raw="Ω/sq", source_ids=(block.source_id,))
    seed_extraction(library, "mineru", samples=[make_sample("A", [stale.model_copy(update={"grounded": False})])])

    lane = library.extraction(DOC_KEY, "mineru")

    assert lane.sample("A").get("sheet_resistance").grounded is True


def test_an_extraction_without_its_artifact_keeps_the_stored_grounding(library: Library):
    # Grounding cannot be re-checked without the blocks, so the stored verdict is the best answer left.
    ungrounded = make_field("sheet_resistance", "12.5", unit_raw="Ω/sq").model_copy(update={"grounded": False})
    seed_extraction(library, "mineru", samples=[make_sample("A", [ungrounded])])

    lane = library.extraction(DOC_KEY, "mineru")

    assert lane.sample("A").get("sheet_resistance").grounded is False


def test_an_upload_carries_the_uploaded_filename_as_its_display_name(library: Library):
    """The stored PDF is always source.pdf, so the export would otherwise name every upload that."""
    document = library.register_upload("Sputtered ITO.pdf", PDF_BYTES)

    assert document.display_name == "Sputtered ITO.pdf"
    assert document.display_filename == "Sputtered ITO.pdf"
    assert document.pdf_path.name == "source.pdf"
    assert library.document(document.document_id[:16]).display_name == "Sputtered ITO.pdf"


def test_a_cli_document_is_displayed_under_the_name_it_was_first_seen_with(library: Library, two_page_pdf: Path):
    seed_cli_document(library, two_page_pdf)

    document = library.document(DOC_KEY)

    assert document.display_filename == two_page_pdf.name


# ---- the VLM's verdicts and the three-key dataset --------------------------------------------------------


def test_the_validation_is_none_while_the_stage_is_off(library: Library):
    seed_artifact(library, "mineru")
    assert library.validation_key is None
    assert library.validation(DOC_KEY) is None


def test_the_validation_is_read_under_the_current_keys_when_the_stage_is_on(settings: Settings):
    on = Library(dataclasses.replace(settings, vlm_enabled=True))
    seed_artifact(on, "mineru")
    assert on.validation(DOC_KEY) is None
    seeded = seed_validation(on)

    assert on.validation(DOC_KEY) == seeded


def test_a_validation_under_other_settings_is_not_served(settings: Settings):
    on = Library(dataclasses.replace(settings, vlm_enabled=True))
    seed_artifact(on, "mineru")
    seed_validation(on, validation_key="ffffffffffff")

    assert on.validation(DOC_KEY) is None


def test_the_dataset_prefers_the_three_key_file_and_falls_back_to_the_two_key_one(settings: Settings):
    on = Library(dataclasses.replace(settings, vlm_enabled=True))
    seed_artifact(on, "mineru")
    two_key = DocumentDataset(DOC_SHA, "two.pdf", {}, (), (), on.extractor_key, on.comparison_key)
    write_dataset_json(two_key, on.layout.dataset_json_path(DOC_SHA, on.extractor_key, on.comparison_key))
    assert on.dataset(DOC_KEY).filename == "two.pdf"

    assert on.validation_key is not None
    three_key = DocumentDataset(
        DOC_SHA, "three.pdf", {}, (), (), on.extractor_key, on.comparison_key, on.validation_key
    )
    write_dataset_json(
        three_key, on.layout.dataset_json_path(DOC_SHA, on.extractor_key, on.comparison_key, on.validation_key)
    )

    assert on.dataset(DOC_KEY).filename == "three.pdf"
    assert on.dataset(DOC_KEY).validation_key == on.validation_key
