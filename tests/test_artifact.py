"""The unified artifact model: idempotency of document_id, look-up-ability of source_id,
round-trip consistency of the artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from paperfacts.models import (
    BACKENDS,
    META_FILENAME,
    DocumentInput,
    PageGeometry,
    ParsedArtifact,
    RawParseOutput,
    SourceBlock,
    make_source_id,
    sha256_of_file,
)
from support.factories import DOC_ID, make_block

# ---- DocumentInput ------------------------------------------------------------------


def test_from_path_uses_content_sha256_as_document_id(two_page_pdf: Path):
    # document_id must be exactly the content sha256, or the "same file is never reprocessed"
    # guarantee doesn't hold.
    document = DocumentInput.from_path(two_page_pdf)

    assert document.document_id == document.sha256
    assert document.document_id == sha256_of_file(two_page_pdf)
    assert len(document.document_id) == 64


def test_from_path_is_idempotent_across_renames(two_page_pdf: Path, tmp_path: Path):
    # Renaming/moving doesn't change the content, so document_id must stay the same --
    # directory reuse depends on exactly this.
    original = DocumentInput.from_path(two_page_pdf)
    renamed = tmp_path / "another-name.pdf"
    renamed.write_bytes(two_page_pdf.read_bytes())

    assert DocumentInput.from_path(renamed).document_id == original.document_id


def test_from_path_resolves_to_an_absolute_path(tmp_path: Path, two_page_pdf: Path):
    document = DocumentInput.from_path(two_page_pdf)

    assert document.pdf_path.is_absolute()


def test_from_path_raises_file_not_found_for_missing_pdf(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="PDF does not exist"):
        DocumentInput.from_path(tmp_path / "nope.pdf")


def test_from_path_raises_file_not_found_for_a_directory(tmp_path: Path):
    # A directory is not a file: is_file() guards ahead of the sha256 call, avoiding a
    # confusing IsADirectoryError.
    with pytest.raises(FileNotFoundError):
        DocumentInput.from_path(tmp_path)


def test_document_input_rejects_short_document_id(two_page_pdf: Path):
    with pytest.raises(ValidationError):
        DocumentInput(document_id="short", pdf_path=two_page_pdf, sha256="a" * 64)


def test_sha256_of_file_streams_large_files(tmp_path: Path):
    # Chunked reading (1 MiB) must not change the result: a file that straddles a chunk
    # boundary must hash the same as reading it all at once.
    import hashlib

    path = tmp_path / "big.bin"
    payload = b"x" * ((1 << 20) + 7)
    path.write_bytes(payload)

    assert sha256_of_file(path) == hashlib.sha256(payload).hexdigest()


# ---- source_id / SourceBlock --------------------------------------------------------


def test_make_source_id_uses_the_documented_format():
    # This string is the unique key between Markdown and a page region; changing the format
    # changes the contract.
    assert make_source_id("mineru", 7, 12) == "mineru_p7_b12"
    assert make_source_id("paddleocr_vl", 0, 0) == "paddleocr_vl_p0_b0"


@pytest.mark.parametrize("field", ["page", "order"])
def test_source_block_rejects_negative_page_or_order(field):
    # Both page number and in-page order are 0-based non-negative integers; a negative value
    # means the adapter miscalculated.
    with pytest.raises(ValidationError):
        SourceBlock.model_validate(make_block().model_dump() | {field: -1})


def test_source_block_rejects_confidence_outside_zero_one():
    with pytest.raises(ValidationError):
        SourceBlock.model_validate(make_block().model_dump() | {"confidence": 1.5})


# ---- ParsedArtifact -----------------------------------------------------------------


def make_artifact(blocks: tuple[SourceBlock, ...] = ()) -> ParsedArtifact:
    return ParsedArtifact(
        document_id=DOC_ID,
        backend="mineru",
        backend_version="3.4.5",
        pages=(
            PageGeometry(index=0, width_pt=595.0, height_pt=842.0),
            PageGeometry(index=1, width_pt=612.0, height_pt=792.0),
        ),
        markdown="# demo",
        blocks=blocks,
    )


def test_block_looks_up_by_source_id():
    blocks = (make_block(page=0, order=0), make_block(page=1, order=3, content="second"))
    artifact = make_artifact(blocks)

    assert artifact.block("mineru_p1_b3").content == "second"


def test_block_raises_key_error_for_an_unknown_source_id():
    # A failed lookup must raise; returning None would let downstream code draw a default box
    # with no visible error.
    artifact = make_artifact((make_block(),))

    with pytest.raises(KeyError, match="mineru_p9_b9"):
        artifact.block("mineru_p9_b9")


def test_blocks_on_page_filters_by_page_and_keeps_order():
    blocks = (
        make_block(page=0, order=0, content="a"),
        make_block(page=1, order=0, content="b"),
        make_block(page=0, order=1, content="c"),
    )
    artifact = make_artifact(blocks)

    assert [b.content for b in artifact.blocks_on_page(0)] == ["a", "c"]
    assert artifact.blocks_on_page(2) == ()


def test_type_counts_counts_each_type_and_sorts_keys():
    blocks = (
        make_block(page=0, order=0, type="text"),
        make_block(page=0, order=1, type="title"),
        make_block(page=0, order=2, type="text"),
        make_block(page=0, order=3, type="caption"),
    )
    artifact = make_artifact(blocks)

    counts = artifact.type_counts()

    assert counts == {"caption": 1, "text": 2, "title": 1}
    assert list(counts) == ["caption", "text", "title"]  # sorted so logs stay diffable


def test_page_count_comes_from_the_page_geometry_table():
    assert make_artifact().page_count == 2


def test_write_then_read_round_trips_to_an_equal_artifact(tmp_path: Path):
    # Writing to disk and reading it back must be exactly equal, or the overlay / extraction
    # stage would get different coordinates than what was parsed.
    blocks = (make_block(page=0, order=0), make_block(page=1, order=0, type="figure"))
    artifact = make_artifact(blocks)
    path = tmp_path / "nested" / "mineru.artifact.json"

    artifact.write(path)
    restored = ParsedArtifact.read(path)

    assert restored == artifact


def test_write_creates_missing_parent_directories(tmp_path: Path):
    path = tmp_path / "a" / "b" / "c" / "mineru.artifact.json"

    make_artifact().write(path)

    assert path.is_file()


def minimal_meta(parser: str, **overrides) -> dict:
    """The minimal meta.json that satisfies ParserMeta validation; used when a unit test only
    cares about the parser / version fields."""
    meta = {
        "parser": parser,
        "source": {"pdf": "sample.pdf", "sha256": "0" * 64, "page_count": 2, "parsed_page_count": 2},
    }
    meta.update(overrides)
    return meta


# ---- RawParseOutput -----------------------------------------------------------------


def test_load_reads_meta_json_and_records_the_parser_version(tmp_path: Path):
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    (out_dir / META_FILENAME).write_text(json.dumps(minimal_meta("mineru", parser_version="3.4.5")), encoding="utf-8")

    raw = RawParseOutput.load(out_dir, "mineru")

    assert raw.backend == "mineru"
    assert raw.backend_version == "3.4.5"
    assert raw.cache_hit is False
    assert raw.meta.parser == "mineru"


def test_load_defaults_the_version_to_unknown_when_meta_omits_it(tmp_path: Path):
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    (out_dir / META_FILENAME).write_text(json.dumps(minimal_meta("paddleocr_vl")), encoding="utf-8")

    assert RawParseOutput.load(out_dir, "paddleocr_vl").backend_version == "unknown"


def test_load_raises_file_not_found_when_meta_json_is_missing(tmp_path: Path):
    # meta.json is only written after full success, so "missing" means "this parse run is unusable".
    out_dir = tmp_path / "raw"
    out_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="parser output is incomplete"):
        RawParseOutput.load(out_dir, "mineru")


def test_load_raises_value_error_when_meta_belongs_to_another_parser(tmp_path: Path):
    # When the directory got mixed up (e.g. moved by hand) it must error, or the paddle adapter
    # would end up reading mineru's output.
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    (out_dir / META_FILENAME).write_text(json.dumps(minimal_meta("paddleocr_vl")), encoding="utf-8")

    with pytest.raises(ValueError, match="belongs to parser='paddleocr_vl'"):
        RawParseOutput.load(out_dir, "mineru")


def test_load_marks_cache_hit_when_asked(tmp_path: Path):
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    (out_dir / META_FILENAME).write_text(json.dumps(minimal_meta("mineru")), encoding="utf-8")

    assert RawParseOutput.load(out_dir, "mineru", cache_hit=True).cache_hit is True


def test_backends_constant_lists_exactly_the_two_supported_parsers():
    assert BACKENDS == ("mineru", "paddleocr_vl")
