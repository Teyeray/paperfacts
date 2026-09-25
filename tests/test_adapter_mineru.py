"""MinerU adapter: native content_list -> SourceBlock.

The adapter is a pure function that only reads fixture JSON — no mineru / torch needed. The
invariants protected here:

- every bbox that lands is within [0, 1];
- a bad box is skipped and recorded in ``collector.skipped``, instead of failing the whole
  document or silently vanishing;
- figure/table captions are split into separate blocks sharing the figure/table's bbox (sample
  IDs often appear only in captions);
- an unknown type converges to ``unknown`` while keeping ``raw_label``;
- when ``page_range`` doesn't start at zero, the page number needs the offset added back.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from paperfacts.adapters import convert, convert_mineru
from paperfacts.models import DocumentGeometry, DocumentInput, ParsedArtifact
from support.factories import RawOutputFactory


@pytest.fixture
def artifact(
    raw_output_factory: RawOutputFactory,
    mineru_content_list: list[dict[str, Any]],
    document: DocumentInput,
    geometry: DocumentGeometry,
) -> ParsedArtifact:
    raw = raw_output_factory.mineru(mineru_content_list)
    return convert(raw, document, geometry)


# ---- Global invariants ----------------------------------------------------------------


def test_every_bbox_is_inside_the_unit_square(artifact: ParsedArtifact):
    for block in artifact.blocks:
        assert 0.0 <= block.bbox.x1 < block.bbox.x2 <= 1.0, block.source_id
        assert 0.0 <= block.bbox.y1 < block.bbox.y2 <= 1.0, block.source_id


def test_artifact_carries_backend_version_and_page_geometry(
    artifact: ParsedArtifact, geometry: DocumentGeometry, document: DocumentInput
):
    assert artifact.backend == "mineru"
    assert artifact.backend_version == "3.4.5"
    assert artifact.pages == geometry.pages
    assert artifact.document_id == document.document_id


def test_raw_output_dir_is_recorded_so_native_files_stay_reachable(
    artifact: ParsedArtifact, raw_output_factory: RawOutputFactory
):
    assert artifact.raw_output_dir == raw_output_factory.base_dir / "raw_mineru"


# ---- Type mapping ----------------------------------------------------------------------


def test_text_level_one_becomes_a_title_block(artifact: ParsedArtifact):
    title = artifact.block("mineru_p0_b0")

    assert title.type == "title"
    assert title.raw_label == "text"
    assert title.content.startswith("Transparent conducting oxides")


def test_plain_text_stays_text(artifact: ParsedArtifact):
    assert artifact.block("mineru_p0_b1").type == "text"


def test_image_becomes_a_figure_whose_content_is_the_image_path(artifact: ParsedArtifact):
    figure = artifact.block("mineru_p0_b2")

    assert figure.type == "figure"
    assert figure.content == "images/fig1.jpg"


def test_table_content_is_the_html_body_not_the_image_path(artifact: ParsedArtifact):
    table = artifact.block("mineru_p1_b0")

    assert table.type == "table"
    assert table.content.startswith("<table>")


def test_equation_becomes_a_formula_block(artifact: ParsedArtifact):
    formula = artifact.block("mineru_p1_b3")

    assert formula.type == "formula"
    assert formula.content == "\\sigma = n e \\mu"


def test_header_maps_to_unknown_but_keeps_its_native_label(artifact: ParsedArtifact):
    # Headers are kept (for reference) but not used for extraction, so they're unified to unknown.
    header = artifact.block("mineru_p0_b5")

    assert header.type == "unknown"
    assert header.raw_label == "header"


def test_undocumented_type_falls_back_to_unknown_and_preserves_raw_label(artifact: ParsedArtifact):
    # Never silently drop: when a new type shows up the block still exists, and raw_label tells
    # us which mapping row to add.
    mystery = artifact.block("mineru_p1_b4")

    assert mystery.type == "unknown"
    assert mystery.raw_label == "sparkle"
    assert mystery.content == "block type MinerU never documented"


def test_unknown_labels_are_logged_once(
    raw_output_factory: RawOutputFactory,
    mineru_content_list: list[dict[str, Any]],
    document: DocumentInput,
    geometry: DocumentGeometry,
    caplog,
):
    raw = raw_output_factory.mineru(mineru_content_list)

    with caplog.at_level(logging.WARNING, logger="paperfacts.adapters"):
        convert(raw, document, geometry)

    assert sum("unknown block label" in record.getMessage() for record in caplog.records) == 1


# ---- Caption splitting ------------------------------------------------------------------


def test_image_captions_become_separate_blocks_sharing_the_figure_bbox(artifact: ParsedArtifact):
    figure = artifact.block("mineru_p0_b2")
    caption_a = artifact.block("mineru_p0_b3")
    caption_b = artifact.block("mineru_p0_b4")

    assert (caption_a.type, caption_b.type) == ("caption", "caption")
    assert caption_a.bbox == figure.bbox
    assert caption_b.bbox == figure.bbox
    assert caption_a.content == "Figure 1. XRD patterns of the as-deposited films."
    assert caption_a.raw_label == "image_caption"


def test_chart_captions_and_footnotes_become_caption_blocks_too(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry
):
    # MinerU 3.x gives a line chart type "chart" and its own caption keys; the conditions a sample is
    # named by are often only in that caption.
    chart = {
        "type": "chart",
        "page_idx": 0,
        "bbox": [100, 100, 500, 400],
        "img_path": "images/chart.jpg",
        "chart_caption": ["Fig. 2. Transmittance spectra of films sputtered at 50-200 W."],
        "chart_footnote": ["Inset: average T in 400-800 nm."],
    }

    blocks = convert(raw_output_factory.mineru([chart]), document, geometry).blocks

    assert [(block.type, block.raw_label) for block in blocks] == [
        ("figure", "chart"),
        ("caption", "chart_caption"),
        ("caption", "chart_footnote"),
    ]
    assert blocks[1].content == "Fig. 2. Transmittance spectra of films sputtered at 50-200 W."
    assert blocks[1].bbox == blocks[0].bbox


def test_table_caption_and_footnote_become_caption_blocks_sharing_the_table_bbox(artifact: ParsedArtifact):
    table = artifact.block("mineru_p1_b0")
    caption = artifact.block("mineru_p1_b1")
    footnote = artifact.block("mineru_p1_b2")

    assert caption.raw_label == "table_caption"
    assert footnote.raw_label == "table_footnote"
    assert caption.bbox == table.bbox == footnote.bbox
    assert footnote.content == "a Measured at 25 °C in air."


def test_blank_attached_text_is_dropped(artifact: ParsedArtifact):
    # The second table_footnote entry in the fixture is pure whitespace; it must not become an
    # empty caption block.
    assert all(block.content.strip() for block in artifact.blocks if block.type == "caption")


# ---- Bad bbox handling ------------------------------------------------------------------


def test_zero_area_bbox_is_skipped_without_failing_the_document(
    artifact: ParsedArtifact, mineru_content_list: list[dict[str, Any]]
):
    # The fixture has 8 entries; 1 bad box among them gets skipped, and captions add 4 more blocks.
    contents = [block.content for block in artifact.blocks]

    assert "zero-area bbox, must be skipped" not in contents
    assert len(artifact.blocks) == 11


def test_skipping_a_bad_bbox_does_not_consume_a_page_order_slot(artifact: ParsedArtifact):
    # A skipped block doesn't consume an order slot; in-page order must stay contiguous from 0,
    # or source_id would have gaps.
    for page in (0, 1):
        orders = [block.order for block in artifact.blocks_on_page(page)]
        assert orders == list(range(len(orders)))


def test_skipped_bad_bbox_is_reported_in_the_warning_log(
    raw_output_factory: RawOutputFactory,
    mineru_content_list: list[dict[str, Any]],
    document: DocumentInput,
    geometry: DocumentGeometry,
    caplog,
):
    # Never silently drop: a bad box must leave a trace in the log, so run-notes can count it.
    raw = raw_output_factory.mineru(mineru_content_list)

    with caplog.at_level(logging.WARNING, logger="paperfacts.adapters"):
        convert(raw, document, geometry)

    skips = [r.getMessage() for r in caplog.records if "skip block" in r.getMessage()]
    assert len(skips) == 1
    assert "page=0" in skips[0]


def test_out_of_range_bbox_is_clamped_to_the_page_edges(artifact: ParsedArtifact):
    header = artifact.block("mineru_p0_b5")

    assert header.bbox.x1 == 0.0
    assert header.bbox.x2 == 1.0
    assert header.bbox.y2 == pytest.approx(0.03)


def test_a_block_without_bbox_is_skipped(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry
):
    raw = raw_output_factory.mineru([{"type": "text", "page_idx": 0, "text": "no bbox at all"}])

    artifact = convert(raw, document, geometry)

    assert artifact.blocks == ()


@pytest.mark.parametrize("page_idx", [None, "1", -1, True])
def test_a_block_without_a_usable_page_is_skipped_with_a_warning_not_filed_on_page_0(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry, caplog, page_idx
):
    item = {"type": "text", "bbox": [0, 0, 500, 100], "text": "where am I"}
    if page_idx is not None:
        item["page_idx"] = page_idx
    raw = raw_output_factory.mineru([item, {"type": "text", "page_idx": 0, "bbox": [0, 0, 500, 100], "text": "ok"}])

    with caplog.at_level("WARNING", logger="paperfacts.adapters"):
        artifact = convert(raw, document, geometry)

    assert [block.content for block in artifact.blocks] == ["ok"]
    assert any("page_idx" in record.getMessage() for record in caplog.records)


def test_list_items_are_joined_into_one_text_block(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry
):
    # MinerU's list entries put the body in list_items; the text field is empty.
    raw = raw_output_factory.mineru(
        [{"type": "list", "page_idx": 0, "bbox": [0, 0, 500, 100], "list_items": ["first", "second"]}]
    )

    block = convert(raw, document, geometry).blocks[0]

    assert block.type == "text"
    assert block.content == "first\nsecond"


def test_a_list_without_items_falls_back_to_the_text_field(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry
):
    raw = raw_output_factory.mineru(
        [{"type": "list", "page_idx": 0, "bbox": [0, 0, 500, 100], "list_items": [], "text": "fallback"}]
    )

    assert convert(raw, document, geometry).blocks[0].content == "fallback"


def test_chart_is_treated_like_an_image(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry
):
    raw = raw_output_factory.mineru(
        [{"type": "chart", "page_idx": 0, "bbox": [0, 0, 500, 100], "img_path": "images/chart.jpg"}]
    )

    block = convert(raw, document, geometry).blocks[0]

    assert block.type == "figure"
    assert block.content == "images/chart.jpg"


# ---- Page offset -----------------------------------------------------------------------


def test_page_range_start_is_added_back_to_every_page_index(
    raw_output_factory: RawOutputFactory,
    mineru_content_list: list[dict[str, Any]],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    # When the runner crops with --start-page 3, MinerU's page_idx restarts from 0; the adapter
    # must add the offset back, or the bbox would get drawn onto page 0.
    raw = raw_output_factory.mineru(mineru_content_list, page_range=[3, None])

    artifact = convert(raw, document, geometry)

    assert sorted({block.page for block in artifact.blocks}) == [3, 4]
    assert artifact.block("mineru_p3_b0").type == "title"


def test_missing_page_range_is_treated_as_no_offset(
    raw_output_factory: RawOutputFactory,
    mineru_content_list: list[dict[str, Any]],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    # source omits page_range entirely (not the same as an empty source, which ParserMeta
    # validation would reject outright as a contract violation)
    source = {
        "pdf": str(document.pdf_path),
        "sha256": document.sha256,
        "page_count": geometry.page_count,
        "parsed_page_count": geometry.page_count,
    }
    raw = raw_output_factory.mineru(mineru_content_list, extra_meta={"source": source})

    artifact = convert(raw, document, geometry)

    assert min(block.page for block in artifact.blocks) == 0


# ---- source_id contract -----------------------------------------------------------------


def test_source_ids_are_unique_and_follow_the_documented_format(artifact: ParsedArtifact):
    ids = [block.source_id for block in artifact.blocks]

    assert len(set(ids)) == len(ids)
    for block in artifact.blocks:
        assert block.source_id == f"mineru_p{block.page}_b{block.order}"


def test_every_source_id_can_be_looked_up(artifact: ParsedArtifact):
    for block in artifact.blocks:
        assert artifact.block(block.source_id) is block


def test_type_counts_reflect_the_fixture(artifact: ParsedArtifact):
    assert artifact.type_counts() == {
        "caption": 4,
        "figure": 1,
        "formula": 1,
        "table": 1,
        "text": 1,
        "title": 1,
        "unknown": 2,
    }


# ---- Dispatch ----------------------------------------------------------------------------


def test_dispatch_convert_routes_to_the_mineru_adapter(
    raw_output_factory: RawOutputFactory,
    mineru_content_list: list[dict[str, Any]],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    raw = raw_output_factory.mineru(mineru_content_list)

    assert convert(raw, document, geometry) == convert_mineru(raw, document, geometry)


def test_meta_without_a_content_list_entry_raises_file_not_found(
    raw_output_factory: RawOutputFactory,
    mineru_content_list: list[dict[str, Any]],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    # meta.files is missing content_list: this output wasn't written by the MinerU runner at
    # all, so the error must list which keys actually exist.
    raw = raw_output_factory.mineru(mineru_content_list, extra_meta={"files": {"middle_json": "x.json"}})

    with pytest.raises(FileNotFoundError) as excinfo:
        convert(raw, document, geometry)

    assert "no 'content_list'" in str(excinfo.value)
    assert "middle_json" in str(excinfo.value)


def test_missing_native_content_list_raises_file_not_found(
    raw_output_factory: RawOutputFactory,
    mineru_content_list: list[dict[str, Any]],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    # When the file meta.json points to has been deleted, it must error, rather than being
    # treated as "this document has no blocks".
    raw = raw_output_factory.mineru(mineru_content_list)
    (raw.out_dir / json.loads((raw.out_dir / "meta.json").read_text())["files"]["content_list"]).unlink()

    with pytest.raises(FileNotFoundError, match="native output is missing file"):
        convert(raw, document, geometry)
