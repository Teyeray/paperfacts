"""PaddleOCR-VL adapter: per-page JSON -> SourceBlock.

Two error-prone spots in this layer get focused coverage:

- **pixel normalization** must divide by the **actual** render size recorded in meta.json, never
  estimate it from a DPI formula;
- **reading order** is determined by ``block_order``; ``None`` (blocks not part of the ordering)
  sort last, otherwise headers/footers would get inserted into the middle of the body text and
  the Markdown would read as a jumble.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest

from paperfacts.adapters import convert as dispatch_convert
from paperfacts.adapters.paddle import convert
from paperfacts.models import DocumentInput, ParsedArtifact
from paperfacts.models.geometry import DocumentGeometry
from support.factories import RawOutputFactory, paddle_page_entry

# Actual pixel size when a 595x842 pt page is rendered at 200 dpi (pypdfium2 rounds).
PAGE0_PX = (1653, 2339)
PAGE1_PX = (1700, 2200)


@pytest.fixture
def artifact(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
) -> ParsedArtifact:
    raw = raw_output_factory.paddle([paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX)])
    return convert(raw, document, geometry)


# ---- Global invariants ----------------------------------------------------------------


def test_every_bbox_is_inside_the_unit_square(artifact: ParsedArtifact):
    for block in artifact.blocks:
        assert 0.0 <= block.bbox.x1 < block.bbox.x2 <= 1.0, block.source_id
        assert 0.0 <= block.bbox.y1 < block.bbox.y2 <= 1.0, block.source_id


def test_markdown_span_invariant_holds_for_the_whole_fixture(artifact: ParsedArtifact):
    for block in artifact.blocks:
        assert artifact.markdown[block.markdown_start : block.markdown_end] == block.content


def test_backend_and_source_id_prefix_are_paddleocr_vl(artifact: ParsedArtifact):
    assert artifact.backend == "paddleocr_vl"
    assert all(block.source_id.startswith("paddleocr_vl_p") for block in artifact.blocks)


# ---- Both wrapper shapes are readable ---------------------------------------------------


def test_wrapped_res_form_is_unwrapped(artifact: ParsedArtifact):
    # save_to_json wraps the content in {"res": {...}}.
    assert len(artifact.blocks) == 8


def test_bare_pruned_result_form_produces_the_same_blocks(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    # The HTTP interface's prunedResult has no res wrapper; both shapes must parse into exactly
    # the same blocks.
    wrapped = raw_output_factory.paddle([paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX)], dir_name="wrapped")
    bare = raw_output_factory.paddle([paddle_page_entry(0, paddle_page_wrapped["res"], PAGE0_PX)], dir_name="bare")

    from_wrapped = convert(wrapped, document, geometry)
    from_bare = convert(bare, document, geometry)

    assert [b.model_dump() for b in from_wrapped.blocks] == [b.model_dump() for b in from_bare.blocks]


def test_a_res_key_without_parsing_res_list_is_not_treated_as_the_wrapper(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry
):
    # Only a res that itself carries parsing_res_list counts as the wrapper; otherwise read the
    # outer dict as-is.
    data = {
        "res": {"something_else": 1},
        "parsing_res_list": [
            {
                "block_id": 0,
                "block_order": 0,
                "block_label": "text",
                "block_bbox": [0, 0, 100, 50],
                "block_content": "x",
            }
        ],
    }
    raw = raw_output_factory.paddle([paddle_page_entry(0, data, PAGE0_PX)])

    assert convert(raw, document, geometry).blocks[0].content == "x"


# ---- Pixels -> normalized ---------------------------------------------------------------


def test_pixel_bbox_is_divided_by_the_recorded_render_size(artifact: ParsedArtifact):
    title = artifact.block("paddleocr_vl_p0_b1")

    assert title.bbox.x1 == pytest.approx(200 / PAGE0_PX[0])
    assert title.bbox.y1 == pytest.approx(150 / PAGE0_PX[1])
    assert title.bbox.x2 == pytest.approx(1450 / PAGE0_PX[0])
    assert title.bbox.y2 == pytest.approx(260 / PAGE0_PX[1])


def test_a_different_render_size_yields_different_normalized_coordinates(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    # Normalization uses the size in meta, not the page's point size; changing width_px must
    # change the result.
    raw = raw_output_factory.paddle([paddle_page_entry(0, paddle_page_wrapped, (826, 1170))])

    artifact = convert(raw, document, geometry)

    assert artifact.block("paddleocr_vl_p0_b1").bbox.x1 == pytest.approx(200 / 826)


def test_bbox_overflowing_the_rendered_page_is_clamped(artifact: ParsedArtifact):
    overflow = artifact.block("paddleocr_vl_p0_b6")

    assert overflow.content.startswith("bbox overflows")
    assert overflow.bbox.x1 == 0.0
    assert overflow.bbox.x2 == 1.0


def test_zero_area_bbox_is_skipped(artifact: ParsedArtifact):
    assert all("zero-area" not in block.content for block in artifact.blocks)


def test_skipping_a_bad_bbox_keeps_the_page_order_dense(artifact: ParsedArtifact):
    orders = [block.order for block in artifact.blocks_on_page(0)]

    assert orders == list(range(len(orders)))


# ---- Reading order --------------------------------------------------------------------


def test_blocks_follow_block_order_not_the_json_array_order(artifact: ParsedArtifact):
    # In the fixture, header (block_order=0) is written after doc_title (block_order=1); the
    # output must be reordered by block_order.
    assert artifact.block("paddleocr_vl_p0_b0").raw_label == "header"
    assert artifact.block("paddleocr_vl_p0_b1").raw_label == "doc_title"
    assert artifact.block("paddleocr_vl_p0_b2").raw_label == "text"


def test_blocks_without_block_order_are_pushed_to_the_end(artifact: ParsedArtifact):
    last = artifact.blocks_on_page(0)[-1]

    assert last.raw_label == "footer"
    assert last.content == "1"


def test_ties_on_block_order_fall_back_to_block_id(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry
):
    data = {
        "parsing_res_list": [
            {
                "block_id": 9,
                "block_order": None,
                "block_label": "text",
                "block_bbox": [0, 0, 10, 10],
                "block_content": "b",
            },
            {
                "block_id": 2,
                "block_order": None,
                "block_label": "text",
                "block_bbox": [0, 0, 10, 10],
                "block_content": "a",
            },
        ]
    }
    raw = raw_output_factory.paddle([paddle_page_entry(0, data, PAGE0_PX)])

    artifact = convert(raw, document, geometry)

    assert [b.content for b in artifact.blocks] == ["a", "b"]


def test_non_dict_entries_in_parsing_res_list_are_ignored(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry
):
    data = {
        "parsing_res_list": [
            "garbage",
            {
                "block_id": 0,
                "block_order": 0,
                "block_label": "text",
                "block_bbox": [0, 0, 10, 10],
                "block_content": "ok",
            },
        ]
    }
    raw = raw_output_factory.paddle([paddle_page_entry(0, data, PAGE0_PX)])

    assert [b.content for b in convert(raw, document, geometry).blocks] == ["ok"]


# ---- Label mapping --------------------------------------------------------------------


def test_doc_title_maps_to_title_and_content_is_stripped(artifact: ParsedArtifact):
    title = artifact.block("paddleocr_vl_p0_b1")

    assert title.type == "title"
    assert title.content == "Transparent conducting oxides for flexible electronics"


def test_table_and_table_title_map_to_table_and_caption(artifact: ParsedArtifact):
    assert artifact.block("paddleocr_vl_p0_b3").type == "table"
    assert artifact.block("paddleocr_vl_p0_b4").type == "caption"


def test_header_and_footer_map_to_unknown(artifact: ParsedArtifact):
    assert artifact.block("paddleocr_vl_p0_b0").type == "unknown"
    assert artifact.blocks_on_page(0)[-1].type == "unknown"


def test_undocumented_label_falls_back_to_unknown_and_keeps_raw_label(artifact: ParsedArtifact):
    mystery = artifact.block("paddleocr_vl_p0_b5")

    assert mystery.type == "unknown"
    assert mystery.raw_label == "sparkle_label"


def test_block_id_is_preserved_as_raw_backend_id(artifact: ParsedArtifact):
    assert artifact.block("paddleocr_vl_p0_b1").raw_backend_id == "0"


def test_missing_block_id_leaves_raw_backend_id_none(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry
):
    data = {"parsing_res_list": [{"block_order": 0, "block_label": "text", "block_bbox": [0, 0, 10, 10]}]}
    raw = raw_output_factory.paddle([paddle_page_entry(0, data, PAGE0_PX)])

    block = convert(raw, document, geometry).blocks[0]

    assert block.raw_backend_id is None
    assert block.content == ""


def test_unknown_label_is_logged_once_per_label(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
    caplog,
):
    raw = raw_output_factory.paddle([paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX)])

    with caplog.at_level(logging.WARNING, logger="paperfacts.adapters.base"):
        convert(raw, document, geometry)

    warnings = [r.getMessage() for r in caplog.records if "unknown block label" in r.getMessage()]
    assert len(warnings) == 1
    assert "sparkle_label" in warnings[0]


# ---- Multiple pages -------------------------------------------------------------------


def test_page_index_comes_from_meta_not_from_the_page_json(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    # We send one page image at a time, so page_index is always null; the real page number can
    # only come from meta.json's pages[].index.
    second = copy.deepcopy(paddle_page_wrapped)
    raw = raw_output_factory.paddle(
        [paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX), paddle_page_entry(1, second, PAGE1_PX)]
    )

    artifact = convert(raw, document, geometry)

    assert sorted({block.page for block in artifact.blocks}) == [0, 1]
    assert artifact.block("paddleocr_vl_p1_b1").type == "title"


def test_each_page_uses_its_own_render_size(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    second = copy.deepcopy(paddle_page_wrapped)
    raw = raw_output_factory.paddle(
        [paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX), paddle_page_entry(1, second, PAGE1_PX)]
    )

    artifact = convert(raw, document, geometry)

    assert artifact.block("paddleocr_vl_p0_b1").bbox.x1 == pytest.approx(200 / PAGE0_PX[0])
    assert artifact.block("paddleocr_vl_p1_b1").bbox.x1 == pytest.approx(200 / PAGE1_PX[0])


def test_meta_without_pages_yields_an_empty_artifact(
    raw_output_factory: RawOutputFactory, document: DocumentInput, geometry: DocumentGeometry
):
    raw = raw_output_factory.paddle([])

    artifact = convert(raw, document, geometry)

    assert artifact.blocks == ()
    assert artifact.markdown == ""


def test_missing_page_json_raises_file_not_found(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    raw = raw_output_factory.paddle([paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX)])
    (raw.out_dir / "pages" / "page_000.json").unlink()

    with pytest.raises(FileNotFoundError, match="native output is missing file"):
        convert(raw, document, geometry)


# ---- A page-level error never degrades into a bad bbox ---------------------------------


@pytest.mark.parametrize("missing", ["width_px", "height_px", "json"])
def test_a_page_missing_its_render_metadata_raises_instead_of_skipping_every_block(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
    missing,
):
    """Missing width_px / height_px / json is a **contract** problem and must raise ValueError.

    If it were skipped block-by-block, a whole page would quietly turn into "nothing was
    recognized on this page" — in run-notes that looks like the parser just did poorly, while
    the real cause (meta.json wasn't written by the Paddle runner) stays hidden.
    """
    raw = raw_output_factory.paddle([paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX, omit=(missing,))])

    with pytest.raises(ValueError, match=r"pages\[0\]"):
        convert(raw, document, geometry)


def test_the_page_level_error_names_the_offending_page(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    # With multiple pages it must say exactly which page, or there's no way to start debugging
    # on a document with dozens of pages.
    raw = raw_output_factory.paddle(
        [
            paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX),
            paddle_page_entry(1, paddle_page_wrapped, PAGE1_PX, omit=("width_px",)),
        ]
    )

    with pytest.raises(ValueError) as excinfo:
        convert(raw, document, geometry)

    assert "pages[1]" in str(excinfo.value)


def test_a_page_level_error_is_not_recorded_as_a_skipped_block(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
    caplog,
):
    raw = raw_output_factory.paddle([paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX, omit=("json",))])

    with caplog.at_level(logging.WARNING, logger="paperfacts.adapters.base"), pytest.raises(ValueError):
        convert(raw, document, geometry)

    assert not [r for r in caplog.records if "skip block" in r.getMessage()]


# ---- Dispatch ----------------------------------------------------------------------------


def test_dispatch_convert_routes_to_the_paddle_adapter(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    raw = raw_output_factory.paddle([paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX)])

    assert dispatch_convert(raw, document, geometry) == convert(raw, document, geometry)


def test_dispatch_convert_rejects_an_unregistered_backend(
    raw_output_factory: RawOutputFactory,
    paddle_page_wrapped: dict[str, Any],
    document: DocumentInput,
    geometry: DocumentGeometry,
):
    raw = raw_output_factory.paddle([paddle_page_entry(0, paddle_page_wrapped, PAGE0_PX)])
    unregistered = raw.model_copy(update={"backend": "brand_new_parser"})

    with pytest.raises(ValueError, match="no adapter registered for backend="):
        dispatch_convert(unregistered, document, geometry)
