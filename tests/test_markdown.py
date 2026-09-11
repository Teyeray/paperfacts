"""Generating Markdown with provenance markers.

**Core invariant** (design doc §8)::

    markdown[block.markdown_start : block.markdown_end] == block.content

As long as it holds, "the LLM cited some span of the Markdown" can be traced precisely back to a
source_id -> page + bbox. Nearly every case in this file asserts it at the end, because once it
breaks, the entire provenance chain fails silently.
"""

from __future__ import annotations

import pytest

from paperfacts.adapters.markdown import (
    BLOCK_SEPARATOR,
    PAGE_MARKER_TEMPLATE,
    build_markdown,
    render_block,
    source_ids_in,
)
from paperfacts.models import BlockType, SourceBlock
from support.factories import make_block

ALL_TYPES: tuple[BlockType, ...] = ("text", "title", "figure", "formula", "table", "caption", "unknown")


def assert_spans_match_content(markdown: str, blocks: tuple[SourceBlock, ...]) -> None:
    """Assertion helper for the core invariant."""
    assert blocks, "must have at least one block, or this assertion verifies nothing"
    for block in blocks:
        assert block.markdown_start is not None
        assert block.markdown_end is not None
        assert markdown[block.markdown_start : block.markdown_end] == block.content, block.source_id


# ---- Single-block rendering -------------------------------------------------------------


def test_render_block_wraps_title_with_a_heading_prefix():
    body, offset = render_block(make_block(type="title", content="3.2 Electrical properties"))

    assert body == "## 3.2 Electrical properties"
    assert body[offset:] == "3.2 Electrical properties"


def test_render_block_wraps_figure_as_an_image_link_carrying_the_source_id():
    block = make_block(page=7, order=12, type="figure", content="images/fig1.jpg")

    body, offset = render_block(block)

    assert body == "![mineru_p7_b12](images/fig1.jpg)"
    assert body[offset : offset + len(block.content)] == "images/fig1.jpg"


def test_render_block_wraps_formula_in_display_math():
    block = make_block(type="formula", content="E = mc^2")

    body, offset = render_block(block)

    assert body == "$$\nE = mc^2\n$$"
    assert body[offset : offset + len(block.content)] == "E = mc^2"


@pytest.mark.parametrize("block_type", ["text", "table", "caption", "unknown"])
def test_render_block_passes_plain_types_through_unchanged(block_type):
    block = make_block(type=block_type, content="<table><tr><td>1</td></tr></table>")

    body, offset = render_block(block)

    assert body == block.content
    assert offset == 0


# ---- Whole-document generation: the core invariant ----------------------------------------


@pytest.mark.parametrize("block_type", ALL_TYPES)
def test_markdown_span_points_exactly_at_the_content_for_every_block_type(block_type):
    # Every type's syntax prefix has a different length; an offset miscalculation would throw
    # off provenance for that whole class of block.
    blocks = [make_block(type=block_type, content=f"content of a {block_type} block")]

    markdown, spanned = build_markdown(blocks)

    assert_spans_match_content(markdown, spanned)


def test_markdown_span_holds_for_a_mixed_multi_page_document():
    blocks = [
        make_block(page=0, order=0, type="title", content="Introduction"),
        make_block(page=0, order=1, type="text", content="ITO films were deposited at 25 °C."),
        make_block(page=0, order=2, type="figure", content="images/fig1.jpg"),
        make_block(page=0, order=3, type="caption", content="Figure 1. XRD patterns."),
        make_block(page=1, order=0, type="table", content="<table><tr><td>32</td></tr></table>"),
        make_block(page=1, order=1, type="formula", content="\\sigma = n e \\mu"),
        make_block(page=1, order=2, type="unknown", content="ACS Applied Materials"),
    ]

    markdown, spanned = build_markdown(blocks)

    assert_spans_match_content(markdown, spanned)


def test_markdown_span_holds_for_content_containing_markers_and_newlines():
    # Content containing <!-- source: ... --> or newlines must not throw off the offset calculation.
    tricky = "line 1\n<!-- source: fake_p0_b0 -->\nline 3"
    blocks = [make_block(content=tricky), make_block(order=1, content="after")]

    markdown, spanned = build_markdown(blocks)

    assert_spans_match_content(markdown, spanned)


def test_markdown_span_holds_for_empty_content():
    blocks = [make_block(content=""), make_block(order=1, content="next")]

    markdown, spanned = build_markdown(blocks)

    assert markdown[spanned[0].markdown_start : spanned[0].markdown_end] == ""
    assert_spans_match_content(markdown, spanned)


def test_build_markdown_sorts_shuffled_blocks_by_page_then_order():
    # Adapters are allowed to feed blocks out of order (MinerU's content_list does interleave
    # across pages); sorting happens here.
    blocks = [
        make_block(page=1, order=1, content="p1b1"),
        make_block(page=0, order=1, content="p0b1"),
        make_block(page=1, order=0, content="p1b0"),
        make_block(page=0, order=0, content="p0b0"),
    ]

    markdown, spanned = build_markdown(blocks)

    assert [b.content for b in spanned] == ["p0b0", "p0b1", "p1b0", "p1b1"]
    assert markdown.index("p0b1") < markdown.index("p1b0")
    assert_spans_match_content(markdown, spanned)


def test_build_markdown_returns_empty_output_for_no_blocks():
    markdown, spanned = build_markdown([])

    assert markdown == ""
    assert spanned == ()


# ---- Page markers -----------------------------------------------------------------------


def test_page_marker_appears_once_per_page_at_the_page_boundary():
    blocks = [
        make_block(page=0, order=0, content="a"),
        make_block(page=0, order=1, content="b"),
        make_block(page=3, order=0, content="c"),
        make_block(page=3, order=1, content="d"),
    ]

    markdown, _ = build_markdown(blocks)

    assert markdown.count(PAGE_MARKER_TEMPLATE.format(page=0)) == 1
    assert markdown.count(PAGE_MARKER_TEMPLATE.format(page=3)) == 1
    assert PAGE_MARKER_TEMPLATE.format(page=1) not in markdown  # a page with no blocks gets no marker


def test_markdown_starts_with_the_page_marker_of_the_first_page():
    markdown, _ = build_markdown([make_block(page=2)])

    assert markdown.startswith(PAGE_MARKER_TEMPLATE.format(page=2) + BLOCK_SEPARATOR)


# ---- source_ids_in ------------------------------------------------------------------


def test_source_ids_in_lists_every_marker_in_reading_order():
    blocks = [
        make_block(page=1, order=0, content="p1b0"),
        make_block(page=0, order=2, content="p0b2"),
        make_block(page=0, order=0, content="p0b0"),
    ]

    markdown, _ = build_markdown(blocks)

    assert source_ids_in(markdown) == ["mineru_p0_b0", "mineru_p0_b2", "mineru_p1_b0"]


def test_source_ids_in_returns_empty_list_for_markdown_without_markers():
    assert source_ids_in("# just a heading\n\nsome text") == []


def test_source_ids_in_ignores_page_markers():
    # Page markers and block markers look similar; the regex must match only the latter.
    markdown, _ = build_markdown([make_block(page=4, order=1)])

    assert source_ids_in(markdown) == ["mineru_p4_b1"]


def test_every_block_has_a_marker_in_the_generated_markdown():
    blocks = [make_block(page=p, order=o) for p in (0, 1) for o in (0, 1, 2)]

    markdown, spanned = build_markdown(blocks)

    assert source_ids_in(markdown) == [b.source_id for b in spanned]
