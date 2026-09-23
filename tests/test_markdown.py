"""Rendering blocks as Markdown with provenance markers.

Every block is preceded by ``<!-- source: id -->`` and every page starts with ``<!-- page: N -->``, so the
model can cite a block and the citation resolves back to a page and a bbox. The same rendering is written to
``parsed/<backend>.md`` and, filtered, is what the extraction model reads, so its shape is a contract.
"""

from __future__ import annotations

import re

import pytest

from paperfacts.adapters import render_markdown
from paperfacts.models import BlockType
from support.factories import make_block

ALL_TYPES: tuple[BlockType, ...] = ("text", "title", "figure", "formula", "table", "caption", "unknown")
_SOURCE_MARKER = re.compile(r"<!-- source: ([A-Za-z0-9_]+) -->")


def source_ids_in(markdown: str) -> list[str]:
    return _SOURCE_MARKER.findall(markdown)


@pytest.mark.parametrize("block_type", ALL_TYPES)
def test_every_block_type_is_rendered_verbatim_after_its_marker(block_type):
    block = make_block(type=block_type, content=f"content of a {block_type} block")

    markdown = render_markdown([block])

    assert f"<!-- source: {block.source_id} -->\n{block.content}\n" in markdown


def test_content_containing_markers_and_newlines_is_kept_as_is():
    tricky = "line 1\n<!-- source: fake_p0_b0 -->\nline 3"
    block = make_block(content=tricky)

    markdown = render_markdown([block, make_block(order=1, content="after")])

    assert f"<!-- source: {block.source_id} -->\n{tricky}\n" in markdown


def test_blocks_are_rendered_in_the_order_given():
    # Sorting is the adapter's job (BlockCollector.artifact); the renderer keeps the order it is handed.
    blocks = [
        make_block(page=0, order=0, content="p0b0"),
        make_block(page=0, order=1, content="p0b1"),
        make_block(page=1, order=0, content="p1b0"),
    ]

    markdown = render_markdown(blocks)

    assert source_ids_in(markdown) == [b.source_id for b in blocks]
    assert markdown.index("p0b1") < markdown.index("p1b0")


def test_no_blocks_render_to_empty_output():
    assert render_markdown([]) == ""


def test_page_marker_appears_once_per_page_at_the_page_boundary():
    blocks = [
        make_block(page=0, order=0, content="a"),
        make_block(page=0, order=1, content="b"),
        make_block(page=3, order=0, content="c"),
        make_block(page=3, order=1, content="d"),
    ]

    markdown = render_markdown(blocks)

    assert markdown.count("<!-- page: 0 -->") == 1
    assert markdown.count("<!-- page: 3 -->") == 1
    assert "<!-- page: 1 -->" not in markdown  # a page with no blocks gets no marker
    page0, page3 = markdown.index("<!-- page: 0 -->"), markdown.index("<!-- page: 3 -->")
    assert page0 < markdown.index("\na\n") < page3 < markdown.index("\nc\n")


def test_markdown_starts_with_the_page_marker_of_the_first_page():
    markdown = render_markdown([make_block(page=2)])

    assert markdown.startswith("<!-- page: 2 -->\n\n<!-- source: mineru_p2_b0 -->\n")


def test_page_markers_are_not_mistaken_for_source_markers():
    markdown = render_markdown([make_block(page=4, order=1)])

    assert source_ids_in(markdown) == ["mineru_p4_b1"]


def test_every_block_has_exactly_one_marker():
    blocks = [make_block(page=p, order=o) for p in (0, 1) for o in (0, 1, 2)]

    markdown = render_markdown(blocks)

    assert source_ids_in(markdown) == [b.source_id for b in blocks]
