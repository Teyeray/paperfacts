"""Regression test against a **real** PaddleOCR-VL 3.7 output.

The fixture comes from an actual run of `runners/paddle_runner.py` on the first two pages of a
TCO paper (2026-09-09, Mac CPU, in-process VLM inference); it keeps only the `meta.json` and the
two pages' JSON that the adapter reads, strips large fields the adapter never reads (like
`layout_det_res` / `block_polygon_points`), and scrubs local absolute paths.

It covers the same two pages of the same paper as `test_adapter_mineru_real.py`: the block count
and type distribution of the two independent parsers on real data can be compared directly,
which is also the first empirical evidence for design doc §28's "two independent parsers" assumption.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperfacts.adapters import convert, render_markdown
from paperfacts.models import DocumentGeometry, DocumentInput, PageGeometry, RawParseOutput

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "paddle_real_sample"
# Expected values measured from the actual run; update them together with an explanation if PADDLE_LABELS changes.
EXPECTED_BLOCK_COUNT = 34
EXPECTED_TYPE_COUNTS = {"caption": 1, "table": 1, "text": 19, "title": 6, "unknown": 7}
# Header/footer-type labels seen in the real output that must map to unknown (kept but not used for extraction)
EXPECTED_UNKNOWN_LABELS = {"footer", "header", "header_image", "number"}


@pytest.fixture
def real_artifact():
    # Arrange: geometry comes from the page sizes the runner recorded in meta.json; load itself
    # is a ParserMeta contract check
    raw = RawParseOutput.load(FIXTURE_DIR, "paddleocr_vl")
    geometry = DocumentGeometry(
        pages=tuple(PageGeometry(index=p.index, width_pt=p.width_pt, height_pt=p.height_pt) for p in raw.meta.pages)
    )
    sha = raw.meta.source.sha256
    document = DocumentInput(document_id=sha, pdf_path=Path("sample.pdf"), sha256=sha)
    # Act
    return convert(raw, document, geometry)


def test_real_sample_block_count_and_type_distribution_are_stable(real_artifact):
    """Block count and type distribution are the regression signals most sensitive to adapter behavior."""
    assert real_artifact.backend_version == "3.7.0"
    assert len(real_artifact.blocks) == EXPECTED_BLOCK_COUNT
    assert real_artifact.type_counts() == EXPECTED_TYPE_COUNTS


def test_real_sample_pixel_boxes_normalize_into_the_unit_square(real_artifact):
    """After dividing the real output's pixel coordinates by the render size in meta, they must
    all land within [0, 1], and only on the two pages that were parsed."""
    for block in real_artifact.blocks:
        bbox = block.bbox
        assert 0.0 <= bbox.x1 < bbox.x2 <= 1.0
        assert 0.0 <= bbox.y1 < bbox.y2 <= 1.0
    assert {block.page for block in real_artifact.blocks} == {0, 1}


def test_real_sample_renders_every_block_with_its_marker_in_block_order(real_artifact):
    """On real data, every block's content appears right after its own marker, in block order."""
    markdown = render_markdown(real_artifact.blocks)
    cursor = 0
    for block in real_artifact.blocks:
        chunk = f"<!-- source: {block.source_id} -->\n{block.content}\n"
        position = markdown.find(chunk, cursor)
        assert position >= cursor, block.source_id
        cursor = position + len(chunk)


def test_real_sample_reading_order_puts_unordered_blocks_after_the_body(real_artifact):
    """Body text that has a block_order comes first in reading order; headers/footers/footnotes
    with block_order None are sorted to the end of the page by block_id."""
    page0 = real_artifact.blocks_on_page(0)
    assert [block.raw_label for block in page0[:2]] == ["text", "doc_title"]
    assert page0[1].content.startswith("Formation and Characterization of Transparent Conductive Oxide")
    # 7 blocks with no reading order in the real output: 3 headers, 2 footnotes, a footer, and a page number
    assert [block.raw_label for block in page0[-7:]] == [
        "header",
        "header_image",
        "header_image",
        "footnote",
        "footnote",
        "footer",
        "number",
    ]
    assert {block.raw_label for block in real_artifact.blocks if block.type == "unknown"} == EXPECTED_UNKNOWN_LABELS
