"""Regression test against a **real** MinerU 3.4.5 output.

The fixture comes from an actual run of `runners/mineru_runner.py` on the first two pages of a
TCO paper (2026-09-09, Mac, pipeline backend); it keeps only the `meta.json` and
`document_content_list.json` the adapter reads, with local absolute paths scrubbed out.

Hand-written fixtures cover "every shape is handled correctly"; this real sample covers "we
didn't guess the real-world shape wrong". If a MinerU upgrade or an adapter change shifts the
block count, type distribution, or coordinate range, this test goes red first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperfacts.adapters import convert
from paperfacts.adapters.markdown import source_ids_in
from paperfacts.models import DocumentGeometry, DocumentInput, PageGeometry, RawParseOutput

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mineru_real_sample"
# Expected values measured from the actual run; update them together with an explanation if the
# adapter's type mapping or caption-splitting rules change.
EXPECTED_BLOCK_COUNT = 34
EXPECTED_TYPE_COUNTS = {"caption": 1, "table": 1, "text": 19, "title": 6, "unknown": 7}


@pytest.fixture
def real_artifact():
    # Arrange: geometry comes straight from the page sizes the runner recorded in meta.json; no
    # original PDF needed
    raw = RawParseOutput.load(FIXTURE_DIR, "mineru")  # also validates the real meta.json against ParserMeta
    geometry = DocumentGeometry(
        pages=tuple(PageGeometry(index=p.index, width_pt=p.width_pt, height_pt=p.height_pt) for p in raw.meta.pages)
    )
    sha = raw.meta.source.sha256
    document = DocumentInput(document_id=sha, pdf_path=Path("sample.pdf"), sha256=sha)
    # Act
    return convert(raw, document, geometry)


def test_real_sample_block_count_and_type_distribution_are_stable(real_artifact):
    """Block count and type distribution are the regression signals most sensitive to adapter behavior."""
    assert real_artifact.backend_version == "3.4.5"
    assert len(real_artifact.blocks) == EXPECTED_BLOCK_COUNT
    assert real_artifact.type_counts() == EXPECTED_TYPE_COUNTS


def test_real_sample_every_bbox_is_normalized(real_artifact):
    """The 0-1000 coordinates in the real output must all land within [0, 1], and appear only
    on the two pages that were parsed."""
    for block in real_artifact.blocks:
        bbox = block.bbox
        assert 0.0 <= bbox.x1 < bbox.x2 <= 1.0
        assert 0.0 <= bbox.y1 < bbox.y2 <= 1.0
    assert {block.page for block in real_artifact.blocks} == {0, 1}


def test_real_sample_markdown_spans_round_trip(real_artifact):
    """The core invariant holds on real data: markdown[start:end] == content, and source_id
    order matches block order."""
    markdown = real_artifact.markdown
    for block in real_artifact.blocks:
        assert markdown[block.markdown_start : block.markdown_end] == block.content
    assert source_ids_in(markdown) == [block.source_id for block in real_artifact.blocks]


def test_real_sample_recognizes_title_and_running_header(real_artifact):
    """A text_level=1 paper title maps to title; a running header maps to unknown but keeps its
    native label — nothing is lost."""
    title = real_artifact.block("mineru_p0_b1")
    assert title.type == "title"
    assert title.content.startswith("Formation and Characterization of Transparent Conductive Oxide")
    headers = [block for block in real_artifact.blocks if block.raw_label == "header"]
    assert headers and all(block.type == "unknown" for block in headers)
