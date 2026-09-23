"""Bbox overlays: the M1 visual-acceptance tool.

Overlays and cropping share the same geometry conversion, so "the boxes in the overlay hug the
text" is equivalent to "the cropped regions are correct." Tested here against small PDFs rendered
by real pypdfium2, asserting output size matches page size × dpi/72, and that the ``pages``
argument only draws the requested pages.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from paperfacts.models import DocumentGeometry, NormalizedBBox, ParsedArtifact
from paperfacts.overlay import DEFAULT_OVERLAY_DPI, TYPE_COLORS, draw_page_overlay, render_overlays
from support.factories import DOC_ID, PAGE_SIZES_PT, make_block


@pytest.fixture
def artifact(geometry: DocumentGeometry) -> ParsedArtifact:
    blocks = (
        # A block flush against the top of the page: the label can only be drawn inside the box, or
        # it would get clipped.
        make_block(page=0, order=0, type="title", bbox=NormalizedBBox(x1=0.1, y1=0.001, x2=0.9, y2=0.05)),
        make_block(page=0, order=1, type="text", bbox=NormalizedBBox(x1=0.1, y1=0.2, x2=0.9, y2=0.4)),
        make_block(page=1, order=0, type="table", bbox=NormalizedBBox(x1=0.05, y1=0.1, x2=0.95, y2=0.5)),
    )
    return ParsedArtifact(
        document_id=DOC_ID,
        backend="mineru",
        backend_version="3.4.5",
        pages=geometry.pages,
        markdown="",
        blocks=blocks,
    )


def test_overlay_image_size_matches_the_page_size_times_the_dpi_ratio(two_page_pdf: Path, artifact: ParsedArtifact):
    width_pt, height_pt = PAGE_SIZES_PT[0]

    image = draw_page_overlay(two_page_pdf, artifact, 0, dpi=150)

    assert image.width == pytest.approx(width_pt * 150 / 72, abs=1)
    assert image.height == pytest.approx(height_pt * 150 / 72, abs=1)


def test_overlay_actually_draws_something_on_the_blank_page(two_page_pdf: Path, artifact: ParsedArtifact):
    # A blank PDF renders as pure white; after drawing boxes, non-white pixels must appear.
    blank = draw_page_overlay(two_page_pdf, artifact, 0, dpi=72)
    empty_artifact = artifact.model_copy(update={"blocks": ()})
    untouched = draw_page_overlay(two_page_pdf, empty_artifact, 0, dpi=72)

    assert blank.tobytes() != untouched.tobytes()
    assert untouched.getcolors() == [(untouched.width * untouched.height, (255, 255, 255))]


def test_every_block_type_has_a_colour(artifact: ParsedArtifact):
    # draw_page_overlay indexes TYPE_COLORS[block.type] directly; a missing entry is a runtime KeyError.
    from typing import get_args

    from paperfacts.models import BlockType

    assert set(TYPE_COLORS) == set(get_args(BlockType))


def test_render_overlays_writes_one_png_per_page_with_blocks(
    two_page_pdf: Path, artifact: ParsedArtifact, tmp_path: Path
):
    out_dir = tmp_path / "overlays" / "mineru"

    written = render_overlays(two_page_pdf, artifact, out_dir, dpi=72)

    assert [p.name for p in written] == ["page_000.png", "page_001.png"]
    assert all(p.is_file() for p in written)


def test_render_overlays_creates_the_output_directory(two_page_pdf: Path, artifact: ParsedArtifact, tmp_path: Path):
    out_dir = tmp_path / "a" / "b" / "overlays"

    render_overlays(two_page_pdf, artifact, out_dir, dpi=72)

    assert out_dir.is_dir()


def test_pages_argument_restricts_output_to_the_requested_pages(
    two_page_pdf: Path, artifact: ParsedArtifact, tmp_path: Path
):
    out_dir = tmp_path / "overlays"

    written = render_overlays(two_page_pdf, artifact, out_dir, dpi=72, pages=[1])

    assert [p.name for p in written] == ["page_001.png"]
    assert not (out_dir / "page_000.png").exists()


def test_pages_argument_may_target_a_page_without_blocks(two_page_pdf: Path, artifact: ParsedArtifact, tmp_path: Path):
    # An explicitly requested page is rendered even without blocks, to confirm "this page really
    # has nothing detected."
    empty = artifact.model_copy(update={"blocks": artifact.blocks_on_page(0)})

    written = render_overlays(two_page_pdf, empty, tmp_path / "overlays", dpi=72, pages=[1])

    assert [p.name for p in written] == ["page_001.png"]


def test_empty_pages_list_writes_nothing(two_page_pdf: Path, artifact: ParsedArtifact, tmp_path: Path):
    written = render_overlays(two_page_pdf, artifact, tmp_path / "overlays", dpi=72, pages=[])

    assert written == []


def test_an_artifact_without_blocks_produces_no_overlays(two_page_pdf: Path, artifact: ParsedArtifact, tmp_path: Path):
    empty = artifact.model_copy(update={"blocks": ()})

    assert render_overlays(two_page_pdf, empty, tmp_path / "overlays", dpi=72) == []


def test_written_png_is_readable_and_matches_the_render_size(
    two_page_pdf: Path, artifact: ParsedArtifact, tmp_path: Path
):
    written = render_overlays(two_page_pdf, artifact, tmp_path / "overlays", dpi=DEFAULT_OVERLAY_DPI, pages=[0])

    with Image.open(written[0]) as image:
        assert image.width == pytest.approx(PAGE_SIZES_PT[0][0] * DEFAULT_OVERLAY_DPI / 72, abs=1)


def test_requesting_a_page_beyond_the_pdf_raises_index_error(
    two_page_pdf: Path, artifact: ParsedArtifact, tmp_path: Path
):
    with pytest.raises(IndexError):
        render_overlays(two_page_pdf, artifact, tmp_path / "overlays", dpi=72, pages=[9])
