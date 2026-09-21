"""``pdf.render_region``: the crop a vision model is shown lines up with the box the viewer draws, and a
crop too large for an endpoint is shrunk here, in code that can be tested, rather than on the server.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from paperfacts.models import NormalizedBBox
from paperfacts.pdf import png_bytes, render_page, render_region
from support.factories import make_blank_pdf


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    # One A4 page: 595 x 842 pt, so at 72 dpi one point is one pixel and sizes are easy to reason about.
    return make_blank_pdf(tmp_path / "one.pdf", sizes=[(595.0, 842.0)])


def test_the_crop_is_the_boxs_pixels_on_the_page_render(pdf: Path):
    box = NormalizedBBox(x1=0.1, y1=0.2, x2=0.5, y2=0.4)

    crop = render_region(pdf, 0, box, dpi=72)

    page = render_page(pdf, 0, dpi=72)
    x1, y1, x2, y2 = box.to_pixels(width_px=page.width, height_px=page.height)
    assert (crop.width, crop.height) == (x2 - x1, y2 - y1)


def test_a_crop_uses_the_same_pixel_mapping_as_the_viewer_at_any_dpi(pdf: Path):
    box = NormalizedBBox(x1=0.25, y1=0.25, x2=0.75, y2=0.5)

    at_72 = render_region(pdf, 0, box, dpi=72)
    at_144 = render_region(pdf, 0, box, dpi=144)

    # Doubling the DPI doubles the page, and the crop is the same fraction of it; rounding may cost a pixel.
    assert abs(at_144.width - 2 * at_72.width) <= 2
    assert abs(at_144.height - 2 * at_72.height) <= 2


def test_a_box_thinner_than_a_pixel_still_yields_an_image(pdf: Path):
    hairline = NormalizedBBox(x1=0.5, y1=0.5, x2=0.5001, y2=0.5001)

    crop = render_region(pdf, 0, hairline, dpi=72)

    assert crop.width >= 1 and crop.height >= 1


def test_a_crop_above_the_pixel_budget_is_shrunk_keeping_its_aspect_ratio(pdf: Path):
    whole_page = NormalizedBBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)

    crop = render_region(pdf, 0, whole_page, dpi=200, max_pixels=100_000)

    assert crop.width * crop.height <= 100_000
    assert crop.width * crop.height > 90_000  # shrunk to fit, not further than needed
    assert abs(crop.width / crop.height - 595 / 842) < 0.02


def test_a_crop_within_the_budget_is_left_at_full_resolution(pdf: Path):
    box = NormalizedBBox(x1=0.0, y1=0.0, x2=0.5, y2=0.5)

    unbounded = render_region(pdf, 0, box, dpi=72)
    bounded = render_region(pdf, 0, box, dpi=72, max_pixels=10_000_000)

    assert (bounded.width, bounded.height) == (unbounded.width, unbounded.height)


def test_a_page_out_of_range_is_an_index_error(pdf: Path):
    with pytest.raises(IndexError):
        render_region(pdf, 3, NormalizedBBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0), dpi=72)


def test_png_bytes_round_trips_through_pillow(pdf: Path):
    crop = render_region(pdf, 0, NormalizedBBox(x1=0.0, y1=0.0, x2=0.2, y2=0.2), dpi=72)

    data = png_bytes(crop)

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    decoded = Image.open(io.BytesIO(data))
    assert (decoded.width, decoded.height) == (crop.width, crop.height)
