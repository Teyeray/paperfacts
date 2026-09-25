"""PDF reading and rendering — the only module in the main package that touches PDF files directly.

This layer is the "source of truth for the page coordinate system": both the runner side and the
main package independently compute geometry with pypdfium2, and the two must agree. All test cases
run against blank PDFs generated on the fly, with no dependency on any copyrighted paper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperfacts.models import NormalizedBBox
from paperfacts.pdf import PDF_POINTS_PER_INCH, crop_region, png_bytes, read_geometry, render_page, render_region
from support.factories import PAGE_SIZES_PT, make_blank_pdf


def test_read_geometry_returns_one_entry_per_page_in_document_order(two_page_pdf: Path):
    geometry = read_geometry(two_page_pdf)

    assert geometry.page_count == 2
    assert [(p.index, p.width_pt, p.height_pt) for p in geometry.pages] == [
        (0, PAGE_SIZES_PT[0][0], PAGE_SIZES_PT[0][1]),
        (1, PAGE_SIZES_PT[1][0], PAGE_SIZES_PT[1][1]),
    ]


def test_read_geometry_keeps_pages_of_different_sizes_apart(two_page_pdf: Path):
    # Page sizes can differ within one PDF; normalizing page 1's box with page 0's size would be a bug.
    geometry = read_geometry(two_page_pdf)

    assert geometry.page(0).width_pt != geometry.page(1).width_pt


def test_read_geometry_handles_a_single_page_document(tmp_path: Path):
    pdf = make_blank_pdf(tmp_path / "one.pdf", [(200.0, 400.0)])

    geometry = read_geometry(pdf)

    assert geometry.page_count == 1
    assert geometry.page(0).height_pt == 400.0


def test_read_geometry_raises_for_a_missing_file(tmp_path: Path):
    with pytest.raises(Exception):  # noqa: B017  pdfium raises its own exception type
        read_geometry(tmp_path / "missing.pdf")


@pytest.mark.parametrize("dpi", [72, 150, 200])
def test_render_page_size_follows_points_times_dpi_over_seventy_two(two_page_pdf: Path, dpi):
    width_pt, height_pt = PAGE_SIZES_PT[0]

    image = render_page(two_page_pdf, 0, dpi=dpi)

    assert image.width == pytest.approx(width_pt * dpi / PDF_POINTS_PER_INCH, abs=1)
    assert image.height == pytest.approx(height_pt * dpi / PDF_POINTS_PER_INCH, abs=1)


def test_render_page_renders_the_requested_page_not_always_the_first(two_page_pdf: Path):
    first = render_page(two_page_pdf, 0, dpi=72)
    second = render_page(two_page_pdf, 1, dpi=72)

    assert first.size != second.size


def test_render_page_returns_an_rgb_image(two_page_pdf: Path):
    # Downstream ImageDraw draws colored boxes; RGBA or grayscale would distort the color semantics.
    assert render_page(two_page_pdf, 0, dpi=72).mode == "RGB"


@pytest.mark.parametrize("dpi", [0, -150])
def test_render_page_rejects_non_positive_dpi(two_page_pdf: Path, dpi):
    with pytest.raises(ValueError, match="dpi must be positive"):
        render_page(two_page_pdf, 0, dpi=dpi)


@pytest.mark.parametrize("page_index", [2, 99, -1])
def test_render_page_rejects_out_of_range_page_index(two_page_pdf: Path, page_index):
    # Negative indices silently wrap to the last page in Python; this must raise instead.
    with pytest.raises(IndexError, match="out of range"):
        render_page(two_page_pdf, page_index, dpi=72)


# ---- render_page_cached ---------------------------------------------------------------


def test_render_page_cached_writes_the_png_under_the_overlay_page_name(two_page_pdf: Path, tmp_path: Path):
    from paperfacts.pdf import render_page_cached

    path = render_page_cached(two_page_pdf, 1, dpi=50, cache_dir=tmp_path / "pages" / "50dpi")

    assert path == tmp_path / "pages" / "50dpi" / "page_001.png"
    assert path.read_bytes().startswith(b"\x89PNG")


def test_render_page_cached_does_not_render_again_when_the_file_exists(two_page_pdf: Path, tmp_path: Path, monkeypatch):
    from paperfacts import pdf as pdf_module

    calls: list[int] = []
    original = pdf_module.render_page

    def counting_render_page(path: Path, page_index: int, *, dpi: int):
        calls.append(page_index)
        return original(path, page_index, dpi=dpi)

    monkeypatch.setattr(pdf_module, "render_page", counting_render_page)
    cache_dir = tmp_path / "pages"

    pdf_module.render_page_cached(two_page_pdf, 0, dpi=50, cache_dir=cache_dir)
    pdf_module.render_page_cached(two_page_pdf, 0, dpi=50, cache_dir=cache_dir)

    assert calls == [0]


@pytest.mark.parametrize("page_index", [2, -1])
def test_render_page_cached_rejects_out_of_range_pages_without_leaving_a_file(
    two_page_pdf: Path, tmp_path: Path, page_index: int
):
    from paperfacts.pdf import render_page_cached

    with pytest.raises(IndexError):
        render_page_cached(two_page_pdf, page_index, dpi=50, cache_dir=tmp_path / "pages")

    assert not (tmp_path / "pages").exists()


# ---- pdfium serialization ---------------------------------------------------------------


def test_pdfium_calls_from_many_threads_never_overlap(two_page_pdf: Path, monkeypatch):
    """PDFium is not thread-safe: opening it concurrently corrupts its global state (after which
    every open fails with a Data format error).

    Uses a fake PdfDocument that "opens slowly" to measure concurrency: at most one thread should
    ever be inside pdfium at a time.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from paperfacts import pdf as pdf_module

    real = pdf_module.pdfium.PdfDocument
    inside = 0
    peak = 0
    counter_lock = threading.Lock()

    class SlowPdfDocument:
        def __init__(self, path: str) -> None:
            nonlocal inside, peak
            with counter_lock:
                inside += 1
                peak = max(peak, inside)
            time.sleep(0.01)
            self._doc = real(path)

        def __len__(self) -> int:
            return len(self._doc)

        def __getitem__(self, index: int):
            return self._doc[index]

        def close(self) -> None:
            nonlocal inside
            self._doc.close()
            with counter_lock:
                inside -= 1

    monkeypatch.setattr(pdf_module.pdfium, "PdfDocument", SlowPdfDocument)

    with ThreadPoolExecutor(max_workers=6) as pool:
        sizes = list(pool.map(lambda i: pdf_module.render_page(two_page_pdf, i % 2, dpi=20).size, range(12)))
        geometries = list(pool.map(lambda _: pdf_module.read_geometry(two_page_pdf).page_count, range(6)))

    assert peak == 1
    assert len(sizes) == 12 and geometries == [2] * 6


# ---- Region crops for the vision model -----------------------------------------------------


def test_render_region_cuts_the_box_out_of_the_page_render(two_page_pdf: Path):
    page = render_page(two_page_pdf, 0, dpi=72)
    bbox = NormalizedBBox(x1=0.1, y1=0.2, x2=0.6, y2=0.5)
    crop = render_region(two_page_pdf, 0, bbox, dpi=72)
    x1, y1, x2, y2 = bbox.to_pixels(width_px=page.width, height_px=page.height)
    assert crop.size == (x2 - x1, y2 - y1)


def test_render_region_shrinks_a_crop_over_the_pixel_bound_keeping_its_shape(two_page_pdf: Path):
    bbox = NormalizedBBox(x1=0.0, y1=0.0, x2=1.0, y2=0.5)
    full = render_region(two_page_pdf, 0, bbox, dpi=144)
    small = render_region(two_page_pdf, 0, bbox, dpi=144, max_pixels=10_000)
    assert small.width * small.height <= 10_000
    assert abs(small.width / small.height - full.width / full.height) < 0.05


def test_png_bytes_is_a_png(two_page_pdf: Path):
    assert png_bytes(render_page(two_page_pdf, 0, dpi=36)).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    "bbox",
    [
        NormalizedBBox(x1=0.9999, y1=0.9999, x2=1.0, y2=1.0),
        NormalizedBBox(x1=0.0, y1=0.0, x2=0.0001, y2=0.0001),
        NormalizedBBox(x1=0.5, y1=0.9995, x2=0.6, y2=1.0),
    ],
)
def test_a_box_at_the_page_edge_or_thinner_than_a_pixel_still_yields_an_image(two_page_pdf: Path, bbox):
    crop = render_region(two_page_pdf, 0, bbox, dpi=36)

    assert crop.width >= 1 and crop.height >= 1


def test_crop_region_matches_render_region(two_page_pdf: Path):
    bbox = NormalizedBBox(x1=0.2, y1=0.2, x2=0.7, y2=0.6)
    page = render_page(two_page_pdf, 1, dpi=72)

    assert crop_region(page, bbox).size == render_region(two_page_pdf, 1, bbox, dpi=72).size
