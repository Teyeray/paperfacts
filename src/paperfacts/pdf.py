"""PDF reading and page rendering: the only module that calls pypdfium2.

The runners use the same library and the same rendering call, so the page geometry a parser records and the
geometry used here for overlays and the web viewer agree numerically.
"""

from __future__ import annotations

import io
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from paperfacts.models import DocumentGeometry, NormalizedBBox, PageGeometry
from paperfacts.storage import overlay_page_name, write_atomic

PDF_POINTS_PER_INCH = 72
# PDFium is not thread-safe. The web server renders pages from worker threads and reads geometry from the
# job thread; concurrent calls corrupt pdfium's global state, after which every later open of the same PDF
# fails with "Data format error" until the process restarts. One process-wide lock serialises every call;
# a page renders in a few hundred milliseconds, so this costs nothing noticeable.
_PDFIUM_LOCK = threading.Lock()


@contextmanager
def _open(pdf_path: Path) -> Iterator[pdfium.PdfDocument]:
    with _PDFIUM_LOCK:
        document = pdfium.PdfDocument(str(pdf_path))
        try:
            yield document
        finally:
            document.close()


def read_geometry(pdf_path: Path) -> DocumentGeometry:
    """Every page's size, in PDF points."""
    with _open(pdf_path) as document:
        pages = tuple(
            PageGeometry(index=index, width_pt=width_pt, height_pt=height_pt)
            for index in range(len(document))
            for width_pt, height_pt in (document[index].get_size(),)
        )
    return DocumentGeometry(pages=pages)


def render_page(pdf_path: Path, page_index: int, *, dpi: int) -> Image.Image:
    """Render one page to RGB; ``px = pt * dpi / 72`` (pdfium rounds up)."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    with _open(pdf_path) as document:
        if not 0 <= page_index < len(document):
            raise IndexError(f"page index {page_index} out of range, document has {len(document)} pages")
        return document[page_index].render(scale=dpi / PDF_POINTS_PER_INCH).to_pil().convert("RGB")


def render_page_cached(pdf_path: Path, page_index: int, *, dpi: int, cache_dir: Path) -> Path:
    """Render and cache one page as PNG. Written atomically, so the file's existence means it is complete."""
    path = cache_dir / overlay_page_name(page_index)
    if not path.is_file():
        image = render_page(pdf_path, page_index, dpi=dpi)
        write_atomic(path, lambda tmp: image.save(tmp, format="PNG"))
    return path


def render_region(
    pdf_path: Path,
    page_index: int,
    bbox: NormalizedBBox,
    *,
    dpi: int,
    max_pixels: int | None = None,
) -> Image.Image:
    """Render one page and cut out ``bbox``, the region a vision model is asked to read.

    The whole page is rendered and then cropped rather than rendering a clip: pdfium's clipped render and
    the full-page render round pixel edges differently, and the crop must line up with the boxes the web
    viewer draws from the same ``NormalizedBBox`` (:meth:`NormalizedBBox.to_pixels` on the same page image).
    A caller cropping several regions of one page renders it once with :func:`render_page` and calls
    :func:`crop_region` itself.
    """
    return crop_region(render_page(pdf_path, page_index, dpi=dpi), bbox, max_pixels=max_pixels)


def crop_region(page: Image.Image, bbox: NormalizedBBox, *, max_pixels: int | None = None) -> Image.Image:
    """Cut ``bbox`` out of a rendered page.

    ``max_pixels`` bounds the crop's area. Hosted vision endpoints resize a large image on their side with
    an algorithm nobody controls; shrinking here with Lanczos keeps that decision, and its effect on small
    tick labels, in this code. The aspect ratio is kept, and a crop already under the bound is left untouched.
    """
    x1, y1, x2, y2 = bbox.to_pixels(width_px=page.width, height_px=page.height)
    # A box at the page's far edge, or thinner than a pixel after rounding, must still yield an image: the
    # near edge is pulled inside the page and the far edge pushed at least one pixel past it.
    x1, y1 = min(x1, page.width - 1), min(y1, page.height - 1)
    x2, y2 = min(max(x2, x1 + 1), page.width), min(max(y2, y1 + 1), page.height)
    crop = page.crop((x1, y1, x2, y2))
    if max_pixels is not None and crop.width * crop.height > max_pixels:
        scale = math.sqrt(max_pixels / (crop.width * crop.height))
        size = (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
        crop = crop.resize(size, Image.Resampling.LANCZOS)
    return crop


def png_bytes(image: Image.Image) -> bytes:
    """The PNG encoding of ``image``: what a vision request carries, and so what its cache key digests."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
