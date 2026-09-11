"""PDF reading and page rendering — the only module in the main package that touches PDF files directly.

Both operations are built on pypdfium2, the same library and rendering logic the two runners use,
so the page geometry a parser records and the page geometry used here for cropping/overlays come
from the same source and agree numerically.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from paperfacts.models.geometry import DocumentGeometry, PageGeometry
from paperfacts.storage.atomic import write_atomic
from paperfacts.storage.paths import overlay_page_name

PDF_POINTS_PER_INCH = 72
# PDFium is not thread-safe (the pypdfium2 docs say so explicitly). The web server renders pages
# from multiple worker threads and reads geometry from background task threads; concurrent calls
# corrupt pdfium's global state — in practice this shows up as every later open of the same PDF
# failing with "Data format error", recoverable only by restarting the process. So we take one
# process-wide lock and serialize all pdfium calls; a page renders in a few hundred milliseconds,
# so serializing is more than fast enough.
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
    """Read every page's size, in PDF points."""
    with _open(pdf_path) as document:
        pages = tuple(
            PageGeometry(index=index, width_pt=width_pt, height_pt=height_pt)
            for index in range(len(document))
            for width_pt, height_pt in (document[index].get_size(),)
        )
    return DocumentGeometry(pages=pages)


def render_page(pdf_path: Path, page_index: int, *, dpi: int) -> Image.Image:
    """Render the given page to an RGB image. ``dpi`` sets the pixel size: ``px = pt * dpi / 72``
    (pdfium rounds up).
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    with _open(pdf_path) as document:
        if not 0 <= page_index < len(document):
            raise IndexError(f"page index {page_index} out of range, document has {len(document)} pages")
        return document[page_index].render(scale=dpi / PDF_POINTS_PER_INCH).to_pil().convert("RGB")


def render_page_cached(pdf_path: Path, page_index: int, *, dpi: int, cache_dir: Path) -> Path:
    """Render and cache one page, returning the PNG path. Atomic write: the file's existence means
    the PNG is complete; an out-of-range page raises ``IndexError``.
    """
    path = cache_dir / overlay_page_name(page_index)
    if not path.is_file():
        image = render_page(pdf_path, page_index, dpi=dpi)
        write_atomic(path, lambda tmp: image.save(tmp, format="PNG"))
    return path
