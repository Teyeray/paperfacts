"""PDF reading and page rendering: the only module that calls pypdfium2.

The runners use the same library and the same rendering call, so the page geometry a parser records and the
geometry used here for overlays and the web viewer agree numerically.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from paperfacts.models import DocumentGeometry, PageGeometry
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
