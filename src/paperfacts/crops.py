"""Page regions rendered for a vision model, and kept on disk so a box is rendered once.

A region is the part of a page a stage wants a vision model to look at: the blocks a value was cited from,
a chart's panel, a whole table. Two stages, and two values within one stage, keep asking for the same box --
two numbers cited from one table share a region, and a re-run asks for every region again -- so the PNG is
content-addressed by page, box, DPI and pixel limit under the document's ``crops/`` directory and answered
from there.
Experiments may also bind the actual PDF byte digest: a merged SI document's logical identity is separate
from its saved PDF bytes, so the same logical document must not reuse another merge's image.

The digest travels with the crop because the LLM cache keys a vision request on the image's sha256 rather
than its base64 (see :mod:`paperfacts.llm`): a stored verdict names the digest, and the file beside it is the
exact bytes a reviewer can open.

Rendering goes through :mod:`paperfacts.pdf` and its process-wide PDFium lock, so any number of worker
threads may ask at once. Kept out of the stages that use it: where a crop lives is not what a model was
asked, so nothing here belongs in a stage's cache key.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from paperfacts.models import NormalizedBBox
from paperfacts.pdf import crop_region, png_bytes, render_page, render_region
from paperfacts.storage import DataLayout, write_bytes_atomic


@dataclass(frozen=True)
class Region:
    """One page's worth of box to render. ``source_ids`` and ``context_ids`` are carried through to the
    stored :class:`RegionCrop` so a verdict can name the blocks the picture was built from, and the
    neighbours it was widened by, apart from each other."""

    page: int
    bbox: NormalizedBBox
    source_ids: tuple[str, ...] = ()
    context_ids: tuple[str, ...] = ()


class RegionCrop(BaseModel):
    """The image a model was shown: which page, which box, at what resolution, and its digest.

    The digest is what the LLM cache keyed the request on, so a stored reading can be traced to the exact
    bytes that produced it, and ``path`` is where those bytes are on disk for a reviewer to look at.
    """

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=0)
    bbox: NormalizedBBox
    dpi: int = Field(gt=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    source_ids: tuple[str, ...] = Field(default=(), description="the cited blocks the box is the union of")
    context_ids: tuple[str, ...] = Field(
        default=(), description="neighbouring blocks a sliding window added around the cited ones"
    )
    image_sha256: str = Field(min_length=64, max_length=64)
    path: str = Field(description="file name under the document's crops/ directory")


def bbox_key(bbox: NormalizedBBox) -> str:
    """A file-name-safe spelling of a box, precise to a ten-thousandth of the page."""
    return f"{bbox.x1:.4f}-{bbox.y1:.4f}-{bbox.x2:.4f}-{bbox.y2:.4f}"


def png_size(data: bytes) -> tuple[int, int]:
    """Width and height from the PNG header, so a cached file need not be decoded to describe it."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CropStore:
    """Renders regions once and keeps the PNGs under ``crops/``.

    Both levels of repetition are answered without work: a second ask within one run comes from memory, and
    a later run's ask comes from the file the first run wrote. A caller cropping many boxes from *one* page
    should pass ``page_cache=True``, which keeps the rendered page instead of rendering it per region; a
    caller whose boxes are scattered over a document should leave it off so the images are not all held.
    """

    def __init__(
        self,
        layout: DataLayout,
        document_id: str,
        pdf_path: Path,
        *,
        dpi: int,
        max_pixels: int | None = None,
        page_cache: bool = False,
        source_pdf_sha256: str | None = None,
    ) -> None:
        self.layout = layout
        self.document_id = document_id
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.max_pixels = max_pixels
        self.source_pdf_sha256 = source_pdf_sha256
        self._lock = threading.Lock()
        self._memory: dict[Path, bytes] = {}
        self._pages: dict[int, Image.Image] | None = {} if page_cache else None

    def crop(self, region: Region) -> tuple[bytes, RegionCrop]:
        """The PNG bytes for ``region``, and the record of what was shown."""
        data = self.png(region.page, region.bbox)
        width, height = png_size(data)
        return data, RegionCrop(
            page=region.page,
            bbox=region.bbox,
            dpi=self.dpi,
            width_px=width,
            height_px=height,
            source_ids=region.source_ids,
            context_ids=region.context_ids,
            image_sha256=sha256_hex(data),
            path=self.path_for(region.page, region.bbox).name,
        )

    def png(self, page: int, bbox: NormalizedBBox) -> bytes:
        """The PNG bytes for one box, from memory, from disk, or rendered and written."""
        path = self.path_for(page, bbox)
        with self._lock:
            data = self._memory.get(path)
            if data is None and path.is_file():
                data = path.read_bytes()
            if data is None:
                data = png_bytes(self._render(page, bbox))
                write_bytes_atomic(path, data)
            self._memory[path] = data
            return data

    def path_for(self, page: int, bbox: NormalizedBBox) -> Path:
        return self.layout.crop_path(
            self.document_id,
            page,
            bbox_key(bbox),
            self.dpi,
            max_pixels=self.max_pixels,
            source_pdf_sha256=self.source_pdf_sha256,
        )

    def _render(self, page: int, bbox: NormalizedBBox) -> Image.Image:
        # Called under the lock, so the page cache needs no separate guard.
        if self._pages is None:
            return render_region(self.pdf_path, page, bbox, dpi=self.dpi, max_pixels=self.max_pixels)
        image = self._pages.get(page)
        if image is None:
            image = render_page(self.pdf_path, page, dpi=self.dpi)
            self._pages[page] = image
        return crop_region(image, bbox, max_pixels=self.max_pixels)
