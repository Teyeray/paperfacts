"""Page geometry and normalized bboxes — the one coordinate system for the whole system.

Design doc §7 mandates: every bbox a parser outputs must be unified in the adapter into
**page-normalized coordinates** (x, y in [0, 1], origin at the page's top-left corner); after
that, retrieval, alignment, cropping, and overlay never again care which parser a bbox came from.

Types turn this rule into something "structurally impossible to violate":

- :class:`NormalizedBBox`'s constructor only accepts coordinates within [0, 1] with top-left <
  bottom-right, otherwise it raises ``ValidationError`` directly;
- the **only** entry points from raw coordinates into normalized coordinates are the three
  ``from_*`` factory methods, each of which requires the geometry info needed for the conversion
  (per-mille scale, pixel size, page point size) — forgetting to pass it makes construction
  impossible;
- cropping / overlay only ever accept a :class:`NormalizedBBox`, never a bare number.

Quick reference for coordinate conventions:

=====================  =========================================================  ====================
Source                 Native unit                                                Entry point
=====================  =========================================================  ====================
MinerU content_list    Integer per-mille of the page (0-1000)                     ``from_thousandths``
PaddleOCR-VL           Pixels of the rendered page image (depends on render DPI)  ``from_pixels``
PDF native / cropping  PDF points (1 pt = 1/72 inch), from pypdfium2              ``from_points``
=====================  =========================================================  ====================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# MinerU content_list's bbox is ``int(x * 1000 / page_width)``, i.e. per-mille of the page.
THOUSANDTHS = 1000.0


class PageGeometry(BaseModel):
    """One page's physical size (PDF points). ``index`` is 0-based, consistent with both
    parsers' page numbering convention."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    width_pt: float = Field(gt=0)
    height_pt: float = Field(gt=0)


class DocumentGeometry(BaseModel):
    """Page-size table for the whole PDF. Read from the PDF by :func:`paperfacts.pdf.read_geometry`."""

    model_config = ConfigDict(frozen=True)

    pages: tuple[PageGeometry, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page(self, index: int) -> PageGeometry:
        """Get geometry by page index; raises ``IndexError`` when out of range, so the error
        surfaces at the call site instead of propagating as a wrong coordinate."""
        try:
            return self.pages[index]
        except IndexError as exc:
            raise IndexError(f"page {index} out of range: document has {self.page_count} pages") from exc


class NormalizedBBox(BaseModel):
    """Page-normalized bounding box: ``(x1, y1)`` top-left, ``(x2, y2)`` bottom-right, all
    within [0, 1]."""

    model_config = ConfigDict(frozen=True)

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_positive_area(self) -> Self:
        # A zero-area or flipped box is garbage output from the parser; better to error out here
        # and let the adapter skip it than let it flow downstream.
        if not (self.x1 < self.x2 and self.y1 < self.y2):
            raise ValueError(f"bbox must satisfy x1 < x2 and y1 < y2, got ({self.x1}, {self.y1}, {self.x2}, {self.y2})")
        return self

    # ---- Factory methods: the only entry point from raw coordinates to normalized coordinates ----

    @classmethod
    def from_thousandths(cls, box: Sequence[float]) -> Self:
        """MinerU content_list's 0-1000 per-mille coordinates."""
        x1, y1, x2, y2 = _four(box)
        return cls._from_scaled(x1, y1, x2, y2, scale_x=THOUSANDTHS, scale_y=THOUSANDTHS)

    @classmethod
    def from_pixels(cls, box: Sequence[float], *, width_px: int, height_px: int) -> Self:
        """Pixel coordinates on the rendered page image (PaddleOCR-VL). The **actual** image
        size must be passed in — it cannot be estimated from a formula."""
        if width_px <= 0 or height_px <= 0:
            raise ValueError(f"image size must be positive, got {width_px}x{height_px}")
        x1, y1, x2, y2 = _four(box)
        return cls._from_scaled(x1, y1, x2, y2, scale_x=width_px, scale_y=height_px)

    @classmethod
    def from_points(cls, box: Sequence[float], *, page: PageGeometry) -> Self:
        """PDF point coordinates (origin top-left)."""
        x1, y1, x2, y2 = _four(box)
        return cls._from_scaled(x1, y1, x2, y2, scale_x=page.width_pt, scale_y=page.height_pt)

    @classmethod
    def _from_scaled(cls, x1: float, y1: float, x2: float, y2: float, *, scale_x: float, scale_y: float) -> Self:
        """Scale proportionally into [0, 1], clamping slight overflow (parser artifacts like
        -1 or 1001) back to the boundary.

        Only overflow gets clamped, not ordering: a flipped or zero-area box still fails in the validator.
        """
        return cls(
            x1=_clamp01(x1 / scale_x),
            y1=_clamp01(y1 / scale_y),
            x2=_clamp01(x2 / scale_x),
            y2=_clamp01(y2 / scale_y),
        )

    # ---- Geometric operations ---------------------------------------------------------------

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_pixels(self, *, width_px: int, height_px: int) -> tuple[int, int, int, int]:
        """Convert back to an integer pixel box on an image of the given size (for overlay / cropping)."""
        return (
            round(self.x1 * width_px),
            round(self.y1 * height_px),
            round(self.x2 * width_px),
            round(self.y2 * height_px),
        )

    def union(self, other: NormalizedBBox) -> NormalizedBBox:
        """The minimal enclosing box of two boxes. Design doc §16: on same-page conflicts, take
        the union of the two bboxes before cropping."""
        return NormalizedBBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def padded(self, pad: float) -> NormalizedBBox:
        """Expand outward by ``pad`` on all four sides (normalized units), clamping to the page
        edge on overflow."""
        if pad < 0:
            raise ValueError("pad must not be negative")
        return NormalizedBBox(
            x1=_clamp01(self.x1 - pad),
            y1=_clamp01(self.y1 - pad),
            x2=_clamp01(self.x2 + pad),
            y2=_clamp01(self.y2 + pad),
        )


def _four(box: Sequence[float]) -> tuple[float, float, float, float]:
    """Split any four-element sequence into four floats; a wrong length errors immediately, so
    downstream code never gets None."""
    if len(box) != 4:
        raise ValueError(f"bbox needs 4 numbers, got {len(box)}: {box!r}")
    x1, y1, x2, y2 = (float(v) for v in box)
    return x1, y1, x2, y2


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))
