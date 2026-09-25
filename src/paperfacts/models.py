"""The data that flows between stages: page geometry, provenance blocks, the unified parse artifact, and
the ``meta.json`` contract a parser runner writes.

Coordinates: every bbox is a :class:`NormalizedBBox` in ``[0, 1]`` page coordinates, origin top-left. Native
coordinates enter only through its ``from_*`` factories:

=====================  =========================================================  ====================
Source                 Native unit                                                Entry point
=====================  =========================================================  ====================
MinerU content_list    Integer per-mille of the page (0-1000)                     ``from_thousandths``
PaddleOCR-VL           Pixels of the rendered page image (depends on render DPI)  ``from_pixels``
PDF native             PDF points (1 pt = 1/72 inch), from pypdfium2              ``from_points``
=====================  =========================================================  ====================

Naming: ``document_id`` is the PDF's content sha256; ``source_id`` is ``{backend}_p{page}_b{order}`` with
0-based pages, the key that leads from an extracted value back to a page region.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paperfacts.storage import write_text_atomic

# The two parsers. The string form appears in source ids, directory names and meta.json.
Backend = Literal["mineru", "paddleocr_vl"]
BACKENDS: tuple[Backend, ...] = ("mineru", "paddleocr_vl")
# Unified block types. Native labels are finer; an adapter maps them down and keeps the original in
# ``raw_label``. Anything unmapped becomes ``unknown``, never dropped.
BlockType = Literal["text", "title", "formula", "table", "figure", "caption", "unknown"]
# MinerU content_list's bbox is ``int(x * 1000 / page_width)``.
THOUSANDTHS = 1000.0
# Written last by every parser, so its existence means the output directory is complete.
META_FILENAME = "meta.json"


# ---- Geometry --------------------------------------------------------------------------------------


class PageGeometry(BaseModel):
    """One page's physical size in PDF points; ``index`` is 0-based."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    width_pt: float = Field(gt=0)
    height_pt: float = Field(gt=0)


class DocumentGeometry(BaseModel):
    model_config = ConfigDict(frozen=True)

    pages: tuple[PageGeometry, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page(self, index: int) -> PageGeometry:
        try:
            return self.pages[index]
        except IndexError as exc:
            raise IndexError(f"page {index} out of range: document has {self.page_count} pages") from exc


class NormalizedBBox(BaseModel):
    """``(x1, y1)`` top-left, ``(x2, y2)`` bottom-right, all within [0, 1], positive area."""

    model_config = ConfigDict(frozen=True)

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_positive_area(self) -> Self:
        # A flipped or zero-area box is parser garbage; fail here so the adapter can skip it.
        if not (self.x1 < self.x2 and self.y1 < self.y2):
            raise ValueError(f"bbox must satisfy x1 < x2 and y1 < y2, got ({self.x1}, {self.y1}, {self.x2}, {self.y2})")
        return self

    @classmethod
    def from_thousandths(cls, box: Sequence[float]) -> Self:
        x1, y1, x2, y2 = _four(box)
        return cls._from_scaled(x1, y1, x2, y2, scale_x=THOUSANDTHS, scale_y=THOUSANDTHS)

    @classmethod
    def from_pixels(cls, box: Sequence[float], *, width_px: int, height_px: int) -> Self:
        """The **actual** rendered image size must be passed in; it cannot be derived from a formula."""
        if width_px <= 0 or height_px <= 0:
            raise ValueError(f"image size must be positive, got {width_px}x{height_px}")
        x1, y1, x2, y2 = _four(box)
        return cls._from_scaled(x1, y1, x2, y2, scale_x=width_px, scale_y=height_px)

    @classmethod
    def from_points(cls, box: Sequence[float], *, page: PageGeometry) -> Self:
        x1, y1, x2, y2 = _four(box)
        return cls._from_scaled(x1, y1, x2, y2, scale_x=page.width_pt, scale_y=page.height_pt)

    @classmethod
    def _from_scaled(cls, x1: float, y1: float, x2: float, y2: float, *, scale_x: float, scale_y: float) -> Self:
        # Slight overflow (-1, 1001) is a parser artefact and is clamped; ordering is not repaired.
        return cls(
            x1=_clamp01(x1 / scale_x),
            y1=_clamp01(y1 / scale_y),
            x2=_clamp01(x2 / scale_x),
            y2=_clamp01(y2 / scale_y),
        )

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
        return (
            round(self.x1 * width_px),
            round(self.y1 * height_px),
            round(self.x2 * width_px),
            round(self.y2 * height_px),
        )

    def union(self, other: NormalizedBBox) -> NormalizedBBox:
        return NormalizedBBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def padded(self, pad: float) -> NormalizedBBox:
        if pad < 0:
            raise ValueError("pad must not be negative")
        return NormalizedBBox(
            x1=_clamp01(self.x1 - pad),
            y1=_clamp01(self.y1 - pad),
            x2=_clamp01(self.x2 + pad),
            y2=_clamp01(self.y2 + pad),
        )


def _four(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"bbox needs 4 numbers, got {len(box)}: {box!r}")
    x1, y1, x2, y2 = (float(v) for v in box)
    return x1, y1, x2, y2


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


# ---- Document input and the unified artifact -------------------------------------------------------


class DocumentInput(BaseModel):
    """A PDF to process. ``document_id`` is the content sha256, so moving or renaming never reprocesses."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=64, max_length=64)
    pdf_path: Path
    sha256: str = Field(min_length=64, max_length=64)
    display_name: str | None = Field(
        default=None,
        description="Name to show a user. Web uploads are all stored as source.pdf, so the on-disk name is "
        "useless there; the real one comes from identity.json.",
    )

    @property
    def display_filename(self) -> str:
        """The filename a user should see, wherever a document is named in output."""
        return self.display_name or self.pdf_path.name

    @classmethod
    def from_path(cls, pdf_path: Path) -> Self:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
        digest = sha256_of_file(pdf_path)
        return cls(document_id=digest, pdf_path=pdf_path.resolve(), sha256=digest)


def sha256_of_file(path: Path) -> str:
    """Streamed sha256. ``runners/*.py`` carry the same implementation; the two must agree."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_source_id(backend: Backend, page: int, order: int) -> str:
    return f"{backend}_p{page}_b{order}"


class SourceBlock(BaseModel):
    """A content block traceable to a page region."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    document_id: str
    backend: Backend
    page: int = Field(ge=0, description="0-based page")
    order: int = Field(ge=0, description="in-page reading order")
    bbox: NormalizedBBox
    type: BlockType
    content: str
    raw_label: str | None = Field(default=None, description="the parser's native label, for debugging")
    raw_backend_id: str | None = Field(default=None, description="the parser's native id, if any")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ParsedArtifact(BaseModel):
    """One parser's unified output: provenance blocks plus page geometry."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    backend: Backend
    backend_version: str | None
    pages: tuple[PageGeometry, ...]
    blocks: tuple[SourceBlock, ...]
    raw_output_dir: Path | None = Field(default=None, description="directory holding the parser's native output")

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def block(self, source_id: str) -> SourceBlock:
        for candidate in self.blocks:
            if candidate.source_id == source_id:
                return candidate
        raise KeyError(f"{self.backend} artifact has no source_id={source_id!r}")

    def blocks_on_page(self, page: int) -> tuple[SourceBlock, ...]:
        return tuple(block for block in self.blocks if block.page == page)

    def content_hash(self) -> str:
        """sha256 over the blocks: what a citation resolves against, nothing that varies between runs.

        Source ids are positional, so a lane derived from one parse and read against another would cite
        whatever block now has that ordinal. Stored results record this hash to notice that.
        """
        blocks = [block.model_dump(mode="json") for block in self.blocks]
        return hashlib.sha256(json.dumps(blocks, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for block in self.blocks:
            counts[block.type] = counts.get(block.type, 0) + 1
        return dict(sorted(counts.items()))

    def write(self, path: Path) -> None:
        """Atomic: a run killed mid-write leaves the previous artifact, never a truncated one."""
        write_text_atomic(path, self.model_dump_json(indent=2))

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---- The meta.json contract with the runners -------------------------------------------------------
#
# Three writers: runners/mineru_runner.py, runners/paddle_runner.py (subprocesses that must not import this
# package) and the HTTP parsers in parsers.py. The runners cannot share code with this module, so the
# contract is pinned by validation: fixtures of real runner output in tests, and the --run-parser suite
# against a fresh meta.json. ``extra="allow"`` lets a runner record fields only it cares about.


class SourceMeta(BaseModel):
    """Which PDF was parsed, and which pages."""

    model_config = ConfigDict(frozen=True, extra="allow")

    pdf: str
    sha256: str = Field(min_length=64, max_length=64)
    page_count: int = Field(ge=0, description="total page count of the PDF")
    parsed_page_count: int = Field(ge=0, description="pages parsed this run, after --start-page/--end-page")
    page_range: tuple[int | None, int | None] = (0, None)

    @property
    def page_offset(self) -> int:
        """MinerU's ``page_idx`` restarts at 0 after ``--start-page``; the real page is ``page_idx + offset``."""
        return self.page_range[0] or 0


class PageMeta(BaseModel):
    """One page. Pixel size and per-page files are written only by the PaddleOCR-VL side, which renders pages."""

    model_config = ConfigDict(frozen=True, extra="allow", populate_by_name=True)

    index: int = Field(ge=0)
    width_pt: float = Field(gt=0)
    height_pt: float = Field(gt=0)
    width_px: int | None = Field(default=None, gt=0)
    height_px: int | None = Field(default=None, gt=0)
    image: str | None = None
    json_path: str | None = Field(default=None, alias="json")
    markdown_dir: str | None = None


class ParserMeta(BaseModel):
    """Top-level structure of ``meta.json``."""

    model_config = ConfigDict(frozen=True, extra="allow")

    parser: Backend
    parser_version: str = "unknown"
    source: SourceMeta
    pages: tuple[PageMeta, ...] = Field(default=(), description="the pages parsed this run, not the whole PDF")
    files: dict[str, str] = Field(default_factory=dict, description="native file paths relative to the output dir")
    runner: dict[str, Any] = Field(default_factory=dict, description="run environment, for troubleshooting only")


class RawParseOutput(BaseModel):
    """A handle to one parser run: the native output directory plus its validated ``meta.json``."""

    model_config = ConfigDict(frozen=True)

    backend: Backend
    out_dir: Path
    meta: ParserMeta
    cache_hit: bool = False

    @property
    def backend_version(self) -> str:
        return self.meta.parser_version

    @classmethod
    def load(cls, out_dir: Path, backend: Backend, *, cache_hit: bool = False) -> Self:
        """Read and validate ``out_dir/meta.json``; a shape mismatch raises ``ValidationError`` (a ``ValueError``)."""
        meta_path = out_dir / META_FILENAME
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} does not exist, parser output is incomplete")
        meta = ParserMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        if meta.parser != backend:
            raise ValueError(f"{meta_path} belongs to parser={meta.parser!r}, expected {backend!r}")
        return cls(backend=backend, out_dir=out_dir, meta=meta, cache_hit=cache_hit)
