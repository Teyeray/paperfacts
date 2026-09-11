"""Core data models for the parsing layer: document input, provenance blocks, unified parse artifact.

Corresponds to design doc §4-§6. No matter what the two parsers (MinerU, PaddleOCR-VL) produce
natively, their respective adapters converge on the same :class:`ParsedArtifact`; everything
downstream (retrieval, extraction, alignment, cropping) only ever deals with this one shape.

Naming conventions:

- ``document_id`` = SHA-256 of the PDF content, naturally idempotent and dedupe-friendly;
- ``source_id`` = ``{backend}_p{page}_b{order}``, e.g. ``mineru_p7_b12`` — the unique key
  between Markdown and a page region: ``source_id -> SourceBlock -> page + bbox -> original PDF``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from paperfacts.models.geometry import NormalizedBBox, PageGeometry

# Literal names of the two parsers. The string form shows up in source_id, directory names, and
# meta.json — changing it changes the contract.
Backend = Literal["mineru", "paddleocr_vl"]
BACKENDS: tuple[Backend, ...] = ("mineru", "paddleocr_vl")

# The unified block type (design doc §6). Native parser labels are finer-grained (Paddle has a
# dozen or so); the adapter maps them down to these few categories. Anything that doesn't map is
# marked ``unknown`` while keeping ``raw_label`` — never silently dropped.
BlockType = Literal["text", "title", "formula", "table", "figure", "caption", "unknown"]


class DocumentInput(BaseModel):
    """A PDF to be processed. ``document_id`` is simply the content SHA-256, so renaming or
    moving the same file never causes it to be reprocessed."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(min_length=64, max_length=64)
    pdf_path: Path
    sha256: str = Field(min_length=64, max_length=64)

    @classmethod
    def from_path(cls, pdf_path: Path) -> Self:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
        digest = sha256_of_file(pdf_path)
        return cls(document_id=digest, pdf_path=pdf_path.resolve(), sha256=digest)


def sha256_of_file(path: Path) -> str:
    """Stream the whole file to compute its SHA-256. runners/*.py has the same implementation;
    the two must stay consistent."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_source_id(backend: Backend, page: int, order: int) -> str:
    """Build a source_id of the form ``{backend}_p{page}_b{order}``."""
    return f"{backend}_p{page}_b{order}"


class SourceBlock(BaseModel):
    """A content block traceable back to a page region — what design doc §6 calls "one of the
    system's most important data structures".

    ``markdown_start`` / ``markdown_end`` are filled in when the Markdown with provenance
    markers is generated, guaranteeing ``artifact.markdown[start:end] == block.content``.
    """

    model_config = ConfigDict(frozen=True)

    source_id: str
    document_id: str
    backend: Backend
    page: int = Field(ge=0, description="page number, 0-based")
    order: int = Field(ge=0, description="in-page reading order")
    bbox: NormalizedBBox
    type: BlockType
    content: str
    raw_label: str | None = Field(default=None, description="parser's native block label, for debugging/statistics")
    raw_backend_id: str | None = Field(default=None, description="parser's native id (if any)")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    markdown_start: int | None = Field(default=None, ge=0)
    markdown_end: int | None = Field(default=None, ge=0)

    def with_markdown_span(self, start: int, end: int) -> SourceBlock:
        """Return a new copy with the Markdown span filled in (the model is frozen, so this
        never mutates in place)."""
        return self.model_copy(update={"markdown_start": start, "markdown_end": end})


class ParsedArtifact(BaseModel):
    """One parser's unified artifact: Markdown with provenance markers + list of provenance
    blocks + page geometry."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    backend: Backend
    backend_version: str | None
    pages: tuple[PageGeometry, ...]
    markdown: str
    blocks: tuple[SourceBlock, ...]
    raw_output_dir: Path | None = Field(default=None, description="directory holding the parser's native output")

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def block(self, source_id: str) -> SourceBlock:
        """Look up a block by source_id; raises ``KeyError`` if not found."""
        for candidate in self.blocks:
            if candidate.source_id == source_id:
                return candidate
        raise KeyError(f"{self.backend} artifact has no source_id={source_id!r}")

    def blocks_on_page(self, page: int) -> tuple[SourceBlock, ...]:
        return tuple(block for block in self.blocks if block.page == page)

    def type_counts(self) -> dict[str, int]:
        """Count of each block type, used in run logs and m1-run-notes."""
        counts: dict[str, int] = {}
        for block in self.blocks:
            counts[block.type] = counts.get(block.type, 0) + 1
        return dict(sorted(counts.items()))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
