"""Constructors for test objects: blank PDFs, SourceBlocks, and faked parser native-output directories.

This is where the "never touch a real parser" ground rule for unit tests is actually
implemented: PDFs are generated on the fly with pypdfium2, and parser native output is
hand-assembled to match the meta.json contract in ``runners/*.py``, so tests depend on none of
mineru / paddleocr / torch / model weights / the network.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from paperfacts.models import (
    META_FILENAME,
    Backend,
    BlockType,
    DocumentGeometry,
    DocumentInput,
    NormalizedBBox,
    RawParseOutput,
    SourceBlock,
    make_source_id,
)
from paperfacts.storage import raw_layout

# Page sizes (PDF points) for the two-page test PDF. Deliberately two different sizes, so a bug
# like "used the wrong page's geometry" shows up immediately as a value mismatch instead of
# accidentally passing.
PAGE_SIZES_PT: tuple[tuple[float, float], ...] = ((595.0, 842.0), (612.0, 792.0))

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"

# Placeholder constants for constructing a SourceBlock. document_id must be 64 hex chars; the
# content itself is meaningless.
DOC_ID = "a" * 64
DEFAULT_BBOX = NormalizedBBox(x1=0.1, y1=0.1, x2=0.9, y2=0.4)


# ---- PDF ----------------------------------------------------------------------------


def make_blank_pdf(path: Path, sizes: Sequence[tuple[float, float]] = PAGE_SIZES_PT) -> Path:
    """Generate a blank multi-page PDF on the fly with pypdfium2; never reference the
    copyrighted PDFs under template_files/."""
    document = pdfium.PdfDocument.new()
    try:
        for width_pt, height_pt in sizes:
            document.new_page(width_pt, height_pt)
        path.parent.mkdir(parents=True, exist_ok=True)
        document.save(str(path))
    finally:
        document.close()
    return path


# ---- SourceBlock --------------------------------------------------------------------


def make_block(
    *,
    page: int = 0,
    order: int = 0,
    type: BlockType = "text",
    content: str = "body text",
    backend: Backend = "mineru",
    bbox: NormalizedBBox = DEFAULT_BBOX,
    document_id: str = DOC_ID,
) -> SourceBlock:
    """A SourceBlock whose source_id is self-consistent with (backend, page, order)."""
    return SourceBlock(
        source_id=make_source_id(backend, page, order),
        document_id=document_id,
        backend=backend,
        page=page,
        order=order,
        bbox=bbox,
        type=type,
        content=content,
    )


# ---- Faked parser native output -------------------------------------------------------


class RawOutputFactory:
    """Fake the artifacts of one parser run inside tmp_path: native files + meta.json.

    The meta.json fields are copied verbatim from ``build_meta`` in
    ``runners/mineru_runner.py`` / ``runners/paddle_runner.py`` (a runner never imports the main
    package, so the contract can only be pinned by writing it twice and cross-validating).
    """

    def __init__(self, base_dir: Path, document: DocumentInput, geometry: DocumentGeometry) -> None:
        self.base_dir = base_dir
        self.document = document
        self.geometry = geometry

    def mineru(
        self,
        content_list: Iterable[Mapping[str, Any]],
        *,
        page_range: Sequence[int | None] = (0, None),
        parser_version: str = "3.4.5",
        parser_name: str = "mineru",
        dir_name: str = "raw_mineru",
        extra_meta: Mapping[str, Any] | None = None,
    ) -> RawParseOutput:
        out_dir = self.base_dir / dir_name
        native_dir = raw_layout.mineru_native_dir(out_dir)
        native_dir.mkdir(parents=True, exist_ok=True)
        content_path = raw_layout.mineru_native_file(native_dir, "_content_list.json")
        content_path.write_text(json.dumps(list(content_list), ensure_ascii=False), encoding="utf-8")
        middle_path = raw_layout.mineru_native_file(native_dir, "_middle.json")
        middle_path.write_text("{}", encoding="utf-8")

        meta: dict[str, Any] = {
            "parser": parser_name,
            "parser_version": parser_version,
            "backend": "pipeline",
            "source": self._source_meta(page_range),
            "pages": [page.model_dump() for page in self.geometry.pages],
            "files": {
                "content_list": _relative(content_path, out_dir),
                "middle_json": _relative(middle_path, out_dir),
            },
            "runner": {"script": "runners/mineru_runner.py"},
        }
        if extra_meta:
            meta.update(extra_meta)
        return self._finish(out_dir, meta, "mineru")

    def paddle(
        self,
        pages: Iterable[Mapping[str, Any]],
        *,
        page_range: Sequence[int | None] = (0, None),
        parser_version: str = "3.7.0",
        dir_name: str = "raw_paddle",
        render_dpi: int = 200,
    ) -> RawParseOutput:
        """Each entry in ``pages`` looks like
        ``{"index": 0, "width_px": .., "height_px": .., "data": {...}}``.

        Optional key ``omit``: field names to delete from that page's meta record, used to
        construct a "runner output is incomplete" scenario.
        """
        out_dir = self.base_dir / dir_name
        pages_dir = raw_layout.paddle_pages_dir(out_dir)
        pages_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        for page in pages:
            index = int(page["index"])
            json_path = raw_layout.paddle_page_json(pages_dir, index)
            json_path.write_text(json.dumps(page["data"], ensure_ascii=False), encoding="utf-8")
            geo = self.geometry.page(index) if index < self.geometry.page_count else self.geometry.page(0)
            record: dict[str, Any] = {
                "index": index,
                "width_pt": geo.width_pt,
                "height_pt": geo.height_pt,
                "width_px": int(page["width_px"]),
                "height_px": int(page["height_px"]),
                "image": _relative(raw_layout.paddle_page_image(pages_dir, index), out_dir),
                "json": _relative(json_path, out_dir),
                "markdown_dir": _relative(raw_layout.paddle_page_markdown_dir(pages_dir, index), out_dir),
            }
            for key in page.get("omit", ()):
                record.pop(key, None)
            records.append(record)

        meta: dict[str, Any] = {
            "parser": "paddleocr_vl",
            "parser_version": parser_version,
            "framework": {},
            "render_dpi": render_dpi,
            "vl_backend": "in-process",
            "source": self._source_meta(page_range),
            "pages": records,
            "files": {"pages_dir": raw_layout.PADDLE_PAGES_DIRNAME},
            "runner": {"script": "runners/paddle_runner.py"},
        }
        return self._finish(out_dir, meta, "paddleocr_vl")

    def _source_meta(self, page_range: Sequence[int | None]) -> dict[str, Any]:
        return {
            "pdf": str(self.document.pdf_path),
            "sha256": self.document.sha256,
            "page_count": self.geometry.page_count,
            "parsed_page_count": self.geometry.page_count,
            "page_range": list(page_range),
        }

    @staticmethod
    def _finish(out_dir: Path, meta: Mapping[str, Any], backend: Backend) -> RawParseOutput:
        # Write meta.json last, just like a runner does: its existence is what signals this
        # output is complete.
        (out_dir / META_FILENAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return RawParseOutput.load(out_dir, backend)


def paddle_page_entry(index: int, data: Mapping[str, Any], size: tuple[int, int], **extra: Any) -> dict[str, Any]:
    """One page's input argument for ``RawOutputFactory.paddle``."""
    return {"index": index, "width_px": size[0], "height_px": size[1], "data": data, **extra}


def _relative(path: Path, out_dir: Path) -> str:
    return str(path.relative_to(out_dir))
