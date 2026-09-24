"""Native parser output -> :class:`ParsedArtifact`, one pure function per backend.

An adapter only reads files: no model, no network, so a hand-written fixture tests it completely. A bad box
(zero area, flipped) never fails the whole document: it is skipped with a warning and counted. A page-level
contract violation (missing pixel size in ``meta.json``) is raised, not skipped.

:func:`render_markdown` turns blocks into Markdown with ``<!-- source: id -->`` markers. The same rendering
is written to ``parsed/<backend>.md`` for eyeballing and, filtered, is what the extraction model reads.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from paperfacts.models import (
    Backend,
    BlockType,
    DocumentGeometry,
    DocumentInput,
    NormalizedBBox,
    PageMeta,
    ParsedArtifact,
    RawParseOutput,
    SourceBlock,
    make_source_id,
)

logger = logging.getLogger(__name__)


def render_markdown(blocks: Iterable[SourceBlock]) -> str:
    """Each block preceded by its source marker, pages separated by a page marker."""
    lines: list[str] = []
    page: int | None = None
    for block in blocks:
        if block.page != page:
            page = block.page
            lines.append(f"<!-- page: {page} -->\n")
        lines.append(f"<!-- source: {block.source_id} -->\n{block.content}\n")
    return "\n".join(lines)


def convert(raw: RawParseOutput, document: DocumentInput, geometry: DocumentGeometry) -> ParsedArtifact:
    """Convert one parser's native output into the unified artifact."""
    if raw.backend == "mineru":
        return convert_mineru(raw, document, geometry)
    if raw.backend == "paddleocr_vl":
        return convert_paddle(raw, document, geometry)
    raise ValueError(f"no adapter for backend={raw.backend!r}")


# ---- Shared skeleton -----------------------------------------------------------------------------------


class BlockCollector:
    """Accumulates blocks per page, assigning reading order and ``source_id``; records skips and unknown labels."""

    def __init__(self, document_id: str, backend: Backend) -> None:
        self.document_id = document_id
        self.backend = backend
        self._blocks: list[SourceBlock] = []
        self._next_order: dict[int, int] = {}
        self.skipped: list[str] = []
        self.unknown_labels: dict[str, int] = {}

    def skip(self, *, page: int, raw_label: str | None, reason: str) -> None:
        entry = f"page={page} label={raw_label}: {reason}"
        self.skipped.append(entry)
        logger.warning("skip block backend=%s %s", self.backend, entry)

    def add(
        self,
        *,
        page: int,
        bbox: NormalizedBBox,
        type: BlockType,
        content: str,
        raw_label: str | None,
        raw_backend_id: str | None = None,
        confidence: float | None = None,
    ) -> SourceBlock:
        order = self._next_order.get(page, 0)
        self._next_order[page] = order + 1
        block = SourceBlock(
            source_id=make_source_id(self.backend, page, order),
            document_id=self.document_id,
            backend=self.backend,
            page=page,
            order=order,
            bbox=bbox,
            type=type,
            content=content,
            raw_label=raw_label,
            raw_backend_id=raw_backend_id,
            confidence=confidence,
        )
        self._blocks.append(block)
        return block

    def note_unknown_label(self, label: str) -> None:
        """Warn once per label so the log is not flooded."""
        if label not in self.unknown_labels:
            logger.warning("unknown block label backend=%s label=%r -> unknown", self.backend, label)
        self.unknown_labels[label] = self.unknown_labels.get(label, 0) + 1

    def artifact(self, raw: RawParseOutput, geometry: DocumentGeometry) -> ParsedArtifact:
        blocks = tuple(sorted(self._blocks, key=lambda b: (b.page, b.order)))
        artifact = ParsedArtifact(
            document_id=self.document_id,
            backend=raw.backend,
            backend_version=raw.backend_version,
            pages=geometry.pages,
            blocks=blocks,
            raw_output_dir=raw.out_dir,
        )
        logger.info(
            "adapted backend=%s doc=%s blocks=%d skipped=%d types=%s",
            artifact.backend,
            artifact.document_id[:16],
            len(blocks),
            len(self.skipped),
            artifact.type_counts(),
        )
        return artifact


def native_path(raw: RawParseOutput, relative: str) -> Path:
    path = raw.out_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"{raw.backend} native output is missing file: {path}")
    return path


def native_file(raw: RawParseOutput, key: str) -> Path:
    """A native file by its key in ``meta.files``; a missing key means this is not the matching runner's output."""
    relative = raw.meta.files.get(key)
    if relative is None:
        raise FileNotFoundError(f"{raw.backend} meta.json files has no {key!r}: {sorted(raw.meta.files)}")
    return native_path(raw, relative)


# ---- MinerU --------------------------------------------------------------------------------------------
#
# ``*_content_list.json`` entries (MinerU 3.x pipeline backend)::
#
#     {"type": "text",  "page_idx": 0, "bbox": [x1, y1, x2, y2], "text": "...", "text_level": 1}
#     {"type": "image", "page_idx": 0, "bbox": [...], "img_path": "...", "image_caption": [...], "image_footnote": []}
#     {"type": "table", "page_idx": 0, "bbox": [...], "table_body": "<table>…</table>", "table_caption": [...]}
#     {"type": "equation", "page_idx": 0, "bbox": [...], "text": "E = mc^2", "text_format": "latex"}
#
# Coordinates are integer per-mille of the page. ``page_idx`` restarts from 0 after ``--start-page``, so the
# real page needs ``meta.source.page_offset``. Captions hang under their figure/table with no bbox of their
# own; they become separate ``caption`` blocks sharing the parent's bbox, because sample ids and conditions
# often appear only there.

MINERU_TYPES: dict[str, BlockType] = {
    "text": "text",
    "list": "text",
    "code": "text",
    "algorithm": "text",
    "ref_text": "text",
    "aside_text": "text",
    "page_footnote": "text",
    "table": "table",
    "image": "figure",
    "chart": "figure",
    "equation": "formula",
    "interline_equation": "formula",
    "header": "unknown",
    "footer": "unknown",
    "page_number": "unknown",
    "discarded": "unknown",
}
# MinerU 3.x files a plotted chart as type "chart" with its own chart_caption / chart_footnote keys. Missing
# them silently cost lane A about 70% of the corpus's figure captions while lane B kept them.
MINERU_ATTACHED_TEXT: tuple[str, ...] = (
    "image_caption",
    "image_footnote",
    "chart_caption",
    "chart_footnote",
    "table_caption",
    "table_footnote",
)


def convert_mineru(raw: RawParseOutput, document: DocumentInput, geometry: DocumentGeometry) -> ParsedArtifact:
    items: list[dict[str, Any]] = json.loads(native_file(raw, "content_list").read_text(encoding="utf-8"))
    offset = raw.meta.source.page_offset
    collector = BlockCollector(document.document_id, "mineru")

    for item in items:
        raw_type = str(item.get("type", "unknown"))
        page = int(item.get("page_idx") or 0) + offset
        try:
            bbox = NormalizedBBox.from_thousandths(item.get("bbox") or ())
        except ValueError as exc:
            collector.skip(page=page, raw_label=raw_type, reason=str(exc))
            continue
        collector.add(
            page=page,
            bbox=bbox,
            type=_mineru_type(collector, item, raw_type),
            content=_mineru_content(item, raw_type),
            raw_label=raw_type,
        )
        for field in MINERU_ATTACHED_TEXT:
            for text in _attached_texts(item.get(field)):
                collector.add(page=page, bbox=bbox, type="caption", content=text, raw_label=field)

    return collector.artifact(raw, geometry)


def _mineru_type(collector: BlockCollector, item: dict[str, Any], raw_type: str) -> BlockType:
    if raw_type == "text" and int(item.get("text_level") or 0) >= 1:
        return "title"
    mapped = MINERU_TYPES.get(raw_type)
    if mapped is None:
        collector.note_unknown_label(raw_type)
        return "unknown"
    return mapped


def _attached_texts(value: Any) -> list[str]:
    """Normally a list of strings; a bare string must never be iterated character by character."""
    if value is None:
        return []
    entries = [value] if isinstance(value, str) else list(value) if isinstance(value, list | tuple) else []
    return [text for text in (str(entry).strip() for entry in entries) if text]


def _mineru_content(item: dict[str, Any], raw_type: str) -> str:
    if raw_type == "table":
        return str(item.get("table_body") or "")
    if raw_type in ("image", "chart"):
        return str(item.get("img_path") or "")
    if raw_type == "list":
        items = item.get("list_items")
        return "\n".join(str(entry) for entry in items) if items else str(item.get("text") or "")
    return str(item.get("text") or "")


# ---- PaddleOCR-VL ----------------------------------------------------------------------------------------
#
# One ``save_to_json`` result per page (the HTTP ``prunedResult`` is the same minus input_path/page_index)::
#
#     {"parsing_res_list": [{"block_id": 0, "block_order": 1, "block_label": "doc_title",
#                            "block_bbox": [x1, y1, x2, y2], "block_content": "..."}, ...]}
#
# Coordinates are pixels of the rendered page image, divided by the actual size recorded in meta.json. The
# page number comes from ``meta.pages[].index``, never from the JSON (we send one page at a time).

PADDLE_LABELS: dict[str, BlockType] = {
    "text": "text",
    "abstract": "text",
    "content": "text",
    "reference": "text",
    "reference_content": "text",
    "footnote": "text",
    "aside_text": "text",
    "algorithm": "text",
    "list": "text",
    "doc_title": "title",
    "paragraph_title": "title",
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "seal": "figure",
    "table": "table",
    "figure_title": "caption",
    "table_title": "caption",
    "chart_title": "caption",
    "image_caption": "caption",
    "table_caption": "caption",
    "vision_footnote": "caption",
    "formula": "formula",
    "display_formula": "formula",
    "inline_formula": "formula",
    "formula_number": "formula",
    "header": "unknown",
    "footer": "unknown",
    "header_image": "unknown",
    "footer_image": "unknown",
    "number": "unknown",
    "page_number": "unknown",
}


def convert_paddle(raw: RawParseOutput, document: DocumentInput, geometry: DocumentGeometry) -> ParsedArtifact:
    collector = BlockCollector(document.document_id, "paddleocr_vl")

    for page_meta in raw.meta.pages:
        page = page_meta.index
        width_px, height_px, json_relative = _rendered_page(page_meta)
        result = _unwrap(json.loads(native_path(raw, json_relative).read_text(encoding="utf-8")))

        for item in _ordered(result.get("parsing_res_list") or ()):
            label = str(item.get("block_label", "unknown"))
            try:
                bbox = NormalizedBBox.from_pixels(item.get("block_bbox") or (), width_px=width_px, height_px=height_px)
            except ValueError as exc:
                collector.skip(page=page, raw_label=label, reason=str(exc))
                continue
            block_type = PADDLE_LABELS.get(label)
            if block_type is None:
                collector.note_unknown_label(label)
                block_type = "unknown"
            block_id = item.get("block_id")
            collector.add(
                page=page,
                bbox=bbox,
                type=block_type,
                content=str(item.get("block_content") or "").strip(),
                raw_label=label,
                raw_backend_id=None if block_id is None else str(block_id),
            )

    return collector.artifact(raw, geometry)


def _rendered_page(page_meta: PageMeta) -> tuple[int, int, str]:
    """Missing render info is a contract violation, not "every block on this page was a bad box"."""
    if page_meta.width_px is None or page_meta.height_px is None or page_meta.json_path is None:
        raise ValueError(
            f"meta.json pages[{page_meta.index}] is missing width_px/height_px/json: "
            "is this really PaddleOCR-VL runner output?"
        )
    return page_meta.width_px, page_meta.height_px, page_meta.json_path


def _unwrap(data: dict[str, Any]) -> dict[str, Any]:
    """``save_to_json`` may wrap the page in ``{"res": {...}}``; the HTTP result does not."""
    inner = data.get("res")
    if isinstance(inner, dict) and "parsing_res_list" in inner:
        return inner
    return data


def _ordered(items: Any) -> list[dict[str, Any]]:
    """Paddle's reading order; blocks without ``block_order`` sort last, ties broken by ``block_id``."""

    def key(item: dict[str, Any]) -> tuple[int, int, int]:
        order = item.get("block_order")
        block_id = item.get("block_id")
        return (
            1 if order is None else 0,
            int(order) if order is not None else 0,
            int(block_id) if block_id is not None else 0,
        )

    return sorted((item for item in items if isinstance(item, dict)), key=key)
