"""MinerU native output (``*_content_list.json``, v1 flat structure) -> ParsedArtifact.

Shape of each content_list entry (MinerU 3.x pipeline backend)::

    {"type": "text",  "page_idx": 0, "bbox": [x1, y1, x2, y2], "text": "...", "text_level": 1}
    {"type": "image", "page_idx": 0, "bbox": [...], "img_path": "images/xxx.jpg",
     "image_caption": ["Figure 1. ..."], "image_footnote": []}
    {"type": "table", "page_idx": 0, "bbox": [...], "img_path": "...", "table_body": "<table>…</table>",
     "table_caption": ["Table 1. ..."], "table_footnote": []}
    {"type": "equation", "page_idx": 0, "bbox": [...], "text": "E = mc^2", "text_format": "latex"}

Coordinates are **integer per-mille of the page** (``int(x * 1000 / page_width)``); they are
divided by 1000 when building a :class:`NormalizedBBox`. ``page_idx`` restarts from 0 when the
runner crops with ``--start-page``, so the real page number needs ``meta.source.page_offset``
added back.

MinerU attaches figure/table captions under the figure/table entry itself, with no bbox of their
own. Here we split them out into separate ``caption`` blocks that **share** the figure/table's
bbox: sample IDs and test conditions often appear only in figure/table captions, and extraction
must be able to retrieve them, while a bbox that merely "points at the whole table" is precise
enough for crop review.
"""

from __future__ import annotations

import json
from typing import Any

from paperfacts.adapters.base import BlockCollector, assemble_artifact, native_file
from paperfacts.models.artifact import BlockType, DocumentInput, ParsedArtifact
from paperfacts.models.geometry import DocumentGeometry, NormalizedBBox
from paperfacts.models.raw_output import RawParseOutput

# MinerU content_list's type field -> unified block type. Anything not in this table is marked
# unknown and warned about.
TYPE_MAP: dict[str, BlockType] = {
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
# Attached text fields under a figure/table entry -> the native label name used when splitting them out
ATTACHED_TEXT_FIELDS: tuple[str, ...] = ("image_caption", "image_footnote", "table_caption", "table_footnote")


def convert(raw: RawParseOutput, document: DocumentInput, geometry: DocumentGeometry) -> ParsedArtifact:
    """Convert one MinerU run's native output into the unified artifact."""
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
            type=_block_type(collector, item, raw_type),
            content=_content_of(item, raw_type),
            raw_label=raw_type,
        )
        # Figure/table captions and footnotes: split into separate caption blocks sharing the same bbox.
        for field in ATTACHED_TEXT_FIELDS:
            for text in _attached_texts(item.get(field)):
                collector.add(page=page, bbox=bbox, type="caption", content=text, raw_label=field)

    return assemble_artifact(collector, raw=raw, geometry=geometry)


def _block_type(collector: BlockCollector, item: dict[str, Any], raw_type: str) -> BlockType:
    # MinerU uses text_level >= 1 to mark a heading (1 = top-level heading).
    if raw_type == "text" and int(item.get("text_level") or 0) >= 1:
        return "title"
    mapped = TYPE_MAP.get(raw_type)
    if mapped is None:
        collector.note_unknown_label(raw_type)
        return "unknown"
    return mapped


def _attached_texts(value: Any) -> list[str]:
    """The caption / footnote field is normally a list of strings; defensively also accept a
    single string — a bare string must never be iterated character-by-character into blocks."""
    if value is None:
        return []
    entries = [value] if isinstance(value, str) else list(value) if isinstance(value, list | tuple) else []
    return [text for text in (str(entry).strip() for entry in entries) if text]


def _content_of(item: dict[str, Any], raw_type: str) -> str:
    """Get the "body" content for an entry by its type: tables use the HTML, images use the
    image path, lists join their items, and everything else uses text."""
    if raw_type == "table":
        return str(item.get("table_body") or "")
    if raw_type in ("image", "chart"):
        return str(item.get("img_path") or "")
    if raw_type == "list":
        # MinerU sometimes gives only text (no list_items) for short lists; fall back to text.
        items = item.get("list_items")
        return "\n".join(str(entry) for entry in items) if items else str(item.get("text") or "")
    return str(item.get("text") or "")
