"""PaddleOCR-VL native output (one ``save_to_json`` result per page) -> ParsedArtifact.

Shape of each page's JSON (PaddleOCR 3.x ``PaddleOCRVL``'s ``res``; the HTTP interface's
``prunedResult`` is isomorphic, just without input_path / page_index)::

    {"input_path": "...", "page_index": null,
     "parsing_res_list": [
        {"block_id": 0, "block_order": 1, "block_label": "doc_title",
         "block_bbox": [x1, y1, x2, y2], "block_content": "..."},
        ...]}

Coordinates are **pixels on the rendered page image**, and must be divided by that page's actual
pixel size recorded in meta.json (``width_px`` / ``height_px``). The page number is not taken
from the JSON's ``page_index`` (we send one page image at a time, so it is always null) but from
``meta.pages[].index``.

``block_label`` comes from PP-DocLayout's label vocabulary, much finer-grained than our unified
types; see :data:`LABEL_MAP` for the mapping. Anything that doesn't map is marked ``unknown``
while keeping the original label in ``raw_label``, so the table can be extended later.
"""

from __future__ import annotations

import json
from typing import Any

from paperfacts.adapters.base import BlockCollector, assemble_artifact, native_path
from paperfacts.models.artifact import BlockType, DocumentInput, ParsedArtifact
from paperfacts.models.geometry import DocumentGeometry, NormalizedBBox
from paperfacts.models.raw_output import PageMeta, RawParseOutput

# PP-DocLayout / PaddleOCR-VL's block_label -> unified block type.
LABEL_MAP: dict[str, BlockType] = {
    # Body text
    "text": "text",
    "abstract": "text",
    "content": "text",
    "reference": "text",
    "reference_content": "text",
    "footnote": "text",
    "aside_text": "text",
    "algorithm": "text",
    "list": "text",
    # Headings
    "doc_title": "title",
    "paragraph_title": "title",
    # Figures / charts
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "seal": "figure",
    "table": "table",
    # Figure/table captions
    "figure_title": "caption",
    "table_title": "caption",
    "chart_title": "caption",
    "image_caption": "caption",
    "table_caption": "caption",
    "vision_footnote": "caption",
    # Formulas
    "formula": "formula",
    "display_formula": "formula",
    "inline_formula": "formula",
    "formula_number": "formula",
    # Headers/footers etc.: kept but not used for extraction
    "header": "unknown",
    "footer": "unknown",
    "header_image": "unknown",
    "footer_image": "unknown",
    "number": "unknown",
    "page_number": "unknown",
}


def convert(raw: RawParseOutput, document: DocumentInput, geometry: DocumentGeometry) -> ParsedArtifact:
    """Convert one PaddleOCR-VL run's native output (per-page JSON) into the unified artifact."""
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
            block_type = LABEL_MAP.get(label)
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

    return assemble_artifact(collector, raw=raw, geometry=geometry)


def _rendered_page(page_meta: PageMeta) -> tuple[int, int, str]:
    """Validate the page-level invariant explicitly, outside the block loop: missing render
    info is a contract violation and must not be disguised as "every block on this page
    happened to be a bad box"."""
    if page_meta.width_px is None or page_meta.height_px is None or page_meta.json_path is None:
        raise ValueError(
            f"meta.json pages[{page_meta.index}] is missing width_px/height_px/json: "
            "is this really PaddleOCR-VL runner output?"
        )
    return page_meta.width_px, page_meta.height_px, page_meta.json_path


def _unwrap(data: dict[str, Any]) -> dict[str, Any]:
    """``save_to_json`` may wrap the content in ``{"res": {...}}``; the HTTP prunedResult has
    no such wrapper."""
    inner = data.get("res")
    if isinstance(inner, dict) and "parsing_res_list" in inner:
        return inner
    return data


def _ordered(items: Any) -> list[dict[str, Any]]:
    """Sort by the reading order Paddle provides: blocks with ``block_order`` None (not part of
    the ordering) sort last, then break ties by block_id."""

    def key(item: dict[str, Any]) -> tuple[int, int, int]:
        order = item.get("block_order")
        block_id = item.get("block_id")
        return (
            1 if order is None else 0,
            int(order) if order is not None else 0,
            int(block_id) if block_id is not None else 0,
        )

    return sorted((item for item in items if isinstance(item, dict)), key=key)
