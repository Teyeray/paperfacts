"""Draw a parser's block boxes back onto the page image, to check provenance by eye.

Rendering goes through :func:`paperfacts.pdf.render_page`, the same call the runners use, so if the boxes hug
the text here they will line up in the web viewer too.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw

from paperfacts.config import DEFAULT_OVERLAY_DPI
from paperfacts.models import BlockType, ParsedArtifact
from paperfacts.pdf import render_page
from paperfacts.storage import overlay_page_name

logger = logging.getLogger(__name__)

LINE_WIDTH = 2
# One colour per block type, shared by both parsers, so overlays compare side by side.
TYPE_COLORS: dict[BlockType, str] = {
    "text": "#1f77b4",
    "title": "#d62728",
    "formula": "#9467bd",
    "table": "#2ca02c",
    "figure": "#ff7f0e",
    "caption": "#17becf",
    "unknown": "#7f7f7f",
}


def draw_page_overlay(
    pdf_path: Path, artifact: ParsedArtifact, page: int, *, dpi: int = DEFAULT_OVERLAY_DPI
) -> Image.Image:
    image = render_page(pdf_path, page, dpi=dpi)
    draw = ImageDraw.Draw(image)
    for block in artifact.blocks_on_page(page):
        x1, y1, x2, y2 = block.bbox.to_pixels(width_px=image.width, height_px=image.height)
        color = TYPE_COLORS[block.type]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=LINE_WIDTH)
        # Label just above the box, or inside it near the page top so it is not clipped.
        label_y = y1 - 12 if y1 >= 12 else y1 + 2
        draw.text((x1 + 2, label_y), block.source_id, fill=color)
    return image


def render_overlays(
    pdf_path: Path,
    artifact: ParsedArtifact,
    out_dir: Path,
    *,
    dpi: int = DEFAULT_OVERLAY_DPI,
    pages: list[int] | None = None,
) -> list[Path]:
    """Overlay PNGs for ``pages`` (default: every page that has blocks); returns the files written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = pages if pages is not None else sorted({block.page for block in artifact.blocks})
    written: list[Path] = []
    for page in targets:
        image = draw_page_overlay(pdf_path, artifact, page, dpi=dpi)
        path = out_dir / overlay_page_name(page)
        image.save(path)
        written.append(path)
    logger.info("overlays backend=%s pages=%d dir=%s", artifact.backend, len(written), out_dir)
    return written
