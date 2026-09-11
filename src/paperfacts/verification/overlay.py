"""Bbox overlays: draw one parser's block boxes back onto the page image for visual acceptance
checks (the core M1 acceptance tool, see plan Task 6).

Rendering uses :func:`paperfacts.pdf.render_page`, the same pypdfium2 logic shared with the runners
and with cropping, so if the boxes in an overlay hug the text correctly, cropping the same geometry
later is guaranteed to be correct too.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw

from paperfacts.models.artifact import BlockType, ParsedArtifact
from paperfacts.pdf import render_page
from paperfacts.storage.paths import overlay_page_name

logger = logging.getLogger(__name__)

DEFAULT_OVERLAY_DPI = 150
LINE_WIDTH = 2
# One fixed color per block type, shared by both parsers, so overlays can be compared side by side.
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
    """Render one page and draw every block's box and source_id on it, returning the image (not
    written to disk).
    """
    image = render_page(pdf_path, page, dpi=dpi)
    draw = ImageDraw.Draw(image)
    for block in artifact.blocks_on_page(page):
        x1, y1, x2, y2 = block.bbox.to_pixels(width_px=image.width, height_px=image.height)
        color = TYPE_COLORS[block.type]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=LINE_WIDTH)
        # Label sits just outside the top-left corner; flip to inside the box near the page top so
        # it doesn't get clipped.
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
    """Generate overlay PNGs for the given pages (default: every page that has blocks), returning
    the files written.
    """
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
