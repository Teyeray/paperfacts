"""Mirrors the runner's native-output directory layout.

These constants and functions **replicate** the literal values ``runners/mineru_runner.py`` /
``runners/paddle_runner.py`` use when writing to disk. A runner can't import the main package
(dependency isolation), so the layout has to be written on both sides; naming it here, in one
place, makes the "changing this means updating the runner too" cost visible instead of scattering
it across f-strings in http_parser.

:mod:`paperfacts.parsers.http_parser` uses these to write the service's response into a layout
**identical** to the runner's, so adapters can't tell (and don't need to) whether native output
came from a subprocess or from HTTP.
"""

from __future__ import annotations

from pathlib import Path

# ---- MinerU: do_parse's <out>/native/<stem>/<parse_method>/<stem>_*.json ------------------
MINERU_NATIVE_DIRNAME = "native"
MINERU_NATIVE_STEM = "document"
MINERU_PARSE_METHOD = "auto"


def mineru_native_dir(out_dir: Path) -> Path:
    return out_dir / MINERU_NATIVE_DIRNAME / MINERU_NATIVE_STEM / MINERU_PARSE_METHOD


def mineru_native_file(native_dir: Path, suffix: str) -> Path:
    """``suffix`` looks like ``_content_list.json`` / ``_middle.json`` / ``.md``."""
    return native_dir / f"{MINERU_NATIVE_STEM}{suffix}"


# ---- PaddleOCR-VL: <out>/pages/page_000.{png,json} and page_000_md/ -------------------------
PADDLE_PAGES_DIRNAME = "pages"


def paddle_pages_dir(out_dir: Path) -> Path:
    return out_dir / PADDLE_PAGES_DIRNAME


def paddle_page_image(pages_dir: Path, index: int) -> Path:
    return pages_dir / f"page_{index:03d}.png"


def paddle_page_json(pages_dir: Path, index: int) -> Path:
    return pages_dir / f"page_{index:03d}.json"


def paddle_page_markdown_dir(pages_dir: Path, index: int) -> Path:
    return pages_dir / f"page_{index:03d}_md"


def paddle_page_markdown(markdown_dir: Path, index: int) -> Path:
    return markdown_dir / f"page_{index:03d}.md"
