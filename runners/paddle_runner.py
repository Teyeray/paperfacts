# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "paddlepaddle==3.2.1; sys_platform == 'darwin'",
#   "paddlepaddle-gpu==3.2.1; sys_platform == 'linux'",
#   "paddleocr[doc-parser]>=3.7,<3.8",
# ]
#
# [[tool.uv.index]]
# name = "paddle-cpu"
# url = "https://www.paddlepaddle.org.cn/packages/stable/cpu/"
# explicit = true
#
# [[tool.uv.index]]
# name = "paddle-cu126"
# url = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
# explicit = true
#
# [tool.uv.sources]
# paddlepaddle = { index = "paddle-cpu" }
# paddlepaddle-gpu = { index = "paddle-cu126" }
# ///
"""PaddleOCR-VL parsing runner — runs inside Paddle's own isolated dependency environment (a PEP 723 script).

Usage (invoked by the main package's SubprocessParser, or run by hand)::

    uv run --locked --script runners/paddle_runner.py --pdf paper.pdf --out data/docs/<sha>/raw/paddleocr_vl

The platform markers in the dependency header let the same script and the same lockfile pick the
right package on each machine: macOS gets the CPU build of ``paddlepaddle``, Linux gets
``paddlepaddle-gpu`` (the CUDA 12.6 index).

The job is deliberately kept thin:

1. Render each page to a PNG **ourselves** with pypdfium2 (at a fixed DPI) before handing it to
   PaddleOCR-VL; this way we know each page image's exact pixel size, which is what makes
   lossless normalization of ``block_bbox`` (pixel coordinates) possible. Handing the whole PDF to
   Paddle directly would let it render at its own internal DPI, leaving the size a guess.
2. Write each page's native result (``res.save_to_json`` / ``save_to_markdown``) as-is to
   ``<out>/pages/``;
3. Write a ``<out>/meta.json`` recording the version, each page's geometry (PDF points + rendered
   pixels), and file locations.

It does **not** do format conversion: the adapter that turns ``parsing_res_list`` into
``SourceBlock`` lives in the main package at ``src/paperfacts/adapters/paddle.py`` as a pure
function, testable with fixtures, and doesn't need paddle installed.

Isolation principle (both directions): the main package never ``import paddleocr``; this script
never ``import paperfacts``.

VLM inference backend: runs in-process by default (slow on a Mac CPU — only good for checking 1-2
pages). In production, use ``--vl-backend`` / ``--vl-server-url`` to offload the VLM stage to a
vLLM / MLX service, or just use the official image under deploy/ (the recommended path on Linux,
where this script doesn't come into play at all).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# ---- Contract constants ------------------------------------------------------------------
PARSER_NAME = "paddleocr_vl"
PAGES_DIRNAME = "pages"
META_FILENAME = "meta.json"
DEFAULT_RENDER_DPI = 200
PDF_POINTS_PER_INCH = 72


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    The main package only needs --pdf / --out; the rest are debug or deployment options.
    """
    parser = argparse.ArgumentParser(
        prog="runners/paddle_runner.py",
        description="Parse a single PDF page by page with PaddleOCR-VL; write native output and meta.json to disk.",
    )
    parser.add_argument("--pdf", required=True, type=Path, help="input PDF path")
    parser.add_argument("--out", required=True, type=Path, help="output directory (created if missing)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_RENDER_DPI, help="page render DPI")
    parser.add_argument("--start-page", type=int, default=0, help="first page, 0-indexed, inclusive")
    parser.add_argument(
        "--end-page", type=int, default=None, help="last page, 0-indexed, inclusive; defaults to the last page"
    )
    parser.add_argument("--device", default=None, help="Paddle device, e.g. cpu / gpu:0; auto-detected if unset")
    parser.add_argument(
        "--vl-backend",
        default=None,
        help="VLM inference backend, e.g. vllm-server / mlx-vlm-server; runs in-process if unset",
    )
    parser.add_argument("--vl-server-url", default=None, help="VLM service URL, used together with --vl-backend")
    parser.add_argument(
        "--vl-model-name",
        default=None,
        help="server-side model name / HF repo id, e.g. PaddlePaddle/PaddleOCR-VL-1.6",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class PageRender:
    """One page's rendered geometry: PDF-point size + actual pixel size + PNG path."""

    index: int
    width_pt: float
    height_pt: float
    width_px: int
    height_px: int
    image: Path


@dataclass(frozen=True)
class PageResult:
    """One page's inference output: render info + where Paddle's native JSON / Markdown landed on disk."""

    render: PageRender
    json_path: Path
    markdown_dir: Path

    def to_meta(self, out_dir: Path) -> dict[str, Any]:
        """One entry of meta.json ``pages[]``, with paths relative to out_dir.

        Field names are a contract with the main package; update the adapter if they change.
        """
        render = self.render
        return {
            "index": render.index,
            "width_pt": render.width_pt,
            "height_pt": render.height_pt,
            "width_px": render.width_px,
            "height_px": render.height_px,
            "image": str(render.image.relative_to(out_dir)),
            "json": str(self.json_path.relative_to(out_dir)),
            "markdown_dir": str(self.markdown_dir.relative_to(out_dir)),
        }


def sha256_of_file(path: Path) -> str:
    """SHA-256 of the whole file, using the same algorithm the main package uses for document_id."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_pages(
    pdf_path: Path, pages_dir: Path, *, dpi: int, start_page: int, end_page: int | None
) -> tuple[list[PageRender], int]:
    """Render the selected pages to PNG; return ``(per-page render records, total PDF page count)``.

    ``width_px``/``height_px`` come from the actual rendered image rather than a computed formula,
    because pypdfium2 rounds the size; the adapter must divide by the real pixel size recorded
    here when normalizing ``block_bbox``.
    """
    import pypdfium2 as pdfium  # already a dependency of paddlex[ocr]

    scale = dpi / PDF_POINTS_PER_INCH
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        last_index = len(document) - 1
        stop = last_index if end_page is None else min(end_page, last_index)
        if start_page > stop:
            raise ValueError(f"page range is empty: start={start_page}, end={stop}")
        renders: list[PageRender] = []
        for index in range(start_page, stop + 1):
            page = document[index]
            width_pt, height_pt = page.get_size()
            image = page.render(scale=scale).to_pil()
            image_path = pages_dir / f"page_{index:03d}.png"
            image.save(image_path)
            renders.append(
                PageRender(
                    index=index,
                    width_pt=width_pt,
                    height_pt=height_pt,
                    width_px=image.width,
                    height_px=image.height,
                    image=image_path,
                )
            )
        return renders, len(document)
    finally:
        document.close()


def bypass_system_proxy_for_loopback(server_url: str | None) -> None:
    """Add localhost to NO_PROXY when the VLM service is on this machine.

    macOS's system-level HTTP proxy (e.g. Clash on 127.0.0.1:8080) gets picked up automatically by
    Python's urllib/httpx, so even requests to localhost get forwarded to the proxy, which then
    replies with a 503 — this shows up as Paddle's 'vlm' worker reporting Error code: 503, while
    the MLX server's own logs show no such request at all. Must be set before the openai client is
    created.
    """
    if not server_url:
        return
    host = urlparse(server_url).hostname or ""
    if host not in ("localhost", "127.0.0.1", "::1"):
        return
    for name in ("NO_PROXY", "no_proxy"):
        current = [h for h in os.environ.get(name, "").split(",") if h]
        for entry in ("localhost", "127.0.0.1", "::1"):
            if entry not in current:
                current.append(entry)
        os.environ[name] = ",".join(current)


def build_pipeline(args: argparse.Namespace) -> Any:
    """Construct ``PaddleOCRVL`` from the parsed arguments.

    Only pass through the options the user explicitly set; leave everything else to Paddle's
    defaults.
    """
    from paddleocr import PaddleOCRVL

    kwargs: dict[str, Any] = {}
    if args.device:
        kwargs["device"] = args.device
    if args.vl_backend:
        kwargs["vl_rec_backend"] = args.vl_backend
    if args.vl_server_url:
        kwargs["vl_rec_server_url"] = args.vl_server_url
    if args.vl_model_name:
        kwargs["vl_rec_api_model_name"] = args.vl_model_name
    return PaddleOCRVL(**kwargs)


def run_paddle(pipeline: Any, renders: list[PageRender], pages_dir: Path) -> list[PageResult]:
    """Run inference page by page and write results as-is; return each page's output (inputs are untouched).

    ``predict`` is a generator that yields results in input order; ``zip(strict=True)`` guarantees
    an immediate error if the page counts don't match, instead of silently dropping a page. JSON
    is written straight to its target file; Markdown goes into a per-page subdirectory because it
    may come with an image directory attached.
    """
    image_paths = [str(render.image) for render in renders]
    results: list[PageResult] = []
    for render, result in zip(renders, pipeline.predict(input=image_paths), strict=True):
        json_path = pages_dir / f"page_{render.index:03d}.json"
        markdown_dir = pages_dir / f"page_{render.index:03d}_md"
        markdown_dir.mkdir(exist_ok=True)
        result.save_to_json(save_path=str(json_path))
        result.save_to_markdown(save_path=str(markdown_dir))
        results.append(PageResult(render=render, json_path=json_path, markdown_dir=markdown_dir))
    return results


def framework_versions() -> dict[str, str]:
    """Record which paddle distribution is actually installed (paddlepaddle on Mac, paddlepaddle-gpu on Linux)."""
    versions: dict[str, str] = {}
    for name in ("paddlepaddle", "paddlepaddle-gpu", "paddlex"):
        try:
            versions[name] = package_version(name)
        except PackageNotFoundError:
            continue
    return versions


def build_meta(
    args: argparse.Namespace,
    *,
    results: list[PageResult],
    total_pages: int,
    out_dir: Path,
    started_at: str,
    duration_s: float,
) -> dict[str, object]:
    """Assemble meta.json, the runner's only contract with the main package.

    Keep the adapter in sync whenever a field here changes.
    """
    pages = [result.to_meta(out_dir) for result in results]
    return {
        "parser": PARSER_NAME,
        "parser_version": package_version("paddleocr"),
        "framework": framework_versions(),
        "render_dpi": args.dpi,
        "vl_backend": args.vl_backend or "in-process",
        "source": {
            "pdf": str(args.pdf.resolve()),
            "sha256": sha256_of_file(args.pdf),
            # page_count is always the PDF's total page count (matches mineru_runner / the HTTP
            # parser); parsed_page_count is the number of pages actually parsed
            "page_count": total_pages,
            "parsed_page_count": len(pages),
            "page_range": [args.start_page, args.end_page],
        },
        "pages": pages,
        "files": {"pages_dir": PAGES_DIRNAME},
        "runner": {
            "script": "runners/paddle_runner.py",
            "argv": sys.argv[1:],
            "python": platform.python_version(),
            "platform": sys.platform,
            "device": args.device,
            "started_at": started_at,
            "duration_s": round(duration_s, 2),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.pdf.is_file():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    out_dir = args.out.resolve()
    pages_dir = out_dir / PAGES_DIRNAME
    pages_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC).isoformat()
    clock = time.monotonic()
    renders, total_pages = render_pages(
        args.pdf, pages_dir, dpi=args.dpi, start_page=args.start_page, end_page=args.end_page
    )
    bypass_system_proxy_for_loopback(args.vl_server_url)
    pipeline = build_pipeline(args)
    results = run_paddle(pipeline, renders, pages_dir)
    duration_s = time.monotonic() - clock

    meta = build_meta(
        args, results=results, total_pages=total_pages, out_dir=out_dir, started_at=started_at, duration_s=duration_s
    )
    # Write meta.json last: its existence marks this parse as complete, which is what the main
    # package uses to detect a cache hit.
    (out_dir / META_FILENAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"ok": True, "out": str(out_dir), "duration_s": round(duration_s, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
