# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "mineru[pipeline]==3.4.5",
#   # MinerU 3.4.5's pytorchocr module imports six but doesn't declare it as a dependency (an
#   # upstream packaging gap), which raises ModuleNotFoundError in a clean environment; the images
#   # under deploy/ and the host setup scripts need the same fix.
#   "six>=1.16",
# ]
# ///
"""MinerU parsing runner — runs inside MinerU's own isolated dependency environment (a PEP 723 script).

Usage (invoked by the main package's SubprocessParser, or run by hand)::

    uv run --locked --script runners/mineru_runner.py --pdf paper.pdf --out data/docs/<sha>/raw/mineru

The job is deliberately kept thin, just three steps:

1. Parse the PDF on MinerU's ``pipeline`` backend (only the pipeline backend produces
   fine-grained layout + bbox output);
2. Write MinerU's **native output** (content_list / middle_json / markdown / images) as-is to
   ``<out>/native/``;
3. Write a ``<out>/meta.json`` recording the parser version, page geometry (PDF points), and the
   relative path of each output file.

It does **not** do any format conversion: the adapter that turns the native JSON into
``SourceBlock`` lives in the main package at ``src/paperfacts/adapters/mineru.py``, where it's a
pure function testable with fixtures and doesn't need torch installed.

Isolation principle (both directions): the main package never ``import mineru``; this script
never ``import paperfacts``. The only contract between them is the structure of ``meta.json``
(see ``build_meta``).

Environment variables passed through to MinerU (this script doesn't interpret them):

- ``MINERU_MODEL_SOURCE``  ``huggingface`` / ``modelscope`` / ``local`` — where model weights come from
- ``MINERU_DEVICE_MODE``   ``cpu`` / ``mps`` / ``cuda`` / ``cuda:0`` — auto-detected if unset
- ``CUDA_VISIBLE_DEVICES`` restricts visible GPUs on Linux (pinned to 4,5,6,7 in the deployment scripts)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path

# ---- Contract constants ------------------------------------------------------------------
# These values get written into meta.json; the main package's adapter relies on them to identify
# which parser produced the output and which layout it uses.
PARSER_NAME = "mineru"
BACKEND = "pipeline"
# do_parse organizes its output as <output_dir>/<pdf_file_name>/<parse_method>/; we pin a safe
# fixed filename so that spaces, non-ASCII characters, or excessive length in the paper's
# original name never trip MinerU's filename rules.
NATIVE_STEM = "document"
NATIVE_DIRNAME = "native"
META_FILENAME = "meta.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments. The surface is deliberately small: the main package only needs --pdf / --out."""
    parser = argparse.ArgumentParser(
        prog="runners/mineru_runner.py",
        description="Parse a single PDF with MinerU (pipeline backend); write native output and meta.json to disk.",
    )
    parser.add_argument("--pdf", required=True, type=Path, help="input PDF path")
    parser.add_argument("--out", required=True, type=Path, help="output directory (created if missing)")
    parser.add_argument("--lang", default="en", help="OCR language hint; defaults to en for scientific papers")
    parser.add_argument(
        "--parse-method",
        default="auto",
        choices=["auto", "txt", "ocr"],
        help="auto=use the text layer when present, else OCR; force ocr for comparison",
    )
    parser.add_argument("--start-page", type=int, default=0, help="first page, 0-indexed, inclusive")
    parser.add_argument(
        "--end-page", type=int, default=None, help="last page, 0-indexed, inclusive; defaults to the last page"
    )
    parser.add_argument("--no-formula", action="store_true", help="disable formula recognition")
    parser.add_argument("--no-table", action="store_true", help="disable table recognition")
    parser.add_argument(
        "--draw-layout-bbox",
        action="store_true",
        help="have MinerU also output a PDF with layout boxes drawn (debug; the main package has its own overlay tool)",
    )
    return parser.parse_args(argv)


def sha256_of_file(path: Path) -> str:
    """SHA-256 of the whole file. The main package uses the same algorithm for document_id, so the two must match."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_geometry(pdf_path: Path) -> list[dict[str, float | int]]:
    """Read each page's size with pypdfium2 (PDF points, 1 pt = 1/72 inch).

    This is the "source of truth" for the page coordinate system: the main package independently
    computes the same thing with pypdfium2 and cross-checks it, so the parser side and the
    cropping side always agree on page geometry.
    """
    import pypdfium2 as pdfium  # already a MinerU dependency; no need to declare it separately

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        pages: list[dict[str, float | int]] = []
        for index in range(len(document)):
            width_pt, height_pt = document[index].get_size()
            pages.append({"index": index, "width_pt": width_pt, "height_pt": height_pt})
        return pages
    finally:
        document.close()


def run_mineru(args: argparse.Namespace, native_dir: Path) -> None:
    """Call MinerU's public ``do_parse`` entry point.

    We use ``do_parse`` rather than the lower-level ``doc_analyze_streaming`` because it's the
    stable entry point shared by the CLI and FastAPI, handling PDF preprocessing, page-range
    cropping, pdfium fallbacks, and other details; we only care about the files it writes.
    """
    from mineru.cli.common import do_parse, read_fn
    from mineru.utils.enum_class import MakeMode

    pdf_bytes = read_fn(args.pdf)
    do_parse(
        output_dir=str(native_dir),
        pdf_file_names=[NATIVE_STEM],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[args.lang],
        backend=BACKEND,
        parse_method=args.parse_method,
        formula_enable=not args.no_formula,
        table_enable=not args.no_table,
        f_draw_layout_bbox=args.draw_layout_bbox,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        start_page_id=args.start_page,
        end_page_id=args.end_page,
    )


def locate_native_outputs(native_dir: Path, out_dir: Path) -> dict[str, str]:
    """Find the files we care about in do_parse's output directory; return paths **relative to out_dir**.

    MinerU's directory structure (``<stem>/<method>/``) and filename suffixes are its own internal
    convention and change between versions (e.g. whether content_list has a ``_v2`` variant). We
    glob for them instead of hardcoding paths, so changes to that internal convention never leak
    into the main package's contract.
    """

    def first(pattern: str) -> Path | None:
        hits = sorted(p for p in native_dir.rglob(pattern) if p.is_file())
        if len(hits) > 1:
            # Multiple files matching the same name can only mean leftovers from a previous run
            # (a different parse_method) got mixed in; picking one alphabetically would be a guess.
            raise RuntimeError(
                f"multiple matches for {pattern} under {native_dir}: {hits}; clear the output dir and rerun"
            )
        return hits[0] if hits else None

    # MinerU 3.x writes both a v1 (flat) and a v2 (nested blocks/sub_type) content_list.
    # The main package's adapter consumes v1: it's flat with stable fields (type / page_idx /
    # bbox / text / table_body, ...), which is enough for provenance; v2's path is only recorded
    # in case a future need calls for finer-grained data.
    content_list = first(f"{NATIVE_STEM}_content_list.json")
    content_list_v2 = first(f"{NATIVE_STEM}_content_list_v2.json")
    middle_json = first(f"{NATIVE_STEM}_middle.json")
    markdown = first(f"{NATIVE_STEM}.md")
    missing = [name for name, path in (("content_list", content_list), ("middle_json", middle_json)) if path is None]
    if missing:
        raise RuntimeError(
            f"MinerU output is missing {missing}; check the contents of {native_dir} and the MinerU logs"
        )
    assert content_list is not None and middle_json is not None  # guaranteed above; satisfies the type checker only

    files = {
        "content_list": str(content_list.relative_to(out_dir)),
        "middle_json": str(middle_json.relative_to(out_dir)),
    }
    if content_list_v2 is not None:
        files["content_list_v2"] = str(content_list_v2.relative_to(out_dir))
    if markdown is not None:
        files["markdown"] = str(markdown.relative_to(out_dir))
    images_dir = content_list.parent / "images"
    if images_dir.is_dir():
        files["images_dir"] = str(images_dir.relative_to(out_dir))
    return files


def parsed_page_count(start_page: int, end_page: int | None, total_pages: int) -> int:
    """Compute the number of pages actually parsed from --start-page/--end-page (inclusive); 0 if the range is empty."""
    last = total_pages - 1 if end_page is None else min(end_page, total_pages - 1)
    return max(0, last - start_page + 1)


def build_meta(
    args: argparse.Namespace,
    *,
    all_pages: list[dict[str, float | int]],
    files: dict[str, str],
    started_at: str,
    duration_s: float,
) -> dict[str, object]:
    """Assemble meta.json, the runner's contract with the main package.

    Keep the adapter in sync whenever a field here changes. ``pages`` lists only the pages parsed
    this run (matches paddle_runner / the HTTP parser); ``page_count`` is still the PDF's total
    page count.
    """
    parsed = parsed_page_count(args.start_page, args.end_page, len(all_pages))
    pages = all_pages[args.start_page : args.start_page + parsed]
    return {
        "parser": PARSER_NAME,
        "parser_version": package_version("mineru"),
        "backend": BACKEND,
        "source": {
            "pdf": str(args.pdf.resolve()),
            "sha256": sha256_of_file(args.pdf),
            # page_count is always the PDF's total page count; parsed_page_count is the number of
            # pages actually parsed this run (after --start-page/--end-page cropping)
            "page_count": len(all_pages),
            "parsed_page_count": parsed,
            "page_range": [args.start_page, args.end_page],
        },
        "pages": pages,
        "files": files,
        "runner": {
            "script": "runners/mineru_runner.py",
            "argv": sys.argv[1:],
            "python": platform.python_version(),
            "platform": sys.platform,
            "device_mode": os.environ.get("MINERU_DEVICE_MODE"),
            "model_source": os.environ.get("MINERU_MODEL_SOURCE"),
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
    native_dir = out_dir / NATIVE_DIRNAME
    native_dir.mkdir(parents=True, exist_ok=True)

    all_pages = page_geometry(args.pdf)
    if not 0 <= args.start_page < len(all_pages):
        print(f"--start-page {args.start_page} out of range: the PDF only has {len(all_pages)} pages", file=sys.stderr)
        return 2
    started_at = datetime.now(UTC).isoformat()
    clock = time.monotonic()
    run_mineru(args, native_dir)
    duration_s = time.monotonic() - clock

    files = locate_native_outputs(native_dir, out_dir)
    meta = build_meta(args, all_pages=all_pages, files=files, started_at=started_at, duration_s=duration_s)
    # Write meta.json last: its existence marks this parse as complete, which is what the main
    # package uses to detect a cache hit.
    (out_dir / META_FILENAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"ok": True, "out": str(out_dir), "duration_s": round(duration_s, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
