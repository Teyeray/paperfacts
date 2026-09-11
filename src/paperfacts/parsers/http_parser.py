"""Server-side parser implementations: call the long-running MinerU / PaddleOCR-VL HTTP services.

The key convention shared with :mod:`paperfacts.parsers.subprocess_parser`: **the directory
layout and meta.json produced here are identical to what the runner scripts write** (layout
constants live in :mod:`paperfacts.storage.raw_layout`). Adapters only ever look at ``meta.json``
and have no idea whether the native output was written by a subprocess or pulled over HTTP.

The two service interfaces (checked against the 3.4.5 / 3.7 source and official docs):

- MinerU ``mineru-api`` / ``mineru-router``: ``POST /file_parse`` (multipart), returns
  ``{"backend", "version", "results": {<filename>: {"md_content", "middle_json", "content_list"}}}``,
  where all three values are **file content as strings** — write them to disk as-is and you get
  the same files the runner would have produced.
- PaddleOCR-VL ``paddleocr-vl-api`` (paddlex --serve): ``POST /layout-parsing`` (JSON),
  ``{"file": <base64>, "fileType": 1}``, returns ``result.layoutParsingResults[0].prunedResult``
  (the ``save_to_json`` content minus input_path/page_index) and ``markdown.text``.
  We send **our own rendered** PNG per page rather than the whole PDF, for the same reason as the
  runner: only our own rendering tells us the exact pixel size.
"""

from __future__ import annotations

import base64
import json
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from paperfacts.models.artifact import Backend, DocumentInput
from paperfacts.models.geometry import DocumentGeometry, PageGeometry
from paperfacts.models.raw_output import PageMeta, ParserMeta, SourceMeta
from paperfacts.parsers.base import CachingParser, ParserError
from paperfacts.pdf import read_geometry, render_page
from paperfacts.storage.raw_layout import (
    MINERU_NATIVE_STEM,
    MINERU_PARSE_METHOD,
    PADDLE_PAGES_DIRNAME,
    mineru_native_dir,
    mineru_native_file,
    paddle_page_image,
    paddle_page_json,
    paddle_page_markdown,
    paddle_page_markdown_dir,
    paddle_pages_dir,
)

logger = logging.getLogger(__name__)

# A several-dozen-page paper can take minutes on the pipeline backend, so the default timeout is
# generous; override via Settings.http_timeout_s.
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_RENDER_DPI = 200


class MinerUHttpParser(CachingParser):
    """Calls the MinerU service via ``POST /file_parse``."""

    backend: Backend = "mineru"

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        lang: str = "en",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client()
        self.timeout_s = timeout_s
        self.lang = lang

    def _produce(self, document: DocumentInput, out_dir: Path) -> ParserMeta:
        started_at = datetime.now(UTC).isoformat()
        clock = time.monotonic()
        payload = self._request(document)
        results: dict[str, Any] = payload.get("results") or {}
        if len(results) != 1:
            raise ParserError(self.backend, "http", f"expected exactly 1 result, got {len(results)}")
        result = next(iter(results.values()))

        native_dir = mineru_native_dir(out_dir)
        native_dir.mkdir(parents=True, exist_ok=True)
        files = _write_mineru_native(native_dir, out_dir, result)

        geometry = read_geometry(document.pdf_path)
        return ParserMeta(
            parser=self.backend,
            parser_version=str(payload.get("version", "unknown")),
            source=_source_meta(document, geometry),
            pages=tuple(PageMeta(index=p.index, width_pt=p.width_pt, height_pt=p.height_pt) for p in geometry.pages),
            files=files,
            runner=_runner_meta(self.base_url, started_at, time.monotonic() - clock),
            # The following are runner-private fields (extra="allow"), matching runners/mineru_runner.py
            backend=str(payload.get("backend", "pipeline")),
        )

    def _request(self, document: DocumentInput) -> dict[str, Any]:
        files = {"files": (f"{MINERU_NATIVE_STEM}.pdf", document.pdf_path.read_bytes(), "application/pdf")}
        data = {
            "backend": "pipeline",
            "parse_method": MINERU_PARSE_METHOD,
            "lang_list": self.lang,
            "formula_enable": "true",
            "table_enable": "true",
            "return_md": "true",
            "return_middle_json": "true",
            "return_content_list": "true",
            "return_images": "false",
            "response_format_zip": "false",
        }
        try:
            response = self.client.post(f"{self.base_url}/file_parse", files=files, data=data, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise ParserError(self.backend, "http", f"{type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            raise ParserError(self.backend, "http", f"HTTP {response.status_code}: {response.text[:500]}")
        return response.json()


def _write_mineru_native(native_dir: Path, out_dir: Path, result: dict[str, Any]) -> dict[str, str]:
    """Write the three response strings using the runner's filenames; return paths relative to out_dir."""
    required = {"content_list": "_content_list.json", "middle_json": "_middle.json"}
    files: dict[str, str] = {}
    for key, suffix in required.items():
        content = result.get(key)
        if not isinstance(content, str):
            raise ParserError("mineru", "http", f"response missing {key}")
        path = mineru_native_file(native_dir, suffix)
        path.write_text(content, encoding="utf-8")
        files[key] = _relative(path, out_dir)
    markdown = result.get("md_content")
    if isinstance(markdown, str):
        path = mineru_native_file(native_dir, ".md")
        path.write_text(markdown, encoding="utf-8")
        files["markdown"] = _relative(path, out_dir)
    return files


class PaddleHttpParser(CachingParser):
    """Calls the PaddleOCR-VL service's ``POST /layout-parsing`` once per page."""

    backend: Backend = "paddleocr_vl"

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        render_dpi: int = DEFAULT_RENDER_DPI,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client()
        self.timeout_s = timeout_s
        self.render_dpi = render_dpi

    def _produce(self, document: DocumentInput, out_dir: Path) -> ParserMeta:
        pages_dir = paddle_pages_dir(out_dir)
        pages_dir.mkdir(parents=True, exist_ok=True)
        geometry = read_geometry(document.pdf_path)
        started_at = datetime.now(UTC).isoformat()
        clock = time.monotonic()

        pages = tuple(self._parse_page(document, page, pages_dir, out_dir) for page in geometry.pages)
        return ParserMeta(
            parser=self.backend,
            # The paddlex service API doesn't report a version; check the deployment image tag
            # for the exact version.
            parser_version="unknown",
            source=_source_meta(document, geometry),
            pages=pages,
            files={"pages_dir": PADDLE_PAGES_DIRNAME},
            runner=_runner_meta(self.base_url, started_at, time.monotonic() - clock),
            # Runner-private fields, matching runners/paddle_runner.py
            framework={},
            render_dpi=self.render_dpi,
            vl_backend="http",
        )

    def _parse_page(self, document: DocumentInput, page: PageGeometry, pages_dir: Path, out_dir: Path) -> PageMeta:
        index = page.index
        image = render_page(document.pdf_path, index, dpi=self.render_dpi)
        image_path = paddle_page_image(pages_dir, index)
        image.save(image_path)

        payload = {
            "file": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            "fileType": 1,  # 1 = image; we always send a single-page PNG
            "visualize": False,
        }
        try:
            response = self.client.post(f"{self.base_url}/layout-parsing", json=payload, timeout=self.timeout_s)
        except httpx.HTTPError as exc:
            raise ParserError(self.backend, "http", f"page {index}: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            raise ParserError(self.backend, "http", f"page {index} HTTP {response.status_code}: {response.text[:500]}")
        try:
            result = response.json()["result"]["layoutParsingResults"][0]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ParserError(self.backend, "http", f"page {index} response has an unexpected shape: {exc}") from exc

        json_path = paddle_page_json(pages_dir, index)
        json_path.write_text(json.dumps(result.get("prunedResult", {}), ensure_ascii=False, indent=2), "utf-8")
        markdown_dir = paddle_page_markdown_dir(pages_dir, index)
        markdown_dir.mkdir(exist_ok=True)
        markdown_text = (result.get("markdown") or {}).get("text", "")
        paddle_page_markdown(markdown_dir, index).write_text(markdown_text, encoding="utf-8")

        return PageMeta(
            index=index,
            width_pt=page.width_pt,
            height_pt=page.height_pt,
            width_px=image.width,
            height_px=image.height,
            image=_relative(image_path, out_dir),
            json_path=_relative(json_path, out_dir),
            markdown_dir=_relative(markdown_dir, out_dir),
        )


def _source_meta(document: DocumentInput, geometry: DocumentGeometry) -> SourceMeta:
    """The HTTP path always parses the whole document: parsed_page_count == page_count."""
    return SourceMeta(
        pdf=str(document.pdf_path),
        sha256=document.sha256,
        page_count=geometry.page_count,
        parsed_page_count=geometry.page_count,
        page_range=(0, None),
    )


def _runner_meta(base_url: str, started_at: str, duration_s: float) -> dict[str, Any]:
    return {
        "script": "http",
        "base_url": base_url,
        "python": platform.python_version(),
        "platform": sys.platform,
        "started_at": started_at,
        "duration_s": round(duration_s, 2),
    }


def _relative(path: Path, out_dir: Path) -> str:
    return str(path.relative_to(out_dir))
