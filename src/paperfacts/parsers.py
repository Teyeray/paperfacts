"""Hand a PDF to MinerU or PaddleOCR-VL and get native output plus ``meta.json`` in a directory.

A parser does nothing else: no format conversion (that is the adapter's job), no knowledge of what happens
downstream. Two implementations per backend produce **identical** directory layouts, so adapters cannot
tell them apart:

- :class:`SubprocessParser` runs ``runners/<name>.py`` with ``uv run --locked --script`` in its own
  environment, which is how MinerU's and PaddleOCR's conflicting dependency trees stay out of this package.
  The runner writes ``meta.json`` itself.
- :class:`MinerUHttpParser` and :class:`PaddleHttpParser` call the long-running services on a GPU server
  and write the response into the runner's layout, then write ``meta.json`` themselves.

Both share the :class:`Parser` template: cache check, clean directory, produce, validate.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import httpx

from paperfacts.errors import ParserError
from paperfacts.models import (
    META_FILENAME,
    Backend,
    DocumentGeometry,
    DocumentInput,
    PageGeometry,
    PageMeta,
    ParserMeta,
    RawParseOutput,
    SourceMeta,
)
from paperfacts.pdf import read_geometry, render_page

logger = logging.getLogger(__name__)

# uv's script mode; --locked makes a stale lockfile fail instead of silently resolving something else.
DEFAULT_COMMAND_PREFIX: tuple[str, ...] = ("uv", "run", "--locked", "--script")
RUNNER_SCRIPTS: dict[Backend, str] = {
    "mineru": "runners/mineru_runner.py",
    "paddleocr_vl": "runners/paddle_runner.py",
}
STDERR_TAIL_LINES = 40
DEFAULT_TIMEOUT_S = 900.0
DEFAULT_RENDER_DPI = 200

# ---- The runners' native layout, replicated here because the HTTP parsers must write the same files. A
# runner cannot import this package, so these literals exist on both sides; changing one means changing
# the runner too.
MINERU_NATIVE_DIRNAME = "native"
MINERU_NATIVE_STEM = "document"
MINERU_PARSE_METHOD = "auto"
PADDLE_PAGES_DIRNAME = "pages"


def mineru_native_dir(out_dir: Path) -> Path:
    """MinerU's ``<out>/native/<stem>/<parse_method>/``, where ``<stem>_content_list.json`` etc. live."""
    return out_dir / MINERU_NATIVE_DIRNAME / MINERU_NATIVE_STEM / MINERU_PARSE_METHOD


def paddle_pages_dir(out_dir: Path) -> Path:
    """PaddleOCR-VL's ``<out>/pages/``, holding ``page_000.png``, ``page_000.json`` and ``page_000_md/``."""
    return out_dir / PADDLE_PAGES_DIRNAME


class PaddlePageFiles(NamedTuple):
    image: Path
    json: Path
    markdown_dir: Path


def paddle_page_files(pages_dir: Path, index: int) -> PaddlePageFiles:
    stem = f"page_{index:03d}"
    return PaddlePageFiles(pages_dir / f"{stem}.png", pages_dir / f"{stem}.json", pages_dir / f"{stem}_md")


def write_meta(out_dir: Path, meta: ParserMeta) -> None:
    """Written last: its existence is what marks the output complete."""
    (out_dir / META_FILENAME).write_text(meta.model_dump_json(indent=2, by_alias=True), encoding="utf-8")


def default_runner_script(repo_root: Path, backend: Backend) -> Path:
    return repo_root / RUNNER_SCRIPTS[backend]


# ---- Template -----------------------------------------------------------------------------------------


class Parser:
    """Cache check, clean directory, produce, write meta.json, validate.

    Subclasses implement :meth:`_produce`; returning a :class:`ParserMeta` makes this class write
    ``meta.json``, returning ``None`` means the producer already wrote it (the runner subprocess does).
    """

    backend: Backend

    def parse(self, document: DocumentInput, out_dir: Path, *, force: bool = False) -> RawParseOutput:
        if not force and (out_dir / META_FILENAME).is_file():
            logger.info("cache_hit backend=%s doc=%s", self.backend, document.document_id[:16])
            try:
                return RawParseOutput.load(out_dir, self.backend, cache_hit=True)
            except ValueError as exc:
                # An older runner's meta.json: fail loudly rather than silently rerun and mask the change.
                raise ParserError(self.backend, "cache", f"{exc}; rerun with --force once confirmed") from exc

        # Cheap pre-flight before anything destructive: a missing runner script must not destroy a previous
        # good output.
        self._check_ready()
        # A cache miss always starts from a clean directory, so leftovers from a half-finished run can never
        # be mistaken for this run's result.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        clock = time.monotonic()
        meta = self._produce(document, out_dir)
        if meta is not None:
            write_meta(out_dir, meta)
        try:
            output = RawParseOutput.load(out_dir, self.backend)
        except (FileNotFoundError, ValueError) as exc:
            raise ParserError(self.backend, "output", str(exc)) from exc

        logger.info(
            "done backend=%s doc=%s version=%s runtime_s=%.1f parsed_pages=%d",
            self.backend,
            document.document_id[:16],
            output.backend_version,
            time.monotonic() - clock,
            output.meta.source.parsed_page_count,
        )
        return output

    def _check_ready(self) -> None:
        """Optional pre-flight; raise :class:`ParserError` on failure."""

    def _produce(self, document: DocumentInput, out_dir: Path) -> ParserMeta | None:
        raise NotImplementedError


# ---- Subprocess ------------------------------------------------------------------------------------


class SubprocessParser(Parser):
    """Runs a ``runners/`` script; only its exit code and stderr tail come back."""

    def __init__(
        self,
        backend: Backend,
        script: Path,
        *,
        command_prefix: Sequence[str] = DEFAULT_COMMAND_PREFIX,
        extra_args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.backend = backend
        self.script = script
        self.command_prefix = tuple(command_prefix)
        self.extra_args = tuple(extra_args)
        self.env = dict(env) if env else None
        self.timeout_s = timeout_s

    def command(self, document: DocumentInput, out_dir: Path) -> list[str]:
        return [
            *self.command_prefix,
            str(self.script),
            "--pdf",
            str(document.pdf_path),
            "--out",
            str(out_dir),
            *self.extra_args,
        ]

    def _check_ready(self) -> None:
        if not self.script.is_file():
            raise ParserError(self.backend, "launch", f"runner script not found: {self.script}")

    def _produce(self, document: DocumentInput, out_dir: Path) -> None:
        cmd = self.command(document, out_dir)
        merged_env = {**os.environ, **self.env} if self.env else None
        logger.info("run backend=%s doc=%s cmd=%s", self.backend, document.document_id[:16], cmd)
        try:
            # The command is built by this module from settings and paths; no user-controlled shell content.
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s, env=merged_env)
        except FileNotFoundError as exc:
            raise ParserError(self.backend, "launch", f"executable not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            detail = f"did not finish within {self.timeout_s}s: {_tail(exc.stderr)}"
            raise ParserError(self.backend, "timeout", detail) from exc
        if completed.returncode != 0:
            raise ParserError(
                self.backend, "run", f"exit code {completed.returncode}, stderr tail:\n{_tail(completed.stderr)}"
            )


def _tail(text: str | bytes | None, lines: int = STDERR_TAIL_LINES) -> str:
    if not text:
        return "(empty)"
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return "\n".join(text.strip().splitlines()[-lines:])


# ---- HTTP services --------------------------------------------------------------------------------------
#
# MinerU ``mineru-api``: ``POST /file_parse`` (multipart) returns ``{"backend", "version", "results":
# {<filename>: {"md_content", "middle_json", "content_list"}}}``, each value the file content as a string.
# PaddleOCR-VL ``paddlex --serve``: ``POST /layout-parsing`` with ``{"file": <base64>, "fileType": 1}`` returns
# ``result.layoutParsingResults[0].prunedResult`` and ``markdown.text``. We send our own rendered PNG per page,
# as the runner does, because only our own rendering tells us the exact pixel size.


class MinerUHttpParser(Parser):
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
            backend=str(payload.get("backend", "pipeline")),  # runner-private field, matching mineru_runner.py
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
    """Write the response strings under the runner's filenames; return paths relative to ``out_dir``."""
    required = {"content_list": "_content_list.json", "middle_json": "_middle.json"}
    files: dict[str, str] = {}
    for key, suffix in required.items():
        content = result.get(key)
        if not isinstance(content, str):
            raise ParserError("mineru", "http", f"response missing {key}")
        path = native_dir / f"{MINERU_NATIVE_STEM}{suffix}"
        path.write_text(content, encoding="utf-8")
        files[key] = str(path.relative_to(out_dir))
    markdown = result.get("md_content")
    if isinstance(markdown, str):
        path = native_dir / f"{MINERU_NATIVE_STEM}.md"
        path.write_text(markdown, encoding="utf-8")
        files["markdown"] = str(path.relative_to(out_dir))
    return files


class PaddleHttpParser(Parser):
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
            parser_version="unknown",  # the paddlex service does not report one; see the image tag
            source=_source_meta(document, geometry),
            pages=pages,
            files={"pages_dir": PADDLE_PAGES_DIRNAME},
            runner=_runner_meta(self.base_url, started_at, time.monotonic() - clock),
            # runner-private fields, matching paddle_runner.py
            framework={},
            render_dpi=self.render_dpi,
            vl_backend="http",
        )

    def _parse_page(self, document: DocumentInput, page: PageGeometry, pages_dir: Path, out_dir: Path) -> PageMeta:
        index = page.index
        image = render_page(document.pdf_path, index, dpi=self.render_dpi)
        image_path, json_path, markdown_dir = paddle_page_files(pages_dir, index)
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

        json_path.write_text(json.dumps(result.get("prunedResult", {}), ensure_ascii=False, indent=2), "utf-8")
        markdown_dir.mkdir(exist_ok=True)
        markdown_text = (result.get("markdown") or {}).get("text", "")
        (markdown_dir / f"page_{index:03d}.md").write_text(markdown_text, encoding="utf-8")

        return PageMeta(
            index=index,
            width_pt=page.width_pt,
            height_pt=page.height_pt,
            width_px=image.width,
            height_px=image.height,
            image=str(image_path.relative_to(out_dir)),
            json_path=str(json_path.relative_to(out_dir)),
            markdown_dir=str(markdown_dir.relative_to(out_dir)),
        )


def _source_meta(document: DocumentInput, geometry: DocumentGeometry) -> SourceMeta:
    """The HTTP path always parses the whole document."""
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
