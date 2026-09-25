"""Hand a PDF to MinerU or PaddleOCR-VL and get native output plus ``meta.json`` in a directory.

A parser does nothing else: no format conversion (that is the adapter's job), no knowledge of what happens
downstream. Two implementations per backend produce **identical** directory layouts, so adapters cannot
tell them apart:

- :class:`SubprocessParser` runs ``runners/<name>.py`` with ``uv run --locked --script`` in its own
  environment, which is how MinerU's and PaddleOCR's conflicting dependency trees stay out of this package.
  The runner writes ``meta.json`` itself.
- :class:`MinerUHttpParser` and :class:`PaddleHttpParser` call the long-running services on a GPU server
  and write the response into the runner's layout, then write ``meta.json`` themselves.

Both share the :class:`Parser` template: cache check, produce into a staging directory, validate,
swap in. A run that fails or is killed never touches the previous output.
"""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, Self

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
# How long a runner gets between SIGTERM and SIGKILL. A runner holding ~10 GB has nothing to flush, so
# this only has to cover the signal round-trip.
RUNNER_KILL_GRACE_S = 5.0
DEFAULT_RENDER_DPI = 200
# A GPU service that times out or answers 5xx is usually busy or restarting, not wrong: each HTTP request (one
# page for PaddleOCR-VL, the whole paper for MinerU) gets this many attempts, with a doubling backoff, before
# the parse fails. Not settings: nothing a user tunes, and never part of what the parser produces.
HTTP_RETRY_ATTEMPTS = 3
HTTP_RETRY_BACKOFF_S = 5.0
HTTP_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

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


def _swap_into_place(staging: Path, out_dir: Path) -> None:
    """Replace ``out_dir`` with ``staging``: move any previous output aside, rename, then delete it.

    ``os.replace`` refuses a non-empty directory as the target, so the previous output has to be renamed
    out of the way first. The window in which neither directory is at ``out_dir`` is one rename long; if
    the process dies inside it the previous output is still on disk under its ``.old.`` name.
    """
    previous = out_dir.with_name(f".{out_dir.name}.old.{uuid.uuid4().hex}") if out_dir.exists() else None
    if previous is not None:
        os.replace(out_dir, previous)
    try:
        os.replace(staging, out_dir)
    except OSError:
        if previous is not None:
            os.replace(previous, out_dir)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)


def _recover_leftovers(out_dir: Path) -> None:
    """Undo what a crash inside :func:`_swap_into_place` can leave behind.

    Dying between the two renames leaves the previous output under its ``.old.`` name and nothing at
    ``out_dir``; putting the newest one back is the difference between a cache hit and re-parsing a
    paper from scratch. Staging directories from a crashed run are just garbage, so they go.
    """
    if not out_dir.parent.is_dir():
        return
    if not out_dir.exists():
        stranded = sorted(out_dir.parent.glob(f".{out_dir.name}.old.*"), key=lambda p: p.stat().st_mtime)
        if stranded:
            recovered = stranded.pop()  # the newest is the one the interrupted swap was replacing
            logger.warning("recovering output stranded by an interrupted swap: %s -> %s", recovered, out_dir)
            os.replace(recovered, out_dir)
        for leftover in stranded:
            shutil.rmtree(leftover, ignore_errors=True)
    with _active_lock:
        in_use = set(_active_staging)
    for staging in out_dir.parent.glob(f".{out_dir.name}.new.*"):
        if staging not in in_use:  # a run happening right now in another thread owns its own directory
            shutil.rmtree(staging, ignore_errors=True)


def default_runner_script(repo_root: Path, backend: Backend) -> Path:
    return repo_root / RUNNER_SCRIPTS[backend]


# ---- Parse locks ---------------------------------------------------------------------------------------
#
# Documents run in parallel, parsing does not: a parser service has one GPU, and two papers sent to it at
# once only compete for its memory. So one run per lock at a time, a lock per backend -- MinerU and
# PaddleOCR-VL are separate services and may parse two different papers side by side. A cache hit never
# takes a lock, and neither does the adaptation after the run.

SUBPROCESS_LOCK_KEY = "runner"
_parse_locks: dict[str, threading.Lock] = {}
_parse_locks_guard = threading.Lock()


@contextmanager
def _exclusive(key: str, backend: Backend, document: DocumentInput) -> Iterator[None]:
    with _parse_locks_guard:
        lock = _parse_locks.setdefault(key, threading.Lock())
    if not lock.acquire(blocking=False):
        # Said once, so a job log explains a parse stage that sits at "running" while another paper parses.
        logger.info(
            "%s waits for parse lock %r: another document is being parsed (doc=%s)",
            backend,
            key,
            document.document_id[:16],
        )
        lock.acquire()
    try:
        yield
    finally:
        lock.release()


# ---- Template -----------------------------------------------------------------------------------------


class Parser:
    """Cache check, produce into a staging directory, write meta.json, validate, swap into place.

    Subclasses implement :meth:`_produce`; returning a :class:`ParserMeta` makes this class write
    ``meta.json``, returning ``None`` means the producer already wrote it (the runner subprocess does).
    """

    backend: Backend

    def is_cached(self, out_dir: Path) -> bool:
        """Would :meth:`parse` return the stored output instead of running?

        ``meta.json`` is written last, so its presence is what marks the directory complete. Recovering
        leftovers first is part of the answer: an interrupted swap hides an output that is still usable.
        """
        _recover_leftovers(out_dir)
        return (out_dir / META_FILENAME).is_file()

    def parse(self, document: DocumentInput, out_dir: Path, *, force: bool = False) -> RawParseOutput:
        if self.is_cached(out_dir) and not force:  # always first: it also recovers interrupted swaps
            logger.info("cache_hit backend=%s doc=%s", self.backend, document.document_id[:16])
            try:
                return RawParseOutput.load(out_dir, self.backend, cache_hit=True)
            except ValueError as exc:
                # An older runner's meta.json: fail loudly rather than silently rerun and mask the change.
                raise ParserError(self.backend, "cache", f"{exc}; rerun with --force once confirmed") from exc

        # Cheap pre-flight before anything destructive: a missing runner script must not destroy a previous
        # good output.
        self._check_ready()

        clock = time.monotonic()
        # The run produces into a fresh directory beside the target and is swapped in only once it has
        # validated: a run that fails, times out or is killed leaves the previous output exactly as it was.
        # The staging directory is a sibling so the swap is a rename within one filesystem.
        staging = out_dir.with_name(f".{out_dir.name}.new.{uuid.uuid4().hex}")
        staging.mkdir(parents=True)
        with _active_lock:
            _active_staging.add(staging)
        try:
            with _exclusive(self.lock_key, self.backend, document):
                meta = self._produce(document, staging)
            if meta is not None:
                write_meta(staging, meta)
            try:
                RawParseOutput.load(staging, self.backend)
            except (FileNotFoundError, ValueError) as exc:
                raise ParserError(self.backend, "output", str(exc)) from exc
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            with _active_lock:
                _active_staging.discard(staging)
        _swap_into_place(staging, out_dir)
        # Reload so the handle points at the final location rather than the staging directory.
        output = RawParseOutput.load(out_dir, self.backend)

        logger.info(
            "done backend=%s doc=%s version=%s runtime_s=%.1f parsed_pages=%d",
            self.backend,
            document.document_id[:16],
            output.backend_version,
            time.monotonic() - clock,
            output.meta.source.parsed_page_count,
        )
        return output

    @property
    def lock_key(self) -> str:
        """Which parse lock this parser's runs take (see :func:`_exclusive`): one per backend by default."""
        return self.backend

    def _check_ready(self) -> None:
        """Optional pre-flight; raise :class:`ParserError` on failure."""

    def _produce(self, document: DocumentInput, out_dir: Path) -> ParserMeta | None:
        raise NotImplementedError

    # A parser may own resources (the HTTP parsers own a connection pool). `parse_document` builds one per
    # call and uses it as a context manager, so nothing outlives the parse in a long-running server.
    def close(self) -> None:
        """Release what this parser owns; nothing by default."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


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

    @property
    def lock_key(self) -> str:
        """Both runners share one lock: each loads its whole model set into this machine's memory, and two
        of them at once do not fit in a laptop's -- the reason ``--backend both`` has always been serial."""
        return SUBPROCESS_LOCK_KEY

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
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # A runner printing non-UTF-8 bytes must not turn a clean exit into a UnicodeDecodeError.
                errors="replace",
                env=merged_env,
                # `uv run` spawns the real parser as a grandchild, so killing the direct child alone leaves
                # a multi-gigabyte python process behind. Its own session makes the whole tree killable as
                # one group. POSIX only; on Windows there is no session and the group kill is skipped.
                start_new_session=_POSIX,
            )
        except FileNotFoundError as exc:
            raise ParserError(self.backend, "launch", f"executable not found: {exc}") from exc

        with _tracked(process) as pgid:
            try:
                _, stderr = process.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired as exc:
                _terminate_tree(process, pgid)
                _, stderr = process.communicate()
                detail = f"did not finish within {self.timeout_s}s: {_tail(stderr)}"
                raise ParserError(self.backend, "timeout", detail) from exc
            except BaseException:
                # Anything else that stops us waiting -- KeyboardInterrupt, a cancelled worker -- must not
                # leave the runner alive holding the GPU and its memory.
                _terminate_tree(process, pgid)
                raise
        if process.returncode != 0:
            raise ParserError(self.backend, "run", f"exit code {process.returncode}, stderr tail:\n{_tail(stderr)}")


# ---- Runner process lifetime ---------------------------------------------------------------------
#
# A runner outliving the process that started it is not a tidiness problem: the PaddleOCR-VL runner holds
# about 10 GB and a GPU, so an orphan starves the machine until someone notices. Every path that stops
# waiting for a runner therefore kills its whole process group, including the parent's own death.

_POSIX = os.name == "posix"
_active_lock = threading.Lock()
# Staging directories a run is using right now, so leftover cleanup cannot delete a live sibling run's.
_active_staging: set[Path] = set()
# pid -> (process, its process-group id). The group is read once at launch: by kill time the pid may
# have been reaped, and a reused pid would hand us a stranger's process group to signal.
_active: dict[int, tuple[subprocess.Popen[str], int | None]] = {}
_cleanup_installed = False


def install_runner_cleanup() -> None:
    """Arrange for running runners to be killed when this process exits. Idempotent.

    Called by the entry points that own a process -- the CLI and the web app's startup -- rather than
    from the launch path: registering an ``atexit`` hook and a signal handler is the process owner's
    decision, and ``signal.signal`` only works on the main thread anyway.
    """
    global _cleanup_installed
    with _active_lock:
        if _cleanup_installed:
            return
        _cleanup_installed = True
    atexit.register(_terminate_all)
    if not _POSIX:
        return
    try:
        previous = signal.getsignal(signal.SIGTERM)

        def on_sigterm(signum: int, frame: Any) -> None:
            _terminate_all()
            if callable(previous):
                previous(signum, frame)
                return
            # Default disposition: die from the signal we were sent, now that the runners are gone.
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGTERM, on_sigterm)
    except ValueError:
        # Not the main thread; atexit still covers a normal interpreter exit.
        logger.debug("SIGTERM handler not installed: not on the main thread")


@contextmanager
def _tracked(process: subprocess.Popen[str]) -> Iterator[int | None]:
    """Register the runner for the exit hooks, yielding its process-group id (``None`` means signal the
    child alone). The group is read here, while the runner is certainly alive."""
    try:
        pgid = os.getpgid(process.pid) if _POSIX else None
    except OSError:  # pragma: no cover - only if the runner died between Popen and this call
        pgid = None
    with _active_lock:
        _active[process.pid] = (process, pgid)
    try:
        yield pgid
    finally:
        with _active_lock:
            _active.pop(process.pid, None)


def _terminate_tree(process: subprocess.Popen[str], pgid: int | None) -> None:
    """SIGTERM the runner's whole process group, then SIGKILL whatever is still there."""
    if process.poll() is not None:
        return
    _signal_group(process, pgid, signal.SIGTERM)
    try:
        process.wait(timeout=RUNNER_KILL_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        logger.warning("runner pid=%s ignored SIGTERM, killing", process.pid)
    _signal_group(process, pgid, signal.SIGKILL)
    try:
        process.wait(timeout=RUNNER_KILL_GRACE_S)
    except subprocess.TimeoutExpired:  # pragma: no cover - a SIGKILLed group does not survive this
        logger.error("runner pid=%s survived SIGKILL", process.pid)


def _signal_group(process: subprocess.Popen[str], pgid: int | None, sig: int) -> None:
    try:
        if pgid is None:  # pragma: no cover - Windows, or a runner that died before we read its group
            process.send_signal(sig)
        else:
            os.killpg(pgid, sig)
    except (OSError, ValueError):
        pass  # already gone, or reaped between the poll and the signal


def _terminate_all() -> None:
    with _active_lock:
        tracked = list(_active.values())
    for process, pgid in tracked:
        _terminate_tree(process, pgid)


def _tail(text: str | bytes | None, lines: int = STDERR_TAIL_LINES) -> str:
    if not text:
        return "(empty)"
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return "\n".join(text.strip().splitlines()[-lines:])


# ---- HTTP services --------------------------------------------------------------------------------------


class _HttpParser(Parser):
    """What the two HTTP parsers share: an owned (or injected) client, closed with the parser, and a bounded
    retry of transient failures around each request."""

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None,
        timeout_s: float,
        retry_attempts: int,
        retry_backoff_s: float,
        sleep: Callable[[float], None] | None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # An injected client belongs to the caller, who closes it; only one made here is closed here.
        self._owns_client = client is None
        self.client = client or httpx.Client()
        self.timeout_s = timeout_s
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff_s = retry_backoff_s
        # Injectable so tests do not wait; resolved per call, so patching time.sleep reaches it too.
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _post(self, path: str, what: str, **kwargs: Any) -> httpx.Response:
        """A 200 response, after up to ``retry_attempts`` tries of a timeout, a connection error or a 5xx.

        Any other status is the request's fault and fails at once: sending it again would get the same answer.
        """
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self.client.post(url, timeout=self.timeout_s, **kwargs)
            except httpx.HTTPError as exc:
                error = f"{what}: {type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return response
                error = f"{what} HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in HTTP_RETRY_STATUS:
                    raise ParserError(self.backend, "http", error)
            if attempt == self.retry_attempts:
                raise ParserError(self.backend, "http", f"{error} (after {attempt} attempts)")
            delay = self.retry_backoff_s * 2 ** (attempt - 1)
            logger.warning("%s retry %d/%d in %.0fs: %s", self.backend, attempt, self.retry_attempts, delay, error)
            (self._sleep or time.sleep)(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises


#
# MinerU ``mineru-api``: ``POST /file_parse`` (multipart) returns ``{"backend", "version", "results":
# {<filename>: {"md_content", "middle_json", "content_list"}}}``, each value the file content as a string.
# PaddleOCR-VL ``paddlex --serve``: ``POST /layout-parsing`` with ``{"file": <base64>, "fileType": 1}`` returns
# ``result.layoutParsingResults[0].prunedResult`` and ``markdown.text``. We send our own rendered PNG per page,
# as the runner does, because only our own rendering tells us the exact pixel size.


class MinerUHttpParser(_HttpParser):
    backend: Backend = "mineru"

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        lang: str = "en",
        retry_attempts: int = HTTP_RETRY_ATTEMPTS,
        retry_backoff_s: float = HTTP_RETRY_BACKOFF_S,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(
            base_url,
            client=client,
            timeout_s=timeout_s,
            retry_attempts=retry_attempts,
            retry_backoff_s=retry_backoff_s,
            sleep=sleep,
        )
        self.lang = lang

    def _produce(self, document: DocumentInput, out_dir: Path) -> ParserMeta:
        started_at = datetime.now(UTC).isoformat()
        clock = time.monotonic()
        payload = self._request(document)
        results = payload.get("results") or {}
        if not isinstance(results, dict) or len(results) != 1:
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
        # One request for the whole paper: MinerU's API has no per-page call, so a retry re-sends all of it.
        response = self._post("/file_parse", "file_parse", files=files, data=data)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ParserError(self.backend, "http", f"response is not JSON: {response.text[:300]}") from exc
        if not isinstance(payload, dict):
            raise ParserError(self.backend, "http", f"response is not a JSON object: {str(payload)[:300]}")
        return payload


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


class PaddleHttpParser(_HttpParser):
    """One request per page, each retried on its own, so a transient error at page 28 costs one page.

    Pages that did finish are not kept across a *failed* parse: the template discards the staging directory
    of any run that fails, which is what guarantees a previous good output is never mixed with a partial new
    one. Resuming would need a staging directory that outlives failures and a rule for when its pages are
    still valid (same PDF, same DPI, same service version), and the per-page retry already covers the
    transient errors that made a resume worth wanting.
    """

    backend: Backend = "paddleocr_vl"

    def __init__(
        self,
        base_url: str,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        render_dpi: int = DEFAULT_RENDER_DPI,
        retry_attempts: int = HTTP_RETRY_ATTEMPTS,
        retry_backoff_s: float = HTTP_RETRY_BACKOFF_S,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__(
            base_url,
            client=client,
            timeout_s=timeout_s,
            retry_attempts=retry_attempts,
            retry_backoff_s=retry_backoff_s,
            sleep=sleep,
        )
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
        response = self._post("/layout-parsing", f"page {index}", json=payload)
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
