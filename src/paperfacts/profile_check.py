"""Checking a pasted profile for the web's check page: the errors ``paperfacts profiles --check`` would print, or the
definition and system prompts the profile would get.

The text is untrusted, and parts of validation are expensive on hostile input: every unit spelling goes through
``text.normalize_text``, whose markup pattern backtracks polynomially in a long run of spaces while holding the GIL, and
the unit tables fill process-wide ``functools.cache``s. So the check never runs in the server: it runs in a short-lived
child process (``run_check``) with a wall-clock timeout, CPU and file-size limits, a memory limit where the OS allows
one (Linux; macOS refuses it), a bounded read of its answer, no environment (so no API key) and a neutral working
directory. Whatever it caches dies with it. Inside the child the text is size-capped by the caller, depth-scanned before
``json.loads`` (a deep nesting would raise RecursionError in the parser and in the loader's error text), and parsed with
NaN and infinities refused.

The pasted profile's name only labels its errors (``<name>.json``); it is never joined to a path and never read
from or written to disk, and nothing here calls ``load_profile`` or a key function.

Unhashed (``tests/test_keys_unhashed.py``): it only reports what the loader decides.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from paperfacts.config import Settings
from paperfacts.errors import ConfigError, ProfileCheckError
from paperfacts.profile_loader import IDENTIFIER, parse_profile
from paperfacts.profile_view import profile_definition, prompt_sections
from paperfacts.workflow import LEGACY_EXPORT_NAME, check_mode

logger = logging.getLogger(__name__)

# The largest shipped profile is about 23 KB.
MAX_CHECK_BYTES = 256 * 1024
MAX_DEPTH = 64
TIMEOUT_S = 10.0
MEMORY_LIMIT_BYTES = 1 << 30
# A valid profile's preview is bounded by the loader's caps (100 fields, 2000-character slots); anything larger is
# not the child's answer, and no more than one byte past it is ever read from the child.
MAX_OUTPUT_BYTES = 16 << 20
# The options line ahead of the text on the child's stdin (the served names and hashes, the mode, the field).
MAX_OPTIONS_BYTES = 1 << 20
# The child finds the package by the path it is handed, not by its environment, which it runs without.
_CHILD = "import sys; sys.path.insert(0, sys.argv[1]); from paperfacts.profile_check import child_main; child_main()"


def run_check(
    raw: bytes, *, served: Mapping[str, str], extraction_mode: str, field: str | None = None, timeout: float = TIMEOUT_S
) -> bytes:
    """The check result of ``raw`` as JSON bytes, computed in a child process. ``served`` maps each served profile's
    name to its content hash. Raises ProfileCheckError when the child cannot start, times out or does not answer.

    The options go on stdin, one JSON line ahead of the text, never on argv: the field name is the caller's, and an
    argument list has an OS-wide size limit."""
    options = json.dumps({"served": dict(served), "extraction_mode": extraction_mode, "field": field})
    package_root = str(Path(__file__).resolve().parent.parent)
    try:
        child = subprocess.Popen(
            # -I: no PYTHON* variables, user site or script directory; -B: no bytecode written.
            [sys.executable, "-I", "-B", "-c", _CHILD, package_root],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={},
            cwd="/",
        )
    except OSError as exc:
        logger.warning("a profile check could not start: %s", exc)
        raise ProfileCheckError("检查未能启动 (the check process could not start)", retryable=True) from None
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        child.kill()

    # Killing the child on time also ends every blocking read and write below. stderr is drained beside stdout (a
    # child blocked on a full stderr pipe would never finish stdout), keeping only its tail for the log.
    timer = threading.Timer(timeout, expire)
    stderr_tail = bytearray()
    drain = threading.Thread(target=_keep_tail, args=(child.stderr, stderr_tail), daemon=True)
    with child:
        timer.start()
        drain.start()
        try:
            # A child that exits (or is killed) without reading it all breaks the pipe; its exit status says why.
            with contextlib.suppress(OSError):
                child.stdin.write(options.encode("utf-8") + b"\n" + raw)
            with contextlib.suppress(OSError):
                child.stdin.close()
            output = child.stdout.read(MAX_OUTPUT_BYTES + 1)
        finally:
            timer.cancel()
            child.kill()  # a no-op once it has exited; ends one that is still writing past the cap
            child.wait()
            drain.join()
    if timed_out.is_set():
        logger.warning("a profile check ran past %.0f s and was killed", timeout)
        raise ProfileCheckError("检查超时 (the check ran out of time)", retryable=True)
    if child.returncode != 0 or not output.startswith(b"{") or len(output) > MAX_OUTPUT_BYTES:
        logger.warning(
            "a profile check failed (exit %s, %d bytes out): %s",
            child.returncode,
            len(output),
            bytes(stderr_tail).decode("utf-8", "replace"),
        )
        raise ProfileCheckError("检查未能完成 (the check process failed)", retryable=False)
    return output


def _keep_tail(stream: Any, tail: bytearray, keep: int = 2000) -> None:
    while chunk := stream.read1(65536):
        tail.extend(chunk)
        del tail[:-keep]


class _Collect(logging.Handler):
    """Every record it is handed, in order: a warning the loader logs is part of the answer and must not be dropped."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def child_main() -> None:
    """The child's body: limits first, then one check of stdin, answered as JSON on stdout."""
    _limit_resources()
    options = json.loads(sys.stdin.buffer.readline(MAX_OPTIONS_BYTES))
    raw = sys.stdin.buffer.read(MAX_CHECK_BYTES + 1)
    result = check_profile(raw, **options)
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _limit_resources() -> None:
    """Best effort: macOS refuses RLIMIT_AS, and the wall-clock timeout is the bound that always holds. Set here, in
    the child, rather than through ``preexec_fn``, which is unsafe in a parent with threads (the server has many)."""
    try:
        import resource
    except ImportError:  # not POSIX
        return
    limits = (
        (resource.RLIMIT_AS, MEMORY_LIMIT_BYTES),
        (resource.RLIMIT_CPU, math.ceil(TIMEOUT_S) + 1),
        # Nothing here writes a file; a write would fail rather than land anywhere.
        (resource.RLIMIT_FSIZE, 0),
    )
    for which, value in limits:
        try:
            resource.setrlimit(which, (value, value))
        except (ValueError, OSError):
            pass


def check_profile(
    raw: bytes, *, served: Mapping[str, str], extraction_mode: str, field: str | None = None
) -> dict[str, Any]:
    """What checking ``raw`` finds: ``errors`` one per line as the CLI prints them, and for a valid profile the
    warnings, notes, a comparison with a served profile of the same name, its definition and its prompts."""
    result: dict[str, Any] = {
        "ok": False,
        "errors": [],
        "warnings": [],
        "notes": [],
        "same_name": None,
        "definition": None,
        "prompts": None,
    }
    data, error = _parse(raw)
    if error is not None:
        result["errors"] = [error]
        return result
    name = data.get("name") if isinstance(data, dict) else None
    # A label for the errors only, never a path that is opened.
    source = Path(f"{name}.json" if isinstance(name, str) and IDENTIFIER.fullmatch(name) else "pasted.json")
    # The loader logs its warnings (a large field table); the CLI prints them, and so does this.
    handler = _Collect()
    handler.setLevel(logging.WARNING)
    package_logger = logging.getLogger("paperfacts")
    package_logger.addHandler(handler)
    try:
        profile = parse_profile(data, source)
    except ConfigError as exc:
        result["errors"] = str(exc).splitlines()
        return result
    finally:
        package_logger.removeHandler(handler)
    if profile.name == LEGACY_EXPORT_NAME:
        result["errors"] = [f"{source}: the profile name {LEGACY_EXPORT_NAME!r} is reserved for old exports"]
        return result
    result["warnings"] = [record.getMessage() for record in handler.records]
    notes: list[str] = []
    if profile.declared_entities:
        names = ", ".join(entity.name for entity in profile.declared_entities)
        notes.append(f"entity types {names}; runs only in passage mode (extraction.mode 'passage')")
    try:
        check_mode(profile, Settings(extraction_mode=extraction_mode))  # type: ignore[arg-type]
    except ConfigError as exc:
        notes.append(str(exc))
    if profile.name in served:
        result["same_name"] = {"name": profile.name, "same_content_hash": served[profile.name] == profile.content_hash}
    try:
        sections = prompt_sections(profile, field)
    except KeyError as exc:
        notes.append(str(exc.args[0]))
        sections = prompt_sections(profile)
    result.update(
        ok=True,
        notes=notes,
        definition=profile_definition(profile),
        prompts=[{"title": title, "text": text} for title, text in sections],
    )
    return result


def _parse(raw: bytes) -> tuple[Any, str | None]:
    """The JSON value of ``raw``, or why there is none."""
    if len(raw) > MAX_CHECK_BYTES:
        return None, f"the profile is larger than {MAX_CHECK_BYTES // 1024} KiB"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, f"the profile is not UTF-8 text: byte {exc.start}"
    if nesting_exceeds(text, MAX_DEPTH):
        return None, f"the profile nests objects and lists more than {MAX_DEPTH} deep"
    try:
        value = json.loads(text, parse_constant=_refuse_constant, parse_float=_finite)
    except (ValueError, RecursionError) as exc:
        return None, f"the profile is not valid JSON: {exc}"
    # json.loads accepts a lone \ud800, which no UTF-8 answer can carry; parse_profile refuses it (message escaped).
    return value, None


def nesting_exceeds(text: str, limit: int) -> bool:
    """Does ``text`` open more than ``limit`` objects or lists inside one another? One pass, no recursion, strings
    skipped; malformed text is left for the parser to name."""
    depth = 0
    in_string = escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > limit:
                return True
        elif character in "]}":
            depth -= 1
    return False


def _refuse_constant(name: str) -> float:
    raise ValueError(f"{name} is not a number a profile may hold")


def _finite(text: str) -> float:
    # A float literal too large to represent (1e999) parses as infinity without passing through parse_constant.
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a finite number")
    return value
