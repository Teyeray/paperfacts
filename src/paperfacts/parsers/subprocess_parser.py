"""Dev-machine parser implementation: runs ``runners/*.py`` as a subprocess.

Each runner is a PEP 723 script that declares its own dependencies in its header and executes via
``uv run --locked --script`` in its own isolated cache environment (MinerU's and PaddleOCR's
dependencies conflict and can't share one env). The main process only builds the command line and
waits for the exit code; ``meta.json`` is written by the runner itself, and caching/validation is
delegated to :class:`~paperfacts.parsers.base.CachingParser`. The runner's stderr only makes it
into the exception message on failure.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from paperfacts.models.artifact import Backend, DocumentInput
from paperfacts.parsers.base import CachingParser, ParserError

logger = logging.getLogger(__name__)

# uv's script mode: --locked makes a stale lockfile fail outright instead of silently resolving a
# different set of dependencies.
DEFAULT_COMMAND_PREFIX: tuple[str, ...] = ("uv", "run", "--locked", "--script")
RUNNER_SCRIPTS: dict[Backend, str] = {
    "mineru": "runners/mineru_runner.py",
    "paddleocr_vl": "runners/paddle_runner.py",
}
# On failure, include this many lines from the end of stderr in the exception: enough to
# diagnose without flooding the terminal.
STDERR_TAIL_LINES = 40


def default_runner_script(repo_root: Path, backend: Backend) -> Path:
    """The runner script path for a given backend (relative to the repo root)."""
    return repo_root / RUNNER_SCRIPTS[backend]


class SubprocessParser(CachingParser):
    """Runs a runner script as a subprocess."""

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
        """Build the full command line; exposed separately so it's easy to test and log."""
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
            completed = subprocess.run(  # command is built by this module, no user-controlled shell content
                cmd, capture_output=True, text=True, timeout=self.timeout_s, env=merged_env
            )
        except FileNotFoundError as exc:
            raise ParserError(self.backend, "launch", f"executable not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            detail = f"did not finish within {self.timeout_s}s: {_tail(exc.stderr)}"
            raise ParserError(self.backend, "timeout", detail) from exc

        if completed.returncode != 0:
            raise ParserError(
                self.backend, "run", f"exit code {completed.returncode}, stderr tail:\n{_tail(completed.stderr)}"
            )
        # meta.json is written by the runner itself; returning None tells the base class to
        # validate and load it directly.


def _tail(text: str | bytes | None, lines: int = STDERR_TAIL_LINES) -> str:
    if not text:
        return "(empty)"
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return "\n".join(text.strip().splitlines()[-lines:])
