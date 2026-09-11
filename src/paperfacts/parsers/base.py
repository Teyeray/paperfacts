"""Protocol, shared error type, and the caching template every parser implementation reuses.

A parser's job is deliberately kept minimal: **turn a PDF into native output plus meta.json** in
some directory, nothing more. It never does format conversion (that's the adapter's job) and
doesn't care how downstream code uses the result. This buys us two things:

- :class:`~paperfacts.parsers.subprocess_parser.SubprocessParser` (dev machine / Mac) and
  :mod:`~paperfacts.parsers.http_parser` (server) produce **identical** directory layouts, so
  adapters and every pure function downstream never need to change;
- the main package never imports mineru / paddleocr — dependency conflicts stay isolated in
  ``runners/`` and the deployment images.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Protocol

from paperfacts.errors import PaperFactsError
from paperfacts.models.artifact import Backend, DocumentInput
from paperfacts.models.raw_output import META_FILENAME, ParserMeta, RawParseOutput

logger = logging.getLogger(__name__)


class ParserError(PaperFactsError):
    """A parser run failed.

    ``stage`` names where it failed (launch / run / timeout / http / output / cache); ``detail``
    keeps the original message.
    """

    def __init__(self, backend: Backend, stage: str, detail: str) -> None:
        self.backend = backend
        self.stage = stage
        self.detail = detail
        super().__init__(f"[{backend}] {stage} failed: {detail}")


class DocumentParser(Protocol):
    """The protocol every parser implementation follows.

    ``parse`` must be idempotent: when ``out_dir`` already holds complete output (a ``meta.json``
    exists) and ``force`` is False, it returns a ``cache_hit=True`` result immediately instead of
    rerunning the expensive model (design doc §22).
    """

    backend: Backend

    def parse(self, document: DocumentInput, out_dir: Path, *, force: bool = False) -> RawParseOutput: ...


class CachingParser:
    """Template method for parsing: check cache, clear the directory, produce, write meta.json, validate.

    Subclasses only implement :meth:`_produce`: returning a :class:`ParserMeta` makes this class
    write ``meta.json`` last; returning ``None`` means ``meta.json`` was already written
    externally (the runner subprocess writes its own).
    """

    backend: Backend

    def parse(self, document: DocumentInput, out_dir: Path, *, force: bool = False) -> RawParseOutput:
        if not force and (out_dir / META_FILENAME).is_file():
            logger.info("cache_hit backend=%s doc=%s", self.backend, document.document_id[:16])
            try:
                return RawParseOutput.load(out_dir, self.backend, cache_hit=True)
            except ValueError as exc:
                # meta.json exists but doesn't match the schema (likely written by an older
                # runner): fail loudly instead of silently rerunning and masking a contract change.
                raise ParserError(self.backend, "cache", f"{exc}; rerun with --force once confirmed") from exc

        # Keep this cheap check before we delete anything: a missing runner script shouldn't
        # destroy a previous good output before we even know we can rerun.
        self._check_ready()
        # A cache miss means we're really going to run, so always start from a clean directory.
        # This covers both --force and leftovers from a run that failed halfway through: stray
        # old files would otherwise make this run's result point at the previous one.
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
        """Optional pre-flight check run before anything destructive. Should raise :class:`ParserError` on failure."""

    def _produce(self, document: DocumentInput, out_dir: Path) -> ParserMeta | None:
        """Write native output into ``out_dir``. Subclasses must implement this."""
        raise NotImplementedError


def write_meta(out_dir: Path, meta: ParserMeta) -> None:
    """Write meta.json last: its existence is what marks this output as complete (matches the runner convention)."""
    (out_dir / META_FILENAME).write_text(meta.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
