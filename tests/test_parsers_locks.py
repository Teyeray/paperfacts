"""The parse locks: one run per parser at a time, the two parsers side by side, cache hits never waiting.

Documents are processed in parallel, but a parser service has one GPU. These tests hold one run inside
``_produce`` with an event and look at what the others do meanwhile; nothing sleeps, and every wait has a
timeout so a regression fails instead of hanging.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from paperfacts.models import Backend, DocumentInput, ParserMeta
from paperfacts.parsers import MinerUHttpParser, PaddleHttpParser, Parser, SubprocessParser
from support.web import WAIT_TIMEOUT_S, wait_until
from test_parsers_base import make_meta


class GatedParser(Parser):
    """Enters ``_produce``, records how many runs of its lock are inside at once, and waits for ``release``."""

    def __init__(self, backend: Backend, *, release: threading.Event, barrier: threading.Barrier | None = None):
        self.backend = backend
        self.release = release
        self.barrier = barrier
        self.inside = 0
        self.peak = 0
        self.entered: list[Path] = []
        self._lock = threading.Lock()

    def _produce(self, document: DocumentInput, out_dir: Path) -> ParserMeta:
        with self._lock:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
            self.entered.append(out_dir)
        try:
            if self.barrier is not None:
                self.barrier.wait(timeout=WAIT_TIMEOUT_S)  # BrokenBarrierError unless the other parser is inside too
            if not self.release.wait(timeout=WAIT_TIMEOUT_S):
                raise AssertionError("the test never released the parser")
            (out_dir / "native.json").write_text("[]", encoding="utf-8")
            return make_meta(self.backend)
        finally:
            with self._lock:
                self.inside -= 1


def test_one_parser_parses_one_document_at_a_time(tmp_path: Path, document: DocumentInput, caplog):
    caplog.set_level(logging.INFO, logger="paperfacts.parsers")
    release = threading.Event()
    parser = GatedParser("mineru", release=release)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(parser.parse, document, tmp_path / "a" / "raw")
        wait_until(lambda: parser.inside == 1, what="the first run to start")
        second = pool.submit(parser.parse, document, tmp_path / "b" / "raw")
        # The second run announces that it waits, and is then really waiting: it has not entered _produce.
        wait_until(lambda: "waiting for the mineru parser" in caplog.text, what="the second run to queue")
        assert len(parser.entered) == 1
        release.set()
        first.result(timeout=WAIT_TIMEOUT_S)
        second.result(timeout=WAIT_TIMEOUT_S)

    assert parser.peak == 1
    assert len(parser.entered) == 2


def test_the_two_parsers_run_side_by_side(tmp_path: Path, document: DocumentInput):
    # A barrier of two only opens when both runs are inside _produce at the same moment.
    release, barrier = threading.Event(), threading.Barrier(2)
    release.set()
    mineru = GatedParser("mineru", release=release, barrier=barrier)
    paddle = GatedParser("paddleocr_vl", release=release, barrier=barrier)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(mineru.parse, document, tmp_path / "a" / "raw"),
            pool.submit(paddle.parse, document, tmp_path / "b" / "raw"),
        ]
        for future in futures:
            future.result(timeout=WAIT_TIMEOUT_S)

    assert not barrier.broken


def test_a_cache_hit_does_not_wait_for_a_running_parse(tmp_path: Path, document: DocumentInput):
    release = threading.Event()
    parser = GatedParser("mineru", release=release)
    cached_dir = tmp_path / "cached" / "raw"
    release.set()
    parser.parse(document, cached_dir)
    release.clear()

    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(parser.parse, document, tmp_path / "running" / "raw")
        wait_until(lambda: parser.inside == 1, what="a run to hold the lock")
        try:
            hit = pool.submit(parser.parse, document, cached_dir).result(timeout=WAIT_TIMEOUT_S)
        finally:
            release.set()
        running.result(timeout=WAIT_TIMEOUT_S)

    assert hit.cache_hit is True


def test_a_failed_run_releases_the_lock(tmp_path: Path, document: DocumentInput):
    class Failing(GatedParser):
        def _produce(self, document: DocumentInput, out_dir: Path) -> ParserMeta:
            raise RuntimeError("the GPU fell over")

    release = threading.Event()
    release.set()
    with pytest.raises(RuntimeError):
        Failing("mineru", release=release).parse(document, tmp_path / "a" / "raw")

    # The next run of the same parser gets the lock; a leaked one would hang it until the timeout.
    with ThreadPoolExecutor(max_workers=1) as pool:
        raw = pool.submit(GatedParser("mineru", release=release).parse, document, tmp_path / "b" / "raw")
        assert raw.result(timeout=WAIT_TIMEOUT_S).cache_hit is False


def test_http_parsers_lock_per_backend_and_the_runners_share_one_lock(tmp_path: Path):
    """On the server the two parsers are separate services; on a laptop the two runners would load both
    model sets into one machine's memory at once."""
    mineru_http = MinerUHttpParser("http://gpu:8002")
    paddle_http = PaddleHttpParser("http://gpu:8080")
    mineru_runner = SubprocessParser("mineru", tmp_path / "mineru_runner.py")
    paddle_runner = SubprocessParser("paddleocr_vl", tmp_path / "paddle_runner.py")

    assert mineru_http.lock_key != paddle_http.lock_key
    assert mineru_runner.lock_key == paddle_runner.lock_key
    assert mineru_runner.lock_key not in {mineru_http.lock_key, paddle_http.lock_key}
