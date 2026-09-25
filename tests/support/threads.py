"""Run a call on a daemon thread and get a Future for it.

The concurrency tests exercise locks that wait without a timeout in production code (a parse lock, an
in-flight slot). Run on a ``ThreadPoolExecutor``, a regression that leaks one would hang the suite: leaving
the ``with`` block joins the stuck worker, and so does the interpreter at exit. A daemon thread per call
lets ``future.result(timeout=...)`` fail the test and the process still end.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any


def submit_daemon[T](fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
    future: Future[T] = Future()

    def run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return future
