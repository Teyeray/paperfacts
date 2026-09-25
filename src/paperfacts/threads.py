"""The pipeline's thread pool: every task runs in a copy of the context of the thread that submitted it.

A plain ``ThreadPoolExecutor`` runs tasks in its worker's own, empty context, so whatever the caller keyed on
a context variable is gone on the pool. The web job queue keys each job's log on one (``web/jobs.py``), and
the pipeline nests pools -- documents, lanes and the figures stage, field questions, chart panels -- so a
task that lost the context would lose its log lines from the job panel, silently. Every pool in the
pipeline is this one, which makes forgetting impossible rather than something each call site must recall.

Deliberately its own module: it decides when work runs, never what is asked, so it stays out of the source
that ``keys.py`` hashes into the cache keys.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """``submit`` (and so ``map``, which submits every item from the calling thread) copies the caller's
    context per task. One copy per task, because a context cannot be entered by two threads at once."""

    def submit[T](self, fn: Callable[..., T], /, *args: object, **kwargs: object) -> Future[T]:
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)
