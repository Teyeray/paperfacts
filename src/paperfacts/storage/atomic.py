"""Atomic writes: write to a temp file in the same directory, then ``os.replace``.

Same contract as ``raw/meta.json`` and ``parsed/*.artifact.json`` — **a file's existence means its
content is complete**. If the process is killed mid-write, the disk holds either the complete old
file or nothing at all; it never leaves a truncated file behind.

The temp filename is unique per call, not just per process: a web server's worker threads may write
the same target concurrently (two tabs uploading the same PDF, or requesting the same rendered page
at once). Sharing one temp name would make the later writer's ``os.replace`` find it already moved.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path


def write_atomic(path: Path, write: Callable[[Path], None]) -> None:
    """``write(tmp_path)`` writes the content to a temp file; on success it is atomically replaced onto ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    write_atomic(path, lambda tmp: tmp.write_bytes(data))


def write_text_atomic(path: Path, text: str) -> None:
    write_atomic(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
