"""Atomic writes: existence means completeness — being killed mid-write must never leave a truncated file."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from paperfacts.compare import compare_lanes
from paperfacts.matching import SampleMatching
from paperfacts.storage import write_atomic, write_bytes_atomic, write_text_atomic
from support.extraction import make_artifact, make_lane
from support.factories import make_block


def test_bytes_are_written_and_parent_directories_are_created(tmp_path: Path):
    target = tmp_path / "deep" / "er" / "file.bin"

    write_bytes_atomic(target, b"\x00\x01")

    assert target.read_bytes() == b"\x00\x01"


def test_text_is_written_as_utf8(tmp_path: Path):
    # Non-ASCII on purpose: identity.json holds uploaded filenames, which are routinely not ASCII.
    target = tmp_path / "identity.json"

    write_text_atomic(target, '{"name": "薄膜.pdf"}')

    assert target.read_text(encoding="utf-8") == '{"name": "薄膜.pdf"}'
    assert "薄膜" in target.read_bytes().decode("utf-8")


def test_a_failing_write_leaves_neither_the_target_nor_a_temp_file(tmp_path: Path):
    target = tmp_path / "out" / "page.png"

    def explode(tmp: Path) -> None:
        tmp.write_bytes(b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError):
        write_atomic(target, explode)

    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_a_failing_rewrite_keeps_the_old_content_intact(tmp_path: Path):
    target = tmp_path / "file.txt"
    write_text_atomic(target, "old")

    def explode(tmp: Path) -> None:
        tmp.write_text("new", encoding="utf-8")
        raise RuntimeError("crashed midway")

    with pytest.raises(RuntimeError):
        write_atomic(target, explode)

    assert target.read_text(encoding="utf-8") == "old"


def test_a_rewrite_replaces_the_content(tmp_path: Path):
    target = tmp_path / "file.txt"
    write_text_atomic(target, "old")

    write_text_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["file.txt"]  # no leftover temp file


def test_concurrent_writers_to_the_same_path_all_succeed(tmp_path: Path):
    """A web server's worker threads can write the same target concurrently (two tabs uploading the
    same PDF, or requesting the same rendered page, at once).

    If the temp filename were only distinguished per process, a later writer's ``os.replace`` would
    find the temp file already moved by an earlier one and fail.
    """
    target = tmp_path / "page.png"
    payloads = [bytes([i]) * 1024 for i in range(16)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda data: write_bytes_atomic(target, data), payloads))

    assert target.read_bytes() in payloads
    assert [p.name for p in tmp_path.iterdir()] == ["page.png"]


def _stored_models():
    lane_a, lane_b = make_lane(backend="mineru"), make_lane(backend="paddleocr_vl")
    return {
        "lane": lane_a,
        "comparison": compare_lanes(lane_a, lane_b, SampleMatching.trivial((), [], [])),
        "artifact": make_artifact((make_block(page=0, order=0),)),
    }


@pytest.mark.parametrize("kind", ["lane", "comparison", "artifact"])
def test_every_stored_model_is_written_atomically(tmp_path: Path, monkeypatch, kind: str):
    # The web server reads these files while a job writes them, and a killed run must not leave a torn one
    # that fails model_validate_json on every later run.
    model = _stored_models()[kind]
    target = tmp_path / f"{kind}.json"
    model.write(target)
    before = target.read_text(encoding="utf-8")

    def interrupted(src, dst):
        raise OSError("killed before the rename")

    monkeypatch.setattr("paperfacts.storage.os.replace", interrupted)
    with pytest.raises(OSError, match="killed"):
        model.write(target)

    assert target.read_text(encoding="utf-8") == before
    assert [path.name for path in tmp_path.iterdir()] == [target.name]
