"""Every request a passage-mode run sends over the two recorded real parses is pinned, with its cache key.

The LLM cache is keyed by the request payload, so a byte that moves anywhere in a request -- the rendered
Markdown, the blocks retrieval picked, the sample list, the matching listing built from cleaned samples --
turns a free replay of the corpus into a paid one. ``tests/fixtures/payloads/b0.json`` was recorded at B0 by
``tests/fixtures/payloads/generate_b0.py``; see its docstring before ever re-recording it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "payloads"))
from generate_b0 import RECORDING, record

# Pinned for the same reason as the prompt snapshot's: the comparison runs the generator's own code, so a
# re-record must not be able to pass unnoticed.
RECORDING_SHA256 = "761fa36ab95399cfba646c1c51f116ccd616e5353af1920410bed651cbb11204"

RECORDED = json.loads(RECORDING.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fresh(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return record(tmp_path_factory.mktemp("payloads"))


def test_the_recording_itself_is_the_pinned_one():
    assert hashlib.sha256(RECORDING.read_bytes()).hexdigest() == RECORDING_SHA256


def test_the_same_questions_are_asked(fresh: dict):
    assert [request["label"] for request in fresh["requests"]] == [request["label"] for request in RECORDED["requests"]]


def test_the_system_prompts_are_byte_identical(fresh: dict):
    assert fresh["systems"] == RECORDED["systems"]


def test_each_request_is_byte_identical_and_keeps_its_cache_key(fresh: dict):
    for new, old in zip(fresh["requests"], RECORDED["requests"], strict=True):
        assert new["user"] == old["user"], new["label"]
        assert new["system"] == old["system"], new["label"]
        assert new["cache_key"] == old["cache_key"], new["label"]
