"""Every prompt the TCO configuration sends is pinned byte for byte.

The LLM cache is keyed by the request payload. A refactor that changes one byte of a prompt silently turns a
free replay of the whole corpus into a paid re-extraction and can move the gold set's cells, so any change to
what the model is asked has to be deliberate: regenerate with ``tests/fixtures/prompts/generate.py`` and say so
in the commit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "prompts"))
from generate import SNAPSHOT, snapshot

RECORDED: dict[str, str] = json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_the_same_prompts_exist_as_were_recorded():
    assert sorted(snapshot()) == sorted(RECORDED)


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_each_prompt_is_byte_identical_to_the_recording(name):
    assert snapshot()[name] == RECORDED[name]
