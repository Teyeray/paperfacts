"""AC-16: README and CLAUDE.md keep up with the code they describe.

A profile author learns the file format from README alone, so every key the loader accepts, every CLI command
and every ``PAPERFACTS_*`` variable must be named there; CLAUDE.md states which modules each cache key hashes,
so it must name every module a key hashes and every module held out of the keys. Names only: the prose that
explains them is reviewed, not tested.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

import paperfacts
from paperfacts import keys, profile_loader
from paperfacts.cli import app
from paperfacts.fields import FieldSpec
from paperfacts.profile import FigureSlots, GroupSpec, PromptSlots
from paperfacts.ui_copy import UiCopy

SOURCE = Path(paperfacts.__file__).parent
REPO = SOURCE.parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")
CLAUDE = (REPO / "CLAUDE.md").read_text(encoding="utf-8")


def _names(cls: type) -> list[str]:
    return [item.name for item in dataclasses.fields(cls)]


def _missing(names: list[str], text: str) -> list[str]:
    return [name for name in names if f"`{name}`" not in text and f"`{name} " not in text]


@pytest.mark.parametrize("cls", [PromptSlots, FigureSlots, UiCopy, FieldSpec, GroupSpec], ids=lambda cls: cls.__name__)
def test_every_profile_attribute_is_in_the_readme(cls):
    assert _missing(_names(cls), README) == []


def test_every_top_level_profile_key_and_unit_key_is_in_the_readme():
    assert _missing(list(profile_loader._TOP_KEYS), README) == []
    assert _missing(list(profile_loader._UNIT_KEYS), README) == []


def test_every_cli_command_is_in_the_readme_command_table():
    commands = [command.name or command.callback.__name__ for command in app.registered_commands]
    assert commands  # the registry was read
    undocumented = [name for name in commands if not re.search(rf"^\| `{name}\b", README, re.MULTILINE)]
    assert undocumented == []


def test_every_environment_variable_config_reads_is_in_the_readme():
    source = (SOURCE / "config.py").read_text(encoding="utf-8")
    variables = set(re.findall(r'(?<![.\w])(?:get|number|_parse_bool)\(\s*"([A-Z][A-Z0-9_]*)"', source))
    assert {"PROFILE", "LLM_OFFLINE", "DATA_ROOT"} <= variables  # the pattern still finds the overrides
    assert sorted(name for name in variables if f"PAPERFACTS_{name}" not in README) == []


def test_the_authoring_walkthrough_names_its_tools():
    for text in (
        "cp profiles/battery_cathode.json",
        "paperfacts profiles --check",
        "paperfacts prompts --profile",
        "PAPERFACTS_PROFILE=",
        "--offline",
        "PAPERFACTS_LLM_OFFLINE",
        "LLM calls ≈",
    ):
        assert text in README, text


def test_claude_md_names_every_hashed_and_every_unhashed_module(monkeypatch, tco_profile):
    hashed: set[str] = set()
    real = keys.source_fingerprint

    def recording(*module_files: str) -> str:
        hashed.update(module_files)
        return real(*module_files)

    monkeypatch.setattr(keys, "source_fingerprint", recording)
    # Through __wrapped__ so an earlier test's cache cannot hide a list.
    keys.extraction_code_fingerprint.__wrapped__()
    keys.normalization_fingerprint.__wrapped__()
    keys.comparison_code_fingerprint.__wrapped__()
    keys.retrieval_fingerprint.__wrapped__(tco_profile)
    keys.figure_key(tco_profile, "model", dpi=200, max_pixels=1, max_per_document=1)
    assert "extract.py" in hashed  # the recording saw the lists

    unhashed = {"workbook.py", "readings.py", "ui_copy.py", "llm.py", "config.py", "cli.py", "workflow.py", "batch.py"}
    policy = CLAUDE[CLAUDE.index("Hashed module sources") :]
    missing = sorted(name for name in hashed | unhashed if f"`{name.removesuffix('.py')}`" not in policy)
    assert missing == []
