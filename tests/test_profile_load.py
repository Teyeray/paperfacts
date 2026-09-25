"""``profiles/tco.json``: the shipped profile says exactly what the running code says about TCO.

Until the code reads its fields from the profile, the field table still comes from ``config.json``. This file
is what keeps the two from drifting apart: every attribute the table had is equal field by field, and the
attributes that replace a special case the code makes by field name carry exactly that case.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from paperfacts import figures, passages
from paperfacts.config import DEFAULT_REPO_ROOT, Settings
from paperfacts.errors import ConfigError
from paperfacts.fields import CONDITION_KEYWORDS, FIELD_SPECS, FieldSpec
from paperfacts.profile import FigureSlots, PromptSlots, load_profile, profile_path
from paperfacts.prompts import (
    extraction_system_prompt,
    field_system_prompt,
    inventory_system_prompt,
    matching_system_prompt,
)
from support.profiles import make_profile

# The attributes that restate a special case the code makes by field name; everything else must equal the
# config.json table as it is.
NAME_BASED = {"condition_rule", "missing_condition_note_zh", "figure_readable", "display_format"}


def test_the_tco_fields_are_the_running_field_table_attribute_by_attribute(tco_profile):
    assert [spec.name for spec in tco_profile.fields] == [spec.name for spec in FIELD_SPECS]
    for loaded, running in zip(tco_profile.fields, FIELD_SPECS, strict=True):
        for attribute in dataclasses.fields(FieldSpec):
            if attribute.name not in NAME_BASED:
                assert getattr(loaded, attribute.name) == getattr(running, attribute.name), (
                    loaded.name,
                    attribute.name,
                )


def test_the_tco_levels_are_the_scope_the_code_applies(tco_profile):
    assert {spec.name for spec in tco_profile.paper_fields} == {
        spec.name for spec in FIELD_SPECS if spec.group == "target"
    }
    assert [(group.name, group.level) for group in tco_profile.groups] == [
        ("target", "paper"),
        ("process", "sample"),
        ("film", "sample"),
    ]


def test_the_name_based_special_cases_became_attributes(tco_profile):
    def where(attribute: str) -> dict[str, object]:
        default = next(item.default for item in dataclasses.fields(FieldSpec) if item.name == attribute)
        return {
            spec.name: getattr(spec, attribute) for spec in tco_profile.fields if getattr(spec, attribute) != default
        }

    # Rule 8 of both extraction prompts, and decide.py's note when the wavelength is missing.
    assert where("condition_rule") == {"transmittance": "the wavelength or spectral range"}
    assert where("missing_condition_note_zh") == {"transmittance": "原文提取结果未注明透光率波长或波段"}
    # figures.figure_fields(): the numeric film fields.
    assert set(where("figure_readable")) == {spec.name for spec in figures.figure_fields()}
    assert [spec.name for spec in tco_profile.figure_fields] == [spec.name for spec in figures.figure_fields()]
    # The workbook's scientific number format.
    assert where("display_format") == {"resistance": "scientific", "resistivity": "scientific"}


def test_the_tco_retrieval_is_the_running_one(tco_profile):
    assert tco_profile.retrieval.condition_keywords == CONDITION_KEYWORDS
    assert tco_profile.retrieval.condition_unit_pattern == passages.CONDITION_UNIT.pattern
    assert passages.CONDITION_UNIT.flags & re.IGNORECASE


def test_every_tco_slot_is_text_the_prompts_already_send(tco_profile):
    # A slot is a slice of today's prompt, so each one must occur in it verbatim; the snapshot proves the
    # reassembly byte for byte once the prompts are rendered from the slots.
    sent = "\n".join(
        (extraction_system_prompt(), inventory_system_prompt(), field_system_prompt(), matching_system_prompt())
    )
    for slot in dataclasses.fields(PromptSlots):
        assert getattr(tco_profile.prompt, slot.name) in sent, slot.name
    assert tco_profile.figures is not None
    for slot in dataclasses.fields(FigureSlots):
        assert getattr(tco_profile.figures, slot.name) in figures.USER_PROMPT, slot.name


def test_the_tco_profile_declares_no_units_of_its_own(tco_profile):
    assert tco_profile.units.declared == ()
    assert tco_profile.units.material() == []


# ---- Selection -------------------------------------------------------------------------


def test_the_shipped_configuration_selects_the_tco_profile():
    settings = Settings.from_env({})

    assert settings.profile == "tco"
    assert profile_path(settings) == DEFAULT_REPO_ROOT / "profiles" / "tco.json"


def test_the_environment_selects_another_profile():
    assert Settings.from_env({"PAPERFACTS_PROFILE": "battery"}).profile == "battery"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("battery", Path("/repo/profiles/battery.json"), id="bare-name"),
        pytest.param("elsewhere/battery.json", Path("elsewhere/battery.json"), id="relative-path"),
        pytest.param("/abs/demo.json", Path("/abs/demo.json"), id="absolute-path"),
        pytest.param("demo.json", Path("demo.json"), id="json-suffix"),
    ],
)
def test_a_bare_name_is_looked_up_in_profiles_and_anything_else_is_a_path(value, expected):
    assert profile_path(Settings(repo_root=Path("/repo"), profile=value)) == expected


def test_a_profile_is_read_once_per_file(monkeypatch):
    shipped = profile_path(Settings())
    monkeypatch.chdir(shipped.parent)

    assert load_profile(Path("tco.json")) is load_profile(shipped)


def test_a_missing_profile_names_the_path(tmp_path):
    with pytest.raises(ConfigError, match="no profile at"):
        load_profile(tmp_path / "absent.json")


# ---- The content hash ------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"title_zh": "别的名字"}, id="title"),
        pytest.param({"description_zh": "说明"}, id="description"),
        pytest.param({"maturity": "production"}, id="maturity"),
        pytest.param({"groups.0.label_zh": "别的"}, id="group-label"),
        pytest.param({"fields.1.label": "厚度"}, id="field-label"),
        pytest.param({"ui": {"entity_label_zh": "涂层"}}, id="ui"),
    ],
)
def test_display_text_leaves_the_content_hash_alone(change):
    assert make_profile(change).content_hash == make_profile().content_hash


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"fields.1.description": "Thickness of the dried coating."}, id="description"),
        pytest.param({"fields.1.rel_tol": 0.1}, id="tolerance"),
        pytest.param({"prompt.fact_noun": "coating facts"}, id="slot"),
        pytest.param({"retrieval.condition_keywords": ["annealed"]}, id="retrieval"),
        pytest.param({"units": {"mg/L": {"aliases": {"mg/L": 1}}}}, id="units"),
    ],
)
def test_everything_else_moves_the_content_hash(change):
    assert make_profile(change).content_hash != make_profile().content_hash


def test_a_profile_hashes_by_its_content_hash():
    first, second = make_profile(), make_profile(source=Path("elsewhere/demo.json"))

    assert first == second and hash(first) == hash(second) == hash(first.content_hash)
