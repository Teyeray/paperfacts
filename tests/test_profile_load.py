"""``profiles/tco.json``: the shipped profile says exactly what the code said about TCO before it had profiles.

The code read its field table, condition keywords and condition pattern from ``config.json`` and ``passages.py``
until S4e deleted them; ``tests/fixtures/b0_field_table/field_table.json`` is what they were, recorded before the
deletion. Every attribute the table had is equal field by field, and the attributes that replace a special case
the code made by field name carry exactly that case.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from paperfacts import figures
from paperfacts.config import DEFAULT_REPO_ROOT, Settings
from paperfacts.errors import ConfigError
from paperfacts.fields import FieldSpec
from paperfacts.profile import FigureSlots, PromptSlots
from paperfacts.profile_loader import load_profile, profile_path
from paperfacts.prompts import (
    extraction_system_prompt,
    field_system_prompt,
    inventory_system_prompt,
    matching_system_prompt,
)
from paperfacts.units import UnitRegistry
from support.profiles import make_profile

# The attributes that restate a special case the code made by field name; everything else must equal the
# config.json table as it was.
NAME_BASED = {"condition_rule", "missing_condition_note_zh", "figure_readable", "display_format"}
# Attributes added after B0 was recorded, with the value that is B0's behaviour.
AFTER_B0 = {"after_clause": "refuse"}
B0 = json.loads(
    (Path(__file__).parent / "fixtures" / "b0_field_table" / "field_table.json").read_text(encoding="utf-8")
)
# The recorded table as plain JSON; a FieldSpec compares through dataclasses.asdict, lists for tuples.
B0_FIELDS: list[dict[str, object]] = B0["fields"]


def as_json(spec: FieldSpec) -> dict[str, object]:
    return json.loads(json.dumps(dataclasses.asdict(spec)))


def test_the_tco_fields_are_the_b0_field_table_attribute_by_attribute(tco_profile):
    assert [spec.name for spec in tco_profile.fields] == [entry["name"] for entry in B0_FIELDS]
    assert {attribute.name for attribute in dataclasses.fields(FieldSpec)} == set(B0_FIELDS[0]) | set(AFTER_B0)
    for loaded, recorded in zip(tco_profile.fields, B0_FIELDS, strict=True):
        for attribute, value in as_json(loaded).items():
            if attribute in AFTER_B0:
                assert value == AFTER_B0[attribute], (loaded.name, attribute)
            elif attribute not in NAME_BASED:
                assert value == recorded[attribute], (loaded.name, attribute)


def test_the_tco_levels_are_the_scope_the_code_applied(tco_profile):
    assert {spec.name for spec in tco_profile.paper_fields} == {
        entry["name"] for entry in B0_FIELDS if entry["group"] == "target"
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
    # The rule figures.figure_fields() applied before the profile said which fields a chart is read for:
    # the numeric film fields, in table order.
    film = [entry["name"] for entry in B0_FIELDS if entry["group"] == "film" and entry["kind"] == "numeric"]
    assert set(where("figure_readable")) == set(film)
    assert [spec.name for spec in tco_profile.figure_fields] == film
    # The workbook's scientific number format.
    assert where("display_format") == {"resistance": "scientific", "resistivity": "scientific"}


def test_the_tco_retrieval_is_the_b0_one(tco_profile):
    assert list(tco_profile.retrieval.condition_keywords) == B0["condition_keywords"]
    # passages compiles the profile's pattern case-insensitively, as the B0 constant was.
    assert tco_profile.retrieval.condition_unit_pattern == B0["condition_unit_pattern"]
    assert B0["condition_unit_ignorecase"]


def test_every_tco_slot_reaches_the_prompts(tco_profile):
    # Every slot must be used by some template; the snapshot proves the reassembly byte for byte.
    sent = "\n".join(
        (
            extraction_system_prompt(tco_profile),
            inventory_system_prompt(tco_profile),
            field_system_prompt(tco_profile),
            matching_system_prompt(tco_profile),
        )
    )
    for slot in dataclasses.fields(PromptSlots):
        assert getattr(tco_profile.prompt, slot.name) in sent, slot.name
    assert tco_profile.figures is not None
    chart = figures.user_prompt("caption", tco_profile.figure_fields, tco_profile.figures)
    for slot in dataclasses.fields(FigureSlots):
        assert getattr(tco_profile.figures, slot.name) in chart, slot.name


def test_the_tco_profile_declares_no_units_of_its_own_only_the_gas_suffixes(tco_profile):
    assert tco_profile.units.declared == ()
    # Exactly the gas names the code set aside for every domain before a profile declared them.
    assert tco_profile.units == UnitRegistry(ignored_suffixes=("Ar", "O2", "N2", "H2", "He", "Kr", "Xe", "air"))
    assert tco_profile.units.material() == [{"ignored_suffixes": ["Ar", "O2", "N2", "H2", "He", "Kr", "Xe", "air"]}]


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


def test_an_unreadable_profile_is_a_config_error_naming_the_path(tmp_path, monkeypatch):
    path = tmp_path / "locked.json"
    path.write_text("{}", encoding="utf-8")

    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_bytes", refuse)

    with pytest.raises(ConfigError, match=r"cannot read the profile at .*locked\.json"):
        load_profile(path)


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
