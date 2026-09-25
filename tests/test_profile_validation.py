"""A profile is edited by hand, so every mistake in one is refused with the file and the key named (AC-12).

Each case starts from the valid demo profile of :mod:`support.profiles` and breaks one thing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from paperfacts.errors import ConfigError
from paperfacts.profile_loader import MAX_SLOT_LENGTH, parse_profile
from support.profiles import DELETE, make_profile, profile_data

SOURCE = Path("profiles/demo.json")


def field(index: int, **changes: Any) -> dict[str, Any]:
    return profile_data()["fields"][index] | changes


def refused(changes: dict[str, Any], source: Path = SOURCE) -> str:
    with pytest.raises(ConfigError) as excinfo:
        parse_profile(profile_data(changes), source)
    message = str(excinfo.value)
    assert str(source) in message
    return message


def test_the_demo_profile_is_valid():
    profile = make_profile()

    assert profile.name == "demo"
    assert [spec.level for spec in profile.fields] == ["paper", "sample", "sample"]
    assert profile.figures is None
    assert profile.prompt.paper_key == "paper"
    assert profile.ui.entity_label_zh == "样品"


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        # Identity
        pytest.param({"feilds": []}, "feilds", id="unknown-top-level-key"),
        pytest.param({"format": 2}, "format", id="format"),
        pytest.param({"format": DELETE}, "format", id="format-missing"),
        pytest.param({"name": "other"}, "file name", id="name-not-the-stem"),
        pytest.param({"maturity": "beta"}, "maturity", id="maturity"),
        # "$" matches before a trailing newline, so every identifier is matched whole.
        pytest.param({"name": "demo\n"}, "name must match", id="name-trailing-newline"),
        pytest.param({"fields.1.name": "coating_thickness\n"}, "name must match", id="field-name-trailing-newline"),
        pytest.param({"prompt.paper_key": "paper\n"}, "paper_key must match", id="paper-key-trailing-newline"),
        pytest.param({"title_zh": 5}, "title_zh", id="title-not-text"),
        # Groups
        pytest.param({"groups.1.level": "global"}, "level", id="group-level"),
        pytest.param({"groups.1.name": "Coating"}, "name must match", id="group-name"),
        pytest.param({"groups.1.colour": "red"}, "colour", id="group-unknown-key"),
        pytest.param({"groups.0.name": "coating"}, "more than one group", id="group-duplicate"),
        pytest.param({"groups.1.level": "paper"}, "level 'sample'", id="no-sample-group"),
        pytest.param({"groups": "coating"}, "groups must be a list", id="groups-not-a-list"),
        # Fields
        pytest.param({"fields.1.group": "film"}, "group must be one of", id="undeclared-group"),
        pytest.param({"fields.1.name": "samples"}, "reserved", id="reserved-name"),
        pytest.param({"fields.1.name": "Thickness"}, "name must match", id="field-name"),
        pytest.param({"fields.2.name": "coating_thickness"}, "more than one entry", id="duplicate-field"),
        pytest.param({"fields.1.level": "paper"}, "unknown key(s) level", id="level-is-derived"),
        pytest.param({"fields.1.kind": "number"}, "kind", id="existing-checks-still-apply"),
        pytest.param({"fields": [field(0)]}, "level 'sample'", id="no-sample-level-field"),
        pytest.param({"fields": [field(1, name=f"f{i}") for i in range(101)]}, "at most 100", id="too-many-fields"),
        pytest.param({"fields.1.condition_rule": "the substrate"}, "condition_hint", id="rule-without-hint"),
        pytest.param(
            {"fields.1.condition_hint": "the substrate", "fields.1.condition_rule": "the substrate"},
            "field 'coating_thickness': condition_rule needs missing_condition_note_zh",
            id="rule-without-note",
        ),
        pytest.param({"fields.1.figure_readable": "yes"}, "figure_readable", id="readable-not-a-boolean"),
        pytest.param({"fields.2.figure_readable": True}, "figure_readable", id="readable-text-field"),
        pytest.param({"fields.1.figure_readable": True}, "figures", id="readable-without-figure-slots"),
        pytest.param({"fields.2.display_format": "plain"}, "display_format", id="format-on-text"),
        pytest.param({"fields.1.display_format": "engineering"}, "display_format", id="format-unknown"),
        pytest.param({"fields.2.range_policy": "reject"}, "range_policy", id="range-policy-on-text"),
        pytest.param({"fields.1.range_policy": "first"}, "range_policy", id="range-policy-unknown"),
        pytest.param({"fields.1.missing_condition_note_zh": " "}, "missing_condition_note_zh", id="blank-note"),
        pytest.param({"fields.1.canonical_unit": "mAh/g"}, "no converter", id="unit-without-converter"),
        # Slots
        pytest.param({"prompt.domain_subject": DELETE}, "domain_subject", id="required-slot-missing"),
        pytest.param({"prompt.fact_nouns": "facts"}, "fact_nouns", id="unknown-slot"),
        pytest.param({"prompt.fact_noun": ""}, "fact_noun", id="empty-slot"),
        pytest.param({"prompt.fact_noun": 3}, "fact_noun", id="slot-not-text"),
        pytest.param({"prompt.field_scope": "x" * (MAX_SLOT_LENGTH + 1)}, "field_scope", id="slot-too-long"),
        pytest.param({"prompt.sample_definition": "A {sample_singular}."}, "marker", id="slot-with-marker"),
        pytest.param({"prompt.sample_definition": "Any {paper_groups}."}, "marker", id="slot-with-computed-marker"),
        pytest.param({"prompt.paper_key": "Paper Record"}, "paper_key", id="paper-key-not-an-identifier"),
        pytest.param({"prompt.no_samples_key": "samples"}, "no_samples_key", id="no-samples-key-is-samples"),
        pytest.param({"prompt.no_samples_key": "paper"}, "must differ", id="keys-collide"),
        pytest.param({"figures": {"subject": "coatings"}}, "missing required key(s)", id="figure-slots-incomplete"),
        pytest.param(
            {"figures": {"subject": "a", "property_noun": "b", "chart_definition": "c", "axis_example": "d"}},
            "figure_readable",
            id="figure-slots-without-a-readable-field",
        ),
        pytest.param({"ui": {"entity_label": "涂层"}}, "entity_label", id="unknown-ui-key"),
        # Retrieval
        pytest.param({"retrieval.condition_unit_pattern": "(unclosed"}, "condition_unit_pattern", id="bad-regex"),
        pytest.param({"retrieval.condition_unit_pattern": "a" * 501}, "condition_unit_pattern", id="long-regex"),
        pytest.param({"retrieval.condition_unit_pattern": r"\d(\s+)+c"}, "exponential", id="nested-repeat"),
        pytest.param({"retrieval.condition_unit_pattern": r"\d(?:a|a)*"}, "exponential", id="repeated-alternation"),
        pytest.param(
            {"units": {"mg/L": {"aliases": {"mg/L": 1}, "retrieval": r"(\d+\s*)*mg"}}}, "exponential", id="unit-redos"
        ),
        pytest.param({"retrieval.condition_keywords": "annealed"}, "condition_keywords", id="keywords-not-a-list"),
        pytest.param({"retrieval.keywords": []}, "keywords", id="unknown-retrieval-key"),
        # Units
        pytest.param({"units": {"mg/L": {"aliases": {"mg/l": 2}}}}, "units['mg/L']", id="unit-rule"),
    ],
)
def test_a_broken_profile_names_the_file_and_the_key(changes, expected):
    assert expected in refused(changes)


def test_a_profile_name_must_be_an_identifier():
    assert "name must match" in refused({"name": "Demo"}, Path("profiles/Demo.json"))


def test_a_readable_field_with_figure_slots_is_accepted():
    slots = {"subject": "coatings", "property_noun": "coating properties", "chart_definition": "a chart"}
    profile = make_profile({"fields.1.figure_readable": True, "figures": slots | {"axis_example": "axis x"}})

    assert [spec.name for spec in profile.figure_fields] == ["coating_thickness"]


def test_a_condition_rule_with_its_hint_is_accepted():
    profile = make_profile(
        {
            "fields.1.condition_hint": "the substrate",
            "fields.1.condition_rule": "the substrate it was coated on",
            "fields.1.missing_condition_note_zh": "未注明基底",
        }
    )

    assert profile.by_name["coating_thickness"].condition_rule == "the substrate it was coated on"


def test_zero_paper_groups_are_allowed():
    profile = make_profile({"groups": [profile_data()["groups"][1]], "fields": [field(1), field(2)]})

    assert profile.paper_groups == () and profile.paper_fields == ()


def test_many_fields_are_allowed_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="paperfacts.profile_loader"):
        profile = make_profile({"fields": [field(1, name=f"f{i}") for i in range(41)]})

    assert len(profile.fields) == 41
    assert "41 fields" in caplog.text


def test_a_profile_that_is_not_an_object_is_refused():
    with pytest.raises(ConfigError, match="JSON object"):
        parse_profile([], SOURCE)


@pytest.mark.parametrize(
    "pattern",
    [
        r"\d\s*(?:°C|℃|K\b|h\b|min\b|(?:wt|at|mol)\.?\s*%)",
        r"\d\s*c\b(?!\s*°)",
        r"\d\s*(?:ab)+",
        r"\d{1,3}(?:\.\d+)?\s*nm",
    ],
)
def test_a_pattern_that_nests_no_repetition_is_accepted(pattern):
    assert make_profile({"retrieval.condition_unit_pattern": pattern}).retrieval.condition_unit_pattern == pattern
