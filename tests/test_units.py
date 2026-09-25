"""The units a profile declares: every mistake in a declaration is refused with the unit and the key named."""

from __future__ import annotations

from typing import Any

import pytest

from paperfacts.errors import ConfigError
from paperfacts.units import BUILTIN_CONVERTERS, BUILTIN_RETRIEVAL, MAX_ALIASES, load_units

WHERE = "profiles/battery.json"
CAPACITY: dict[str, Any] = {"aliases": {"mAh/g": 1, "mAh g-1": 1, "Ah/kg": 1, "Ah/g": 1000}, "case_sensitive": True}


def refused(declared: dict[str, Any]) -> str:
    with pytest.raises(ConfigError) as excinfo:
        load_units(declared, WHERE)
    message = str(excinfo.value)
    assert WHERE in message
    return message


def test_no_declared_unit_leaves_the_built_ins():
    registry = load_units({}, WHERE)

    assert registry.known() == tuple(BUILTIN_CONVERTERS)
    assert registry.material() == []
    assert all(registry.knows(unit) and registry.has_retrieval(unit) for unit in BUILTIN_CONVERTERS)


def test_every_built_in_unit_has_both_a_converter_and_a_retrieval_pattern():
    assert BUILTIN_CONVERTERS.keys() == BUILTIN_RETRIEVAL.keys()


def test_a_declared_unit_is_known_with_its_spellings_cleaned():
    registry = load_units({"mAh/g": CAPACITY}, WHERE)

    assert registry.knows("mAh/g") and registry.has_retrieval("mAh/g")
    (unit,) = registry.declared
    # Cleaned as a quoted unit is: whitespace removed, case kept for a case-sensitive unit.
    assert unit.aliases == (("mAh/g", 1.0, 0.0), ("mAhg-1", 1.0, 0.0), ("Ah/kg", 1.0, 0.0), ("Ah/g", 1000.0, 0.0))


def test_a_case_insensitive_unit_folds_its_spellings():
    (unit,) = load_units({"V": {"aliases": {"V": 1, "mV": 0.001}}}, WHERE).declared

    assert [spelling for spelling, _, _ in unit.aliases] == ["v", "mv"]


def test_a_temperature_extension_may_carry_an_offset():
    registry = load_units({"℃": {"extends_builtin": True, "aliases": {"K": {"factor": 1, "offset": -273.15}}}}, WHERE)

    assert registry.declared[0].aliases == (("k", 1.0, -273.15),)
    assert registry.known() == tuple(BUILTIN_CONVERTERS)


def test_a_declared_unit_with_its_own_pattern_is_accepted():
    registry = load_units({"C": {"aliases": {"C": 1}, "case_sensitive": True, "retrieval": r"\d\s*c\b"}}, WHERE)

    assert registry.declared[0].retrieval == r"\d\s*c\b"


def test_an_unknown_canonical_unit_is_refused_with_the_known_ones():
    with pytest.raises(ConfigError, match=r"has no converter; known units are .*mAh/g") as excinfo:
        load_units({"mAh/g": CAPACITY}, WHERE).check("Wh/kg", f"{WHERE}: field 'energy'")

    assert "field 'energy'" in str(excinfo.value)


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        pytest.param({"nm": {"aliases": {"nm": 1}}}, "built-in", id="redefines-a-built-in"),
        pytest.param(
            {"mAh/g": {"extends_builtin": True, "aliases": {"mAh/g": 1}}}, "extends_builtin", id="extends-nothing"
        ),
        pytest.param({"mAh/g": {"aliases": {"Ah/g": 1000}}}, "itself", id="own-spelling-missing"),
        pytest.param({"mAh/g": {"aliases": {"mAh/g": 2}}}, "itself", id="own-spelling-not-factor-one"),
        pytest.param({"mAh/g": {"aliases": {}}}, "aliases", id="no-aliases"),
        pytest.param(
            {"mAh/g": {"aliases": {f"u{i}": 1 for i in range(MAX_ALIASES + 1)}}}, "aliases", id="too-many-aliases"
        ),
        pytest.param({"mAh/g": {"aliases": ["mAh/g"]}}, "aliases", id="aliases-not-an-object"),
        pytest.param({"mAh/g": {"aliases": {"mAh/g": 1, "Ah/g": 0}}}, "greater than 0", id="zero-factor"),
        pytest.param({"mAh/g": {"aliases": {"mAh/g": 1, "Ah/g": -1}}}, "greater than 0", id="negative-factor"),
        pytest.param({"mAh/g": {"aliases": {"mAh/g": 1, "Ah/g": float("inf")}}}, "finite", id="infinite-factor"),
        pytest.param({"mAh/g": {"aliases": {"mAh/g": 1, "Ah/g": True}}}, "finite", id="boolean-factor"),
        pytest.param({"mAh/g": {"aliases": {"mAh/g": 1, "Ah/g": "1000"}}}, "finite", id="factor-as-text"),
        pytest.param(
            {"mAh/g": {"aliases": {"mAh/g": 1, "Ah/g": {"factor": 1000, "offset": 1}}}},
            "offset",
            id="offset-on-capacity",
        ),
        pytest.param({"K": {"aliases": {"K": {"factor": 1, "offset": float("nan")}}}}, "finite", id="nan-offset"),
        pytest.param({"mAh/g": {"aliases": {"mAh/g": {"scale": 1}}}}, "factor", id="unknown-alias-key"),
        pytest.param({"mS": {"aliases": {"mS": 1, "MS": 1e6}}}, "same spelling", id="case-collision"),
        pytest.param({"mAh/g": {"aliases": {"mAh/g": 1, "mAh / g": 1}}}, "same spelling", id="whitespace-collision"),
        pytest.param({"mAh/g": {"aliases": {"mAh/g": 1, " ": 1}}}, "empty spelling", id="blank-spelling"),
        pytest.param({"mAh/g": CAPACITY | {"units": "x"}}, "unknown key", id="unknown-unit-key"),
        pytest.param({"mAh/g": CAPACITY | {"case_sensitive": "yes"}}, "case_sensitive", id="flag-not-a-boolean"),
        pytest.param({"mAh/g": CAPACITY | {"retrieval": "(unclosed"}}, "retrieval", id="bad-regex"),
        pytest.param({"mAh/g": CAPACITY | {"retrieval": "a" * 501}}, "retrieval", id="long-regex"),
        pytest.param({"mAh/g": "mAh/g"}, "must be an object", id="unit-not-an-object"),
        pytest.param({" ": CAPACITY}, "non-empty name", id="blank-unit-name"),
    ],
)
def test_a_broken_declaration_names_the_unit_and_the_problem(declared, expected):
    message = refused(declared)

    assert expected in message
    assert f"units[{next(iter(declared))!r}]" in message


def test_case_sensitive_spellings_may_differ_only_in_case():
    registry = load_units({"mS": {"aliases": {"mS": 1, "MS": 1e9}, "case_sensitive": True}}, WHERE)

    assert [spelling for spelling, _, _ in registry.declared[0].aliases] == ["mS", "MS"]


def test_units_that_are_not_an_object_are_refused():
    with pytest.raises(ConfigError, match="units must be an object"):
        load_units(["mAh/g"], WHERE)
