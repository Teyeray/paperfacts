"""The units a profile declares: every mistake in a declaration is refused with the unit and the key named."""

from __future__ import annotations

from typing import Any

import pytest

from paperfacts.errors import ConfigError
from paperfacts.fields import FIELD_BY_NAME, FieldSpec
from paperfacts.models import SourceBlock
from paperfacts.normalize import convert_to_canonical, normalize_field
from paperfacts.passages import DEFAULT_RETRIEVAL, candidate_blocks, inventory_blocks, searchable
from paperfacts.profile import RetrievalSpec
from paperfacts.units import (
    BUILTIN_CONVERTERS,
    BUILTIN_RETRIEVAL,
    BUILTIN_UNITS,
    MAX_ALIASES,
    DeclaredUnit,
    UnitRegistry,
    load_units,
)
from support.extraction import make_field
from support.factories import make_block

WHERE = "profiles/battery.json"
CAPACITY: dict[str, Any] = {"aliases": {"mAh/g": 1, "mAh g-1": 1, "Ah/kg": 1, "Ah/g": 1000}, "case_sensitive": True}
# A capacity table as a battery profile declares it, with the superscript and middle-dot spellings papers print.
# "mA h g-1" is not listed: cleaned, it is "mAh g-1" again, and the loader refuses the pair as one spelling.
BATTERY_CAPACITY: dict[str, Any] = {
    "aliases": {"mAh/g": 1, "mAh g-1": 1, "mAh g^-1": 1, "mAh·g^-1": 1, "Ah/kg": 1, "Ah/g": 1000},
    "case_sensitive": True,
}
KELVIN = {"℃": {"extends_builtin": True, "aliases": {"K": {"factor": 1, "offset": -273.15}}}}


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


# ---- Conversion through a registry -----------------------------------------------------------


def probe(canonical: str, **attributes: Any) -> FieldSpec:
    return FieldSpec(
        name="probe",
        group="film",
        kind="numeric",
        description="probe",
        canonical_unit=canonical,
        **({"keywords": ()} | attributes),
    )


def block(content: str) -> SourceBlock:
    return make_block(type="text", content=content)


def test_a_declared_factor_converts_a_quoted_value():
    registry = load_units({"mAh/g": BATTERY_CAPACITY}, WHERE)

    value, unit, note = convert_to_canonical(probe("mAh/g"), 0.15, "Ah/g", registry)

    assert (value, unit, note) == (pytest.approx(150.0), "mAh/g", None)


@pytest.mark.parametrize("spelling", ["mAh g^-1", "mAh g⁻¹", "mAh·g⁻¹", "mAh·g^-1", "mA h g-1", "mAhg-1", " mAh/g "])
def test_printed_spellings_of_a_declared_unit_resolve(spelling):
    registry = load_units({"mAh/g": BATTERY_CAPACITY}, WHERE)

    assert convert_to_canonical(probe("mAh/g"), 2.0, spelling, registry) == (2.0, "mAh/g", None)


def test_a_case_sensitive_unit_does_not_read_another_case():
    registry = load_units({"mAh/g": BATTERY_CAPACITY}, WHERE)

    assert convert_to_canonical(probe("mAh/g"), 2.0, "MAh/g", registry) == (
        None,
        None,
        "unknown unit 'MAh/g' for mAh/g",
    )


def test_a_case_insensitive_unit_reads_any_case():
    registry = load_units({"V": {"aliases": {"V": 1, "mV": 0.001}}}, WHERE)

    assert registry.convert("V", "MV") == (0.001, 0.0)
    assert registry.convert("V", "v") == (1.0, 0.0)


def test_the_built_in_registry_converts_as_the_built_in_converters_do():
    # The registry is asked with a cleaned unit, in which "·" is already ".".
    assert BUILTIN_UNITS.convert("Ω·cm", "mΩ.cm") == (1e-3, 0.0)
    assert BUILTIN_UNITS.convert("nm", "furlong") is None
    assert BUILTIN_UNITS.convert("mAh/g", "mAh/g") is None


def test_kelvin_reaches_celsius_through_the_declared_offset():
    registry = load_units(KELVIN, WHERE)

    value, unit, note = convert_to_canonical(FIELD_BY_NAME["annealing_temperature"], 573.0, "K", registry)

    assert (value, unit, note) == (pytest.approx(299.85), "℃", None)


def test_kelvin_stays_ambiguous_without_the_extension():
    spec = FIELD_BY_NAME["annealing_temperature"]

    assert convert_to_canonical(spec, 573.0, "K") == (None, None, "unknown unit 'K' for ℃")
    assert convert_to_canonical(spec, 573.0, "K", BUILTIN_UNITS) == (None, None, "unknown unit 'K' for ℃")


def test_an_extension_never_changes_a_spelling_the_built_in_reads():
    registry = load_units(KELVIN, WHERE)

    assert registry.convert("℃", "°C") == (1.0, 0.0)
    assert convert_to_canonical(FIELD_BY_NAME["annealing_temperature"], 300.0, "℃", registry) == (300.0, "℃", None)


def test_a_scale_factor_is_applied_before_the_offset():
    registry = load_units(KELVIN, WHERE)

    value, unit, note = convert_to_canonical(probe("℃"), 5.73, "×10^2 K", registry)

    # 5.73 × 100 is 573 K, then the offset; the other order would give (5.73 − 273.15) × 100.
    assert (value, unit) == (pytest.approx(299.85), "℃")
    assert note == "scale factor 100 taken from the header in the unit"


def test_an_offset_is_never_applied_to_a_bare_number():
    registry = load_units(KELVIN, WHERE)

    assert convert_to_canonical(probe("℃", bare_number="assume_canonical"), 300.0, None, registry) == (
        300.0,
        "℃",
        "no unit; assumed ℃",
    )


def test_a_gas_name_after_a_declared_unit_is_set_aside():
    registry = load_units({"mAh/g": BATTERY_CAPACITY}, WHERE)

    value, unit, note = convert_to_canonical(probe("mAh/g"), 0.15, "Ah/g (Ar)", registry)

    assert (value, unit) == (pytest.approx(150.0), "mAh/g")
    assert note == "gas name in the unit (Ar) set aside"


def test_normalize_field_converts_with_the_registry_it_is_given():
    registry = load_units(KELVIN, WHERE)
    field = make_field("annealing_temperature", "573", unit_raw="K")

    assert normalize_field(field, FIELD_BY_NAME["annealing_temperature"], registry).value == pytest.approx(299.85)
    assert normalize_field(field, FIELD_BY_NAME["annealing_temperature"], BUILTIN_UNITS).value is None


@pytest.mark.parametrize(("spelling", "factor"), list(BATTERY_CAPACITY["aliases"].items()))
def test_every_alias_converts_by_its_factor_and_is_found_after_a_number(spelling, factor):
    registry = load_units({"mAh/g": BATTERY_CAPACITY}, WHERE)
    pattern = registry.retrieval("mAh/g")

    assert convert_to_canonical(probe("mAh/g"), 2.0, spelling, registry)[0] == pytest.approx(2.0 * factor)
    assert pattern is not None
    assert pattern.search(searchable(block(f"a capacity of 12 {spelling} after 50 cycles")))
    assert not pattern.search(searchable(block(f"the capacity in {spelling} is shown")))


# ---- Retrieval through a registry --------------------------------------------------------------


def test_the_built_in_registry_keeps_the_built_in_pattern_objects():
    assert all(BUILTIN_UNITS.retrieval(unit) is pattern for unit, pattern in BUILTIN_RETRIEVAL.items())


def test_a_derived_pattern_needs_a_digit_and_a_word_boundary():
    registry = load_units({"V": {"aliases": {"V": 1, "mV": 0.001}}}, WHERE)
    pattern = registry.retrieval("V")

    assert pattern is not None
    assert pattern.search(searchable(block("charged to 4.3 V")))
    assert pattern.search(searchable(block("a 150 mV plateau")))
    assert not pattern.search(searchable(block("the V content")))
    assert not pattern.search(searchable(block("in 4 vials")))


def test_a_derived_pattern_makes_spaces_optional():
    registry = load_units({"mAh/g": BATTERY_CAPACITY}, WHERE)
    pattern = registry.retrieval("mAh/g")

    assert pattern is not None
    assert pattern.search(searchable(block("152 mAhg-1")))
    assert pattern.search(searchable(block("152 mAh  g-1")))


def test_an_authors_pattern_is_used_as_written():
    registry = load_units({"C": {"aliases": {"C": 1}, "case_sensitive": True, "retrieval": r"\d\s*c\b(?!\s*°)"}}, WHERE)
    pattern = registry.retrieval("C")

    assert pattern is not None and pattern.pattern == r"\d\s*c\b(?!\s*°)"


def test_an_extension_is_searched_beside_the_built_in_pattern():
    registry = load_units({"nm": {"extends_builtin": True, "aliases": {"angstroms": 0.1}}}, WHERE)
    pattern = registry.retrieval("nm")

    assert pattern is not None
    assert pattern.search(searchable(block("films of 120 nm")))
    assert pattern.search(searchable(block("films of 1200 angstroms")))
    assert registry.convert("nm", "angstroms") == (0.1, 0.0)


def test_a_unit_with_no_pattern_is_refused():
    registry = UnitRegistry((DeclaredUnit(canonical="Wh/kg", aliases=(("wh/kg", 1.0, 0.0),)),))

    assert registry.knows("Wh/kg") and not registry.has_retrieval("Wh/kg")
    with pytest.raises(ConfigError, match="has no retrieval pattern"):
        registry.check("Wh/kg", f"{WHERE}: field 'energy'")


def test_candidate_blocks_find_a_declared_unit_through_the_registry():
    registry = load_units({"mAh/g": BATTERY_CAPACITY}, WHERE)
    spec = probe("mAh/g", keywords=("specific capacity",))
    unit_only = block("the cell delivered 152 mAh g-1 at 0.1 C")

    assert candidate_blocks(spec, [unit_only]) == []
    assert candidate_blocks(spec, [unit_only], units=registry) == [unit_only]


def test_inventory_blocks_use_the_condition_pattern_they_are_given():
    cycled = block("cells were cycled at 0.5 C between two limits")
    retrieval = RetrievalSpec(condition_keywords=(), condition_unit_pattern=r"\d\s*C\b")

    assert inventory_blocks([cycled]) == []
    # Compiled case-insensitively: the pattern is matched on lower-cased searchable() text.
    assert inventory_blocks([cycled], retrieval) == [cycled]


def test_the_default_retrieval_is_the_shipped_profiles(tco_profile):
    assert tco_profile.retrieval == DEFAULT_RETRIEVAL
