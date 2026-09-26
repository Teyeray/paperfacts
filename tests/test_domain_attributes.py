"""The attributes and slots that took the last TCO assumptions out of the code (S16).

Each one is TCO's old behaviour as the TCO profile's value, and something neutral for every other profile: the
rule-2 table-header examples and the plausible-range sentence of the extraction prompts, the chart prompt's axis
and tick examples, the gas names set aside after a unit, a value quoted with an "after ..." clause, and a
plausible range on a field that has no unit. TCO's own bytes are pinned by test_prompt_snapshot.py and
test_payload_pins.py; this file pins what the others get.
"""

from __future__ import annotations

import pytest

from paperfacts import prompts
from paperfacts.errors import ConfigError
from paperfacts.kinds import RULES
from paperfacts.normalize import drop_implausible, normalize_field, parse_number, split_after_clause
from paperfacts.profile import DomainProfile, PromptSlots
from paperfacts.profile_loader import load_profile
from paperfacts.records import NO_CONTEXT, ExtractedRecords, SampleRecord
from support.extraction import make_field
from support.profiles import SHIPPED_PROFILE_PATH, make_profile

TCO_HEADER = '(e.g. "ρ × 10^4 (Ω cm)" or "ρ (×10^-4 Ω·cm)")'
TCO_PLAIN = '(a column "Thickness (nm)" gives just "nm")'
TCO_ORIGIN = "almost always belongs to a different layer, process step or quantity"


@pytest.fixture(scope="module")
def battery() -> DomainProfile:
    return load_profile(SHIPPED_PROFILE_PATH.with_name("battery_cathode.json"))


def system_prompts(profile: DomainProfile) -> list[str]:
    return [prompts.extraction_system_prompt(profile), prompts.field_system_prompt(profile)]


# ---- Prompt slots ----------------------------------------------------------------------------------------


def test_tco_keeps_its_header_examples_and_range_sentence(tco_profile):
    for text in system_prompts(tco_profile):
        assert TCO_HEADER in text and TCO_PLAIN in text
    assert TCO_ORIGIN in prompts.extraction_system_prompt(tco_profile)


def test_a_profile_without_its_own_examples_gets_generic_ones():
    profile = make_profile({"fields.1.valid_range": {"max": 500}})

    for text in system_prompts(profile):
        assert f"(e.g. {PromptSlots.scaled_header_examples})" in text
        assert f"({PromptSlots.plain_header_example})" in text
        assert "ρ" not in text and "Thickness" not in text
    extraction = prompts.extraction_system_prompt(profile)
    assert f"almost always belongs to {PromptSlots.implausible_origin}, so check" in extraction
    assert "layer" not in extraction


def test_the_battery_profile_gets_its_own_examples(battery):
    for text in system_prompts(battery):
        assert '"D_Li+ × 10^11 (cm2 s-1)"' in text
        assert 'a column "Discharge capacity (mAh g-1)" gives just "mAh g-1"' in text
        assert "ρ" not in text
    spec = battery.by_name["capacity_retention"]
    question = prompts.field_user_prompt(spec, "- S1", "text", battery.prompt.implausible_origin)
    assert "belongs to a different electrode component, test condition or quantity" in question


# ---- after_clause ----------------------------------------------------------------------------------------


def test_split_after_clause_cuts_only_after_a_number():
    assert split_after_clause("92.5% after 100 cycles") == ("92.5%", "after 100 cycles")
    assert split_after_clause("92.5%") == ("92.5%", "")
    assert split_after_clause("retained after 100 cycles") == ("retained after 100 cycles", "")


def test_by_default_a_value_after_a_treatment_is_refused():
    value, note = parse_number("92.5% after 100 cycles")

    assert value is None and "another state" in note


def test_under_condition_the_number_is_read_and_the_clause_becomes_the_condition(battery):
    spec = battery.by_name["capacity_retention"]
    assert spec.after_clause == "condition"

    field = normalize_field(
        make_field("capacity_retention", "92.5% after 100 cycles", unit_raw="%"), spec, battery.units, NO_CONTEXT
    )

    assert (field.value, field.unit) == (92.5, "%")
    assert field.condition == "after 100 cycles"
    assert "'after 100 cycles' moved into the condition" in field.normalization_note


def test_the_clause_joins_a_condition_the_model_gave_once_only(battery):
    spec = battery.by_name["capacity_retention"]
    raw = make_field("capacity_retention", "92.5% after 100 cycles", unit_raw="%", condition="at 1 C")

    once = normalize_field(raw, spec, battery.units, NO_CONTEXT)
    twice = normalize_field(once, spec, battery.units, NO_CONTEXT)

    assert once.condition == "at 1 C; after 100 cycles"
    assert twice == once


def test_the_dataset_cell_reads_the_same_number(battery):
    spec = battery.by_name["capacity_retention"]

    value, note = RULES["numeric"].cell(
        make_field("capacity_retention", "92.5% after 100 cycles", unit_raw="%"), spec, battery.units, NO_CONTEXT
    )

    assert value == 92.5
    assert "after 100 cycles" in note


# ---- Ignored unit suffixes -------------------------------------------------------------------------------


def test_tco_sets_a_gas_name_aside_and_other_profiles_do_not(tco_profile):
    pressure = tco_profile.by_name["working_pressure"]
    field = make_field("working_pressure", "1.1", unit_raw="Pa Ar")
    assert normalize_field(field, pressure, tco_profile.units, NO_CONTEXT).value == pytest.approx(1.1)

    other = make_profile({"fields.1.canonical_unit": "Pa"})
    thickness = other.by_name["coating_thickness"]
    read = normalize_field(make_field("coating_thickness", "1.1", unit_raw="Pa Ar"), thickness, other.units, NO_CONTEXT)
    assert read.value is None and "unknown unit" in read.normalization_note


# ---- valid_range without a unit --------------------------------------------------------------------------


def test_a_unitless_field_takes_a_range_judged_on_the_number(battery):
    spec = battery.by_name["cycle_number"]
    assert spec.canonical_unit is None
    assert spec.describe_range() == "between 1 and 100000"
    records = ExtractedRecords(
        paper=None,
        samples=(
            SampleRecord(
                sample_id="S1",
                fields=(make_field("cycle_number", "100"), make_field("cycle_number", "2500000")),
            ),
        ),
        invalid_source_ids=(),
        dropped=(),
    )

    cleaned = drop_implausible(records, battery)

    assert [value.value_raw for value in cleaned.samples[0].fields] == ["100"]
    assert cleaned.dropped == (
        "cycle_number: '2500000' is 2.5e+06, outside the plausible range (between 1 and 100000)",
    )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        pytest.param({"fields.2.after_clause": "condition"}, "only meaningful for a numeric field", id="text-field"),
        pytest.param({"fields.1.after_clause": "keep"}, "after_clause must be one of", id="unknown-policy"),
    ],
)
def test_a_misplaced_after_clause_is_refused(changes, expected):
    with pytest.raises(ConfigError, match=expected):
        make_profile(changes)
