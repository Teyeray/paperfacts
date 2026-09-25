"""Fill in the canonical value for every FieldValue in a LaneExtraction from its raw text.

This layer is pure: the raw text (transcribed by the LLM) is read-only, and the canonical value (computed
by code) is written into separate fields. So there are only two core invariants — **always return a new
object**, **never change a single character of the raw text** — and violating either one breaks the
evidence trail.
"""

from __future__ import annotations

import pytest

from paperfacts.normalize import drop_implausible, normalize_field, normalize_lane
from paperfacts.records import ExtractedRecords, FieldValue, TargetRecord
from paperfacts.units import BUILTIN_UNITS
from support.extraction import make_field, make_lane, make_sample
from support.profiles import shipped_profile

# The shipped profile's field table, at module level because constants and parametrize lists need it before
# any fixture runs.
FIELD_BY_NAME = shipped_profile().by_name

# ---- normalize_field ----------------------------------------------------------------


def test_a_numeric_field_gets_value_and_unit_filled_in():
    field = make_field("thickness", "1.2", unit_raw="μm")

    normalized = normalize_field(field, FIELD_BY_NAME["thickness"], BUILTIN_UNITS)

    assert normalized.value == 1200.0
    assert normalized.unit == "nm"
    assert normalized.normalization_note is None


def test_a_numeric_field_keeps_the_original_text_untouched():
    # The raw text is the provenance evidence; normalization may only add fields alongside it, never
    # change value_raw / unit_raw.
    field = make_field("sheet_resistance", "1.2 × 10⁻⁴", unit_raw="kΩ/sq")

    normalized = normalize_field(field, FIELD_BY_NAME["sheet_resistance"], BUILTIN_UNITS)

    assert normalized.value_raw == "1.2 × 10⁻⁴"
    assert normalized.unit_raw == "kΩ/sq"
    assert normalized.value == pytest.approx(0.12)


def test_the_parse_note_and_the_unit_note_are_joined():
    field = make_field("transmittance", "> 80")

    normalized = normalize_field(field, FIELD_BY_NAME["transmittance"], BUILTIN_UNITS)

    assert normalized.normalization_note == "qualifier '>' dropped; no unit; read as percent"


def test_an_unparseable_number_clears_value_and_unit_and_explains_why():
    field = make_field("thickness", "n.a.", unit_raw="nm")

    normalized = normalize_field(field, FIELD_BY_NAME["thickness"], BUILTIN_UNITS)

    assert (normalized.value, normalized.unit) == (None, None)
    assert normalized.normalization_note == "no number found"


@pytest.mark.parametrize(
    ("raw", "unit_raw", "expected"),
    [
        ("3 h 30 min", "h", 210.0),
        ("3 h 30 min", None, 210.0),
        ("2 hours and 15 minutes", None, 135.0),
        ("1 min 30 s", "min", 1.5),
        # A qualifier or a condition around it is set aside first, as for any number.
        ("~3 h 30 min", "h", 210.0),
        ("3 h 30 min at 400 °C", "h", 210.0),
    ],
)
def test_a_compound_duration_is_one_value(raw, unit_raw, expected):
    # Read as its first number, "3 h 30 min" became 180 min: a common annealing-time spelling, silently wrong.
    normalized = normalize_field(
        make_field("annealing_time", raw, unit_raw=unit_raw), FIELD_BY_NAME["annealing_time"], BUILTIN_UNITS
    )

    assert normalized.value == pytest.approx(expected)
    assert normalized.unit == "min"
    assert "compound" in normalized.normalization_note


@pytest.mark.parametrize(
    "raw",
    [
        "30 min 3 h",
        "30 min 20 min",
        "3 h 30 nm",
        "1 h 90 min",
        "2 h 60 min",
        # Nobody writes a sum with a fractional larger part: "0.5 h 30 min" restates 30 min.
        "0.5 h 30 min",
        "0.5 min 30 s",
        "1.5 h 30 min",
    ],
)
def test_only_a_descending_pair_of_one_quantity_is_a_compound(raw):
    normalized = normalize_field(
        make_field("annealing_time", raw, unit_raw="min"), FIELD_BY_NAME["annealing_time"], BUILTIN_UNITS
    )

    assert normalized.value is None


@pytest.mark.parametrize(
    ("field", "raw", "unit_raw"),
    [
        ("working_pressure", "0.5 Pa 3.75 mTorr", "Pa"),
        ("working_pressure", "1 Torr 133 Pa", "Torr"),
        ("target_substrate_distance", "10 cm 100 mm", "cm"),
        ("inch", "2 in 50 mm", "in"),
        ("thickness", "1 um 1000 nm", "um"),
    ],
)
def test_a_value_restated_in_a_second_unit_is_never_added_up(field, raw, unit_raw):
    # Only a duration is written as a sum of units; anywhere else a second unit restates the same value, and
    # adding the two doubled it (0.5 Pa 3.75 mTorr read as 1.0 Pa).
    assert normalize_field(make_field(field, raw, unit_raw=unit_raw), FIELD_BY_NAME[field], BUILTIN_UNITS).value is None


@pytest.mark.parametrize(
    ("field", "raw", "unit_raw", "expected"),
    [
        ("annealing_temperature", "400 °C for 2 h", "°C", 400.0),
        ("sputtering_time", "deposited for 10 min", "min", 10.0),
        ("annealing_time", "annealed for 2 h", "h", 120.0),
        ("annealing_time", "held for 30 min", "min", 30.0),
        # The tail holds the field's own quantity and the value does not: the time is in the tail.
        ("annealing_time", "400 °C for 2 h", "h", None),
        ("sheet_resistance", "increase of 5 % after 1000 cycles", "Ω/sq", None),
    ],
)
def test_a_condition_tail_is_set_aside_only_when_it_is_not_the_value(field, raw, unit_raw, expected):
    # "400 °C for 2 h" on annealing_time read 400 h (24000 min) once "for" opened a condition.
    value = normalize_field(make_field(field, raw, unit_raw=unit_raw), FIELD_BY_NAME[field], BUILTIN_UNITS).value

    assert value == (pytest.approx(expected) if expected is not None else None)


def test_a_text_field_is_returned_untouched():
    # Text/composition comparison is computed on the fly in the comparison layer via normalize_key, rather
    # than caching a canonical text copy on the record (to avoid the two rule sets drifting apart).
    field = make_field("component", "SnO₂:Ta (2 wt% Ta₂O₅)")

    normalized = normalize_field(field, FIELD_BY_NAME["component"], BUILTIN_UNITS)

    assert normalized is field
    assert normalized.value is None and normalized.unit is None


# ---- normalize_lane -----------------------------------------------------------------


def test_normalize_lane_returns_a_new_object_and_leaves_the_input_alone(tco_profile):
    # The models are frozen, but a misplaced model_copy could still leak the normalized result back
    # through the original object reference.
    lane = make_lane(samples=[make_sample("A", [make_field("thickness", "1.2", unit_raw="μm")])])

    normalized = normalize_lane(lane, tco_profile)

    assert normalized is not lane
    assert lane.samples[0].fields[0].value is None
    assert normalized.samples[0].fields[0].value == 1200.0


def test_normalize_lane_covers_the_target_record_too(tco_profile):
    lane = make_lane(target=TargetRecord(fields=(make_field("density", "98.5", unit_raw="%"),)))

    normalized = normalize_lane(lane, tco_profile)

    assert normalized.target.fields[0].value == 98.5
    assert normalized.target.fields[0].unit == "%"


def test_normalize_lane_keeps_a_missing_target_as_none(tco_profile):
    assert normalize_lane(make_lane(), tco_profile).target is None


def test_normalize_lane_normalizes_every_field_of_every_sample(tco_profile):
    lane = make_lane(
        samples=[
            make_sample(
                "A", [make_field("thickness", "300", unit_raw="nm"), make_field("transmittance", "85", unit_raw="%")]
            ),
            make_sample("B", [make_field("sputtering_time", "2", unit_raw="h")]),
        ]
    )

    normalized = normalize_lane(lane, tco_profile)

    assert [f.value for f in normalized.samples[0].fields] == [300.0, 85.0]
    assert normalized.samples[1].fields[0].value == 120.0


def test_a_field_outside_the_schema_survives_untouched(tco_profile):
    # The extraction layer already filtered these out once; this is a defensive second line — better to
    # leave it untouched than let a KeyError blow up the whole document.
    unknown = FieldValue(field="carrier_concentration", value_raw="1e20")
    lane = make_lane(samples=[make_sample("A", [unknown])])

    normalized = normalize_lane(lane, tco_profile)

    assert normalized.samples[0].fields[0] == unknown


def test_normalize_lane_preserves_everything_that_is_not_a_field_value(tco_profile):
    lane = make_lane(
        backend="paddleocr_vl",
        samples=[
            make_sample(
                "A", [make_field("thickness", "300", unit_raw="nm")], label="O2 100 sccm", conditions={"O2": "100"}
            )
        ],
        usage={"total_tokens": 42},
        raw_response='{"samples": []}',
    )

    normalized = normalize_lane(lane, tco_profile)

    assert normalized.backend == "paddleocr_vl"
    assert normalized.usage == {"total_tokens": 42}
    assert normalized.raw_response == '{"samples": []}'
    assert normalized.samples[0].label == "O2 100 sccm"
    assert normalized.samples[0].conditions == {"O2": "100"}


def test_normalizing_twice_changes_nothing_further(tco_profile):
    # Normalization is redone every time a lane is read (so a rule change doesn't require re-calling the
    # LLM), which means it must be idempotent.
    lane = normalize_lane(
        make_lane(samples=[make_sample("A", [make_field("thickness", "1.2", unit_raw="μm")])]), tco_profile
    )

    assert normalize_lane(lane, tco_profile) == lane


# ---- drop_implausible --------------------------------------------------------------------


def _records(*, samples=(), unattributed=()) -> ExtractedRecords:
    return ExtractedRecords(
        target=None, samples=tuple(samples), invalid_source_ids=(), dropped=(), unattributed=tuple(unattributed)
    )


def test_a_value_outside_its_range_is_dropped_with_the_reason(tco_profile):
    # Shipped range: thickness at most 5000 nm. A 280 µm wafer read as the film is the observed confusion.
    records = _records(samples=(make_sample("S1", [make_field("thickness", "6000", unit_raw="nm")]),))

    kept = drop_implausible(records, tco_profile)

    assert kept.samples[0].fields == ()
    assert len(kept.dropped) == 1
    assert "thickness" in kept.dropped[0] and "6000" in kept.dropped[0] and "at most 5000 nm" in kept.dropped[0]


def test_the_range_is_judged_after_conversion_to_the_canonical_unit(tco_profile):
    # "0.006" has digits well under 5000, but in mm it is 6000 nm.
    inside = make_field("thickness", "3", unit_raw="μm")
    outside = make_field("thickness", "0.006", unit_raw="mm")
    records = _records(samples=(make_sample("S1", [inside, outside]),))

    assert drop_implausible(records, tco_profile).samples[0].fields == (inside,)


def test_unattributed_values_are_held_to_the_same_range(tco_profile):
    records = _records(unattributed=(make_field("rotation_speed", "3000", unit_raw="rpm"),))

    kept = drop_implausible(records, tco_profile)

    assert kept.unattributed == ()
    assert "rotation_speed" in kept.dropped[0]


def test_a_value_that_cannot_be_converted_is_kept_since_there_is_nothing_to_judge(tco_profile):
    unconvertible = make_field("thickness", "600", unit_raw="furlongs")
    records = _records(samples=(make_sample("S1", [unconvertible]),))

    assert drop_implausible(records, tco_profile) == records


def test_fields_without_a_range_and_values_inside_one_leave_the_records_as_they_were(tco_profile):
    records = _records(
        samples=(
            make_sample(
                "S1",
                [
                    make_field("sheet_resistance", "1e6", unit_raw="Ω/sq"),
                    make_field("transmittance", "85", unit_raw="%"),
                ],
            ),
        )
    )

    assert drop_implausible(records, tco_profile) is records


def test_a_target_whose_every_field_is_dropped_keeps_its_citations(tco_profile):
    spec = FIELD_BY_NAME["thickness"]  # any ranged field will do; the target is judged like a sample
    records = ExtractedRecords(
        target=TargetRecord(source_ids=("mineru_p0_b1",), fields=(make_field(spec.name, "9", unit_raw="μm"),)),
        samples=(),
        invalid_source_ids=(),
        dropped=(),
    )

    kept = drop_implausible(records, tco_profile)

    assert kept.target == TargetRecord(source_ids=("mineru_p0_b1",), fields=())
