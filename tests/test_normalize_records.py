"""Fill in the canonical value for every FieldValue in a LaneExtraction from its raw text.

This layer is pure: the raw text (transcribed by the LLM) is read-only, and the canonical value (computed
by code) is written into separate fields. So there are only two core invariants — **always return a new
object**, **never change a single character of the raw text** — and violating either one breaks the
evidence trail.
"""

from __future__ import annotations

import pytest

from paperfacts.fields import FIELD_BY_NAME
from paperfacts.normalize import drop_implausible, normalize_field, normalize_lane
from paperfacts.records import ExtractedRecords, FieldValue, TargetRecord
from support.extraction import make_field, make_lane, make_sample

# ---- normalize_field ----------------------------------------------------------------


def test_a_numeric_field_gets_value_and_unit_filled_in():
    field = make_field("thickness", "1.2", unit_raw="μm")

    normalized = normalize_field(field, FIELD_BY_NAME["thickness"])

    assert normalized.value == 1200.0
    assert normalized.unit == "nm"
    assert normalized.normalization_note is None


def test_a_numeric_field_keeps_the_original_text_untouched():
    # The raw text is the provenance evidence; normalization may only add fields alongside it, never
    # change value_raw / unit_raw.
    field = make_field("sheet_resistance", "1.2 × 10⁻⁴", unit_raw="kΩ/sq")

    normalized = normalize_field(field, FIELD_BY_NAME["sheet_resistance"])

    assert normalized.value_raw == "1.2 × 10⁻⁴"
    assert normalized.unit_raw == "kΩ/sq"
    assert normalized.value == pytest.approx(0.12)


def test_the_parse_note_and_the_unit_note_are_joined():
    field = make_field("transmittance", "> 80")

    normalized = normalize_field(field, FIELD_BY_NAME["transmittance"])

    assert normalized.normalization_note == "qualifier '>' dropped; no unit; read as percent"


def test_an_unparseable_number_clears_value_and_unit_and_explains_why():
    field = make_field("thickness", "n.a.", unit_raw="nm")

    normalized = normalize_field(field, FIELD_BY_NAME["thickness"])

    assert (normalized.value, normalized.unit) == (None, None)
    assert normalized.normalization_note == "no number found"


def test_a_text_field_is_returned_untouched():
    # Text/composition comparison is computed on the fly in the comparison layer via normalize_key, rather
    # than caching a canonical text copy on the record (to avoid the two rule sets drifting apart).
    field = make_field("component", "SnO₂:Ta (2 wt% Ta₂O₅)")

    normalized = normalize_field(field, FIELD_BY_NAME["component"])

    assert normalized is field
    assert normalized.value is None and normalized.unit is None


# ---- normalize_lane -----------------------------------------------------------------


def test_normalize_lane_returns_a_new_object_and_leaves_the_input_alone():
    # The models are frozen, but a misplaced model_copy could still leak the normalized result back
    # through the original object reference.
    lane = make_lane(samples=[make_sample("A", [make_field("thickness", "1.2", unit_raw="μm")])])

    normalized = normalize_lane(lane)

    assert normalized is not lane
    assert lane.samples[0].fields[0].value is None
    assert normalized.samples[0].fields[0].value == 1200.0


def test_normalize_lane_covers_the_target_record_too():
    lane = make_lane(target=TargetRecord(fields=(make_field("density", "98.5", unit_raw="%"),)))

    normalized = normalize_lane(lane)

    assert normalized.target.fields[0].value == 98.5
    assert normalized.target.fields[0].unit == "%"


def test_normalize_lane_keeps_a_missing_target_as_none():
    assert normalize_lane(make_lane()).target is None


def test_normalize_lane_normalizes_every_field_of_every_sample():
    lane = make_lane(
        samples=[
            make_sample(
                "A", [make_field("thickness", "300", unit_raw="nm"), make_field("transmittance", "85", unit_raw="%")]
            ),
            make_sample("B", [make_field("sputtering_time", "2", unit_raw="h")]),
        ]
    )

    normalized = normalize_lane(lane)

    assert [f.value for f in normalized.samples[0].fields] == [300.0, 85.0]
    assert normalized.samples[1].fields[0].value == 120.0


def test_a_field_outside_the_schema_survives_untouched():
    # The extraction layer already filtered these out once; this is a defensive second line — better to
    # leave it untouched than let a KeyError blow up the whole document.
    unknown = FieldValue(field="carrier_concentration", value_raw="1e20")
    lane = make_lane(samples=[make_sample("A", [unknown])])

    normalized = normalize_lane(lane)

    assert normalized.samples[0].fields[0] == unknown


def test_normalize_lane_preserves_everything_that_is_not_a_field_value():
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

    normalized = normalize_lane(lane)

    assert normalized.backend == "paddleocr_vl"
    assert normalized.usage == {"total_tokens": 42}
    assert normalized.raw_response == '{"samples": []}'
    assert normalized.samples[0].label == "O2 100 sccm"
    assert normalized.samples[0].conditions == {"O2": "100"}


def test_normalizing_twice_changes_nothing_further():
    # Normalization is redone every time a lane is read (so a rule change doesn't require re-calling the
    # LLM), which means it must be idempotent.
    lane = normalize_lane(make_lane(samples=[make_sample("A", [make_field("thickness", "1.2", unit_raw="μm")])]))

    assert normalize_lane(lane) == lane


# ---- drop_implausible --------------------------------------------------------------------


def _records(*, samples=(), unattributed=()) -> ExtractedRecords:
    return ExtractedRecords(
        target=None, samples=tuple(samples), invalid_source_ids=(), dropped=(), unattributed=tuple(unattributed)
    )


def test_a_value_outside_its_range_is_dropped_with_the_reason():
    # Shipped range: thickness below 500 nm. A perovskite absorber's 600 nm is the observed confusion.
    records = _records(samples=(make_sample("S1", [make_field("thickness", "600", unit_raw="nm")]),))

    kept = drop_implausible(records)

    assert kept.samples[0].fields == ()
    assert len(kept.dropped) == 1
    assert "thickness" in kept.dropped[0] and "600" in kept.dropped[0] and "below 500 nm" in kept.dropped[0]


def test_the_range_is_judged_after_conversion_to_the_canonical_unit():
    # "0.6" has digits well under 500, but in μm it is 600 nm.
    inside = make_field("thickness", "0.3", unit_raw="μm")
    outside = make_field("thickness", "0.6", unit_raw="μm")
    records = _records(samples=(make_sample("S1", [inside, outside]),))

    assert drop_implausible(records).samples[0].fields == (inside,)


def test_unattributed_values_are_held_to_the_same_range():
    records = _records(unattributed=(make_field("rotation_speed", "3000", unit_raw="rpm"),))

    kept = drop_implausible(records)

    assert kept.unattributed == ()
    assert "rotation_speed" in kept.dropped[0]


def test_a_value_that_cannot_be_converted_is_kept_since_there_is_nothing_to_judge():
    unconvertible = make_field("thickness", "600", unit_raw="furlongs")
    records = _records(samples=(make_sample("S1", [unconvertible]),))

    assert drop_implausible(records) == records


def test_fields_without_a_range_and_values_inside_one_leave_the_records_as_they_were():
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

    assert drop_implausible(records) is records
