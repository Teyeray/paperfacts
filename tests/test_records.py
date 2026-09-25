"""Data models for extraction results: validating the LLM response and converting it into stored records.

The two layers are deliberately kept apart -- :class:`ExtractionResponse` only has the verbatim fields, so
the LLM has no place to do unit conversion even structurally. What this file guards is the boundary of
"response shape": a field missing its verbatim value must fail on the spot, not carry a null value
downstream.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from paperfacts.fields import FIELD_BY_NAME
from paperfacts.records import ExtractionResponse, ResponseCleaning, response_to_records

# Cleaning needs to know both "which source_ids actually exist" and "which fields are in the schema";
# tests are given one fixed known set.
KNOWN_IDS = frozenset({"b1", "b2", "b8", "b9", "mineru_p1_b9"})


def to_records(response: ExtractionResponse, *, known_ids: frozenset[str] = KNOWN_IDS):
    return response_to_records(response, known_ids=known_ids)


# ---- ExtractionResponse validation -----------------------------------------------------


def make_response(**overrides) -> dict:
    field = {"field": "thickness", "value_raw": "300", "unit_raw": "nm", "source_ids": ["mineru_p1_b9"]}
    field.update(overrides)
    return {"samples": [{"sample_id": "A", "fields": [field]}]}


def test_a_well_formed_response_validates():
    response = ExtractionResponse.model_validate(make_response())

    assert response.samples[0].fields[0].value_raw == "300"


def test_a_field_without_value_raw_is_rejected():
    # value_raw is the sole verbatim evidence across the whole chain; without it this fact cannot be
    # traced back to anything, so it must fail immediately.
    payload = make_response()
    del payload["samples"][0]["fields"][0]["value_raw"]

    with pytest.raises(ValidationError, match="value_raw"):
        ExtractionResponse.model_validate(payload)


@pytest.mark.parametrize("empty", ["", None])
def test_a_field_with_an_empty_value_raw_is_rejected(empty):
    with pytest.raises(ValidationError, match="value_raw"):
        ExtractionResponse.model_validate(make_response(value_raw=empty))


def test_a_sample_without_an_id_is_rejected():
    with pytest.raises(ValidationError, match="sample_id"):
        ExtractionResponse.model_validate({"samples": [{"sample_id": "", "fields": []}]})


def test_unknown_keys_in_the_response_are_ignored_rather_than_fatal():
    # The model tacking on an extra key is common; failing the whole extraction over it is not worth it.
    response = ExtractionResponse.model_validate(
        {"samples": [{"sample_id": "A", "fields": [], "confidence": 0.9}], "commentary": "…"}
    )

    assert response.samples[0].sample_id == "A"


def test_an_empty_response_is_valid_and_means_nothing_was_found():
    response = ExtractionResponse.model_validate({})

    assert response.target is None and response.samples == []


# ---- response_to_records ---------------------------------------------------------------


def test_duplicate_source_ids_are_removed_but_the_order_is_kept():
    # Citing the same id twice is not an error, but the duplicate would make the provenance look like it
    # has two pieces of corroborating evidence when it only has one.
    response = ExtractionResponse.model_validate(
        {
            "samples": [
                {
                    "sample_id": "A",
                    "source_ids": ["b1", "b2", "b1"],
                    "fields": [{"field": "thickness", "value_raw": "300", "source_ids": ["b9", "b9", "b8"]}],
                }
            ]
        }
    )

    records = to_records(response)

    assert records.samples[0].source_ids == ("b1", "b2")
    assert records.samples[0].fields[0].source_ids == ("b9", "b8")


def test_surrounding_whitespace_is_stripped_from_every_text_field():
    response = ExtractionResponse.model_validate(
        {
            "samples": [
                {
                    "sample_id": "  A  ",
                    "label": "  O2 100 sccm ",
                    "conditions": {"  O2  ": "  100 sccm "},
                    "fields": [
                        {
                            "field": "thickness",
                            "value_raw": "  300 ",
                            "unit_raw": " nm ",
                            "condition": " RT ",
                            "note": "  from SEM ",
                        }
                    ],
                }
            ]
        }
    )

    sample = to_records(response).samples[0]

    assert (sample.sample_id, sample.label) == ("A", "O2 100 sccm")
    assert sample.conditions == {"O2": "100 sccm"}
    assert (sample.fields[0].value_raw, sample.fields[0].unit_raw) == ("300", "nm")
    assert (sample.fields[0].condition, sample.fields[0].note) == ("RT", "from SEM")


@pytest.mark.parametrize("blank", ["", "   "])
def test_optional_text_that_is_only_whitespace_becomes_none(blank):
    # "" and None mean two different things downstream (written but empty vs. never written); folding
    # them into one None keeps every comparison from having to re-decide which case it is.
    response = ExtractionResponse.model_validate(
        {"samples": [{"sample_id": "A", "fields": [{"field": "thickness", "value_raw": "300", "unit_raw": blank}]}]}
    )

    records = to_records(response)

    assert records.samples[0].fields[0].unit_raw is None


def test_a_target_without_fields_becomes_none():
    # An empty target record would add a spurious "paper level" tier to the report for nothing; better to
    # have no target at all.
    response = ExtractionResponse.model_validate({"target": {"source_ids": ["b1"], "fields": []}, "samples": []})

    assert to_records(response).target is None


def test_a_target_with_fields_is_kept_with_its_provenance():
    response = ExtractionResponse.model_validate(
        {"target": {"source_ids": ["b1"], "fields": [{"field": "density", "value_raw": "98.5", "unit_raw": "%"}]}}
    )

    target = to_records(response).target

    assert target.source_ids == ("b1",)
    assert target.get("density").value_raw == "98.5"


def test_records_carry_no_normalized_values_yet():
    # This step only carries the verbatim text across; value/unit are filled in by the normalisation
    # layer, and mixing the two would blur who computed what.
    response = ExtractionResponse.model_validate(make_response())

    field = to_records(response).samples[0].fields[0]

    assert (field.value, field.unit, field.normalization_note) == (None, None, None)


# ---- Cleaning ---------------------------------------------------------------------------


def test_a_source_id_that_does_not_exist_is_removed_and_reported():
    # The LLM does invent source_ids; keeping one would leave the provenance pointing at a page region
    # that does not exist.
    response = ExtractionResponse.model_validate(
        {
            "samples": [
                {
                    "sample_id": "A",
                    "fields": [{"field": "thickness", "value_raw": "300", "source_ids": ["b9", "ghost"]}],
                }
            ]
        }
    )

    records = to_records(response)

    assert records.samples[0].fields[0].source_ids == ("b9",)
    assert records.invalid_source_ids == ("ghost",)


def test_invalid_source_ids_are_deduplicated_and_sorted():
    response = ExtractionResponse.model_validate(
        {
            "samples": [
                {
                    "sample_id": "A",
                    "source_ids": ["zzz", "aaa"],
                    "fields": [{"field": "thickness", "value_raw": "1", "source_ids": ["zzz"]}],
                }
            ]
        }
    )

    assert to_records(response).invalid_source_ids == ("aaa", "zzz")


def test_a_field_outside_the_schema_is_dropped_and_logged():
    response = ExtractionResponse.model_validate(
        {"samples": [{"sample_id": "A", "fields": [{"field": "carrier_concentration", "value_raw": "1e20"}]}]}
    )

    records = to_records(response)

    assert records.samples[0].fields == ()
    assert records.dropped == ("carrier_concentration: not in schema",)


# ---- Field scope enforcement -----------------------------------------------------------


def test_a_target_level_field_reported_under_a_sample_is_dropped_and_logged():
    # `resistance` belongs to the sputtering target, not to any one film. The prompt has always said so,
    # but a prompt is a request, not an enforcement mechanism -- a real run reported a film's dopant
    # concentration as the target's own composition, which is exactly this failure mode.
    response = ExtractionResponse.model_validate(
        {"samples": [{"sample_id": "A", "fields": [{"field": "resistance", "value_raw": "0.3", "unit_raw": "Ω cm"}]}]}
    )

    records = to_records(response)

    assert records.samples[0].fields == ()
    assert records.dropped == ("resistance: target-level field reported under a sample",)


def test_a_sample_level_field_reported_under_the_target_is_dropped_and_logged():
    response = ExtractionResponse.model_validate(
        {"target": {"fields": [{"field": "thickness", "value_raw": "300", "unit_raw": "nm"}]}}
    )

    records = to_records(response)

    assert records.target is None
    assert records.dropped == ("thickness: film-level field reported under the target",)


# ---- The sample list: the same audit and placement rules as passage mode --------------------------


def thickness(value_raw: str = "300", **extra) -> dict:
    return {"field": "thickness", "value_raw": value_raw, "unit_raw": "nm", **extra}


def test_a_sample_with_no_usable_id_keeps_its_values_unattributed_and_is_audited():
    # It used to vanish in a generator filter, values and all, with nothing in the audit.
    response = ExtractionResponse.model_validate(
        {"samples": [{"sample_id": "A", "fields": [thickness()]}, {"sample_id": "  ", "fields": [thickness("250")]}]}
    )

    records = to_records(response)

    assert [sample.sample_id for sample in records.samples] == ["A"]
    assert [value.value_raw for value in records.unattributed] == ["250"]
    assert records.dropped == ("inventory: a sample was listed with no usable id",)


def test_a_sample_listed_twice_is_kept_once_its_values_filed_under_the_first_and_audited():
    response = ExtractionResponse.model_validate(
        {
            "samples": [
                {"sample_id": "S-1", "label": "first", "fields": [thickness()]},
                {"sample_id": "S 1", "label": "second", "fields": [{"field": "transmittance", "value_raw": "85"}]},
            ]
        }
    )

    records = to_records(response)

    assert [sample.label for sample in records.samples] == ["first"]
    assert [value.field for value in records.samples[0].fields] == ["thickness", "transmittance"]
    assert records.dropped == ("inventory: sample 'S 1' repeats an id already listed",)


def test_a_series_value_under_the_target_is_written_onto_every_sample():
    # Document mode's prompt asks for a whole-series value once, under the target, flagged; it is placed the
    # way passage mode places the same flag.
    response = ExtractionResponse.model_validate(
        {
            "target": {"fields": [thickness(applies_to_all_samples=True)]},
            "samples": [{"sample_id": "A"}, {"sample_id": "B"}],
        }
    )

    records = to_records(response)

    assert records.target is None
    for sample in records.samples:
        assert [(value.value_raw, value.series) for value in sample.fields] == [("300", True)]
    assert records.dropped == ()


def test_a_series_value_with_no_sample_to_carry_it_is_kept_unattributed():
    response = ExtractionResponse.model_validate({"target": {"fields": [thickness(applies_to_all_samples=True)]}})

    records = to_records(response)

    assert records.samples == ()
    assert [(value.value_raw, value.series) for value in records.unattributed] == [("300", False)]


def test_the_series_flag_changes_nothing_on_a_paper_level_field_or_under_a_sample():
    response = ExtractionResponse.model_validate(
        {
            "target": {"fields": [{"field": "component", "value_raw": "ITO", "applies_to_all_samples": True}]},
            "samples": [
                {"sample_id": "A", "fields": [thickness(applies_to_all_samples=True)]},
                {"sample_id": "B"},
            ],
        }
    )

    records = to_records(response)

    assert records.target.get("component").series is False
    assert [(value.value_raw, value.series) for value in records.samples[0].fields] == [("300", False)]
    assert records.samples[1].fields == ()


def test_a_correctly_scoped_field_of_every_group_survives():
    # Regression guard for the scope rule itself: it must reject the wrong scope without becoming
    # overzealous and rejecting the right one, for every group (target, process, film).
    response = ExtractionResponse.model_validate(
        {
            "target": {"fields": [{"field": "resistance", "value_raw": "0.3", "unit_raw": "Ω cm"}]},
            "samples": [
                {
                    "sample_id": "A",
                    "fields": [
                        {"field": "sputtering_time", "value_raw": "30", "unit_raw": "min"},
                        {"field": "resistivity", "value_raw": "1.2e-3", "unit_raw": "Ω cm"},
                    ],
                }
            ],
        }
    )

    records = to_records(response)

    assert records.target.get("resistance").value_raw == "0.3"
    assert records.samples[0].get("sputtering_time").value_raw == "30"
    assert records.samples[0].get("resistivity").value_raw == "1.2e-3"
    assert records.dropped == ()


@pytest.mark.parametrize("value_raw", ["minimum", "n.a.", "high", "a dozen", "none"])
def test_a_numeric_field_without_any_digit_is_dropped_and_logged(value_raw):
    # "minimum" is not a fact value; keeping it would only normalize to None and then show up as a line
    # of noise in the report.
    response = ExtractionResponse.model_validate(
        {"samples": [{"sample_id": "A", "fields": [{"field": "sheet_resistance", "value_raw": value_raw}]}]}
    )

    records = to_records(response)

    assert records.samples[0].fields == ()
    assert records.dropped == (f"sheet_resistance: non-numeric value {value_raw!r}",)


def test_a_numeric_field_containing_a_digit_survives():
    response = ExtractionResponse.model_validate(
        {"samples": [{"sample_id": "A", "fields": [{"field": "thickness", "value_raw": "2 μm"}]}]}
    )

    records = to_records(response)

    assert records.samples[0].get("thickness").value_raw == "2 μm"
    assert records.dropped == ()


def test_a_text_field_without_digits_is_not_affected_by_the_numeric_rule():
    response = ExtractionResponse.model_validate(
        {"target": {"fields": [{"field": "component", "value_raw": "SnO2:Ta"}]}}
    )

    records = to_records(response)

    assert records.target.get("component").value_raw == "SnO2:Ta"
    assert records.dropped == ()


def test_a_target_whose_fields_are_all_dropped_becomes_none():
    response = ExtractionResponse.model_validate(
        {"target": {"source_ids": ["ghost"], "fields": [{"field": "not_a_field", "value_raw": "x"}]}}
    )

    records = to_records(response)

    assert records.target is None
    # The target's own invented id still belongs in the audit, even though every one of its fields was
    # dropped.
    assert records.invalid_source_ids == ("ghost",)


def test_a_clean_response_reports_nothing_dropped_or_invalid():
    records = to_records(ExtractionResponse.model_validate(make_response(source_ids=["mineru_p1_b9"])))

    assert records.invalid_source_ids == () and records.dropped == ()


# ---- ResponseCleaning: the rules both extraction modes share ---------------------------
# Document mode reaches these through response_to_records; passage mode calls them directly. Testing the
# class itself is what keeps the two modes from drifting into different notions of a clean value.


def test_kept_citations_are_deduplicated_and_stay_in_the_order_the_model_gave_them():
    # Order carries meaning: the prompt asks for the most specific block first, and a caller showing
    # provenance lists them as cited.
    cleaning = ResponseCleaning()

    kept = cleaning.keep_ids(["b2", "b1", "b2", "b8"], KNOWN_IDS)

    assert kept == ("b2", "b1", "b8")


def test_a_citation_that_was_never_shown_is_stripped_and_recorded():
    # Silently dropping an invented id would leave a value looking traceable when its provenance is gone.
    cleaning = ResponseCleaning()

    kept = cleaning.keep_ids(["b1", "ghost", "b9"], KNOWN_IDS)

    assert kept == ("b1", "b9")
    assert cleaning.invalid == {"ghost"}


def test_an_invented_citation_is_recorded_once_however_often_it_is_repeated():
    cleaning = ResponseCleaning()

    cleaning.keep_ids(["ghost", "ghost"], KNOWN_IDS)
    cleaning.keep_ids(["ghost"], KNOWN_IDS)

    assert cleaning.invalid == {"ghost"}


def clean_value(field: str, value_raw: str, **overrides):
    cleaning = ResponseCleaning()
    arguments = {"unit_raw": None, "condition": None, "source_ids": [], "note": None, "known_ids": KNOWN_IDS}
    arguments.update(overrides)
    return cleaning, cleaning.value(FIELD_BY_NAME[field], value_raw=value_raw, **arguments)


def test_a_numeric_value_without_a_digit_is_dropped_with_the_reason():
    # "minimum" is a word, not a measurement; keeping it would put an unparseable value into the comparison.
    cleaning, value = clean_value("sheet_resistance", "minimum")

    assert value is None
    assert cleaning.dropped == ["sheet_resistance: non-numeric value 'minimum'"]


@pytest.mark.parametrize(
    "value_raw", ["one of the samples", "five to ten", "one-third", "ten-fold", "two-step", "one order of magnitude"]
)
def test_a_number_word_inside_other_words_is_still_non_numeric(value_raw):
    cleaning, value = clean_value("inch", value_raw, unit_raw="inch")

    assert value is None
    assert cleaning.dropped == [f"inch: non-numeric value {value_raw!r}"]


def test_a_number_word_without_a_unit_is_still_non_numeric():
    cleaning, value = clean_value("inch", "four")

    assert value is None
    assert cleaning.dropped == ["inch: non-numeric value 'four'"]


@pytest.mark.parametrize("value_raw", ["four", "four-inch", "Two", "twelve"])
def test_a_numeric_value_written_as_a_number_word_survives(value_raw):
    # metals: "a four-inch ITO target" was dropped as non-numeric in both lanes.
    cleaning, value = clean_value("inch", value_raw, unit_raw="inch")

    assert value is not None and value.value_raw == value_raw
    assert cleaning.dropped == []


def test_a_composition_value_without_a_digit_survives_the_numeric_rule():
    # The rule is about numeric fields only: a target composition is text and is allowed to have no digits.
    cleaning, value = clean_value("component", "SnO2:Ta")

    assert value is not None
    assert (value.field, value.value_raw) == ("component", "SnO2:Ta")
    assert cleaning.dropped == []


def test_the_cleaned_value_is_stamped_with_the_field_that_was_asked_about():
    # Passage mode never sends a field name, so the spec is the only source of truth for it.
    _, value = clean_value("thickness", "300", unit_raw="nm")

    assert value.field == "thickness"


def test_surrounding_whitespace_is_stripped_and_blank_optional_text_becomes_none():
    _, value = clean_value("thickness", "  300  ", unit_raw="  nm  ", condition="   ", note="")

    assert (value.value_raw, value.unit_raw) == ("300", "nm")
    assert value.condition is None and value.note is None


def test_a_cleaned_value_carries_only_the_citations_it_was_shown():
    cleaning, value = clean_value("thickness", "300", source_ids=["b1", "ghost"])

    assert value.source_ids == ("b1",)
    assert cleaning.invalid == {"ghost"}
