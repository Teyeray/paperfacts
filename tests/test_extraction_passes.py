"""Merging several extraction passes over the same lane by majority vote.

Even at temperature 0 a model is not perfectly deterministic across calls, so a single pass cannot tell a
genuine parser-level disagreement from ordinary sampling noise. ``merge_passes`` runs the same lane N
times and keeps only what a majority of passes agree on -- see :mod:`paperfacts.extraction.passes` for the
full rationale. This file pins the voting rules themselves: value-level agreement, sample identity, the
target record, and how the audit trails (``invalid_source_ids``, ``dropped``) from every pass combine.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from paperfacts.extraction.passes import TARGET_SCOPE, merge_passes
from paperfacts.extraction.records import ExtractedRecords, SampleRecord, TargetRecord
from support.extraction import make_field, make_sample


def records(
    *,
    target: TargetRecord | None = None,
    samples: Sequence[SampleRecord] = (),
    invalid: Sequence[str] = (),
    dropped: Sequence[str] = (),
) -> ExtractedRecords:
    return ExtractedRecords(
        target=target, samples=tuple(samples), invalid_source_ids=tuple(invalid), dropped=tuple(dropped)
    )


def find(merged: ExtractedRecords, sample_id: str) -> SampleRecord | None:
    return next((s for s in merged.samples if s.sample_id == sample_id), None)


# ---- A single pass is a no-op ----------------------------------------------------------


def test_a_single_pass_is_returned_unchanged():
    # With nothing to vote against, merging must not even reconstruct the object -- the same result
    # single-pass extraction already produces.
    one = records(samples=[make_sample("A", [make_field("sheet_resistance", "12.5")])])

    assert merge_passes([one]) is one


# ---- Majority voting on values -----------------------------------------------------------


def test_a_value_produced_by_every_pass_gets_full_agreement():
    passes = [records(samples=[make_sample("A", [make_field("sheet_resistance", "12.5")])]) for _ in range(3)]

    merged = merge_passes(passes)

    field = find(merged, "A").get("sheet_resistance")
    assert field.value_raw == "12.5"
    assert field.agreement == 1.0


def test_a_value_produced_by_a_bare_majority_survives_with_partial_agreement():
    passes = [
        records(samples=[make_sample("A", [make_field("sheet_resistance", "12.5")])]),
        records(samples=[make_sample("A", [make_field("sheet_resistance", "12.5")])]),
        records(samples=[make_sample("A", [])]),  # this pass saw the sample but found nothing on it
    ]

    merged = merge_passes(passes)

    field = find(merged, "A").get("sheet_resistance")
    assert field.agreement == pytest.approx(2 / 3)


def test_a_value_produced_by_only_one_pass_is_dropped_with_a_reason():
    passes = [
        records(samples=[make_sample("A", [make_field("thickness", "300")])]),
        records(samples=[make_sample("A", [])]),
        records(samples=[make_sample("A", [])]),
    ]

    merged = merge_passes(passes)

    assert find(merged, "A").get("thickness") is None
    assert "thickness: only 1/3 passes produced '300'" in merged.dropped


def test_a_condition_spelled_differently_across_passes_still_counts_as_one_vote():
    # normalize_key folds the condition text, so "RT" and " rt " must be recognised as the same value,
    # not split into two spellings that each fail to reach a majority on their own.
    passes = [
        records(samples=[make_sample("A", [make_field("transmittance", "80", condition="RT")])]),
        records(samples=[make_sample("A", [make_field("transmittance", "80", condition=" rt ")])]),
        records(samples=[make_sample("A", [])]),
    ]

    merged = merge_passes(passes)

    field = find(merged, "A").get("transmittance")
    assert field.agreement == pytest.approx(2 / 3)


def test_the_surviving_values_source_ids_come_from_the_first_pass_that_produced_it():
    # _value_key ignores source_ids, so two passes citing different blocks for the same number still count
    # as one vote; the exemplar that is kept is whichever pass was seen first.
    first = make_field("sheet_resistance", "12.5", source_ids=("mineru_p0_b1",))
    second = make_field("sheet_resistance", "12.5", source_ids=("paddleocr_vl_p0_b3",))
    passes = [records(samples=[make_sample("A", [first])]), records(samples=[make_sample("A", [second])])]

    merged = merge_passes(passes)

    field = find(merged, "A").get("sheet_resistance")
    assert field.source_ids == ("mineru_p0_b1",)
    assert field.agreement == 1.0


# ---- Sample identity is voted on separately from its values --------------------------------


def test_a_sample_agreed_on_by_a_majority_survives_even_when_none_of_its_fields_do():
    # "this sample exists, under these conditions" is itself a finding, reported the same way a
    # single-pass extraction would report it; it must not vanish just because its measurements were noisy.
    passes = [
        records(samples=[make_sample("B", [make_field("thickness", "300")])]),
        records(samples=[make_sample("B", [])]),
        records(samples=[make_sample("B", [])]),
    ]

    merged = merge_passes(passes)

    sample = find(merged, "B")
    assert sample is not None
    assert sample.fields == ()


def test_a_sample_reported_by_only_a_minority_of_passes_is_dropped_entirely():
    passes = [records(samples=[make_sample("C", [make_field("thickness", "300")])]), records(), records()]

    merged = merge_passes(passes)

    assert find(merged, "C") is None


def test_passes_spelling_the_sample_id_differently_still_merge_into_one_sample():
    # normalize_key keeps a hyphen but drops a space, so the three spellings below have to differ only by
    # case and whitespace -- not by punctuation -- to actually land on the same key.
    passes = [
        records(samples=[make_sample("Sample A", [make_field("sheet_resistance", "12.5")])]),
        records(samples=[make_sample("sample a", [make_field("sheet_resistance", "12.5")])]),
        records(samples=[make_sample(" SAMPLE A ", [make_field("sheet_resistance", "12.5")])]),
    ]

    merged = merge_passes(passes)

    assert len(merged.samples) == 1
    assert merged.samples[0].sample_id == "Sample A"  # the spelling of the first pass to report it wins


# ---- The target record follows the same value-level rule, with no separate identity vote --------


def test_the_targets_fields_are_merged_by_the_same_majority_rule():
    with_density = TargetRecord(source_ids=("mineru_p0_b0",), fields=(make_field("density", "98.5", unit_raw="%"),))
    without_density = TargetRecord(source_ids=("mineru_p0_b0",), fields=())
    passes = [records(target=with_density), records(target=with_density), records(target=without_density)]

    merged = merge_passes(passes)

    assert merged.target.get("density").agreement == pytest.approx(2 / 3)
    assert merged.target.source_ids == ("mineru_p0_b0",)


def test_a_target_with_no_surviving_fields_is_dropped_entirely():
    # Unlike a sample, the target has no separate identity vote -- there is at most one per paper -- so it
    # disappears whenever every one of its fields fails to reach a majority.
    passes = [
        records(target=TargetRecord(fields=(make_field("density", "98.5"),))),
        records(target=None),
        records(target=None),
    ]

    merged = merge_passes(passes)

    assert merged.target is None


# ---- Audit trails from every pass are combined, never silently narrowed to one -------------------


def test_invalid_source_ids_are_the_sorted_union_of_every_pass():
    passes = [records(invalid=["b"]), records(invalid=["a"]), records(invalid=["a", "c"])]

    merged = merge_passes(passes)

    assert merged.invalid_source_ids == ("a", "b", "c")


def test_pre_existing_dropped_reasons_are_kept_and_exact_duplicates_collapsed():
    passes = [
        records(dropped=["carrier_concentration: not in schema"]),
        records(dropped=["carrier_concentration: not in schema"]),
        records(dropped=[]),
    ]

    merged = merge_passes(passes)

    assert merged.dropped.count("carrier_concentration: not in schema") == 1


# ---- the target scope cannot collide with a sample scope -------------------------------


def test_a_sample_value_never_leaks_into_the_target_record():
    """The paper-level record is keyed by ``None``, not by a reserved string.

    With a string sentinel, a sample whose id normalises to the empty string shared the target's scope key
    and its measurements were merged into the target record -- reintroducing, one layer later, exactly the
    scope confusion that ``response_to_records`` exists to prevent.
    """
    blank = SampleRecord(sample_id=" ", fields=(make_field("sheet_resistance", "15.6", unit_raw="Ω/sq"),))
    results = [ExtractedRecords(target=None, samples=(blank,), invalid_source_ids=(), dropped=()) for _ in range(2)]

    merged = merge_passes(results)

    assert merged.target is None
    assert [v.field for s in merged.samples for v in s.fields] == ["sheet_resistance"]


def test_the_target_scope_is_not_a_string():
    assert TARGET_SCOPE is None
