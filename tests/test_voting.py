"""Merging several extraction passes over the same lane by majority vote.

Even at temperature 0 a model is not perfectly deterministic across calls, so a single pass cannot tell a
genuine parser-level disagreement from ordinary sampling noise. ``merge_passes`` runs the same lane N
times and keeps only what a majority of passes agree on -- see :func:`paperfacts.voting.merge_passes` for the
full rationale. This file pins the voting rules themselves: value-level agreement, sample identity, the
target record, and how the audit trails (``invalid_source_ids``, ``dropped``) from every pass combine.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from paperfacts.records import ExtractedRecords, SampleRecord, TargetRecord
from paperfacts.voting import Scope, merge_passes
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


def test_the_surviving_value_keeps_the_first_passs_wording_and_every_passs_citation():
    # The vote ignores source_ids, so two passes citing different blocks for the same number still count as
    # one vote. The exemplar kept is the first pass's, but the evidence is the union: a block only the
    # second pass quoted is still a block that supports the value.
    first = make_field("sheet_resistance", "12.5", source_ids=("mineru_p0_b1",))
    second = make_field("sheet_resistance", "12.5", source_ids=("paddleocr_vl_p0_b3",))
    passes = [records(samples=[make_sample("A", [first])]), records(samples=[make_sample("A", [second])])]

    merged = merge_passes(passes)

    field = find(merged, "A").get("sheet_resistance")
    assert field.source_ids == ("mineru_p0_b1", "paddleocr_vl_p0_b3")
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
    # sample_key folds the case of words and treats spaces and hyphens as separators; only a single-letter
    # suffix keeps its case ("A" vs "a" may be two samples), so the three spellings keep that letter.
    passes = [
        records(samples=[make_sample("Sample A", [make_field("sheet_resistance", "12.5")])]),
        records(samples=[make_sample("sample-A", [make_field("sheet_resistance", "12.5")])]),
        records(samples=[make_sample(" SAMPLE  A ", [make_field("sheet_resistance", "12.5")])]),
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


def test_the_scopes_that_are_not_a_sample_cannot_collide_with_a_sample_id():
    # A sample's scope is its normalised id, always a string; these two are not strings at all, so no
    # paper can name a sample in a way that lands its values in the target or unattributed bucket.
    assert not isinstance(Scope.TARGET, str)
    assert not isinstance(Scope.UNATTRIBUTED, str)


# ---- The vote ignores the condition wording; the entries still carry it ------------------------


def test_a_paraphrased_condition_is_one_value_carrying_the_first_passs_wording():
    # The measured defect: the model rewords the condition on every pass, and a vote on the full key gave
    # each wording a single vote, so nothing reached a majority and two passes destroyed the extraction.
    passes = [
        records(samples=[make_sample("A", [make_field("transmittance", "85", condition="at 550 nm")])]),
        records(samples=[make_sample("A", [make_field("transmittance", "85", condition="550 nm wavelength")])]),
    ]

    merged = merge_passes(passes)

    fields = [v for v in find(merged, "A").fields if v.field == "transmittance"]
    assert len(fields) == 1
    assert fields[0].condition == "at 550 nm"  # the exemplar is the first pass to report the number
    assert fields[0].agreement == 1.0


def test_two_different_numbers_under_the_same_condition_still_fail_the_vote():
    # Voting condition-free must not make disagreement disappear: these are two different measurements,
    # each seen once, and both are dropped exactly as before.
    passes = [
        records(samples=[make_sample("A", [make_field("transmittance", "85", condition="at 550 nm")])]),
        records(samples=[make_sample("A", [make_field("transmittance", "90", condition="at 550 nm")])]),
    ]

    merged = merge_passes(passes)

    assert [v for v in find(merged, "A").fields if v.field == "transmittance"] == []
    assert "transmittance: only 1/2 passes produced '85'" in merged.dropped
    assert "transmittance: only 1/2 passes produced '90'" in merged.dropped


def test_one_number_under_two_genuine_conditions_keeps_both_entries():
    # 85 % at 550 nm and 85 % at 600 nm are two measurements that happen to share a number. Collapsing
    # them would merge conditions, which the extraction never does.
    def pass_with_both():
        return records(
            samples=[
                make_sample(
                    "A",
                    [
                        make_field("transmittance", "85", condition="at 550 nm"),
                        make_field("transmittance", "85", condition="at 600 nm"),
                    ],
                )
            ]
        )

    merged = merge_passes([pass_with_both(), pass_with_both()])

    fields = [v for v in find(merged, "A").fields if v.field == "transmittance"]
    assert [v.condition for v in fields] == ["at 550 nm", "at 600 nm"]
    assert [v.agreement for v in fields] == [1.0, 1.0]


def test_a_paraphrase_seen_in_two_of_three_passes_keeps_the_partial_agreement():
    passes = [
        records(samples=[make_sample("A", [make_field("transmittance", "85", condition="at 550 nm")])]),
        records(samples=[make_sample("A", [make_field("transmittance", "85", condition="measured at 550 nm")])]),
        records(samples=[make_sample("A", [])]),
    ]

    merged = merge_passes(passes)

    fields = [v for v in find(merged, "A").fields if v.field == "transmittance"]
    assert len(fields) == 1
    assert fields[0].agreement == pytest.approx(2 / 3)


def test_unattributed_values_are_voted_on_condition_free_too():
    def unplaced(condition: str) -> ExtractedRecords:
        return ExtractedRecords(
            target=None,
            samples=(),
            invalid_source_ids=(),
            dropped=(),
            unattributed=(make_field("transmittance", "85", condition=condition),),
        )

    merged = merge_passes([unplaced("at 550 nm"), unplaced("550 nm wavelength")])

    assert len(merged.unattributed) == 1
    assert merged.unattributed[0].condition == "at 550 nm"
    assert merged.unattributed[0].agreement == 1.0


def test_a_paraphrase_that_loses_its_wording_still_contributes_its_citation():
    # Only one wording survives, but both passes supported the number: the block pass 2 quoted has to
    # reach the kept value, or merging passes would quietly narrow the evidence.
    passes = [
        records(
            samples=[
                make_sample(
                    "A", [make_field("transmittance", "85", condition="at 550 nm", source_ids=("mineru_p0_b1",))]
                )
            ]
        ),
        records(
            samples=[
                make_sample(
                    "A",
                    [make_field("transmittance", "85", condition="550 nm wavelength", source_ids=("mineru_p0_b7",))],
                )
            ]
        ),
    ]

    merged = merge_passes(passes)

    fields = [v for v in find(merged, "A").fields if v.field == "transmittance"]
    assert len(fields) == 1
    assert fields[0].source_ids == ("mineru_p0_b1", "mineru_p0_b7")


def test_a_second_condition_only_one_pass_reported_is_dropped_like_any_lone_value():
    # Both passes place the number at 550 nm, so that entry is unanimous; the extra 600 nm entry has no
    # counterpart in pass 1 and must not be carried in on the back of the agreement about 550 nm.
    passes = [
        records(samples=[make_sample("A", [make_field("transmittance", "85", condition="at 550 nm")])]),
        records(
            samples=[
                make_sample(
                    "A",
                    [
                        make_field("transmittance", "85", condition="at 550 nm"),
                        make_field("transmittance", "85", condition="at 600 nm"),
                    ],
                )
            ]
        ),
    ]

    merged = merge_passes(passes)

    fields = [v for v in find(merged, "A").fields if v.field == "transmittance"]
    assert [(v.condition, v.agreement) for v in fields] == [("at 550 nm", 1.0)]
    assert "transmittance: only 1/2 passes produced '85'" in merged.dropped


def test_citations_are_not_carried_across_conditions_when_the_passes_disagree_on_order():
    # Rank matching pairs entries by position, so opposite orders pair 550 nm with 600 nm. Both ranks are
    # unanimous, but the citation union must not move pass 2's 600 nm block onto the 550 nm exemplar.
    first = records(
        samples=[
            make_sample(
                "A",
                [
                    make_field("transmittance", "85", condition="at 550 nm", source_ids=("mineru_p0_b1",)),
                    make_field("transmittance", "85", condition="at 600 nm", source_ids=("mineru_p0_b2",)),
                ],
            )
        ]
    )
    second = records(
        samples=[
            make_sample(
                "A",
                [
                    make_field("transmittance", "85", condition="at 600 nm", source_ids=("mineru_p0_b9",)),
                    make_field("transmittance", "85", condition="at 550 nm", source_ids=("mineru_p0_b8",)),
                ],
            )
        ]
    )

    merged = merge_passes([first, second])

    fields = [v for v in find(merged, "A").fields if v.field == "transmittance"]
    assert [(v.condition, v.agreement) for v in fields] == [("at 550 nm", 1.0), ("at 600 nm", 1.0)]
    # Each entry keeps its own citation; only a matching condition key may add to it.
    assert [v.source_ids for v in fields] == [("mineru_p0_b1",), ("mineru_p0_b2",)]
