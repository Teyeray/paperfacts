"""Render extraction results and comparison reports into human-readable lines.

Rendering is the only observation window during research: these are the lines you look at when
eyeballing a failure case. So what must be guarded is "not a single piece of expected information
is missing" — especially provenance ids and anything dropped during cleaning. Without them, the
terminal looks fine while the problem has already been silently swallowed.
"""

from __future__ import annotations

from paperfacts.compare import compare_lanes
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.records import TargetRecord
from paperfacts.report import render_lane, render_report
from support.extraction import comparison_options, make_field, make_lane, make_sample

# ---- render_lane --------------------------------------------------------------------


def lane_text(**kwargs) -> str:
    return "\n".join(render_lane(make_lane(**kwargs)))


def test_the_header_summarises_the_counts_the_cost_and_the_extractor():
    text = lane_text(
        samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm")])],
        target=TargetRecord(fields=(make_field("density", "98.5", unit_raw="%"),)),
        invalid_source_ids=("ghost",),
        dropped=("sheet_resistance: non-numeric value 'minimum'",),
        usage={"total_tokens": 1234},
    )
    header = text.splitlines()[0]

    assert "samples=1" in header and "target_fields=1" in header
    assert "invalid_source_ids=1" in header and "dropped=1" in header
    assert "tokens=1234" in header
    assert "model=fake-model" in header


def test_a_lane_without_usage_shows_a_question_mark_rather_than_crashing():
    assert "tokens=?" in lane_text().splitlines()[0]


def test_every_field_shows_its_raw_text_its_unit_and_its_provenance():
    # Without a source id there's no way back to the PDF to verify, which makes the line meaningless.
    text = lane_text(
        samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm", source_ids=("mineru_p1_b9",))])]
    )

    assert "thickness: 300 nm" in text
    assert "← mineru_p1_b9" in text


def test_a_field_without_provenance_says_so_explicitly():
    text = lane_text(samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm")])])

    assert "(no source)" in text


def test_a_normalized_value_is_shown_next_to_the_raw_one():
    # Showing both side by side makes a conversion error visible at a glance.
    text = lane_text(
        samples=[make_sample("A", [make_field("thickness", "0.3", unit_raw="μm", value=300.0, unit="nm")])]
    )

    assert "0.3 μm" in text and "= 300 nm" in text


def test_a_measurement_condition_is_shown_with_the_value():
    text = lane_text(samples=[make_sample("A", [make_field("transmittance", "85", unit_raw="%", condition="550 nm")])])

    assert "@550 nm" in text


def test_target_fields_are_labelled_as_such():
    text = lane_text(target=TargetRecord(fields=(make_field("density", "98.5", unit_raw="%"),)))

    assert "target.density: 98.5 %" in text


def test_each_sample_shows_its_id_and_label():
    text = lane_text(samples=[make_sample("A", label="O2 100 sccm")])

    assert "A  (O2 100 sccm)" in text


def test_an_empty_lane_renders_just_the_header():
    assert len(lane_text().splitlines()) == 1


# ---- render_report ------------------------------------------------------------------


def report_text(lane_a, lane_b, matching) -> str:
    return "\n".join(render_report(compare_lanes(lane_a, lane_b, matching, comparison_options())))


def test_the_counts_line_comes_first():
    text = report_text(make_lane(), make_lane(backend="paddleocr_vl"), SampleMatching())

    assert text.splitlines()[0].startswith("counts: ")
    assert "'agree': 0" in text


def test_each_pair_shows_both_ids_the_confidence_the_method_and_the_reason():
    fields = [make_field("thickness", "300", unit_raw="nm")]
    lane_a = make_lane(backend="mineru", samples=[make_sample("A1", fields)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("B1", fields)])
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="A1", b_id="B1", confidence=0.85, justification="both 100 sccm", method="llm"),)
    )

    text = report_text(lane_a, lane_b, matching)

    assert "match A1 ↔ B1  conf=0.85 (llm): both 100 sccm" in text


def test_unmatched_samples_are_listed_per_backend():
    lane_a = make_lane(backend="mineru", samples=[make_sample("S1", [make_field("thickness", "1", unit_raw="nm")])])
    lane_b = make_lane(
        backend="paddleocr_vl", samples=[make_sample("T1", [make_field("thickness", "2", unit_raw="nm")])]
    )

    text = report_text(lane_a, lane_b, SampleMatching(unmatched_a=("S1",), unmatched_b=("T1",)))

    assert "only in mineru: S1" in text
    assert "only in paddleocr_vl: T1" in text


def test_a_matching_failure_is_shouted_rather_than_hidden():
    """A matching failure turns every unpaired sample into "ambiguous"; staying silent about it
    would make the reader think that's a genuine disagreement.
    """
    lane_a = make_lane(backend="mineru", samples=[make_sample("S1", [make_field("thickness", "1", unit_raw="nm")])])
    lane_b = make_lane(
        backend="paddleocr_vl", samples=[make_sample("T1", [make_field("thickness", "2", unit_raw="nm")])]
    )
    matching = SampleMatching(unmatched_a=("S1",), unmatched_b=("T1",), failed=True, failure="two bad answers")

    text = report_text(lane_a, lane_b, matching)

    assert "sample matching FAILED: two bad answers" in text


def test_a_comparison_line_shows_both_candidate_values_and_the_reason():
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", [make_field("thickness", "300", unit_raw="nm")])])
    lane_b = make_lane(
        backend="paddleocr_vl", samples=[make_sample("A", [make_field("thickness", "900", unit_raw="nm")])]
    )
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, justification="", method="exact"),)
    )

    line = next(ln for ln in report_text(lane_a, lane_b, matching).splitlines() if "thickness" in ln)

    assert "conflict" in line
    assert "300 nm" in line and "900 nm" in line
    assert "300 vs 900" in line


def test_a_missing_fact_names_the_backend_that_lacks_it():
    # Saying "missing" without naming which side is missing says nothing useful.
    lane_a = make_lane(backend="mineru", samples=[make_sample("S1", [make_field("thickness", "1", unit_raw="nm")])])
    lane_b = make_lane(backend="paddleocr_vl")

    text = report_text(lane_a, lane_b, SampleMatching(unmatched_a=("S1",)))

    assert "missing(paddleocr_vl)" in text
    assert "—" in text  # the missing side is shown as an em dash placeholder


def test_a_low_confidence_pair_shows_its_confidence_on_every_line():
    fields = [make_field("thickness", "300", unit_raw="nm")]
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", fields)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", fields)])
    weak = SampleMatching(pairs=(SampleMatch(a_id="A", b_id="A", confidence=0.3, justification="weak", method="llm"),))

    line = next(ln for ln in report_text(lane_a, lane_b, weak).splitlines() if "thickness" in ln)

    assert "conf=0.30" in line


def test_an_exact_pair_shows_no_confidence_on_the_comparison_lines():
    fields = [make_field("thickness", "300", unit_raw="nm")]
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", fields)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", fields)])
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, justification="", method="exact"),)
    )

    line = next(ln for ln in report_text(lane_a, lane_b, matching).splitlines() if "thickness" in ln)

    assert "conf=" not in line


def test_the_condition_is_shown_on_the_comparison_line():
    fields = [make_field("transmittance", "85", unit_raw="%", condition="550 nm")]
    lane_a = make_lane(backend="mineru", samples=[make_sample("A", fields)])
    lane_b = make_lane(backend="paddleocr_vl", samples=[make_sample("A", fields)])
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, justification="", method="exact"),)
    )

    assert "@550 nm" in report_text(lane_a, lane_b, matching)


def test_both_renderers_return_iterators_so_the_cli_can_stream_them():
    # Returns an iterator rather than printing directly: tests can assert on content, and there's a
    # place to add --json later.
    lane = make_lane()

    assert iter(render_lane(lane)) is not None
    assert list(render_lane(lane)) == list(render_lane(lane))
