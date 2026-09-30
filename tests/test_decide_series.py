"""A value the paper states for the whole series never outranks the lane's own value for the sample.

om0035's PaddleOCR lane read "~75–81 %" for all four films beside each film's own table value, and the range cited
a block the rest of the row cites, so narrowing rule 1 kept the range and the cell went blank. The rule is judged
per lane and per measurement: only a scalar for the sample at the same condition supersedes the series value.
"""

from __future__ import annotations

import json
from pathlib import Path

from paperfacts.compare import FieldComparison, Supervision, SupervisorScore
from paperfacts.decide import decide, decide_many
from paperfacts.records import NO_CONTEXT, FieldValue
from test_cardinality_many import SPEC as LIST_SPEC

REAL = Path(__file__).parent / "fixtures" / "real" / "om0035" / "transmittance.json"
EXCLUDED = "已排除整系列表述的候选"


def val(raw, *, condition=None, backend="mineru", block=1, series=False, field="transmittance", unit="%", **kw):
    return FieldValue(
        field=field,
        value_raw=raw,
        unit_raw=unit,
        condition=condition,
        source_ids=(f"{backend}_p0_b{block}",),
        series=series,
        **kw,
    )


def compared(status, a=None, b=None, *, field="transmittance", **kw):
    return FieldComparison(scope="sample:A|A", field=field, status=status, a=a, b=b, **kw)


def cell(spec, evidence, comparisons, **kw):
    return decide(spec, evidence, comparisons, units=kw.pop("units"), ctx=NO_CONTEXT, **kw)


def om0035_shape(own_mineru: str, own_paddle: str):
    mineru = val(own_mineru, condition="400–700 nm", block=6)
    paddle = val(own_paddle, condition="average 400–700 nm", backend="paddleocr_vl", block=2)
    series = FieldValue(
        field="transmittance",
        value_raw="~75–81",
        unit_raw="%",
        condition="400–700 nm",
        source_ids=("paddleocr_vl_p0_b2", "paddleocr_vl_p5_b3"),
        series=True,
    )
    return mineru, paddle, series


def test_a_series_range_never_outranks_the_lanes_own_sample_value(tco_profile):
    mineru, paddle, series = om0035_shape("77.0", "77.0")
    evidence = [("mineru", mineru), ("paddleocr_vl", paddle), ("paddleocr_vl", series)]
    comparisons = [compared("agree", mineru, series), compared("missing", b=paddle, missing_in="mineru")]

    result = cell(
        tco_profile.by_name["transmittance"],
        evidence,
        comparisons,
        units=tco_profile.units,
        row_sources=frozenset({"paddleocr_vl_p5_b3"}),
    )

    assert (result.status, result.value, result.series) == ("agree", 77.0, False)
    assert EXCLUDED in result.detail and "~75–81" in result.detail


def test_a_conflict_with_a_superseded_series_value_does_not_refuse_the_cell(tco_profile):
    mineru, paddle, series = om0035_shape("75.4", "75.4")
    evidence = [("mineru", mineru), ("paddleocr_vl", paddle), ("paddleocr_vl", series)]
    comparisons = [compared("conflict", mineru, series), compared("missing", b=paddle, missing_in="mineru")]

    result = cell(tco_profile.by_name["transmittance"], evidence, comparisons, units=tco_profile.units)

    assert (result.status, result.value) == ("agree", 75.4)


def test_a_series_value_at_another_condition_is_not_superseded(tco_profile):
    series = val("85", condition="550 nm", series=True, block=1)
    peak = val("92", condition="600 nm (peak)", block=2)
    comparisons = [compared("missing", a=series, missing_in="paddleocr_vl"), compared("missing", a=peak)]

    result = cell(
        tco_profile.by_name["transmittance"],
        [("mineru", series), ("mineru", peak)],
        comparisons,
        units=tco_profile.units,
    )

    # The preferred 550 nm measurement is the series one, as before the rule existed.
    assert (result.status, result.value) == ("single_source", 85.0)
    assert EXCLUDED not in result.detail


def test_a_sample_bound_does_not_supersede_a_series_scalar(tco_profile):
    series = val("85", condition="550 nm", series=True, block=1)
    bound = val("80", condition="550 nm", block=2, bound="above")
    comparisons = [compared("missing", a=series), compared("missing", a=bound)]

    result = cell(
        tco_profile.by_name["transmittance"],
        [("mineru", series), ("mineru", bound)],
        comparisons,
        units=tco_profile.units,
    )

    assert (result.status, result.value) == ("single_source", 85.0)
    assert EXCLUDED not in result.detail


def test_a_lane_holding_only_series_values_keeps_them(tco_profile):
    a = val("100", unit="nm", field="thickness", series=True)
    b = val("100", unit="nm", field="thickness", series=True, backend="paddleocr_vl")

    result = cell(
        tco_profile.by_name["thickness"],
        [("mineru", a), ("paddleocr_vl", b)],
        [compared("agree", a, b, field="thickness")],
        units=tco_profile.units,
    )

    assert (result.status, result.value, result.series) == ("agree", 100.0, True)


def test_a_series_value_in_one_lane_is_still_compared_with_the_others_sample_value(tco_profile):
    a = val("100", unit="nm", field="thickness", series=True)
    b = val("120", unit="nm", field="thickness", backend="paddleocr_vl")

    result = cell(
        tco_profile.by_name["thickness"],
        [("mineru", a), ("paddleocr_vl", b)],
        [compared("conflict", a, b, field="thickness")],
        units=tco_profile.units,
    )

    assert result.status == "conflict"


def test_a_stale_conflict_matching_no_candidate_still_refuses(tco_profile):
    mineru, paddle, series = om0035_shape("77.0", "77.0")
    stale = compared("conflict", val("60", condition="400–700 nm", block=9), paddle)

    result = cell(
        tco_profile.by_name["transmittance"],
        [("mineru", mineru), ("paddleocr_vl", paddle), ("paddleocr_vl", series)],
        [stale],
        units=tco_profile.units,
    )

    assert result.status == "conflict"


def test_a_settled_conflict_is_applied_before_sample_over_series(tco_profile):
    # The supervisor rules mineru's own 70.0 out first; the lane is then left with its series range alone, which
    # nothing supersedes, and the cell rests on the trusted PaddleOCR value.
    own = val("70.0", condition="400–700 nm", block=6)
    series = val("~75–81", condition="400–700 nm", block=7, series=True)
    paddle = val("77.0", condition="400–700 nm", backend="paddleocr_vl", block=2)
    settled = compared(
        "conflict",
        own,
        paddle,
        supervision=Supervision(
            reason="conflict",
            a=SupervisorScore(score=0.2, flag="value_not_in_passage", verdict="doubted"),
            b=SupervisorScore(score=0.9, flag="correct", verdict="trusted"),
        ),
    )

    result = cell(
        tco_profile.by_name["transmittance"],
        [("mineru", own), ("mineru", series), ("paddleocr_vl", paddle)],
        [settled],
        units=tco_profile.units,
    )

    assert (result.status, result.value) == ("supervised", 77.0)
    assert EXCLUDED not in result.detail


def test_decide_many_keeps_a_series_element_beside_a_sample_element():
    def element(raw, *, series):
        return FieldValue(field=LIST_SPEC.name, value_raw=raw, source_ids=("mineru_p0_b1",), series=series)

    xrd, xps = element("XRD", series=True), element("XPS", series=False)
    comparisons = [FieldComparison(scope="paper", field=LIST_SPEC.name, status="missing", a=xrd)]

    decision = decide_many(LIST_SPEC, [("mineru", xrd), ("mineru", xps)], comparisons)

    assert decision.value == ["XRD", "XPS"]


def test_om0035_z1_and_az1_read_their_table_values(tco_profile):
    real = json.loads(REAL.read_text(encoding="utf-8"))["samples"]
    spec = tco_profile.by_name["transmittance"]

    def decided(sample):
        data = real[sample]
        evidence = [(backend, FieldValue.model_validate(value)) for backend, value in data["evidence"]]
        comparisons = [FieldComparison.model_validate(c | {"scope": c["scope"]}) for c in data["comparisons"]]
        return cell(spec, evidence, comparisons, units=tco_profile.units, row_sources=frozenset(data["row_sources"]))

    z1, az1 = decided("Z-1"), decided("AZ-1")

    assert (z1.status, z1.value) == ("agree", 75.4)
    assert (az1.status, az1.value) == ("agree", 77.0)


def test_a_peak_cited_by_the_row_does_not_win_over_the_preferred_average(tco_profile):
    # om0035 AZ-2: both lanes also quote the abstract's peak, which the row's other fields cite too.
    average_a = val("80.8", condition="400–700 nm", block=6)
    peak_a = val("86.4", condition="498 nm", block=13)
    average_b = val("80.8", condition="average 400–700 nm", backend="paddleocr_vl", block=2)
    peak_b = val("86.4", condition="at 498 nm", backend="paddleocr_vl", block=12)
    comparisons = [compared("agree", average_a, average_b), compared("agree", peak_a, peak_b)]

    result = cell(
        tco_profile.by_name["transmittance"],
        [("mineru", average_a), ("mineru", peak_a), ("paddleocr_vl", average_b), ("paddleocr_vl", peak_b)],
        comparisons,
        units=tco_profile.units,
        row_sources=frozenset({"mineru_p0_b13", "paddleocr_vl_p0_b12"}),
    )

    assert (result.status, result.value) == ("agree", 80.8)


def test_the_row_still_decides_between_preferred_conditions(tco_profile):
    at_550 = val("85", condition="550 nm", block=1)
    average = val("83.5", condition="average 400-800 nm", block=2)
    comparisons = [compared("missing", a=at_550), compared("missing", a=average)]

    result = cell(
        tco_profile.by_name["transmittance"],
        [("mineru", at_550), ("mineru", average)],
        comparisons,
        units=tco_profile.units,
        row_sources=frozenset({"mineru_p0_b1"}),
    )

    assert result.value == 85.0


def test_a_series_value_worded_as_another_quantity_is_not_superseded(tco_profile):
    # materials-13-00113: the IWO layers' 50 min is not the Cu interlayer's 1 min, and neither condition names a
    # number that could show it; the cell stays for review, as before.
    cu = val("1", unit="min", field="sputtering_time", condition="Cu layer sputtering time (4 nm Cu)")
    iwo = val(
        "50", unit="min", field="sputtering_time", condition="IWO layer (upper and bottom) sputtering time", series=True
    )

    result = cell(
        tco_profile.by_name["sputtering_time"],
        [("mineru", cu), ("mineru", iwo)],
        [compared("missing", a=cu, field="sputtering_time"), compared("missing", a=iwo, field="sputtering_time")],
        units=tco_profile.units,
    )

    assert result.value is None
    assert EXCLUDED not in result.detail


def test_a_series_statement_with_no_condition_yields_to_the_samples_value(tco_profile):
    # 34858.pdf: "over 70 %" for every film beside each film's own 76.9 % (380-780 nm average).
    own = val("76.9", condition="average 380-780 nm", block=2)
    summary = val("over 70", condition=None, series=True, block=1)
    other = val("76.9", condition="average transmittance in 380-780 nm", backend="paddleocr_vl", block=2)

    result = cell(
        tco_profile.by_name["transmittance"],
        [("mineru", own), ("mineru", summary), ("paddleocr_vl", other)],
        [compared("agree", own, other), compared("missing", a=summary)],
        units=tco_profile.units,
    )

    assert (result.status, result.value) == ("agree", 76.9)
