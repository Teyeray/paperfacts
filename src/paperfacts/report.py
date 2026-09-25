"""Render extraction results and comparison reports into human-readable text lines.

Returns an iterator of lines rather than printing directly: the CLI does the echoing, tests can
assert on content directly, and there's a natural place to add ``--json`` later.
"""

from __future__ import annotations

from collections.abc import Iterator

from paperfacts.compare import ComparisonReport, FieldComparison
from paperfacts.records import FieldValue, LaneExtraction


def render_lane(lane: LaneExtraction) -> Iterator[str]:
    passes = f" passes={lane.passes}" if lane.passes != 1 else ""
    yield (
        f"[{lane.backend}] samples={len(lane.samples)} target_fields={len(lane.target.fields) if lane.target else 0} "
        f"invalid_source_ids={len(lane.invalid_source_ids)} dropped={len(lane.dropped)} "
        f"ungrounded={len(lane.ungrounded())} unattributed={len(lane.unattributed)}{passes} "
        f"tokens={lane.usage.get('total_tokens', '?')} model={lane.model} key={lane.extractor_key}"
    )
    for question in lane.failed_questions:
        yield f"    (no valid answer to the {question.field} question; asked again next run: {question.detail})"
    if lane.target:
        for field in lane.target.fields:
            yield f"    target.{_value(field)}  ← {', '.join(field.source_ids) or '(no source)'}"
    for sample in lane.samples:
        yield f"    {sample.sample_id}  ({sample.label})"
        for field in sample.fields:
            yield f"        {_value(field)}  ← {', '.join(field.source_ids) or '(no source)'}"
    if lane.unattributed:
        # Printed apart from the samples because nothing compares them: they are findings without a place.
        yield "    (unattributed)"
        for field in lane.unattributed:
            yield f"        ? {_value(field)}  ← {', '.join(field.source_ids) or '(no source)'}"


def render_report(report: ComparisonReport) -> Iterator[str]:
    yield f"counts: {report.counts.model_dump()}"
    for pair in report.matching.pairs:
        yield f"  match {pair.a_id} ↔ {pair.b_id}  conf={pair.confidence:.2f} ({pair.method}): {pair.justification}"
    for sample_id in report.matching.unmatched_a:
        yield f"  only in {report.backend_a}: {sample_id}"
    for sample_id in report.matching.unmatched_b:
        yield f"  only in {report.backend_b}: {sample_id}"
    if report.matching.failed:
        yield f"  sample matching FAILED: {report.matching.failure}"
    for comparison in report.comparisons:
        yield _comparison_line(comparison)


def _value(field: FieldValue) -> str:
    condition = f" @{field.condition}" if field.condition else ""
    normalized = f" = {field.value:g} {field.unit}" if field.value is not None else ""
    return f"{field.field}: {field.value_raw} {field.unit_raw or ''}{condition}{normalized}{_caveats(field)}".rstrip()


def _caveats(field: FieldValue) -> str:
    """Flag a value the reader should not take at face value.

    Both signals are cheap to compute and useless if nobody sees them: an ungrounded value could not be
    located in the block it cites, and an agreement below 1.0 means repeated extractions disagreed about it.
    """
    marks = []
    if not field.grounded:
        marks.append("UNGROUNDED")
    if field.agreement < 1.0:
        marks.append(f"agreement {field.agreement:.0%}")
    return f"  [{', '.join(marks)}]" if marks else ""


def _comparison_line(comparison: FieldComparison) -> str:
    a = _raw(comparison.a)
    b = _raw(comparison.b)
    condition = f" @{comparison.condition}" if comparison.condition else ""
    status = comparison.status if comparison.missing_in is None else f"missing({comparison.missing_in})"
    confidence = f" conf={comparison.match_confidence:.2f}" if comparison.match_confidence is not None else ""
    return (
        f"  {status:<24} {comparison.scope} {comparison.field}{condition}{confidence}: {a} | {b}   {comparison.detail}"
    )


def _raw(field: FieldValue | None) -> str:
    if field is None:
        return "—"
    return f"{field.value_raw} {field.unit_raw or ''}".strip()
