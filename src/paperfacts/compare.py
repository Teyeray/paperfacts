"""Field-level comparison (the rule layer): the deterministic parts happen here, the fuzzy judgment calls
are left to the model layer.

The unit of comparison is "one value of one field": first pair exactly by normalized measurement condition
(550 nm and a full-spectrum average are two different facts), then pair whatever numeric values are left by
numeric proximity — so when the two lanes phrase the same fact's condition differently ("ellipsometric" vs.
"from ellipsometric"), it doesn't get split into two records that each look like they're missing the
other's value. Numeric comparison uses ``math.isclose`` (|a-b| <= max(rel_tol*max(|a|,|b|), abs_tol)), with
per-field tolerances configured in :mod:`paperfacts.fields`; text/composition comparison uses
equality of the normalized key.

``status`` is always the **actual** comparison outcome; sample-pairing confidence is recorded separately in
``match_confidence``. Whether a low-confidence pairing should be escalated for review is a decision left to
downstream consumers — the raw observation is never discarded.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from paperfacts.fields import AMBIGUOUS_MATCH_CONFIDENCE, SAMPLE_FIELDS, TARGET_FIELDS, FieldSpec
from paperfacts.keys import comparison_key
from paperfacts.matching import SampleMatching
from paperfacts.models import Backend
from paperfacts.normalize import clean_unit, normalize_key, normalize_lane
from paperfacts.records import FieldValue, LaneExtraction

FactStatus = Literal["agree", "conflict", "ambiguous", "missing"]


class FieldComparison(BaseModel):
    """The comparison outcome for one fact across both lanes, keeping both sides' candidate values and provenance."""

    model_config = ConfigDict(frozen=True)

    scope: str = Field(description='"target", "sample:<a_id>|<b_id>" (matched), or "sample:<id>" (unmatched)')
    field: str
    condition: str | None = None
    status: FactStatus
    missing_in: Backend | None = Field(
        default=None, description='which lane is missing the value when status == "missing"; None otherwise'
    )
    match_confidence: float | None = Field(
        default=None, description="confidence of the LLM sample pairing; None for paper-level or exact-matched facts"
    )
    a: FieldValue | None = None
    b: FieldValue | None = None
    detail: str = ""


class ComparisonCounts(BaseModel):
    """The evaluation counts the PRD asks for.

    Every count is present even at zero, so a consumer never guards against a missing key. The two per-lane
    dictionaries are the exception: a lane with nothing to report is simply absent from them.
    """

    model_config = ConfigDict(frozen=True)

    agree: int = 0
    conflict: int = 0
    ambiguous: int = 0
    missing: int = 0
    total: int = 0
    missing_by_backend: dict[str, int] = Field(default_factory=dict)
    samples_matched: int = 0
    samples_unmatched: int = 0
    low_confidence_matches: int = 0
    matching_failed: bool = False
    unattributed_by_backend: dict[str, int] = Field(
        default_factory=dict,
        description="per lane, values extracted but not placed on a sample; they take part in no comparison",
    )


class ComparisonReport(BaseModel):
    """The full report of aligning and comparing both lanes for one document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    extractor_key: str
    comparison_key: str
    backend_a: Backend
    backend_b: Backend
    matching: SampleMatching
    comparisons: tuple[FieldComparison, ...] = ()
    counts: ComparisonCounts = Field(default_factory=ComparisonCounts)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def compare_lanes(lane_a: LaneExtraction, lane_b: LaneExtraction, matching: SampleMatching) -> ComparisonReport:
    if lane_a.extractor_key != lane_b.extractor_key:
        raise ValueError(
            f"lanes have different extractor_key ({lane_a.extractor_key} vs {lane_b.extractor_key}); cannot compare"
        )
    # Normalization is a pure, idempotent function, so it is unconditionally redone here: callers never have
    # to remember to normalize first, which rules out "forgot to normalize, so the answer was silently wrong"
    lane_a, lane_b = normalize_lane(lane_a), normalize_lane(lane_b)
    a_name, b_name = lane_a.backend, lane_b.backend
    comparisons: list[FieldComparison] = []

    # Target: paper-level, does not go through sample pairing
    comparisons += _compare_records(
        "target",
        lane_a.target.fields if lane_a.target else (),
        lane_b.target.fields if lane_b.target else (),
        TARGET_FIELDS,
        a_name,
        b_name,
    )
    # Matched samples: status is the real comparison outcome; pairing confidence is recorded separately
    for pair in matching.pairs:
        sample_a, sample_b = lane_a.sample(pair.a_id), lane_b.sample(pair.b_id)
        if sample_a is None or sample_b is None:
            continue
        comparisons += _compare_records(
            f"sample:{pair.a_id}|{pair.b_id}",
            sample_a.fields,
            sample_b.fields,
            SAMPLE_FIELDS,
            a_name,
            b_name,
            match_confidence=pair.confidence if pair.method == "llm" else None,
        )
    # Unmatched samples: normally this just means the other lane doesn't have it; when the matching model
    # itself failed we can't tell whether it's really missing, so mark it ambiguous and send it for review
    one_sided: FactStatus = "ambiguous" if matching.failed else "missing"
    reason = f"sample matching failed: {matching.failure}" if matching.failed else None
    for sample_id in matching.unmatched_a:
        if (sample := lane_a.sample(sample_id)) is not None:
            comparisons += _compare_records(
                f"sample:{sample_id}",
                sample.fields,
                (),
                SAMPLE_FIELDS,
                a_name,
                b_name,
                one_sided=one_sided,
                one_sided_detail=reason,
            )
    for sample_id in matching.unmatched_b:
        if (sample := lane_b.sample(sample_id)) is not None:
            comparisons += _compare_records(
                f"sample:{sample_id}",
                (),
                sample.fields,
                SAMPLE_FIELDS,
                a_name,
                b_name,
                one_sided=one_sided,
                one_sided_detail=reason,
            )

    return ComparisonReport(
        document_id=lane_a.document_id,
        extractor_key=lane_a.extractor_key,
        comparison_key=comparison_key(),
        backend_a=a_name,
        backend_b=b_name,
        matching=matching,
        comparisons=tuple(comparisons),
        counts=_count(comparisons, matching, (lane_a, lane_b)),
    )


def compare_values(a: FieldValue, b: FieldValue, spec: FieldSpec) -> tuple[FactStatus, str]:
    """Decide the outcome when both sides have a value."""
    if spec.kind != "numeric":
        if normalize_key(a.value_raw) == normalize_key(b.value_raw):
            return "agree", "identical after text normalization"
        return "conflict", f"{a.value_raw!r} vs {b.value_raw!r}"

    if a.value is None or b.value is None:
        # At least one side failed to parse as a number: if the raw text (including unit, case-sensitive)
        # is identical on both sides, that still counts as agreement; otherwise there's no way to judge
        if normalize_key(a.value_raw) == normalize_key(b.value_raw) and _unit_key(a.unit_raw) == _unit_key(b.unit_raw):
            return "agree", "identical raw text (not parsed as a number)"
        return (
            "ambiguous",
            f"unparsed: {a.normalization_note or a.value_raw!r} vs {b.normalization_note or b.value_raw!r}",
        )
    if a.unit != b.unit:
        return "ambiguous", f"units differ after normalization: {a.unit} vs {b.unit}"
    if math.isclose(a.value, b.value, rel_tol=spec.rel_tol, abs_tol=spec.abs_tol):
        return "agree", f"{a.value:g} ≈ {b.value:g} {a.unit or ''} (rel_tol={spec.rel_tol:g}, abs_tol={spec.abs_tol:g})"
    return "conflict", f"{a.value:g} vs {b.value:g} {a.unit or ''}"


def _compare_records(
    scope: str,
    fields_a: Sequence[FieldValue],
    fields_b: Sequence[FieldValue],
    specs: Sequence[FieldSpec],
    backend_a: Backend,
    backend_b: Backend,
    *,
    match_confidence: float | None = None,
    one_sided: FactStatus = "missing",
    one_sided_detail: str | None = None,
) -> list[FieldComparison]:
    """Pair up both sides' values field by field and compare them. When ``fields_b`` is empty this
    naturally degenerates to "everything is only in lane a"."""
    spec_by_name = {spec.name: spec for spec in specs}
    names = sorted({f.field for f in (*fields_a, *fields_b)} & spec_by_name.keys())
    out: list[FieldComparison] = []
    for name in names:
        spec = spec_by_name[name]
        values_a = [f for f in fields_a if f.field == name]
        values_b = [f for f in fields_b if f.field == name]
        for a, b in _pair_values(values_a, values_b, spec):
            if a is None or b is None:
                present = a if a is not None else b
                assert present is not None
                missing_in = backend_a if a is None else backend_b
                out.append(
                    FieldComparison(
                        scope=scope,
                        field=name,
                        condition=present.condition,
                        status=one_sided,
                        missing_in=missing_in if one_sided == "missing" else None,
                        match_confidence=match_confidence,
                        a=a,
                        b=b,
                        detail=one_sided_detail or f"only in {backend_b if a is None else backend_a}",
                    )
                )
                continue
            status, detail = compare_values(a, b, spec)
            if normalize_key(a.condition) != normalize_key(b.condition):
                # A pair matched by numeric proximity, with condition wording that differs: if the values
                # still agree, call it agreement (with a note); if not, there's no way to tell whether
                # that's a real conflict or just two different conditions
                detail = f"condition texts differ ({a.condition!r} vs {b.condition!r}); {detail}"
                if status == "conflict":
                    status = "ambiguous"
            out.append(
                FieldComparison(
                    scope=scope,
                    field=name,
                    condition=a.condition,
                    status=status,
                    match_confidence=match_confidence,
                    a=a,
                    b=b,
                    detail=detail,
                )
            )
    return out


def _pair_values(
    values_a: Sequence[FieldValue], values_b: Sequence[FieldValue], spec: FieldSpec
) -> list[tuple[FieldValue | None, FieldValue | None]]:
    """Pair up both sides' values for one field, so that every value is accounted for.

    Values under the same measurement condition are paired first, by numeric proximity within that
    condition; whatever is left over can still pair across differently worded conditions ("ellipsometric"
    against "from ellipsometric"), again by proximity. Anything still unpaired is reported one-sided.

    Every value gets a row. An earlier version indexed each side by condition and kept the first value per
    condition, which silently discarded a second reading of the same quantity -- exactly the case worth
    reporting, since a lane that reads a resistivity as both ``10^-2`` and ``10^2`` has an OCR problem the
    other lane may not share.
    """
    groups_a, groups_b = _by_condition(values_a), _by_condition(values_b)
    pairs: list[tuple[FieldValue | None, FieldValue | None]] = []
    rest_a: list[FieldValue] = []
    rest_b: list[FieldValue] = []
    for key in sorted(groups_a.keys() | groups_b.keys()):
        group_a, group_b = list(groups_a.get(key, ())), list(groups_b.get(key, ()))
        if spec.kind == "numeric":
            pairs += _closest_pairs(group_a, group_b)
        # Same condition, so whatever remains still describes the same fact: pair it in the order the paper
        # gave, rather than leaving both sides looking like the other is missing a value.
        while group_a and group_b:
            pairs.append((group_a.pop(0), group_b.pop(0)))
        rest_a += group_a
        rest_b += group_b
    if spec.kind == "numeric":
        pairs += _closest_pairs(rest_a, rest_b)
    pairs += [(a, None) for a in rest_a]
    pairs += [(None, b) for b in rest_b]
    return pairs


def _closest_pairs(
    rest_a: list[FieldValue], rest_b: list[FieldValue]
) -> list[tuple[FieldValue | None, FieldValue | None]]:
    """Greedily pair the two closest numbers, removing them from both lists as it goes.

    Stops as soon as a side runs out or nothing left parsed as a number, leaving those for the caller to
    report one-sided.
    """
    pairs: list[tuple[FieldValue | None, FieldValue | None]] = []
    while rest_a and rest_b:
        candidates = [
            (abs(a.value - b.value), i, j)
            for i, a in enumerate(rest_a)
            for j, b in enumerate(rest_b)
            if a.value is not None and b.value is not None
        ]
        if not candidates:
            break
        _, i, j = min(candidates)
        pairs.append((rest_a.pop(i), rest_b.pop(j)))
    return pairs


def _by_condition(values: Sequence[FieldValue]) -> dict[str, list[FieldValue]]:
    """Group values by normalized condition, keeping every one of them in the order they were extracted."""
    grouped: dict[str, list[FieldValue]] = {}
    for value in values:
        grouped.setdefault(normalize_key(value.condition), []).append(value)
    return grouped


def _unit_key(unit_raw: str | None) -> str:
    return clean_unit(unit_raw) if unit_raw else ""


def _count(
    comparisons: Sequence[FieldComparison], matching: SampleMatching, lanes: Sequence[LaneExtraction]
) -> ComparisonCounts:
    tally = Counter(c.status for c in comparisons)
    missing_by_backend = Counter(c.missing_in for c in comparisons if c.missing_in is not None)
    return ComparisonCounts(
        agree=tally["agree"],
        conflict=tally["conflict"],
        ambiguous=tally["ambiguous"],
        missing=tally["missing"],
        total=len(comparisons),
        missing_by_backend={str(k): v for k, v in sorted(missing_by_backend.items())},
        samples_matched=len(matching.pairs),
        samples_unmatched=len(matching.unmatched_a) + len(matching.unmatched_b),
        low_confidence_matches=sum(1 for p in matching.pairs if p.confidence < AMBIGUOUS_MATCH_CONFIDENCE),
        matching_failed=matching.failed,
        unattributed_by_backend={lane.backend: len(lane.unattributed) for lane in lanes if lane.unattributed},
    )
