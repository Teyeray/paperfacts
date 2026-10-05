"""Pure, experimental C-lane adoption. Returns an audit; never patches an A/B dataset.

Only explicitly enabled scalar numeric sample fields participate. Existing populated cells,
uncertain identity/conditions, incomplete crops and contradictory C readings always abstain.
The receipt checks establish traceability, not the scientific truth of a model's attribution.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field

from paperfacts.compare import ComparisonReport
from paperfacts.dataset import DatasetPayload, _matching_blocked, _Scope, _scopes
from paperfacts.fields import FieldSpec
from paperfacts.keys import profile_comparison_fingerprint, profile_extraction_fingerprint
from paperfacts.kinds import _PARENTHESISED_UNCERTAINTY, CellValue, rules_for
from paperfacts.normalize import _QUALIFIERS, SCALAR, normalize_text, read_number, read_value, typeset
from paperfacts.profile import DomainProfile
from paperfacts.readers import _ONE_SIDED
from paperfacts.records import NO_CONTEXT, FieldValue, LaneExtraction, sample_key
from paperfacts.visual_evidence import CandidateReading, VisualEvidenceReport, VisualObservation, parse_response


class AdoptionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    revision: str = Field(min_length=1)
    fields: tuple[str, ...] = Field(min_length=1)
    ambiguous_match_confidence: float = Field(default=0.6, ge=0, le=1)


class CellAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    entity: str
    sample_id: str
    field: str
    before_status: str
    before: CellValue
    after: CellValue
    unit: str | None
    condition: str | None
    adopted: bool = False
    reason: str
    evidence: tuple[tuple[int, int], ...] = ()


class UnassignedObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    reading_index: int
    fact_index: int
    reason: str


class AdoptionAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    policy_revision: str
    cells: tuple[CellAudit, ...]
    unassigned: tuple[UnassignedObservation, ...]


def _receipt_problem(reading: CandidateReading, profile: DomainProfile) -> str | None:
    if "multipage_source" in reading.candidate.reasons or "incomplete_source" in reading.candidate.reasons:
        return "incomplete_context"
    if reading.status != "observed" or not reading.attempts:
        return "invalid_reading_receipt"
    attempt = reading.attempts[-1]
    crop = attempt.crop
    if attempt.error or not attempt.requested or crop is None or len(crop.image_sha256) != 64:
        return "invalid_reading_receipt"
    # A zoom can lose headers/footnotes. Initial adoption requires the complete selected region.
    if crop.page != reading.candidate.page or crop.bbox != reading.candidate.bbox:
        return "incomplete_context"
    try:
        response = parse_response(attempt.raw_response, profile)
    except ValueError:
        return "invalid_reading_receipt"
    if response.outcome != "observed" or response.observations != tuple(f.raw for f in reading.facts):
        return "invalid_reading_receipt"
    return None


def _number(raw: VisualObservation, spec: FieldSpec, profile: DomainProfile) -> tuple[float | None, str | None]:
    text = typeset(raw.value_raw)
    number = read_number(raw.value_raw, range_policy="reject")
    scalar = SCALAR.fullmatch(text)
    uncertain = (
        scalar is not None and scalar.group("uncertainty") is not None
    ) or _PARENTHESISED_UNCERTAINTY.fullmatch(text)
    if number.ends is not None or _QUALIFIERS.match(text) or _ONE_SIDED.match(text) or uncertain:
        return None, "non_exact_value"
    field = FieldValue(
        field=raw.field, value_raw=raw.value_raw, unit_raw=raw.unit_raw, condition=raw.condition, grounded=False
    )
    reading = read_value(field, spec, profile.units)
    if reading.clause or reading.condition:
        return None, "embedded_condition"
    value, _ = rules_for(spec).cell(field, replace(spec, range_policy="reject"), profile.units, NO_CONTEXT)
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        return None, "invalid_value"
    low, high = spec.valid_range
    if (low is not None and value < low) or (high is not None and value > high):
        return None, "out_of_range"
    return float(value), None


def _cell_decision(
    row: Mapping[str, CellValue],
    scope: _Scope,
    observations: Sequence[tuple[int, int]],
    readings: Sequence[CandidateReading],
    lanes: Sequence[LaneExtraction],
    spec: FieldSpec,
    profile: DomainProfile,
    policy: AdoptionPolicy,
) -> tuple[float | None, str | None, str]:
    if row["value"] is not None:
        return None, None, "existing_value"
    if _matching_blocked(scope, policy.ambiguous_match_confidence):
        return None, None, "sample_identity_blocked"
    if any(q.field == spec.name for lane in lanes for q in lane.failed_questions):
        return None, None, "unanswered"
    if row["decision"] not in ("missing", "conflict"):
        return None, None, "baseline_status_not_enabled"
    if not observations:
        return None, None, "no_visual_evidence"
    baseline_conditions = {
        normalize_text(f.condition or "")
        for sample in (scope.a, scope.b)
        if sample is not None
        for f in sample.fields
        if f.field == spec.name
    }
    if len(baseline_conditions) > 1:
        return None, None, "condition_mismatch"
    numbers = []
    conditions = []
    for reading_index, fact_index in observations:
        reading = readings[reading_index]
        problem = _receipt_problem(reading, profile)
        if problem:
            return None, None, problem
        raw = reading.facts[fact_index].raw
        if raw.evidence_type != "printed":
            return None, None, "not_printed"
        if raw.condition_status == "unclear" or (
            raw.condition_status == "not_stated" and (spec.condition_rule or spec.condition_hint)
        ):
            return None, None, "unclear_condition"
        condition = normalize_text(raw.condition or "")
        if baseline_conditions and baseline_conditions != {condition}:
            return None, None, "condition_mismatch"
        value, problem = _number(raw, spec, profile)
        if problem:
            return None, None, problem
        numbers.append(value)
        conditions.append(condition)
    if len(set(conditions)) != 1 or any(not rules_for(spec).same(numbers[0], number, spec) for number in numbers[1:]):
        return None, None, "visual_disagreement"
    raw = readings[observations[0][0]].facts[observations[0][1]].raw
    return numbers[0], raw.condition, "adopted_from_visual"


def check_policy(profile: DomainProfile, policy: AdoptionPolicy) -> None:
    """Reject unsupported policy fields before any experiment output is produced."""
    if len(set(policy.fields)) != len(policy.fields):
        raise ValueError("duplicate policy field")
    for name in policy.fields:
        spec = profile.by_name.get(name)
        if spec is None or spec.kind != "numeric" or spec.cardinality != "one" or not spec.is_sample_level:
            raise ValueError(f"policy needs a scalar numeric sample field: {name}")


def replay_document(
    *,
    dataset: DatasetPayload,
    lanes: Sequence[LaneExtraction],
    comparison: ComparisonReport,
    evidence: VisualEvidenceReport,
    profile: DomainProfile,
    policy: AdoptionPolicy,
    pdf_sha256: str,
) -> AdoptionAudit:
    """Evaluate frozen evidence against existing cells; no I/O, clients, gold or state mutation.

    Hash/identity mismatches are bad replay inputs, not a reason to silently change the baseline.
    Valid but unusable facts stay in the report and receive an explicit refusal in this audit.
    """
    check_policy(profile, policy)
    by_backend = {lane.backend: lane for lane in lanes}
    if len(lanes) != 2 or set(by_backend) != {comparison.backend_a, comparison.backend_b}:
        raise ValueError("baseline lane identity mismatch")
    expected = {lane.backend: hashlib.sha256(lane.model_dump_json().encode()).hexdigest() for lane in lanes}
    if (
        evidence.document_id != dataset.document_id
        or comparison.document_id != dataset.document_id
        or any(lane.document_id != dataset.document_id for lane in lanes)
        or evidence.pdf_sha256 != pdf_sha256
        or evidence.profile_hash != profile.content_hash
        or evidence.baseline_sha256 != expected
        or dataset.extractor_key != comparison.extractor_key
        or dataset.comparison_key != comparison.comparison_key
        or any(lane.extractor_key != dataset.extractor_key for lane in lanes)
        or dataset.profile_fingerprint != profile_comparison_fingerprint(profile)
        or comparison.profile_fingerprint != dataset.profile_fingerprint
        or any(lane.profile_fingerprint != profile_extraction_fingerprint(profile) for lane in lanes)
        or set(comparison.matchings) != {entity.name for entity in profile.entities}
    ):
        raise ValueError("frozen baseline/report identity mismatch")
    artifacts = {lane.backend: lane.artifact_sha256 for lane in lanes}
    if dataset.artifact_sha256 != artifacts or artifacts != {
        comparison.backend_a: comparison.artifact_sha256_a,
        comparison.backend_b: comparison.artifact_sha256_b,
    }:
        raise ValueError("frozen artifact identity mismatch")
    scopes = [scope for entity in profile.entities for scope in _scopes(by_backend, comparison, entity.name)]
    scope_by_key = {(s.entity, s.sample_id): s for s in scopes}
    if len(scope_by_key) != len(scopes):
        raise ValueError("duplicate baseline sample scope")
    sample_rows = {(str(r.get("entity", profile.primary.name)), str(r["sample_id"])): r for r in dataset.sample_rows}
    if len(sample_rows) != len(dataset.sample_rows) or set(sample_rows) != set(scope_by_key):
        raise ValueError("baseline sample rows differ from frozen scopes")
    assigned: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    unassigned = []
    for ri, reading in enumerate(evidence.readings):
        for fi, bound in enumerate(reading.facts):
            raw = bound.raw
            reason = "field_not_enabled"
            if raw.field in policy.fields:
                spec = profile.by_name[raw.field]
                reason = "unknown_or_ambiguous_sample"
                if raw.scope == "sample" and raw.entity == (spec.entity or profile.primary.name):
                    key = sample_key(raw.sample_raw)
                    # Resolve actual lane records, not the cached C mapping or a parsed display ID.
                    found = [
                        (lane.backend, sample.sample_id)
                        for lane in lanes
                        for sample in lane.samples
                        if sample.entity == raw.entity and sample_key(sample.sample_id) == key
                    ]
                    unique = all(sum(backend == lane.backend for backend, _ in found) <= 1 for lane in lanes)
                    targets = []
                    for s in scopes:
                        members = {(comparison.backend_a, s.a.sample_id)} if s.a else set()
                        if s.b:
                            members.add((comparison.backend_b, s.b.sample_id))
                        if s.entity == raw.entity and found and set(found) <= members:
                            targets.append(s)
                    if unique and len(targets) == 1:
                        target = targets[0]
                        assigned.setdefault((target.entity, target.sample_id, raw.field), []).append((ri, fi))
                        continue
            unassigned.append(UnassignedObservation(reading_index=ri, fact_index=fi, reason=reason))
    cells = []
    quality = {}
    for row in dataset.quality_rows:
        if row.get("field") not in policy.fields:
            continue
        key = (str(row.get("entity", profile.primary.name)), str(row["sample_id"]), str(row["field"]))
        if key in quality:
            raise ValueError("duplicate baseline quality cell")
        quality[key] = row
    for scope in scopes:
        for field in policy.fields:
            spec = profile.by_name[field]
            if (spec.entity or profile.primary.name) != scope.entity:
                continue
            key = (scope.entity, scope.sample_id, field)
            row = quality.get(key)
            if row is None or field not in sample_rows[key[:2]] or row.get("value") != sample_rows[key[:2]][field]:
                raise ValueError("baseline quality and sample cell mismatch")
            pointers = assigned.get(key, [])
            value, condition, reason = _cell_decision(
                row, scope, pointers, evidence.readings, lanes, spec, profile, policy
            )
            adopted = reason == "adopted_from_visual"
            cells.append(
                CellAudit(
                    document_id=dataset.document_id,
                    entity=scope.entity,
                    sample_id=scope.sample_id,
                    field=field,
                    before_status=str(row["decision"]),
                    before=row["value"],
                    after=value if adopted else row["value"],
                    unit=spec.canonical_unit,
                    condition=condition if adopted else row.get("conditions") or None,
                    adopted=adopted,
                    reason=reason,
                    evidence=tuple(pointers),
                )
            )
    if len(quality) != len(cells):
        raise ValueError("quality cells do not belong to known sample scopes")
    return AdoptionAudit(policy_revision=policy.revision, cells=tuple(cells), unassigned=tuple(unassigned))
