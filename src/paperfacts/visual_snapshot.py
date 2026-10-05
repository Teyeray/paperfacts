"""Independent experimental dataset copies, always rebuilt from the original A/B evidence.

This is not a workflow stage. The wrapper keeps counterfactual provenance separate from the
baseline dataset contract; existing consolidation and workbook code remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from paperfacts.compare import ComparisonReport
from paperfacts.dataset import DatasetPayload, _scopes, consolidate_document
from paperfacts.keys import ComparisonOptions
from paperfacts.kinds import joined
from paperfacts.models import DocumentInput
from paperfacts.profile import DomainProfile
from paperfacts.records import LaneExtraction
from paperfacts.storage import write_bytes_if_absent
from paperfacts.visual_adoption import AdoptionAudit, AdoptionPolicy, check_policy, replay_document
from paperfacts.visual_evidence import VisualEvidenceReport


def _digest(value: BaseModel) -> str:
    return hashlib.sha256(value.model_dump_json().encode()).hexdigest()


class ExperimentSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal[1] = 1
    scope: Literal["experimental-copy"] = "experimental-copy"
    scientific_acceptance: Literal["not_established"] = "not_established"
    fingerprint: str
    baseline_sha256: str
    evidence_sha256: str | None
    policy: AdoptionPolicy
    report_status: Literal["current", "missing", "stale"]
    reason: str
    audit: AdoptionAudit | None
    dataset: DatasetPayload
    provenance: dict[str, Any] = Field(default_factory=dict)

    def write(self, path: Path) -> None:
        if not write_bytes_if_absent(path, self.model_dump_json(indent=2).encode()):
            raise FileExistsError("experimental snapshot already exists; choose a new export directory")


def _apply(
    baseline: DatasetPayload,
    audit: AdoptionAudit,
    evidence: VisualEvidenceReport,
    lanes: Sequence[LaneExtraction],
    comparison: ComparisonReport,
    profile: DomainProfile,
    fingerprint: str,
) -> DatasetPayload:
    # Copies only. A snapshot is never accepted as the baseline of another adoption replay.
    quality = [dict(row) for row in baseline.quality_rows]
    by_cell = {(c.entity, c.sample_id, c.field): c for c in audit.cells}
    for row in quality:
        cell = by_cell.get((row.get("entity", profile.primary.name), row.get("sample_id"), row.get("field")))
        if cell is None:
            continue
        row["detail"] = joined([str(row.get("detail") or ""), f"visual: {cell.reason}"])
        if cell.adopted:
            pointers = [
                f"visual:{evidence.readings[ri].attempts[-1].crop.image_sha256}:r{ri}:f{fi}" for ri, fi in cell.evidence
            ]
            row.update(
                value=cell.after,
                decision="adopted_from_visual",
                conditions=cell.condition or "",
                source_ids=joined([str(row.get("source_ids") or ""), *pointers]),
                lanes=joined([str(row.get("lanes") or ""), "visual"]),
                series=False,
            )
    by_quality = {(r.get("entity", profile.primary.name), r.get("sample_id"), r.get("field")): r for r in quality}
    scopes = {
        (s.entity, s.sample_id): s
        for entity in profile.entities
        for s in _scopes({lane.backend: lane for lane in lanes}, comparison, entity.name)
    }
    sample_rows = []
    for original in baseline.sample_rows:
        row = dict(original)
        entity = next(entity for entity in profile.entities if entity.name == row.get("entity", profile.primary.name))
        key = (entity.name, row["sample_id"])
        fields = (*profile.paper_fields, *profile.entity_fields(entity))
        rows = [
            by_quality[(*key, spec.name)]
            if spec.is_sample_level
            else by_quality[(profile.primary.name, "paper", spec.name)]
            for spec in fields
        ]
        for value in rows:
            row[str(value["field"])] = value["value"]
        row["available_fields"] = sum(value["value"] is not None for value in rows)
        row["agree_fields"] = sum(value["decision"] == "agree" for value in rows)
        scope = scopes[key]
        row["conditions"] = joined(
            [
                f"{name}={value}"
                for sample in (scope.a, scope.b)
                if sample is not None
                for name, value in sorted(sample.conditions.items())
            ]
            + [
                f"{spec.name}: {by_quality[(*key, spec.name)]['conditions']}"
                for spec in profile.entity_fields(entity)
                if by_quality[(*key, spec.name)].get("conditions")
            ]
        )
        sample_rows.append(row)
    primary = [row for row in sample_rows if row.get("entity", profile.primary.name) == profile.primary.name]
    # Keep dataset.py's whole-row selection order; never combine fields across samples.
    paper = (
        min(
            primary, key=lambda r: (-int(r["available_fields"] or 0), -int(r["agree_fields"] or 0), str(r["sample_id"]))
        )
        if primary
        else dict(baseline.paper_row)
    )
    for row in quality:
        if row.get("field") == "__selection__":
            row["sample_id"] = paper["sample_id"]
    return baseline.model_copy(
        update={
            "sample_rows": tuple(sample_rows),
            "paper_row": dict(paper),
            "quality_rows": tuple(quality),
            "comparison_key": f"visual-experiment-{fingerprint[:16]}",
        }
    )


def build_snapshot(
    *,
    document: DocumentInput,
    lanes: Sequence[LaneExtraction],
    comparison: ComparisonReport,
    profile: DomainProfile,
    policy: AdoptionPolicy,
    pdf_sha256: str,
    evidence: VisualEvidenceReport | None,
    expected_model: str,
    expected_prompt_sha256: str,
    expected_strategy: str,
) -> ExperimentSnapshot:
    """No I/O or clients: rebuild from A/B and apply only a matching current report."""
    check_policy(profile, policy)
    baseline = consolidate_document(
        document,
        {lane.backend: lane for lane in lanes},
        comparison,
        ComparisonOptions(profile=profile, ambiguous_match_confidence=policy.ambiguous_match_confidence),
    ).to_payload()
    baseline_sha = _digest(baseline)
    evidence_sha = _digest(evidence) if evidence else None
    material = dict(
        baseline=baseline_sha,
        lanes=[_digest(lane) for lane in lanes],
        comparison=_digest(comparison),
        profile=profile.content_hash,
        policy=_digest(policy),
        pdf=pdf_sha256,
        evidence=evidence_sha,
        model=expected_model,
        prompt=expected_prompt_sha256,
        strategy=expected_strategy,
    )
    fingerprint = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    audit = None
    status, reason = "missing", "no matching saved report"
    if evidence is not None:
        status, reason = "stale", "reader model, prompt or strategy changed"
        if (
            evidence.model == expected_model
            and evidence.prompt_sha256 == expected_prompt_sha256
            and evidence.strategy == expected_strategy
        ):
            try:
                audit = replay_document(
                    dataset=baseline,
                    lanes=lanes,
                    comparison=comparison,
                    evidence=evidence,
                    profile=profile,
                    policy=policy,
                    pdf_sha256=pdf_sha256,
                )
            except ValueError as exc:
                reason = str(exc)
            else:
                status, reason = "current", "recomputed from frozen A/B and matching C"
    derived = (
        _apply(baseline, audit, evidence, lanes, comparison, profile, fingerprint) if audit is not None else baseline
    )
    return ExperimentSnapshot(
        fingerprint=fingerprint,
        baseline_sha256=baseline_sha,
        evidence_sha256=evidence_sha,
        policy=policy,
        report_status=status,
        reason=reason,
        audit=audit,
        dataset=derived,
    )
