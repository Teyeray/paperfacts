"""Offline G2 counterfactual replay. Frozen inputs in, audit JSON on stdout; no model calls.

Gold labels judge complete facts (including measurement state), never steer adoption. A label
must explicitly review every adopted observation. Synthetic scores do not unlock production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paperfacts.compare import ComparisonReport
from paperfacts.dataset import DatasetPayload
from paperfacts.normalize import normalize_text
from paperfacts.pdf import read_geometry
from paperfacts.profile_loader import parse_profile
from paperfacts.records import LaneExtraction
from paperfacts.visual_adoption import AdoptionPolicy, CellAudit, replay_document
from paperfacts.visual_evidence import VisualEvidenceReport


class ReplayModel(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", str_strip_whitespace=True, allow_inf_nan=False, hide_input_in_errors=True
    )


class FrozenRef(ReplayModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReplayCase(ReplayModel):
    pdf: FrozenRef
    dataset: FrozenRef
    comparison: FrozenRef
    lane_a: FrozenRef
    lane_b: FrozenRef
    evidence: FrozenRef


class ObservationReview(ReplayModel):
    reading_index: int = Field(ge=0, strict=True)
    fact_index: int = Field(ge=0, strict=True)
    verdict: Literal["correct", "wrong", "uncertain"]


class GoldCell(ReplayModel):
    document_id: str = Field(min_length=1)
    entity: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    value: float | None
    unit: str | None
    condition: str | None
    condition_status: Literal["stated", "not_stated", "unclear"]
    measurement_state: str = Field(
        min_length=1, description="Human-reviewed state; observation verdicts must check it, not just the number"
    )
    before_correct: bool | None = Field(
        description="Human full-tuple review of the frozen baseline; null means unreviewed"
    )
    observations: tuple[ObservationReview, ...] = ()

    @model_validator(mode="after")
    def consistent_condition(self) -> Self:
        if self.condition_status == "stated" and not self.condition:
            raise ValueError("stated condition needs text")
        if self.condition_status == "not_stated" and self.condition is not None:
            raise ValueError("not_stated condition must be null")
        return self


class ReplayInput(ReplayModel):
    format: Literal[1]
    evaluation_set: Literal["synthetic", "development", "c_holdout"] = "synthetic"
    gold_revision: str = Field(min_length=1)
    profile: FrozenRef
    policy: AdoptionPolicy
    cases: tuple[ReplayCase, ...] = Field(min_length=1)
    gold: tuple[GoldCell, ...] = ()


def _key(cell: CellAudit | GoldCell) -> tuple[str, str, str, str]:
    return cell.document_id, cell.entity, cell.sample_id, cell.field


def _counts() -> dict:
    return dict(
        cells=0,
        opportunities=0,
        triggered=0,
        adopted=0,
        improved=0,
        incorrect_adoptions=0,
        correct_value_regressions=0,
        pending_review=0,
        documents=set(),
    )


def _metric(counts: dict) -> str:
    if counts["incorrect_adoptions"] or counts["correct_value_regressions"]:
        return "fail"
    if counts["pending_review"]:
        return "needs_review"
    if not counts["opportunities"] or not counts["adopted"]:
        return "insufficient"
    return "pass" if counts["improved"] > 0 else "fail"


def _finalize(counts: dict) -> dict:
    return {
        **counts,
        "documents": len(counts["documents"]),
        "net": counts["improved"] - counts["incorrect_adoptions"],
        "metric_result": _metric(counts),
    }


def replay(path: Path) -> dict:
    path = path.resolve()
    raw = path.read_bytes()
    config = ReplayInput.model_validate_json(raw)
    contents = {}
    hashes = {}

    def resolve(ref: FrozenRef) -> Path:
        return (path.parent / ref.path).resolve()

    refs = [config.profile]
    for case in config.cases:
        refs.extend(getattr(case, name) for name in ReplayCase.model_fields)
    # Check ALL byte references before making a decision. Never refresh or fetch an input.
    for ref in refs:
        target = resolve(ref)
        data = contents.setdefault(target, target.read_bytes())
        if hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ValueError(f"SHA mismatch for frozen input: {target}")
        hashes[str(target)] = ref.sha256

    def load(model: type[BaseModel], ref: FrozenRef) -> BaseModel:
        return model.model_validate_json(contents[resolve(ref)])

    profile = parse_profile(json.loads(contents[resolve(config.profile)]), resolve(config.profile))
    audits = []
    documents = set()
    for case in config.cases:
        dataset = load(DatasetPayload, case.dataset)
        if dataset.document_id in documents:
            raise ValueError("duplicate document in replay")
        documents.add(dataset.document_id)
        evidence = load(VisualEvidenceReport, case.evidence)
        pages = read_geometry(resolve(case.pdf)).page_count
        if any(reading.candidate.page >= pages for reading in evidence.readings):
            raise ValueError("visual reading page is outside original PDF")
        audit = replay_document(
            dataset=dataset,
            lanes=(load(LaneExtraction, case.lane_a), load(LaneExtraction, case.lane_b)),
            comparison=load(ComparisonReport, case.comparison),
            evidence=evidence,
            profile=profile,
            policy=config.policy,
            pdf_sha256=case.pdf.sha256,
        )
        audits.append(audit)
    cells = {_key(cell): cell for audit in audits for cell in audit.cells}
    gold = {}
    for label in config.gold:
        key = _key(label)
        if key in gold or key not in cells:
            raise ValueError("duplicate or unknown gold cell")
        cell = cells[key]
        pointers = [(r.reading_index, r.fact_index) for r in label.observations]
        if len(pointers) != len(set(pointers)) or not set(pointers) <= set(cell.evidence):
            raise ValueError("duplicate or misattributed observation review")
        if label.before_correct and (
            cell.before is None or label.value is None or cell.before != label.value or cell.unit != label.unit
        ):
            raise ValueError("before_correct contradicts frozen baseline value/unit")
        gold[key] = label
    goals = {name: _counts() for name in ("correction", "shared_missing")}
    by_rule = {(field, status): _counts() for field in config.policy.fields for status in ("missing", "conflict")}
    for key, cell in cells.items():
        goal = "shared_missing" if cell.before_status == "missing" else "correction"
        counters = [goals[goal]]
        if (cell.field, cell.before_status) in by_rule:
            counters.append(by_rule[(cell.field, cell.before_status)])
        label = gold.get(key)
        pending = label is None or label.before_correct is None or label.condition_status == "unclear"
        after_correct = False
        known_wrong = False
        if label is not None and cell.adopted:
            reviews = {(r.reading_index, r.fact_index): r.verdict for r in label.observations}
            verdicts = [reviews.get(pointer, "uncertain") for pointer in cell.evidence]
            pending = pending or "uncertain" in verdicts
            tuple_matches = (
                label.value is not None
                and cell.after == label.value
                and cell.unit == label.unit
                and normalize_text(cell.condition or "") == normalize_text(label.condition or "")
            )
            known_wrong = not tuple_matches or "wrong" in verdicts
            after_correct = tuple_matches and bool(verdicts) and all(v == "correct" for v in verdicts)
        for counts in counters:
            counts["cells"] += 1
            counts["documents"].add(cell.document_id)
            counts["triggered"] += bool(cell.evidence)
            counts["adopted"] += cell.adopted
            counts["pending_review"] += pending
            if label is not None and label.before_correct is False and label.value is not None:
                counts["opportunities"] += 1
            counts["incorrect_adoptions"] += known_wrong
            counts["correct_value_regressions"] += known_wrong and label is not None and bool(label.before_correct)
            if not pending:
                counts["improved"] += cell.adopted and after_correct and not label.before_correct
    goals = {name: _finalize(counts) for name, counts in goals.items()}
    rules = [
        dict(field=field, baseline_status=status, evidence_type="printed", **_finalize(counts))
        for (field, status), counts in by_rule.items()
    ]
    outcomes = {item["metric_result"] for item in goals.values()} | {item["metric_result"] for item in rules}
    result = next((status for status in ("fail", "needs_review", "insufficient") if status in outcomes), "pass")
    root = Path(__file__).resolve().parents[1]
    code = hashlib.sha256()
    for source in [*sorted((root / "src/paperfacts").glob("*.py")), Path(__file__).resolve()]:
        code.update(str(source.relative_to(root)).encode() + b"\0" + source.read_bytes() + b"\0")
    return dict(
        format=1,
        scope="counterfactual-reviewed-scope",
        scientific_acceptance="not_established",
        evaluation_set=config.evaluation_set,
        gold_revision=config.gold_revision,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        code_sha256=code.hexdigest(),
        frozen_files=hashes,
        policy=config.policy.model_dump(mode="json"),
        metric_result=result,
        goals=goals,
        rules=rules,
        audits=[audit.model_dump(mode="json") for audit in audits],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--replay", type=Path)
    choice.add_argument("--schema", action="store_true")
    args = parser.parse_args()
    if args.schema:
        print(json.dumps(ReplayInput.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    try:
        result = replay(args.replay)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
