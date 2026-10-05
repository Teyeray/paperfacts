"""Score human-reviewed complete facts in frozen reports, never infer truth or sample alignment.

The review JSON owns canonical identities, baseline classes and observation judgments. This script
checks their provenance and counts them in the reviewed scope; it establishes neither holdout
eligibility nor scientific acceptance. Run with ``--review path/to/review.json``; JSON goes to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pypdfium2 import PdfiumError

from paperfacts.models import NormalizedBBox
from paperfacts.pdf import read_geometry
from paperfacts.records import LaneExtraction
from paperfacts.visual_evidence import VisualEvidenceReport

IDENTITY_FIELDS = (
    "document_id",
    "scope",
    "entity",
    "sample",
    "field",
    "condition",
    "condition_status",
    "measurement_state",
)
GOALS = ("correction", "shared_missing")


class ReviewModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)


class FrozenRef(ReviewModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BaselineRef(FrozenRef):
    document_id: str = Field(min_length=1)
    backend: Literal["mineru", "paddleocr_vl"]


class FactSource(ReviewModel):
    pdf: FrozenRef
    page: int = Field(ge=0, strict=True)
    bbox: NormalizedBBox


class ReviewedFact(ReviewModel):
    key: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    scope: Literal["paper", "sample"]
    entity: str | None
    sample: str | None
    field: str = Field(min_length=1)
    condition: str | None
    condition_status: Literal["stated", "not_stated", "unclear"]
    measurement_state: str = Field(min_length=1)
    value_raw: str | None = None
    unit_raw: str | None = None
    baseline: Literal["correction_opportunity", "shared_missing", "already_correct", "uncertain", "excluded"]
    source: FactSource
    reason: str = ""
    penalty_goal: Literal["correction", "shared_missing"] | None = None

    @model_validator(mode="after")
    def consistent_fact(self) -> Self:
        if self.scope == "sample" and (not self.entity or not self.sample):
            raise ValueError("sample fact needs entity and sample")
        if self.scope == "paper" and (self.entity is not None or self.sample is not None):
            raise ValueError("paper fact cannot name entity or sample")
        if self.condition_status == "stated" and not self.condition:
            raise ValueError("stated condition needs condition text")
        if self.condition_status == "not_stated" and self.condition is not None:
            raise ValueError("not_stated condition must be null")
        if self.baseline == "excluded":
            if not self.reason or self.penalty_goal is None:
                raise ValueError("excluded fact needs reason and penalty_goal")
        elif self.penalty_goal is not None or not self.value_raw:
            raise ValueError("reviewed fact needs value_raw and cannot override penalty_goal")
        return self


class ObservationReview(ReviewModel):
    report_path: Path
    reading_index: int = Field(ge=0, strict=True)
    fact_index: int = Field(ge=0, strict=True)
    fact_key: str = Field(min_length=1)
    verdict: Literal["correct", "wrong", "uncertain"]
    reason: str = Field(min_length=1)


class ReviewDocument(ReviewModel):
    format: Literal[1]
    review_version: str = Field(min_length=1)
    gold_revision: str = Field(min_length=1)
    evaluation_set: Literal["synthetic", "development", "c_holdout"] = "synthetic"
    baseline_sources: tuple[BaselineRef, ...] = Field(min_length=2)
    reports: tuple[FrozenRef, ...] = Field(min_length=1)
    facts: tuple[ReviewedFact, ...]
    observations: tuple[ObservationReview, ...]


def score_review(path: Path) -> dict:
    path = path.resolve()
    review_bytes = path.read_bytes()
    review = ReviewDocument.model_validate_json(review_bytes)
    verified: dict[Path, bytes] = {}
    hashes: dict[str, str] = {}

    def resolve(file_path: Path) -> Path:
        return (path.parent / file_path).resolve()

    # Freeze every byte reference before parsing any result or counting any observation.
    for ref in (*review.baseline_sources, *review.reports, *(fact.source.pdf for fact in review.facts)):
        target = resolve(ref.path)
        data = verified.get(target)
        if data is None:
            data = target.read_bytes()
            verified[target] = data
        digest = hashlib.sha256(data).hexdigest()
        if digest != ref.sha256:
            raise ValueError(f"SHA mismatch for frozen input: {target}")
        hashes[str(target)] = digest

    facts: dict[str, ReviewedFact] = {}
    identities: set[tuple] = set()
    pdfs_by_doc: dict[str, set[str]] = {}
    for fact in review.facts:
        identity = tuple(getattr(fact, name) for name in IDENTITY_FIELDS)
        if fact.key in facts or identity in identities:
            raise ValueError(f"duplicate or conflicting canonical fact definition: {fact.key}")
        identities.add(identity)
        facts[fact.key] = fact
        pdfs_by_doc.setdefault(fact.document_id, set()).add(fact.source.pdf.sha256)
    if any(len(digests) != 1 for digests in pdfs_by_doc.values()):
        raise ValueError("fact sources disagree on the original PDF SHA for one document")
    page_counts: dict[Path, int] = {}
    for fact in review.facts:
        target = resolve(fact.source.pdf.path)
        if target not in page_counts:
            try:
                page_counts[target] = read_geometry(target).page_count
            except (PdfiumError, OSError, ValueError) as exc:
                raise ValueError(f"reviewed source PDF is unreadable: {target}") from exc
        if fact.source.page >= page_counts[target]:
            raise ValueError(f"reviewed source page is outside the frozen PDF: {fact.key}")

    lanes: dict[tuple[str, str], LaneExtraction] = {}
    for ref in review.baseline_sources:
        lane = LaneExtraction.model_validate_json(verified[resolve(ref.path)])
        key = (ref.document_id, ref.backend)
        if key in lanes or lane.document_id != ref.document_id or lane.backend != ref.backend:
            raise ValueError("duplicate or inconsistent baseline lane identity")
        if lane.failed_questions:
            raise ValueError("incomplete baseline lane has failed extraction questions")
        lanes[key] = lane

    reports: dict[Path, VisualEvidenceReport] = {}
    report_pairs: set[tuple[str, str]] = set()
    for ref in review.reports:
        target = resolve(ref.path)
        report = VisualEvidenceReport.model_validate_json(verified[target])
        pair = (report.document_id, report.strategy)
        if target in reports or pair in report_pairs:
            raise ValueError("duplicate report for document and strategy")
        if report.strategy not in ("balanced", "risk_only") or report.selection.strategy != report.strategy:
            raise ValueError("report strategy does not match its selection")
        if any((reading.status == "observed") != bool(reading.facts) for reading in report.readings):
            raise ValueError("reading status and complete fact tuple list are inconsistent")
        if report.document_id in pdfs_by_doc and report.pdf_sha256 not in pdfs_by_doc[report.document_id]:
            raise ValueError("report PDF SHA differs from reviewed original source")
        reports[target] = report
        report_pairs.add(pair)
    documents = {fact.document_id for fact in facts.values()} | {report.document_id for report in reports.values()}
    for document_id in documents:
        pair = [lanes.get((document_id, backend)) for backend in ("mineru", "paddleocr_vl")]
        if any(lane is None for lane in pair):
            raise ValueError(f"both complete baseline lanes are required: {document_id}")
        if pair[0].extractor_key != pair[1].extractor_key:
            raise ValueError(f"baseline lanes have different extractor keys: {document_id}")
    for report in reports.values():
        expected = {
            backend: hashlib.sha256(lanes[(report.document_id, backend)].model_dump_json().encode()).hexdigest()
            for backend in ("mineru", "paddleocr_vl")
        }
        if report.baseline_sha256 != expected:
            raise ValueError("baseline lane content does not match the frozen report digests")

    actual = {
        (target, reading_index, fact_index): (report, reading, fact)
        for target, report in reports.items()
        for reading_index, reading in enumerate(report.readings)
        for fact_index, fact in enumerate(reading.facts)
    }
    labels: dict[tuple, ObservationReview] = {}
    for label in review.observations:
        pointer = (resolve(label.report_path), label.reading_index, label.fact_index)
        if pointer in labels:
            raise ValueError("duplicate observation review")
        if pointer not in actual:
            raise ValueError("observation pointer does not identify a fact in a frozen report")
        fact = facts.get(label.fact_key)
        if fact is None or fact.document_id != actual[pointer][0].document_id:
            raise ValueError("observation maps to an unknown fact or different document")
        if fact.baseline == "excluded" and label.verdict == "correct":
            raise ValueError("excluded observation cannot be judged correct in the reviewed scope")
        labels[pointer] = label

    observations = []
    states: dict[str, dict[str, set[str]]] = {strategy: {} for _, strategy in report_pairs}
    pending: Counter[str] = Counter()
    for pointer, (report, reading, observation) in actual.items():
        label = labels.get(pointer)
        if label is None:
            pending[report.strategy] += 1
        else:
            states[report.strategy].setdefault(label.fact_key, set()).add(label.verdict)
        observations.append(
            {
                "report_path": str(pointer[0]),
                "report_sha256": hashes[str(pointer[0])],
                "reading_index": pointer[1],
                "fact_index": pointer[2],
                "strategy": report.strategy,
                "fact_key": label.fact_key if label else None,
                "verdict": label.verdict if label else "pending_review",
                "reason": label.reason if label else "no human full-tuple review supplied",
                "observation": observation.model_dump(mode="json"),
                "candidate": reading.candidate.model_dump(mode="json"),
            }
        )

    strategies = {}
    for strategy, fact_states in sorted(states.items()):
        absent = sorted(documents - {doc for doc, name in report_pairs if name == strategy})
        strategies[strategy] = _score_strategy(facts, fact_states, pending[strategy], absent)
    return {
        "format": 1,
        "scope": "reviewed-scope",
        "scientific_acceptance": "not_established_by_scorer",
        "g2": "not_evaluated",
        "evaluation_set": review.evaluation_set,
        "review": {"path": str(path), "sha256": hashlib.sha256(review_bytes).hexdigest()},
        "review_version": review.review_version,
        "gold_revision": review.gold_revision,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_files": hashes,
        "reports": [
            {
                "path": str(target),
                "sha256": hashes[str(target)],
                "document_id": report.document_id,
                "strategy": report.strategy,
            }
            for target, report in reports.items()
        ],
        "baseline_sources": [ref.model_dump(mode="json") for ref in review.baseline_sources],
        "reviewed_facts": [fact.model_dump(mode="json") for fact in review.facts],
        "strategies": strategies,
        "observations": observations,
    }


def _score_strategy(
    facts: dict[str, ReviewedFact],
    states: dict[str, set[str]],
    pending: int,
    absent_documents: list[str],
) -> dict:
    counts = {goal: {"opportunities": 0, "correct": 0, "wrong": 0, "net": 0} for goal in GOALS}
    uncertain = set()
    fact_results = []
    for key, fact in sorted(facts.items()):
        verdicts = states.get(key, set())
        if fact.baseline == "uncertain" or "uncertain" in verdicts:
            uncertain.add(key)
        goal = {
            "correction_opportunity": "correction",
            "already_correct": "correction",
            "shared_missing": "shared_missing",
        }.get(fact.baseline)
        if fact.baseline == "excluded":
            goal = fact.penalty_goal
        gain = int(fact.baseline in ("correction_opportunity", "shared_missing") and "correct" in verdicts)
        penalty = int(goal is not None and "wrong" in verdicts)
        if goal is not None:
            counts[goal]["opportunities"] += int(fact.baseline in ("correction_opportunity", "shared_missing"))
            counts[goal]["correct"] += gain
            counts[goal]["wrong"] += penalty
        fact_results.append(
            {"fact_key": key, "baseline": fact.baseline, "verdicts": sorted(verdicts), "gain": gain, "penalty": penalty}
        )
    for goal in GOALS:
        counts[goal]["net"] = counts[goal]["correct"] - counts[goal]["wrong"]
    if pending or uncertain:
        result = "needs_review"
    elif absent_documents or any(counts[goal]["opportunities"] == 0 for goal in GOALS):
        result = "insufficient"
    else:
        result = "pass" if all(counts[goal]["net"] > 0 for goal in GOALS) else "fail"
    return {
        **counts,
        "pending_review": pending,
        "uncertain": len(uncertain),
        "metric_result": result,
        "absent_report_documents": absent_documents,
        "facts": fact_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--review", type=Path)
    choice.add_argument("--schema", action="store_true")
    args = parser.parse_args()
    if args.schema:
        print(json.dumps(ReviewDocument.model_json_schema(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    try:
        result = score_review(args.review)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
