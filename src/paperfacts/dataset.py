"""Conservative, one-value-per-field datasets.

The paper table selects a complete sample row. It must never manufacture a sample by combining the best
measurement of each field from different experimental conditions. Which value a cell holds -- and whether it
holds one at all -- is decided per cell by :mod:`paperfacts.decide`; this module gathers each cell's evidence
and assembles the rows. Writing them to Excel is :mod:`paperfacts.workbook`'s, which is not hashed into any
cache key: how a sheet looks is no verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from paperfacts.columns import FieldColumn
from paperfacts.compare import PAPER_SCOPE, ComparisonReport, FieldComparison, check_profile
from paperfacts.decide import Decision, decide_cell
from paperfacts.fields import FieldSpec
from paperfacts.keys import ComparisonOptions, profile_comparison_fingerprint
from paperfacts.kinds import CellValue, joined
from paperfacts.models import Backend, DocumentInput
from paperfacts.normalize import normalize_lane
from paperfacts.records import LaneExtraction, SampleRecord
from paperfacts.storage import write_atomic

Row = Mapping[str, CellValue]
# The paper-level quality rows' sample_id as files written before round 2 spell it.
_LEGACY_PAPER_ID = "target"


class DatasetPayload(BaseModel):
    """One document's consolidated dataset as it crosses the disk and HTTP boundaries.

    The same model is written to ``dataset.json``, parsed back from it and returned by the endpoint, so
    the browser's contract is declared once and FastAPI can publish a schema for it. The field list is
    the endpoint's alone: the rows carry values only, so the browser needs the canonical unit and the
    paper/sample scope to build a header it can trust, but the list is display text built from the profile
    the server runs under (:func:`paperfacts.columns.field_columns`). It is never written to disk, and a
    file that still has one is read without it.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = ""
    filename: str = ""
    extractor_key: str = ""
    comparison_key: str = ""
    # Per lane, the content hash of the parse the table was built from (see ParsedArtifact.content_hash): its
    # cells cite blocks by position. A lane absent here is unknown (a file from before it was recorded).
    artifact_sha256: dict[Backend, str] = {}
    fields: tuple[FieldColumn, ...] = ()
    paper_row: dict[str, CellValue] = {}
    sample_rows: tuple[dict[str, CellValue], ...] = ()
    quality_rows: tuple[dict[str, CellValue], ...] = ()
    # keys.profile_comparison_fingerprint of the profile the table was consolidated under (None: an older file).
    profile_fingerprint: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _current_paper_id(cls, data: Any) -> Any:
        # A file written before round 2 ids its paper-level quality rows "target". Every sample has a sample
        # row, so a quality row whose id no sample row has can only be the paper's.
        if not isinstance(data, dict) or not isinstance(data.get("quality_rows"), list | tuple):
            return data
        samples = {row.get("sample_id") for row in data.get("sample_rows") or () if isinstance(row, dict)}
        if _LEGACY_PAPER_ID in samples:
            return data
        rows = [
            row | {"sample_id": PAPER_SCOPE}
            if isinstance(row, dict) and row.get("sample_id") == _LEGACY_PAPER_ID
            else row
            for row in data["quality_rows"]
        ]
        return data | {"quality_rows": rows}


@dataclass(frozen=True)
class DocumentDataset:
    document_id: str
    filename: str
    paper_row: Row
    sample_rows: tuple[Row, ...]
    quality_rows: tuple[Row, ...]
    extractor_key: str = ""
    comparison_key: str = ""
    artifact_sha256: Mapping[Backend, str] = field(default_factory=dict)
    # Why this run's table is not a finished result (incomplete_reason), or "". Such a table is written to the
    # run's workbook but never stored as dataset.json, so it never crosses the payload.
    incomplete: str = ""
    profile_fingerprint: str | None = None

    def to_payload(self) -> DatasetPayload:
        """The serialisable view the web UI and ``dataset.json`` share.

        Rows are ``MappingProxyType`` so nothing downstream can mutate a consolidated row; pydantic copies
        each into a plain dict here rather than weakening the model.
        """
        return DatasetPayload(
            document_id=self.document_id,
            filename=self.filename,
            extractor_key=self.extractor_key,
            comparison_key=self.comparison_key,
            artifact_sha256=dict(self.artifact_sha256),
            paper_row=dict(self.paper_row),
            sample_rows=tuple(dict(row) for row in self.sample_rows),
            quality_rows=tuple(dict(row) for row in self.quality_rows),
            profile_fingerprint=self.profile_fingerprint,
        )

    @classmethod
    def from_payload(cls, payload: DatasetPayload) -> DocumentDataset:
        """The exact inverse of :meth:`to_payload`, so a dataset read back from disk can be exported again
        without re-running the pipeline.

        The payload arrives already validated -- a malformed file fails at the disk boundary, where the
        caller can decide whether to skip that document or raise.
        """
        return cls(
            document_id=payload.document_id,
            filename=payload.filename,
            paper_row=MappingProxyType(dict(payload.paper_row)),
            sample_rows=tuple(MappingProxyType(dict(row)) for row in payload.sample_rows),
            quality_rows=tuple(MappingProxyType(dict(row)) for row in payload.quality_rows),
            extractor_key=payload.extractor_key,
            comparison_key=payload.comparison_key,
            artifact_sha256=dict(payload.artifact_sha256),
            profile_fingerprint=payload.profile_fingerprint,
        )


def write_dataset_json(dataset: DocumentDataset, path: Path) -> None:
    """Write one document's consolidated dataset for the web UI, atomically like every other artifact."""
    payload = dataset.to_payload().model_dump_json(indent=2, exclude={"fields"})
    write_atomic(path, lambda tmp: tmp.write_text(payload, encoding="utf-8"))


@dataclass(frozen=True)
class _Scope:
    sample_id: str
    report_scope: str
    a: SampleRecord | None
    b: SampleRecord | None
    confidence: float | None = None
    matching_failed: bool = False


def _matching_blocked(scope: _Scope | None, ambiguous_match_confidence: float) -> str | None:
    """Why nothing measured on this scope may be committed, or None if it may.

    Scope-wide rather than per-field: if the two lanes' samples were not confidently identified as the
    same sample, no value on them can be trusted, whatever the per-field comparison says. The paper-level
    row has no scope and so is never blocked this way.
    """
    if scope is None:
        return None
    if scope.matching_failed:
        return "样品匹配失败，无法确认跨通道身份"
    if scope.confidence is not None and scope.confidence < ambiguous_match_confidence:
        return "样品匹配置信度低于阈值"
    return None


def _scopes(lanes: Mapping[Backend, LaneExtraction], report: ComparisonReport) -> tuple[_Scope, ...]:
    lane_a, lane_b = lanes[report.backend_a], lanes[report.backend_b]
    matching = report.sample_matching()
    scopes: list[_Scope] = []
    for pair in matching.pairs:
        a, b = lane_a.sample(pair.a_id), lane_b.sample(pair.b_id)
        if a is not None and b is not None:
            sample_id = pair.a_id if pair.a_id == pair.b_id else f"{pair.a_id} | {pair.b_id}"
            scopes.append(_Scope(sample_id, f"sample:{pair.a_id}|{pair.b_id}", a, b, pair.confidence))
    for backend, ids, side in (
        (report.backend_a, matching.unmatched_a, "a"),
        (report.backend_b, matching.unmatched_b, "b"),
    ):
        for sample_id in ids:
            sample = lanes[backend].sample(sample_id)
            if sample is not None:
                scopes.append(
                    _Scope(
                        f"{backend}:{sample_id}",
                        f"sample:{sample_id}",
                        sample if side == "a" else None,
                        sample if side == "b" else None,
                        matching_failed=matching.failed,
                    )
                )
    return tuple(sorted(scopes, key=lambda scope: (scope.sample_id, scope.report_scope)))


def _scope_comparisons(scope: _Scope, report: ComparisonReport) -> tuple[FieldComparison, ...]:
    # Unmatched IDs can be identical across lanes; the comparison's populated side
    # disambiguates them without parsing the report's deliberately lossy scope string.
    return tuple(
        comparison
        for comparison in report.comparisons
        if comparison.scope == scope.report_scope
        and (scope.a is not None or comparison.a is None)
        and (scope.b is not None or comparison.b is None)
    )


def incomplete_reason(lanes: Mapping[Backend, LaneExtraction], report: ComparisonReport) -> str:
    """Why a run's comparison and consolidated table must not be stored as finished, or "" when they may be.

    Each reason is a model that answered badly this time, not a verdict about the paper. Stored, the result
    would be served on every later run -- and the stored table marks the paper finished, so "run all" would
    never retry it; unstored, the next run asks again, and only the failed request reaches the model, since
    invalid answers are never cached (llm.complete_validated).
    """
    if report.sample_matching().failed:
        return "sample matching failed"
    unanswered = [f"{backend}:{q.field}" for backend, lane in lanes.items() for q in lane.failed_questions]
    return f"no valid answer to {', '.join(unanswered)}" if unanswered else ""


def consolidate_document(
    document: DocumentInput,
    lanes: Mapping[Backend, LaneExtraction],
    report: ComparisonReport,
    options: ComparisonOptions,
) -> DocumentDataset:
    """Collapse source evidence, then select the most complete trustworthy sample row."""
    profile = options.profile
    fingerprint = profile_comparison_fingerprint(profile)
    check_profile(report.profile_fingerprint, fingerprint, "the comparison report")
    if report.document_id != document.document_id or any(
        lane.document_id != document.document_id for lane in lanes.values()
    ):
        raise ValueError("document, extraction lanes and comparison report must refer to the same PDF")
    if any(lane.extractor_key != report.extractor_key for lane in lanes.values()):
        raise ValueError("extraction lanes and comparison report have different extractor keys")
    incomplete = incomplete_reason(lanes, report)
    unanswered = {question.field for lane in lanes.values() for question in lane.failed_questions}
    lanes = {backend: normalize_lane(lane, profile) for backend, lane in lanes.items()}
    metadata: dict[str, CellValue] = {"document_id": document.document_id, "filename": document.display_filename}
    quality: list[Row] = []

    def record(sample_id: str, spec: FieldSpec, decision: Decision) -> None:
        quality.append(
            MappingProxyType(
                {
                    **metadata,
                    "sample_id": sample_id,
                    "field": spec.name,
                    "decision": decision.status,
                    "value": decision.value,
                    "unit": spec.canonical_unit,
                    "conditions": decision.conditions,
                    "source_ids": decision.sources,
                    "lanes": "; ".join(decision.lanes),
                    "series": decision.series,
                    "detail": decision.detail,
                }
            )
        )

    paper: dict[str, Decision] = {}
    for spec in profile.paper_fields:
        evidence = [
            (backend, field)
            for backend, lane in lanes.items()
            if lane.paper
            for field in lane.paper.fields
            if field.field == spec.name
        ]
        paper[spec.name] = decide_cell(
            spec,
            evidence,
            [c for c in report.comparisons if c.scope == PAPER_SCOPE and c.field == spec.name],
            units=profile.units,
            unanswered=spec.name in unanswered,
        )
    for spec in profile.paper_fields:
        record(PAPER_SCOPE, spec, paper[spec.name])

    sample_rows: list[Row] = []
    for scope in _scopes(lanes, report):
        scope_comparisons = _scope_comparisons(scope, report)
        decisions = dict(paper)
        for spec in profile.sample_fields:
            evidence = [
                (backend, field)
                for backend, sample in ((report.backend_a, scope.a), (report.backend_b, scope.b))
                if sample is not None
                for field in sample.fields
                if field.field == spec.name
            ]
            row_sources = frozenset(
                source
                for sample in (scope.a, scope.b)
                if sample is not None
                for field in sample.fields
                if field.field != spec.name and field.grounded
                for source in field.source_ids
            )
            decision = decide_cell(
                spec,
                evidence,
                [c for c in scope_comparisons if c.field == spec.name],
                units=profile.units,
                blocked=_matching_blocked(scope, options.ambiguous_match_confidence),
                unanswered=spec.name in unanswered,
                row_sources=row_sources,
            )
            decisions[spec.name] = decision
            record(scope.sample_id, spec, decision)
        samples = [sample for sample in (scope.a, scope.b) if sample is not None]
        conditions = joined(
            [f"{key}={value}" for sample in samples for key, value in sorted(sample.conditions.items())]
            + [
                f"{spec.name}: {decisions[spec.name].conditions}"
                for spec in profile.sample_fields
                if decisions[spec.name].conditions
            ]
        )
        sample_rows.append(
            MappingProxyType(
                {
                    **metadata,
                    "sample_id": scope.sample_id,
                    "sample_label": joined([sample.label for sample in samples]),
                    "conditions": conditions,
                    "available_fields": sum(d.value is not None for d in decisions.values()),
                    "agree_fields": sum(d.status == "agree" for d in decisions.values()),
                    **{name: decision.value for name, decision in decisions.items()},
                }
            )
        )
    if sample_rows:
        paper_row = min(
            sample_rows,
            key=lambda row: (-int(row["available_fields"] or 0), -int(row["agree_fields"] or 0), str(row["sample_id"])),
        )
        selection = "按可用字段数最多、双路一致字段数最多、样品ID稳定排序，选择整行；未跨样品拼接字段"
    else:
        paper_row = MappingProxyType(
            {
                **metadata,
                "sample_id": "",
                "sample_label": "",
                "conditions": "",
                "available_fields": sum(d.value is not None for d in paper.values()),
                "agree_fields": sum(d.status == "agree" for d in paper.values()),
                **{spec.name: paper[spec.name].value if spec.name in paper else None for spec in profile.fields},
            }
        )
        selection = "未提取到可匹配样品；论文行仅保留唯一的论文级字段"
    quality.append(
        MappingProxyType(
            {
                **metadata,
                "sample_id": paper_row["sample_id"],
                "field": "__selection__",
                "decision": "selected_sample",
                "detail": selection,
            }
        )
    )
    return DocumentDataset(
        document.document_id,
        document.display_filename,
        paper_row,
        tuple(sample_rows),
        tuple(quality),
        report.extractor_key,
        report.comparison_key,
        {
            backend: sha
            for backend, sha in (
                (report.backend_a, report.artifact_sha256_a),
                (report.backend_b, report.artifact_sha256_b),
            )
            if sha is not None
        },
        incomplete,
        fingerprint,
    )
