"""Conservative, one-value-per-field datasets and an atomic Excel export.

The paper table selects a complete sample row. It must never manufacture a sample by
combining the best measurement of each field from different experimental conditions. "Different
conditions" is judged within a lane: the two lanes paraphrase the same condition differently, so
comparing their wording across lanes would refuse values the comparison report already agreed on.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.worksheet.worksheet import Worksheet
from pydantic import BaseModel, ConfigDict

from paperfacts.compare import ComparisonReport, FieldComparison
from paperfacts.fields import (
    AMBIGUOUS_MATCH_CONFIDENCE,
    CONDITION_NUMBER,
    FIELD_SPECS,
    SAMPLE_FIELDS,
    TARGET_FIELDS,
    FieldSpec,
)
from paperfacts.models import BACKENDS, Backend, DocumentInput
from paperfacts.normalize import (
    clean_unit,
    convert_to_canonical,
    delatex,
    normalize_key,
    normalize_lane,
    normalize_text,
    parse_number,
    text_key,
)
from paperfacts.records import FieldValue, LaneExtraction, SampleRecord
from paperfacts.storage import write_atomic

CellValue = str | float | int | bool | None
Row = Mapping[str, CellValue]

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+\.\d*|\.\d+|\d+)"
_ATOM = rf"(?:{_NUMBER}\s*x\s*10\s*\^?\s*[-+]?\d+|10\s*\^\s*[-+]?\d+|{_NUMBER}(?:[eE][-+]?\d+)?)"
_SCALAR = re.compile(rf"^(?P<center>{_ATOM})(?:\s*(?:±|\+/-|\+-|\\pm)\s*(?P<uncertainty>{_ATOM}))?(?P<tail>.*)$")
# "100 nm (± 5 nm)": the uncertainty in parentheses after the unit, read as "100 ± 5 nm" when both units agree.
_PARENTHESISED_UNCERTAINTY = re.compile(
    rf"^(?P<center>{_ATOM})\s*(?P<unit>[^\d\s(±][^(±]*?)?\s*\(\s*(?:±|\+/-|\+-)\s*(?P<uncertainty>{_ATOM})\s*(?P<again>[^)]*)\)$"
)
# The tilde operator U+223C and its friends are folded to "~" by normalize_text, which runs first.
_APPROX = re.compile(r"^(?:approximately|approx\.?|roughly|around|about|circa|ca\.?|[~≈≃≅])\s*", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_DATA_COLUMNS = (
    ("document_id", "文档ID"),
    ("filename", "文件名"),
    ("sample_id", "样品ID"),
    ("sample_label", "样品标签"),
    ("conditions", "样品及测量条件"),
    ("available_fields", "可用字段数"),
    ("agree_fields", "双路一致字段数"),
    *((spec.name, spec.name) for spec in FIELD_SPECS),
)
_QUALITY_COLUMNS = (
    ("document_id", "文档ID"),
    ("filename", "文件名"),
    ("sample_id", "样品ID"),
    ("field", "字段"),
    ("decision", "最终决策"),
    ("value", "输出值"),
    ("unit", "标准单位"),
    ("conditions", "条件"),
    ("source_ids", "合并证据来源"),
    ("lanes", "证据来源通道"),
    ("series", "系列级"),
    ("detail", "说明"),
)


class FieldColumn(BaseModel):
    """What a reader needs to know about one column, built once for both the web UI and the Excel sheet.

    ``label`` and ``description`` are display only and may be empty when ``config.json`` declares neither;
    ``unit`` is absent for a text field.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    label: str = ""
    unit: str | None = None
    scope: str
    description: str = ""


class DatasetPayload(BaseModel):
    """One document's consolidated dataset as it crosses the disk and HTTP boundaries.

    The same model is written to ``dataset.json``, parsed back from it and returned by the endpoint, so
    the browser's contract is declared once and FastAPI can publish a schema for it. The field list
    travels with the data because the rows carry values only: the browser needs the canonical unit and
    the paper/sample scope to build a header it can trust.
    """

    model_config = ConfigDict(frozen=True)

    document_id: str = ""
    filename: str = ""
    extractor_key: str = ""
    comparison_key: str = ""
    fields: tuple[FieldColumn, ...] = ()
    paper_row: dict[str, CellValue] = {}
    sample_rows: tuple[dict[str, CellValue], ...] = ()
    quality_rows: tuple[dict[str, CellValue], ...] = ()


def field_columns() -> tuple[FieldColumn, ...]:
    """The field table as columns, in the order the dataset writes them."""
    return tuple(
        FieldColumn(
            name=spec.name,
            label=spec.label,
            unit=spec.canonical_unit,
            scope="sample" if spec.is_sample_level else "target",
            description=spec.description_zh,
        )
        for spec in FIELD_SPECS
    )


@dataclass(frozen=True)
class DocumentDataset:
    document_id: str
    filename: str
    paper_row: Row
    sample_rows: tuple[Row, ...]
    quality_rows: tuple[Row, ...]
    extractor_key: str = ""
    comparison_key: str = ""

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
            fields=field_columns(),
            paper_row=dict(self.paper_row),
            sample_rows=tuple(dict(row) for row in self.sample_rows),
            quality_rows=tuple(dict(row) for row in self.quality_rows),
        )

    @classmethod
    def from_payload(cls, payload: DatasetPayload) -> DocumentDataset:
        """The exact inverse of :meth:`to_payload`, so a dataset read back from disk can be exported again
        without re-running the pipeline. The field list is not restored: it is derived from FIELD_SPECS on
        the way out, and a payload written under a different field table lives under a different
        extractor_key and is never read next to this one.

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
        )


def write_dataset_json(dataset: DocumentDataset, path: Path) -> None:
    """Write one document's consolidated dataset for the web UI, atomically like every other artifact."""
    payload = dataset.to_payload().model_dump_json(indent=2)
    write_atomic(path, lambda tmp: tmp.write_text(payload, encoding="utf-8"))


@dataclass(frozen=True)
class _Scope:
    sample_id: str
    report_scope: str
    a: SampleRecord | None
    b: SampleRecord | None
    confidence: float | None = None
    matching_failed: bool = False


@dataclass(frozen=True)
class _Decision:
    value: CellValue
    status: str
    conditions: str
    sources: str
    detail: str
    # The committed value rests entirely on evidence the paper stated for the whole sample series,
    # never for this sample on its own. False for a rejected decision, which commits to nothing.
    series: bool = False
    # The backends whose trusted, parsed evidence produced the committed value. A reader seeing a
    # single-source cell needs to know which lane it came from; empty for a rejected decision.
    lanes: tuple[Backend, ...] = ()


def _joined(values: Sequence[str]) -> str:
    return "; ".join(dict.fromkeys(value for value in values if value))


def _scalar(value: FieldValue, spec: FieldSpec) -> tuple[CellValue, str | None]:
    if spec.kind != "numeric":
        return value.value_raw.strip(), None
    text = delatex(normalize_text(value.value_raw)).strip()
    approx = _APPROX.match(text)
    if approx:
        text = text[approx.end() :].strip()
    parenthesised = _PARENTHESISED_UNCERTAINTY.fullmatch(text)
    if parenthesised:
        center, unit, uncertainty, again = parenthesised.group("center", "unit", "uncertainty", "again")
        if clean_unit(unit or "") == clean_unit(again):
            text = f"{center} ± {uncertainty} {unit or ''}"
    match = _SCALAR.fullmatch(text)
    if match is None:
        return None, "不是唯一精确标量（含上下界、区间、尺寸组合或无法解析的文字）"
    tail = match.group("tail").strip()
    allowed_units = {clean_unit(unit) for unit in (value.unit_raw, spec.canonical_unit) if unit}
    if tail and clean_unit(tail) not in allowed_units:
        return None, "含多个数值、范围、上下界或附加条件，不能取中点或第一个数"
    number, _ = parse_number(match.group("center"))
    if number is None or not math.isfinite(number):
        return None, "数值不可解析或非有限数"
    canonical, _, note = convert_to_canonical(spec, number, value.unit_raw, value_text=match.group("center"))
    if canonical is None or not math.isfinite(canonical):
        return None, note or "单位无法转换为标准单位"
    notes = [note or ""]
    if approx:
        notes.append("原文为近似值，保留中心值")
    if match.group("uncertainty"):
        notes.append(f"原文不确定度 ±{match.group('uncertainty')} {value.unit_raw or ''}；保留中心值")
    return canonical, _joined(notes) or None


def _same_value(a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
    """Whether two candidate cells state the same thing. Text fields with a closed category set are judged
    on the category, so "DC and RF" and "DC and RF magnetron co-sputtering" are one answer rather than a
    refusal; a field without one falls back to folded-text equality."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    if spec.categories:
        return text_key(spec, a) == text_key(spec, b)
    return normalize_text(a) == normalize_text(b)


def _within_tolerance(a: CellValue, b: CellValue, spec: FieldSpec) -> bool:
    """Whether two candidate cells agree the way compare.py judges agreement: within the field's tolerance."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=spec.rel_tol, abs_tol=spec.abs_tol)
    return _same_value(a, b, spec)


def _matching_blocked(scope: _Scope | None) -> str | None:
    """Why nothing measured on this scope may be committed, or None if it may.

    Scope-wide rather than per-field: if the two lanes' samples were not confidently identified as the
    same sample, no value on them can be trusted, whatever the per-field comparison says. The paper-level
    target row has no scope and so is never blocked this way.
    """
    if scope is None:
        return None
    if scope.matching_failed:
        return "样品匹配失败，无法确认跨通道身份"
    if scope.confidence is not None and scope.confidence < AMBIGUOUS_MATCH_CONFIDENCE:
        return "样品匹配置信度低于阈值"
    return None


def _commit(
    spec: FieldSpec,
    chosen: tuple[Backend, FieldValue, CellValue],
    parsed: Sequence[tuple[Backend, FieldValue, CellValue]],
    *,
    agreed: bool,
    conditions: str,
    sources: str,
    details: list[str],
) -> _Decision:
    """Nothing refused the evidence: record the value and how it was arrived at."""
    chosen_backend, chosen_field, value = chosen
    if spec.name == "transmittance" and not conditions:
        details.append("原文提取结果未注明透光率波长或波段")
    details.append(f"采用 {chosen_backend}；抽取重复一致率 {chosen_field.agreement:g}；合并重复证据")
    series = all(field.series for _, field, _ in parsed)
    return _Decision(
        value,
        "agree" if agreed else "single_source",
        conditions,
        sources,
        _joined(details),
        series=series,
        lanes=tuple(dict.fromkeys(backend for backend, _, _ in parsed)),
    )


def _one_condition(
    spec: FieldSpec, trusted: Sequence[tuple[Backend, FieldValue]], row_sources: frozenset[str]
) -> tuple[list[tuple[Backend, FieldValue]], str] | None:
    """Of a sample's several measurements, the one the cell should state, with the reason; or None.

    Tried in order, the first that settles it wins:

    1. The condition stated in a block the rest of the row also cites. "Resistivity of 5.74e-4 Ω·cm and a
       transmittance of 83.5 % (400-1800 nm)" ties one of several transmittances to the rest of its row;
       that is the one a reader expects in the cell.
    2. The field's ``condition_preference``, entry by entry: a condition matches an entry when it names
       exactly the entry's numbers, so "average 400–800 nm" and "from 400 to 800 nm" both match "400-800".

    Once a rule chooses, every lane is held to it: a lane keeps only its values the rule picks, so a lane
    quoting a different condition cannot vouch for the one chosen. A rule that picks two conditions in one
    lane, or nothing in any, settles nothing and the next is tried.
    """
    rules: list[tuple[str, Callable[[FieldValue], bool]]] = [
        ("采用与本行其他字段引用同一原文块的条件", lambda value: bool(row_sources.intersection(value.source_ids)))
    ]
    for entry in spec.condition_preference:
        numbers = _condition_numbers(entry)
        rules.append(
            (
                f"按字段配置的优先条件 {entry} 选取",
                lambda value, numbers=numbers: _condition_numbers(value.condition) == numbers,
            )
        )
    for reason, picks in rules:
        kept = _held_to(trusted, picks)
        if kept:
            return kept, reason
    return None


def _condition_numbers(condition: str | None) -> tuple[float, ...]:
    return tuple(float(number) for number in CONDITION_NUMBER.findall(delatex(normalize_text(condition or ""))))


def _held_to(
    trusted: Sequence[tuple[Backend, FieldValue]], picks: Callable[[FieldValue], bool]
) -> list[tuple[Backend, FieldValue]] | None:
    kept: list[tuple[Backend, FieldValue]] = []
    for backend in BACKENDS:
        groups: dict[str, list[FieldValue]] = {}
        for lane, value in trusted:
            if lane == backend and picks(value):
                groups.setdefault(normalize_key(value.condition), []).append(value)
        if len(groups) > 1:
            return None
        kept += [(backend, value) for values in groups.values() for value in values]
    return kept or None


def _decide(
    spec: FieldSpec,
    evidence: Sequence[tuple[Backend, FieldValue]],
    comparisons: Sequence[FieldComparison],
    *,
    scope: _Scope | None = None,
    row_sources: frozenset[str] = frozenset(),
) -> _Decision:
    conditions = _joined([value.condition or "" for _, value in evidence])
    sources = _joined(sorted({source for _, value in evidence for source in value.source_ids}))
    details: list[str] = []

    def reject(status: str, reason: str) -> _Decision:
        raw = _joined([f"{backend}: {value.value_raw} {value.unit_raw or ''}" for backend, value in evidence])
        return _Decision(None, status, conditions, sources, _joined([reason, raw]))

    if not evidence:
        return reject("missing", "未提取到该字段；留空，不填 0")
    blocked = _matching_blocked(scope)
    if blocked:
        return reject("ambiguous", blocked)
    if any(c.status in {"conflict", "ambiguous"} for c in comparisons):
        status = "conflict" if any(c.status == "conflict" for c in comparisons) else "ambiguous"
        return reject(status, "双路比较存在冲突或歧义，需人工复核")
    if not comparisons:
        return reject("unreviewed", "比较报告没有覆盖该字段")
    if any(c.match_confidence is not None and c.match_confidence < AMBIGUOUS_MATCH_CONFIDENCE for c in comparisons):
        return reject("ambiguous", "样品匹配置信度低于阈值")
    trusted = [(backend, value) for backend, value in evidence if value.grounded and value.source_ids]
    if not trusted:
        return reject("ungrounded", "没有同时通过原文定位且包含有效引用的证据")
    if len(trusted) != len(evidence):
        details.append("已排除未定位到原文或缺少有效引用的候选")
    # Per lane only: the two lanes word the same condition differently ("after sputtering" vs
    # "after deposition"), so only a lane disagreeing with itself is evidence of several measurements.
    # The key is normalize_key, the same one compare.py and extract.py judge conditions by, so a
    # condition the comparison report called one thing is never two here.
    narrowed = any(
        len({normalize_key(value.condition) for lane, value in trusted if lane == backend}) > 1 for backend in BACKENDS
    )
    if narrowed:
        chosen_condition = _one_condition(spec, trusted, row_sources)
        if chosen_condition is None:
            return reject("multiple_conditions", "同一解析通道记录了多种测量条件，无法唯一确定")
        # The cell now states one of several measurements, so it names only that one's condition and blocks.
        trusted, reason = chosen_condition
        conditions = _joined([value.condition or "" for _, value in trusted])
        sources = _joined(sorted({source for _, value in trusted for source in value.source_ids}))
        details.append(f"该样品有多种测量条件；{reason}")
    parsed: list[tuple[Backend, FieldValue, CellValue]] = []
    for backend, value in trusted:
        scalar, note = _scalar(value, spec)
        if scalar is None:
            return reject("non_scalar", note or "无法生成唯一标量")
        parsed.append((backend, value, scalar))
        if note:
            details.append(note)
    # compare.py keeps the first value per condition. Inspect every extraction candidate
    # here so two different same-condition values cannot disappear behind that first one.
    for backend in BACKENDS:
        same_lane = [scalar for lane, _, scalar in parsed if lane == backend]
        if same_lane and any(not _same_value(same_lane[0], scalar, spec) for scalar in same_lane[1:]):
            return reject("multiple_values", "同一解析通道在相同条件下记录了多个不同值")
    chosen = min(parsed, key=lambda item: (-item[1].agreement, BACKENDS.index(item[0]), item[1].value_raw))
    # Agreement is two lanes vouching for the very value committed. The comparison's "agree" can be about a
    # value this cell no longer holds -- a condition set aside by narrowing, or a lane's value that failed
    # grounding and was left out of `trusted` -- so the remaining values have to match as well.
    agreed = (
        len({backend for backend, _, _ in parsed}) == 2
        and all(_within_tolerance(chosen[2], scalar, spec) for _, _, scalar in parsed)
        and (narrowed or any(c.status == "agree" for c in comparisons))
    )
    if not agreed and any(not _same_value(chosen[2], scalar, spec) for _, _, scalar in parsed):
        return reject("multiple_values", "多个候选值未经双路一致确认，无法唯一确定")
    return _commit(spec, chosen, parsed, agreed=agreed, conditions=conditions, sources=sources, details=details)


def _scopes(lanes: Mapping[Backend, LaneExtraction], report: ComparisonReport) -> tuple[_Scope, ...]:
    lane_a, lane_b = lanes[report.backend_a], lanes[report.backend_b]
    scopes: list[_Scope] = []
    for pair in report.matching.pairs:
        a, b = lane_a.sample(pair.a_id), lane_b.sample(pair.b_id)
        if a is not None and b is not None:
            sample_id = pair.a_id if pair.a_id == pair.b_id else f"{pair.a_id} | {pair.b_id}"
            scopes.append(_Scope(sample_id, f"sample:{pair.a_id}|{pair.b_id}", a, b, pair.confidence))
    for backend, ids, side in (
        (report.backend_a, report.matching.unmatched_a, "a"),
        (report.backend_b, report.matching.unmatched_b, "b"),
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
                        matching_failed=report.matching.failed,
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


def consolidate_document(
    document: DocumentInput, lanes: Mapping[Backend, LaneExtraction], report: ComparisonReport
) -> DocumentDataset:
    """Collapse source evidence, then select the most complete trustworthy sample row."""
    if report.document_id != document.document_id or any(
        lane.document_id != document.document_id for lane in lanes.values()
    ):
        raise ValueError("document, extraction lanes and comparison report must refer to the same PDF")
    if any(lane.extractor_key != report.extractor_key for lane in lanes.values()):
        raise ValueError("extraction lanes and comparison report have different extractor keys")
    lanes = {backend: normalize_lane(lane) for backend, lane in lanes.items()}
    metadata: dict[str, CellValue] = {"document_id": document.document_id, "filename": document.display_filename}
    quality: list[Row] = []

    def record(sample_id: str, spec: FieldSpec, decision: _Decision) -> None:
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

    target: dict[str, _Decision] = {}
    for spec in TARGET_FIELDS:
        evidence = [
            (backend, field)
            for backend, lane in lanes.items()
            if lane.target
            for field in lane.target.fields
            if field.field == spec.name
        ]
        target[spec.name] = _decide(
            spec, evidence, [c for c in report.comparisons if c.scope == "target" and c.field == spec.name]
        )
    for spec in TARGET_FIELDS:
        record("target", spec, target[spec.name])

    sample_rows: list[Row] = []
    for scope in _scopes(lanes, report):
        scope_comparisons = _scope_comparisons(scope, report)
        decisions = dict(target)
        for spec in SAMPLE_FIELDS:
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
            decision = _decide(
                spec,
                evidence,
                [c for c in scope_comparisons if c.field == spec.name],
                scope=scope,
                row_sources=row_sources,
            )
            decisions[spec.name] = decision
            record(scope.sample_id, spec, decision)
        samples = [sample for sample in (scope.a, scope.b) if sample is not None]
        conditions = _joined(
            [f"{key}={value}" for sample in samples for key, value in sorted(sample.conditions.items())]
            + [
                f"{spec.name}: {decisions[spec.name].conditions}"
                for spec in SAMPLE_FIELDS
                if decisions[spec.name].conditions
            ]
        )
        sample_rows.append(
            MappingProxyType(
                {
                    **metadata,
                    "sample_id": scope.sample_id,
                    "sample_label": _joined([sample.label for sample in samples]),
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
                "available_fields": sum(d.value is not None for d in target.values()),
                "agree_fields": sum(d.status == "agree" for d in target.values()),
                **{spec.name: target[spec.name].value if spec.name in target else None for spec in FIELD_SPECS},
            }
        )
        selection = "未提取到可匹配样品；论文行仅保留唯一的靶材字段"
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
    )


def _worksheet(
    workbook: Workbook, title: str, columns: Sequence[tuple[str, str]], rows: Sequence[Row], table_id: str
) -> Worksheet:
    sheet = workbook.create_sheet(title)
    sheet.append([label for _, label in columns])
    for row in rows:
        sheet.append([row.get(key) for key, _ in columns])
    sheet.freeze_panes = "D2" if title in {"论文数据", "样品数据", "数据质量"} else "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[1].height = 30
    for cell in sheet[1]:
        cell.font = Font(name="Calibri", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="17365D")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for column, (key, _) in enumerate(columns, start=1):
        width = {
            "document_id": 20,
            "filename": 48,
            "conditions": 48,
            "source_ids": 48,
            "detail": 80,
            "sample_id": 28,
            "sample_label": 35,
            "description": 68,
            "rule": 70,
        }.get(key, 23)
        sheet.column_dimensions[get_column_letter(column)].width = width
        for cells in sheet.iter_rows(min_row=2, min_col=column, max_col=column):
            cell = cells[0]
            if isinstance(cell.value, bool):
                # openpyxl writes a bool as Excel TRUE/FALSE; a number format would be misleading.
                pass
            elif isinstance(cell.value, str):
                cell.value = _CONTROL.sub("", cell.value)
                # PDF-derived strings are data even when their first character is '='.
                cell.data_type = "s"
            elif isinstance(cell.value, (int, float)):
                cell.number_format = "0.0000E+00" if key in {"resistance", "resistivity"} else "0.############"
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    if rows:
        table = Table(displayName=table_id, ref=sheet.dimensions)
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        sheet.add_table(table)
    return sheet


def write_dataset(
    documents: Sequence[DocumentDataset], output: Path, *, failures: Sequence[dict[str, str]] = ()
) -> None:
    """Replace a workbook atomically; repeated PDF hashes produce exactly one paper row."""
    unique = sorted(
        {document.document_id: document for document in documents}.values(), key=lambda document: document.document_id
    )
    workbook = Workbook()
    workbook.remove(workbook.active)
    _worksheet(workbook, "论文数据", _DATA_COLUMNS, [doc.paper_row for doc in unique], "Papers")
    _worksheet(workbook, "样品数据", _DATA_COLUMNS, [row for doc in unique for row in doc.sample_rows], "Samples")
    descriptions: list[Row] = [
        column.model_dump()
        | {
            # The sheet says the same things in Chinese, for a reader who opens the workbook alone.
            "scope": "样品级" if column.scope == "sample" else "靶材（论文级）",
            "unit": column.unit or "文本",
            "rule": "冲突、多条件、多值、范围、上下界或无引用定位时留空；近似值和 ± 不确定度保留中心值并备注。",
        }
        for column in field_columns()
    ]
    _worksheet(
        workbook,
        "字段说明",
        (
            ("name", "字段"),
            ("label", "中文名"),
            ("scope", "层级"),
            ("unit", "标准单位"),
            ("description", "中文说明"),
            ("rule", "单值与缺失规则"),
        ),
        descriptions,
        "Fields",
    )
    _worksheet(workbook, "数据质量", _QUALITY_COLUMNS, [row for doc in unique for row in doc.quality_rows], "Quality")
    runs: list[Row] = [
        {
            "document_id": doc.document_id,
            "filename": doc.filename,
            "status": "success",
            "samples": len(doc.sample_rows),
            "extractor_key": doc.extractor_key,
            "comparison_key": doc.comparison_key,
            "detail": "论文行采用一个完整样品；空白为缺失或未通过唯一值质量规则。",
        }
        for doc in unique
    ]
    runs.extend(
        {
            "document_id": failure.get("document_id", ""),
            "filename": failure.get("filename") or failure.get("pdf") or failure.get("path", ""),
            "status": "failed",
            "detail": failure.get("error") or failure.get("detail", str(failure)),
        }
        for failure in failures
    )
    _worksheet(
        workbook,
        "运行记录",
        (
            ("document_id", "文档ID"),
            ("filename", "文件名"),
            ("status", "状态"),
            ("samples", "合并后样品数"),
            ("extractor_key", "抽取版本"),
            ("comparison_key", "比较版本"),
            ("detail", "说明"),
        ),
        runs,
        "Runs",
    )
    write_atomic(output, workbook.save)
