"""Conservative, one-value-per-field datasets and an atomic Excel export.

The paper table selects a complete sample row. It must never manufacture a sample by
combining the best measurement of each field from different experimental conditions. "Different
conditions" is judged within a lane: the two lanes paraphrase the same condition differently, so
comparing their wording across lanes would refuse values the comparison report already agreed on.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.worksheet.worksheet import Worksheet

from paperfacts.compare import ComparisonReport, FieldComparison
from paperfacts.fields import AMBIGUOUS_MATCH_CONFIDENCE, FIELD_SPECS, SAMPLE_FIELDS, TARGET_FIELDS, FieldSpec
from paperfacts.models import BACKENDS, Backend, DocumentInput
from paperfacts.normalize import clean_unit, convert_to_canonical, delatex, normalize_lane, normalize_text, parse_number
from paperfacts.records import FieldValue, LaneExtraction, SampleRecord
from paperfacts.storage import write_atomic

CellValue = str | float | int | None
Row = Mapping[str, CellValue]

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+\.\d*|\.\d+|\d+)"
_ATOM = rf"(?:{_NUMBER}\s*x\s*10\s*\^?\s*[-+]?\d+|10\s*\^\s*[-+]?\d+|{_NUMBER}(?:[eE][-+]?\d+)?)"
_SCALAR = re.compile(rf"^(?P<center>{_ATOM})(?:\s*(?:±|\+/-|\+-|\\pm)\s*(?P<uncertainty>{_ATOM}))?(?P<tail>.*)$")
_APPROX = re.compile(r"^(?:approximately|approx\.?|about|ca\.?|[~≈≃≅])\s*", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_DESCRIPTIONS = {
    "component": "溅射靶材的化学组成；保留唯一组成文本，不拆选多个靶材。",
    "resistance": "靶材电阻率；与薄膜电阻率 resistivity 区分。",
    "density": "靶材相对理论密度，数值 95 表示 95%。",
    "inch": "靶材直径或唯一长度；矩形长宽、范围值不转成单个数值。",
    "sputtering_time": "所选样品的溅射沉积时间。",
    "sputtering_power": "所选样品的溅射功率（W）。",
    "mode": "溅射工作模式：DC、RF 或 pulsed DC。",
    "ar_flow_rate": "所选样品的氩气流量（sccm）。",
    "o2_flow_rate": "所选样品的氧气流量（sccm）。",
    "h2_flow_rate": "所选样品的氢气流量（sccm）。",
    "target_substrate_distance": "靶到基片/样品台的间距（cm）。",
    "substrate_axis_distance": "基片到样品台中心的偏轴距离（cm）。",
    "substrate_temperature": "沉积时的基片/样品台温度（°C）。",
    "annealing_temperature": "沉积后退火处理的温度（°C）。",
    "annealing_time": "沉积后退火处理的时长（min）。",
    "rotation_speed": "沉积时样品台的旋转速度（rpm）。",
    "sheet_resistance": "所选样品的薄膜方块电阻。",
    "resistivity": "所选样品的薄膜电阻率。",
    "transmittance": "所选样品的透光率，数值 85 表示 85%；测量波段见条件及数据质量。",
    "thickness": "所选样品的薄膜厚度。",
}
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
    ("detail", "说明"),
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

    def as_dict(self) -> dict[str, object]:
        """A JSON-serialisable view for the web UI.

        Rows are ``MappingProxyType`` so nothing downstream can mutate a consolidated row; ``json`` cannot
        dump one, so copy each into a plain dict here rather than weakening the model. The field list
        travels with the data because the rows carry values only: the browser needs the canonical unit and
        the paper/sample scope to build a header it can trust.
        """
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "extractor_key": self.extractor_key,
            "comparison_key": self.comparison_key,
            "fields": [
                {
                    "name": spec.name,
                    "unit": spec.canonical_unit,
                    "scope": "sample" if spec.is_sample_level else "target",
                }
                for spec in FIELD_SPECS
            ],
            "paper_row": dict(self.paper_row),
            "sample_rows": [dict(row) for row in self.sample_rows],
            "quality_rows": [dict(row) for row in self.quality_rows],
        }


def write_dataset_json(dataset: DocumentDataset, path: Path) -> None:
    """Write one document's consolidated dataset for the web UI, atomically like every other artifact."""
    payload = json.dumps(dataset.as_dict(), ensure_ascii=False, indent=2)
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


def _condition_key(value: str | None) -> str:
    # Preserve non-Latin text and meaningful operators, unlike a formula-oriented ASCII key.
    return re.sub(r"\s+", "", normalize_text(value or "")).casefold()


def _joined(values: Sequence[str]) -> str:
    return "; ".join(dict.fromkeys(value for value in values if value))


def _scalar(value: FieldValue, spec: FieldSpec) -> tuple[CellValue, str | None]:
    if spec.kind != "numeric":
        return value.value_raw.strip(), None
    text = delatex(normalize_text(value.value_raw)).strip()
    approx = _APPROX.match(text)
    if approx:
        text = text[approx.end() :].strip()
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
    canonical, _, note = convert_to_canonical(spec, number, value.unit_raw)
    if canonical is None or not math.isfinite(canonical):
        return None, note or "单位无法转换为标准单位"
    notes = [note or ""]
    if approx:
        notes.append("原文为近似值，保留中心值")
    if match.group("uncertainty"):
        notes.append(f"原文不确定度 ±{match.group('uncertainty')} {value.unit_raw or ''}；保留中心值")
    return canonical, _joined(notes) or None


def _same_value(a: CellValue, b: CellValue) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
    return isinstance(a, str) and isinstance(b, str) and normalize_text(a) == normalize_text(b)


def _decide(
    spec: FieldSpec,
    evidence: Sequence[tuple[Backend, FieldValue]],
    comparisons: Sequence[FieldComparison],
    *,
    blocked: str | None = None,
) -> _Decision:
    conditions = _joined([value.condition or "" for _, value in evidence])
    sources = _joined(sorted({source for _, value in evidence for source in value.source_ids}))
    details: list[str] = []

    def reject(status: str, reason: str) -> _Decision:
        raw = _joined([f"{backend}: {value.value_raw} {value.unit_raw or ''}" for backend, value in evidence])
        return _Decision(None, status, conditions, sources, _joined([reason, raw]))

    if not evidence:
        return reject("missing", "未提取到该字段；留空，不填 0")
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
    # "after deposition"), and compare.py has already judged whether their values and conditions
    # correspond. Several distinct conditions inside one lane really are several measurements.
    if any(
        len({_condition_key(value.condition) for lane, value in trusted if lane == backend}) > 1 for backend in BACKENDS
    ):
        return reject("multiple_conditions", "同一解析通道记录了多种测量条件，无法唯一确定")
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
        if same_lane and any(not _same_value(same_lane[0], scalar) for scalar in same_lane[1:]):
            return reject("multiple_values", "同一解析通道在相同条件下记录了多个不同值")
    chosen_backend, chosen_field, chosen = min(
        parsed, key=lambda item: (-item[1].agreement, BACKENDS.index(item[0]), item[1].value_raw)
    )
    agreed = any(c.status == "agree" for c in comparisons) and len({backend for backend, _, _ in parsed}) == 2
    if not agreed and any(not _same_value(chosen, scalar) for _, _, scalar in parsed):
        return reject("multiple_values", "多个候选值未经双路一致确认，无法唯一确定")
    if spec.name == "transmittance" and not conditions:
        details.append("原文提取结果未注明透光率波长或波段")
    details.append(f"采用 {chosen_backend}；抽取重复一致率 {chosen_field.agreement:g}；合并重复证据")
    return _Decision(chosen, "agree" if agreed else "single_source", conditions, sources, _joined(details))


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
    metadata: dict[str, CellValue] = {"document_id": document.document_id, "filename": document.pdf_path.name}
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
        blocked = None
        if scope.matching_failed:
            blocked = "样品匹配失败，无法确认跨通道身份"
        elif scope.confidence is not None and scope.confidence < AMBIGUOUS_MATCH_CONFIDENCE:
            blocked = "样品匹配置信度低于阈值"
        for spec in SAMPLE_FIELDS:
            evidence = [
                (backend, field)
                for backend, sample in ((report.backend_a, scope.a), (report.backend_b, scope.b))
                if sample is not None
                for field in sample.fields
                if field.field == spec.name
            ]
            decision = _decide(spec, evidence, [c for c in scope_comparisons if c.field == spec.name], blocked=blocked)
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
        document.pdf_path.name,
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
            if isinstance(cell.value, str):
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
    descriptions = [
        {
            "field": spec.name,
            "scope": "靶材（论文级）" if not spec.is_sample_level else "样品级",
            "unit": spec.canonical_unit or "文本",
            "description": _DESCRIPTIONS[spec.name],
            "rule": "冲突、多条件、多值、范围、上下界或无引用定位时留空；近似值和 ± 不确定度保留中心值并备注。",
        }
        for spec in FIELD_SPECS
    ]
    _worksheet(
        workbook,
        "字段说明",
        (
            ("field", "字段"),
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
