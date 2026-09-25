"""The Excel export: a consolidated dataset written to an atomic workbook.

Kept apart from :mod:`paperfacts.dataset` on purpose: that module's source is hashed into ``comparison_key``,
because the rows it assembles are verdicts. How those rows are laid out on a sheet (column order, widths,
number formats) decides nothing, so editing it must never rename a stored comparison. This module is in no
cache key.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.worksheet.worksheet import Worksheet

from paperfacts.dataset import DocumentDataset, Row, data_columns, field_columns
from paperfacts.profile import DomainProfile
from paperfacts.storage import write_atomic

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
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
# Values a vision model read off charts. The rows come from paperfacts.readings (which this module does not
# import: the readings are no part of any verdict here); this is only the sheet's layout.
_FIGURE_COLUMNS = (
    ("document_id", "文档ID"),
    ("filename", "文件名"),
    ("figure", "图"),
    ("page", "页码"),
    ("source_id", "图块来源"),
    ("panel", "子图"),
    ("field", "字段"),
    ("series", "系列"),
    ("x", "横轴（仅供参考，不用于对应样品）"),
    ("value", "读数（近似值）"),
    ("unit", "标准单位"),
    ("precision", "精度"),
    ("value_raw", "图中原始读数"),
    ("scale", "纵轴刻度"),
    ("caption", "图注"),
    ("detail", "说明"),
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
            "caption": 60,
            "x": 36,
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
    documents: Sequence[DocumentDataset],
    output: Path,
    profile: DomainProfile,
    *,
    failures: Sequence[dict[str, str]] = (),
    figure_rows: Sequence[Row] = (),
) -> None:
    """Replace a workbook atomically; repeated PDF hashes produce exactly one paper row.

    ``figure_rows`` (from :func:`paperfacts.readings.figure_rows`) only fill the 图中读数 sheet: chart readings
    are approximate and never compared, so they never reach a sample or paper row.
    """
    unique = sorted(
        {document.document_id: document for document in documents}.values(), key=lambda document: document.document_id
    )
    workbook = Workbook()
    workbook.remove(workbook.active)
    columns = data_columns(profile)
    _worksheet(workbook, "论文数据", columns, [doc.paper_row for doc in unique], "Papers")
    _worksheet(workbook, "样品数据", columns, [row for doc in unique for row in doc.sample_rows], "Samples")
    descriptions: list[Row] = [
        column.model_dump()
        | {
            # The sheet says the same things in Chinese, for a reader who opens the workbook alone.
            "scope": "样品级" if column.scope == "sample" else "靶材（论文级）",
            "unit": column.unit or "文本",
            "rule": "冲突、多条件、多值、范围、上下界或无引用定位时留空；近似值和 ± 不确定度保留中心值并备注。",
        }
        for column in field_columns(profile)
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
    _worksheet(workbook, "图中读数", _FIGURE_COLUMNS, figure_rows, "Figures")
    runs: list[Row] = [
        {
            "document_id": doc.document_id,
            "filename": doc.filename,
            "status": "incomplete" if doc.incomplete else "success",
            "samples": len(doc.sample_rows),
            "extractor_key": doc.extractor_key,
            "comparison_key": doc.comparison_key,
            "detail": f"未完成，下次运行重问：{doc.incomplete}"
            if doc.incomplete
            else "论文行采用一个完整样品；空白为缺失或未通过唯一值质量规则。",
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
