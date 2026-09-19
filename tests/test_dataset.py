"""An ML row must describe one actual sample and contain only defensible scalar values."""

import json
from pathlib import Path

import pytest
from openpyxl import load_workbook

from paperfacts.compare import compare_lanes
from paperfacts.dataset import DocumentDataset, consolidate_document, write_dataset, write_dataset_json
from paperfacts.fields import FIELD_SPECS
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import DocumentInput
from paperfacts.records import FieldValue, TargetRecord
from support.extraction import make_lane, make_sample
from support.factories import DOC_ID


def value(name, raw, unit=None, *, condition=None, backend="mineru", **kwargs):
    return FieldValue(
        field=name, value_raw=raw, unit_raw=unit, condition=condition, source_ids=(f"{backend}_p0_b1",), **kwargs
    )


def dataset(a, b=None, matching=None, *, filename="paper.pdf"):
    b = b or make_lane(backend="paddleocr_vl")
    matching = matching or SampleMatching(
        unmatched_a=tuple(s.sample_id for s in a.samples), unmatched_b=tuple(s.sample_id for s in b.samples)
    )
    document = DocumentInput(document_id=DOC_ID, sha256=DOC_ID, pdf_path=Path(filename))
    report = compare_lanes(a, b, matching)
    return consolidate_document(document, {a.backend: a, b.backend: b}, report)


def paired(a, b, *, confidence=1.0, filename="paper.pdf"):
    return dataset(
        make_lane(samples=[make_sample("A", a)]),
        make_lane(backend="paddleocr_vl", samples=[make_sample("A", b)]),
        SampleMatching(
            pairs=(SampleMatch(a_id="A", b_id="A", confidence=confidence, method="llm", justification="test"),)
        ),
        filename=filename,
    )


def decision(result, field, sample_id=None):
    return next(
        row
        for row in result.quality_rows
        if row["field"] == field and (sample_id is None or row["sample_id"] == sample_id)
    )


def test_paper_row_selects_one_complete_sample_without_cross_sample_fill():
    result = dataset(
        make_lane(
            samples=[
                make_sample(
                    "A",
                    [value("thickness", "300", "nm"), value("transmittance", "85", "%", condition="550 nm")],
                    label="as deposited",
                    conditions={"temperature": "300 K"},
                ),
                make_sample("B", [value("resistivity", "1e-4", "Ω cm")], label="annealed"),
            ]
        )
    )

    assert result.paper_row["sample_id"] == "mineru:A"
    assert result.paper_row["thickness"] == 300
    assert result.paper_row["transmittance"] == 85
    assert result.paper_row["resistivity"] is None
    assert "temperature=300 K" in result.paper_row["conditions"]
    assert len(result.sample_rows) == 2
    assert {spec.name for spec in FIELD_SPECS} <= result.paper_row.keys()


def test_unit_conversion_and_duplicate_sources_yield_one_numeric_value():
    a = value("thickness", "0.3", "μm")
    result = paired(
        [a, a.model_copy(update={"source_ids": ("mineru_p1_b1",)})],
        [value("thickness", "300", "nm", backend="paddleocr_vl")],
    )

    assert result.paper_row["thickness"] == 300
    assert decision(result, "thickness")["decision"] == "agree"
    assert "mineru_p1_b1" in decision(result, "thickness")["source_ids"]
    assert len([row for row in result.quality_rows if row["field"] == "thickness"]) == 1


def test_agree_chooses_highest_repeat_agreement_without_averaging():
    result = paired(
        [value("thickness", "300", "nm", agreement=0.5)],
        [value("thickness", "305", "nm", agreement=1.0, backend="paddleocr_vl")],
    )
    assert result.paper_row["thickness"] == 305


def test_conflict_and_low_confidence_matching_remain_empty():
    result = paired([value("thickness", "300", "nm")], [value("thickness", "900", "nm", backend="paddleocr_vl")])
    assert result.paper_row["thickness"] is None
    assert decision(result, "thickness")["decision"] == "conflict"
    result = paired(
        [value("thickness", "300", "nm")], [value("thickness", "300", "nm", backend="paddleocr_vl")], confidence=0.2
    )
    assert result.paper_row["thickness"] is None
    assert decision(result, "thickness")["decision"] == "ambiguous"


@pytest.mark.parametrize(
    "raw", ["10-20", "<10", ">=10", "10 (20)", "10 × 20", "10, 20", "1e-4 to 2e-4", "10 nm at 300 K"]
)
def test_lossy_numeric_interpretations_are_never_exported(raw):
    result = paired([value("thickness", raw, "nm")], [value("thickness", raw, "nm", backend="paddleocr_vl")])
    assert result.paper_row["thickness"] is None
    assert decision(result, "thickness")["decision"] == "non_scalar"


@pytest.mark.parametrize(
    ("raw", "expected"), [("~300", 300), ("300 ± 2", 300), ("1e-4 ± 2e-5", 1e-4), ("300 ± 2e-1", 300)]
)
def test_approximation_and_uncertainty_keep_documented_center(raw, expected):
    result = paired([value("thickness", raw, "nm")], [value("thickness", raw, "nm", backend="paddleocr_vl")])
    assert result.paper_row["thickness"] == expected
    assert "中心值" in decision(result, "thickness")["detail"]


def test_a_comparison_cannot_hide_different_same_condition_values_in_one_lane():
    result = paired(
        [value("thickness", "300", "nm"), value("thickness", "400", "nm")],
        [value("thickness", "300", "nm", backend="paddleocr_vl")],
    )
    assert result.paper_row["thickness"] is None
    assert decision(result, "thickness")["decision"] == "multiple_values"


def test_multiple_conditions_are_not_collapsed_even_with_identical_numbers():
    fields = [value("transmittance", "85", "%", condition=condition) for condition in ("550 nm", "600 nm")]
    result = paired(fields, [v.model_copy(update={"source_ids": ("paddleocr_vl_p0_b1",)}) for v in fields])
    assert result.paper_row["transmittance"] is None
    assert decision(result, "transmittance")["decision"] == "multiple_conditions"


def test_differently_worded_conditions_across_lanes_still_agree():
    """The lanes paraphrase one condition; only a lane's own evidence may signal several conditions."""
    result = paired(
        [value("transmittance", "85", "%", condition="ITO monolayer thickness")],
        [value("transmittance", "85", "%", condition="ITO monolayer film thickness", backend="paddleocr_vl")],
    )
    assert result.paper_row["transmittance"] == 85
    assert decision(result, "transmittance")["decision"] == "agree"
    row = decision(result, "transmittance")
    assert "ITO monolayer thickness" in row["conditions"] and "ITO monolayer film thickness" in row["conditions"]


def test_one_lane_with_two_conditions_is_still_refused():
    result = paired(
        [
            value("transmittance", "85", "%", condition="550 nm"),
            value("transmittance", "85", "%", condition="600 nm"),
        ],
        [value("transmittance", "85", "%", condition="550 nm", backend="paddleocr_vl")],
    )
    assert result.paper_row["transmittance"] is None
    assert decision(result, "transmittance")["decision"] == "multiple_conditions"


def test_one_mode_quoted_two_ways_is_one_answer_not_a_refusal():
    """A closed category set is judged on the category, so the paper's phrasing cannot manufacture a
    multiple_values refusal out of a single co-sputtering run."""
    result = paired(
        [value("mode", "DC and RF")],
        [value("mode", "DC and RF magnetron co-sputtering", backend="paddleocr_vl")],
    )

    assert decision(result, "mode")["decision"] == "agree"
    assert result.paper_row["mode"] in {"DC and RF", "DC and RF magnetron co-sputtering"}


def test_two_genuinely_different_modes_are_still_refused():
    result = paired([value("mode", "DC")], [value("mode", "RF", backend="paddleocr_vl")])

    assert result.paper_row["mode"] is None
    assert decision(result, "mode")["decision"] == "conflict"


def test_one_lane_quoting_two_spellings_of_one_mode_is_not_multiple_values():
    result = paired(
        [value("mode", "DC and RF"), value("mode", "DC and RF co-sputtering")],
        [value("mode", "DC and RF", backend="paddleocr_vl")],
    )

    assert decision(result, "mode")["decision"] != "multiple_values"
    assert result.paper_row["mode"] is not None


def test_one_lane_quoting_two_different_modes_stays_refused():
    result = paired(
        [value("mode", "DC"), value("mode", "RF")],
        [value("mode", "DC", backend="paddleocr_vl")],
    )

    assert result.paper_row["mode"] is None
    assert decision(result, "mode")["decision"] in {"conflict", "multiple_values"}


def test_different_target_compositions_cannot_be_picked_or_joined():
    result = dataset(make_lane(target=TargetRecord(fields=(value("component", "SnO2"), value("component", "ZnO")))))
    assert result.paper_row["component"] is None
    assert decision(result, "component")["decision"] == "multiple_values"


def test_unmatched_samples_with_the_same_id_stay_separate_per_backend():
    a = make_lane(samples=[make_sample("001", [value("thickness", "300", "nm")])])
    b = make_lane(
        backend="paddleocr_vl",
        samples=[make_sample("001", [value("resistivity", "1e-4", "Ω cm", backend="paddleocr_vl")])],
    )
    result = dataset(a, b)
    assert len(result.sample_rows) == 2
    assert {row["sample_id"] for row in result.sample_rows} == {"mineru:001", "paddleocr_vl:001"}
    assert sum(row["thickness"] is not None for row in result.sample_rows) == 1
    assert sum(row["resistivity"] is not None for row in result.sample_rows) == 1
    assert result.paper_row["resistivity"] is None


def test_a_failed_match_does_not_turn_into_trusted_single_source_rows():
    a = make_lane(samples=[make_sample("A", [value("thickness", "300", "nm")])])
    result = dataset(a, matching=SampleMatching(unmatched_a=("A",), failed=True, failure="model failed"))
    assert result.paper_row["thickness"] is None


@pytest.mark.parametrize("updates", [{"grounded": False}, {"source_ids": ()}])
def test_values_without_grounded_citations_stay_empty(updates):
    field = value("thickness", "300", "nm").model_copy(update=updates)
    result = dataset(make_lane(samples=[make_sample("A", [field])]))
    assert result.paper_row["thickness"] is None
    assert decision(result, "thickness")["decision"] == "ungrounded"


def test_agreement_with_one_ungrounded_side_uses_only_trusted_evidence():
    result = paired(
        [value("thickness", "300", "nm", grounded=False)], [value("thickness", "305", "nm", backend="paddleocr_vl")]
    )
    assert result.paper_row["thickness"] == 305
    assert decision(result, "thickness")["decision"] == "single_source"


def test_paper_selection_prefers_two_lane_agreement_then_stable_sample_id():
    a = make_lane(
        samples=[
            make_sample("A", [value("thickness", "300", "nm")]),
            make_sample("B", [value("thickness", "400", "nm")]),
        ]
    )
    b = make_lane(
        backend="paddleocr_vl", samples=[make_sample("B", [value("thickness", "400", "nm", backend="paddleocr_vl")])]
    )
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="B", b_id="B", confidence=1.0, method="exact", justification="same"),),
        unmatched_a=("A",),
    )
    assert dataset(a, b, matching).paper_row["sample_id"] == "B"


def test_excel_reopens_with_numeric_fields_text_ids_and_no_pdf_formulas(tmp_path):
    a = make_lane(
        samples=[
            make_sample(
                "001",
                [value("thickness", "0.3", "μm"), value("resistivity", "1e-4", "Ω cm")],
                label='=HYPERLINK("https://example.invalid")',
                conditions={"temperature": "001"},
            )
        ],
        target=TargetRecord(fields=(value("component", "=1+1"),)),
    )
    b = make_lane(backend="paddleocr_vl", samples=[make_sample("001", a.samples[0].fields)], target=a.target)
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="001", b_id="001", confidence=1.0, method="exact", justification="same"),)
    )
    result = dataset(a, b, matching, filename="=paper.pdf")
    output = tmp_path / "dataset.xlsx"
    write_dataset(
        [result, result],
        output,
        failures=[{"document_id": "b" * 64, "filename": "failed.pdf", "error": "parser failed"}],
    )
    workbook = load_workbook(output)
    assert workbook.sheetnames == ["论文数据", "样品数据", "字段说明", "数据质量", "运行记录"]
    sheet = workbook["论文数据"]
    assert sheet.max_row == 2
    assert sheet.freeze_panes == "D2"
    assert len(sheet.tables) == 1
    columns = {cell.value: cell.column for cell in sheet[1]}
    assert sheet.cell(2, columns["thickness"]).value == 300
    assert sheet.cell(2, columns["thickness"]).data_type == "n"
    assert sheet.cell(2, columns["样品ID"]).value == "001"
    assert sheet.cell(2, columns["样品ID"]).data_type == "s"
    assert sheet.cell(2, columns["component"]).value == "=1+1"
    assert sheet.cell(2, columns["component"]).data_type == "s"
    assert sheet.cell(2, columns["文件名"]).data_type == "s"
    assert sheet.cell(2, columns["density"]).value is None
    assert "E+00" in sheet.cell(2, columns["resistivity"]).number_format
    assert workbook["运行记录"].max_row == 3
    assert workbook["运行记录"].cell(3, 3).value == "failed"
    assert not list(tmp_path.glob("*.tmp"))
    assert all(cell.data_type != "f" for page in workbook for row in page for cell in row)


def test_an_empty_export_still_records_failures(tmp_path):
    output = tmp_path / "empty.xlsx"
    write_dataset([], output, failures=[{"filename": "bad.pdf", "error": "unreadable"}])
    workbook = load_workbook(output)
    assert workbook["论文数据"].max_row == 1
    assert workbook["运行记录"].cell(2, 3).value == "failed"


def test_mismatched_document_ids_are_rejected():
    a, b = make_lane(), make_lane(backend="paddleocr_vl")
    report = compare_lanes(a, b, SampleMatching())
    document = DocumentInput(document_id="b" * 64, sha256="b" * 64, pdf_path=Path("other.pdf"))
    with pytest.raises(ValueError, match="same PDF"):
        consolidate_document(document, {a.backend: a, b.backend: b}, report)


def test_the_json_view_survives_a_round_trip(tmp_path):
    # Rows are MappingProxyType, which json refuses; the web UI reads this file, so the copy must be real.
    result = paired([value("thickness", "300", "nm")], [value("thickness", "300", "nm")])

    path = tmp_path / "dataset.json"
    write_dataset_json(result, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["document_id"] == result.document_id
    assert loaded["filename"] == "paper.pdf"
    assert loaded["extractor_key"] == result.extractor_key
    assert loaded["comparison_key"] == result.comparison_key
    assert [field["name"] for field in loaded["fields"]] == [spec.name for spec in FIELD_SPECS]
    assert loaded["paper_row"] == dict(result.paper_row)
    assert loaded["sample_rows"] == [dict(row) for row in result.sample_rows]
    assert loaded["quality_rows"] == [dict(row) for row in result.quality_rows]
    assert loaded["sample_rows"][0]["thickness"] == 300
    assert not list(tmp_path.glob("*.tmp"))


def test_from_dict_reverses_as_dict_exactly():
    # The corpus export rebuilds datasets from their JSON, so the inverse has to be lossless.
    result = paired([value("thickness", "300", "nm")], [value("thickness", "300", "nm")])

    restored = DocumentDataset.from_dict(json.loads(json.dumps(result.as_dict())))

    assert restored == result
    assert restored.as_dict() == result.as_dict()


def test_the_json_field_list_carries_the_unit_and_the_scope():
    fields = {field["name"]: field for field in dataset(make_lane()).as_dict()["fields"]}

    assert fields["thickness"]["scope"] == "sample"
    assert fields["component"]["scope"] == "target"
    assert fields["thickness"]["unit"] == "nm"
