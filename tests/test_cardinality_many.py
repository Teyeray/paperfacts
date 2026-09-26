"""``cardinality: many``: a text or composition field holding several values at once (round 2 spec §2).

The loader accepts it for text and composition only; the field line asks for one entry per value; the lanes pair
as a set, so a list never reports a conflict; the dataset cell is the union of what either lane grounded, with
each element's lanes in the detail; and the workbook, the payload and the scorer read the cell as a list.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from openpyxl import load_workbook

from paperfacts.columns import field_columns
from paperfacts.compare import compare_lanes
from paperfacts.dataset import DatasetPayload, consolidate_document
from paperfacts.decide import decide_many
from paperfacts.errors import ConfigError
from paperfacts.fields import FieldRole
from paperfacts.keys import (
    ComparisonOptions,
    ExtractionOptions,
    _field_material,
    extractor_key,
    profile_extraction_fingerprint,
)
from paperfacts.kinds import LIST_NOTE, element_key
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import DocumentInput
from paperfacts.profile import DomainProfile
from paperfacts.profile_loader import parse_profile
from paperfacts.prompts import render_field_table
from paperfacts.records import FieldValue, LaneExtraction, PaperRecord
from paperfacts.workbook import write_dataset
from support.extraction import make_lane, make_sample
from support.factories import DOC_ID
from support.profiles import profile_data
from test_score import score

TECHNIQUES = ["XRD", "XPS", "TEM", "SEM"]
LIST_FIELD = {
    "name": "characterization_techniques",
    "group": "precursor",
    "kind": "text",
    "cardinality": "many",
    "categories": TECHNIQUES,
    "description": "Each characterization technique the paper applies to its coatings.",
    "keywords": ["characterization"],
    "label": "表征手段",
}


def list_profile(**extra: object) -> DomainProfile:
    """The demo profile with a paper-level categorical list and its sample-level ``solvent`` made a plain list."""
    data = profile_data({"fields.2.cardinality": "many"})
    data["fields"].append(LIST_FIELD | extra)
    return parse_profile(data, Path("profiles/demo.json"))


PROFILE = list_profile()
SPEC = PROFILE.by_name["characterization_techniques"]
SOLVENT = PROFILE.by_name["solvent"]


def entry(**changes: object) -> dict[str, object]:
    return {"name": "f", "group": "coating", "kind": "text", "description": "A list.", "cardinality": "many"} | changes


def load_field(field: dict[str, object]):
    data = profile_data()
    data["fields"].append(field)
    return parse_profile(data, Path("profiles/demo.json")).by_name[str(field["name"])]


# ---- the loader --------------------------------------------------------------------------------------------------


def test_a_list_field_loads_with_its_categories_as_prompt_categories():
    assert SPEC.cardinality == "many"
    assert SPEC.prompt_categories == tuple(TECHNIQUES)
    assert SOLVENT.cardinality == "many" and SOLVENT.prompt_categories == ()
    assert load_field(entry(kind="composition")).cardinality == "many"


def test_a_single_valued_field_with_categories_derives_no_prompt_categories():
    spec = load_field(entry(cardinality="one", categories=["DC", "RF"]))

    assert spec.cardinality == "one" and spec.prompt_categories == ()
    unstated = {key: value for key, value in entry(categories=["DC", "RF"]).items() if key != "cardinality"}
    assert load_field(unstated).cardinality == "one" and load_field(unstated).prompt_categories == ()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param({"kind": "numeric", "canonical_unit": "nm"}, "needs a text or composition field", id="numeric"),
        pytest.param({"figure_readable": True}, "cannot be combined with figure_readable", id="figure_readable"),
        pytest.param({"figure_readable": False}, "cannot be combined with figure_readable", id="figure_readable-false"),
        pytest.param({"condition_preference": ["550"]}, "condition_preference", id="condition_preference"),
        pytest.param(
            {"condition_hint": "the step", "condition_rule": "the step", "missing_condition_note_zh": "未注明"},
            "cannot be combined with condition_rule",
            id="condition_rule",
        ),
        pytest.param({"rel_tol": 0.1}, "cannot be combined with rel_tol", id="rel_tol"),
        pytest.param({"canonical_unit": "nm"}, "cannot be combined with canonical_unit", id="canonical_unit"),
        pytest.param({"valid_range": {"min": 0}}, "cannot be combined with valid_range", id="valid_range"),
        pytest.param({"cardinality": "several"}, "cardinality must be one of one, many", id="unknown"),
        pytest.param({"prompt_categories": ["XRD"]}, "unknown key(s) prompt_categories", id="derived-not-written"),
    ],
)
def test_the_loader_refuses_a_list_it_cannot_honour(changes, message):
    with pytest.raises(ConfigError, match=message.replace("(", r"\(").replace(")", r"\)")):
        load_field(entry(**changes))


# ---- the prompt and the keys -------------------------------------------------------------------------------------


def test_the_field_line_asks_for_each_value_and_names_the_categories():
    (line,) = render_field_table([SPEC], "another quantity").splitlines()
    (solvent,) = render_field_table([SOLVENT], "another quantity").splitlines()

    assert line.endswith(f"its coatings. {LIST_NOTE} Name each with one of: XRD, XPS, TEM, SEM.")
    assert solvent.endswith(f"made in. {LIST_NOTE}")


def test_a_single_valued_field_line_is_unchanged():
    single = dataclasses.replace(SPEC, cardinality="one", prompt_categories=())

    assert render_field_table([single], "x").endswith("applies to its coatings.")


def test_cardinality_and_prompt_categories_reach_the_extraction_key_only_off_their_defaults():
    single = dataclasses.replace(SPEC, cardinality="one", prompt_categories=())
    material = _field_material(SPEC, FieldRole.PROMPT, FieldRole.CLEANING)

    assert material["cardinality"] == "many" and material["prompt_categories"] == tuple(TECHNIQUES)
    assert not {"cardinality", "prompt_categories"} & set(_field_material(single, FieldRole.PROMPT))
    # A list's categories are told to the model, so editing them re-keys extraction.
    edited = list_profile(categories=[*TECHNIQUES, "BET"])
    options = {"model": "m", "mode": "passage"}
    assert extractor_key(ExtractionOptions(profile=edited, **options)) != extractor_key(
        ExtractionOptions(profile=PROFILE, **options)
    )


# ---- comparison: pairing as a set --------------------------------------------------------------------------------


def lane(backend, *, paper=(), solvent=(), unattributed=()) -> LaneExtraction:
    def values(field, raws):
        return [
            FieldValue(field=field, value_raw=raw, source_ids=(f"{backend}_p0_b{index}",))
            for index, raw in enumerate(raws)
        ]

    return make_lane(
        backend=backend,
        paper=PaperRecord(fields=tuple(values(SPEC.name, paper))) if paper else None,
        samples=[make_sample("A", values("solvent", solvent))],
        unattributed=values("solvent", unattributed),
    ).model_copy(update={"profile_fingerprint": profile_extraction_fingerprint(PROFILE)})


OPTIONS = ComparisonOptions(profile=PROFILE, ambiguous_match_confidence=0.6)
MATCHING = SampleMatching(
    pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, justification="same", method="exact"),)
)


def report(a: LaneExtraction, b: LaneExtraction):
    return compare_lanes(a, b, MATCHING, OPTIONS)


def rows(result, field):
    found = [
        (c.status, c.a.value_raw if c.a else None, c.b.value_raw if c.b else None)
        for c in result.comparisons
        if c.field == field
    ]
    return sorted(found, key=str)


def test_two_lanes_reading_the_same_list_in_another_order_agree_element_by_element():
    result = report(lane("mineru", solvent=["water", "ethanol"]), lane("paddleocr_vl", solvent=["Ethanol", "water"]))

    assert rows(result, "solvent") == [("agree", "ethanol", "Ethanol"), ("agree", "water", "water")]


def test_a_partial_overlap_is_agreement_plus_one_sided_missing_never_a_conflict():
    result = report(
        lane("mineru", paper=["XRD", "X-ray photoelectron spectroscopy (XPS)"]), lane("paddleocr_vl", paper=["XPS"])
    )

    assert rows(result, SPEC.name) == [
        ("agree", "X-ray photoelectron spectroscopy (XPS)", "XPS"),
        ("missing", "XRD", None),
    ]


def test_disjoint_lists_are_missing_on_each_side_never_a_conflict():
    # Positional pairing would have called LiOH against NiSO4 a conflict.
    result = report(lane("mineru", solvent=["LiOH"]), lane("paddleocr_vl", solvent=["NiSO4"]))

    assert rows(result, "solvent") == [("missing", "LiOH", None), ("missing", None, "NiSO4")]
    assert result.counts.conflict == 0


def test_elements_differing_only_in_a_greek_letter_or_a_compositions_case_stay_two():
    # same_text drops Greek letters and folds case: in a union that would silently lose one of the two.
    phases = report(lane("mineru", solvent=["α-Al2O3"]), lane("paddleocr_vl", solvent=["γ-Al2O3"]))
    oxide = load_field(entry(kind="composition"))

    assert rows(phases, "solvent") == [("missing", "α-Al2O3", None), ("missing", None, "γ-Al2O3")]
    assert element_key(oxide, "Co3O4") != element_key(oxide, "CO3O4")
    assert element_key(SOLVENT, " Ethanol ") == element_key(SOLVENT, "ethanol")


def test_unplaced_list_values_are_not_paired_by_position():
    result = report(lane("mineru", unattributed=["water"]), lane("paddleocr_vl", unattributed=["toluene"]))

    assert [c for c in result.comparisons if c.scope == "unattributed"] == []


# ---- the dataset cell --------------------------------------------------------------------------------------------


def cited(backend: str, raw: str, *, grounded: bool = True, index: int = 1) -> tuple[str, FieldValue]:
    return backend, FieldValue(
        field=SPEC.name, value_raw=raw, source_ids=(f"{backend}_p0_b{index}",), grounded=grounded
    )


def cell(evidence, comparisons=None, **kwargs):
    if comparisons is None:
        a = [value for backend, value in evidence if backend == "mineru"]
        b = [value for backend, value in evidence if backend == "paddleocr_vl"]
        comparisons = report(
            lane("mineru", paper=[v.value_raw for v in a]), lane("paddleocr_vl", paper=[v.value_raw for v in b])
        ).comparisons
    return decide_many(SPEC, evidence, comparisons, **kwargs)


def test_the_cell_is_the_union_in_category_order_with_each_elements_lanes():
    decision = cell([cited("mineru", "TEM"), cited("mineru", "X-ray diffraction (XRD)"), cited("paddleocr_vl", "XRD")])

    assert decision.value == ["XRD", "TEM"]
    assert decision.status == "single_source"
    assert "XRD（mineru, paddleocr_vl）" in decision.detail and "TEM（mineru）" in decision.detail
    assert decision.lanes == ("mineru", "paddleocr_vl")
    assert decision.sources == "mineru_p0_b1; paddleocr_vl_p0_b1"


def test_the_cell_agrees_when_both_lanes_hold_every_element():
    decision = cell(
        [cited("mineru", "XPS"), cited("mineru", "XRD"), cited("paddleocr_vl", "XRD"), cited("paddleocr_vl", "XPS")]
    )

    assert (decision.value, decision.status) == (["XRD", "XPS"], "agree")


def test_a_quote_naming_two_categories_is_refused_as_an_element():
    decision = cell([cited("mineru", "XRD and XPS"), cited("mineru", "SEM"), cited("paddleocr_vl", "SEM")])

    assert (decision.value, decision.status) == (["SEM"], "agree")
    assert "XRD and XPS" in decision.detail and "未对应唯一类别" in decision.detail


def test_a_cell_left_with_no_element_is_non_scalar():
    decision = cell([cited("mineru", "XRD and XPS"), cited("paddleocr_vl", "XRD and XPS")])

    assert (decision.value, decision.status) == (None, "non_scalar")


def test_an_ungrounded_value_is_no_element():
    decision = cell([cited("mineru", "XRD"), cited("paddleocr_vl", "TEM", grounded=False)])

    assert (decision.value, decision.status) == (["XRD"], "single_source")
    assert "已排除未定位" in decision.detail


def test_without_categories_elements_keep_first_seen_order_mineru_first_and_one_spelling():
    spec = SOLVENT
    evidence = [
        ("paddleocr_vl", FieldValue(field="solvent", value_raw="toluene", source_ids=("paddleocr_vl_p0_b1",))),
        ("mineru", FieldValue(field="solvent", value_raw="ethanol", source_ids=("mineru_p0_b1",))),
        ("mineru", FieldValue(field="solvent", value_raw="  Toluene ", source_ids=("mineru_p0_b2",))),
    ]
    comparisons = report(
        lane("mineru", solvent=["ethanol", "Toluene"]), lane("paddleocr_vl", solvent=["toluene"])
    ).comparisons
    decision = decide_many(spec, evidence, [c for c in comparisons if c.field == "solvent"])

    assert decision.value == ["ethanol", "Toluene"]
    assert decision.status == "single_source"


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"unanswered": True, "blocked": "样品匹配失败"}, "unanswered"),
        ({"blocked": "样品匹配失败"}, "ambiguous"),
    ],
)
def test_the_refusals_are_decides_in_its_order(kwargs, status):
    decision = cell([cited("mineru", "XRD")], **kwargs)

    assert (decision.value, decision.status) == (None, status)


def test_no_evidence_is_missing_and_no_comparison_is_unreviewed():
    assert decide_many(SPEC, [], []).status == "missing"
    assert cell([cited("mineru", "XRD")], comparisons=[]).status == "unreviewed"


# ---- the dataset, payload, workbook and web columns --------------------------------------------------------------


def consolidated(a: LaneExtraction, b: LaneExtraction):
    document = DocumentInput(document_id=DOC_ID, sha256=DOC_ID, pdf_path=Path("paper.pdf"))
    return consolidate_document(document, {a.backend: a, b.backend: b}, report(a, b), OPTIONS)


A = lane("mineru", paper=["XRD", "TEM"], solvent=["water", "ethanol"])
B = lane("paddleocr_vl", paper=["XRD"], solvent=["water"])


def test_a_list_fills_its_dataset_cell_and_counts_as_one_available_field():
    result = consolidated(A, B)
    (row,) = result.sample_rows

    assert row["solvent"] == ["water", "ethanol"]
    assert row[SPEC.name] == ["XRD", "TEM"]
    # coating_thickness and precursor_purity are empty: the two lists are two fields.
    assert row["available_fields"] == 2
    quality = {(q["sample_id"], q["field"]): q for q in result.quality_rows}
    assert quality[("paper", SPEC.name)]["value"] == ["XRD", "TEM"]
    assert quality[("paper", SPEC.name)]["decision"] == "single_source"


def test_a_list_cell_survives_the_payload_round_trip():
    payload = consolidated(A, B).to_payload()

    assert DatasetPayload.model_validate_json(payload.model_dump_json()).sample_rows[0]["solvent"] == [
        "water",
        "ethanol",
    ]


def test_the_columns_carry_the_cardinality():
    columns = {column.name: column for column in field_columns(PROFILE)}

    assert columns["solvent"].cardinality == "many" and columns[SPEC.name].cardinality == "many"
    assert columns["coating_thickness"].cardinality == "one"


def test_the_workbook_joins_a_list_and_marks_its_column(tmp_path):
    path = tmp_path / "demo.xlsx"
    write_dataset([consolidated(A, B)], path, PROFILE)
    workbook = load_workbook(path)

    samples = list(workbook["样品数据"].values)
    assert samples[1][samples[0].index("solvent")] == "water; ethanol"
    quality = list(workbook["数据质量"].values)
    value = quality[0].index("输出值")
    assert "XRD; TEM" in [line[value] for line in quality[1:]]
    fields = {line[0]: line for line in workbook["字段说明"].values}
    assert fields["solvent"][3] == "文本（多值）" and fields["coating_thickness"][3] == "nm"


# ---- the scorer --------------------------------------------------------------------------------------------------


def test_the_scorer_scores_a_list_per_element():
    gold = [{"value": "XRD"}, {"value": "XPS"}, {"value": "TEM", "ambiguous": True}]

    outcomes = score.outcomes(SPEC, ["XRD", "TEM", "SEM"], gold, False)

    assert sorted(outcomes, key=str) == sorted(
        [("correct", "XRD"), ("soft", "TEM"), ("extra", "SEM"), ("missing", None)], key=str
    )
    assert score.outcomes(SPEC, None, gold, False) == [("missing", None), ("missing", None)]
    assert score.outcomes(SPEC, ["XRD"], [], False) == [("extra", "XRD")]
    assert score.outcomes(SPEC, ["XRD"], [{"value": "TEM", "ambiguous": True}], False) == [("disputed", "XRD")]
    # One to one: two elements matching the one required cell find it once.
    both = score.outcomes(SOLVENT, ["water", "Water "], [{"value": "water"}], False)
    assert both == [("correct", "water"), ("extra", "Water ")]


def test_a_single_valued_field_still_scores_one_cell():
    single = dataclasses.replace(SPEC, cardinality="one")

    assert score.outcomes(single, "XRD", [{"value": "XRD"}, {"value": "XPS"}], False) == [("correct", "XRD")]


def test_a_list_cell_matches_a_gold_value_for_alignment_when_one_element_does():
    assert score.value_matches(SOLVENT, ["water", "ethanol"], {"value": "Ethanol"})
    assert not score.value_matches(SOLVENT, ["water"], {"value": "ethanol"})


def test_a_union_keeps_two_elements_differing_only_in_a_greek_letter():
    evidence = [
        ("mineru", FieldValue(field="solvent", value_raw="α-Al2O3", source_ids=("mineru_p0_b1",))),
        ("paddleocr_vl", FieldValue(field="solvent", value_raw="γ-Al2O3", source_ids=("paddleocr_vl_p0_b1",))),
    ]
    comparisons = report(lane("mineru", solvent=["α-Al2O3"]), lane("paddleocr_vl", solvent=["γ-Al2O3"])).comparisons

    decision = decide_many(SOLVENT, evidence, [c for c in comparisons if c.field == "solvent"])

    assert decision.value == ["α-Al2O3", "γ-Al2O3"] and decision.status == "single_source"
