"""Deterministic candidate coverage and budgets, using synthetic evidence only."""

from __future__ import annotations

import importlib

import pytest

from paperfacts.compare import ComparisonReport, FieldComparison
from paperfacts.matching import SampleMatching
from paperfacts.models import NormalizedBBox
from paperfacts.pdf import NativePage
from support.extraction import make_artifact, make_field, make_lane
from support.factories import DOC_ID, make_block
from support.profiles import make_profile

FULL_PAGE = NormalizedBBox(x1=0, y1=0, x2=1, y2=1)
TABLE_BOX = NormalizedBBox(x1=0.1, y1=0.2, x2=0.8, y2=0.5)
PADDED_TABLE_BOX = NormalizedBBox(x1=0.09, y1=0.19, x2=0.81, y2=0.51)


def assert_box(actual: NormalizedBBox, expected: NormalizedBBox):
    assert actual.model_dump() == pytest.approx(expected.model_dump())


def report(*comparisons):
    return ComparisonReport(
        document_id=DOC_ID,
        extractor_key="synthetic",
        comparison_key="synthetic",
        backend_a="mineru",
        backend_b="paddleocr_vl",
        matchings={"sample": SampleMatching()},
        comparisons=comparisons,
    )


def select(pages, *, blocks=(), comparisons=(), lanes=(), limit=4, strategy="balanced"):
    candidates = importlib.import_module("paperfacts.visual_candidates")
    return candidates.select_candidates(
        pages=pages,
        artifacts={"mineru": make_artifact(blocks)},
        comparison=report(*comparisons),
        lanes=lanes,
        profile=make_profile(),
        limit=limit,
        strategy=strategy,
    )


def conflict(block, *, status="conflict", grounded=True):
    value = make_field("coating_thickness", "42", source_ids=(block.source_id,))
    return FieldComparison(
        scope="sample:A|A",
        field="coating_thickness",
        status=status,
        a=value.model_copy(update={"grounded": grounded}),
    )


def test_empty_lanes_still_find_native_table_and_keyword_pages():
    result = select((NativePage(0, "Table 2. Coating thickness", ()), NativePage(1, "Introduction", ())))

    assert [candidate.page for candidate in result.candidates] == [0]
    assert result.candidates[0].bbox == FULL_PAGE
    assert result.candidates[0].sources == ("independent",)
    assert result.candidates[0].selected_by == "independent"
    assert result.candidates[0].fields == ("coating_thickness",)
    assert result.unselected_pages == (1,)


def test_a_page_already_cited_by_both_lanes_remains_an_independent_candidate():
    block = make_block(type="table", bbox=TABLE_BOX, content="plain numbers")
    comparison = conflict(block, status="agree")
    comparison = comparison.model_copy(update={"b": comparison.a})

    result = select(
        (NativePage(0, "Table 2. Another thickness measurement", ()),), blocks=(block,), comparisons=(comparison,)
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].sources == ("independent",)
    assert result.candidates[0].bbox == FULL_PAGE


def test_risk_overflow_cannot_take_the_independent_half_of_the_budget():
    blocks = tuple(make_block(page=page, type="table", bbox=TABLE_BOX) for page in range(5))
    pages = tuple(NativePage(page, "Background" if page < 5 else "Thickness", ()) for page in range(7))

    result = select(pages, blocks=blocks, comparisons=tuple(conflict(block) for block in blocks))

    assert [(candidate.page, candidate.selected_by) for candidate in result.candidates] == [
        (0, "risk"),
        (5, "independent"),
        (1, "risk"),
        (6, "independent"),
    ]
    assert result.unselected_pages == (2, 3, 4)
    assert any("candidate_limit" in issue for issue in result.issues)


def test_duplicate_risk_and_independent_region_uses_one_slot_and_keeps_zoom_box():
    block = make_block(page=0, type="table", bbox=TABLE_BOX)
    pages = tuple(NativePage(page, "Thickness", ()) for page in range(4))

    result = select(pages, blocks=(block,), comparisons=(conflict(block),))

    assert [candidate.page for candidate in result.candidates] == [0, 1, 2, 3]
    assert len({(candidate.page, candidate.bbox) for candidate in result.candidates}) == 4
    merged = result.candidates[0]
    assert merged.sources == ("risk", "independent")
    assert merged.selected_by == "risk"
    assert merged.bbox == FULL_PAGE
    assert len(merged.zoom_boxes) == 1
    assert_box(merged.zoom_boxes[0], PADDED_TABLE_BOX)
    assert merged.source_ids == (block.source_id,)


@pytest.mark.parametrize("risk_count,independent_count", [(4, 0), (0, 4), (1, 3), (3, 1)])
def test_an_exhausted_pool_lends_unused_slots(risk_count, independent_count):
    blocks = tuple(make_block(page=page, type="table", bbox=TABLE_BOX) for page in range(risk_count))
    pages = tuple(
        NativePage(page, "Background" if page < risk_count else "Thickness", ())
        for page in range(risk_count + independent_count)
    )

    result = select(pages, blocks=blocks, comparisons=tuple(conflict(block) for block in blocks))

    assert len(result.candidates) == 4
    assert sum(candidate.selected_by == "risk" for candidate in result.candidates) == risk_count


@pytest.mark.parametrize("content", ["Thickness (×10⁻³ nm)", "Thickness<sup>a</sup>; a: after treatment"])
def test_an_agreed_table_with_multiplier_or_footnote_is_a_risk(content):
    block = make_block(type="table", bbox=TABLE_BOX, content=content)
    comparison = conflict(block, status="agree")
    comparison = comparison.model_copy(update={"b": comparison.a})

    result = select((NativePage(0, "Background", ()),), blocks=(block,), comparisons=(comparison,))

    assert len(result.candidates) == 1
    assert "complex_table" in result.candidates[0].reasons
    assert result.candidates[0].selected_by == "risk"


def test_scanned_page_stays_visible_as_a_candidate_and_coverage_issue():
    result = select((NativePage(0, "  \n", ()), NativePage(1, "Background", ())))

    assert [candidate.page for candidate in result.candidates] == [0]
    assert "no_native_text" in result.candidates[0].reasons
    assert any("no_native_text" in issue for issue in result.issues)
    assert result.unselected_pages == (1,)


def test_multpage_citations_keep_each_page_and_explicitly_report_missing_context():
    blocks = tuple(make_block(page=page, type="table", bbox=TABLE_BOX) for page in (0, 1))
    value = make_field("coating_thickness", "42", source_ids=tuple(block.source_id for block in blocks))
    comparison = FieldComparison(scope="sample:A|A", field=value.field, status="conflict", a=value)

    result = select(
        tuple(NativePage(page, "Background", ()) for page in (0, 1)), blocks=blocks, comparisons=(comparison,)
    )

    assert [candidate.page for candidate in result.candidates] == [0, 1]
    assert all("multipage_source" in candidate.reasons for candidate in result.candidates)
    assert all(candidate.bbox == FULL_PAGE and candidate.zoom_boxes == () for candidate in result.candidates)
    assert any("multipage_source" in issue for issue in result.issues)


def test_missing_and_out_of_range_sources_never_invent_regions_or_consume_slots():
    block = make_block(page=99, type="table", bbox=TABLE_BOX)
    missing = FieldComparison(
        scope="sample:A|A",
        field="coating_thickness",
        status="ambiguous",
        a=make_field("coating_thickness", "42", source_ids=("missing_id",)),
    )

    result = select((NativePage(0, "Thickness", ()),), blocks=(block,), comparisons=(conflict(block), missing))

    assert [candidate.page for candidate in result.candidates] == [0]
    assert result.candidates[0].selected_by == "independent"
    assert any("missing_source" in issue for issue in result.issues)
    assert any("invalid_source_page" in issue for issue in result.issues)


def test_risk_priorities_precede_page_order_and_input_permutations_are_deterministic():
    blocks = tuple(make_block(page=page, type="table", bbox=TABLE_BOX) for page in range(3))
    comparisons = (conflict(blocks[0], status="missing"), conflict(blocks[2]), conflict(blocks[1], status="ambiguous"))
    pages = tuple(NativePage(page, "Background", ()) for page in range(3))

    forward = select(pages, blocks=blocks, comparisons=comparisons)
    backward = select(pages[::-1], blocks=blocks[::-1], comparisons=comparisons[::-1])

    assert forward == backward
    assert [candidate.page for candidate in forward.candidates] == [1, 2, 0]


def test_overlapping_regions_merge_transitively_but_never_across_pages():
    boxes = (
        NormalizedBBox(x1=0.1, y1=0.1, x2=0.4, y2=0.4),
        NormalizedBBox(x1=0.3, y1=0.3, x2=0.6, y2=0.6),
        NormalizedBBox(x1=0.5, y1=0.5, x2=0.8, y2=0.8),
    )
    blocks = tuple(make_block(page=0, order=index, type="table", bbox=box) for index, box in enumerate(boxes))
    other = make_block(page=1, type="table", bbox=boxes[0])
    all_blocks = (*blocks, other)

    result = select(
        (NativePage(0, "Background", ()), NativePage(1, "Background", ())),
        blocks=all_blocks,
        comparisons=tuple(conflict(block) for block in all_blocks),
    )

    assert len(result.candidates) == 2
    assert_box(result.candidates[0].bbox, NormalizedBBox(x1=0.09, y1=0.09, x2=0.81, y2=0.81))
    assert len(result.candidates[0].zoom_boxes) == 3
    assert result.candidates[1].page == 1


def test_caption_only_source_uses_whole_page_without_guessing_a_figure_box():
    caption = make_block(type="caption", content="Fig. 1", bbox=TABLE_BOX)

    result = select((NativePage(0, "Background", ()),), blocks=(caption,), comparisons=(conflict(caption),))

    assert result.candidates[0].bbox == FULL_PAGE
    assert result.candidates[0].zoom_boxes == ()


def test_ungrounded_unattributed_lane_value_can_trigger_risk_without_a_comparison():
    block = make_block(type="table", bbox=TABLE_BOX)
    value = make_field("coating_thickness", "42", source_ids=(block.source_id,)).model_copy(update={"grounded": False})
    lane = make_lane(unattributed=(value,))

    result = select((NativePage(0, "Background", ()),), blocks=(block,), lanes=(lane,))

    assert result.candidates[0].reasons == ("ungrounded",)


def test_risk_only_strategy_does_not_promote_independent_regions():
    block = make_block(type="table", bbox=TABLE_BOX)
    result = select(
        (NativePage(0, "Thickness", ()), NativePage(1, "Thickness", ())),
        blocks=(block,),
        comparisons=(conflict(block),),
        strategy="risk_only",
    )

    assert result.strategy == "risk_only"
    assert [candidate.page for candidate in result.candidates] == [0]
    assert_box(result.candidates[0].bbox, PADDED_TABLE_BOX)
    assert result.candidates[0].sources == ("risk",)
    assert result.unselected_pages == (1,)


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="limit"):
        select((NativePage(0, "Thickness", ()),), limit=limit)


def test_zero_limit_preserves_explicit_unselected_page_coverage():
    result = select((NativePage(0, "Thickness", ()),), limit=0)

    assert result.candidates == ()
    assert result.unselected_pages == (0,)
    assert any("candidate_limit" in issue for issue in result.issues)


@pytest.mark.parametrize("text", ["Fig.1 shows the result", "Annealed at elevated temperature"])
def test_caption_spacing_and_profile_condition_keywords_can_locate_independent_pages(text):
    result = select((NativePage(0, text, ()),))

    assert [candidate.page for candidate in result.candidates] == [0]
    assert result.candidates[0].selected_by == "independent"


def test_partial_missing_reference_marks_the_located_region_as_incomplete():
    block = make_block(type="table", bbox=TABLE_BOX)
    value = make_field("coating_thickness", "42", source_ids=(block.source_id, "missing_other_page"))
    comparison = FieldComparison(scope="sample:A|A", field=value.field, status="conflict", a=value)

    result = select((NativePage(0, "Background", ()),), blocks=(block,), comparisons=(comparison,))

    assert "incomplete_source" in result.candidates[0].reasons
    assert result.candidates[0].bbox == FULL_PAGE
    assert result.candidates[0].zoom_boxes == ()
    assert any("missing_source" in issue for issue in result.issues)


def test_incompatible_document_evidence_is_refused_before_selection():
    candidates = importlib.import_module("paperfacts.visual_candidates")
    artifact = make_artifact(document_id="b" * 64)

    with pytest.raises(ValueError, match="document"):
        candidates.select_candidates(
            pages=(NativePage(0, "Thickness", ()),),
            artifacts={"mineru": artifact},
            comparison=report(),
            lanes=(),
            profile=make_profile(),
        )


def test_continued_table_does_not_offer_a_fragment_as_a_complete_zoom_box():
    block = make_block(type="table", bbox=TABLE_BOX, content="Table 1 (continued)")

    result = select((NativePage(0, "Background", ()),), blocks=(block,), comparisons=(conflict(block),))

    assert result.candidates[0].bbox == FULL_PAGE
    assert result.candidates[0].zoom_boxes == ()
    assert "incomplete_source" in result.candidates[0].reasons


def test_odd_budget_reserves_its_extra_slot_for_independent_discovery():
    blocks = tuple(make_block(page=page, type="table", bbox=TABLE_BOX) for page in range(2))
    pages = tuple(NativePage(page, "Background" if page < 2 else "Thickness", ()) for page in range(4))

    result = select(pages, blocks=blocks, comparisons=tuple(conflict(block) for block in blocks), limit=3)

    assert [(candidate.page, candidate.selected_by) for candidate in result.candidates] == [
        (0, "risk"),
        (2, "independent"),
        (3, "independent"),
    ]


def test_disjoint_tables_on_the_same_page_remain_separate_candidates():
    first = make_block(page=0, order=0, type="table", bbox=TABLE_BOX)
    first_note = make_block(page=0, order=1, bbox=NormalizedBBox(x1=0.1, y1=0.52, x2=0.8, y2=0.54))
    second_caption = make_block(page=0, order=2, bbox=NormalizedBBox(x1=0.1, y1=0.64, x2=0.8, y2=0.66))
    second = make_block(page=0, order=3, type="table", bbox=NormalizedBBox(x1=0.1, y1=0.7, x2=0.8, y2=0.9))

    result = select(
        (NativePage(0, "Background", ()),),
        blocks=(first, first_note, second_caption, second),
        comparisons=(conflict(first), conflict(second)),
        limit=1,
    )

    assert len(result.candidates) == 1
    assert_box(result.candidates[0].bbox, NormalizedBBox(x1=0.09, y1=0.19, x2=0.81, y2=0.55))
    assert result.candidates[0].source_ids == (first.source_id,)
    assert any("candidate_limit" in issue for issue in result.issues)


def test_visual_crop_keeps_external_caption_and_footnote_even_if_blocks_arrive_unsorted():
    caption = make_block(
        order=0, type="caption", bbox=NormalizedBBox(x1=0.1, y1=0.1, x2=0.8, y2=0.15), content="Table 1"
    )
    table = make_block(order=1, type="table", bbox=TABLE_BOX)
    footnote = make_block(
        order=2, type="text", bbox=NormalizedBBox(x1=0.1, y1=0.55, x2=0.8, y2=0.6), content="a: treated sample"
    )

    result = select(
        (NativePage(0, "Background", ()),), blocks=(footnote, table, caption), comparisons=(conflict(table),)
    )

    candidate = result.candidates[0]
    assert_box(candidate.bbox, NormalizedBBox(x1=0.09, y1=0.09, x2=0.81, y2=0.61))
    assert candidate.zoom_boxes == (candidate.bbox,)
    assert candidate.context_ids == (caption.source_id, footnote.source_id)


def test_continuation_notice_in_neighbour_context_prevents_a_partial_zoom():
    table = make_block(order=0, type="table", bbox=TABLE_BOX)
    caption = make_block(order=1, type="caption", content="Table 1 (continued)")

    result = select((NativePage(0, "Background", ()),), blocks=(table, caption), comparisons=(conflict(table),))

    assert result.candidates[0].bbox == FULL_PAGE
    assert result.candidates[0].zoom_boxes == ()
    assert "incomplete_source" in result.candidates[0].reasons
