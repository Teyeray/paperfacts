"""The visual validation stage: which values are checked, what the model is shown, and how its reading
becomes a verdict.

Everything runs on :class:`support.llm.FakeVisionClient` and a blank PDF generated on the fly; no model,
no network. The three rules the module docstring states are each pinned by a test: the prompt never
carries the value, the selection is lane-blind, and the verdict comes from grounding's matcher.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paperfacts.compare import compare_lanes
from paperfacts.errors import LlmError
from paperfacts.grounding import ground_lane
from paperfacts.matching import SampleMatch, SampleMatching
from paperfacts.models import NormalizedBBox, ParsedArtifact
from paperfacts.normalize import normalize_lane
from paperfacts.records import TargetRecord
from paperfacts.storage import DataLayout
from paperfacts.validate import (
    FILL_SOURCE_PREFIX,
    CropStore,
    Reading,
    Region,
    ValidationReport,
    adjudicate,
    cites_table,
    fill_from_tables,
    missing_fields,
    owner_of,
    parse_reading,
    region_for,
    region_of_blocks,
    select_targets,
    table_regions_of,
    validate_lanes,
    value_key,
)
from support.extraction import DEFAULT_EXTRACTOR_KEY, make_artifact, make_field, make_lane, make_sample
from support.factories import DOC_ID, make_blank_pdf, make_block
from support.llm import FakeLlmClient, FakeVisionClient, VisionCall

BOX_A = NormalizedBBox(x1=0.1, y1=0.1, x2=0.9, y2=0.3)
BOX_B = NormalizedBBox(x1=0.1, y1=0.4, x2=0.9, y2=0.6)
BOX_C = NormalizedBBox(x1=0.1, y1=0.65, x2=0.9, y2=0.75)
BOX_D = NormalizedBBox(x1=0.1, y1=0.8, x2=0.9, y2=0.9)


def reading(transcription: str, legible: bool = True) -> str:
    return json.dumps({"transcription": transcription, "legible": legible}, ensure_ascii=False)


def exact_match() -> SampleMatching:
    return SampleMatching(pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, justification="", method="exact"),))


def artifact_for(backend: str) -> ParsedArtifact:
    """Two blocks per lane, both on page 0: a paragraph and a table, each with its own box."""
    return make_artifact(
        [
            make_block(page=0, order=0, backend=backend, content="Sample A was deposited at 150 W.", bbox=BOX_A),
            make_block(
                page=0,
                order=1,
                type="table",
                backend=backend,
                content="<table><tr><td>Rs</td><td>12.5 Ω/sq</td></tr></table>",
                bbox=BOX_B,
            ),
        ],
        backend=backend,
    )


def lanes_and_report(a_value: str, b_value: str, *, field: str = "sheet_resistance", unit: str = "Ω/sq"):
    """Two lanes with one matched sample carrying one value each, and the comparison report between them.

    Grounding is applied against each lane's own artifact, exactly as the pipeline does, so the values'
    ``grounded`` flags are real.
    """
    artifacts = {"mineru": artifact_for("mineru"), "paddleocr_vl": artifact_for("paddleocr_vl")}
    lanes = {}
    for backend, raw in (("mineru", a_value), ("paddleocr_vl", b_value)):
        lane = make_lane(
            backend=backend,
            samples=[make_sample("A", [make_field(field, raw, unit_raw=unit, source_ids=[f"{backend}_p0_b1"])])],
        )
        blocks = {block.source_id: block.content for block in artifacts[backend].blocks}
        lanes[backend] = normalize_lane(ground_lane(lane, blocks))
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, justification="", method="exact"),)
    )
    report = compare_lanes(lanes["mineru"], lanes["paddleocr_vl"], matching)
    return lanes, report, artifacts


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    return make_blank_pdf(tmp_path / "paper.pdf", sizes=[(595.0, 842.0)])


@pytest.fixture
def layout(tmp_path: Path) -> DataLayout:
    return DataLayout(tmp_path / "data")


# ---- value_key and owner_of --------------------------------------------------------------------------------


def test_the_value_key_is_backend_owner_field_quote_condition_and_citations():
    value = make_field(
        "thickness", "250", unit_raw="nm", condition="at 550 nm", source_ids=["mineru_p0_b1", "mineru_p0_b0"]
    )
    assert value_key("mineru", "sample:A", value) == "mineru|sample:A|thickness|250|at 550 nm|mineru_p0_b1,mineru_p0_b0"


def test_the_key_ignores_what_normalisation_and_grounding_fill_in():
    stored = make_field("thickness", "250", unit_raw="nm", source_ids=["mineru_p0_b1"])
    derived = stored.model_copy(update={"value": 250.0, "unit": "nm", "grounded": False, "agreement": 0.5})
    assert value_key("mineru", "sample:A", stored) == value_key("mineru", "sample:A", derived)


def test_owner_of_finds_target_sample_and_unattributed_values():
    target_value = make_field("component", "SnO2:Ta", source_ids=["mineru_p0_b0"])
    sample_value = make_field("thickness", "250", unit_raw="nm", source_ids=["mineru_p0_b1"])
    stray = make_field("transmittance", "> 80", unit_raw="%", source_ids=["mineru_p0_b0"])
    lane = make_lane(
        target=TargetRecord(fields=(target_value,)),
        samples=[make_sample("S1", [sample_value])],
        unattributed=[stray],
    )

    assert owner_of(lane, target_value) == "target"
    assert owner_of(lane, sample_value) == "sample:S1"
    assert owner_of(lane, stray) == "unattributed"
    assert owner_of(lane, make_field("thickness", "999", source_ids=["mineru_p0_b1"])) is None


# ---- select_targets: the lane-blind rule ---------------------------------------------------------------------


def test_a_conflict_puts_both_sides_on_the_list():
    lanes, report, _ = lanes_and_report("12.5", "125")
    assert [c.status for c in report.comparisons] == ["conflict"]

    targets = select_targets(lanes, report, policy="disputed")

    assert [(t.backend, t.value.value_raw, t.reason) for t in targets] == [
        ("mineru", "12.5", "conflict"),
        ("paddleocr_vl", "125", "conflict"),
    ]


def test_an_agreement_is_not_checked_under_the_disputed_policy():
    lanes, report, _ = lanes_and_report("12.5", "12.5")
    assert [c.status for c in report.comparisons] == ["agree"]

    assert select_targets(lanes, report, policy="disputed") == ()


def test_the_tables_policy_adds_every_value_cited_from_a_table():
    # The fixtures' values cite the table block b1: agreeing values the disputed policy leaves alone are
    # checked under the default policy, with their own reason, and a conflict keeps its reason.
    lanes, report, artifacts = lanes_and_report("12.5", "12.5")

    targets = select_targets(lanes, report, policy="tables", artifacts=artifacts)

    assert [(t.backend, t.reason) for t in targets] == [("mineru", "table"), ("paddleocr_vl", "table")]
    disputed, disputed_report, _ = lanes_and_report("12.5", "125")
    conflict = select_targets(disputed, disputed_report, policy="tables", artifacts=artifacts)
    assert {t.reason for t in conflict} == {"conflict"}


def test_the_tables_policy_leaves_text_cited_values_to_the_disputed_rule():
    artifacts = {"mineru": artifact_for("mineru"), "paddleocr_vl": artifact_for("paddleocr_vl")}
    lanes = {}
    for backend in artifacts:
        power = make_field("sputtering_power", "150", unit_raw="W", source_ids=[f"{backend}_p0_b0"])
        lane = make_lane(backend=backend, samples=[make_sample("A", [power])])
        blocks = {block.source_id: block.content for block in artifacts[backend].blocks}
        lanes[backend] = normalize_lane(ground_lane(lane, blocks))
    report = compare_lanes(lanes["mineru"], lanes["paddleocr_vl"], exact_match())
    assert [c.status for c in report.comparisons] == ["agree"]

    assert select_targets(lanes, report, policy="tables", artifacts=artifacts) == ()
    assert not cites_table(lanes["mineru"].sample("A").fields[0], artifacts["mineru"])  # type: ignore[union-attr]


def test_the_tables_policy_without_artifacts_degrades_to_disputed(caplog):
    lanes, report, _ = lanes_and_report("12.5", "12.5")

    with caplog.at_level("WARNING", logger="paperfacts.validate"):
        targets = select_targets(lanes, report, policy="tables")

    assert targets == ()
    assert "without artifacts" in caplog.text


def test_the_all_policy_checks_every_value_in_both_lanes():
    lanes, report, _ = lanes_and_report("12.5", "12.5")

    targets = select_targets(lanes, report, policy="all")

    assert [(t.backend, t.reason) for t in targets] == [("mineru", "all"), ("paddleocr_vl", "all")]


def test_a_one_sided_value_is_checked_as_missing():
    lane_a = normalize_lane(
        make_lane(
            backend="mineru",
            samples=[make_sample("A", [make_field("thickness", "250", unit_raw="nm", source_ids=["mineru_p0_b1"])])],
        )
    )
    lane_b = normalize_lane(make_lane(backend="paddleocr_vl", samples=[make_sample("A")]))
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="A", b_id="A", confidence=1.0, justification="", method="exact"),)
    )
    report = compare_lanes(lane_a, lane_b, matching)

    targets = select_targets({"mineru": lane_a, "paddleocr_vl": lane_b}, report, policy="disputed")

    assert [(t.backend, t.owner, t.reason) for t in targets] == [("mineru", "sample:A", "missing")]


def test_an_ungrounded_value_is_checked_whatever_the_comparison_said():
    # Lane B quotes a number its own block does not contain: grounding flags it, and both lanes' values
    # "agree" numerically, yet the flagged one goes to the model.
    lanes, report, _ = lanes_and_report("12.5", "12.5")
    flagged = lanes["paddleocr_vl"]
    sample = flagged.sample("A")
    assert sample is not None
    bad = sample.fields[0].model_copy(update={"grounded": False})
    lanes["paddleocr_vl"] = flagged.model_copy(update={"samples": (sample.model_copy(update={"fields": (bad,)}),)})

    targets = select_targets(lanes, report, policy="disputed")

    assert [(t.backend, t.reason) for t in targets] == [("paddleocr_vl", "ungrounded")]


def test_the_same_value_is_listed_once_under_the_first_reason_that_named_it():
    lanes, report, _ = lanes_and_report("12.5", "125")
    a = lanes["mineru"]
    sample = a.sample("A")
    assert sample is not None
    lanes["mineru"] = a.model_copy(
        update={
            "samples": (
                sample.model_copy(update={"fields": (sample.fields[0].model_copy(update={"grounded": False}),)}),
            )
        }
    )

    targets = select_targets(lanes, report, policy="disputed")

    mineru = [t for t in targets if t.backend == "mineru"]
    assert len(mineru) == 1 and mineru[0].reason == "conflict"


def test_the_selection_order_is_fixed():
    lanes, report, _ = lanes_and_report("12.5", "125")
    assert select_targets(lanes, report) == select_targets(lanes, report)


def test_an_unknown_policy_is_refused():
    lanes, report, _ = lanes_and_report("12.5", "125")
    with pytest.raises(ValueError, match="policy"):
        select_targets(lanes, report, policy="some")  # type: ignore[arg-type]


# ---- region_for: the crop is the union of the cited blocks on one page ----------------------------------------


def test_the_region_is_the_cited_block_padded():
    artifact = artifact_for("mineru")
    value = make_field("sheet_resistance", "12.5", source_ids=["mineru_p0_b1"])

    region = region_for(value, artifact, padding=0.01, context=0)

    assert region is not None
    assert region.page == 0 and region.source_ids == ("mineru_p0_b1",)
    assert region.context_ids == ()
    assert region.bbox == BOX_B.padded(0.01)


def test_two_cited_blocks_on_one_page_are_joined():
    artifact = artifact_for("mineru")
    value = make_field("sheet_resistance", "12.5", source_ids=["mineru_p0_b0", "mineru_p0_b1"])

    region = region_for(value, artifact, padding=0.0, context=0)

    assert region is not None
    assert region.bbox == BOX_A.union(BOX_B)
    assert region.source_ids == ("mineru_p0_b0", "mineru_p0_b1")


# ---- The sliding window: the neighbours before and after the cited block -------------------------------------


def four_blocks(backend: str = "mineru") -> ParsedArtifact:
    """Paragraph, table, caption, paragraph -- in reading order on page 0 -- plus a figure between the table
    and its caption that carries no text."""
    return make_artifact(
        [
            make_block(page=0, order=0, backend=backend, content="Sample A was deposited at 150 W.", bbox=BOX_A),
            make_block(page=0, order=1, type="table", backend=backend, content="Rs | 12.5 Ω/sq", bbox=BOX_B),
            make_block(
                page=0,
                order=2,
                type="figure",
                backend=backend,
                content="",
                bbox=NormalizedBBox(x1=0.1, y1=0.61, x2=0.9, y2=0.64),
            ),
            make_block(
                page=0, order=3, type="caption", backend=backend, content="Table 1. Sheet resistance.", bbox=BOX_C
            ),
            make_block(page=0, order=4, backend=backend, content="The films were annealed.", bbox=BOX_D),
        ],
        backend=backend,
    )


def test_one_neighbour_on_each_side_joins_the_crop_and_is_named_as_context():
    artifact = four_blocks()
    value = make_field("sheet_resistance", "12.5", source_ids=["mineru_p0_b1"])

    region = region_for(value, artifact, padding=0.0, context=1)

    assert region is not None
    assert region.source_ids == ("mineru_p0_b1",)
    assert region.context_ids == ("mineru_p0_b0", "mineru_p0_b3")
    assert region.bbox == BOX_A.union(BOX_B).union(BOX_C)


def test_a_figure_is_skipped_over_and_not_counted_as_a_neighbour():
    # The caption, not the figure, is the neighbour after the table: the window walks past blocks with no
    # text to read, and the figure's box is not drawn into the crop.
    artifact = four_blocks()
    region = region_of_blocks([artifact.block("mineru_p0_b1")], artifact, padding=0.0, context=1)
    assert "mineru_p0_b2" not in region.context_ids
    assert region.bbox.y2 == BOX_C.y2


def test_a_wider_window_takes_more_neighbours_and_stops_at_the_page_edge():
    artifact = four_blocks()
    region = region_of_blocks([artifact.block("mineru_p0_b1")], artifact, padding=0.0, context=5)
    assert region.context_ids == ("mineru_p0_b0", "mineru_p0_b3", "mineru_p0_b4")
    assert region.bbox == BOX_A.union(BOX_D)


def test_a_window_of_zero_is_the_cited_block_alone():
    artifact = four_blocks()
    region = region_of_blocks([artifact.block("mineru_p0_b1")], artifact, padding=0.0, context=0)
    assert region.context_ids == () and region.bbox == BOX_B


def test_neighbours_never_come_from_another_page():
    artifact = make_artifact(
        [
            make_block(page=0, order=0, content="page one", bbox=BOX_A),
            make_block(page=1, order=0, type="table", content="Rs | 12.5", bbox=BOX_B),
            make_block(page=1, order=1, content="page two text", bbox=BOX_C),
        ]
    )
    region = region_of_blocks([artifact.block("mineru_p1_b0")], artifact, padding=0.0, context=1)
    assert region.page == 1
    assert region.context_ids == ("mineru_p1_b1",)


def test_a_cited_neighbour_is_not_listed_twice():
    artifact = four_blocks()
    cited = [artifact.block("mineru_p0_b0"), artifact.block("mineru_p0_b1")]
    region = region_of_blocks(cited, artifact, padding=0.0, context=1)
    assert region.source_ids == ("mineru_p0_b0", "mineru_p0_b1")
    assert region.context_ids == ("mineru_p0_b3",)


def test_the_stage_shows_the_model_the_window_and_records_it(pdf: Path, layout: DataLayout):
    lanes, report, _ = lanes_and_report("12.5", "125")
    artifacts = {"mineru": four_blocks("mineru"), "paddleocr_vl": four_blocks("paddleocr_vl")}
    client = FakeVisionClient(lambda call: reading("x"))

    validation = run_stage(lanes, report, artifacts, client, pdf, layout, context_blocks=1)

    crops = {value.backend: value.crop for value in validation.values}
    assert crops["mineru"] is not None
    assert crops["mineru"].source_ids == ("mineru_p0_b1",)
    assert crops["mineru"].context_ids == ("mineru_p0_b0", "mineru_p0_b3")
    # A wider crop is a different file: the window is part of what the model was shown.
    narrow = run_stage(lanes, report, artifacts, client, pdf, layout, context_blocks=0)
    assert {v.crop.path for v in narrow.values if v.crop} != {v.crop.path for v in validation.values if v.crop}


def test_a_citation_on_another_page_is_left_out_of_the_box():
    artifact = make_artifact(
        [
            make_block(page=0, order=0, content="page one", bbox=BOX_A),
            make_block(page=1, order=0, content="page two", bbox=BOX_B),
        ]
    )
    value = make_field("thickness", "250", source_ids=["mineru_p0_b0", "mineru_p1_b0"])

    region = region_for(value, artifact, padding=0.0, context=1)

    assert region == Region(page=0, bbox=BOX_A, source_ids=("mineru_p0_b0",))


def test_no_valid_citation_means_no_region():
    artifact = artifact_for("mineru")
    assert region_for(make_field("thickness", "250", source_ids=[]), artifact) is None
    assert region_for(make_field("thickness", "250", source_ids=["mineru_p9_b9"]), artifact) is None


# ---- CropStore: rendered once, kept on disk ---------------------------------------------------------------------


def test_a_crop_is_rendered_once_and_then_read_back(pdf: Path, layout: DataLayout):
    store = CropStore(layout, DOC_ID, pdf, dpi=72)
    region = Region(page=0, bbox=BOX_B, source_ids=("mineru_p0_b1",))

    first, crop = store.crop(region)
    second, again = store.crop(region)

    assert first == second and crop == again
    assert (layout.crops_dir(DOC_ID) / crop.path).read_bytes() == first
    assert crop.image_sha256 == hashlib.sha256(first).hexdigest()
    assert crop.width_px > 0 and crop.height_px > 0 and crop.dpi == 72


def test_the_crop_file_is_named_by_what_it_shows(pdf: Path, layout: DataLayout):
    store = CropStore(layout, DOC_ID, pdf, dpi=72)
    _, crop = store.crop(Region(page=0, bbox=BOX_B, source_ids=("x",)))
    assert crop.path == "p000_0.1000-0.4000-0.9000-0.6000_72dpi.png"


# ---- parse_reading: lenient about the wrapper, strict about the content ---------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"transcription": "Rs = 12.5 Ω/sq", "legible": true}',
        '```json\n{"transcription": "Rs = 12.5 Ω/sq", "legible": true}\n```',
        'Here is the JSON: {"transcription": "Rs = 12.5 Ω/sq", "legible": true} done.',
    ],
)
def test_the_json_object_is_found_wherever_the_model_put_it(text):
    parsed, note = parse_reading(text)
    assert parsed == Reading(transcription="Rs = 12.5 Ω/sq", legible=True)
    assert note == ""


def test_a_reply_with_no_json_is_taken_as_the_transcription_with_a_note():
    parsed, note = parse_reading("Rs = 12.5 Ω/sq")
    assert parsed == Reading(transcription="Rs = 12.5 Ω/sq", legible=True)
    assert "not the JSON object" in note


def test_an_empty_reply_is_illegible():
    parsed, _ = parse_reading("   ")
    assert parsed.legible is False


# ---- adjudicate: grounding's matcher, applied to the model's reading --------------------------------------------


@pytest.mark.parametrize(
    ("quoted", "transcription"),
    [
        ("12.5", "Rs | 12.5 Ω/sq"),
        ("1.2 × 10^-4", "ρ = 1.2×10⁻⁴ Ω·cm"),  # superscript vs caret, spaces around × vs none
        ("1.2×10^-4", "ρ = 1.2 × 10^-4 Ω·cm"),  # and the other way round
        ("40 × 10 cm", "target size (40 x 10 cm)"),
        ("40 × 10 cm", "target size (40×10 cm)"),
        ("SnO2:Ta", "$\\mathrm{SnO}_2$:Ta target"),
        ("250", "thickness of 250 nm"),
    ],
)
def test_a_value_the_model_read_is_confirmed(quoted, transcription):
    verdict, _ = adjudicate(make_field("thickness", quoted), Reading(transcription=transcription))
    assert verdict == "confirmed"


@pytest.mark.parametrize(
    ("quoted", "transcription"),
    [
        ("12.5", "Rs | 125 Ω/sq"),  # the digits differ
        ("10^2", "ρ = 10^-2 Ω·cm"),  # the sign the parser lost
        ("4", "deposited for 40 minutes"),  # never a number inside a longer number
        ("5 nm", "235 nm thick"),  # the fold closes only a gap between digits around ×, nothing else
        ("250", "Rs | 12.5 Ω/sq"),
    ],
)
def test_a_value_the_model_did_not_read_is_contradicted(quoted, transcription):
    verdict, _ = adjudicate(make_field("thickness", quoted), Reading(transcription=transcription))
    assert verdict == "contradicted"


def test_an_unreadable_region_is_illegible_not_contradicted():
    assert adjudicate(make_field("thickness", "250"), Reading(transcription="", legible=True))[0] == "illegible"
    assert adjudicate(make_field("thickness", "250"), Reading(transcription="250", legible=False))[0] == "illegible"


# ---- validate_lanes: the stage end to end -----------------------------------------------------------------------


def run_stage(lanes, report, artifacts, client, pdf, layout, **overrides) -> ValidationReport:
    kwargs = dict(
        document_id=DOC_ID,
        pdf_path=pdf,
        lanes=lanes,
        report=report,
        artifacts=artifacts,
        client=client,
        layout=layout,
        validation_key="vvvvvvvvvvvv",
        crop_dpi=72,
        policy="disputed",
        context_blocks=0,
    )
    kwargs.update(overrides)
    return validate_lanes(**kwargs)


def test_a_conflict_is_read_off_the_page_for_both_lanes(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    # The page (as the fake model reads it) says 12.5: lane A is confirmed, lane B contradicted.
    client = FakeVisionClient(lambda call: reading("Rs | 12.5 Ω/sq"))

    validation = run_stage(lanes, report, artifacts, client, pdf, layout)

    assert client.call_count == 2
    by_backend = {value.backend: value for value in validation.values}
    assert by_backend["mineru"].verdict == "confirmed"
    assert by_backend["paddleocr_vl"].verdict == "contradicted"
    assert by_backend["paddleocr_vl"].transcription == "Rs | 12.5 Ω/sq"
    assert validation.counts.confirmed == 1 and validation.counts.contradicted == 1 and validation.counts.total == 2
    assert validation.counts.by_backend == {"mineru": {"confirmed": 1}, "paddleocr_vl": {"contradicted": 1}}
    assert validation.counts.by_reason == {"conflict": {"confirmed": 1, "contradicted": 1}}


def test_the_model_is_never_told_the_value_it_is_checking(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    client = FakeVisionClient(lambda call: reading("whatever"))

    run_stage(lanes, report, artifacts, client, pdf, layout)

    for call in client.calls:
        assert "12.5" not in call.user and "125" not in call.user
        assert "12.5" not in call.system and "125" not in call.system
        # The field is named, so the model knows which small print matters; the wording is the same for both.
        assert "sheet_resistance" in call.user
    assert len({call.user for call in client.calls}) == 1
    assert len({call.system for call in client.calls}) == 1


def test_both_lanes_are_shown_their_own_lanes_region_at_the_same_settings(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    client = FakeVisionClient(lambda call: reading("x"))

    validation = run_stage(lanes, report, artifacts, client, pdf, layout)

    crops = {value.backend: value.crop for value in validation.values}
    assert crops["mineru"] is not None and crops["paddleocr_vl"] is not None
    assert crops["mineru"].source_ids == ("mineru_p0_b1",)
    assert crops["paddleocr_vl"].source_ids == ("paddleocr_vl_p0_b1",)
    assert crops["mineru"].dpi == crops["paddleocr_vl"].dpi == 72
    # Same box in both artifacts here, so the same pixels: the two lanes' images are byte-identical.
    assert crops["mineru"].image_sha256 == crops["paddleocr_vl"].image_sha256


def test_the_report_carries_the_keys_the_model_and_the_policy(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    client = FakeVisionClient(lambda call: reading("x"), model="qwen3-vl-test")

    validation = run_stage(lanes, report, artifacts, client, pdf, layout, policy="all")

    assert validation.document_id == DOC_ID
    assert validation.extractor_key == report.extractor_key == DEFAULT_EXTRACTOR_KEY
    assert validation.comparison_key == report.comparison_key
    assert validation.validation_key == "vvvvvvvvvvvv"
    assert validation.model == "qwen3-vl-test"
    assert validation.policy == "all"


def test_usage_is_summed_across_requests(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    client = FakeVisionClient(lambda call: reading("x"), usage={"total_tokens": 700})

    validation = run_stage(lanes, report, artifacts, client, pdf, layout)

    assert validation.usage == {"total_tokens": 1400}
    assert all(value.usage == {"total_tokens": 700} for value in validation.values)


def test_without_a_pdf_every_value_is_not_checked_and_no_request_is_made(layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    client = FakeVisionClient(lambda call: reading("x"))

    validation = run_stage(lanes, report, artifacts, client, None, layout)

    assert client.call_count == 0
    assert {value.verdict for value in validation.values} == {"not_checked"}
    assert validation.counts.not_checked == 2


def test_a_value_without_a_region_is_not_checked(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    # Take lane B's artifact away: its value cites a block that no longer exists anywhere.
    empty = artifacts["paddleocr_vl"].model_copy(update={"blocks": ()})
    client = FakeVisionClient(lambda call: reading("x"))

    validation = run_stage(lanes, report, {**artifacts, "paddleocr_vl": empty}, client, pdf, layout)

    by_backend = {value.backend: value for value in validation.values}
    assert by_backend["paddleocr_vl"].verdict == "not_checked"
    assert by_backend["paddleocr_vl"].crop is None
    assert by_backend["mineru"].verdict in {"confirmed", "contradicted"}
    assert client.call_count == 1


def test_one_failed_request_is_an_error_verdict_not_a_failed_stage(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    calls = 0

    def flaky(call: VisionCall) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LlmError("HTTP 500: boom")
        return reading("Rs | 12.5 Ω/sq")

    validation = run_stage(lanes, report, artifacts, FakeVisionClient(flaky), pdf, layout, concurrency=1)

    verdicts = sorted(value.verdict for value in validation.values)
    assert verdicts == ["confirmed", "error"] or verdicts == ["contradicted", "error"]
    assert validation.counts.error == 1
    error = next(value for value in validation.values if value.verdict == "error")
    assert "boom" in error.detail


def test_every_request_failing_is_a_failed_stage(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")

    def broken(call: VisionCall) -> str:
        raise LlmError("HTTP 404: model not found")

    with pytest.raises(LlmError, match="every vision request failed"):
        run_stage(lanes, report, artifacts, FakeVisionClient(broken), pdf, layout)


def test_nothing_to_check_is_an_empty_report_not_an_error(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "12.5")
    client = FakeVisionClient([])

    validation = run_stage(lanes, report, artifacts, client, pdf, layout)

    assert validation.values == () and validation.counts.total == 0
    assert client.call_count == 0


def test_a_reply_that_was_not_json_is_still_adjudicated_and_says_so(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    client = FakeVisionClient(lambda call: "Rs | 12.5 Ω/sq")

    validation = run_stage(lanes, report, artifacts, client, pdf, layout)

    by_backend = {value.backend: value for value in validation.values}
    assert by_backend["mineru"].verdict == "confirmed"
    assert "not the JSON object" in by_backend["mineru"].detail


def test_refresh_is_passed_to_every_request(pdf: Path, layout: DataLayout):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    client = FakeVisionClient(lambda call: reading("x"))

    run_stage(lanes, report, artifacts, client, pdf, layout, refresh=True)

    assert all(call.refresh for call in client.calls)


def test_the_report_survives_a_round_trip_through_disk(pdf: Path, layout: DataLayout, tmp_path: Path):
    lanes, report, artifacts = lanes_and_report("12.5", "125")
    validation = run_stage(lanes, report, artifacts, FakeVisionClient(lambda call: reading("Rs | 12.5")), pdf, layout)

    path = tmp_path / "validation.json"
    validation.write(path)

    assert ValidationReport.read(path) == validation
    assert set(ValidationReport.read(path).verdicts()) == {value.key for value in validation.values}


# ---- Filling blanks from the tables ---------------------------------------------------------------------------


TABLE_TEXT = "Table 1. Properties.\nSample | Rs (Ω/sq) | Thickness (nm)\nA | 12.5 | 250\nB | 40 | 310"


def extractor_answer(fields_by_sample: dict[str, list[tuple[str, str, str | None]]]):
    """A responder that cites the one source marker the fill prompt carries and answers per sample."""

    def respond(system: str, user: str) -> str:
        marker = user.split("<!-- source: ", 1)[1].split(" -->", 1)[0]
        return json.dumps(
            {
                "samples": [
                    {
                        "sample_id": sample_id,
                        "fields": [
                            {"field": name, "value_raw": raw, "unit_raw": unit, "source_ids": [marker]}
                            for name, raw, unit in fields
                        ],
                    }
                    for sample_id, fields in fields_by_sample.items()
                ]
            }
        )

    return respond


def table_lanes(*, with_b: bool = True):
    """Both lanes read sheet_resistance for sample A (and B) from the table block b1; nothing else."""
    artifacts = {"mineru": artifact_for("mineru"), "paddleocr_vl": artifact_for("paddleocr_vl")}
    lanes = {}
    for backend in artifacts:
        cited = [f"{backend}_p0_b1"]
        samples = [make_sample("A", [make_field("sheet_resistance", "12.5", unit_raw="Ω/sq", source_ids=cited)])]
        if with_b:
            samples.append(make_sample("B", [make_field("sheet_resistance", "40", unit_raw="Ω/sq", source_ids=cited)]))
        lane = make_lane(backend=backend, samples=samples)
        blocks = {block.source_id: block.content for block in artifacts[backend].blocks}
        lanes[backend] = normalize_lane(ground_lane(lane, blocks))
    return lanes, artifacts


def test_missing_fields_are_the_sample_level_fields_the_sample_lacks():
    sample = make_sample("A", [make_field("sheet_resistance", "12.5", unit_raw="Ω/sq")])
    names = [spec.name for spec in missing_fields(sample)]
    assert "sheet_resistance" not in names
    assert "thickness" in names
    assert all(spec.is_sample_level for spec in missing_fields(sample))


def test_the_tables_a_lane_read_from_are_listed_once_with_their_samples():
    lanes, artifacts = table_lanes()
    regions = table_regions_of(lanes["mineru"], artifacts["mineru"], padding=0.0, context=0)
    assert list(regions) == ["mineru_p0_b1"]
    region, sample_ids = regions["mineru_p0_b1"]
    assert sample_ids == ("A", "B")
    assert region.bbox == BOX_B


def test_a_value_quoted_from_the_transcription_fills_the_blank(pdf: Path, layout: DataLayout):
    lanes, artifacts = table_lanes(with_b=False)
    store = CropStore(layout, DOC_ID, pdf, dpi=72)
    vlm = FakeVisionClient(lambda call: reading(TABLE_TEXT))
    llm = FakeLlmClient(extractor_answer({"A": [("thickness", "250", "nm")]}))

    fills, dropped, usage = fill_from_tables(lanes, artifacts, store, vlm, llm, padding=0.0, context=0, refresh=False)

    assert dropped == ()
    assert [(f.backend, f.owner, f.field, f.value_raw, f.unit_raw) for f in fills] == [
        ("mineru", "sample:A", "thickness", "250", "nm"),
        ("paddleocr_vl", "sample:A", "thickness", "250", "nm"),
    ]
    assert all(f.source_id.startswith(FILL_SOURCE_PREFIX) and f.transcription == TABLE_TEXT for f in fills)
    assert fills[0].key == value_key("mineru", "sample:A", fills[0].as_field_value())
    assert fills[0].as_field_value().grounded is True
    # One table per lane: the VLM read it once per lane and the extractor was asked once per lane.
    assert vlm.call_count == 2 and llm.call_count == 2
    assert usage


def test_the_extractor_is_asked_only_for_the_missing_fields_of_the_cited_samples(pdf: Path, layout: DataLayout):
    lanes, artifacts = table_lanes()
    store = CropStore(layout, DOC_ID, pdf, dpi=72)
    vlm = FakeVisionClient(lambda call: reading(TABLE_TEXT))
    llm = FakeLlmClient(extractor_answer({}))

    fill_from_tables(lanes, artifacts, store, vlm, llm, padding=0.0, context=0, refresh=False)

    user = llm.calls[0].user
    assert "- id: A" in user and "- id: B" in user
    assert "sheet_resistance" not in user.split("Table transcription:")[0].split("Fields still missing")[1]
    assert "thickness" in user
    assert TABLE_TEXT in user
    # The table is asked for as a whole, no field named, so the transcription serves every blank.
    assert "sheet_resistance" not in vlm.calls[0].user and "caption" in vlm.calls[0].user


def test_a_fill_the_transcription_does_not_contain_is_dropped(pdf: Path, layout: DataLayout):
    lanes, artifacts = table_lanes(with_b=False)
    store = CropStore(layout, DOC_ID, pdf, dpi=72)
    vlm = FakeVisionClient(lambda call: reading(TABLE_TEXT))
    llm = FakeLlmClient(extractor_answer({"A": [("thickness", "999", "nm")]}))

    fills, dropped, _ = fill_from_tables(lanes, artifacts, store, vlm, llm, padding=0.0, context=0, refresh=False)

    assert fills == ()
    assert len(dropped) == 2 and all("is not in the transcription" in reason for reason in dropped)


def test_a_field_the_lane_already_holds_or_a_sample_it_did_not_cite_is_refused(pdf: Path, layout: DataLayout):
    lanes, artifacts = table_lanes(with_b=False)
    store = CropStore(layout, DOC_ID, pdf, dpi=72)
    vlm = FakeVisionClient(lambda call: reading(TABLE_TEXT))
    llm = FakeLlmClient(
        extractor_answer({"A": [("sheet_resistance", "12.5", "Ω/sq")], "B": [("thickness", "310", "nm")]})
    )

    fills, dropped, _ = fill_from_tables(lanes, artifacts, store, vlm, llm, padding=0.0, context=0, refresh=False)

    assert fills == ()
    assert any("was not asked for" in reason for reason in dropped)
    assert any("is not a sample cited from this table" in reason for reason in dropped)


def test_an_illegible_table_asks_the_extractor_nothing(pdf: Path, layout: DataLayout):
    lanes, artifacts = table_lanes(with_b=False)
    store = CropStore(layout, DOC_ID, pdf, dpi=72)
    vlm = FakeVisionClient(lambda call: reading("", legible=False))
    llm = FakeLlmClient(extractor_answer({"A": [("thickness", "250", "nm")]}))

    fills, dropped, _ = fill_from_tables(lanes, artifacts, store, vlm, llm, padding=0.0, context=0, refresh=False)

    assert fills == () and llm.call_count == 0
    assert all("illegible" in reason for reason in dropped) and len(dropped) == 2


def test_a_failed_table_read_drops_that_table_and_reads_the_next(pdf: Path, layout: DataLayout):
    lanes, artifacts = table_lanes(with_b=False)
    store = CropStore(layout, DOC_ID, pdf, dpi=72)
    calls = 0

    def flaky(call: VisionCall) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LlmError("HTTP 500: boom")
        return reading(TABLE_TEXT)

    llm = FakeLlmClient(extractor_answer({"A": [("thickness", "250", "nm")]}))
    fills, dropped, _ = fill_from_tables(
        lanes, artifacts, store, FakeVisionClient(flaky), llm, padding=0.0, context=0, refresh=False
    )

    assert [f.backend for f in fills] == ["paddleocr_vl"]
    assert len(dropped) == 1 and "boom" in dropped[0] and dropped[0].startswith("mineru")


def test_a_sample_with_nothing_missing_costs_no_request(pdf: Path, layout: DataLayout):
    lanes, artifacts = table_lanes(with_b=False)
    full = {}
    for backend, lane in lanes.items():
        sample = lane.sample("A")
        assert sample is not None
        fields = tuple(sample.fields) + tuple(
            make_field(spec.name, "1", source_ids=[f"{backend}_p0_b1"]) for spec in missing_fields(sample)
        )
        full[backend] = lane.model_copy(update={"samples": (sample.model_copy(update={"fields": fields}),)})
    store = CropStore(layout, DOC_ID, pdf, dpi=72)
    vlm, llm = FakeVisionClient([]), FakeLlmClient([])

    fills, dropped, _ = fill_from_tables(full, artifacts, store, vlm, llm, padding=0.0, context=0, refresh=False)

    assert fills == () and dropped == () and vlm.call_count == 0 and llm.call_count == 0


def test_the_stage_carries_the_fills_and_counts_them(pdf: Path, layout: DataLayout):
    lanes, artifacts = table_lanes(with_b=False)
    report = compare_lanes(lanes["mineru"], lanes["paddleocr_vl"], exact_match())
    vlm = FakeVisionClient(lambda call: reading(TABLE_TEXT))
    llm = FakeLlmClient(extractor_answer({"A": [("thickness", "250", "nm")]}))

    validation = run_stage(lanes, report, artifacts, vlm, pdf, layout, llm=llm, fill_blanks=True)

    assert validation.counts.filled == 2
    assert set(validation.fills_by_cell()) == {
        ("mineru", "sample:A", "thickness"),
        ("paddleocr_vl", "sample:A", "thickness"),
    }
    assert validation.fill_dropped == ()
    # The fills survive the file exactly like the verdicts.
    path = layout.doc_dir(DOC_ID) / "v.json"
    validation.write(path)
    assert ValidationReport.read(path) == validation


def test_filling_needs_the_extractor(pdf: Path, layout: DataLayout):
    lanes, artifacts = table_lanes(with_b=False)
    report = compare_lanes(lanes["mineru"], lanes["paddleocr_vl"], exact_match())
    with pytest.raises(ValueError, match="llm"):
        run_stage(lanes, report, artifacts, FakeVisionClient([]), pdf, layout, fill_blanks=True)


def test_without_a_pdf_nothing_is_filled_and_the_extractor_is_not_asked(layout: DataLayout):
    lanes, artifacts = table_lanes(with_b=False)
    report = compare_lanes(lanes["mineru"], lanes["paddleocr_vl"], exact_match())
    llm = FakeLlmClient([])

    validation = run_stage(lanes, report, artifacts, FakeVisionClient([]), None, layout, llm=llm, fill_blanks=True)

    assert validation.fills == () and llm.call_count == 0
