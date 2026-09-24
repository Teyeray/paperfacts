"""Figure reading: which charts are read, how the answer is parsed, and what a reading claims.

No model and no PDF: the vision client is :class:`support.vision.FakeVisionClient` and the crop renderer is
a lambda returning fixed bytes, so every test is about this module's own decisions.
"""

from __future__ import annotations

import dataclasses

import pytest

from paperfacts import figures, keys
from paperfacts.config import Settings
from paperfacts.errors import LlmError
from paperfacts.figures import (
    FigureReadings,
    figure_groups,
    parse_answer,
    precision_for,
    read_figures,
    select_panels,
)
from paperfacts.keys import figure_key, figure_key_for
from paperfacts.models import NormalizedBBox, PageGeometry, ParsedArtifact
from support.factories import DOC_ID, make_block
from support.vision import NOT_A_CHART, FakeVisionClient, chart_answer

BOX = NormalizedBBox(x1=0.1, y1=0.1, x2=0.5, y2=0.4)


def fig(page: int, order: int):
    return make_block(page=page, order=order, type="figure", content=f"images/p{page}_{order}.jpg", bbox=BOX)


def cap(page: int, order: int, text: str):
    return make_block(page=page, order=order, type="caption", content=text, bbox=BOX)


def text(page: int, order: int, content: str = "A long paragraph of discussion about the films."):
    return make_block(page=page, order=order, type="text", content=content)


def artifact(*blocks) -> ParsedArtifact:
    return ParsedArtifact(
        document_id=DOC_ID,
        backend="mineru",
        backend_version="3.4.5",
        pages=(PageGeometry(index=0, width_pt=595, height_pt=842), PageGeometry(index=1, width_pt=595, height_pt=842)),
        blocks=tuple(blocks),
    )


def run(art: ParsedArtifact, client: FakeVisionClient, *, limit: int = 12) -> FigureReadings:
    return read_figures(art, lambda page, bbox: b"png", client, figure_key="k" * 12, max_per_document=limit)


# ---- Figure groups and the whole-figure caption ---------------------------------------------------


def test_panels_of_a_multi_panel_figure_share_the_whole_figure_caption():
    # MinerU gives each panel the neighbours' labels as its caption; only the last caption names the figure.
    groups = figure_groups(
        [
            fig(0, 0),
            cap(0, 1, "(a) (c)"),
            fig(0, 2),
            cap(0, 3, "(b)"),
            fig(0, 4),
            cap(0, 5, "Fig. 3. (a) Sheet resistance vs O2 flow; (b) transmittance spectra; (c) XRD."),
        ]
    )

    assert len(groups) == 1
    assert [block.order for block in groups[0].panels] == [0, 2, 4]
    assert groups[0].caption.startswith("Fig. 3.")
    assert groups[0].label == "Fig. 3"


def test_two_figures_on_one_page_are_two_groups():
    groups = figure_groups(
        [fig(0, 0), cap(0, 1, "Figure 1 Resistivity of the films."), fig(0, 2), cap(0, 3, "FIGURE 2 Thickness map.")]
    )

    assert [(g.label, len(g.panels)) for g in groups] == [("Figure 1", 1), ("FIGURE 2", 1)]


def test_a_caption_above_its_figure_opens_the_group():
    groups = figure_groups([cap(0, 0, "Fig. 5 Sheet resistance."), fig(0, 1), fig(0, 2), text(0, 3)])

    assert len(groups) == 1
    assert groups[0].label == "Fig. 5"
    assert len(groups[0].panels) == 2


def test_prose_or_a_page_break_ends_a_group_but_a_panel_label_does_not():
    groups = figure_groups([fig(0, 0), text(0, 1, "(b)"), fig(0, 2), text(0, 3), fig(0, 4), fig(1, 0)])

    assert [len(group.panels) for group in groups] == [2, 1, 1]


def test_without_a_figure_caption_the_group_keeps_whatever_captions_it_has():
    [group] = figure_groups([fig(0, 0), cap(0, 1, "Sheet resistance of the films")])

    assert group.label is None
    assert group.caption == "Sheet resistance of the films"


# ---- Selection -----------------------------------------------------------------------------------------


def test_a_figure_is_selected_when_its_caption_names_a_film_property():
    [request] = select_panels([fig(0, 0), cap(0, 1, "Fig. 2 Sheet resistance of ITO films")], limit=12)

    assert [spec.name for spec in request.fields] == ["sheet_resistance"]


def test_a_panel_caption_alone_does_not_select_a_figure_whose_caption_names_nothing():
    # "(a) Rs" on a panel, but the figure's own caption is about XRD: the figure caption is the one that counts.
    blocks = [fig(0, 0), cap(0, 1, "(a) sheet resistance"), fig(0, 2), cap(0, 3, "Fig. 4 XRD patterns of the films.")]

    assert select_panels(blocks, limit=12) == ()


def test_a_process_condition_in_the_caption_does_not_select_a_figure():
    blocks = [fig(0, 0), cap(0, 1, "Fig. 1 Spectra of films deposited at a substrate temperature of 300 °C.")]

    assert select_panels(blocks, limit=12) == ()


def test_every_panel_of_a_selected_figure_is_asked_separately_up_to_the_limit():
    blocks = [fig(0, 0), fig(0, 1), fig(0, 2), cap(0, 3, "Fig. 3 (a-c) Resistivity of the films")]
    blocks += [fig(1, 0), cap(1, 1, "Fig. 4 Thickness of the films")]

    chosen = select_panels(blocks, limit=2)

    assert [(r.block.page, r.block.order, r.panel) for r in chosen] == [(0, 0, 1), (0, 1, 2)]
    assert [(r.block.page, r.panel) for r in select_panels(blocks, limit=12)] == [(0, 1), (0, 2), (0, 3), (1, 1)]


def test_the_question_carries_the_whole_caption_and_only_the_candidate_fields():
    client = FakeVisionClient(chart_answer())
    run(artifact(fig(0, 0), cap(0, 1, "(a)"), fig(0, 2), cap(0, 3, "Fig. 3 Sheet resistance vs flow")), client)

    assert len(client.calls) == 2
    for call in client.calls:
        assert "<<<Fig. 3 Sheet resistance vs flow>>>" in call.user
        assert "- sheet_resistance:" in call.user
        assert "- resistivity:" not in call.user
        assert call.system == figures.SYSTEM_PROMPT


# ---- Lenient parsing -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"chart_type": "property_vs_condition", "points": []}',
        '```json\n{"chart_type": "property_vs_condition", "points": []}\n```',
        'Here you go:\n{"chart_type": "property_vs_condition", // the type\n "points": [],}\nThanks.',
        '<think>{not json}</think>{"chart_type": "property_vs_condition", /* c */ "points": []}',
    ],
)
def test_the_answer_is_read_leniently(reply: str):
    assert parse_answer(reply) == {"chart_type": "property_vs_condition", "points": []}


def test_a_double_slash_inside_a_string_is_not_a_comment():
    answer = parse_answer('{"reason": "see http://x.org", "chart": false}')

    assert answer == {"reason": "see http://x.org", "chart": False}


def test_an_answer_without_an_object_is_refused():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_answer("I cannot read this chart.")


# ---- Readings -----------------------------------------------------------------------------------------


SELECTED = (fig(0, 0), cap(0, 1, "Fig. 3 Sheet resistance and resistivity vs O2 flow"))


def test_a_reading_cites_the_figure_block_and_converts_y_with_the_axis_multiplier():
    readings = run(artifact(*SELECTED), FakeVisionClient(chart_answer(unit="[10^2 ohm/sq]"))).readings

    first = readings[0]
    assert (first.source_id, first.page, first.bbox) == ("mineru_p0_b0", 0, BOX)
    assert first.figure == "Fig. 3"
    assert (first.y_raw, first.y_unit_raw) == (25.0, "10^2 ohm/sq")
    assert first.y == pytest.approx(2500.0)
    assert first.unit == "Ω/sq"
    assert first.approximate is True
    assert first.x_value == 100 and first.x_unit == "sccm"


def test_a_unit_that_will_not_convert_keeps_the_raw_reading_with_a_note():
    [reading, _] = run(artifact(*SELECTED), FakeVisionClient(chart_answer(unit="arb. units"))).readings

    assert reading.y is None and reading.unit is None
    assert "unknown unit" in (reading.note or "")
    assert reading.y_raw == 25.0


def test_an_axis_the_model_did_not_tie_to_a_listed_field_is_not_read():
    answer = chart_answer(field="carrier_concentration")

    result = run(artifact(*SELECTED), FakeVisionClient(answer))

    assert result.readings == ()
    assert result.panels[0].status == "read"


def test_each_series_is_read_against_its_own_axis():
    answer = chart_answer()
    answer["y_axes"].append({"id": "right", "field": "resistivity", "unit": "10^-3 Ω cm", "scale": "linear"})
    answer["series"].append({"label": "rho", "y_axis": "right"})
    answer["points"].append({"series": "rho", "x": 100, "y": 4.0, "confidence": 0.9})

    readings = run(artifact(*SELECTED), FakeVisionClient(answer)).readings

    rho = [r for r in readings if r.field == "resistivity"]
    assert len(rho) == 1 and rho[0].y == pytest.approx(0.004) and rho[0].unit == "Ω·cm"


@pytest.mark.parametrize(
    ("scale", "series", "expected"),
    [("linear", 1, 0.10), ("linear", 3, 0.10), ("log", 1, 0.20), ("linear", 4, 0.20), ("logarithmic", 2, 0.20)],
)
def test_precision_is_ten_percent_linear_and_twenty_percent_on_a_log_or_crowded_chart(scale, series, expected):
    assert precision_for(scale, series) == expected


def test_the_precision_label_follows_the_axis_and_the_series_count():
    log = run(artifact(*SELECTED), FakeVisionClient(chart_answer(scale="log"))).readings
    crowded = run(artifact(*SELECTED), FakeVisionClient(chart_answer(series=("a", "b", "c", "d")))).readings
    plain = run(artifact(*SELECTED), FakeVisionClient(chart_answer())).readings

    assert {r.precision for r in log} == {0.20} and {r.scale for r in log} == {"log"}
    assert {r.precision for r in crowded} == {0.20}
    assert {r.precision for r in plain} == {0.10}


def test_a_low_confidence_marker_is_kept_with_a_note():
    [reading, _] = run(artifact(*SELECTED), FakeVisionClient(chart_answer(confidence=0.6))).readings

    assert "hidden or overlapping" in (reading.note or "")


def test_a_malformed_point_costs_only_itself():
    answer = chart_answer()
    answer["points"].append({"series": "Rs", "x": 300, "y": "about forty"})

    assert len(run(artifact(*SELECTED), FakeVisionClient(answer)).readings) == 2


# ---- Panel outcomes -------------------------------------------------------------------------------------


def test_a_refusal_is_recorded_and_yields_nothing():
    result = run(artifact(*SELECTED), FakeVisionClient(NOT_A_CHART))

    assert result.readings == ()
    assert result.panels[0].status == "not_chart"
    assert result.panels[0].detail == "a transmittance spectrum"
    assert result.complete


def test_the_short_refusal_is_accepted_too():
    assert run(artifact(*SELECTED), FakeVisionClient({"chart": False})).panels[0].status == "not_chart"


def test_an_unparseable_answer_is_final_but_a_failed_request_asks_again_next_time():
    unreadable = run(artifact(*SELECTED), FakeVisionClient("no idea"))
    failed = run(artifact(*SELECTED), FakeVisionClient(LlmError("HTTP 504")))

    assert unreadable.panels[0].status == "unreadable" and unreadable.complete
    assert failed.panels[0].status == "error" and not failed.complete


def test_one_failed_panel_does_not_cost_the_others():
    blocks = [fig(0, 0), fig(0, 1), cap(0, 2, "Fig. 3 Sheet resistance")]

    def responder(user: str, image: bytes):
        return LlmError("timeout") if image == b"p0_0" else chart_answer()

    images = iter([b"p0_0", b"p0_1"])
    result = read_figures(
        artifact(*blocks),
        lambda page, bbox: next(images),
        FakeVisionClient(responder),
        figure_key="k",
        max_per_document=12,
        concurrency=2,
    )

    assert [panel.status for panel in result.panels] == ["error", "read"]
    assert len(result.readings) == 2
    assert {r.panel for r in result.readings} == {2}


def test_usage_is_summed_over_panels_and_the_model_is_recorded():
    blocks = [fig(0, 0), fig(0, 1), cap(0, 2, "Fig. 3 Sheet resistance")]

    result = run(artifact(*blocks), FakeVisionClient(chart_answer(), model="qwen3.7-plus"))

    assert result.usage["total_tokens"] == 2 * 1300
    assert result.model == "qwen3.7-plus" and result.backend == "mineru"


def test_no_selected_figure_means_no_request():
    client = FakeVisionClient(chart_answer())

    result = run(artifact(fig(0, 0), cap(0, 1, "Fig. 1 Photograph of the sputtering system")), client)

    assert client.calls == [] and result.panels == ()


def test_the_readings_round_trip_through_their_file(tmp_path):
    result = run(artifact(*SELECTED), FakeVisionClient(chart_answer()))
    path = tmp_path / "figures" / "k.json"

    result.write(path)

    assert FigureReadings.read(path) == result


# ---- figure_key: what makes a stored reading stale -------------------------------------------------------


def key(**changes) -> str:
    arguments = {"dpi": 200, "max_pixels": 2_000_000, "max_per_document": 12} | changes
    model = arguments.pop("model", "qwen3.7-plus")
    return figure_key(model, **arguments)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "qwen3.6-plus"},
        {"dpi": 150},
        {"max_pixels": 1_000_000},
        {"max_per_document": 3},
        {"temperature": 0.2},
        {"max_tokens": 4096},
    ],
)
def test_the_figure_key_moves_with_everything_that_changes_the_question(change):
    assert key(**change) != key()


def test_the_figure_key_is_stable_for_the_same_settings():
    assert key() == key()


def test_the_figure_key_moves_when_a_film_field_keyword_changes(monkeypatch):
    specs = tuple(
        dataclasses.replace(spec, keywords=(*spec.keywords, "Rsq2")) if spec.name == "sheet_resistance" else spec
        for spec in figures.FIELD_SPECS
    )
    before = key()
    monkeypatch.setattr(figures, "FIELD_SPECS", specs)
    keys.figure_field_fingerprint.cache_clear()
    try:
        assert key() != before
    finally:
        monkeypatch.undo()
        keys.figure_field_fingerprint.cache_clear()


def test_the_figure_key_reads_its_settings():
    assert figure_key_for(Settings()) == key()
    assert figure_key_for(Settings(figures_dpi=100)) == key(dpi=100)
