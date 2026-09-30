"""The article type: a review is detected on its first page, once per document, and both lanes are told the same.

A review lists the films of the works it reviews, and both lanes agree on them, so the comparison cannot catch it.
Detection is deterministic and high-precision (a wrong "review" costs a research paper its samples); what the lanes
are told is decided from both parses, so a badge one parser dropped still reaches the other lane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperfacts.config import Settings
from paperfacts.extract import detect_article_type, extract_lane
from paperfacts.models import BACKENDS, DocumentInput, ParsedArtifact
from paperfacts.prompts import article_note, inventory_system_prompt
from paperfacts.records import LaneExtraction
from paperfacts.storage import DataLayout
from paperfacts.workflow import document_article_type
from support.extraction import lane_options, make_artifact, make_lane
from support.factories import make_block
from support.llm import FakeLlmClient
from support.profiles import shipped_profile

TSF2011 = Path(__file__).parent / "fixtures" / "real" / "tsf2011"


def first_page(*contents: str, page: int = 0, type: str = "text", backend: str = "mineru") -> tuple:
    return tuple(
        make_block(page=page, order=order, type=type, content=content, backend=backend)
        for order, content in enumerate(contents)
    )


# ---- detection ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "badge", ["Critical review", "REVIEW", "Review article", "Mini-review", "Mini review", "Topical Review", "Reviews"]
)
def test_an_article_type_badge_on_the_first_page_marks_a_review(badge):
    assert detect_article_type(first_page(badge, "TCO/metal/TCO structures for energy")) == "review"


def test_the_badge_may_be_a_title_block():
    assert detect_article_type(first_page("Critical review", type="title")) == "review"


@pytest.mark.parametrize(
    "sentence",
    [
        "This review summarizes recent progress in transparent electrodes.",
        "In the present review we compile the sheet resistances reported so far.",
        "This mini-review covers metal nanowire networks.",
    ],
)
def test_a_first_page_sentence_calling_the_paper_a_review_marks_it(sentence):
    assert detect_article_type(first_page("Metal-microstructure electrodes", sentence)) == "review"


@pytest.mark.parametrize(
    "text",
    [
        "Peer review under responsibility of the Chinese Academy of Sciences.",
        "Review of the manuscript was handled by the editor.",
        "ITO thin films: an overview",
        "Perspective",
        "We review the deposition conditions of our films below.",
        "The films were reviewed by XRD.",
        "Sputtered ITO films for flexible electronics",
    ],
)
def test_research_paper_front_matter_is_not_a_review(text):
    assert detect_article_type(first_page(text)) is None


def test_only_the_first_page_counts():
    assert detect_article_type(first_page("This review summarizes recent progress.", page=1)) is None


def test_a_badge_in_a_table_or_caption_is_not_one():
    assert detect_article_type(first_page("Review", type="caption")) is None
    assert detect_article_type(first_page("Review", type="table")) is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_tsf2011s_recorded_first_page_is_a_review_in_both_lanes(backend):
    # "Critical review" is block 0 of page 0 in both parses of tsf.2011 (88115225).
    artifact = ParsedArtifact.read(TSF2011 / f"{backend}.artifact.json")

    assert detect_article_type(artifact.blocks) == "review"


# ---- once per document --------------------------------------------------------------------------------------


def write_parses(tmp_path: Path, contents: dict[str, str]) -> tuple[DocumentInput, Settings]:
    settings = Settings(data_root=tmp_path / "data", repo_root=tmp_path)
    document = DocumentInput(document_id="a" * 64, sha256="a" * 64, pdf_path=tmp_path / "paper.pdf")
    layout = DataLayout(settings.data_root)
    for backend in BACKENDS:
        blocks = first_page(contents.get(backend, "Sputtered ITO films"), backend=backend)
        artifact = make_artifact(
            tuple(block.model_copy(update={"document_id": document.document_id}) for block in blocks),
            backend=backend,
            document_id=document.document_id,
        )
        artifact.write(layout.artifact_path(document.document_id, backend))
    return document, settings


def test_the_document_is_a_review_when_either_parse_shows_the_badge(tmp_path: Path):
    # xu2020 (90239d24): MinerU keeps the "REVIEW" badge, PaddleOCR-VL drops it. Both lanes must be told.
    document, settings = write_parses(tmp_path, {"mineru": "REVIEW"})

    assert document_article_type(document, settings) == "review"


def test_the_document_is_ordinary_when_neither_parse_shows_one(tmp_path: Path):
    document, settings = write_parses(tmp_path, {})

    assert document_article_type(document, settings) is None


def test_the_type_needs_both_parses(tmp_path: Path):
    document, settings = write_parses(tmp_path, {})
    DataLayout(settings.data_root).artifact_path(document.document_id, "paddleocr_vl").unlink()

    with pytest.raises(FileNotFoundError, match="paddleocr_vl"):
        document_article_type(document, settings)


# ---- what the lanes are told ----------------------------------------------------------------------------------


def run_passage_lane(backend: str, article_type: str | None) -> tuple[LaneExtraction, FakeLlmClient]:
    profile = shipped_profile()

    def respond(system: str, user: str) -> str:
        if system == inventory_system_prompt(profile):
            return json.dumps({"samples": [], "no_tco_film": True})
        return json.dumps({"values": []})

    client = FakeLlmClient(respond)
    blocks = first_page("Critical review", "ITO films were sputtered at 100 W.", backend=backend)
    lane = extract_lane(
        make_artifact(blocks, backend=backend), client, lane_options(client, mode="passage"), article_type=article_type
    )
    return lane, client


def inventory_user(client: FakeLlmClient) -> str:
    profile = shipped_profile()
    return next(call.user for call in client.calls if call.system == inventory_system_prompt(profile))


def test_a_reviews_inventory_question_carries_the_same_note_in_both_lanes():
    note = article_note(shipped_profile(), "review")
    users = [inventory_user(run_passage_lane(backend, "review")[1]) for backend in BACKENDS]

    assert all(user.startswith(f"{note}\n\nPaper excerpts") for user in users)
    assert note.startswith("Note on this paper: its front matter marks it as a review.")


def test_an_ordinary_papers_inventory_question_carries_no_note_in_either_lane():
    users = [inventory_user(run_passage_lane(backend, None)[1]) for backend in BACKENDS]

    assert all(user.startswith("Paper excerpts") for user in users)


def test_only_the_inventory_question_carries_the_note():
    _, client = run_passage_lane("mineru", "review")
    others = [call.user for call in client.calls if call.system != inventory_system_prompt(shipped_profile())]

    assert not any("Note on this paper" in user for user in others)


def test_the_lane_records_its_type_and_the_audit_says_what_was_told():
    lane, _ = run_passage_lane("mineru", "review")

    assert lane.article_type == "review"
    assert "article type: review; the inventory question was told" in lane.dropped
    ordinary, _ = run_passage_lane("mineru", None)
    assert ordinary.article_type is None
    assert not any(entry.startswith("article type") for entry in ordinary.dropped)


def test_document_mode_tells_its_one_question_with_its_own_note():
    client = FakeLlmClient([json.dumps({"target": None, "samples": []})])
    artifact = make_artifact(first_page("Critical review", "ITO films were sputtered at 100 W."))

    lane = extract_lane(artifact, client, lane_options(client, mode="document"), article_type="review")

    (call,) = client.calls
    assert call.user.startswith("Note on this paper: its front matter marks it as a review.")
    assert 'return "samples" empty.\n\nPaper (Markdown' in call.user
    assert "no_tco_film" not in call.user  # document mode's answer has no such key
    assert "article type: review; the extraction question was told" in lane.dropped


# ---- stored ---------------------------------------------------------------------------------------------------


def test_the_article_type_round_trips_and_is_left_out_of_an_ordinary_lanes_file(tmp_path: Path):
    path = tmp_path / "lane.json"
    make_lane().model_copy(update={"article_type": "review"}).write(path)
    assert LaneExtraction.read(path).article_type == "review"

    make_lane().write(path)
    assert "article_type" not in json.loads(path.read_text(encoding="utf-8"))
    assert LaneExtraction.read(path).article_type is None
