"""Rendering the Markdown document that is actually sent to the extraction model.

``build_extraction_document`` trims a parser artifact down to what can plausibly hold a fact: page
furniture (``unknown`` / ``figure`` blocks, blank blocks) is dropped, and everything from a
references/bibliography heading onward is dropped outright. What survives keeps its ``<!-- page: N -->``
and ``<!-- source: id -->`` markers, since citations only resolve back to a page and a bounding box
through those.
"""

from __future__ import annotations

import pytest

from paperfacts.extract import build_extraction_document
from paperfacts.keys import extraction_code_fingerprint
from support.extraction import make_artifact
from support.factories import make_block

# ---- Noise filtering ------------------------------------------------------------------


def test_unknown_blocks_are_dropped():
    body = make_block(page=0, order=0, type="text", content="Sample A was sputtered at 100 sccm O2.")
    running_head = make_block(page=0, order=1, type="unknown", content="J. Mater. Sci. 2020, page 3")
    artifact = make_artifact([body, running_head])

    document = build_extraction_document(artifact)

    assert body.source_id in document.blocks
    assert running_head.source_id not in document.blocks
    assert running_head.content not in document.markdown


def test_figure_blocks_are_dropped():
    # A figure block's content is the image path, not the figure -- there is nothing there for the model
    # to cite a value from.
    body = make_block(page=0, order=0, type="text", content="See Figure 2 for the SEM cross section.")
    figure = make_block(page=0, order=1, type="figure", content="images/fig2.png")
    artifact = make_artifact([body, figure])

    document = build_extraction_document(artifact)

    assert body.source_id in document.blocks
    assert figure.source_id not in document.blocks
    assert "images/fig2.png" not in document.markdown


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_blocks_with_no_real_content_are_dropped_regardless_of_type(blank):
    body = make_block(page=0, order=0, type="text", content="Sample A was sputtered at 100 sccm O2.")
    empty = make_block(page=0, order=1, type="text", content=blank)
    artifact = make_artifact([body, empty])

    document = build_extraction_document(artifact)

    assert empty.source_id not in document.blocks
    assert document.kept_blocks == 1


def test_kept_and_dropped_block_counts_add_up_to_the_total():
    kept_block = make_block(page=0, order=0, type="text", content="Sample A was sputtered at 100 sccm O2.")
    noise = make_block(page=0, order=1, type="unknown", content="page 3 of 10")
    artifact = make_artifact([kept_block, noise])

    document = build_extraction_document(artifact)

    assert document.kept_blocks == 1
    assert document.dropped_blocks == 1
    assert document.kept_blocks + document.dropped_blocks == len(artifact.blocks)


# ---- References / bibliography cutoff --------------------------------------------------


@pytest.mark.parametrize(
    "heading",
    [
        "References",
        "REFERENCES",
        "# References",
        "Reference",
        "Bibliography",
        "Literature cited",
        "References and Notes",
        # Numbered, as MinerU and PaddleOCR-VL both keep a section number in the heading text.
        "6. References",
        "6 References",
        "## 7. REFERENCES",
        "VI. REFERENCES",
        "**References**",
        "References:",
        "Notes and references",
    ],
)
def test_a_references_style_title_and_everything_after_it_is_dropped(heading):
    body = make_block(page=0, order=0, type="text", content="Sample A had a sheet resistance of 12.5 Ω/sq.")
    heading_block = make_block(page=1, order=0, type="title", content=heading)
    citation = make_block(page=1, order=1, type="text", content="[1] Smith et al., Journal of Materials, 2020.")
    artifact = make_artifact([body, heading_block, citation])

    document = build_extraction_document(artifact)

    assert body.source_id in document.blocks
    assert heading_block.source_id not in document.blocks
    assert citation.source_id not in document.blocks
    assert "Smith et al." not in document.markdown


def test_a_title_merely_mentioning_references_without_starting_with_it_is_kept():
    # The cutoff is anchored at the start of the (stripped) title text, not a substring search, or a
    # section like "Cross-References Between Samples" would wrongly truncate the paper.
    heading = make_block(page=0, order=0, type="title", content="Cross-References Between Samples")
    after = make_block(page=0, order=1, type="text", content="Sample A had a sheet resistance of 12.5 Ω/sq.")
    artifact = make_artifact([heading, after])

    document = build_extraction_document(artifact)

    assert heading.source_id in document.blocks
    assert after.source_id in document.blocks


@pytest.mark.parametrize(
    "heading",
    ["Reference electrode", "Reference samples", "References to Table 2", "2. Reference cells and substrates"],
)
def test_a_title_that_merely_starts_with_reference_is_kept(heading):
    # A prefix match cut the paper at these, dropping every result after them.
    title = make_block(page=0, order=0, type="title", content=heading)
    after = make_block(page=0, order=1, type="text", content="Sample A had a sheet resistance of 12.5 Ω/sq.")

    document = build_extraction_document(make_artifact([title, after]))

    assert title.source_id in document.blocks
    assert after.source_id in document.blocks


def test_a_references_heading_the_parser_labelled_as_text_also_ends_the_document():
    # PaddleOCR-VL sometimes labels the heading as plain text. Only the whole block being the heading counts,
    # so a lane is not left carrying the bibliography because of how its parser labelled one line.
    body = make_block(page=0, order=0, type="text", content="Sample A had a sheet resistance of 12.5 Ω/sq.")
    heading = make_block(page=1, order=0, type="text", content="References")
    citation = make_block(page=1, order=1, type="text", content="[1] Smith et al., Journal of Materials, 2020.")

    document = build_extraction_document(make_artifact([body, heading, citation]))

    assert list(document.blocks) == [body.source_id]


def test_an_early_references_cut_is_logged(caplog):
    heading = make_block(page=0, order=0, type="title", content="References")
    rest = [make_block(page=0, order=order, type="text", content=f"Result {order}") for order in range(1, 4)]

    with caplog.at_level("WARNING", logger="paperfacts.extract"):
        build_extraction_document(make_artifact([heading, *rest]))

    assert "everything after it is left out" in caplog.text


def test_a_paragraph_mentioning_references_does_not_trigger_the_cutoff():
    # The word "References" inside an ordinary paragraph is not a section heading; only a block that is
    # nothing but the heading ends the document.
    body = make_block(page=0, order=0, type="text", content="References to prior work are listed below.")
    after = make_block(page=0, order=1, type="text", content="Sample A had a sheet resistance of 12.5 Ω/sq.")
    artifact = make_artifact([body, after])

    document = build_extraction_document(artifact)

    assert body.source_id in document.blocks
    assert after.source_id in document.blocks


def test_a_normal_title_is_kept():
    heading = make_block(page=0, order=0, type="title", content="Results and Discussion")
    body = make_block(page=0, order=1, type="text", content="Sample A had a sheet resistance of 12.5 Ω/sq.")
    artifact = make_artifact([heading, body])

    document = build_extraction_document(artifact)

    assert heading.source_id in document.blocks
    assert body.source_id in document.blocks


# ---- Markers survive into the rendered markdown -----------------------------------------


def test_page_and_source_markers_appear_in_the_rendered_markdown():
    block = make_block(page=2, order=0, type="text", content="Sample A had a sheet resistance of 12.5 Ω/sq.")
    artifact = make_artifact([block])

    document = build_extraction_document(artifact)

    assert "<!-- page: 2 -->" in document.markdown
    assert f"<!-- source: {block.source_id} -->" in document.markdown


def test_the_page_marker_is_not_repeated_for_consecutive_blocks_on_the_same_page():
    first = make_block(page=0, order=0, type="text", content="Sample A was sputtered at 100 sccm O2.")
    second = make_block(page=0, order=1, type="text", content="Sample A had a thickness of 300 nm.")
    artifact = make_artifact([first, second])

    document = build_extraction_document(artifact)

    assert document.markdown.count("<!-- page: 0 -->") == 1


def test_a_new_page_marker_is_emitted_when_the_page_changes():
    first = make_block(page=0, order=0, type="text", content="Sample A was sputtered at 100 sccm O2.")
    second = make_block(page=1, order=0, type="text", content="Sample A had a thickness of 300 nm.")
    artifact = make_artifact([first, second])

    document = build_extraction_document(artifact)

    assert document.markdown.count("<!-- page: 0 -->") == 1
    assert document.markdown.count("<!-- page: 1 -->") == 1


def test_the_blocks_mapping_matches_what_is_in_the_markdown():
    first = make_block(page=0, order=0, type="text", content="Sample A was sputtered at 100 sccm O2.")
    second = make_block(page=0, order=1, type="table", content="<table><tr><td>Rs</td><td>12.5 Ω/sq</td></tr></table>")
    artifact = make_artifact([first, second])

    document = build_extraction_document(artifact)

    assert document.blocks == {first.source_id: first.content, second.source_id: second.content}
    for source_id, content in document.blocks.items():
        assert f"<!-- source: {source_id} -->\n{content}\n" in document.markdown


# ---- estimated_tokens -----------------------------------------------------------------


def test_estimated_tokens_follows_the_measured_chars_per_token_ratio():
    # 3 characters per token was measured against a real 10-page paper; pin the arithmetic itself here so
    # a change to that ratio is a deliberate edit, not an accident.
    block = make_block(page=0, order=0, type="text", content="x" * 300)
    artifact = make_artifact([block])

    document = build_extraction_document(artifact)

    assert document.estimated_tokens == int(len(document.markdown) / 3.0) + 1


# ---- extraction_code_fingerprint --------------------------------------------------------


def test_extraction_code_fingerprint_is_a_stable_short_hex_string():
    # Rendering logic is hashed into extractor_key so that changing how the document is built
    # invalidates the extraction cache automatically; that only works if the fingerprint is stable and
    # actually looks like a hash.
    first = extraction_code_fingerprint()
    second = extraction_code_fingerprint()

    assert first == second
    assert len(first) == 12
    assert all(ch in "0123456789abcdef" for ch in first)
