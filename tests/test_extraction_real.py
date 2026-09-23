"""A full extraction -> normalization regression run against a **real** MinerU output.

The fixture is `runners/mineru_runner.py`'s actual run result on the first two pages of a TCO paper (see
:mod:`tests.test_adapter_mineru_real`). The LLM's answer is hand-written, but the source_id it cites and
the numeric spellings it copies (``$6.4 \\times 10^{-3}$``, ``225 °C``, ``40 × 10 cm``) all come from the
real text of those two pages.

The hand-written fixtures cover "every shape is handled correctly"; this file covers "the spellings found
in a real paper were not guessed wrong": the source_id format, the LaTeX/HTML superscripts in the
Markdown, the scientific notation inside table cells -- any drift in any of these makes
``invalid_source_ids`` or a normalized value light up red first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperfacts.adapters import convert, render_markdown
from paperfacts.extract import extract_lane
from paperfacts.models import DocumentGeometry, DocumentInput, PageGeometry, RawParseOutput
from paperfacts.normalize import normalize_lane
from support.llm import FakeLlmClient

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mineru_real_sample"
# These ids all come from the real output: b9 is TABLE 1 (resistivity), b4 is the methods paragraph
# (target size and deposition temperature).
TABLE_ID = "mineru_p1_b9"
METHOD_ID = "mineru_p1_b4"
ABSTRACT_ID = "mineru_p0_b8"


@pytest.fixture
def real_artifact():
    raw = RawParseOutput.load(FIXTURE_DIR, "mineru")
    geometry = DocumentGeometry(
        pages=tuple(PageGeometry(index=p.index, width_pt=p.width_pt, height_pt=p.height_pt) for p in raw.meta.pages)
    )
    sha = raw.meta.source.sha256
    document = DocumentInput(document_id=sha, pdf_path=Path("sample.pdf"), sha256=sha)
    return convert(raw, document, geometry)


# A hand-written "model answer": the fields and their spelling are copied straight from the paper, and the
# source_ids point at real blocks.
REAL_RESPONSE = json.dumps(
    {
        "target": {
            "source_ids": [METHOD_ID],
            "fields": [
                {
                    "field": "component",
                    "value_raw": "Sn/Ta target 95:5 wt.%",
                    "source_ids": [METHOD_ID],
                },
                {
                    "field": "inch",
                    "value_raw": "40 × 10",
                    "unit_raw": "cm",
                    "condition": "rectangular magnetron",
                    "source_ids": [METHOD_ID],
                },
            ],
        },
        "samples": [
            {
                "sample_id": "this-work-225C",
                "label": "reactive DC-MS, Sn + SnTa targets, 225 °C",
                "conditions": {"deposition temperature": "225 °C", "technique": "reactive DC–MS"},
                "source_ids": [TABLE_ID],
                "fields": [
                    {
                        "field": "resistivity",
                        "value_raw": "0.3",
                        "unit_raw": "Ω cm",
                        "source_ids": [TABLE_ID],
                    },
                    {
                        "field": "thickness",
                        "value_raw": "3",
                        "unit_raw": "mm",
                        "condition": "fused silica substrate",
                        "source_ids": [METHOD_ID],
                    },
                ],
            },
            {
                "sample_id": "muto-200C",
                "label": "pulsed-MS of SnTa targets, 200 °C (reference [19])",
                "conditions": {"deposition temperature": "200 °C"},
                "source_ids": [TABLE_ID],
                "fields": [
                    {
                        "field": "resistivity",
                        "value_raw": "6.4 × 10⁻³",
                        "unit_raw": "Ω cm",
                        "source_ids": [TABLE_ID],
                    }
                ],
            },
            {
                "sample_id": "mientus-25C",
                "label": "reactive MS, ceramic SnO2:Ta target, 25 °C (reference [22])",
                "conditions": {"deposition temperature": "25 °C"},
                "source_ids": [TABLE_ID],
                "fields": [
                    {
                        "field": "resistivity",
                        "value_raw": "4 \\times 1 0 ^ { - 3 }",
                        "unit_raw": "Ω cm",
                        "note": "copied from the LaTeX-ish table cell",
                        "source_ids": [TABLE_ID],
                    }
                ],
            },
        ],
    }
)


def test_the_real_source_ids_all_exist_in_the_artifact(real_artifact):
    """The most important thing: the provenance id format matches the real artifact, and not one of them
    gets thrown out as invented."""
    lane = extract_lane(real_artifact, FakeLlmClient([REAL_RESPONSE]), mode="document")

    assert lane.invalid_source_ids == ()
    assert {ABSTRACT_ID, TABLE_ID, METHOD_ID} <= {block.source_id for block in real_artifact.blocks}


def test_nothing_from_the_real_response_is_dropped_by_the_cleaning_rules(real_artifact):
    lane = extract_lane(real_artifact, FakeLlmClient([REAL_RESPONSE]), mode="document")

    assert lane.dropped == ()
    assert len(lane.samples) == 3
    assert lane.target is not None


def test_the_resistivity_written_in_plain_decimal_normalizes(real_artifact):
    lane = normalize_lane(extract_lane(real_artifact, FakeLlmClient([REAL_RESPONSE]), mode="document"))

    field = lane.sample("this-work-225C").get("resistivity")
    assert field.value == pytest.approx(0.3)
    assert field.unit == "Ω·cm"


def test_the_scientific_notation_from_the_real_table_cell_normalizes(real_artifact):
    # The table cell is written as $6 . 4 \times 1 0 ^ { - 3 }$; when the model copies it as the
    # superscript "6.4 × 10⁻³" it must be read as 0.0064, not 6.4.
    lane = normalize_lane(extract_lane(real_artifact, FakeLlmClient([REAL_RESPONSE]), mode="document"))

    field = lane.sample("muto-200C").get("resistivity")
    assert field.value == pytest.approx(6.4e-3)
    assert field.unit == "Ω·cm"


def test_a_latex_exponent_copied_verbatim_is_parsed(real_artifact):
    """MinerU's table cell is LaTeX (``$4 \\times 1 0 ^ { - 3 }$``, with spaces between characters), and
    the prompt asks for a verbatim transcription.

    This shape used to be misread as 4.0 (three orders of magnitude off, with only a weak note left
    behind) -- exactly the kind of silent error this project exists to catch. Now ``delatex`` strips the
    ``$`` / ``\\times`` / braces and rejoins the split-up digits first, so the exponent must parse
    correctly.
    """
    lane = normalize_lane(extract_lane(real_artifact, FakeLlmClient([REAL_RESPONSE]), mode="document"))

    field = lane.sample("mientus-25C").get("resistivity")
    assert field.value == pytest.approx(4e-3)
    assert field.normalization_note is None


@pytest.mark.parametrize(
    ("value_raw", "expected"),
    [
        ("6.4 × 10⁻³", 6.4e-3),  # superscript
        ("6.4 × 10^-3", 6.4e-3),  # already the ^ spelling
        ("6.4e-3", 6.4e-3),  # exponent spelling
        ("6.4 × 10<sup>-3</sup>", 6.4e-3),  # HTML superscript
        ("6.4 × 10-3", 6.4e-3),  # OCR lost the superscript but "× 10" survives
    ],
)
def test_the_spellings_a_well_behaved_model_produces_all_parse(value_raw, expected):
    """The same table cell could be copied by the model as any of these spellings; the result must be
    the same regardless."""
    from paperfacts.normalize import parse_number

    assert parse_number(value_raw)[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("value_raw", "expected"),
    [
        ("6.4 × 10^{-3}", 6.4e-3),  # braced exponent
        ("4 \\times 1 0 ^ { - 3 }", 4e-3),  # MinerU formula block: backslash command + scattered spaces
        ("$6 . 4 \\times 1 0 ^ { - 3 }$", 6.4e-3),  # MinerU table cell: also wrapped in $
        ("$2 \\mu m$", 2.0),  # a plain number with LaTeX around it is unaffected
    ],
)
def test_the_latex_spellings_from_mineru_tables_parse_to_the_exponent(value_raw, expected):
    """LaTeX spellings that really occur in MinerU's tables and formula blocks must parse to the correct
    exponent, and leave no weak "found N numbers" note behind."""
    from paperfacts.normalize import parse_number

    value, note = parse_number(value_raw)

    assert value == pytest.approx(expected)
    assert note is None


def test_spaces_between_digits_are_only_merged_inside_latex():
    """ "10 20" is two numbers, and must not be merged into 1020 just because LaTeX handling ran; only
    text carrying a $ or a backslash command has the spaces between its digits collapsed."""
    from paperfacts.normalize import parse_number

    assert parse_number("10 20") == (10.0, "2 numbers found, first used")


def test_the_thickness_in_millimetres_converts_to_nanometres(real_artifact):
    lane = normalize_lane(extract_lane(real_artifact, FakeLlmClient([REAL_RESPONSE]), mode="document"))

    field = lane.sample("this-work-225C").get("thickness")
    assert field.value == pytest.approx(3e6)
    assert field.unit == "nm"


def test_the_target_size_taken_from_the_methods_section_converts_to_inches(real_artifact):
    # "40 × 10 cm" contains two numbers: the first is taken, with a note left so the reader knows the
    # value is incomplete.
    lane = normalize_lane(extract_lane(real_artifact, FakeLlmClient([REAL_RESPONSE]), mode="document"))

    field = lane.target.get("inch")
    assert field.value == pytest.approx(40 / 2.54)
    assert field.unit == "inch"
    assert "2 numbers found" in field.normalization_note


def test_the_composition_text_is_kept_verbatim(real_artifact):
    lane = normalize_lane(extract_lane(real_artifact, FakeLlmClient([REAL_RESPONSE]), mode="document"))

    field = lane.target.get("component")
    assert field.value_raw == "Sn/Ta target 95:5 wt.%"
    assert field.value is None  # composition is text and should not have a number forced into it


def test_the_provenance_survives_all_the_way_to_the_normalized_lane(real_artifact):
    """Normalization only adds fields alongside; source_id must survive unchanged all the way through, or
    the value could never be traced back to a PDF page."""
    lane = normalize_lane(extract_lane(real_artifact, FakeLlmClient([REAL_RESPONSE]), mode="document"))

    field = lane.sample("muto-200C").get("resistivity")
    assert field.source_ids == (TABLE_ID,)
    block = real_artifact.block(TABLE_ID)
    assert block.type == "table"
    assert "6 . 4 \\times 1 0 ^ { - 3 }" in block.content


def test_every_informative_block_of_the_real_paper_reaches_the_model(real_artifact):
    """Sample-level records need the whole body: where a sample is defined and where its value is
    tabulated are usually pages apart. Only page furniture is withheld."""
    client = FakeLlmClient([REAL_RESPONSE])

    extract_lane(real_artifact, client, mode="document")

    prompt = client.users[0]
    assert f"<!-- source: {TABLE_ID} -->" in prompt
    for block in real_artifact.blocks:
        if block.type not in {"unknown", "figure"}:
            assert block.content in prompt, f"{block.source_id} ({block.type}) was withheld from the model"


def test_page_furniture_is_withheld_from_the_model(real_artifact):
    """Running heads, page numbers and figure image paths cost context and invite bad citations."""
    client = FakeLlmClient([REAL_RESPONSE])

    extract_lane(real_artifact, client, mode="document")

    prompt = client.users[0]
    furniture = [b for b in real_artifact.blocks if b.type in {"unknown", "figure"}]
    assert furniture, "the fixture should contain some page furniture for this test to mean anything"
    for block in furniture:
        assert f"<!-- source: {block.source_id} -->" not in prompt
    assert len(prompt) < len(render_markdown(real_artifact.blocks))
