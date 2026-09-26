"""``eval/score.py``: how a dataset cell is matched against a gold cell.

The script lives outside the package, so it is loaded from its path. Only the matching rules are pinned
here; the alignment and report are exercised by running it against the gold set.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from paperfacts.profile_loader import load_profile
from support.profiles import SHIPPED_PROFILE_PATH

SCRIPT = Path(__file__).resolve().parent.parent / "eval" / "score.py"
_spec = importlib.util.spec_from_file_location("paperfacts_eval_score", SCRIPT)
assert _spec is not None and _spec.loader is not None
score = importlib.util.module_from_spec(_spec)
# dataclasses look their module up in sys.modules while the class body is built.
sys.modules[_spec.name] = score
_spec.loader.exec_module(score)

MODE = score.load_specs(SHIPPED_PROFILE_PATH)["mode"]
assert MODE.categories == ("DC", "RF", "pulsed DC", "DC+RF", "HiPIMS")


@pytest.mark.parametrize(
    ("got", "gold"),
    [
        ("HiPIMS", "HiPIMS"),
        ("high-power impulse magnetron sputtering (HiPIMS)", "HiPIMS"),
        ("RF magnetron sputtering", "RF"),
        ("DC pulsed", "pulsed DC"),
        ("DC and RF co-sputtering", "DC+RF"),
    ],
)
def test_a_category_value_matches_its_gold_category(got, gold):
    # The package's own category rule: before, HiPIMS matched nothing, so every HiPIMS cell scored wrong.
    assert score.value_matches(MODE, got, {"value": gold})


@pytest.mark.parametrize(("got", "gold"), [("DC", "RF"), ("pulsed DC", "DC"), ("RF", "DC+RF")])
def test_a_different_category_does_not_match(got, gold):
    assert not score.value_matches(MODE, got, {"value": gold})


def test_the_specs_are_the_packages_own_field_table():
    # Scoring reads the profile through the package, so its tolerances, categories and levels cannot drift from
    # what the run compared with.
    assert score.load_specs(SHIPPED_PROFILE_PATH) == load_profile(SHIPPED_PROFILE_PATH).by_name


def _dataset(root: Path, name: str, fingerprint: str | None, mtime: int) -> Path:
    path = root / "docs" / "0000000000000001" / "datasets" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"profile_fingerprint": fingerprint, "paper_row": {}, "sample_rows": []}))
    os.utime(path, (mtime, mtime))
    return path


def test_without_keys_the_newest_dataset_of_the_scored_profile_is_chosen(tmp_path: Path):
    # Another profile's table is newer, but scoring it against this profile's gold set would be meaningless.
    own = _dataset(tmp_path, "aaaa.bbbb", "tco-fingerprint", mtime=1_000)
    _dataset(tmp_path, "cccc.dddd", "battery-fingerprint", mtime=2_000)

    assert score.find_dataset(tmp_path, "0000000000000001", None, "tco-fingerprint") == own
    with pytest.raises(FileNotFoundError, match="of profile fingerprint"):
        score.find_dataset(tmp_path, "0000000000000001", None, "other-fingerprint")


def test_a_dataset_from_before_profiles_is_still_a_candidate(tmp_path: Path):
    old = _dataset(tmp_path, "aaaa.bbbb", None, mtime=1_000)

    assert score.find_dataset(tmp_path, "0000000000000001", None, "tco-fingerprint") == old


def test_a_gold_sample_aligns_only_to_rows_of_its_entity_and_scores_its_fields():
    from support.profiles import make_entity_profile

    specs = make_entity_profile().by_name
    gold = {
        "doc_id": "d",
        "paper": {},
        "samples": [
            {"id": "c", "entity": "coating", "match": {"label": "S1"}, "fields": {"solvent": [{"value": "ethanol"}]}},
            {
                "id": "t",
                "entity": "wear_test",
                "match": {"label": "S1"},
                "fields": {"wear_mode": [{"value": "sliding"}]},
            },
        ],
    }
    rows = [
        {"entity": "wear_test", "sample_id": "S1", "wear_mode": "sliding", "test_temperature": None},
        {"entity": "coating", "sample_id": "S1", "solvent": "ethanol", "coating_thickness": None},
    ]
    dataset = {
        "sample_rows": rows,
        "quality_rows": [{"entity": "coating", "sample_id": "S1", "field": "solvent", "decision": "agree"}],
    }

    cells = score.score_document(specs, gold, dataset)

    assert {(cell.sample, cell.field, cell.outcome) for cell in cells if cell.outcome != "missing"} == {
        ("c", "solvent", "correct"),
        ("t", "wear_mode", "correct"),
    }
    # The trace finds the quality row of the gold sample's entity.
    assert next(cell.detail for cell in cells if cell.field == "solvent") == "agree"


def test_a_profile_declaring_one_entity_scores_the_dataset_consolidation_wrote():
    # With one declared entity the rows used to carry no `entity`, while the fields and the gold name it: every
    # sample cell then aligned to nothing and came out missing.
    from paperfacts.compare import compare_lanes
    from paperfacts.dataset import consolidate_document
    from paperfacts.keys import ComparisonOptions, profile_extraction_fingerprint
    from paperfacts.matching import SampleMatch, SampleMatching
    from paperfacts.models import DocumentInput
    from paperfacts.records import FieldValue, LaneExtraction, SampleRecord
    from support.profiles import make_one_entity_profile

    profile = make_one_entity_profile()
    options = ComparisonOptions(profile=profile, ambiguous_match_confidence=0.6)

    def lane(backend: str, sample_id: str) -> LaneExtraction:
        solvent = FieldValue(field="solvent", value_raw="ethanol", source_ids=("b",), grounded=True)
        return LaneExtraction(
            document_id="d" * 64,
            backend=backend,  # type: ignore[arg-type]
            extractor_key="k" * 12,
            model="m",
            profile_fingerprint=profile_extraction_fingerprint(profile),
            samples=(SampleRecord(sample_id=sample_id, entity="coating", fields=(solvent,)),),
        )

    lanes = {"mineru": lane("mineru", "S1"), "paddleocr_vl": lane("paddleocr_vl", "coat-1")}
    matching = SampleMatching(
        pairs=(SampleMatch(a_id="S1", b_id="coat-1", confidence=0.9, justification="j", method="llm"),)
    )
    report = compare_lanes(lanes["mineru"], lanes["paddleocr_vl"], {"coating": matching}, options)
    document = DocumentInput(document_id="d" * 64, sha256="d" * 64, pdf_path=Path("paper.pdf"))
    dataset = consolidate_document(document, lanes, report, options)
    payload = {"sample_rows": [dict(row) for row in dataset.sample_rows], "quality_rows": list(dataset.quality_rows)}
    gold = {
        "doc_id": "d",
        "samples": [
            {"id": "c1", "entity": "coating", "match": {"label": "S1"}, "fields": {"solvent": [{"value": "ethanol"}]}}
        ],
    }

    assert [row["entity"] for row in dataset.sample_rows] == ["coating"]
    cells = score.score_document(profile.by_name, gold, payload)
    assert [(cell.field, cell.outcome) for cell in cells if cell.field == "solvent"] == [("solvent", "correct")]


@pytest.mark.parametrize("entity", [None, "coatings"])
def test_a_gold_sample_must_name_a_declared_entity(entity):
    from support.profiles import make_entity_profile

    sample = {"id": "c1", "match": {"label": "S1"}, "fields": {}} | ({"entity": entity} if entity else {})
    gold = {"doc_id": "doc-7", "samples": [sample]}

    with pytest.raises(ValueError, match=r"doc-7: gold sample 'c1' names entity .*coating, wear_test"):
        score.score_document(make_entity_profile().by_name, gold, {"sample_rows": [], "quality_rows": []})
