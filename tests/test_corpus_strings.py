"""The number parser and the sample key over every string the real corpus holds.

``tests/fixtures/corpus/`` records the 571 distinct numeric ``value_raw`` strings and every lane's sample ids
from a copy of the server's data, each with how the reference commit (main before this rework) read it and
how the code reads it now (``generate.py`` there writes both). Two things are pinned: the code still reads
every string as recorded, and the recorded changes against the reference are exactly the intended ones --
so a rule change that moves anything else on real data fails here, with no corpus and no network.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from paperfacts.normalize import parse_number
from paperfacts.records import sample_key

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
VALUES = json.loads((CORPUS / "values.json").read_text(encoding="utf-8"))
PAIRS = json.loads((CORPUS / "sample_ids.json").read_text(encoding="utf-8"))

# Every value string whose number differs from the reference's, and why. Nothing else may differ.
INTENDED_VALUE_CHANGES = {
    "1:9": "a gas ratio; read as 1, it became 100 % through the fraction rule",
    "10/10": "a slash ratio, the same class as 1:9",
    "12/10/3": "a three-way slash ratio",
    "100 to 0": "a power ramp, a descending pair rather than a value",
    "$ \\sim $25 and 70": "two thicknesses joined by 'and'",
    "500 °C to 530 °C": "a range with a unit on each bound; the midpoint, not the first bound",
}


def _same(a: float | None, b: float | None) -> bool:
    return a == b or (a is not None and b is not None and a == pytest.approx(b))


@pytest.mark.parametrize("row", VALUES, ids=lambda row: f"{row['field']}:{row['value_raw']}")
def test_every_corpus_value_string_reads_as_recorded(row):
    value, note = parse_number(row["value_raw"])

    assert _same(value, row["expected"][0])
    assert note == row["expected"][1]


def test_the_changes_against_the_reference_are_only_the_intended_ones():
    changed = {row["value_raw"] for row in VALUES if not _same(row["reference"][0], row["expected"][0])}

    assert changed == set(INTENDED_VALUE_CHANGES)


def test_every_corpus_sample_id_keys_as_recorded():
    mismatched = [
        entry["id"]
        for pair in PAIRS
        for entries in pair["lanes"].values()
        for entry in entries
        if sample_key(entry["id"]) != entry["expected"]
    ]

    assert mismatched == []


def test_no_two_ids_of_one_lane_share_a_key():
    # The stored lanes were de-duplicated by the reference key; a key that merged two of their ids would put
    # one sample's values on another.
    merged = []
    for pair in PAIRS:
        for entries in pair["lanes"].values():
            keys = collections.Counter(sample_key(entry["id"]) for entry in entries)
            merged.extend(key for key, count in keys.items() if count > 1)

    assert merged == []


def test_every_exact_cross_lane_pair_of_the_reference_still_pairs():
    # The key may pair more ids across lanes than normalize_key did (separator differences), never fewer.
    lost, gained = [], 0
    for pair in PAIRS:
        if set(pair["lanes"]) != {"mineru", "paddleocr_vl"}:
            continue
        paddle = pair["lanes"]["paddleocr_vl"]
        by_reference = {entry["reference"] for entry in paddle}
        by_key = {sample_key(entry["id"]) for entry in paddle}
        for entry in pair["lanes"]["mineru"]:
            key = sample_key(entry["id"])
            if entry["reference"] in by_reference and key not in by_key:
                lost.append(entry["id"])
            gained += entry["reference"] not in by_reference and bool(key) and key in by_key

    assert lost == []
    assert gained > 0
