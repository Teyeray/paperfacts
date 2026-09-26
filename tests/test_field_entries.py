"""A field entry in a profile: which keys it takes, what the optional ones default to, how each is
validated, and which cache key each one moves. Split from ``test_config_file.py`` when the field table left
``config.json`` for ``profiles/<name>.json``."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from paperfacts import keys
from paperfacts.config import (
    Settings,
)
from paperfacts.errors import ConfigError
from paperfacts.fields import FieldRole, FieldSpec
from paperfacts.profile_loader import parse_profile
from support.profiles import SHIPPED_PROFILE_PATH, make_profile

# A field entry with only the keys that have no default, used to check what the optional ones fall back to.
MINIMAL_FIELD: dict[str, Any] = {
    "name": "thickness",
    "group": "film",
    "kind": "numeric",
    "description": "Film thickness.",
    "keywords": ["thickness"],
}


def load_fields(entries: list[Any]) -> tuple[FieldSpec, ...]:
    """``entries`` as the shipped profile's field table, validated the way every field table is: config.json no
    longer has one of its own that anything reads, and the entry rules are the profile's."""
    # No chart slots: none of these entries is figure_readable, and a profile has slots only when one is.
    data = json.loads(SHIPPED_PROFILE_PATH.read_text(encoding="utf-8")) | {"fields": entries, "figures": None}
    return parse_profile(data, Path("profiles/tco.json")).fields


# ---- The field table -------------------------------------------------------------------


def test_a_minimal_field_entry_fills_in_the_optional_keys():
    specs = load_fields([MINIMAL_FIELD])

    assert len(specs) == 1
    spec = specs[0]
    assert (spec.name, spec.group, spec.kind) == ("thickness", "film", "numeric")
    assert spec.keywords == ("thickness",)
    assert spec.canonical_unit is None
    assert (spec.rel_tol, spec.abs_tol) == (0.0, 0.0)
    assert spec.condition_hint is None
    assert spec.bare_number == "reject"


def test_every_optional_key_is_read_when_it_is_given():
    entry = MINIMAL_FIELD | {
        "canonical_unit": "nm",
        "rel_tol": 0.05,
        "abs_tol": 1,
        "condition_hint": "wavelength",
        "bare_number": "assume_canonical",
    }

    spec = load_fields([entry])[0]

    assert spec.canonical_unit == "nm"
    assert (spec.rel_tol, spec.abs_tol) == (0.05, 1.0)
    assert spec.condition_hint == "wavelength"
    assert spec.bare_number == "assume_canonical"


def test_a_field_label_is_optional_and_read_when_it_is_given():
    assert load_fields([MINIMAL_FIELD])[0].label == ""

    spec = load_fields([MINIMAL_FIELD | {"label": "厚度"}])[0]

    assert spec.label == "厚度"


@pytest.mark.parametrize("bad", ["", "  ", 5, None])
def test_a_label_that_is_not_a_non_empty_string_is_refused(bad):
    with pytest.raises(ConfigError, match="label"):
        load_fields([MINIMAL_FIELD | {"label": bad}])


def test_every_shipped_field_has_a_chinese_label(tco_profile):
    assert all(spec.label for spec in tco_profile.fields)


def keys_under(profile, specs) -> tuple[str, str, str]:
    """The document- and passage-mode extractor keys and the comparison key of ``profile`` with its field table
    replaced by ``specs``."""
    edited = dataclasses.replace(profile, fields=specs)
    return (
        keys.extractor_key(keys.ExtractionOptions(edited, "a-model", mode="document")),
        keys.extractor_key(keys.ExtractionOptions(edited, "a-model", mode="passage")),
        keys.comparison_key_for(Settings(), edited),
    )


def test_a_label_changes_neither_cache_key(tco_profile):
    # It is a column header, nothing more: adding or editing one must not re-extract or re-compare a paper.
    plain = load_fields([MINIMAL_FIELD])
    labelled = load_fields([MINIMAL_FIELD | {"label": "厚度"}])

    assert keys_under(tco_profile, plain) == keys_under(tco_profile, labelled)


def test_a_chinese_description_is_optional_and_read_when_it_is_given():
    assert load_fields([MINIMAL_FIELD])[0].description_zh == ""

    spec = load_fields([MINIMAL_FIELD | {"description_zh": "薄膜厚度"}])[0]

    assert spec.description_zh == "薄膜厚度"


@pytest.mark.parametrize("bad", ["", "  ", 5, None])
def test_a_chinese_description_that_is_not_a_non_empty_string_is_refused(bad):
    with pytest.raises(ConfigError, match="description_zh"):
        load_fields([MINIMAL_FIELD | {"description_zh": bad}])


def test_every_shipped_field_has_a_chinese_description(tco_profile):
    assert all(spec.description_zh for spec in tco_profile.fields)


def test_a_chinese_description_changes_neither_cache_key(tco_profile):
    # Like the label: it reaches a tooltip and a spreadsheet sheet, never a prompt and never a verdict.
    plain = load_fields([MINIMAL_FIELD])
    described = load_fields([MINIMAL_FIELD | {"description_zh": "薄膜厚度"}])

    assert keys_under(tco_profile, plain) == keys_under(tco_profile, described)


def test_an_unknown_key_in_a_field_names_the_field_and_the_valid_keys():
    # The likeliest edit is a typo, and "keyword" for "keywords" would otherwise extract nothing.
    entry = MINIMAL_FIELD | {"keyword": ["thickness"]}

    with pytest.raises(ConfigError) as excinfo:
        load_fields([entry])

    message = str(excinfo.value)
    assert "thickness" in message and "keyword" in message and "keywords" in message


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        pytest.param({"group": "films"}, "group", id="group"),
        pytest.param({"kind": "number"}, "kind", id="kind"),
        pytest.param({"bare_number": "percent"}, "bare_number", id="bare-number"),
        pytest.param({"bare_number": None}, "bare_number", id="bare-number-null"),
        pytest.param({"rel_tol": "0.05"}, "rel_tol", id="tolerance-as-text"),
        pytest.param({"abs_tol": True}, "abs_tol", id="tolerance-as-boolean"),
        pytest.param({"keywords": "thickness"}, "keywords", id="keywords-not-a-list"),
        pytest.param({"keywords": ["thickness", ""]}, "keywords", id="keywords-with-an-empty-string"),
        pytest.param({"keywords": ["thickness", 7]}, "keywords", id="keywords-with-a-number"),
        pytest.param({"description": "   "}, "description", id="blank-description"),
        pytest.param({"canonical_unit": 5}, "canonical_unit", id="unit-not-a-string"),
        pytest.param({"rel_tol": -0.05}, "rel_tol", id="negative-relative-tolerance"),
        pytest.param({"abs_tol": -1}, "abs_tol", id="negative-absolute-tolerance"),
        pytest.param(
            {"canonical_unit": "nm", "bare_number": "percent_or_fraction"}, "bare_number", id="fraction-on-nm"
        ),
        pytest.param({"bare_number": "percent_or_fraction"}, "bare_number", id="fraction-without-a-unit"),
    ],
)
def test_an_invalid_field_value_names_the_field_and_the_key(change: dict[str, Any], expected: str):
    entry = MINIMAL_FIELD | change

    with pytest.raises(ConfigError) as excinfo:
        load_fields([entry])

    message = str(excinfo.value)
    assert "thickness" in message and expected in message


@pytest.mark.parametrize("name", [None, "", "   ", 7], ids=["missing", "empty", "blank", "not-a-string"])
def test_a_field_without_a_usable_name_is_refused(name: Any):
    entry = MINIMAL_FIELD | {"name": name}
    if name is None:
        del entry["name"]

    with pytest.raises(ConfigError, match="name"):
        load_fields([entry])


def test_a_field_without_a_description_is_refused():
    entry = {key: value for key, value in MINIMAL_FIELD.items() if key != "description"}

    with pytest.raises(ConfigError, match="description"):
        load_fields([entry])


def test_a_field_that_is_not_an_object_names_its_position():
    with pytest.raises(ConfigError, match=r"fields\[1\]"):
        load_fields([MINIMAL_FIELD, "thickness"])


def test_an_empty_field_table_is_refused():
    # An empty table would extract nothing at all, and would do it without a word.
    with pytest.raises(ConfigError, match="at least one field"):
        load_fields([])


def test_two_fields_with_the_same_name_are_refused():
    # A table by name would keep the last one, so the earlier entry would be configured but never used.
    with pytest.raises(ConfigError, match="thickness"):
        load_fields([MINIMAL_FIELD, MINIMAL_FIELD | {"canonical_unit": "nm"}])


@pytest.mark.parametrize("words", [["power", ""], ["power", 7], ["power", None]], ids=["empty", "number", "null"])
def test_condition_keywords_must_all_be_non_empty_strings(words: list[Any]):
    with pytest.raises(ConfigError, match="condition_keywords"):
        make_profile({"retrieval.condition_keywords": words})


def test_condition_keywords_come_back_in_the_order_they_were_written():
    words = make_profile({"retrieval.condition_keywords": ["power", "pressure", "flow rate"]})

    assert words.retrieval.condition_keywords == ("power", "pressure", "flow rate")


# ---- valid_range -------------------------------------------------------------------------

RANGED_FIELD: dict[str, Any] = MINIMAL_FIELD | {"canonical_unit": "nm"}


def test_a_field_without_a_range_accepts_every_value():
    spec = load_fields([RANGED_FIELD])[0]

    assert spec.valid_range == (None, None)
    assert spec.describe_range() is None
    assert spec.in_range(1e9)


@pytest.mark.parametrize(
    ("bounds", "described", "inside", "outside"),
    [
        pytest.param({"max": 500}, "at most 500 nm", 500.0, 501.0, id="ceiling"),
        pytest.param({"min": 60}, "at least 60 nm", 60.0, 59.9, id="floor"),
        pytest.param({"min": 1, "max": 2.5}, "between 1 and 2.5 nm", 2.0, 3.0, id="both"),
    ],
)
def test_a_range_is_read_in_the_canonical_unit_with_either_end_open(bounds, described, inside, outside):
    spec = load_fields([RANGED_FIELD | {"valid_range": bounds}])[0]

    assert spec.describe_range() == described
    assert spec.in_range(inside)
    assert not spec.in_range(outside)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(RANGED_FIELD | {"valid_range": [0, 500]}, "object", id="not-an-object"),
        pytest.param(RANGED_FIELD | {"valid_range": {"maximum": 500}}, "object", id="unknown-bound"),
        pytest.param(RANGED_FIELD | {"valid_range": {}}, "at least one", id="empty"),
        pytest.param(RANGED_FIELD | {"valid_range": {"max": "500"}}, "valid_range.max", id="bound-as-text"),
        pytest.param(RANGED_FIELD | {"valid_range": {"max": True}}, "valid_range.max", id="bound-as-boolean"),
        pytest.param(RANGED_FIELD | {"valid_range": {"min": 5, "max": 5}}, "below", id="empty-interval"),
        pytest.param(MINIMAL_FIELD | {"kind": "text", "valid_range": {"max": 5}}, "numeric", id="not-numeric"),
    ],
)
def test_a_malformed_range_names_the_field_and_the_problem(entry, expected):
    with pytest.raises(ConfigError, match=expected) as excinfo:
        load_fields([entry])

    assert "thickness" in str(excinfo.value)


def test_a_range_moves_both_cache_keys_and_its_absence_moves_neither(tco_profile):
    # It changes what the model is told and which values survive, so it must re-extract; but a table that
    # declares no range has to keep the keys it had before ranges existed.
    plain = load_fields([RANGED_FIELD])
    ranged = load_fields([RANGED_FIELD | {"valid_range": {"max": 500}}])

    everything = (FieldRole.PROMPT, FieldRole.CLEANING, FieldRole.VERDICT)
    assert "valid_range" not in keys._field_material(plain[0], *everything)
    assert "valid_range" in keys._field_material(ranged[0], *everything)
    document_key, passage_key, comparison = keys_under(tco_profile, plain)
    ranged_document, ranged_passage, ranged_comparison = keys_under(tco_profile, ranged)
    assert ranged_document != document_key and ranged_passage != passage_key and ranged_comparison != comparison


# Every FieldSpec cell the extraction key hashes: what the model is told, and what cleaning its answer reads.
EXTRACTION_CELLS = {
    "name",
    "group",
    "kind",
    "description",
    "canonical_unit",
    "condition_hint",
    "bare_number",
    "valid_range",
    "level",
    "condition_rule",
    "range_policy",
    "after_clause",
    "cardinality",
    "prompt_categories",
    "entity",
    "references",
}


def test_every_field_cell_is_classified_for_the_cache_keys():
    # A cell nobody classified lands in no key or the wrong one. Its roles (fields.py) decide; the extraction
    # key hashes the PROMPT and CLEANING ones.
    assert set(keys.attributes_with(FieldRole.PROMPT, FieldRole.CLEANING)) == EXTRACTION_CELLS


def test_a_tolerance_moves_only_the_comparison_key(tco_profile):
    # A tolerance decides whether two quoted values agree; the model is never told it and no cleaning rule
    # reads it, so editing one must leave every stored extraction where it is.
    plain = load_fields([RANGED_FIELD])
    tolerant = load_fields([RANGED_FIELD | {"rel_tol": 0.1, "abs_tol": 2}])

    document_key, passage_key, comparison = keys_under(tco_profile, plain)
    document_after, passage_after, comparison_after = keys_under(tco_profile, tolerant)
    assert (document_key, passage_key) == (document_after, passage_after)
    assert comparison != comparison_after


@pytest.mark.parametrize("mode", ["document", "passage"])
def test_the_range_sentence_the_model_reads_is_part_of_the_extractor_key(monkeypatch, mode, tco_profile):
    # The "Plausible values are ..." line is written by FieldSpec.describe_range and reaches every question,
    # in passage mode through the field question's user half, which is not hashed by value there. The
    # rendered field table is in the key by value (the document prompt carries it), so a change in how
    # the range is worded re-keys both modes, not just a change of the range itself.

    before = keys.extractor_key(keys.ExtractionOptions(tco_profile, "a-model", mode=mode))
    monkeypatch.setattr(FieldSpec, "describe_range", lambda self: "no more than a little")

    assert keys.extractor_key(keys.ExtractionOptions(tco_profile, "a-model", mode=mode)) != before


def test_fields_py_is_part_of_the_extraction_code_fingerprint(monkeypatch):
    # It also decides which fields are sample-level, which gates which questions are asked at all.
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(keys, "source_fingerprint", lambda *files: seen.append(files) or "x")
    keys.extraction_code_fingerprint.cache_clear()
    try:
        keys.extraction_code_fingerprint()
    finally:
        keys.extraction_code_fingerprint.cache_clear()

    assert "fields.py" in seen[0]


# ---- condition_preference -----------------------------------------------------------------


def test_a_condition_preference_is_read_in_order():
    spec = load_fields([RANGED_FIELD | {"condition_preference": ["400-800", "550"]}])[0]

    assert spec.condition_preference == ("400-800", "550")


@pytest.mark.parametrize("bad", ["400-800", [""], ["visible"], [550]])
def test_a_condition_preference_must_be_a_list_of_entries_naming_numbers(bad):
    with pytest.raises(ConfigError, match="condition_preference"):
        load_fields([RANGED_FIELD | {"condition_preference": bad}])


def test_a_condition_preference_moves_only_the_comparison_key(tco_profile):
    # It picks a dataset cell among values already extracted; the model never hears of it.
    plain = load_fields([RANGED_FIELD])
    preferring = load_fields([RANGED_FIELD | {"condition_preference": ["550"]}])

    (document_key, passage_key, comparison), after = keys_under(tco_profile, plain), keys_under(tco_profile, preferring)
    assert (document_key, passage_key) == after[:2]
    assert comparison != after[2]


def test_a_canonical_unit_with_no_converter_names_the_field_and_the_file():
    with pytest.raises(ConfigError, match=r"tco\.json: field 'thickness': canonical_unit 'furlong'"):
        load_fields([MINIMAL_FIELD | {"canonical_unit": "furlong"}])
