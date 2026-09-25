"""The cache keys follow the roles of what a profile edit touches, and nothing else.

A display edit must rename no stored result, a verdict edit only the comparisons, a retrieval edit only the
passage-mode extractions: each wrong answer either throws away results that are still valid or serves ones
that are not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperfacts import keys
from paperfacts.config import Settings
from paperfacts.profile import DomainProfile
from support.profiles import make_profile


def all_keys(profile: DomainProfile) -> dict[str, str]:
    return {
        "document": keys.extractor_key(keys.ExtractionOptions(profile, "a-model", mode="document")),
        "passage": keys.extractor_key(keys.ExtractionOptions(profile, "a-model", mode="passage")),
        "comparison": keys.comparison_key_for(Settings(), profile),
    }


def moved_keys(changes: dict[str, object]) -> set[str]:
    before, after = all_keys(make_profile()), all_keys(make_profile(changes))
    return {name for name in before if before[name] != after[name]}


@pytest.mark.parametrize(
    "changes",
    [
        {"fields.1.label": "厚度"},
        {"fields.1.description_zh": "涂层的厚度"},
        {"fields.1.display_format": "scientific"},
        {"groups.1.label_zh": "膜层"},
        {"title_zh": "另一个标题"},
        {"description_zh": "另一段说明"},
        {"maturity": "production"},
        {"ui": {"entity_label_zh": "涂层"}},
    ],
    ids=lambda changes: next(iter(changes)),
)
def test_a_display_edit_re_keys_nothing(changes):
    assert moved_keys(changes) == set()


@pytest.mark.parametrize(
    "changes",
    [
        {"fields.1.rel_tol": 0.05},
        {"fields.1.abs_tol": 2},
        {"fields.2.categories": ["water", "ethanol"]},
        {"fields.1.condition_preference": ["550"]},
    ],
    ids=lambda changes: next(iter(changes)),
)
def test_a_verdict_edit_changes_only_the_comparison_key(changes):
    assert moved_keys(changes) == {"comparison"}


@pytest.mark.parametrize(
    "changes",
    [
        {"fields.1.keywords": ["thickness", "film thickness"]},
        {"retrieval.condition_keywords": ["annealed", "coating", "cured"]},
        {"retrieval.condition_unit_pattern": r"\d\s*(?:°C|min\b|h\b)"},
    ],
    ids=lambda changes: next(iter(changes)),
)
def test_a_retrieval_edit_changes_only_the_passage_mode_extractor_key(changes):
    assert moved_keys(changes) == {"passage"}


@pytest.mark.parametrize(
    "changes",
    [
        {"fields.1.description": "Thickness of the dried coating."},
        {"fields.1.valid_range": {"max": 5000}},
    ],
    ids=lambda changes: next(iter(changes)),
)
def test_a_field_prompt_edit_changes_every_key(changes):
    # What the model is told re-extracts in both modes, and the comparison of those extractions is recomputed.
    assert moved_keys(changes) == {"document", "passage", "comparison"}


def test_a_prompt_slot_edit_re_extracts_in_both_modes():
    # A comparison is stored under both keys, so a new extractor key already gives it a new file; the
    # comparison key moves only for the slots the matching prompt reads.
    assert moved_keys({"prompt.domain_subject": "dip-coated films"}) == {"document", "passage"}
    assert moved_keys({"prompt.matching_justification_example": "both were dried at 80 °C"}) == {"comparison"}


def test_the_profile_file_name_is_not_hashed():
    # Copying a profile under another name (a second instance, a renamed file) keeps every stored result.
    renamed = make_profile({"name": "other"}, source=Path("elsewhere/other.json"))

    assert all_keys(renamed) == all_keys(make_profile())
