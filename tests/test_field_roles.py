"""Every FieldSpec attribute says which stages read it, and the cache keys agree with what it says.

An attribute nobody classified would land in a key by accident, or miss one it should be in. The roles are
the classification: this pins them, and pins that today's key material follows from them.
"""

from __future__ import annotations

import dataclasses

from paperfacts import keys
from paperfacts.fields import FieldRole, FieldSpec, field_roles

P, C, V, R, F, D = (
    FieldRole.PROMPT,
    FieldRole.CLEANING,
    FieldRole.VERDICT,
    FieldRole.RETRIEVAL,
    FieldRole.FIGURE,
    FieldRole.DISPLAY,
)
# The role table of the profile spec (section 3.1). Changing a row is a decision about which key an
# attribute belongs to.
ROLES = {
    "name": {P, F},
    "group": {P},
    "kind": {P},
    "description": {P, F},
    "keywords": {R, F},
    "canonical_unit": {P, F},
    "label": {D},
    "description_zh": {D},
    "rel_tol": {V},
    "abs_tol": {V},
    "condition_hint": {P},
    "bare_number": {C, F},
    "categories": {V},
    "valid_range": {P, C},
    "condition_preference": {V},
    "level": {P, C, V},
    "condition_rule": {P},
    "missing_condition_note_zh": {V},
    "figure_readable": {F},
    "display_format": {D},
    "range_policy": {C, V},
}
# Attributes no stage reads yet: excluded from every key until the code that reads them lands.
NOT_YET_READ = {
    "level",
    "condition_rule",
    "missing_condition_note_zh",
    "figure_readable",
    "display_format",
    "range_policy",
}


def test_every_attribute_has_its_roles():
    assert {item.name: set(item.metadata["roles"]) for item in dataclasses.fields(FieldSpec)} == ROLES


def test_display_is_a_role_of_its_own():
    for name, roles in ROLES.items():
        assert D not in roles or roles == {D}, name


def test_field_roles_reads_the_metadata():
    assert field_roles("valid_range") == frozenset({P, C})


def test_the_extraction_schema_hashes_exactly_the_prompt_and_cleaning_attributes_in_use():
    hashed = set(ROLES) - keys._SCHEMA_EXCLUDED - keys._VERDICT_ONLY

    assert hashed == {name for name, roles in ROLES.items() if roles & {P, C}} - NOT_YET_READ


def test_display_text_reaches_no_key():
    assert {name for name, roles in ROLES.items() if roles == {D}} <= keys._SCHEMA_EXCLUDED


def test_a_verdict_only_attribute_stays_out_of_the_extraction_key():
    for name in keys._VERDICT_ONLY:
        assert V in ROLES[name] and not ROLES[name] & {P, C}, name


def test_the_attributes_no_stage_reads_yet_are_in_no_key():
    assert NOT_YET_READ <= keys._SCHEMA_EXCLUDED
