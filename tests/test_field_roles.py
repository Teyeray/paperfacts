"""Every FieldSpec attribute says which stages read it, and the cache keys agree with what it says.

An attribute nobody classified would land in a key by accident, or miss one it should be in. The roles are
the classification: this pins them, and pins that the key material follows from them.
"""

from __future__ import annotations

import dataclasses

import pytest

from paperfacts import keys
from paperfacts.fields import FieldRole, FieldSpec, field_roles
from support.profiles import make_profile

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
# A value off each attribute's default, set on one field of the demo profile to see which fingerprints move.
EDITS = {
    "name": "coating_depth",
    "group": "precursor",
    "kind": "text",
    "description": "Something else.",
    "keywords": ("depth",),
    "canonical_unit": "μm",
    "label": "厚度",
    "description_zh": "涂层厚度",
    "rel_tol": 0.1,
    "abs_tol": 1.0,
    "condition_hint": "the wavelength",
    "bare_number": "assume_canonical",
    "categories": ("thin", "thick"),
    "valid_range": (0.0, 500.0),
    "condition_preference": ("550",),
    "level": "paper",
    "condition_rule": "the wavelength or spectral range",
    "missing_condition_note_zh": "未注明波长",
    "figure_readable": True,
    "display_format": "scientific",
    "range_policy": "reject",
}


def test_every_attribute_has_its_roles():
    assert {item.name: set(item.metadata["roles"]) for item in dataclasses.fields(FieldSpec)} == ROLES


def test_display_is_a_role_of_its_own():
    for name, roles in ROLES.items():
        assert D not in roles or roles == {D}, name


def test_field_roles_reads_the_metadata():
    assert field_roles("valid_range") == frozenset({P, C})


def _fingerprints(profile) -> dict[str, str]:
    return {
        "extraction": keys.profile_extraction_fingerprint(profile),
        "comparison": keys.profile_comparison_fingerprint(profile),
        "retrieval": keys.retrieval_fingerprint(profile),
    }


@pytest.mark.parametrize("attribute", sorted(ROLES))
def test_an_attribute_moves_exactly_the_fingerprints_its_roles_name(attribute):
    profile = make_profile()
    spec = profile.fields[1]
    edited = dataclasses.replace(
        profile,
        fields=(profile.fields[0], dataclasses.replace(spec, **{attribute: EDITS[attribute]}), *profile.fields[2:]),
    )
    assert getattr(spec, attribute) != EDITS[attribute]

    before, after = _fingerprints(profile), _fingerprints(edited)
    moved = {name for name in before if before[name] != after[name]}

    roles = ROLES[attribute]
    expected = {
        name for name, reads in (("extraction", {P, C}), ("comparison", {P, C, V}), ("retrieval", {R})) if roles & reads
    }
    assert moved == expected


def test_the_figure_material_is_exactly_the_figure_attributes():
    assert set(keys.attributes_with(F)) == {name for name, roles in ROLES.items() if F in roles}


def test_range_policy_enters_the_material_only_away_from_its_default():
    profile = make_profile()
    spec = profile.fields[1]
    assert spec.range_policy == "midpoint"
    assert "range_policy" not in keys._field_material(spec, C, V)
    rejecting = dataclasses.replace(spec, range_policy="reject")
    assert keys._field_material(rejecting, C, V)["range_policy"] == "reject"
