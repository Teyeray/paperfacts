"""Synthetic domain profiles: a small, valid profile of a made-up domain, edited per test.

A test that needs a profile other than TCO's builds it here rather than writing its own JSON, so every
synthetic profile starts valid and a test changes only the part it is about.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from paperfacts.profile import DomainProfile, parse_profile

# One paper-level group and one sample-level group, a numeric field in each, and a text field: the least a
# profile needs to exercise both scopes.
_DEMO: dict[str, Any] = {
    "format": 1,
    "name": "demo",
    "title_zh": "示例领域",
    "maturity": "example",
    "groups": [
        {"name": "precursor", "level": "paper", "label_zh": "前驱体"},
        {"name": "coating", "level": "sample", "label_zh": "涂层"},
    ],
    "prompt": {
        "domain_subject": "sol-gel coatings",
        "sample_definition": "A sample is a coating this paper prepares itself.",
        "field_scope": "Every field describes the coating or its precursor.",
    },
    "retrieval": {"condition_keywords": ["annealed", "coating"], "condition_unit_pattern": r"\d\s*(?:°C|min\b)"},
    "fields": [
        {
            "name": "precursor_purity",
            "group": "precursor",
            "kind": "numeric",
            "description": "Purity of the precursor.",
            "keywords": ["purity"],
            "canonical_unit": "%",
        },
        {
            "name": "coating_thickness",
            "group": "coating",
            "kind": "numeric",
            "description": "Coating thickness.",
            "keywords": ["thickness"],
            "canonical_unit": "nm",
        },
        {
            "name": "solvent",
            "group": "coating",
            "kind": "text",
            "description": "Solvent the sol was made in.",
            "keywords": ["solvent"],
        },
    ],
}
# A change whose value is this removes the key instead of setting it.
DELETE = object()


def profile_data(changes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The demo profile's JSON with some dotted keys replaced (``"prompt.paper_key"``) or, for a value of
    :data:`DELETE`, removed. A list is indexed by position (``"fields.0.canonical_unit"``)."""
    data = copy.deepcopy(_DEMO)
    for dotted, value in (changes or {}).items():
        *parents, leaf = dotted.split(".")
        node: Any = data
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node[part]
        key: Any = int(leaf) if isinstance(node, list) else leaf
        if value is DELETE:
            del node[key]
        else:
            node[key] = value
    return data


def make_profile(changes: Mapping[str, Any] | None = None, *, source: Path | None = None) -> DomainProfile:
    """The demo profile, validated, with ``changes`` applied as in :func:`profile_data`."""
    data = profile_data(changes)
    return parse_profile(data, source or Path(f"profiles/{data['name']}.json"))
