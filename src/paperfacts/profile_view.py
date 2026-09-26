"""What a profile looks like to a reader: the small view the page names things by, the full read-only definition,
and the system prompts it renders, as ``paperfacts prompts`` prints them and the web shows them.

Unhashed on purpose (``tests/test_keys_unhashed.py``): it only presents what the hashed modules decide. The prompt
preview is assembled here rather than in ``prompts.py``, whose source is hashed into every extraction key, so a
change to how the preview is laid out renames no stored result. Nothing here caches: a profile that was never
loaded from a file can be shown without growing any process-wide table.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from paperfacts.fields import FieldSpec
from paperfacts.figures import user_prompt as figure_user_prompt
from paperfacts.profile import DomainProfile
from paperfacts.prompts import (
    extraction_system_prompt,
    field_system_prompt,
    field_user_prompt,
    inventory_system_prompt,
    matching_system_prompt,
    render_field_table,
)
from paperfacts.records import KindContext


def profile_view(profile: DomainProfile) -> dict[str, Any]:
    """What the page needs to name things the profile's way: its title, its copy and its groups and fields.
    Display text only; the prompts, units and retrieval stay on the server."""
    return {
        "name": profile.name,
        "title_zh": profile.title_zh,
        "maturity": profile.maturity,
        "description_zh": profile.description_zh,
        "ui": dataclasses.asdict(profile.ui),
        "groups": [
            {"name": group.name, "level": group.level, "label_zh": group.label_zh, "entity": group.entity}
            for group in profile.groups
        ],
        # The kinds of sample, the primary first; a profile without entity types has the one implicit entity.
        "entities": [
            {
                "name": entity.name,
                "label_zh": entity.label_zh,
                "fields": [spec.name for spec in profile.entity_fields(entity)],
            }
            for entity in profile.entities
        ],
        "fields": [
            {
                "name": spec.name,
                "label": spec.label,
                "group": spec.group,
                "level": spec.level,
                "unit": spec.canonical_unit,
                "entity": spec.entity,
                "references": spec.references,
            }
            for spec in profile.fields
        ],
        "field_count": {"paper": len(profile.paper_fields), "sample": len(profile.sample_fields)},
    }


def profile_definition(profile: DomainProfile) -> dict[str, Any]:
    """Everything a reader needs to see what a profile asks and how it judges, for the read-only profile page.

    A field is every ``FieldSpec`` attribute, read off the dataclass rather than listed here, so an attribute added
    later shows up without an edit to this function or the page."""
    attributes = [attribute.name for attribute in dataclasses.fields(FieldSpec)]
    return {
        "name": profile.name,
        "title_zh": profile.title_zh,
        "maturity": profile.maturity,
        "description_zh": profile.description_zh,
        "content_hash": profile.content_hash[:12],
        "ui": dataclasses.asdict(profile.ui),
        "groups": [
            {"name": group.name, "level": group.level, "label_zh": group.label_zh, "entity": group.entity}
            for group in profile.groups
        ],
        "entities": [
            {
                "name": entity.name,
                "label_zh": entity.label_zh,
                "fields": [spec.name for spec in profile.entity_fields(entity)],
                "sample_list_heading": entity.prompt.sample_list_heading,
            }
            for entity in profile.entities
        ],
        "fields": [{name: getattr(spec, name) for name in attributes} for spec in profile.fields],
        "units": {
            "declared": [
                {
                    "canonical": unit.canonical,
                    "aliases": [
                        {"spelling": spelling, "factor": factor, "offset": offset}
                        for spelling, factor, offset in unit.aliases
                    ],
                    "case_sensitive": unit.case_sensitive,
                    "extends_builtin": unit.extends_builtin,
                    "exclude": list(unit.exclude),
                    "retrieval": unit.retrieval,
                }
                for unit in profile.units.declared
            ],
            "ignored_suffixes": list(profile.units.ignored_suffixes),
            "known": list(profile.units.known()),
        },
        "retrieval": {
            "condition_keywords": list(profile.retrieval.condition_keywords),
            "condition_unit_pattern": profile.retrieval.condition_unit_pattern,
        },
    }


def prompt_sections(profile: DomainProfile, field: str | None = None) -> list[tuple[str, str]]:
    """The system prompts ``profile`` renders, exactly as the model gets them, each titled with the mode that
    sends it; with ``field``, that field's question instead. Raises ``KeyError`` naming the profile's fields when
    it has no ``field``."""
    if field is not None:
        spec = profile.by_name.get(field)
        if spec is None:
            raise KeyError(f"no field {field!r} in {profile.name}; it has: {', '.join(profile.by_name)}")
        # Asked with its entity's system prompt and sample list, and a reference field with the list of the entity
        # it names too.
        entity = profile.entity_of(spec)
        entities = {declared.name: declared for declared in profile.entities}
        referenced = None if spec.references is None else (entities[spec.references], "<referenced sample list>")
        sections = {
            "field system prompt (passage mode)": field_system_prompt(profile, entity),
            f"field line ({field})": render_field_table(
                (spec,), profile.prompt.implausible_origin, KindContext(entities=entities)
            ),
            # The question's framing; the placeholders are what a run fills from the paper.
            f"field user prompt ({field}, passage mode)": field_user_prompt(
                spec,
                "<sample list>",
                "<excerpts>",
                profile.prompt.implausible_origin,
                entity.prompt.sample_list_heading,
                referenced,
            ),
        }
    elif len(profile.entities) > 1:
        # Passage mode only: each entity type has its own inventory, field and matching prompts.
        sections = {
            f"{title} ({entity.name})": render(profile, entity)
            for entity in profile.entities
            for title, render in (
                ("inventory system prompt", inventory_system_prompt),
                ("field system prompt", field_system_prompt),
                ("matching system prompt", matching_system_prompt),
            )
        }
    else:
        # Passage mode (the default) sends the inventory and field prompts and never the extraction prompt;
        # document mode sends only the extraction prompt. Matching is the compare stage's, under either mode.
        sections = {
            "inventory system prompt (passage mode)": inventory_system_prompt(profile),
            "field system prompt (passage mode)": field_system_prompt(profile),
            "extraction system prompt (document mode)": extraction_system_prompt(profile),
            "matching system prompt (compare, both modes)": matching_system_prompt(profile),
        }
    if field is None and profile.figures is not None and profile.figure_fields:
        # Sent once per chart panel, with that figure's own caption and only the fields the caption names.
        sections["figure user prompt (figures stage, only when figures.enabled; one per chart panel)"] = (
            figure_user_prompt("<caption>", profile.figure_fields, profile.figures)
        )
    return list(sections.items())
