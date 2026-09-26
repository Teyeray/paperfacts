"""Every profile value that reaches a request the pipeline sends moves the key the stored answer is filed under.

The keys hash the system prompts by value and the code by source, so a value that reaches only a *user* prompt
(``implausible_origin`` in a passage-mode field line) would otherwise be served stale lanes after an edit. Rather
than list the slots and attributes by hand, this walks every leaf of a profile -- each prompt slot, each attribute
of each field, the retrieval section, the ignored unit suffixes, the display copy -- changes it, re-renders every
request, and checks the implication in the direction that matters: a changed request means a changed key. A value
that is display only must change no key at all.
"""

from __future__ import annotations

import dataclasses
import re
import typing
from collections.abc import Iterator
from pathlib import Path

import pytest

from paperfacts.adapters import render_markdown
from paperfacts.fields import FieldRole, FieldSpec, field_roles
from paperfacts.keys import ComparisonOptions, ExtractionOptions, comparison_key, extractor_key
from paperfacts.passages import candidate_blocks, inventory_blocks
from paperfacts.profile import DomainProfile, EntitySpec, PromptSlots
from paperfacts.profile_loader import ENTITY_SLOTS, load_profile, parse_profile
from paperfacts.prompts import (
    extraction_system_prompt,
    field_system_prompt,
    field_user_prompt,
    inventory_system_prompt,
    inventory_user_prompt,
    matching_system_prompt,
)
from support.factories import make_block
from support.profiles import SHIPPED_PROFILE_PATH, make_entity_profile, make_profile, profile_data

_REPOSITORY = Path(__file__).resolve().parents[1]
# A few blocks every field can find something in: a title, prose with numbers and units, a table and its caption.
_BLOCKS = (
    make_block(order=0, type="title", content="2. Experimental"),
    make_block(
        order=1,
        content=(
            "Films were annealed at 500 °C for 2 h; sheet resistance 12 Ω/sq, thickness 100 nm, transmittance"
            " 85 % at 550 nm, capacity 150 mAh/g at 0.1 C between 2.8-4.3 V, coated at 100 °C for 30 min."
        ),
    ),
    make_block(order=2, type="table", content="| Sample | Rs (Ω/sq) | d (nm) |\n| S1 | 12 | 100 |"),
    make_block(order=3, type="caption", content="Table 1. Films deposited at 100 W and 0.5 Pa."),
)
_SAMPLE_LIST = "- S1: films annealed at 500 °C"
_FIELD_TYPES = typing.get_type_hints(FieldSpec)
# One field of each kind that adds to a request (the holds key, a field-line note).
_KIND_FIELDS = [
    {"name": "doped", "group": "coating", "kind": "boolean", "description": "Doped or not.", "keywords": ["doped"]},
    {"name": "made_on", "group": "precursor", "kind": "date", "description": "Date made.", "keywords": ["prepared"]},
    {
        "name": "anneal_window",
        "group": "coating",
        "kind": "interval",
        "description": "Annealing window.",
        "keywords": ["annealed"],
        "canonical_unit": "℃",
        "valid_range": {"max": 1500},
    },
]


def _profiles() -> dict[str, DomainProfile]:
    return {
        "tco": load_profile(SHIPPED_PROFILE_PATH),
        "battery_cathode": load_profile(_REPOSITORY / "profiles" / "battery_cathode.json"),
        "demo": make_profile(),
        "many": _list_profile(),
        "kinds": make_profile({"fields": [*profile_data()["fields"], *_KIND_FIELDS]}),
        "entities": make_entity_profile(),
    }


def _list_profile() -> DomainProfile:
    """The demo profile with list fields (``cardinality: many``): a categorical one, whose categories reach the
    field line as ``prompt_categories``, and a plain one."""
    data = profile_data({"fields.2.cardinality": "many"})
    data["fields"].append(
        {
            "name": "characterization_techniques",
            "group": "precursor",
            "kind": "text",
            "cardinality": "many",
            "categories": ["XRD", "XPS", "TEM"],
            "description": "Each characterization technique the paper applies.",
            "keywords": ["characterization"],
        }
    )
    return parse_profile(data, Path("profiles/demo.json"))


def _requests(profile: DomainProfile) -> dict[str, tuple[str, ...]]:
    """Every request text the pipeline can send under ``profile``, per key that must cover it: each entity type's
    inventory, field and matching questions, and each field's question with its entity's sample list heading."""
    field_users = tuple(
        field_user_prompt(
            spec,
            _SAMPLE_LIST,
            render_markdown(candidate_blocks(spec, _BLOCKS, units=profile.units)),
            profile.prompt.implausible_origin,
            profile.entity_of(spec).prompt.sample_list_heading,
        )
        for spec in profile.fields
    )
    entities = profile.entities
    return {
        "document": (extraction_system_prompt(profile),),
        "passage": (
            *(inventory_system_prompt(profile, entity) for entity in entities),
            *(inventory_user_prompt(render_markdown(inventory_blocks(_BLOCKS, e.retrieval))) for e in entities),
            *(field_system_prompt(profile, entity) for entity in entities),
            *field_users,
        ),
        "comparison": tuple(matching_system_prompt(profile, entity) for entity in entities),
    }


def _keys(profile: DomainProfile) -> dict[str, str]:
    return {
        "document": extractor_key(ExtractionOptions(profile=profile, model="m", mode="document")),
        "passage": extractor_key(ExtractionOptions(profile=profile, model="m", mode="passage")),
        "comparison": comparison_key(ComparisonOptions(profile=profile, ambiguous_match_confidence=0.6)),
    }


def _other(value: object, annotation: object = None) -> object:
    """A different value of the same type."""
    options = typing.get_args(annotation) if typing.get_origin(annotation) is typing.Literal else ()
    if options:
        return next(option for option in options if option != value)
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 0.5
    if isinstance(value, str):
        return f"{value} edited"
    if value is None:
        return "edited"
    if isinstance(value, tuple) and len(value) == 2 and all(end is None or isinstance(end, float) for end in value):
        low, high = value
        return (0.0 if low is None else low - 1, 1e6 if high is None else high + 1)
    if isinstance(value, tuple):
        return (*value, "edited")
    raise AssertionError(f"no other value for {value!r}")


def _edited(profile: DomainProfile, label: str, **changes: object) -> DomainProfile:
    # A distinct content_hash spreads the variants over the profile-keyed caches instead of one hash bucket.
    edited = dataclasses.replace(profile, content_hash=label, **changes)
    if "declared_entities" in changes or not profile.declared_entities:
        return edited
    # A declared entity holds the profile's slots and retrieval resolved under its overrides, as the loader
    # builds it: an edit of the profile's own reaches every entity that does not override it.
    entities = tuple(
        dataclasses.replace(
            entity,
            prompt=dataclasses.replace(edited.prompt, **{n: getattr(entity.prompt, n) for n in entity.overrides}),
            retrieval=dataclasses.replace(
                edited.retrieval,
                **{
                    item.name: getattr(entity.retrieval, item.name)
                    for item in dataclasses.fields(entity.retrieval)
                    if getattr(entity.retrieval, item.name) != getattr(profile.retrieval, item.name)
                },
            ),
        )
        for entity in profile.declared_entities
    )
    return dataclasses.replace(edited, declared_entities=entities)


def _variants(profile: DomainProfile) -> Iterator[tuple[str, DomainProfile, bool]]:
    """``(what was changed, the edited profile, whether it is display only)`` for every leaf value."""
    for slot in dataclasses.fields(PromptSlots):
        value = _other(getattr(profile.prompt, slot.name))
        label = f"prompt.{slot.name}"
        yield label, _edited(profile, label, prompt=dataclasses.replace(profile.prompt, **{slot.name: value})), False
    for index, spec in enumerate(profile.fields):
        for attribute in dataclasses.fields(FieldSpec):
            if attribute.name in ("level", "entity"):
                continue  # the group's, never set on its own
            if attribute.name == "group":
                others = [g.name for g in profile.groups if g.level == spec.level and g.name != spec.group]
                if not others:
                    continue
                value: object = others[0]
            else:
                value = _other(getattr(spec, attribute.name), _FIELD_TYPES[attribute.name])
            fields = list(profile.fields)
            fields[index] = dataclasses.replace(spec, **{attribute.name: value})
            label = f"{spec.name}.{attribute.name}"
            display = field_roles(attribute.name) == {FieldRole.DISPLAY}
            yield label, _edited(profile, label, fields=tuple(fields)), display
    retrieval = profile.retrieval
    for name, value in (
        ("condition_keywords", (*retrieval.condition_keywords, "deposited")),
        ("condition_unit_pattern", f"{retrieval.condition_unit_pattern}|\\d\\s*W\\b"),
    ):
        label = f"retrieval.{name}"
        yield label, _edited(profile, label, retrieval=dataclasses.replace(retrieval, **{name: value})), False
    yield from _entity_variants(profile)
    units = dataclasses.replace(profile.units, ignored_suffixes=(*profile.units.ignored_suffixes, "Xe"))
    yield "units.ignored_suffixes", _edited(profile, "units.ignored_suffixes", units=units), False
    for name in ("title_zh", "description_zh"):
        yield name, _edited(profile, name, **{name: f"{getattr(profile, name)} edited"}), True
    for item in dataclasses.fields(profile.ui):
        label = f"ui.{item.name}"
        ui = dataclasses.replace(profile.ui, **{item.name: _other(getattr(profile.ui, item.name))})
        yield label, _edited(profile, label, ui=ui), True


def _entity_variants(profile: DomainProfile) -> Iterator[tuple[str, DomainProfile, bool]]:
    """Every value a declared entity type sets: each slot it may override (set as the loader would, recorded
    among its overrides), each key of its retrieval, and its display label."""

    def with_entity(label: str, index: int, entity: EntitySpec) -> DomainProfile:
        entities = list(profile.declared_entities)
        entities[index] = entity
        return _edited(profile, label, declared_entities=tuple(entities))

    for index, entity in enumerate(profile.declared_entities):
        for slot in dataclasses.fields(PromptSlots):
            if slot.name not in ENTITY_SLOTS and not slot.name.startswith("matching_"):
                continue
            prompt = dataclasses.replace(entity.prompt, **{slot.name: _other(getattr(entity.prompt, slot.name))})
            edited = dataclasses.replace(entity, prompt=prompt, overrides=entity.overrides | {slot.name})
            label = f"entities.{entity.name}.prompt.{slot.name}"
            yield label, with_entity(label, index, edited), False
        retrieval = entity.retrieval
        for name, value in (
            ("condition_keywords", (*retrieval.condition_keywords, "deposited")),
            ("condition_unit_pattern", f"{retrieval.condition_unit_pattern}|\\d\\s*W\\b"),
        ):
            label = f"entities.{entity.name}.retrieval.{name}"
            edited = dataclasses.replace(entity, retrieval=dataclasses.replace(retrieval, **{name: value}))
            yield label, with_entity(label, index, edited), False
        label = f"entities.{entity.name}.label_zh"
        yield label, with_entity(label, index, dataclasses.replace(entity, label_zh="改名")), True


PROFILES = ["tco", "battery_cathode", "demo", "many", "kinds", "entities"]


@pytest.mark.parametrize("name", PROFILES)
def test_every_value_that_changes_a_request_changes_its_key(name: str):
    profile = _profiles()[name]
    requests, keys = _requests(profile), _keys(profile)
    uncovered: list[str] = []
    for label, edited, display in _variants(profile):
        edited_requests, edited_keys = _requests(edited), _keys(edited)
        for target in keys:
            if display and edited_keys[target] != keys[target]:
                uncovered.append(f"{label}: display text moved the {target} key")
            if edited_requests[target] != requests[target] and edited_keys[target] == keys[target]:
                uncovered.append(f"{label}: a {target} request changed and its key did not")

    assert uncovered == []


@pytest.mark.parametrize("name", PROFILES)
def test_every_prompt_slot_is_hashed_by_value(name: str):
    # Whether or not a system prompt shows it today: a slot at its default in one profile, or read only by a user
    # prompt, must still move the key its prompts are filed under.
    profile = _profiles()[name]
    keys = _keys(profile)
    unmoved = []
    for label, edited, _ in _variants(profile):
        slot = re.fullmatch(r"(?:entities\.\w+\.)?prompt\.(\w+)", label)
        if slot is None:
            continue
        edited_keys = _keys(edited)
        targets = ("comparison",) if slot.group(1).startswith("matching_") else ("document", "passage")
        unmoved += [f"{label}: {target}" for target in targets if edited_keys[target] == keys[target]]

    assert unmoved == []
