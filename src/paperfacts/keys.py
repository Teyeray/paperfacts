"""Cache keys: content fingerprints of everything a stored result depends on.

Two keys name the files under a document directory:

- :func:`extractor_key` covers the model, the part of the profile extraction reads (every field attribute
  with the PROMPT or CLEANING role, the groups, the declared units), the extraction prompts, how the document
  is rendered for the model, and the code that decides which of the model's claims survive. Changing any of
  them invalidates the stored extraction. The model's own answer is cached separately by request payload,
  so a code-only change re-derives records for free.
- :func:`comparison_key` covers that plus every VERDICT attribute (tolerances, categories, condition
  preferences), the normalisation rules and the sample-matching prompt. Changing a tolerance recomputes the
  comparison and leaves every stored extraction where it is.

A third, :func:`figure_key`, names the figure readings, which belong to neither lane.

Which key a field attribute reaches follows from its :class:`~paperfacts.fields.FieldRole` set alone; DISPLAY
text and the profile's file name reach none. Module sources are hashed instead of versioned by hand, so nobody
has to remember to bump a number.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from paperfacts import figures
from paperfacts.config import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_LLM_CONTEXT_TOKENS,
    DEFAULT_LLM_INVENTORY_REASONING_EFFORT,
    DEFAULT_LLM_REASONING_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    ExtractionMode,
    InventoryReasoningEffort,
    ReasoningEffort,
    Settings,
)
from paperfacts.fields import FieldRole, FieldSpec
from paperfacts.profile import DomainProfile, FigureSlots, PromptSlots
from paperfacts.prompts import (
    extraction_system_prompt,
    field_system_prompt,
    inventory_system_prompt,
    matching_system_prompt,
)

# Long enough that a collision is not a practical concern, short enough to read in a filename.
FINGERPRINT_LENGTH = 12
_PACKAGE_DIR = Path(__file__).parent
_NO_DEFAULT = object()
# Built once: the material of every field of every key compares against it.
_FIELD_DEFAULTS = {item.name: item.default for item in dataclasses.fields(FieldSpec)}
_FIGURE_SLOT_DEFAULTS = {item.name: item.default for item in dataclasses.fields(FigureSlots)}
_PROMPT_SLOT_DEFAULTS = {item.name: item.default for item in dataclasses.fields(PromptSlots)}
# The slots only the sample-matching prompt reads: comparison material, never extraction material.
_MATCHING_SLOT_PREFIX = "matching_"


def content_fingerprint(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def source_fingerprint(*module_files: str) -> str:
    """Fingerprint of the given files of this package, in the order given."""
    return content_fingerprint("\n".join((_PACKAGE_DIR / name).read_text(encoding="utf-8") for name in module_files))


def _dumps(material: object) -> str:
    return json.dumps(material, ensure_ascii=False, sort_keys=True)


@cache
def attributes_with(*roles: FieldRole) -> tuple[str, ...]:
    """The :class:`FieldSpec` attributes a stage with any of ``roles`` reads, in declaration order. Which key an
    attribute lands in follows from its roles alone, so adding one is a decision made where it is declared."""
    wanted = set(roles)
    return tuple(item.name for item in dataclasses.fields(FieldSpec) if item.metadata["roles"] & wanted)


def _field_material(spec: FieldSpec, *roles: FieldRole) -> dict[str, object]:
    """The attributes of ``spec`` with any of ``roles``, each left out while at its dataclass default: a field
    that does not use an attribute keeps the fingerprint it had before the attribute existed."""
    return {
        name: getattr(spec, name)
        for name in attributes_with(*roles)
        if getattr(spec, name) != _FIELD_DEFAULTS.get(name, _NO_DEFAULT)
    }


def _schema_material(profile: DomainProfile, *roles: FieldRole) -> dict[str, object]:
    material: dict[str, object] = {
        "fields": [_field_material(spec, *roles) for spec in profile.fields],
        # A group's level decides the scope rules and its entity which samples its fields describe (written only
        # when set, so a profile without entity types keeps its material); its Chinese label is display text.
        "groups": [[group.name, group.level, *([group.entity] if group.entity else [])] for group in profile.groups],
    }
    if profile.units.material():
        material["units"] = profile.units.material()
    if profile.declared_entities:
        # In order: the first is the primary entity, which the paper-level questions are asked beside.
        material["entities"] = [entity.name for entity in profile.declared_entities]
    return material


@cache
def profile_extraction_fingerprint(profile: DomainProfile) -> str:
    """The profile as extraction reads it: every field attribute that is rendered into a prompt or decides which
    of the model's values survive (and what they convert to), the groups, and the units the profile declares.
    Tolerances, categories, condition preferences and display text are left out: they only decide verdicts or
    what a page prints, so editing one never renames a stored extraction. Recorded on every lane
    (``LaneExtraction.profile_fingerprint``), so two lanes can be checked to come from the same profile."""
    return content_fingerprint(_dumps(_schema_material(profile, FieldRole.PROMPT, FieldRole.CLEANING)))


@cache
def profile_comparison_fingerprint(profile: DomainProfile) -> str:
    """The profile as a comparison reads it: what extraction reads plus every verdict attribute (tolerances,
    categories, condition preferences). The keywords stay out: they steer retrieval and nothing else, so a new
    synonym must not recompute every stored comparison."""
    return content_fingerprint(
        _dumps(_schema_material(profile, FieldRole.PROMPT, FieldRole.CLEANING, FieldRole.VERDICT))
    )


@cache
def retrieval_fingerprint(profile: DomainProfile) -> str:
    """Everything that decides which blocks a passage-mode question is shown: the fields' retrieval attributes
    (their keywords), the profile's condition words and unit pattern for the inventory question, how its own
    units are found in running text, and the code that applies them (``fields.py`` for which kinds need a digit)."""
    material: dict[str, object] = {
        # In question order, not by name: a field's name reaches the prompts, which the extraction schema covers.
        "fields": [_field_material(spec, FieldRole.RETRIEVAL) for spec in profile.fields],
        "retrieval": dataclasses.asdict(profile.retrieval),
        "code": source_fingerprint("passages.py", "continuation.py", "units.py", "text.py", "fields.py"),
    }
    # A unit's excluded spellings change what its built-in pattern finds; listed only when there are some, so a
    # unit that excludes nothing keeps the material it had before exclusions existed.
    unit_patterns = [
        [unit.canonical, unit.retrieval, *([list(unit.exclude)] if unit.exclude else [])]
        for unit in profile.units.declared
        if unit.retrieval or unit.exclude
    ]
    if unit_patterns:
        material["units"] = unit_patterns
    if profile.declared_entities:
        # Each entity's inventory is shown the blocks its own condition words and unit pattern find.
        material["entities"] = {
            entity.name: dataclasses.asdict(entity.retrieval) for entity in profile.declared_entities
        }
    return content_fingerprint(_dumps(material))


@cache
def extraction_code_fingerprint() -> str:
    """Every module that decides what the model is asked and which of its claims are stored.

    ``adapters.py`` renders the bytes the model reads; ``prompts.py`` wraps them (only the system prompts are
    hashed by value, so the user half would otherwise be invisible); ``fields.py`` writes the field lines
    of every question (the plausible-range sentence) and decides which fields are sample-level, which
    gates the questions asked; ``normalize.py`` and ``grounding.py`` fold the text that decides which values
    are duplicates of each other and which sample a value lands on, and ``normalize.py`` converts the value
    ``drop_implausible`` judges (``continuation.py`` decides which blocks grounding joins across a page
    break); ``voting.py`` decides which of the model's repeated claims survive the majority vote.
    ``text.py`` and ``units.py`` hold the folding and the unit tables ``normalize.py`` applies, and
    ``profile.py`` the prompt-slot defaults a profile falls back on. ``kinds.py`` reads a numeric value (which
    ``drop_implausible`` judges) and adds its kind's note to a field line. ``profile_loader.py`` is left out: what it
    reads from a file reaches this key as values (the schema fingerprint and the rendered prompts), and a
    default it leaves in place is declared in ``fields.py``, ``profile.py`` or ``units.py``, all hashed.
    Over-invalidation is cheap here: an unchanged request replays from the LLM cache, so re-deriving the
    records costs nothing but a second of CPU.

    ``llm.py`` is left out on purpose: it transports a request and validates the reply's shape, and the
    repair question it sends on a malformed reply is written in ``prompts.py``, which is hashed.
    """
    return source_fingerprint(
        "extract.py",
        "fields.py",
        "profile.py",
        "units.py",
        "text.py",
        "voting.py",
        "records.py",
        "adapters.py",
        "prompts.py",
        "normalize.py",
        "grounding.py",
        "continuation.py",
        "kinds.py",
    )


@cache
def normalization_fingerprint() -> str:
    # kinds.py: normalize_field reads a value as its kind's row says.
    return source_fingerprint("normalize.py", "units.py", "text.py", "kinds.py")


@cache
def comparison_code_fingerprint() -> str:
    """The rules that pair samples and values and decide a verdict. Recomputing a comparison is free -- it
    re-reads two stored extractions -- so a change here must never be served from a file written by the old
    rules. ``matching.py`` is in here because which samples were paired decides every verdict below them, and
    ``dataset.py`` and ``decide.py`` because the consolidated table they write is stored under this key and is
    itself a set of verdicts (which cells are committed, which are refused). ``fields.py`` declares the
    attribute defaults the schema material leaves out, and ``profile.py`` the defaults the matching prompt's
    slots fall back on. ``kinds.py`` holds the per-kind verdicts: when two values agree, what a cell holds."""
    return source_fingerprint(
        "compare.py", "matching.py", "dataset.py", "decide.py", "kinds.py", "fields.py", "profile.py"
    )


@dataclass(frozen=True)
class ExtractionOptions:
    """Every setting that decides what one lane's model is asked, in one value.

    Built once per document from the settings and the profile (:meth:`from_settings`) and handed to both
    lanes' :func:`paperfacts.extract.extract_lane`, which records ``extractor_key(options)`` on the lane; every
    reader computes the same key from the same settings (:func:`extractor_key_for`). Two hand-spelled argument
    lists once disagreed about ``context_tokens``; one object cannot. The defaults are the built-in baselines,
    which the key leaves out.
    """

    # Hashed through its role-derived material and its rendered prompts, never as a whole: display text and
    # the file name must not rename a stored extraction.
    profile: DomainProfile = dataclasses.field(metadata={"by_roles": True})
    model: str
    mode: ExtractionMode
    passes: int = 1
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning_effort: ReasoningEffort | None = DEFAULT_LLM_REASONING_EFFORT
    # Passage mode only: document mode never asks an inventory question, retrieves no blocks and trims
    # nothing to fit (it only refuses a paper too long for the window), so these change none of its requests.
    inventory_reasoning_effort: InventoryReasoningEffort = dataclasses.field(
        default=DEFAULT_LLM_INVENTORY_REASONING_EFFORT, metadata={"passage_only": True}
    )
    candidate_limit: int = dataclasses.field(default=DEFAULT_CANDIDATE_LIMIT, metadata={"passage_only": True})
    context_tokens: int = dataclasses.field(default=DEFAULT_LLM_CONTEXT_TOKENS, metadata={"passage_only": True})

    @classmethod
    def from_settings(cls, settings: Settings, profile: DomainProfile, model: str | None = None) -> ExtractionOptions:
        # Refused with a ConfigError where a profile is loaded to run (workflow.check_mode); only asserted here,
        # where readers pass too.
        assert not (profile.declared_entities and settings.extraction_mode == "document"), (
            f"{profile.name} declares entity types, which document mode cannot ask about"
        )
        return cls(
            profile=profile,
            model=model or settings.llm_model,
            mode=settings.extraction_mode,
            passes=settings.extraction_passes,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            reasoning_effort=settings.llm_reasoning_effort,
            inventory_reasoning_effort=settings.llm_inventory_reasoning_effort,
            candidate_limit=settings.candidate_limit,
            context_tokens=settings.llm_context_tokens,
        )


def _slot_material(profile: DomainProfile, *, matching: bool) -> dict[str, object]:
    """The prompt slots that differ from their :class:`PromptSlots` default: the matching ones or all the others,
    plus, per declared entity, the ones it overrides (under ``"entities"``, only when an entity overrides one).

    Hashed by value because a slot may reach only a user prompt (``implausible_origin`` in a passage-mode field
    line, an entity's ``sample_list_heading``), which no system-prompt hash sees; a slot at its default is left
    out like a field attribute, and a required slot (no default) is always in. An entity's override is hashed
    whatever its value: it is what that entity's requests are rendered from."""

    def wanted(name: str) -> bool:
        return name.startswith(_MATCHING_SLOT_PREFIX) == matching

    material: dict[str, object] = {
        name: value
        for name, value in dataclasses.asdict(profile.prompt).items()
        if wanted(name) and value != _PROMPT_SLOT_DEFAULTS.get(name, dataclasses.MISSING)
    }
    overrides = {
        entity.name: {name: getattr(entity.prompt, name) for name in sorted(entity.overrides) if wanted(name)}
        for entity in profile.declared_entities
    }
    if any(overrides.values()):
        material["entities"] = overrides
    return material


def _per_entity(profile: DomainProfile, render: Callable[..., str]) -> str | dict[str, str]:
    """A system prompt as key material: the one text of a profile with one entity (rendered for the primary one,
    which for a profile without entity types is ``profile.prompt`` itself), or each entity's text by name."""
    if len(profile.entities) == 1:
        return render(profile)
    return {entity.name: render(profile, entity) for entity in profile.entities}


def extractor_key(options: ExtractionOptions) -> str:
    """The one key a stored lane is named by.

    The matching prompt is deliberately absent: it drives sample pairing, which lives in the comparison
    report, so tuning it must not throw away the expensive per-lane extractions. Every option at its
    built-in baseline is left out of the material, so an unedited config.json keeps the filenames it has.
    """
    profile = options.profile
    material: dict[str, object] = {
        "model": options.model,
        "schema": profile_extraction_fingerprint(profile),
        "extraction_system": extraction_system_prompt(profile),
        "code": extraction_code_fingerprint(),
        "slots": _slot_material(profile, matching=False),
    }
    # Pinned to "document" rather than to the configured default: whole-document mode sends exactly the
    # request it always sent, so its keys must stay as they were, and changing the default in config.json
    # must never rename anybody's stored facts.
    passage = options.mode != "document"
    if passage:
        material["mode"] = options.mode
        material["inventory_system"] = _per_entity(profile, inventory_system_prompt)
        material["field_system"] = _per_entity(profile, field_system_prompt)
        material["retrieval"] = retrieval_fingerprint(profile)
    for option in dataclasses.fields(ExtractionOptions):
        if (
            option.name in {"model", "mode"}
            or option.metadata.get("by_roles")
            or (option.metadata.get("passage_only") and not passage)
        ):
            continue
        value = getattr(options, option.name)
        if value != option.default:
            material[option.name] = value
    return content_fingerprint(_dumps(material))


def extractor_key_for(settings: Settings, profile: DomainProfile, model: str | None = None) -> str:
    """The key a run with these settings writes, so the reader and the writer cannot disagree about it."""
    return extractor_key(ExtractionOptions.from_settings(settings, profile, model))


@dataclass(frozen=True)
class ComparisonOptions:
    """Every setting that decides a stored comparison and the table consolidated from it, in one value.

    Built once per document (:meth:`from_settings`) and handed to :func:`paperfacts.compare.compare_lanes` and
    :func:`paperfacts.dataset.consolidate_document`, so the verdicts and the key they are stored under cannot
    come from two different profiles.
    """

    profile: DomainProfile
    ambiguous_match_confidence: float

    @classmethod
    def from_settings(cls, settings: Settings, profile: DomainProfile) -> ComparisonOptions:
        return cls(profile=profile, ambiguous_match_confidence=settings.ambiguous_match_confidence)


def comparison_key(options: ComparisonOptions) -> str:
    profile = options.profile
    material = {
        "schema": profile_comparison_fingerprint(profile),
        "ambiguous_confidence": options.ambiguous_match_confidence,
        "normalization": normalization_fingerprint(),
        "code": comparison_code_fingerprint(),
        "matching_system": _per_entity(profile, matching_system_prompt),
        "matching_slots": _slot_material(profile, matching=True),
    }
    return content_fingerprint(_dumps(material))


def comparison_key_for(settings: Settings, profile: DomainProfile) -> str:
    """The key a comparison with these settings is stored under, so the reader and the writer cannot disagree."""
    return comparison_key(ComparisonOptions.from_settings(settings, profile))


@cache
def figure_profile_fingerprint(profile: DomainProfile) -> str:
    """The part of the profile figure reading uses: every FIGURE attribute of the fields a chart may be read
    for (which fields a caption can name, what the model is told about them, how a reading is converted), the
    chart prompt's slots, and the units the profile declares. A field no chart is read for is left out whole,
    so editing it never renames a stored reading; an attribute or a slot at its dataclass default is left out
    as in the other keys (``fields.py`` and ``profile.py``, which declare the defaults, are hashed with the
    figure code)."""
    slots = None
    if profile.figures is not None:
        slots = {
            name: value
            for name, value in dataclasses.asdict(profile.figures).items()
            if value != _FIGURE_SLOT_DEFAULTS.get(name, _NO_DEFAULT)
        }
    material: dict[str, object] = {
        "fields": [_field_material(spec, FieldRole.FIGURE) for spec in profile.figure_fields],
        "slots": slots,
    }
    if profile.units.material():
        material["units"] = profile.units.material()
    return content_fingerprint(_dumps(material))


def figure_key(
    profile: DomainProfile,
    model: str,
    *,
    temperature: float = figures.TEMPERATURE,
    max_tokens: int = figures.MAX_TOKENS,
    dpi: int,
    max_pixels: int,
    max_per_document: int,
) -> str:
    """Everything the stored figure readings depend on. Every setting is in the material whatever its value;
    only the profile's part omits what is at its default (:func:`figure_profile_fingerprint`).

    ``dpi`` and ``max_pixels`` change the image the model is shown; ``max_per_document`` which charts are
    read. ``figures.py`` holds the prompt template, the selection and the conversion into readings;
    ``normalize.py``, ``units.py`` and ``text.py`` the unit arithmetic; ``passages.py`` the keyword matching
    that selects a chart; ``fields.py`` the attribute defaults the profile material leaves out and
    ``profile.py`` which fields are figure-readable.
    """
    material = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "dpi": dpi,
        "max_pixels": max_pixels,
        "max_per_document": max_per_document,
        "fields": figure_profile_fingerprint(profile),
        "code": source_fingerprint(
            "figures.py", "normalize.py", "passages.py", "units.py", "text.py", "fields.py", "profile.py"
        ),
    }
    return content_fingerprint(_dumps(material))


def figure_key_for(settings: Settings, profile: DomainProfile) -> str:
    """The key a figures run with these settings writes. The stage's client is built from the same settings
    and the module's sampling constants (workflow.build_vision_client), so reader and writer agree by
    construction."""
    return figure_key(
        profile,
        settings.figures_model,
        dpi=settings.figures_dpi,
        max_pixels=settings.figures_max_pixels,
        max_per_document=settings.figures_max_per_document,
    )
