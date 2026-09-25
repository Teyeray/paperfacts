"""Cache keys: content fingerprints of everything a stored result depends on.

Two keys name the files under a document directory:

- :func:`extractor_key` covers the model, the part of the field table extraction reads (everything but
  the tolerances, categories, preferences and display text), the extraction prompts, how the document is
  rendered for the model, and the code that decides which of the model's claims survive. Changing any of
  them invalidates the stored extraction. The model's own answer is cached separately by request payload,
  so a code-only change re-derives records for free.
- :func:`comparison_key` covers the whole field table including the tolerances, the normalisation rules
  and the sample-matching prompt. Changing a tolerance recomputes the comparison and leaves every stored
  extraction where it is.

A third, :func:`figure_key`, names the figure readings, which belong to neither lane.

Module sources are hashed instead of versioned by hand, so nobody has to remember to bump a number.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
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
from paperfacts.fields import AMBIGUOUS_MATCH_CONFIDENCE, CONDITION_KEYWORDS, FIELD_SPECS
from paperfacts.profile import default_profile
from paperfacts.prompts import (
    extraction_system_prompt,
    field_system_prompt,
    inventory_system_prompt,
    matching_system_prompt,
)

# Long enough that a collision is not a practical concern, short enough to read in a filename.
FINGERPRINT_LENGTH = 12
_PACKAGE_DIR = Path(__file__).parent
# Cells that change a verdict or retrieval but never what the model is asked; each has its own fingerprint.
# ``label`` is excluded outright: it is a Chinese column header for the UI, so it changes no prompt and no
# verdict and gets no fingerprint of its own -- renaming a column must never re-extract or re-compare.
_SCHEMA_EXCLUDED = {"keywords", "categories", "label", "description_zh", "condition_preference"} | {
    # No stage reads these yet, so no stored result depends on them: ``level`` restates the group, which is
    # hashed, and the rest keep today's behaviour at their defaults until the code that reads them lands.
    "level",
    "condition_rule",
    "missing_condition_note_zh",
    "figure_readable",
    "display_format",
    "range_policy",
}
# Cells only a verdict reads: the numeric tolerance of a comparison. Nothing in extraction -- no prompt, no
# cleaning rule, not drop_implausible -- looks at them, so they stay out of extractor_key.
_VERDICT_ONLY = {"rel_tol", "abs_tol"}
# Cells added after stored results existed: left out of the material while at their default, so a field that
# does not use one keeps the fingerprint it had before the cell was introduced.
_SCHEMA_OMITTED_AT_DEFAULT = {"valid_range": (None, None)}
_NO_DEFAULT = object()


def content_fingerprint(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def source_fingerprint(*module_files: str) -> str:
    """Fingerprint of the given files of this package, in the order given."""
    return content_fingerprint("\n".join((_PACKAGE_DIR / name).read_text(encoding="utf-8") for name in module_files))


def _table_fingerprint(excluded: set[str]) -> str:
    table = [
        {
            name: value
            for name, value in dataclasses.asdict(spec).items()
            if name not in excluded and _SCHEMA_OMITTED_AT_DEFAULT.get(name, _NO_DEFAULT) != value
        }
        for spec in FIELD_SPECS
    ]
    return content_fingerprint(json.dumps(table, ensure_ascii=False, sort_keys=True))


@cache
def extraction_schema_fingerprint() -> str:
    """The field table as extraction reads it: what the model is told (name, group, kind, description,
    canonical unit, condition hint, plausible range) and what the cleaning of its answer reads (the bare
    number policy, through ``drop_implausible``). The tolerances are left out: they only decide verdicts,
    so editing one must never rename a stored extraction."""
    return _table_fingerprint(_SCHEMA_EXCLUDED | _VERDICT_ONLY)


@cache
def schema_fingerprint() -> str:
    """The field table as the comparison sees it: descriptions, units, tolerances, policies.

    ``keywords`` is deliberately left out. It steers passage-mode retrieval and nothing else -- never a
    prompt, never a tolerance -- so folding it in here would invalidate document-mode extractions and every
    stored comparison each time a synonym is added. :func:`retrieval_fingerprint` covers it instead.

    ``label`` is left out because nothing downstream of it is cached: it is only what the web table prints
    above a column.

    ``categories`` is left out for the same reason in the other direction: it renames nothing the model is
    asked and only decides whether two quoted spellings count as the same answer, which is a verdict.
    :func:`category_fingerprint` folds it into ``comparison_key`` alone.
    """
    return _table_fingerprint(_SCHEMA_EXCLUDED)


@cache
def category_fingerprint() -> str:
    """The closed answer sets of text fields, empty for a table that declares none -- so a checkout without
    any keeps the comparison keys it already has."""
    table = {spec.name: list(spec.categories) for spec in FIELD_SPECS if spec.categories}
    return content_fingerprint(json.dumps(table, ensure_ascii=False, sort_keys=True))


@cache
def preference_fingerprint() -> str:
    """The condition preferences that pick a dataset cell among several measurements; a verdict rule."""
    table = {spec.name: list(spec.condition_preference) for spec in FIELD_SPECS if spec.condition_preference}
    return content_fingerprint(json.dumps(table, ensure_ascii=False, sort_keys=True))


@cache
def retrieval_fingerprint() -> str:
    """Everything that decides which blocks a passage-mode question is shown."""
    material = {
        "keywords": {spec.name: list(spec.keywords) for spec in FIELD_SPECS},
        "condition_keywords": list(CONDITION_KEYWORDS),
        "code": source_fingerprint("passages.py", "continuation.py", "units.py", "text.py"),
    }
    return content_fingerprint(json.dumps(material, ensure_ascii=False, sort_keys=True))


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
    ``profile.py`` the prompt-slot defaults a profile falls back on.
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
    )


@cache
def normalization_fingerprint() -> str:
    return source_fingerprint("normalize.py", "units.py", "text.py")


@cache
def comparison_code_fingerprint() -> str:
    """The rules that pair samples and values and decide a verdict. Recomputing a comparison is free -- it
    re-reads two stored extractions -- so a change here must never be served from a file written by the old
    rules. ``matching.py`` is in here because which samples were paired decides every verdict below them, and
    ``dataset.py`` and ``decide.py`` because the consolidated table they write is stored under this key and is
    itself a set of verdicts (which cells are committed, which are refused). ``profile.py`` holds the defaults
    the matching prompt's slots fall back on."""
    return source_fingerprint("compare.py", "matching.py", "dataset.py", "decide.py", "profile.py")


@dataclass(frozen=True)
class ExtractionOptions:
    """Every setting that decides what one lane's model is asked, in one value.

    Built once from the settings (:meth:`from_settings`) and handed to
    :func:`paperfacts.extract.extract_lane`, which records ``extractor_key(options)`` on the lane; every reader
    computes the same key from the same settings (:func:`extractor_key_for`). Two hand-spelled argument
    lists once disagreed about ``context_tokens``; one object cannot. The defaults are the built-in baselines,
    which the key leaves out.
    """

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
    def from_settings(cls, settings: Settings, model: str | None = None) -> ExtractionOptions:
        return cls(
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


def extractor_key(options: ExtractionOptions) -> str:
    """The one key a stored lane is named by.

    The matching prompt is deliberately absent: it drives sample pairing, which lives in the comparison
    report, so tuning it must not throw away the expensive per-lane extractions. Every option at its
    built-in baseline is left out of the material, so an unedited config.json keeps the filenames it has.
    """
    material: dict[str, object] = {
        "model": options.model,
        "schema": extraction_schema_fingerprint(),
        "extraction_system": extraction_system_prompt(default_profile()),
        "code": extraction_code_fingerprint(),
    }
    # Pinned to "document" rather than to the configured default: whole-document mode sends exactly the
    # request it always sent, so its keys must stay as they were, and changing the default in config.json
    # must never rename anybody's stored facts.
    passage = options.mode != "document"
    if passage:
        material["mode"] = options.mode
        material["inventory_system"] = inventory_system_prompt(default_profile())
        material["field_system"] = field_system_prompt(default_profile())
        material["retrieval"] = retrieval_fingerprint()
    for option in dataclasses.fields(ExtractionOptions):
        if option.name in {"model", "mode"} or (option.metadata.get("passage_only") and not passage):
            continue
        value = getattr(options, option.name)
        if value != option.default:
            material[option.name] = value
    return content_fingerprint(json.dumps(material, ensure_ascii=False, sort_keys=True))


def extractor_key_for(settings: Settings, model: str | None = None) -> str:
    """The key a run with these settings writes, so the reader and the writer cannot disagree about it."""
    return extractor_key(ExtractionOptions.from_settings(settings, model))


def comparison_key() -> str:
    material = {
        "schema": schema_fingerprint(),
        "ambiguous_confidence": AMBIGUOUS_MATCH_CONFIDENCE,
        "normalization": normalization_fingerprint(),
        "code": comparison_code_fingerprint(),
        "matching_system": matching_system_prompt(default_profile()),
    }
    # Only present when a field declares categories: a table without any keeps the filenames it had before
    # the concept existed, the same way every other baseline stays out of the material.
    if any(spec.categories for spec in FIELD_SPECS):
        material["categories"] = category_fingerprint()
    if any(spec.condition_preference for spec in FIELD_SPECS):
        material["condition_preference"] = preference_fingerprint()
    return content_fingerprint(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


@cache
def figure_field_fingerprint() -> str:
    """The part of the field table figure reading uses: which fields a caption can name (the keywords),
    what the model is told about them, and how a reading is converted."""
    table = [
        {
            "name": spec.name,
            "description": spec.description,
            "keywords": list(spec.keywords),
            "canonical_unit": spec.canonical_unit,
            "bare_number": spec.bare_number,
        }
        for spec in figures.figure_fields()
    ]
    return content_fingerprint(json.dumps(table, ensure_ascii=False, sort_keys=True))


def figure_key(
    model: str,
    *,
    temperature: float = figures.TEMPERATURE,
    max_tokens: int = figures.MAX_TOKENS,
    dpi: int,
    max_pixels: int,
    max_per_document: int,
) -> str:
    """Everything the stored figure readings depend on. A new key with no stored files behind it yet, so
    nothing here is omitted at a baseline: every input is in the material from the start.

    ``dpi`` and ``max_pixels`` change the image the model is shown; ``max_per_document`` which charts are
    read. ``figures.py`` holds the prompt, the selection and the conversion into readings; ``normalize.py``,
    ``units.py`` and ``text.py`` the unit arithmetic; ``passages.py`` the keyword matching that selects a chart.
    """
    material = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "dpi": dpi,
        "max_pixels": max_pixels,
        "max_per_document": max_per_document,
        "fields": figure_field_fingerprint(),
        "code": source_fingerprint("figures.py", "normalize.py", "passages.py", "units.py", "text.py"),
    }
    return content_fingerprint(json.dumps(material, ensure_ascii=False, sort_keys=True))


def figure_key_for(settings: Settings) -> str:
    """The key a figures run with these settings writes. The stage's client is built from the same settings
    and the module's sampling constants (workflow.build_vision_client), so reader and writer agree by
    construction."""
    return figure_key(
        settings.figures_model,
        dpi=settings.figures_dpi,
        max_pixels=settings.figures_max_pixels,
        max_per_document=settings.figures_max_per_document,
    )
