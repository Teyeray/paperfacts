"""Cache keys: content fingerprints of everything a stored result depends on.

Two keys name the files under a document directory:

- :func:`extractor_key` covers the model, the field schema, the extraction prompt, how the document is
  rendered for the model, and the code that decides which of the model's claims survive. Changing any of
  them invalidates the stored extraction. The model's own answer is cached separately by request payload,
  so a code-only change re-derives records for free.
- :func:`comparison_key` covers the tolerances, the normalisation rules and the sample-matching prompt.
  Changing a tolerance recomputes the comparison without paying for extraction again.
- :func:`validation_key` covers the vision model, its prompt, how the cited region is rendered for it, which
  values are selected, and the code that turns its reading into a verdict. It is deliberately not folded
  into the two keys above: a change to the validation prompt must re-ask the VLM, never re-run the far more
  expensive extraction, and never rename a comparison that did not change.

Module sources are hashed instead of versioned by hand, so nobody has to remember to bump a number.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from functools import cache
from pathlib import Path

from paperfacts.config import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_LLM_CONTEXT_TOKENS,
    DEFAULT_LLM_INVENTORY_REASONING_EFFORT,
    DEFAULT_LLM_REASONING_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_VLM_CONTEXT_BLOCKS,
    DEFAULT_VLM_CROP_DPI,
    DEFAULT_VLM_CROP_MAX_PIXELS,
    DEFAULT_VLM_CROP_PADDING,
    DEFAULT_VLM_FILL_BLANKS,
    DEFAULT_VLM_MAX_TOKENS,
    DEFAULT_VLM_POLICY,
    DEFAULT_VLM_TEMPERATURE,
    ExtractionMode,
    InventoryReasoningEffort,
    ReasoningEffort,
    Settings,
    ValidationPolicy,
)
from paperfacts.fields import AMBIGUOUS_MATCH_CONFIDENCE, CONDITION_KEYWORDS, FIELD_SPECS
from paperfacts.prompts import (
    extraction_system_prompt,
    field_system_prompt,
    fill_system_prompt,
    inventory_system_prompt,
    matching_system_prompt,
    validation_system_prompt,
)

# Long enough that a collision is not a practical concern, short enough to read in a filename.
FINGERPRINT_LENGTH = 12
_PACKAGE_DIR = Path(__file__).parent
# Cells that change a verdict or retrieval but never what the model is asked; each has its own fingerprint.
# ``label`` is excluded outright: it is a Chinese column header for the UI, so it changes no prompt and no
# verdict and gets no fingerprint of its own -- renaming a column must never re-extract or re-compare.
_SCHEMA_EXCLUDED = {"keywords", "categories", "label", "description_zh"}


def content_fingerprint(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def source_fingerprint(*module_files: str) -> str:
    """Fingerprint of the given files of this package, in the order given."""
    return content_fingerprint("\n".join((_PACKAGE_DIR / name).read_text(encoding="utf-8") for name in module_files))


@cache
def schema_fingerprint() -> str:
    """The field table as the model and the comparison see it: descriptions, units, tolerances, policies.

    ``keywords`` is deliberately left out. It steers passage-mode retrieval and nothing else -- never a
    prompt, never a tolerance -- so folding it in here would invalidate document-mode extractions and every
    stored comparison each time a synonym is added. :func:`retrieval_fingerprint` covers it instead.

    ``label`` is left out because nothing downstream of it is cached: it is only what the web table prints
    above a column.

    ``categories`` is left out for the same reason in the other direction: it renames nothing the model is
    asked and only decides whether two quoted spellings count as the same answer, which is a verdict.
    :func:`category_fingerprint` folds it into ``comparison_key`` alone.
    """
    table = [
        {name: value for name, value in dataclasses.asdict(spec).items() if name not in _SCHEMA_EXCLUDED}
        for spec in FIELD_SPECS
    ]
    return content_fingerprint(json.dumps(table, ensure_ascii=False, sort_keys=True))


@cache
def category_fingerprint() -> str:
    """The closed answer sets of text fields, empty for a table that declares none -- so a checkout without
    any keeps the comparison keys it already has."""
    table = {spec.name: list(spec.categories) for spec in FIELD_SPECS if spec.categories}
    return content_fingerprint(json.dumps(table, ensure_ascii=False, sort_keys=True))


@cache
def retrieval_fingerprint() -> str:
    """Everything that decides which blocks a passage-mode question is shown."""
    material = {
        "keywords": {spec.name: list(spec.keywords) for spec in FIELD_SPECS},
        "condition_keywords": list(CONDITION_KEYWORDS),
        "code": source_fingerprint("passages.py"),
    }
    return content_fingerprint(json.dumps(material, ensure_ascii=False, sort_keys=True))


@cache
def extraction_code_fingerprint() -> str:
    """Every module that decides what the model is asked and which of its claims are stored.

    ``adapters.py`` renders the bytes the model reads; ``prompts.py`` wraps them (only the system prompts are
    hashed by value, so the user half would otherwise be invisible); ``normalize.py`` and ``grounding.py``
    fold the text that decides which values are duplicates of each other and which sample a value lands on;
    ``voting.py`` decides which of the model's repeated claims survive the majority vote.
    Over-invalidation is cheap here: an unchanged request replays from the LLM cache, so re-deriving the
    records costs nothing but a second of CPU.
    """
    return source_fingerprint(
        "extract.py", "voting.py", "records.py", "adapters.py", "prompts.py", "normalize.py", "grounding.py"
    )


@cache
def normalization_fingerprint() -> str:
    return source_fingerprint("normalize.py")


@cache
def comparison_code_fingerprint() -> str:
    """The rules that pair samples and values and decide a verdict. Recomputing a comparison is free -- it
    re-reads two stored extractions -- so a change here must never be served from a file written by the old
    rules. ``matching.py`` is in here because which samples were paired decides every verdict below them, and
    ``dataset.py`` because the consolidated table it writes is stored under this key and is itself a set of
    verdicts (which cells are committed, which are refused)."""
    return source_fingerprint("compare.py", "matching.py", "dataset.py")


def extractor_key(
    model: str,
    *,
    passes: int = 1,
    mode: ExtractionMode = "document",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: ReasoningEffort | None = DEFAULT_LLM_REASONING_EFFORT,
    inventory_reasoning_effort: InventoryReasoningEffort = DEFAULT_LLM_INVENTORY_REASONING_EFFORT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    context_tokens: int = DEFAULT_LLM_CONTEXT_TOKENS,
) -> str:
    """The matching prompt is deliberately absent: it drives sample pairing, which lives in the comparison
    report, so tuning it must not throw away the expensive per-lane extractions. ``passes`` is only mixed in
    when it is not the default, so single-pass keys stay stable.
    """
    material: dict[str, object] = {
        "model": model,
        "schema": schema_fingerprint(),
        "extraction_system": extraction_system_prompt(),
        "code": extraction_code_fingerprint(),
    }
    if passes != 1:
        material["passes"] = passes
    # Sampling settings reach the model, so a stored extraction made with different ones is a different
    # answer to a different question. They are mixed in only when moved, so an unedited config.json keeps
    # the filenames it already has.
    if temperature != DEFAULT_TEMPERATURE:
        material["temperature"] = temperature
    if max_tokens != DEFAULT_MAX_TOKENS:
        material["max_tokens"] = max_tokens
    if reasoning_effort != DEFAULT_LLM_REASONING_EFFORT:
        material["reasoning_effort"] = reasoning_effort
    # Pinned to "document" rather than to the configured default: whole-document mode sends exactly the
    # request it always sent, so its keys must stay as they were, and changing the default in config.json
    # must never rename anybody's stored facts.
    if mode != "document":
        material["mode"] = mode
        material["inventory_system"] = inventory_system_prompt()
        material["field_system"] = field_system_prompt()
        material["retrieval"] = retrieval_fingerprint()
        # Passage mode only: it overrides the effort of the inventory question, which document mode never
        # asks. Out of the material at its baseline, like every other unedited setting.
        if inventory_reasoning_effort != DEFAULT_LLM_INVENTORY_REASONING_EFFORT:
            material["inventory_reasoning_effort"] = inventory_reasoning_effort
        if candidate_limit != DEFAULT_CANDIDATE_LIMIT:
            material["candidate_limit"] = candidate_limit  # how many blocks each question saw
        if context_tokens != DEFAULT_LLM_CONTEXT_TOKENS:
            # Passage mode trims each question to fit this; a smaller window is a different question. In
            # document mode it only decides whether to refuse, so it changes no answer and stays out.
            material["context_tokens"] = context_tokens
    return content_fingerprint(json.dumps(material, ensure_ascii=False, sort_keys=True))


def extractor_key_for(settings: Settings, model: str | None = None) -> str:
    """The key a run with these settings writes, so the reader and the writer cannot disagree about it."""
    return extractor_key(
        model or settings.llm_model,
        passes=settings.extraction_passes,
        mode=settings.extraction_mode,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_effort=settings.llm_reasoning_effort,
        inventory_reasoning_effort=settings.llm_inventory_reasoning_effort,
        candidate_limit=settings.candidate_limit,
        context_tokens=settings.llm_context_tokens,
    )


def comparison_key() -> str:
    material = {
        "schema": schema_fingerprint(),
        "ambiguous_confidence": AMBIGUOUS_MATCH_CONFIDENCE,
        "normalization": normalization_fingerprint(),
        "code": comparison_code_fingerprint(),
        "matching_system": matching_system_prompt(),
    }
    # Only present when a field declares categories: a table without any keeps the filenames it had before
    # the concept existed, the same way every other baseline stays out of the material.
    if any(spec.categories for spec in FIELD_SPECS):
        material["categories"] = category_fingerprint()
    return content_fingerprint(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


@cache
def validation_code_fingerprint() -> str:
    """The code between the model's reading and the verdict.

    ``validate.py`` selects the values, cuts the region and adjudicates; ``grounding.py`` and ``normalize.py``
    are the matcher it adjudicates with, so a more lenient fold there is a different verdict; ``prompts.py``
    carries the user half of the question (only the system half is hashed by value below). ``pdf.py`` is in
    because how the region is rendered -- padding of a thin box, the downscale -- is part of what the model
    saw.
    """
    return source_fingerprint("validate.py", "grounding.py", "normalize.py", "prompts.py", "pdf.py")


def validation_key(
    model: str,
    *,
    temperature: float = DEFAULT_VLM_TEMPERATURE,
    max_tokens: int = DEFAULT_VLM_MAX_TOKENS,
    crop_dpi: int = DEFAULT_VLM_CROP_DPI,
    crop_padding: float = DEFAULT_VLM_CROP_PADDING,
    crop_max_pixels: int = DEFAULT_VLM_CROP_MAX_PIXELS,
    policy: ValidationPolicy = DEFAULT_VLM_POLICY,
    context_blocks: int = DEFAULT_VLM_CONTEXT_BLOCKS,
    fill_blanks: bool = DEFAULT_VLM_FILL_BLANKS,
) -> str:
    """What a stored validation depends on. Same baseline rule as the other two keys: a setting at its
    built-in value stays out of the material, so tightening one knob renames only the files it affects.

    The fill prompt is hashed by value like the validation prompt: it decides what the extractor is asked
    over a transcription, and a filled value is stored in this report and nowhere else.
    """
    material: dict[str, object] = {
        "model": model,
        "validation_system": validation_system_prompt(),
        "fill_system": fill_system_prompt(),
        "code": validation_code_fingerprint(),
    }
    if temperature != DEFAULT_VLM_TEMPERATURE:
        material["temperature"] = temperature
    if max_tokens != DEFAULT_VLM_MAX_TOKENS:
        material["max_tokens"] = max_tokens
    if crop_dpi != DEFAULT_VLM_CROP_DPI:
        material["crop_dpi"] = crop_dpi
    if crop_padding != DEFAULT_VLM_CROP_PADDING:
        material["crop_padding"] = crop_padding
    if crop_max_pixels != DEFAULT_VLM_CROP_MAX_PIXELS:
        material["crop_max_pixels"] = crop_max_pixels
    if policy != DEFAULT_VLM_POLICY:
        material["policy"] = policy  # which values were checked is part of what the report says
    if context_blocks != DEFAULT_VLM_CONTEXT_BLOCKS:
        material["context_blocks"] = context_blocks  # a wider window is a different picture
    if fill_blanks != DEFAULT_VLM_FILL_BLANKS:
        material["fill_blanks"] = fill_blanks  # a report with fills says more than one without
    return content_fingerprint(json.dumps(material, ensure_ascii=False, sort_keys=True))


def validation_key_for(settings: Settings) -> str:
    return validation_key(
        settings.vlm_model,
        temperature=settings.vlm_temperature,
        max_tokens=settings.vlm_max_tokens,
        crop_dpi=settings.vlm_crop_dpi,
        crop_padding=settings.vlm_crop_padding,
        crop_max_pixels=settings.vlm_crop_max_pixels,
        policy=settings.vlm_policy,
        context_blocks=settings.vlm_context_blocks,
        fill_blanks=settings.vlm_fill_blanks,
    )
