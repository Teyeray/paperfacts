"""Cache keys: content fingerprints of everything a stored result depends on.

Two keys name the files under a document directory:

- :func:`extractor_key` covers the model, the field schema, the extraction prompt, how the document is
  rendered for the model, and the code that decides which of the model's claims survive. Changing any of
  them invalidates the stored extraction. The model's own answer is cached separately by request payload,
  so a code-only change re-derives records for free.
- :func:`comparison_key` covers the tolerances, the normalisation rules and the sample-matching prompt.
  Changing a tolerance recomputes the comparison without paying for extraction again.

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
    DEFAULT_LLM_REASONING_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    ExtractionMode,
    Settings,
)
from paperfacts.fields import AMBIGUOUS_MATCH_CONFIDENCE, CONDITION_KEYWORDS, FIELD_SPECS
from paperfacts.prompts import (
    extraction_system_prompt,
    field_system_prompt,
    inventory_system_prompt,
    matching_system_prompt,
)

# Long enough that a collision is not a practical concern, short enough to read in a filename.
FINGERPRINT_LENGTH = 12
_PACKAGE_DIR = Path(__file__).parent


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
    """
    table = [
        {name: value for name, value in dataclasses.asdict(spec).items() if name != "keywords"} for spec in FIELD_SPECS
    ]
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
    fold the text that decides which values are duplicates of each other and which sample a value lands on.
    Over-invalidation is cheap here: an unchanged request replays from the LLM cache, so re-deriving the
    records costs nothing but a second of CPU.
    """
    return source_fingerprint("extract.py", "records.py", "adapters.py", "prompts.py", "normalize.py", "grounding.py")


@cache
def normalization_fingerprint() -> str:
    return source_fingerprint("normalize.py")


@cache
def comparison_code_fingerprint() -> str:
    """The rules that pair samples and values and decide a verdict. Recomputing a comparison is free -- it
    re-reads two stored extractions -- so a change here must never be served from a file written by the old
    rules. ``matching.py`` is in here because which samples were paired decides every verdict below them."""
    return source_fingerprint("compare.py", "matching.py")


def extractor_key(
    model: str,
    *,
    passes: int = 1,
    mode: ExtractionMode = "document",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str | None = DEFAULT_LLM_REASONING_EFFORT,
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
        candidate_limit=settings.candidate_limit,
        context_tokens=settings.llm_context_tokens,
    )


def comparison_key() -> str:
    return content_fingerprint(
        json.dumps(
            {
                "schema": schema_fingerprint(),
                "ambiguous_confidence": AMBIGUOUS_MATCH_CONFIDENCE,
                "normalization": normalization_fingerprint(),
                "code": comparison_code_fingerprint(),
                "matching_system": matching_system_prompt(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
