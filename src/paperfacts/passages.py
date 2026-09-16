"""Which blocks the model needs to see for each question — deterministic retrieval, no model involved.

Passage-mode extraction asks the model one small question at a time, so something has to decide what goes
into each prompt. That decision is made here, by ordinary code, for one reason above all: **both lanes must
be given the same treatment**. A model-driven selector would add its own noise to what is supposed to be a
measurement of parser disagreement.

Three selectors, one per question the extractor asks:

- :func:`inventory_blocks` — "which samples does this paper report?" Needs breadth: section titles, every
  table and caption, and the prose that carries deposition conditions.
- :func:`candidate_blocks` — "where might this one field's value be?" Needs precision: keyword and unit
  matching, ranked, capped.
- :func:`fit_budget` — keep a prompt inside the context window without ever silently dropping a table.

Matching rules that were tuned against real papers, and why:

- **Word boundaries.** ``Rs`` as a plain substring matches "years", "layers" and "parameters"; the sheet
  resistance of a paper would then be looked for in half the document.
- **A numeric field needs a digit.** A paragraph saying resistance "decreased sharply" cannot contain the
  number, so it is not a candidate however often it says the word.
- **A unit is a weak signal, not a filter.** Selection ranks and then cuts, so a unit match can qualify a
  block while scoring below any block that matched by name. Filtering on units instead would either drown
  the prompt (every paper is full of percentages and wavelengths) or lose the paper that writes "films of
  2108 nm" without the word "thickness".

Measured against the blocks earlier whole-document runs cited on the papers in ``data/docs``: the default
limit recalls 97% of them. See ``.omc/research/extraction-modes.md``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from paperfacts.config import DEFAULT_CANDIDATE_LIMIT
from paperfacts.fields import CONDITION_KEYWORDS, FieldSpec
from paperfacts.models import SourceBlock
from paperfacts.normalize import normalize_text

logger = logging.getLogger(__name__)

# Blocks that are dense in values and cheap to include: a table holds the numbers, a caption holds the
# condition they were measured under, and the two are useless apart.
DENSE_TYPES: frozenset[str] = frozenset({"table", "caption"})
# How many blocks one question carries is extraction.candidate_limit in config.json. Its built-in baseline
# was measured against the blocks earlier whole-document runs cited: 4 recalls 77% of them, 6 recalls 95%,
# 8 recalls 97%, and 12 adds nothing but prose.
# Score weights. A name match is worth more than a unit match, and a table or caption more than prose.
KEYWORD_SCORE = 2
UNIT_SCORE = 1
DENSE_SCORE = 1


# How each canonical unit is recognised **inside running text**. :mod:`paperfacts.normalize` has unit
# patterns too, but those are anchored: they answer "is this whole string the unit?" for a value the model
# already quoted. Searching prose for a unit is a different question and needs looser expressions.
#
# A unit match is worth less than a name match rather than being filtered out, because the two failure modes
# are not symmetric. "%" and "nm" appear in every paper, so treating them as proof would drown the prompt;
# refusing them outright loses the paper that writes "films of 2108 nm" without the word "thickness". As a
# low score they only fill places no named block wanted.
# Lowercase omega, not the ohm sign: _searchable() lowercases, and "Ω".lower() is "ω". normalize_key has to
# undo the same fold for the same reason. Spelling it uppercase here would silently match only "ohm".
_OHM = r"(?:ohms?|ω)"
UNIT_PATTERNS: dict[str, re.Pattern[str]] = {
    "Ω/sq": re.compile(rf"{_OHM}\s*(?:/|per)?\s*(?:sq|square|□)"),
    "Ω·cm": re.compile(rf"{_OHM}\s*[.x*·-]?\s*cm"),
    # "4 in." and "2 inch" are target sizes; a bare "in" is the English word, so a digit must precede it.
    "inch": re.compile(r"\d\s*(?:inch|inches|in\.|\")"),
    "nm": re.compile(r"\d\s*(?:nm|µm|μm|um)\b"),
    "min": re.compile(r"\d\s*(?:min|mins|minutes?|h|hr|hrs|hours?)\b"),
    "%": re.compile(r"\d\s*%"),
}

# A deposition condition stated as a number with its unit. This is what distinguishes one sample from
# another ("100 sccm", "150 W", "300 °C"), so a block carrying one belongs in the inventory question even
# when it uses none of the condition words.
CONDITION_UNIT = re.compile(r"\d\s*(?:sccm|W\b|°C|℃|K\b|Pa\b|mtorr|torr|mbar|kv\b|ma\b|rpm|min\b|h\b)", re.IGNORECASE)

# Compiled on first use and kept: the keyword tables are small and fixed at import time.
_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _pattern(keyword: str) -> re.Pattern[str]:
    """Match a keyword as a whole token. A keyword ending in punctuation (``d =``, ``%T``) keeps that edge
    open, since ``\\b`` would demand a word character that is not there."""
    cached = _PATTERN_CACHE.get(keyword)
    if cached is None:
        folded = normalize_text(keyword).lower()
        prefix = r"\b" if folded[:1].isalnum() else ""
        suffix = r"\b" if folded[-1:].isalnum() else ""
        cached = _PATTERN_CACHE[keyword] = re.compile(prefix + re.escape(folded) + suffix)
    return cached


def _searchable(block: SourceBlock) -> str:
    return normalize_text(block.content).lower()


def inventory_blocks(blocks: Sequence[SourceBlock]) -> list[SourceBlock]:
    """The blocks that could name a sample or the conditions that distinguish one.

    Section titles come along because they are nearly free and tell the model which part of the paper it is
    reading; tables and captions because samples are usually enumerated there; prose only when it mentions a
    deposition condition. Document order is preserved, so the model sees the paper's own narrative.
    """
    kept = [block for block in blocks if _is_inventory_block(block)]
    logger.debug("inventory blocks %d/%d", len(kept), len(blocks))
    return kept


def _is_inventory_block(block: SourceBlock) -> bool:
    if block.type in DENSE_TYPES or block.type == "title":
        return True
    text = _searchable(block)
    if CONDITION_UNIT.search(text):
        return True
    # A condition word alone is not enough: "the deposition process" appears in every discussion paragraph.
    # Paired with a number it is almost always the sentence that states how a sample was made.
    if not any(character.isdigit() for character in text):
        return False
    return any(_pattern(keyword).search(text) for keyword in CONDITION_KEYWORDS)


def candidate_blocks(
    spec: FieldSpec, blocks: Sequence[SourceBlock], *, limit: int = DEFAULT_CANDIDATE_LIMIT
) -> list[SourceBlock]:
    """The blocks that could hold a value of ``spec``, ranked by score and returned in document order.

    Ranking rather than plain filtering is what keeps the prompt small on a paper that says "sheet
    resistance" thirty times: the blocks that say it *and* carry numbers *and* are a table win the places.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    unit = UNIT_PATTERNS.get(spec.canonical_unit or "")
    scored: list[tuple[int, int]] = []
    for index, block in enumerate(blocks):
        score = _score(spec, block, unit)
        if score is not None:
            scored.append((score, index))

    # Ties break on document order, so the selection is reproducible run to run.
    chosen = {index for _, index in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]}
    chosen |= _dense_neighbours(chosen, blocks)
    selected = [blocks[index] for index in sorted(chosen)]
    logger.debug(
        "candidates field=%s matched=%d selected=%d chars=%d",
        spec.name,
        len(scored),
        len(selected),
        sum(len(block.content) for block in selected),
    )
    return selected


def _score(spec: FieldSpec, block: SourceBlock, unit: re.Pattern[str] | None) -> int | None:
    """This block's score for ``spec``, or None when it does not qualify at all."""
    text = _searchable(block)
    if spec.kind == "numeric" and not any(character.isdigit() for character in text):
        return None  # a number cannot be quoted from a block that has none
    names = sum(1 for keyword in spec.keywords if _pattern(keyword).search(text))
    has_unit = unit is not None and bool(unit.search(text))
    if not names and not has_unit:
        return None
    return KEYWORD_SCORE * names + (UNIT_SCORE if has_unit else 0) + (DENSE_SCORE if block.type in DENSE_TYPES else 0)


def _dense_neighbours(chosen: set[int], blocks: Sequence[SourceBlock]) -> set[int]:
    """Pull in a selected block's adjacent table or caption.

    One fact is routinely split across two blocks: the number is in the table and the wavelength it was
    measured at is in the caption beside it. Whichever half matched, the other half has to come too, and it
    arrives outside the ranking so it never costs another block its place.
    """
    extra: set[int] = set()
    for index in chosen:
        for neighbour in (index - 1, index + 1):
            if not 0 <= neighbour < len(blocks) or neighbour in chosen:
                continue
            if blocks[neighbour].type in DENSE_TYPES and blocks[neighbour].page == blocks[index].page:
                extra.add(neighbour)
    return extra


def fit_budget(blocks: Sequence[SourceBlock], *, budget_chars: int) -> list[SourceBlock]:
    """Trim a selection to ``budget_chars``, dropping prose from the end and never a table.

    Tables are the densest evidence in a paper, so cutting one to fit a budget would throw away the very
    thing the question is about. Whatever is dropped is logged, never silent.
    """
    total = sum(len(block.content) for block in blocks)
    if total <= budget_chars:
        return list(blocks)

    dropped: set[int] = set()
    for index in range(len(blocks) - 1, -1, -1):
        if total <= budget_chars:
            break
        if blocks[index].type == "table":
            continue
        total -= len(blocks[index].content)
        dropped.add(index)
    kept = [block for index, block in enumerate(blocks) if index not in dropped]
    if dropped:
        logger.warning(
            "context budget: dropped %d of %d blocks (%s)",
            len(dropped),
            len(blocks),
            ", ".join(blocks[index].source_id for index in sorted(dropped)),
        )
    if total > budget_chars:
        logger.warning("selection still exceeds the budget after dropping prose: %d > %d chars", total, budget_chars)
    return kept
