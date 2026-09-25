"""Which blocks the model needs to see for each question — deterministic retrieval, no model involved.

Passage-mode extraction asks the model one small question at a time, so something has to decide what goes
into each prompt. That decision is made here, by ordinary code, for one reason above all: **both lanes must
be given the same treatment**. A model-driven selector would add its own noise to what is supposed to be a
measurement of parser disagreement.

Three selectors, one per question the extractor asks:

- :func:`inventory_blocks` — "which samples does this paper report?" Needs breadth: section titles, every
  table and caption, and the prose that carries deposition conditions.
- :func:`candidate_blocks` — "where might this one field's value be?" Needs precision: every block naming
  the field by a keyword, plus the blocks carrying only its unit, ranked and capped.
- :func:`fit_budget` — keep a prompt inside the context window without ever silently dropping a table.

Matching rules that were tuned against real papers, and why:

- **Word boundaries.** ``Rs`` as a plain substring matches "years", "layers" and "parameters"; the sheet
  resistance of a paper would then be looked for in half the document.
- **A numeric field needs a digit.** A paragraph saying resistance "decreased sharply" cannot contain the
  number, so it is not a candidate however often it says the word.
- **A unit is a weak signal, not a filter.** A block that names the field is always shown; a block that
  only carries the unit competes for the places left under the limit. Filtering on units instead would either drown
  the prompt (every paper is full of percentages and wavelengths) or lose the paper that writes "films of
  2108 nm" without the word "thickness".

The limit's baseline was measured when it capped every block (``.omc/research/extraction-modes.md``); it
now caps only what the named blocks leave room for.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Literal

from paperfacts.config import DEFAULT_CANDIDATE_LIMIT
from paperfacts.continuation import continuation_partners
from paperfacts.fields import CONDITION_KEYWORDS, FieldSpec
from paperfacts.models import SourceBlock
from paperfacts.normalize import delatex, normalize_text

logger = logging.getLogger(__name__)

# Blocks that are dense in values and cheap to include: a table holds the numbers, a caption holds the
# condition they were measured under, and the two are useless apart.
DENSE_TYPES: frozenset[str] = frozenset({"table", "caption"})
# extraction.candidate_limit in config.json caps the unit-only blocks, together with the named ones: named
# blocks are never cut, and the unit-only ones fill whatever places they left.


# How each canonical unit is recognised **inside running text**. :mod:`paperfacts.normalize` has unit
# patterns too, but those are anchored: they answer "is this whole string the unit?" for a value the model
# already quoted. Searching prose for a unit is a different question and needs looser expressions.
#
# A unit match ranks below a name match rather than being filtered out, because the two failure modes are
# not symmetric. "%" and "nm" appear in every paper, so treating them as proof would drown the prompt;
# refusing them outright loses the paper that writes "films of 2108 nm" without the word "thickness". As a
# weaker class they only fill places no named block wanted.
# Lowercase omega, not the ohm sign: _searchable() lowercases, and "Ω".lower() is "ω". normalize_key has to
# undo the same fold for the same reason. Spelling it uppercase here would silently match only "ohm".
_OHM = r"(?:ohms?|ω)"
# Every pattern runs on _searchable() text, which is lower case: an upper-case letter in one never matches.
# "W" was written that way once, and sputtering_power went unasked in 22 of 54 lanes that said "60 W".
UNIT_PATTERNS: dict[str, re.Pattern[str]] = {
    "Ω/sq": re.compile(rf"{_OHM}\s*(?:/|per)?\s*(?:sq|square|□)"),
    "Ω·cm": re.compile(rf"{_OHM}\s*[.x*·-]?\s*cm"),
    # "4 in." and "2 inch" are target sizes; a bare "in" is the English word, so a digit must precede it.
    "inch": re.compile(r"\d\s*(?:inch|inches|in\.|\")"),
    "nm": re.compile(r"\d\s*(?:nm|µm|μm|um)\b"),
    "min": re.compile(r"\d\s*(?:min|mins|minutes?|h|hr|hrs|hours?|s|sec|secs|seconds?)\b"),
    "%": re.compile(r"\d\s*%"),
    # K is admitted as a retrieval signal even though the converter refuses it: a block saying "annealed
    # at 573 K" belongs in the prompt, and the honest ambiguous verdict is the comparison's job, not
    # retrieval's.
    "℃": re.compile(r"\d\s*(?:°\s*[ck]\b|℃|c\b|k\b)"),
    "cm": re.compile(r"\d\s*(?:cm|mm|m|µm|μm|um)\b"),
    "W": re.compile(r"\d\s*[km]?w\b"),
    "sccm": re.compile(r"\d\s*(?:sccm|slm)\b"),
    "rpm": re.compile(r"\d\s*(?:rpm|r/min)\b"),
    "Pa": re.compile(r"\d\s*(?:[mkh]?pa|m?torr|m?bar)\b"),
}

# A deposition condition stated as a number with its unit. This is what distinguishes one sample from
# another ("100 sccm", "150 W", "300 °C"), so a block carrying one belongs in the inventory question even
# when it uses none of the condition words.
# A bare "%" is deliberately absent: it appears in every results paragraph (transmittance, ratios),
# so it would flood the inventory selection with prose. Only explicit composition ratios count here.
# "\d\s*s\b" does not match "2 samples": \b requires a non-word character after the "s", and the "a" of
# "amples" is a word character, so the boundary fails and the block stays out of the inventory.
CONDITION_UNIT = re.compile(
    r"\d\s*(?:sccm|W\b|°C|℃|K\b|Pa\b|mtorr|torr|mbar|kv\b|ma\b|rpm|min\b|h\b|s\b|(?:vol|at)\.?\s*%)",
    re.IGNORECASE,
)

# Compiled on first use and kept: the keyword tables are small and fixed at import time.
_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


# MinerU drops one letter of a doubled pair on some papers ("transmitance" for "transmittance"): 410 words
# across 15 papers, none in the other lane. Keywords are matched with every run of a repeated letter squeezed
# to one on both sides, so the two spellings meet.
_DOUBLED_LETTER = re.compile(r"([a-z])\1+")
# A LaTeX command left over after delatex (\text, \mathrm) splits a unit: "\Omega\cdot\text{cm}".
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+")


def _pattern(keyword: str) -> re.Pattern[str]:
    """Match a keyword as a whole token. A keyword ending in punctuation (``d =``, ``%T``) keeps that edge
    open, since ``\\b`` would demand a word character that is not there."""
    cached = _PATTERN_CACHE.get(keyword)
    if cached is None:
        folded = _DOUBLED_LETTER.sub(r"\1", normalize_text(keyword).lower())
        prefix = r"\b" if folded[:1].isalnum() else ""
        suffix = r"\b" if folded[-1:].isalnum() else ""
        cached = _PATTERN_CACHE[keyword] = re.compile(prefix + re.escape(folded) + suffix)
    return cached


def _searchable(block: SourceBlock) -> str:
    """The block as units are searched for: folded, LaTeX undone (``delatex`` restores "°" and "%"), lower
    case."""
    return _LATEX_COMMAND.sub(" ", delatex(normalize_text(block.content))).lower()


def _names(keywords: Sequence[str], text: str) -> int:
    """How many of ``keywords`` occur in ``text`` (a :func:`_searchable` string) as whole tokens."""
    squeezed = _DOUBLED_LETTER.sub(r"\1", text)
    return sum(1 for keyword in keywords if _pattern(keyword).search(squeezed))


def inventory_blocks(blocks: Sequence[SourceBlock]) -> list[SourceBlock]:
    """The blocks that could name a sample or the conditions that distinguish one.

    Section titles come along because they are nearly free and tell the model which part of the paper it is
    reading; tables and captions because samples are usually enumerated there; prose only when it mentions a
    deposition condition. Document order is preserved, so the model sees the paper's own narrative.
    """
    chosen = {index for index, block in enumerate(blocks) if _is_inventory_block(block)}
    chosen |= continuation_partners(chosen, blocks)
    logger.debug("inventory blocks %d/%d", len(chosen), len(blocks))
    return [blocks[index] for index in sorted(chosen)]


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
    return _names(CONDITION_KEYWORDS, text) > 0


def candidate_blocks(
    spec: FieldSpec,
    blocks: Sequence[SourceBlock],
    *,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
    sample_blocks: frozenset[str] = frozenset(),
) -> list[SourceBlock]:
    """The blocks that could hold a value of ``spec``, returned in document order.

    Every block naming the field comes along; the blocks that only carry its unit are capped by ``limit``,
    which is what keeps a paper full of percentages from putting every caption into every question.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    unit = UNIT_PATTERNS.get(spec.canonical_unit or "")
    named: set[int] = set()
    unit_only: list[tuple[bool, int]] = []
    for index, block in enumerate(blocks):
        match = _classify(spec, block, unit)
        if match == "named":
            named.add(index)
        elif match == "unit":
            unit_only.append((block.type in DENSE_TYPES, index))

    # Every block that names the field is shown: ranking named blocks against each other cut the later
    # pages of a paper (ties break on document order), which is where the results section is. "Named"
    # means a keyword matched, nothing else: a table or caption carrying only the unit ("XRD at 20%
    # power") is unit-only however dense it is, or every caption with a "%" would ride along uncapped.
    # The unit-only blocks compete for the places the named ones left, tables and captions first, then
    # document order, so the selection is reproducible run to run.
    spare = max(limit - len(named), 0)
    chosen = named | {index for _, index in sorted(unit_only, key=lambda item: (not item[0], item[1]))[:spare]}
    if spec.is_sample_level:
        # The blocks the inventory said describe the samples. A methods paragraph stating "240 nm thick" or
        # "the substrate to target distance is 69 mm" names the recipe in words no keyword list foresees,
        # and then only its unit qualifies it -- where it loses its place to earlier blocks. It still has to
        # carry a number to be of use.
        chosen |= {
            index
            for index, block in enumerate(blocks)
            if block.source_id in sample_blocks and (spec.kind != "numeric" or _has_digit(block))
        }
    chosen |= _dense_neighbours(chosen, blocks)
    chosen |= continuation_partners(chosen, blocks)
    selected = [blocks[index] for index in sorted(chosen)]
    logger.debug(
        "candidates field=%s matched=%d selected=%d chars=%d",
        spec.name,
        len(named) + len(unit_only),
        len(selected),
        sum(len(block.content) for block in selected),
    )
    return selected


def _has_digit(block: SourceBlock) -> bool:
    return any(character.isdigit() for character in block.content)


def _classify(spec: FieldSpec, block: SourceBlock, unit: re.Pattern[str] | None) -> Literal["named", "unit"] | None:
    """Whether ``block`` names ``spec`` by a keyword, only carries its unit, or does not qualify at all."""
    text = _searchable(block)
    if spec.kind == "numeric" and not any(character.isdigit() for character in text):
        return None  # a number cannot be quoted from a block that has none
    if _names(spec.keywords, text):
        return "named"
    if unit is not None and unit.search(text):
        return "unit"
    return None


def _dense_neighbours(chosen: set[int], blocks: Sequence[SourceBlock]) -> set[int]:
    """Pull in a selected block's adjacent table or caption.

    One fact is routinely split across two blocks: the number is in the table and the wavelength it was
    measured at is in the caption beside it. Whichever half matched, the other half has to come too, and it
    arrives outside the ranking so it never costs another block its place.

    A page break does not break that pair: a table at the foot of one page and its caption at the head of
    the next are adjacent in reading order, so a neighbour one page away still qualifies — but only for a
    table/caption pair, never two tables or a figure, which across a break are merely consecutive.
    """
    extra: set[int] = set()
    for index in chosen:
        for neighbour in (index - 1, index + 1):
            if not 0 <= neighbour < len(blocks) or neighbour in chosen:
                continue
            if blocks[neighbour].type not in DENSE_TYPES:
                continue
            page_gap = abs(blocks[neighbour].page - blocks[index].page)
            if page_gap == 0:
                extra.add(neighbour)
            elif page_gap == 1 and {blocks[index].type, blocks[neighbour].type} == DENSE_TYPES:
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
