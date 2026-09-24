"""Paragraphs a page or column break cut in two: which blocks continue which.

Parsers cut a block wherever the page (or column) ends, so the sentence naming a sample can sit on one page
and the value it has on the next. The halves are linked, never merged: each keeps its own source_id and its
single-page bbox, and retrieval brings one along whenever it picks the other.

Retrieval (:mod:`paperfacts.passages`) and grounding (:mod:`paperfacts.grounding`) both read these links, so
they live in a module of their own that both cache fingerprints hash.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from paperfacts.models import SourceBlock

# Body text interrupted by something that is not part of it: footnotes and sidebars are "text" to the
# adapters, so only the parser's own label tells them apart.
_NOT_BODY_LABELS: frozenset[str] = frozenset({"page_footnote", "footnote", "aside_text"})
# A block that ends one of these finished its sentence.
_SENTENCE_END = re.compile(r"[.!?。](?:[\s\"'”’)\]]|\$)*$")
# How a continuation may start: lower case ("... the films | were annealed"), a number or formula
# ("... composed of | 95% SnO2"), or a parenthesis ("... sputtered | (Ar 20 sccm)").
_CONTINUATION_START = re.compile(r"^[a-z0-9$(]")
# ...but not a sub-figure label "(a)" or a numbered heading "3 Results", which start something new.
_NEW_START = re.compile(r"^(?:\([a-z]\)|\d+(?:\.\d+)*\.?\s+[A-Z])")


def _is_body(block: SourceBlock) -> bool:
    return block.type == "text" and block.raw_label not in _NOT_BODY_LABELS


def continuation_pairs(blocks: Sequence[SourceBlock]) -> list[tuple[int, int]]:
    """``(i, j)`` index pairs where body block ``j`` continues the sentence body block ``i`` left unfinished.

    ``j`` is the next body block after ``i`` in reading order; figures, captions, tables, formulas and
    footnotes between them are skipped, a title is not (a heading starts a new section). Conservative on
    purpose: 49 of 50 sampled pairs found this way were real continuations.
    """
    pairs: list[tuple[int, int]] = []
    previous: int | None = None
    for index, block in enumerate(blocks):
        if block.type == "title":
            previous = None
            continue
        if not _is_body(block):
            continue
        if previous is not None and _continues(blocks[previous].content, block.content):
            pairs.append((previous, index))
        previous = index
    return pairs


def _continues(before: str, after: str) -> bool:
    head, tail = after.lstrip(), before.rstrip()
    if not head or not tail or _SENTENCE_END.search(tail):
        return False
    return bool(_CONTINUATION_START.match(head)) and not _NEW_START.match(head)


def continuation_partners(chosen: set[int], blocks: Sequence[SourceBlock]) -> set[int]:
    """The other half of every chosen block that was cut in two. One step only: a partner's own partner
    stays out, so a long run of continuations never drags in a whole section."""
    extra: set[int] = set()
    for first, second in continuation_pairs(blocks):
        if first in chosen and second not in chosen:
            extra.add(second)
        elif second in chosen and first not in chosen:
            extra.add(first)
    return extra
