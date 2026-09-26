"""Check that an extracted value really occurs in the block it cites.

Validating that a ``source_id`` exists only proves the model named a real block; it does not prove the
value came from there. A model can quote a plausible number and attach a nearby, real id, and the result
looks perfectly traceable all the way to a page and a bounding box. Grounding closes that gap: a value is
grounded when its text can be found in the text of one of its cited blocks.

Matching is deliberately lenient about formatting, because the two parsers write the same number very
differently -- MinerU emits ``$( 4 0 \\times 1 0 \\mathrm { c m }$`` where PaddleOCR-VL emits ``(40 x 10 cm``
-- while staying strict about digits, which is what actually matters.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from paperfacts.continuation import continuation_pairs
from paperfacts.models import SourceBlock
from paperfacts.profile import DomainProfile
from paperfacts.records import FieldValue, LaneExtraction, resolve_reference
from paperfacts.text import KEY_CHARACTERS, LATEX_WRAPPERS, delatex, normalize_text

# LaTeX expands to " x ", Unicode papers use "×"; fold both so the two spellings compare equal.
_MULTIPLICATION = re.compile(r"[×✕✖⋅·]")


# Decoration collapses to a single space rather than vanishing. `normalize_key` deletes it, which is right
# for asking "are these two values equal" but wrong here: with separators gone, "76.7, 71.3, 68.4" becomes
# one 12-digit run in which none of the three numbers has a boundary any more.
# The caret is kept, glued to its exponent: it is what tells the boundary rule that "10" in "10^-4" is the
# base of a power, not a number of its own.
_DECORATION = re.compile(f"[^^{KEY_CHARACTERS}]+")
_CARET = re.compile(r"\s*\^\s*")
_EXPONENT = re.compile(r"[-+]?\d")


def grounding_key(text: str) -> str:
    """Reduce text to the form used for the containment test: no LaTeX, no case, no decoration."""
    return _fold(text, _DECORATION)


def _fold(text: str, decoration: re.Pattern[str]) -> str:
    folded = LATEX_WRAPPERS.sub(" ", delatex(normalize_text(text)))
    folded = _MULTIPLICATION.sub("x", folded).lower().replace("ω", "Ω")
    return _POWER_OF_TEN.sub(r"x 10^\1", _CARET.sub("^", decoration.sub(" ", folded))).strip()


# A power of ten after a multiplication sign is an exponent whether or not the caret survived: a table writes
# "6.58 x 10<sup>-4</sup>" (a caret once folded) while the model quotes "6.58 x 10-4". Only after "x": a bare
# "10-4" elsewhere may be a range and keeps its hyphen.
_POWER_OF_TEN = re.compile(r"x\s*10\s*\^?\s*(-?\s*\d+)")

_MIN_SQUEEZED_LENGTH = 4


def block_adjacency(blocks: Sequence[SourceBlock]) -> dict[str, tuple[str | None, str | None]]:
    """Map each block's source_id to ``(previous_id, next_id)`` within the given sequence.

    A neighbour is the adjacent entry in reading order *on the same page*, or the other half of a sentence
    :func:`paperfacts.continuation.continuation_pairs` found cut by a page or column break. Any other pair
    of blocks across a page break comes from two different regions of the document, and joining them would
    manufacture text that appears nowhere in the PDF. A linked half's partner replaces its reading-order
    neighbour on that side, so a formula between the two halves is no longer tried as the neighbour.
    """
    adjacency: dict[str, list[str | None]] = {}
    for index, block in enumerate(blocks):
        previous = blocks[index - 1] if index > 0 and blocks[index - 1].page == block.page else None
        nxt = blocks[index + 1] if index + 1 < len(blocks) and blocks[index + 1].page == block.page else None
        adjacency[block.source_id] = [
            previous.source_id if previous is not None else None,
            nxt.source_id if nxt is not None else None,
        ]
    for first, second in continuation_pairs(blocks):
        adjacency[blocks[first].source_id][1] = blocks[second].source_id
        adjacency[blocks[second].source_id][0] = blocks[first].source_id
    return {source_id: (previous, nxt) for source_id, (previous, nxt) in adjacency.items()}


def is_grounded(
    value: FieldValue,
    blocks: Mapping[str, str],
    *,
    adjacency: Mapping[str, tuple[str | None, str | None]] | None = None,
) -> bool:
    """True when ``value.value_raw`` appears in at least one of the blocks it cites."""
    needle = grounding_key(value.value_raw)
    if not needle or not value.source_ids:
        return False
    cited = [grounding_key(blocks.get(source_id, "")) for source_id in value.source_ids]
    if any(_contains(haystack, needle) for haystack in cited):
        return True
    squeezed = _squeeze(needle)
    if not _may_be_squeezed(squeezed):
        return _grounded_across_boundary(needle, squeezed, False, value.source_ids, blocks, adjacency)
    if any(squeezed in _squeeze(haystack) for haystack in cited):
        return True
    return _grounded_across_boundary(needle, squeezed, True, value.source_ids, blocks, adjacency)


def _squeeze(text: str) -> str:
    return text.replace(" ", "")


def _may_be_squeezed(squeezed: str) -> bool:
    r"""Whether a value may be matched with its separators discarded.

    Formulas earn this: MinerU writes ``SnO2:Ta`` as ``$\mathrm { S n O } _ { 2 } : \mathrm { T a }$``, so
    its spacing is an artefact of the typesetting and comparing without spaces is the only way to match it.

    Numbers do not, however they are written. A measurement squeezed against a squeezed block is exactly
    the case that produces silent false confirmations -- "5 nm" found inside "235 nm" -- so anything
    starting with a digit keeps its separators and has to match properly.
    """
    return len(squeezed) >= _MIN_SQUEEZED_LENGTH and not squeezed[0].isdigit()


def _contains(haystack: str, needle: str) -> bool:
    """Substring search that will not match a number inside a longer number.

    Grounding keys have no spaces, so a plain ``in`` test lets a short value match a digit run that has
    nothing to do with it: a thickness of "4" would be "found" in a block reading "deposited for 40
    minutes", and a sputtering time of "10" inside "page 10 of 12". Those are precisely the fields whose
    values are short and round, so the check would pass most often exactly where it is needed most --
    and, unlike a failed match, a wrong success is silent.
    """
    for match in re.finditer(re.escape(needle), haystack):
        if not (_continues_before(haystack, match.start()) or _continues_after(haystack, match.end())):
            return True
    return False


def _continues_before(text: str, start: int) -> bool:
    """Whether the number ending at ``text[start - 1]`` runs on into position ``start``.

    A digit does, and so does a decimal point after a digit ("5" in "0.5") and a caret or an exponent's sign
    ("4" in "10^4", "10^-4"): each makes the match the tail of a longer number, not a number of its own.
    """
    before = text[start - 1] if start else ""
    before2 = text[start - 2] if start > 1 else ""
    return (
        before.isdigit()
        or before == "^"
        or (before == "." and before2.isdigit())
        or (before in "-+" and before2 == "^")
    )


def _continues_after(text: str, end: int) -> bool:
    """Whether a number continues past ``end``: a digit, a decimal point before a digit ("5" in "5.2"), or a
    caret before an exponent ("10" in "10^-4"). A caret before anything else is no exponent -- a raised
    unit, or a degree sign that escaped folding -- and the number before it is a number of its own.
    """
    after = text[end] if end < len(text) else ""
    after2 = text[end + 1] if end + 1 < len(text) else ""
    exponent = after == "^" and _EXPONENT.match(text, end + 1) is not None
    return after.isdigit() or (after == "." and after2.isdigit()) or exponent


def _grounded_across_boundary(
    needle: str,
    squeezed: str,
    squeezable: bool,
    source_ids: tuple[str, ...],
    blocks: Mapping[str, str],
    adjacency: Mapping[str, tuple[str | None, str | None]] | None,
) -> bool:
    """Grounding across a block boundary, tried only after both in-block checks have failed.

    The README documents a real failure this repairs: the model quoted "95% SnO2 and 5% Sb2O3" citing one
    block, but the sentence straddles two adjacent blocks and only the second was cited -- the value is
    real, yet was flagged ungrounded. A quote found in the join of the cited block and a same-page
    neighbour, crossing the junction between them, is therefore accepted.

    A quote lying entirely inside the neighbour is *not* accepted: the model cited a block that carries
    none of the quote, and the neighbour merely happens to contain the words.
    """
    if adjacency is None:
        return False
    for source_id in source_ids:
        cited = grounding_key(blocks.get(source_id, ""))
        for neighbour_id in adjacency.get(source_id, (None, None)):
            # A neighbour absent from the block map (e.g. pruned) has no text to join.
            if neighbour_id is None or neighbour_id not in blocks:
                continue
            neighbour = grounding_key(blocks[neighbour_id])
            # The cited block's side comes second for a previous neighbour, first for a next one, so the
            # junction sits where the cited block's text begins/ends in the join.
            pairs = (
                (neighbour + " " + cited, len(neighbour)),
                (cited + " " + neighbour, len(cited)),
            )
            for joined, junction in pairs:
                if _straddles(joined, needle, junction):
                    return True
                if squeezable and _straddles(_squeeze(joined), squeezed, len(_squeeze(joined[:junction]))):
                    return True
    return False


def _straddles(joined: str, needle: str, junction: int) -> bool:
    """Whether ``needle`` crosses the junction between two joined blocks, obeying the digit rule.

    Strictly one side is not enough: a match must reach from one block's text into the other's
    (``start < junction <= end`` -- a needle ending exactly at the junction already touches both).
    The number-boundary strictness of :func:`_contains` applies in the joined text too, or a thickness
    of "4" would ground via "deposited for 4" + "0 min" reading as "40", and "5" via "0." + "5 nm".
    """
    for match in re.finditer(re.escape(needle), joined):
        if _continues_before(joined, match.start()) or _continues_after(joined, match.end()):
            continue
        if not (match.start() < junction <= match.end()):
            continue
        # The digit rule again, in squeezed coordinates: the space at the junction is an artefact of the
        # join, not spacing the PDF had, so "4" ending at the junction of "…for 4" + "0 min…" is really
        # the "4" of "40" and must not ground. Position mapping is a space count because squeezing only
        # removes spaces; when ``joined`` is already squeezed the mapping is the identity. Only a needle
        # that itself begins/ends with a digit can merge into a longer number, so the check is gated on
        # that -- otherwise "and" following "SnO2" in the squeeze would be rejected as a digit run.
        squeezed = _squeeze(joined)
        start = match.start() - joined[: match.start()].count(" ")
        end = match.end() - joined[: match.end()].count(" ")
        if needle[0].isdigit() and _continues_before(squeezed, start):
            continue
        if needle[-1].isdigit() and _continues_after(squeezed, end):
            continue
        return True
    return False


# ---- A bound outside the quote --------------------------------------------------------------------------
# "above 90 %" quoted whole is a bound; "90" quoted out of the same sentence read as the scalar 90 and filled six
# cells with a number the paper never stated. What stands right before the quote in its block settles which it
# is, so grounding, the one step that locates the quote there, records it; normalize and decide then read the
# value as if the bound had been quoted. "~" is no bound: an approximate value is still the value. Nor is "under",
# which papers write as a condition ("sputtered under 0.5 Pa", "500 °C under N2"), as normalize's condition rule
# reads it.
_BOUND_SIGNS = {"geq": "≥", "ge": "≥", "leq": "≤", "le": "≤", "gt": ">", "lt": "<"}
# LaTeX ("\geq", not "\left") and HTML ("&gt;") spellings of the signs.
_BOUND_SIGN = re.compile(r"\\(geq|ge|leq|le|gt|lt)(?![A-Za-z])|&(gt|lt);")
_BOUND_DECORATION = re.compile(f"[^^<>≥≤{KEY_CHARACTERS}]+")
# The words that make the number after them a one-sided bound, lower then upper. normalize reads a quote's own
# bound with the same words, so a bound found before a quote and one quoted with it read alike.
LOWER_BOUND_WORDS = ("above", "over", "more than", "greater than", "higher than", "exceeding", "at least")
UPPER_BOUND_WORDS = ("below", "less than", "lower than", "up to", "at most")
_BOUND_BEFORE = re.compile(rf"(?:^|[^a-z])({'|'.join((*LOWER_BOUND_WORDS, *UPPER_BOUND_WORDS))}|[<>≥≤])\s*$")


def _bound_key(text: str) -> str:
    """:func:`grounding_key` keeping the comparison signs, which the containment test folds away."""
    text = _BOUND_SIGN.sub(lambda match: f" {_BOUND_SIGNS[match.group(1) or match.group(2)]} ", text)
    return _fold(text, _BOUND_DECORATION)


def quoted_bound(value: FieldValue, blocks: Mapping[str, str]) -> str | None:
    """The bound a cited block writes right before ``value.value_raw``, or None.

    Any in-block occurrence carrying one is enough: the quote does not say which occurrence it copied, and a
    bound read as a scalar is a wrong number where a bound refused is only a blank cell. The number-boundary rule
    of :func:`_contains` applies, so "90" in "above 90.5 %" is no occurrence of it. A quote found only across a
    block boundary is not looked at."""
    needle = _bound_key(value.value_raw)
    if not needle:
        return None
    for source_id in value.source_ids:
        haystack = _bound_key(blocks.get(source_id, ""))
        for match in re.finditer(re.escape(needle), haystack):
            if _continues_before(haystack, match.start()) or _continues_after(haystack, match.end()):
                continue
            bound = _BOUND_BEFORE.search(haystack, 0, match.start())
            if bound is not None:
                return bound.group(1)
    return None


def ground_values(
    values: tuple[FieldValue, ...],
    blocks: Mapping[str, str],
    *,
    adjacency: Mapping[str, tuple[str | None, str | None]] | None = None,
) -> tuple[FieldValue, ...]:
    """Return ``values`` with :attr:`FieldValue.grounded` and :attr:`FieldValue.bound` filled in."""
    return tuple(
        value.model_copy(
            update={
                "grounded": is_grounded(value, blocks, adjacency=adjacency),
                "bound": quoted_bound(value, blocks),
            }
        )
        for value in values
    )


def ground_lane(
    lane: LaneExtraction,
    blocks: Mapping[str, str],
    *,
    adjacency: Mapping[str, tuple[str | None, str | None]] | None = None,
    profile: DomainProfile,
) -> LaneExtraction:
    """Re-check every value in ``lane`` against ``blocks``, a reference field's against the lane's samples.

    Grounding needs no model, so it is redone whenever a lane is read rather than trusted from the stored
    file. Improving the matcher therefore costs nothing and never leaves a stale verdict behind -- the same
    bargain normalisation makes.

    A ``reference`` field quotes the id of a sample of another entity type, which the question showed in a list
    rather than in an excerpt. Its value is grounded when it resolves to one of the lane's samples of that entity
    (:func:`~paperfacts.records.resolve_reference`), and no bound applies. Resolving here rather than once at
    extraction is what keeps it: every read re-grounds, and a text check would refuse every reference again.
    """
    references = {spec.name: spec.references for spec in profile.fields if spec.references is not None}
    listed = lane.listed() if references else {}

    def resolved(value: FieldValue) -> FieldValue:
        found = resolve_reference(listed.get(references[value.field], {}), value.value_raw)
        return value.model_copy(update={"grounded": found is not None, "bound": None})

    def grounded(values: tuple[FieldValue, ...]) -> tuple[FieldValue, ...]:
        return tuple(
            resolved(value) if value.field in references else ground_values((value,), blocks, adjacency=adjacency)[0]
            for value in values
        )

    paper = lane.paper
    if paper is not None:
        paper = paper.model_copy(update={"fields": grounded(paper.fields)})
    return lane.model_copy(
        update={
            "paper": paper,
            "samples": tuple(sample.model_copy(update={"fields": grounded(sample.fields)}) for sample in lane.samples),
            "unattributed": grounded(lane.unattributed),
        }
    )
