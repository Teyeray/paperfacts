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
from collections.abc import Mapping

from paperfacts.normalize import KEY_CHARACTERS, delatex, normalize_text
from paperfacts.records import FieldValue, LaneExtraction

# LaTeX expands to " x ", Unicode papers use "×"; fold both so the two spellings compare equal.
_MULTIPLICATION = re.compile(r"[×✕✖⋅·]")
# Formatting commands that survive delatex and would otherwise split a chemical formula: a table cell
# holding "$\mathrm { S n O } _ { 2 } : \mathrm { S b } _ { 2 }$" has to match a quoted "SnO2:Sb2O3".
_LATEX_WRAPPERS = re.compile(r"\\(?:mathrm|mathbf|mathit|mathsf|mathcal|text|rm|it|bf|left|right|operatorname)\b")


# Decoration collapses to a single space rather than vanishing. `normalize_key` deletes it, which is right
# for asking "are these two values equal" but wrong here: with separators gone, "76.7, 71.3, 68.4" becomes
# one 12-digit run in which none of the three numbers has a boundary any more.
_DECORATION = re.compile(f"[^{KEY_CHARACTERS}]+")


def grounding_key(text: str) -> str:
    """Reduce text to the form used for the containment test: no LaTeX, no case, no decoration."""
    folded = _LATEX_WRAPPERS.sub(" ", delatex(normalize_text(text)))
    folded = _MULTIPLICATION.sub("x", folded).lower().replace("ω", "Ω")
    return _DECORATION.sub(" ", folded).strip()


_MIN_SQUEEZED_LENGTH = 4


def is_grounded(value: FieldValue, blocks: Mapping[str, str]) -> bool:
    """True when ``value.value_raw`` appears in at least one of the blocks it cites."""
    needle = grounding_key(value.value_raw)
    if not needle or not value.source_ids:
        return False
    cited = [grounding_key(blocks.get(source_id, "")) for source_id in value.source_ids]
    if any(_contains(haystack, needle) for haystack in cited):
        return True
    squeezed = _squeeze(needle)
    if not _may_be_squeezed(squeezed):
        return False
    return any(squeezed in _squeeze(haystack) for haystack in cited)


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
        before = haystack[match.start() - 1] if match.start() else ""
        after = haystack[match.end()] if match.end() < len(haystack) else ""
        if not (before.isdigit() or after.isdigit()):
            return True
    return False


def ground_values(values: tuple[FieldValue, ...], blocks: Mapping[str, str]) -> tuple[FieldValue, ...]:
    """Return ``values`` with :attr:`FieldValue.grounded` filled in."""
    return tuple(value.model_copy(update={"grounded": is_grounded(value, blocks)}) for value in values)


def ground_lane(lane: LaneExtraction, blocks: Mapping[str, str]) -> LaneExtraction:
    """Re-check every value in ``lane`` against ``blocks``.

    Grounding needs no model, so it is redone whenever a lane is read rather than trusted from the stored
    file. Improving the matcher therefore costs nothing and never leaves a stale verdict behind -- the same
    bargain normalisation makes.
    """
    target = lane.target
    if target is not None:
        target = target.model_copy(update={"fields": ground_values(target.fields, blocks)})
    return lane.model_copy(
        update={
            "target": target,
            "samples": tuple(
                sample.model_copy(update={"fields": ground_values(sample.fields, blocks)}) for sample in lane.samples
            ),
            "unattributed": ground_values(lane.unattributed, blocks),
        }
    )
