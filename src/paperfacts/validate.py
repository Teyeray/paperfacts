"""Visual validation: a vision model transcribes the region a value was cited from, and the code decides.

The model **transcribes; this module adjudicates**. It is never told the value under check and never asked
"is this right?": the verdict is :func:`paperfacts.grounding.is_grounded` run against its transcription, so
the matcher that judges a parser's text judges the model's text -- same leniency, same strictness. Adding a
yes/no question to the prompt, or a second matcher here, would break that.

This module holds the parts of the stage that are decided by code alone, and nothing that talks to a model:

- :class:`Reading` and :func:`parse_reading` -- the model's reply, read leniently.
- :func:`adjudicate` -- the verdict, from grounding's own matcher.
- :func:`region_for` and :func:`region_of_blocks` -- which pixels the model is shown, in the page's reading
  order, never crossing a page.
- the stored shapes a report is written from.

The crops themselves come from :mod:`paperfacts.crops`, which renders a box once and keeps it; where a crop
lives is not what a model was asked, so nothing about storage belongs in ``validation_key``.

Verdicts are kept apart -- ``confirmed``, ``contradicted``, ``illegible``, ``not_checked``, ``error`` -- so a
value nobody could check never looks like one that was checked and passed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperfacts.config import DEFAULT_VLM_CONTEXT_BLOCKS, DEFAULT_VLM_CROP_PADDING
from paperfacts.crops import Region, RegionCrop
from paperfacts.fields import FieldSpec
from paperfacts.grounding import is_grounded
from paperfacts.models import ParsedArtifact, SourceBlock
from paperfacts.profile import DomainProfile, EntitySpec

# ``_values`` is reached for deliberately: adding a public alias to ``prompts.py`` would change that module's
# bytes, and ``extraction_code_fingerprint`` hashes its source -- every stored extraction would be renamed by a
# change that only concerns the vision stage. The validation prompts live here for the same reason.
from paperfacts.prompts import _values as slot_values
from paperfacts.prompts import render, render_field_table
from paperfacts.records import FieldValue, KindContext

# What the reading said about a value. ``not_checked`` is a value the stage could not put in front of the
# model at all (no citation, no PDF); ``error`` is a request that failed after its retries. Both are kept in
# the report so "not validated" and "validated and wrong" never look alike.
Verdict = Literal["confirmed", "contradicted", "illegible", "not_checked", "error"]
VERDICTS: tuple[Verdict, ...] = ("confirmed", "contradicted", "illegible", "not_checked", "error")
# Why a value was selected. The comparison status that put it on the list, ``ungrounded`` for a value
# grounding flagged in either lane, ``table`` for a value cited from a table under the tables policy, ``all``
# under the exhaustive policy.
Reason = Literal["conflict", "ambiguous", "missing", "ungrounded", "table", "all"]
# Block types worth including as sliding-window context. A figure's block is a picture (its content is a file
# path) and page furniture is noise; both would only make the crop larger.
CONTEXT_TYPES: frozenset[str] = frozenset({"text", "title", "table", "caption", "formula"})
# The source id a fill cites: the transcription is one block with this id, so the extractor can cite nothing
# else and citation validation keeps working unchanged.
FILL_SOURCE_PREFIX = "vlm:"
# Where in its lane a value lives. Together with the backend this is what makes a value's key unique, and it
# is spelled the way the comparison report spells scopes, so the web UI can rebuild the key from a row.
OWNER_TARGET = "target"
OWNER_UNATTRIBUTED = "unattributed"
# The lenient parse of the model's reply: the first JSON object in the text, code fences or not.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
# A multiplication sign between two digits, with or without spaces around it. The extractor quotes the
# parser's spacing ("1.2 × 10^-4") and the model writes its own ("1.2×10⁻⁴"); grounding is deliberately
# strict about spaces inside numbers, so this one fold is applied to *both* sides before it judges them.
_DIGIT_TIMES_DIGIT = re.compile(r"(?<=\d)\s*[×✕✖x]\s*(?=\d)")


# ---- Stored shapes -----------------------------------------------------------------------------------------


class ValueValidation(BaseModel):
    """One value, one region, one reading, one verdict."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="value_key: the lane, the owner and the value's identity")
    backend: str
    owner: str = Field(description=f"{OWNER_TARGET}:<sample_id>, or {OWNER_UNATTRIBUTED}")
    field: str
    value_raw: str
    reason: Reason = Field(description="why this value was selected for a check")
    verdict: Verdict
    detail: str = Field(default="", description="why the verdict is what it is, in words")
    transcription: str = Field(default="", description="what the model read, verbatim")
    crop: RegionCrop | None = Field(default=None, description="the image shown; None when none could be made")


class FilledValue(BaseModel):
    """A value no lane held, quoted by the extraction model out of a transcription and grounded in it."""

    model_config = ConfigDict(frozen=True)

    backend: str
    sample_id: str
    field: str
    value_raw: str
    unit_raw: str | None = None
    condition: str | None = None
    source_id: str = Field(description=f"the {FILL_SOURCE_PREFIX}<crop> block the quote was taken from")
    transcription: str = ""
    crop: RegionCrop | None = None


class ValidationCounts(BaseModel):
    """One line of arithmetic over the verdicts, so a report need not be re-counted to be described."""

    model_config = ConfigDict(frozen=True)

    checked: int = 0
    confirmed: int = 0
    contradicted: int = 0
    illegible: int = 0
    not_checked: int = 0
    error: int = 0
    filled: int = 0


# ---- The region a value was cited from ---------------------------------------------------------------------


def region_for(
    value: FieldValue,
    artifact: ParsedArtifact,
    *,
    padding: float = DEFAULT_VLM_CROP_PADDING,
    context: int = DEFAULT_VLM_CONTEXT_BLOCKS,
) -> Region | None:
    """The union of the cited blocks on the first cited page and their sliding-window neighbours, padded;
    None when nothing was cited.

    Citations on other pages are dropped rather than joined: a box spanning two pages is not a region of any
    page. The first cited block decides which page, because the extractor is asked to cite the most specific
    block first.
    """
    blocks = []
    for source_id in value.source_ids:
        try:
            blocks.append(artifact.block(source_id))
        except KeyError:
            continue  # an invalid id was already audited by records.py; nothing to draw for it
    if not blocks:
        return None
    page = blocks[0].page
    same_page = [block for block in blocks if block.page == page]
    return region_of_blocks(same_page, artifact, padding=padding, context=context)


def region_of_blocks(
    cited: Sequence[SourceBlock],
    artifact: ParsedArtifact,
    *,
    padding: float = DEFAULT_VLM_CROP_PADDING,
    context: int = DEFAULT_VLM_CONTEXT_BLOCKS,
) -> Region:
    """The crop for some blocks on one page: their union, widened by ``context`` neighbours on each side.

    Neighbours are the adjacent blocks in the page's reading order, the same order grounding's
    ``block_adjacency`` walks, so a quote grounding accepts across a boundary is inside the crop. Figures and
    page furniture are skipped over (not counted) because they carry no text worth reading, and a neighbour is
    never taken from another page.
    """
    page = cited[0].page
    on_page = list(artifact.blocks_on_page(page))
    index_of = {block.source_id: i for i, block in enumerate(on_page)}
    context_ids: list[str] = []
    cited_ids = {block.source_id for block in cited}
    for block in cited:
        position = index_of.get(block.source_id)
        if position is None:
            continue
        for step in (-1, 1):
            taken, i = 0, position + step
            while taken < context and 0 <= i < len(on_page):
                neighbour = on_page[i]
                if neighbour.type in CONTEXT_TYPES:
                    if neighbour.source_id not in cited_ids and neighbour.source_id not in context_ids:
                        context_ids.append(neighbour.source_id)
                    taken += 1
                i += step
    bbox = cited[0].bbox
    for block in list(cited[1:]) + [artifact.block(i) for i in context_ids]:
        bbox = bbox.union(block.bbox)
    return Region(
        page=page,
        bbox=bbox.padded(padding),
        source_ids=tuple(block.source_id for block in cited),
        context_ids=tuple(sorted(context_ids, key=lambda i: index_of[i])),
    )


# ---- The reading and the verdict ---------------------------------------------------------------------------


class Reading(BaseModel):
    """What the model is asked to return. ``extra="ignore"`` keeps it tolerant of a chatty model."""

    model_config = ConfigDict(extra="ignore")

    transcription: str = ""
    legible: bool = True


def parse_reading(text: str) -> tuple[Reading, str]:
    """The model's reply as a :class:`Reading`, plus a note when it had to be coaxed.

    A vision endpoint without JSON mode may wrap the object in a code fence or a sentence; the first
    ``{...}`` in the text is taken. A reply with no JSON at all is treated as the transcription itself: a
    model that ignored the format and simply transcribed has still done the useful part of the job, and the
    note records that the structure was missing.
    """
    stripped = _FENCE.sub("", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return Reading.model_validate_json(stripped[start : end + 1]), ""
        except ValidationError:
            pass
    note = "reply was not the JSON object asked for; taken as the transcription"
    return Reading(transcription=stripped, legible=bool(stripped.strip())), note


def transcription_fold(text: str) -> str:
    """The one spelling difference between a parser and a model that grounding's matcher does not absorb.

    Both parsers space a product the way the PDF's typesetting did and the extractor copies it verbatim; a
    vision model writes ``1.2×10⁻⁴`` however it likes. ``grounding_key`` folds ``×`` to ``x`` but keeps the
    spaces, because for a *number* a space is a boundary it must not invent or drop. Closing the gap only
    between two digits, and on both sides alike, keeps that guarantee: "5 nm" still cannot be found in
    "235 nm", and "40 × 10" now equals "40×10".
    """
    return _DIGIT_TIMES_DIGIT.sub("x", text)


def adjudicate(value: FieldValue, reading: Reading) -> tuple[Verdict, str]:
    """Does the quoted value occur in what the model read? Decided by grounding's own matcher.

    The transcription is stood in as a block the value cites, so the exact leniency grounding extends to a
    parser's text -- LaTeX, ``×`` versus ``x``, superscripts, case, decoration -- is extended to the model's,
    and the exact strictness too: a number never matches inside a longer number. :func:`transcription_fold`
    is applied to both sides first.
    """
    if not reading.legible or not reading.transcription.strip():
        return "illegible", "the model could not read the region"
    probe = value.model_copy(update={"source_ids": ("vlm",), "value_raw": transcription_fold(value.value_raw)})
    if is_grounded(probe, {"vlm": transcription_fold(reading.transcription)}):
        return "confirmed", "value_raw occurs in the model's transcription of the cited region"
    return "contradicted", "value_raw does not occur in the model's transcription of the cited region"


# ---- A value's identity ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationTarget:
    """One value selected for a check, with the lane and owner that make its key unique."""

    backend: str
    owner: str
    value: FieldValue
    field: str
    reason: Reason


def value_key(backend: str, owner: str, value: FieldValue) -> str:
    """A value's identity across a stored report and the web UI.

    ``facts.js`` rebuilds this from a comparison row, so the field order here is a contract between the two
    files: changing it means changing both.
    """
    condition = value.condition or ""
    return "|".join((backend, owner, value.field, value.value_raw, condition))


# ---- The prompts ------------------------------------------------------------------------------------------
# Here rather than in ``prompts.py`` because ``extraction_code_fingerprint`` hashes that module's source: a
# word changed in a validation prompt must re-ask the vision model and nothing else, so it may not touch the
# bytes ``extractor_key`` reads. ``validation_code_fingerprint`` hashes this module instead.

_VALIDATION_SYSTEM = """You are a transcription instrument. You are shown a cropped region of one page of a scientific paper (a paragraph, a table, a caption or a formula) as an image.

Transcribe EXACTLY the text visible in the image. Do not summarise, interpret, correct, convert or complete anything. Copy every number, sign, unit and symbol as printed:
- keep superscripts and subscripts readable, e.g. write 10^-4 (or 10⁻⁴), SnO2 (or SnO₂), cm^-3;
- keep ±, ≈, ~, <, > and ranges exactly as printed;
- transcribe a table cell by cell, row by row, one row per line, cells separated by " | ";
- transcribe formulas as they read.
If a part of the image is unreadable, write [illegible] in its place rather than guessing. If the whole image contains no text, return an empty transcription and set legible to false.

Output ONLY a JSON object with this exact shape and nothing else (no prose, no code fences):
{"transcription": "<the verbatim text>", "legible": <true|false>}"""


def validation_system_prompt() -> str:
    """Identical for both lanes and every field: what varies is the picture, never the instructions. Carries no
    domain words at all -- a transcription instrument needs none, so this one text serves every profile."""
    return _VALIDATION_SYSTEM


def validation_user_prompt(spec: FieldSpec) -> str:
    """The user half names the quantity the region was cited for -- so the model knows which small print to take
    care over -- but never the value or the unit the extractor quoted: those are what is being checked, and
    stating them would invite the model to see them. The wording is identical for both lanes."""
    return (
        f"This region was cited as the source of a value of: {spec.name} ({spec.description}).\n"
        "Transcribe the whole region exactly as printed. Return the JSON object only."
    )


def table_transcription_user_prompt() -> str:
    """The question for a whole table region read for the fill step: no field is named, because the
    transcription serves every field the table might hold, and the caption and footnote that the sliding window
    brought along are asked for explicitly -- they carry the sample names and the units."""
    return (
        "This region contains a table from the paper, possibly with its caption above and notes below.\n"
        "Transcribe everything exactly as printed: the caption, every header cell, every row (one row per "
        'line, cells separated by " | "), and any footnote. Return the JSON object only.'
    )


# The fill question. The transcription is one block with one source id, so the model can cite nothing but it;
# the sample ids are given so a value lands on a sample the lane already knows; the fields are limited to the
# ones that sample lacks, so a value the lane already holds is never second-guessed here.
_FILL_SYSTEM = """You are given the transcription of ONE table from a scientific paper about {domain_subject}, preceded by a provenance marker `<!-- source: <id> -->`. The table was read from the page image by a vision model; treat the transcription as the paper's own text.

A list of {sample_plural} the paper reports is given, each with a stable `sample_id`. Your only job is to read, for those {sample_plural}, the fields listed below that are still missing, and to quote them exactly as they appear in the transcription.

Output ONLY a JSON object with this exact shape (no prose):

{
  "samples": [
    {"sample_id": "<one of the given sample ids, exactly>", "fields": [FIELD, ...]},
    ...
  ]
}

FIELD = {"field": "<field name from the table below>", "value_raw": "<exactly as written in the transcription>",
         "unit_raw": "<unit exactly as written, or null>", "condition": "<measurement condition, or null>",
         "source_ids": ["<the marker id>"], "note": "<optional remark or null>"}

Rules:
1. `value_raw` must be copied verbatim from the transcription. Never convert units or round numbers.
2. Use only the given sample ids. If a row of the table cannot be matched to one of them with confidence, leave it out.
3. Report a field only when the transcription states it for that {sample_singular}. Never guess, never fill defaults, never reuse a value from another row.
4. Only the listed fields. A field that is not listed for a {sample_singular} is not wanted for that {sample_singular}.
5. Units: put the unit in `unit_raw` exactly as written in the header or the cell. If a column header carries the unit, that is the unit of every value in the column.
6. `source_ids` must be the marker id shown; there is no other.

Fields that may be requested:
{fields}

Return the JSON object only."""


def fill_system_prompt(profile: DomainProfile, entity: EntitySpec | None = None) -> str:
    """The extraction question asked over a transcription.

    It is the *extraction* model's prompt, not the vision model's: the VLM transcribes, and quoting the missing
    fields out of that transcription is an extraction like any other, so it is rendered from the same profile
    slots and the same field table. Letting the VLM answer this question itself would put a value in the mouth
    of the instrument that is supposed to be checking values.
    """
    ctx = KindContext(entities={e.name: e for e in profile.entities})
    fields = render_field_table(profile.fields, profile.prompt.implausible_origin, ctx)
    return render(_FILL_SYSTEM, {**slot_values(profile, entity), "fields": fields})


def fill_user_prompt(sample_list: str, missing: str, markdown: str) -> str:
    """``sample_list`` is the lane's own samples, ``missing`` names which fields each still lacks, and
    ``markdown`` is the transcription behind its one source marker."""
    return (
        f"Samples (use these ids exactly):\n{sample_list}\n\n"
        f"Fields still missing, per sample:\n{missing}\n\n"
        f"Table transcription:\n{markdown}\n\nReturn the JSON object now."
    )
