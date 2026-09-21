"""Visual validation: a vision-language model reads the cited page region back; the code decides.

Everything upstream of this stage works on *text the parsers produced*. Both lanes are extracted from
parser text, grounding checks a quote against parser text, and the comparison compares two parser texts.
So when both parsers misread the same character -- a lost minus sign in ``10^-2``, a shifted table cell --
the pipeline reports AGREE at full confidence and nothing anywhere has looked at the page. This module is
the one place that does: it renders the region a value was cited from and asks a third, independent model
what is printed there.

Three rules keep it honest, and they mirror the rules extraction lives by:

- **The model transcribes; the code adjudicates.** The VLM is never told which value is being checked and
  is never asked "is this right?". It is asked to transcribe the region, and the verdict is whether the
  quoted ``value_raw`` occurs in that transcription -- decided by :func:`paperfacts.grounding.is_grounded`,
  the same test grounding applies to parser text. A model asked to confirm a number tends to; a model asked
  what it sees has no side to take.
- **Both lanes get the same treatment.** Which values are checked is decided by a lane-blind rule
  (:func:`select_targets`), and every value is checked with the same prompt, model, DPI and padding. A
  verdict that favoured one lane by construction would contaminate the disagreement signal the tool exists
  to measure.
- **A verdict is a fourth reading, not the truth.** ``confirmed`` means a model that never saw the parser's
  text also reads these characters in this region; ``contradicted`` means it does not. Both are recorded
  with the transcription, so a reviewer can overrule either. Nothing here ever deletes a value.

The stage runs after comparison and before export, on the values the two lanes could not settle between
them (``policy="disputed"``): conflicts, ambiguities, one-sided values, and anything grounding could not
locate. ``policy="all"`` checks every value, which is how the "both lanes agree and both are wrong" rate --
the number that justifies this stage -- is measured.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperfacts.compare import ComparisonReport
from paperfacts.config import (
    DEFAULT_VLM_CONCURRENCY,
    DEFAULT_VLM_CROP_DPI,
    DEFAULT_VLM_CROP_MAX_PIXELS,
    DEFAULT_VLM_CROP_PADDING,
    DEFAULT_VLM_POLICY,
    VALIDATION_POLICIES,
    ValidationPolicy,
)
from paperfacts.errors import LlmError
from paperfacts.fields import FIELD_BY_NAME
from paperfacts.grounding import is_grounded
from paperfacts.llm import VisionClient
from paperfacts.models import Backend, NormalizedBBox, ParsedArtifact
from paperfacts.pdf import png_bytes, render_region
from paperfacts.prompts import validation_system_prompt, validation_user_prompt
from paperfacts.records import FieldValue, LaneExtraction
from paperfacts.storage import DataLayout, write_bytes_atomic

logger = logging.getLogger(__name__)

# What the VLM's reading said about a value. ``not_checked`` is a value the stage could not put in front of
# the model at all (no citation, no PDF); ``error`` is a request that failed after its retries. Both are
# kept in the report so "not validated" and "validated and wrong" never look alike.
Verdict = Literal["confirmed", "contradicted", "illegible", "not_checked", "error"]
VERDICTS: tuple[Verdict, ...] = ("confirmed", "contradicted", "illegible", "not_checked", "error")
# Why a value was selected. The comparison status that put it on the list, ``ungrounded`` for a value
# grounding flagged in either lane, ``all`` under the exhaustive policy.
Reason = Literal["conflict", "ambiguous", "missing", "ungrounded", "all"]
# Where in its lane a value lives. Together with the backend this is what makes a value's key unique, and
# it is spelled the way the comparison report spells scopes so the web UI can rebuild the key from a row.
OWNER_TARGET = "target"
OWNER_UNATTRIBUTED = "unattributed"
# The lenient parse of the model's reply: the first JSON object in the text, code fences or not.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
# A multiplication sign between two digits, with or without spaces around it. The extractor quotes the
# parser's spacing ("1.2 × 10^-4") and the model writes its own ("1.2×10⁻⁴"); grounding is deliberately
# strict about spaces inside numbers, so this one fold is applied to *both* sides before it judges them.
_DIGIT_TIMES_DIGIT = re.compile(r"(?<=\d)\s*[×✕✖x]\s*(?=\d)")


# ---- Stored models ------------------------------------------------------------------------------------


class RegionCrop(BaseModel):
    """The image the model was shown: which page, which box, at what resolution, and its digest.

    The digest is what the LLM cache keyed the request on, so a stored verdict can be traced to the exact
    bytes that produced it, and ``path`` is where those bytes are on disk for a reviewer to look at.
    """

    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=0)
    bbox: NormalizedBBox
    dpi: int = Field(gt=0)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    source_ids: tuple[str, ...] = Field(description="the cited blocks the box is the union of (same page only)")
    image_sha256: str = Field(min_length=64, max_length=64)
    path: str = Field(description="file name under the document's crops/ directory")


class ValueValidation(BaseModel):
    """One value, one region, one reading, one verdict."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="value_key(): backend|owner|field|value_raw|condition|source_ids")
    backend: Backend
    owner: str = Field(description='"target", "sample:<id>" or "unattributed"')
    field: str
    value_raw: str
    unit_raw: str | None = None
    condition: str | None = None
    source_ids: tuple[str, ...] = ()
    reason: Reason
    verdict: Verdict
    transcription: str = Field(default="", description="the model's verbatim reading of the region")
    legible: bool | None = Field(default=None, description="the model's own judgement of the region")
    crop: RegionCrop | None = None
    detail: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    cached: bool = Field(default=False, description="the reading came from the LLM cache, not a new request")


class ValidationCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    confirmed: int = 0
    contradicted: int = 0
    illegible: int = 0
    not_checked: int = 0
    error: int = 0
    total: int = 0
    by_backend: dict[str, dict[str, int]] = Field(default_factory=dict, description="per lane, verdict -> count")
    by_reason: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="per selection reason, verdict -> count"
    )


class ValidationReport(BaseModel):
    """Every value the stage looked at for one document, under one set of keys."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    extractor_key: str
    comparison_key: str
    validation_key: str
    model: str
    policy: ValidationPolicy
    values: tuple[ValueValidation, ...] = ()
    counts: ValidationCounts = Field(default_factory=ValidationCounts)
    usage: dict[str, int] = Field(default_factory=dict)

    def verdicts(self) -> dict[str, ValueValidation]:
        return {value.key: value for value in self.values}

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---- Which values are checked ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationTarget:
    backend: Backend
    owner: str
    value: FieldValue
    reason: Reason


def value_key(backend: Backend, owner: str, value: FieldValue) -> str:
    """The identity of one piece of evidence: where it sits and what it quotes.

    A ``FieldValue`` has no id of its own, and the normalised copies that travel through comparison differ
    from the stored ones in ``value``/``unit``, so identity is the raw quote plus its citations. The web UI
    builds the same string from a comparison row (``facts.js``), so the field order here is a contract.
    """
    return "|".join((backend, owner, value.field, value.value_raw, value.condition or "", ",".join(value.source_ids)))


def _same_evidence(a: FieldValue, b: FieldValue) -> bool:
    """Equal on what identifies the evidence, ignoring the derived fields normalisation and grounding fill."""
    return (a.field, a.value_raw, a.unit_raw, a.condition, a.source_ids) == (
        b.field,
        b.value_raw,
        b.unit_raw,
        b.condition,
        b.source_ids,
    )


def owner_of(lane: LaneExtraction, value: FieldValue) -> str | None:
    """Where ``value`` lives in ``lane``, or None when the lane does not hold it.

    Looked up in the lane rather than parsed from the comparison report's scope string: the scope joins two
    sample ids with ``|`` and would be ambiguous for a sample whose id contains one.
    """
    if lane.target is not None and any(_same_evidence(value, other) for other in lane.target.fields):
        return OWNER_TARGET
    for sample in lane.samples:
        if any(_same_evidence(value, other) for other in sample.fields):
            return f"sample:{sample.sample_id}"
    if any(_same_evidence(value, other) for other in lane.unattributed):
        return OWNER_UNATTRIBUTED
    return None


def select_targets(
    lanes: Mapping[Backend, LaneExtraction],
    report: ComparisonReport,
    *,
    policy: ValidationPolicy = DEFAULT_VLM_POLICY,
) -> tuple[ValidationTarget, ...]:
    """The values to show the model, in a fixed order, each once.

    The rule is lane-blind by construction: a conflict or an ambiguity puts *both* sides on the list, a
    one-sided value is the one side there is, and grounding's flag is applied to both lanes alike. The same
    value reached from two comparison rows is listed once, under the first reason that named it; the
    reasons rank conflict > ambiguous > missing > ungrounded only through that order of traversal.
    """
    if policy not in VALIDATION_POLICIES:
        raise ValueError(f"unknown validation policy: {policy!r}, expected one of {', '.join(VALIDATION_POLICIES)}")
    chosen: dict[str, ValidationTarget] = {}

    def add(backend: Backend, value: FieldValue, reason: Reason) -> None:
        lane = lanes.get(backend)
        if lane is None:
            return
        owner = owner_of(lane, value)
        if owner is None:
            # A comparison row quoting a value its lane no longer holds: the report and the lane disagree
            # about the evidence, which is worth a log line but not a crash of the whole stage.
            logger.warning("validation target not found in its lane backend=%s field=%s", backend, value.field)
            return
        chosen.setdefault(value_key(backend, owner, value), ValidationTarget(backend, owner, value, reason))

    if policy == "all":
        for backend, lane in lanes.items():
            for value in lane.values():
                add(backend, value, "all")
    else:
        for status in ("conflict", "ambiguous", "missing"):
            for comparison in report.comparisons:
                if comparison.status != status:
                    continue
                if comparison.a is not None:
                    add(report.backend_a, comparison.a, status)
                if comparison.b is not None:
                    add(report.backend_b, comparison.b, status)
        for backend, lane in lanes.items():
            for value in lane.ungrounded():
                add(backend, value, "ungrounded")
    return tuple(
        sorted(
            chosen.values(),
            key=lambda t: (t.backend, t.owner, t.value.field, t.value.value_raw, t.value.condition or ""),
        )
    )


# ---- The region a value was cited from ------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    page: int
    bbox: NormalizedBBox
    source_ids: tuple[str, ...]


def region_for(
    value: FieldValue, artifact: ParsedArtifact, *, padding: float = DEFAULT_VLM_CROP_PADDING
) -> Region | None:
    """The union of the cited blocks on the first cited page, padded; None when nothing was cited.

    Citations on other pages are dropped rather than joined: a box spanning two pages is not a region of any
    page. The first cited block decides which page, because the extractor is asked to cite the most
    specific block first.
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
    bbox = same_page[0].bbox
    for block in same_page[1:]:
        bbox = bbox.union(block.bbox)
    return Region(page=page, bbox=bbox.padded(padding), source_ids=tuple(block.source_id for block in same_page))


def bbox_key(bbox: NormalizedBBox) -> str:
    """A file-name-safe spelling of a box, precise to a ten-thousandth of the page."""
    return f"{bbox.x1:.4f}-{bbox.y1:.4f}-{bbox.x2:.4f}-{bbox.y2:.4f}"


class CropStore:
    """Renders regions once and keeps the PNGs under ``crops/``.

    Two values cited from the same table ask for the same box, and a re-run asks for every box again; both
    are answered from disk. Rendering goes through :mod:`paperfacts.pdf` and its process-wide lock, so any
    number of worker threads may ask at once.
    """

    def __init__(
        self,
        layout: DataLayout,
        document_id: str,
        pdf_path: Path,
        *,
        dpi: int = DEFAULT_VLM_CROP_DPI,
        max_pixels: int = DEFAULT_VLM_CROP_MAX_PIXELS,
    ) -> None:
        self.layout = layout
        self.document_id = document_id
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.max_pixels = max_pixels
        self._lock = threading.Lock()
        self._memory: dict[Path, bytes] = {}

    def crop(self, region: Region) -> tuple[bytes, RegionCrop]:
        path = self.layout.crop_path(self.document_id, region.page, bbox_key(region.bbox), self.dpi)
        with self._lock:
            data = self._memory.get(path)
            if data is None and path.is_file():
                data = path.read_bytes()
            if data is None:
                image = render_region(self.pdf_path, region.page, region.bbox, dpi=self.dpi, max_pixels=self.max_pixels)
                data = png_bytes(image)
                write_bytes_atomic(path, data)
            self._memory[path] = data
        width, height = _png_size(data)
        return data, RegionCrop(
            page=region.page,
            bbox=region.bbox,
            dpi=self.dpi,
            width_px=width,
            height_px=height,
            source_ids=region.source_ids,
            image_sha256=_sha256(data),
            path=path.name,
        )


def _png_size(data: bytes) -> tuple[int, int]:
    """Width and height from the PNG header, so a cached file need not be decoded to describe it."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---- The reading and the verdict -------------------------------------------------------------------------


class Reading(BaseModel):
    """What the model is asked to return. ``extra="ignore"`` keeps it tolerant of a chatty model."""

    model_config = ConfigDict(extra="ignore")

    transcription: str = ""
    legible: bool = True


def parse_reading(text: str) -> tuple[Reading, str]:
    """The model's reply as a :class:`Reading`, plus a note when it had to be coaxed.

    A vision endpoint without JSON mode may wrap the object in a code fence or a sentence; the first
    ``{...}`` in the text is taken. A reply with no JSON at all is treated as the transcription itself:
    a model that ignored the format and simply transcribed has still done the useful part of the job, and
    the note records that the structure was missing.
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


# ---- The stage ---------------------------------------------------------------------------------------------


def validate_lanes(
    *,
    document_id: str,
    pdf_path: Path | None,
    lanes: Mapping[Backend, LaneExtraction],
    report: ComparisonReport,
    artifacts: Mapping[Backend, ParsedArtifact],
    client: VisionClient,
    layout: DataLayout,
    validation_key: str,
    policy: ValidationPolicy = DEFAULT_VLM_POLICY,
    crop_dpi: int = DEFAULT_VLM_CROP_DPI,
    crop_padding: float = DEFAULT_VLM_CROP_PADDING,
    crop_max_pixels: int = DEFAULT_VLM_CROP_MAX_PIXELS,
    concurrency: int = DEFAULT_VLM_CONCURRENCY,
    refresh: bool = False,
) -> ValidationReport:
    """Check every selected value and return the report. Nothing is written to disk but the crops.

    Requests run ``concurrency`` at a time; the report is assembled in the fixed order of
    :func:`select_targets`, so two runs over the same inputs produce the same file. One failed request is
    an ``error`` verdict, not a failed stage; a stage where *every* request failed raises, because that is a
    misconfigured endpoint and not a hundred coincidences.
    """
    targets = select_targets(lanes, report, policy=policy)
    store = (
        None
        if pdf_path is None or not pdf_path.is_file()
        else CropStore(layout, document_id, pdf_path, dpi=crop_dpi, max_pixels=crop_max_pixels)
    )
    system = validation_system_prompt()

    def check(target: ValidationTarget) -> ValueValidation:
        return _validate_one(target, artifacts, store, client, system, padding=crop_padding, refresh=refresh)

    with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="paperfacts-vlm") as pool:
        values = tuple(pool.map(check, targets))

    errors = [value for value in values if value.verdict == "error"]
    if values and len(errors) == len(values):
        raise LlmError(f"every vision request failed ({len(errors)}); first error: {errors[0].detail}")

    usage: dict[str, int] = {}
    for value in values:
        for key, amount in value.usage.items():
            usage[key] = usage.get(key, 0) + amount
    validation = ValidationReport(
        document_id=document_id,
        extractor_key=report.extractor_key,
        comparison_key=report.comparison_key,
        validation_key=validation_key,
        model=client.model,
        policy=policy,
        values=values,
        counts=_count(values),
        usage=usage,
    )
    logger.info("validated doc=%s policy=%s counts=%s", document_id[:16], policy, validation.counts.model_dump())
    return validation


def _validate_one(
    target: ValidationTarget,
    artifacts: Mapping[Backend, ParsedArtifact],
    store: CropStore | None,
    client: VisionClient,
    system: str,
    *,
    padding: float,
    refresh: bool,
) -> ValueValidation:
    value = target.value
    base = {
        "key": value_key(target.backend, target.owner, value),
        "backend": target.backend,
        "owner": target.owner,
        "field": value.field,
        "value_raw": value.value_raw,
        "unit_raw": value.unit_raw,
        "condition": value.condition,
        "source_ids": value.source_ids,
        "reason": target.reason,
    }
    spec = FIELD_BY_NAME.get(value.field)
    if spec is None:
        return ValueValidation(**base, verdict="not_checked", detail=f"{value.field}: not in the field table")
    if store is None:
        return ValueValidation(**base, verdict="not_checked", detail="PDF not available; no region could be rendered")
    artifact = artifacts.get(target.backend)
    region = None if artifact is None else region_for(value, artifact, padding=padding)
    if region is None:
        return ValueValidation(**base, verdict="not_checked", detail="no valid citation, so there is no region to read")
    try:
        image, crop = store.crop(region)
        result = client.complete_vision(
            system=system, user=validation_user_prompt(spec), image_png=image, refresh=refresh
        )
    except (LlmError, OSError, ValueError, IndexError) as exc:
        logger.warning("vision request failed backend=%s field=%s: %s", target.backend, value.field, exc)
        return ValueValidation(**base, verdict="error", detail=f"{type(exc).__name__}: {exc}")
    reading, note = parse_reading(result.text)
    verdict, detail = adjudicate(value, reading)
    return ValueValidation(
        **base,
        verdict=verdict,
        transcription=reading.transcription,
        legible=reading.legible,
        crop=crop,
        detail="; ".join(part for part in (detail, note) if part),
        usage=dict(result.usage),
        cached=result.cached,
    )


def _count(values: Sequence[ValueValidation]) -> ValidationCounts:
    tally = Counter(value.verdict for value in values)
    by_backend: dict[str, Counter[str]] = {}
    by_reason: dict[str, Counter[str]] = {}
    for value in values:
        by_backend.setdefault(value.backend, Counter())[value.verdict] += 1
        by_reason.setdefault(value.reason, Counter())[value.verdict] += 1
    return ValidationCounts(
        confirmed=tally["confirmed"],
        contradicted=tally["contradicted"],
        illegible=tally["illegible"],
        not_checked=tally["not_checked"],
        error=tally["error"],
        total=len(values),
        by_backend={backend: dict(sorted(counts.items())) for backend, counts in sorted(by_backend.items())},
        by_reason={reason: dict(sorted(counts.items())) for reason, counts in sorted(by_reason.items())},
    )
