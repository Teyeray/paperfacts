"""Figure reading: a vision model reads the values a property-vs-condition chart plots.

Many papers state a sample's sheet resistance or resistivity only as a marker on a chart ("Rs vs O2 flow"),
so the text-only lanes can never find it. This stage crops those charts out of the page and asks a vision
model to read them. It is paper-level, not a lane: both parsers cropped the same chart, so there is no
second, independent reading to compare, and the result stays out of the two-lane measurement entirely.

Boundaries, all from the measurement in ``.omc/research/figure-reading-accuracy.md``:

- **Only y is used.** The models snap x to the nearest tick label (1.25 comes back as 1.5), so a chart's x
  never creates a sample and never identifies one. It is stored for the reader, nothing more.
- **Approximate, and labelled so.** Every reading carries a precision: ±10 % on a linear axis, ±20 % on a
  log axis or a chart with four or more series (the p90 error there was about twice as large).
- **Never in the dataset's main cells and never compared.** Readings get their own sheet and their own
  section in the web page; :mod:`paperfacts.dataset` and :mod:`paperfacts.compare` never mix them in.
- **Selection is deterministic code.** A chart is read when the caption of its *figure* (not of the panel:
  MinerU gives a panel of a multi-panel figure the caption "(a) (c)", the neighbours' labels) names a film
  property by one of its retrieval keywords. The model's own refusal handles the charts that are not
  property-vs-condition after all (spectra, XRD patterns); it refused both negatives in the measurement.
- **The model quotes, the code converts.** y comes back in the axis's own unit, multiplier included
  ("10^2 ohm/sq"), and :func:`paperfacts.normalize.convert_to_canonical` does the arithmetic.

The prompt lives here rather than in :mod:`paperfacts.prompts` on purpose: that module's source is hashed
into ``extractor_key``, so putting it there would rename every stored extraction whenever this prompt is
tuned. This module's source is hashed into ``figure_key`` instead.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperfacts.fields import FIELD_SPECS, FieldSpec
from paperfacts.llm import VisionClient
from paperfacts.models import Backend, NormalizedBBox, ParsedArtifact, SourceBlock
from paperfacts.normalize import convert_to_canonical, normalize_text

# Keyword matching is passages.py's, so a caption names a field under exactly the rules a passage-mode
# question uses to find it in the text. The two helpers are private there; they are imported rather than
# made public because renaming them would change passages.py's source, which is hashed into every
# passage-mode extractor_key and would rename every stored extraction for no change in behaviour.
from paperfacts.passages import _names, _searchable
from paperfacts.storage import write_text_atomic

logger = logging.getLogger(__name__)

# Sampling for the vision request. Temperature 0 is what was measured. The token budget has room for the
# model's reasoning, which was most of the 3-9k tokens a chart cost, plus the JSON after it.
TEMPERATURE = 0.0
MAX_TOKENS = 16384
# One retry: a request that hangs for five minutes twice is not going to answer on the third try, and
# every retry of a slow chart holds up the whole paper.
RETRY_ATTEMPTS = 2
# The precision a reading is labelled with, from the measured p90 plus headroom.
LINEAR_PRECISION = 0.10
LOG_PRECISION = 0.20
# Four or more series on one chart crowd the markers; the crowded chart had twice the p90 error.
CROWDED_SERIES = 4
# The one signal worth passing on from the model's self-reported confidence: it only went below this on
# markers hidden behind other markers, which were also its worst readings (27-78 % error).
LOW_CONFIDENCE = 0.8
# A panel label MinerU files as its own block ("(b)") must not split a figure group in two.
_PANEL_LABEL_CHARS = 12
# The caption that describes a whole figure: "Fig. 3", "FIGURE 2", "Figure S1", with any leading markup.
_FIGURE_CAPTION = re.compile(r"^[\W_]*(?P<label>fig(?:ure)?s?\.?\s*S?\d+[a-z]?)", re.IGNORECASE)

SYSTEM_PROMPT = "You read numeric data off charts in scientific papers. Answer with one JSON object only."

# The "Final prompt" of the accuracy report with its two edits marked ⊕ there applied: interpolate x between
# ticks, and strict JSON. Added for the pipeline: the list of fields and a "field" per y axis, so the model
# answers only for fields this table has and the code never guesses which axis is which property.
USER_PROMPT = """\
You are reading data off a chart from a scientific paper about thin films (e.g. sputtered transparent conductive oxides).
The image may be one panel of a multi-panel figure. The figure caption is:
<<<{caption}>>>

Only these film properties are of interest (name: meaning, usual unit):
{fields}

Step 1 - decide whether this chart is a "property-vs-condition" chart: a film property (e.g. sheet resistance, resistivity, carrier concentration, mobility, thickness, average transmittance, figure of merit) plotted against a preparation or treatment condition (e.g. gas flow or ratio, power, pressure, temperature, thickness, doping level, sample name), with one discrete marker per sample.
Spectra, XRD/XPS/Raman patterns, J-V curves, images, maps, schematics and analysis plots (Tauc, Williamson-Hall, fits) are NOT property-vs-condition charts. If it is not one, or none of its y axes plots one of the properties listed above, output {{"chart_type": "not_property_vs_condition", "reason": "..."}} and nothing else.

Step 2 - otherwise read the chart carefully:
- Read every plotted data MARKER (squares, circles, triangles, stars...). Do NOT read points from fitted curves, splines or guide-to-the-eye lines; a line vertex without a marker is not a data point. Legend symbols are not data.
- For each y axis, determine the scale from the tick labels: linear or logarithmic (ticks like 10^1, 10^2, 10^3 evenly spaced => log; interpolate logarithmically between them). Note axis breaks.
- For each y axis, set "field" to the name of the listed property it plots, or null if it plots none of them. Report points only for series on an axis whose field is not null.
- If there are several y axes (left/right, or several right axes), decide which axis each series belongs to (colour of axis and labels, arrows, legend) and read each series against ITS OWN axis.
- Report y in the units printed on that axis INCLUDING any multiplier written in the axis title (e.g. axis "Sheet resistance [10^2 ohm/sq]" with a marker at the "25" gridline => y = 25, unit "10^2 ohm/sq"). Do not convert.
- If the chart prints the numeric value next to a point, use the printed value.
- Report x exactly as the tick label/category of that marker (e.g. 400, 1.5, "As-deposited", "ITO-RT"); if markers of one series are shifted slightly sideways to avoid overlap, still report the nominal x of the group.
  If a marker lies between labelled ticks, interpolate its x position instead of rounding to the nearest tick, and set "x_on_tick": false.
- Markers hidden behind other markers: include them if you can infer their position, with low confidence.
- Include error-bar half-width if error bars are visible, else null.
- Per point, give a confidence in [0,1] for the y reading.

Output ONLY this JSON (strict JSON: no comments, no trailing text):
{{"chart_type": "property_vs_condition",
 "x_axis": {{"quantity": "...", "unit": "...", "scale": "linear|log|categorical"}},
 "y_axes": [{{"id": "left", "field": "<listed property name or null>", "quantity": "...", "unit": "... (with multiplier)", "scale": "linear|log", "broken": false}}],
 "series": [{{"label": "legend text or quantity name", "y_axis": "left|right|right2", "marker": "..."}}],
 "points": [{{"series": "<label>", "x": <number or string>, "x_on_tick": true, "y": <number>, "y_error": <number or null>, "confidence": <0..1>}}]}}
"""


def figure_fields() -> tuple[FieldSpec, ...]:
    """The fields a chart may be read for: numeric film properties.

    Only the ``film`` group, although process fields are sample-level too: a chart's y axis is what gets
    read, and a y axis plots a property of the film. Letting "annealing temperature" select charts would
    spend the per-paper budget on every spectrum whose caption says what the film was annealed at.
    """
    return tuple(spec for spec in FIELD_SPECS if spec.group == "film" and spec.kind == "numeric")


# ---- Stored result -----------------------------------------------------------------------------------

PanelStatus = Literal["read", "not_chart", "unreadable", "error"]


class FigureReading(BaseModel):
    """One marker read off one chart panel, with the figure block it came from as its citation."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    page: int = Field(ge=0, description="0-based page")
    bbox: NormalizedBBox
    figure: str | None = Field(default=None, description='the whole figure\'s label, e.g. "Fig. 3"')
    caption: str
    panel: int = Field(ge=1, description="1-based position of this panel within its figure group")
    field: str
    series: str | None = None
    x_quantity: str | None = None
    x_value: float | str | None = Field(default=None, description="for the reader only; never identifies a sample")
    x_unit: str | None = None
    x_on_tick: bool | None = None
    y_raw: float
    y_unit_raw: str | None = None
    y_error_raw: float | None = None
    y: float | None = Field(default=None, description="y in the field's canonical unit; None when it would not convert")
    unit: str | None = None
    scale: str = "linear"
    precision: float = Field(description="relative half-width to show, e.g. 0.1 for ±10 %")
    confidence: float | None = None
    approximate: bool = True
    note: str | None = None


class FigurePanel(BaseModel):
    """What happened to one panel, including the ones that yielded nothing: the audit trail of the stage."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    page: int
    figure: str | None = None
    caption: str
    fields: tuple[str, ...]
    status: PanelStatus
    detail: str = ""
    readings: int = 0
    usage: dict[str, int] = Field(default_factory=dict)


class FigureReadings(BaseModel):
    """One document's figure readings, stored as ``figures/<figure_key>.json``."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    figure_key: str
    model: str
    backend: Backend | None = Field(default=None, description="whose figure blocks were cropped")
    panels: tuple[FigurePanel, ...] = ()
    readings: tuple[FigureReading, ...] = ()
    usage: dict[str, int] = Field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """False when a request failed outright: a rerun should ask again, not serve this file."""
        return all(panel.status != "error" for panel in self.panels)

    def write(self, path: Path) -> None:
        write_text_atomic(path, self.model_dump_json(indent=2))

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


# ---- Selection -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureGroup:
    """Adjacent figure and caption blocks on one page: the panels of one figure plus its caption."""

    page: int
    panels: tuple[SourceBlock, ...]
    captions: tuple[SourceBlock, ...]
    # The caption block that describes the whole figure, when one of them does.
    figure_caption: SourceBlock | None = None

    @property
    def caption(self) -> str:
        """The whole figure's caption, or, when no caption says "Fig. N", everything the group has."""
        if self.figure_caption is not None:
            return normalize_text(self.figure_caption.content)
        return " ".join(normalize_text(block.content) for block in self.captions)

    @property
    def label(self) -> str | None:
        match = _FIGURE_CAPTION.match(self.caption)
        return " ".join(match.group("label").split()) if match else None

    def fields(self, specs: Sequence[FieldSpec]) -> tuple[FieldSpec, ...]:
        """The fields the figure's caption names. Only the whole-figure caption counts when there is one: a
        panel's "(a) (c)" names nothing, and a neighbouring figure's caption is about another chart."""
        blocks = (self.figure_caption,) if self.figure_caption is not None else self.captions
        texts = [_searchable(block) for block in blocks]
        return tuple(spec for spec in specs if any(_names(spec.keywords, text) for text in texts))


def _is_figure_caption(block: SourceBlock) -> bool:
    return block.type == "caption" and _FIGURE_CAPTION.match(normalize_text(block.content)) is not None


def _is_panel_label(block: SourceBlock) -> bool:
    return block.type in {"text", "unknown"} and len(block.content.strip()) <= _PANEL_LABEL_CHARS


def figure_groups(blocks: Sequence[SourceBlock]) -> tuple[FigureGroup, ...]:
    """Group a document's blocks into figures, in document order.

    A run of figure and caption blocks on one page is one figure until a caption that starts "Fig. N" ends
    it; that caption belongs to the panels before it. A "Fig. N" caption that arrives before any panel (a
    caption placed above its figure) opens the group instead, and the next one closes it. Anything else on
    the page -- except a short panel label like "(b)" -- ends the run, and so does a page break.

    MinerU hangs every caption under the image it was attached to, so "Fig. N" may follow the *first* panel
    and the remaining panels arrive after it with only "(b)"-style captions. A run with no figure caption
    that directly follows a closed figure on the same page, with nothing but figures and captions between
    them, is therefore the rest of that figure and joins it.
    """
    groups: list[FigureGroup] = []
    panels: list[SourceBlock] = []
    captions: list[SourceBlock] = []
    whole: SourceBlock | None = None
    page: int | None = None
    # Whether the last group was closed by its own caption with nothing after it yet but figure blocks.
    continues_last = False

    def close(*, by_caption: bool = False) -> None:
        nonlocal panels, captions, whole, continues_last
        if panels and whole is None and continues_last and groups and groups[-1].page == panels[0].page:
            last = groups[-1]
            groups[-1] = dataclasses.replace(
                last, panels=last.panels + tuple(panels), captions=last.captions + tuple(captions)
            )
        elif panels:
            groups.append(
                FigureGroup(page=panels[0].page, panels=tuple(panels), captions=tuple(captions), figure_caption=whole)
            )
        continues_last = by_caption
        panels, captions, whole = [], [], None

    for block in blocks:
        if block.page != page:
            close()
            continues_last = False
            page = block.page
        if block.type == "figure":
            panels.append(block)
        elif _is_figure_caption(block):
            if whole is not None:
                close()  # the previous figure was opened by its caption; this one starts the next
            below_its_panels = bool(panels)
            captions.append(block)
            whole = block
            if below_its_panels:
                close(by_caption=True)
        elif block.type == "caption":
            captions.append(block)
        elif not _is_panel_label(block):
            close()
            continues_last = False
    close()
    return tuple(groups)


@dataclass(frozen=True)
class PanelRequest:
    """One panel the model will be shown, with the question's context."""

    block: SourceBlock
    panel: int
    group: FigureGroup
    fields: tuple[FieldSpec, ...]


def select_panels(blocks: Sequence[SourceBlock], *, limit: int) -> tuple[PanelRequest, ...]:
    """The panels worth a question, in document order, at most ``limit`` of them."""
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    specs = figure_fields()
    chosen: list[PanelRequest] = []
    for group in figure_groups(blocks):
        fields = group.fields(specs)
        if not fields:
            continue
        for index, block in enumerate(group.panels, start=1):
            if len(chosen) == limit:
                logger.info("figure limit %d reached; later charts are not read", limit)
                return tuple(chosen)
            chosen.append(PanelRequest(block=block, panel=index, group=group, fields=fields))
    return tuple(chosen)


def user_prompt(caption: str, fields: Sequence[FieldSpec]) -> str:
    listed = "\n".join(f"- {spec.name}: {spec.description} ({spec.canonical_unit})" for spec in fields)
    return USER_PROMPT.format(caption=caption, fields=listed)


# ---- Parsing the answer ---------------------------------------------------------------------------------

_FENCE = re.compile(r"```[a-zA-Z]*")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments and trailing commas outside strings; qwen3-vl-plus put ``//``
    notes in its JSON."""
    out: list[str] = []
    index, in_string = 0, False
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if char == "\\" and index + 1 < len(text):
                out.append(text[index + 1])
                index += 1
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            out.append(char)
        elif text.startswith("//", index):
            newline = text.find("\n", index)
            index = len(text) if newline == -1 else newline
            continue
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end == -1 else end + 2
            continue
        elif char in "}]" and (trimmed := "".join(out).rstrip()).endswith(","):
            out = [trimmed[:-1], char]
        else:
            out.append(char)
        index += 1
    return "".join(out)


def parse_answer(text: str) -> dict[str, Any]:
    """The first JSON object in the reply, read leniently.

    No JSON mode is requested (not every vision endpoint accepts it), so the reply may come fenced, with
    reasoning before it, with comments inside it or with a trailing comma. Only an object that says what it
    is (``chart_type`` or ``chart``) counts: when the outer object is broken, the first inner one that parses
    is an axis or a point, and reading that as the answer would file a real chart as refused, for good.
    ``NaN`` and ``Infinity`` are refused too; they would be stored as ``null`` and break the file.
    """
    cleaned = _strip_comments(_FENCE.sub("", _THINK.sub("", text)))
    decoder = json.JSONDecoder(parse_constant=_no_constant)
    start = cleaned.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(cleaned, start)
        except ValueError:
            start = cleaned.find("{", start + 1)
            continue
        if isinstance(value, dict) and ("chart_type" in value or "chart" in value):
            return value
        start = cleaned.find("{", start + 1)
    raise ValueError(f"no chart answer in the reply: {text[:200]!r}")


def _no_constant(name: str) -> float:
    raise ValueError(f"{name} is not a reading")


class _Axis(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    # Everything optional: a null in one key must not cost the whole axis.
    id: str | None = None
    field: str | None = None
    quantity: str | None = None
    unit: str | None = None
    scale: str | None = None
    broken: bool | None = None


class _XAxis(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    quantity: str | None = None
    unit: str | None = None
    scale: str | None = None


class _Series(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    label: str | None = None
    y_axis: str | None = None


class _Point(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    series: str | None = None
    x: float | str | None = None
    x_on_tick: bool | None = None
    y: float = Field(allow_inf_nan=False)
    y_error: float | None = Field(default=None, allow_inf_nan=False)
    confidence: float | None = None


def is_refusal(answer: dict[str, Any]) -> bool:
    """The tested refusal is ``chart_type: not_property_vs_condition``; ``{"chart": false}`` is accepted too.

    Only an explicit refusal counts: a variant spelling of the chart type ("property-vs-condition") on an
    answer that carries points is still a chart, and refusing it would be cached for good.
    """
    if answer.get("chart") is False:
        return True
    kind = re.sub(r"[^a-z]", "", str(answer.get("chart_type") or "").lower())
    return kind.startswith("not") or (kind != "propertyvscondition" and not answer.get("points"))


def _items[M: BaseModel](answer: dict[str, Any], key: str, model: type[M]) -> list[M]:
    """The entries of a list that validate; one malformed point must not cost the rest of the chart."""
    raw = answer.get(key)
    if not isinstance(raw, list):
        return []
    items: list[M] = []
    for entry in raw:
        try:
            items.append(model.model_validate(entry))
        except ValidationError as exc:
            logger.warning("figure answer: skipping a malformed %s entry: %s", key, str(exc)[:200])
    return items


def _axis_unit(unit: str | None) -> str | None:
    """The axis unit without the brackets an axis title wraps it in: "[10^2 ohm/sq]" -> "10^2 ohm/sq"."""
    if unit is None:
        return None
    stripped = unit.strip()
    while len(stripped) >= 2 and stripped[0] + stripped[-1] in {"()", "[]", "{}"}:
        stripped = stripped[1:-1].strip()
    return stripped or None


def precision_for(scale: str, series_count: int) -> float:
    """±20 % on a log axis or a crowded chart, ±10 % otherwise: the measured p90 with headroom."""
    return LOG_PRECISION if scale.lower().startswith("log") or series_count >= CROWDED_SERIES else LINEAR_PRECISION


def readings_from_answer(answer: dict[str, Any], request: PanelRequest) -> tuple[FigureReading, ...]:
    """Turn a chart answer into readings: only for the fields asked about, y converted, precision labelled."""
    specs = {spec.name.lower(): spec for spec in request.fields}
    axes = {axis.id or "left": axis for axis in _items(answer, "y_axes", _Axis)}
    series = {entry.label or "": entry for entry in _items(answer, "series", _Series)}
    points = _items(answer, "points", _Point)
    try:
        x_axis = _XAxis.model_validate(answer.get("x_axis") or {})
    except ValidationError:
        x_axis = _XAxis()
    series_count = max(len(series), len({point.series for point in points}))
    only_axis = next(iter(axes.values())) if len(axes) == 1 else None

    readings: list[FigureReading] = []
    for point in points:
        entry = series.get(point.series or "")
        axis = axes.get(entry.y_axis) if entry is not None and entry.y_axis else None
        axis = axis or only_axis
        if axis is None or axis.field is None:
            continue
        spec = specs.get(axis.field.strip().lower())
        if spec is None:
            continue  # a field nobody asked about, or one this table does not have
        unit_raw = _axis_unit(axis.unit)
        value, unit, note = convert_to_canonical(spec, point.y, unit_raw)
        notes = [note] if note else []
        if point.confidence is not None and point.confidence < LOW_CONFIDENCE:
            notes.append(f"model confidence {point.confidence:g}: likely a hidden or overlapping marker")
        if axis.broken:
            notes.append("broken axis")
        readings.append(
            FigureReading(
                source_id=request.block.source_id,
                page=request.block.page,
                bbox=request.block.bbox,
                figure=request.group.label,
                caption=request.group.caption,
                panel=request.panel,
                field=spec.name,
                series=point.series,
                x_quantity=x_axis.quantity,
                x_value=point.x,
                x_unit=x_axis.unit,
                x_on_tick=point.x_on_tick,
                y_raw=point.y,
                y_unit_raw=unit_raw,
                y_error_raw=point.y_error,
                y=value,
                unit=unit if value is not None else None,
                scale="log" if (axis.scale or "").lower().startswith("log") else "linear",
                precision=precision_for(axis.scale or "linear", series_count),
                confidence=point.confidence,
                note="; ".join(notes) or None,
            )
        )
    return tuple(readings)


# ---- The stage -------------------------------------------------------------------------------------------

# (page index, bbox) -> PNG bytes. Injected so the stage never opens a PDF itself: pdf.py does, behind its lock.
CropRenderer = Callable[[int, NormalizedBBox], bytes]


def _read_panel(
    request: PanelRequest, image: bytes | Exception, client: VisionClient, *, refresh: bool
) -> tuple[FigurePanel, tuple[FigureReading, ...]]:
    block, group = request.block, request.group
    base = {
        "source_id": block.source_id,
        "page": block.page,
        "figure": group.label,
        "caption": group.caption,
        "fields": tuple(spec.name for spec in request.fields),
    }
    if isinstance(image, Exception):
        return FigurePanel(**base, status="error", detail=f"crop failed: {image}"[:500]), ()
    try:
        result = client.complete_vision(
            system=SYSTEM_PROMPT, user=user_prompt(group.caption, request.fields), image_png=image, refresh=refresh
        )
    except Exception as exc:  # any failure is this panel's alone; LlmError is the usual one
        logger.warning("figure %s: the vision request failed: %s", block.source_id, exc)
        return FigurePanel(**base, status="error", detail=f"{type(exc).__name__}: {exc}"[:500]), ()
    try:
        answer = parse_answer(result.text)
    except ValueError as exc:
        return FigurePanel(**base, status="unreadable", detail=str(exc)[:500], usage=result.usage), ()
    if is_refusal(answer):
        reason = str(answer.get("reason") or "not a property-vs-condition chart")
        return FigurePanel(**base, status="not_chart", detail=reason[:500], usage=result.usage), ()
    try:
        readings = readings_from_answer(answer, request)
    except (ValueError, TypeError) as exc:
        return FigurePanel(**base, status="unreadable", detail=str(exc)[:500], usage=result.usage), ()
    return FigurePanel(**base, status="read", readings=len(readings), usage=result.usage), readings


def _crop(render: CropRenderer, request: PanelRequest) -> bytes | Exception:
    """The panel's PNG, or the error that stopped it: one bad box must not cost the other panels."""
    try:
        return render(request.block.page, request.block.bbox)
    except Exception as exc:
        logger.warning("figure %s: the crop failed: %s", request.block.source_id, exc)
        return exc


def read_figures(
    artifact: ParsedArtifact,
    render: CropRenderer,
    client: VisionClient,
    *,
    figure_key: str,
    max_per_document: int,
    concurrency: int = 1,
    refresh: bool = False,
) -> FigureReadings:
    """Read every selected chart panel of ``artifact``: one vision request per panel.

    Crops are rendered first, one after another, because rendering goes through pdf.py's process-wide lock
    anyway; the requests then overlap, ``concurrency`` at a time, since each spends a minute waiting.
    """
    requests = select_panels(artifact.blocks, limit=max_per_document)
    images = [_crop(render, request) for request in requests]
    with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="paperfacts-figure") as pool:
        futures = [
            pool.submit(_read_panel, request, image, client, refresh=refresh)
            for request, image in zip(requests, images, strict=True)
        ]
        results = [future.result() for future in futures]
    usage: dict[str, int] = {}
    for panel, _ in results:
        for key, value in panel.usage.items():
            usage[key] = usage.get(key, 0) + value
    return FigureReadings(
        document_id=artifact.document_id,
        figure_key=figure_key,
        model=client.model,
        backend=artifact.backend,
        panels=tuple(panel for panel, _ in results),
        readings=tuple(reading for _, readings in results for reading in readings),
        usage=usage,
    )
