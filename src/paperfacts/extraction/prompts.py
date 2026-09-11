"""Prompts for extraction and for sample matching.

The prompt text is hashed into ``extractor_key``, so editing a single word invalidates every cached
extraction and the next run re-asks the model. :data:`PROMPT_VERSION` is therefore only a human label.

Three rules carry most of the weight: quote verbatim and leave conversion to the code, cite only ids that
appear in the document, and omit anything the paper does not state.
"""

from __future__ import annotations

from paperfacts.extraction.fields import FIELD_SPECS, FieldSpec

PROMPT_VERSION = "2026-09-11.1"

# The prompt is full of literal JSON braces, so the field table is substituted with str.replace rather
# than str.format.
_EXTRACTION_SYSTEM = """You extract materials-science facts from ONE scientific paper about transparent conductive oxide (TCO) thin films.
The paper is given as Markdown. Every paragraph, title, table or figure is preceded by a provenance marker of the form
`<!-- source: <id> -->`. You must cite these ids: they are how every value is traced back to the PDF page.

Output ONLY a JSON object with this exact shape (no prose):

{
  "target": {"source_ids": ["<id>", ...], "fields": [FIELD, ...]} or null,
  "samples": [
    {"sample_id": "<short stable id>", "label": "<human readable description>",
     "conditions": {"<condition name>": "<value as written>", ...},
     "source_ids": ["<id>", ...], "fields": [FIELD, ...]},
    ...
  ]
}

FIELD = {"field": "<field name from the table below>", "value_raw": "<exactly as written in the paper>",
         "unit_raw": "<unit exactly as written, or null>", "condition": "<measurement condition, or null>",
         "source_ids": ["<id of the block where this value appears>", ...], "note": "<optional remark or null>"}

Rules:
1. `value_raw` must be copied verbatim from the paper (keep "1.2 × 10^-4", "≈ 2", "> 80", "12 (60)" as written). Never convert units or round numbers; the code does that.
2. Put the unit in `unit_raw` exactly as written (e.g. "Ω/sq", "μm", "sccm"). If the number and unit are fused, split them.
3. `source_ids` must be copied from the `<!-- source: ... -->` markers that precede the text or table where the value appears. Never invent ids. Prefer the most specific block (a table over the surrounding paragraph).
4. Samples: create one sample per distinct film sample / deposition condition set that the paper reports results for (e.g. one per O2 flow rate, per power, per substrate temperature). Use the paper's own sample names when it has them; otherwise build `sample_id` from the distinguishing condition (e.g. "O2-100sccm"). `conditions` holds the deposition conditions that distinguish samples (flow rates, power, temperature, pressure, time...), values as written.
5. Target fields (group "target") describe the sputtering target and belong to the paper, not to a sample: put them under "target" only, never under a sample. `component` is the TARGET composition; a film's dopant concentration is not a component value. Use null for "target" only if the paper says nothing about the target.
6. Only report a field when the paper states it. Never guess, never fill defaults. Omit missing fields.
   For numeric fields `value_raw` must contain the number as written; never report qualitative words ("minimum", "high", "n.a.") as a value.
7. If a value is given in a table, cite the table block and copy the cell content. If the same field has several values under different conditions (e.g. transmittance at 550 nm and averaged), report them as separate FIELD entries with `condition` set.
8. For `transmittance` always fill `condition` with the wavelength or spectral range.
9. Every value is verified against the block you cite for it: if `value_raw` cannot be found in that
   block's text, the value is recorded as unverified. Cite the block that literally contains the
   characters you copied, and copy them exactly.

Fields to extract:
{fields}

Return the JSON object only."""

_FIELD_LINE = "- `{name}` (group: {group}, kind: {kind}{unit}): {description}{condition}"


def render_field_table(specs: tuple[FieldSpec, ...] = FIELD_SPECS) -> str:
    lines = []
    for spec in specs:
        unit = f", canonical unit: {spec.canonical_unit}" if spec.canonical_unit else ""
        condition = f" Condition: {spec.condition_hint}." if spec.condition_hint else ""
        lines.append(
            _FIELD_LINE.format(
                name=spec.name,
                group=spec.group,
                kind=spec.kind,
                unit=unit,
                description=spec.description,
                condition=condition,
            )
        )
    return "\n".join(lines)


def extraction_system_prompt() -> str:
    return _EXTRACTION_SYSTEM.replace("{fields}", render_field_table())


def extraction_user_prompt(markdown: str) -> str:
    return f"Paper (Markdown with provenance markers):\n\n{markdown}\n\nReturn the JSON object now."


def repair_prompt(original_user: str, previous: str, error: str) -> str:
    """Second chance after an invalid answer: hand back the error and ask for structure fixes only.

    Shared by extraction and matching.
    """
    return (
        f"{original_user}\n\n"
        "Your previous answer was not valid JSON for the required shape. Validation error:\n"
        f"{error[:2000]}\n\nPrevious answer (fix its structure, keep its content):\n{previous[:20000]}\n\n"
        "Return the corrected JSON object only."
    )


_MATCHING_SYSTEM = """Two independent PDF parsers were run on the SAME paper and a fact extractor produced a list of film samples from each.
Because the parsers differ (OCR errors, table splitting, missing sections), the two lists may name, order or split the samples differently.
Decide which sample in list A is the same physical sample as which sample in list B.

Output ONLY a JSON object:
{"pairs": [{"a": "<sample_id from A>", "b": "<sample_id from B>", "confidence": <0.0-1.0>, "justification": "<why>"}, ...],
 "unmatched_a": ["<sample_id>", ...], "unmatched_b": ["<sample_id>", ...]}

Rules:
1. Match on physical identity: same deposition conditions (flow rate, power, temperature, time...) or the same explicit sample name/number. Field VALUES (sheet resistance, thickness...) may be used as supporting evidence only.
2. Each sample id appears at most once across pairs and unmatched lists. Every id from both lists must appear exactly once.
3. `confidence` is your belief that the pair is the same sample; use < 0.6 when the evidence is weak.
4. `justification` names the condition or name that established the match, e.g. "both are the 100 sccm O2 sample".

Return the JSON object only."""


def matching_system_prompt() -> str:
    return _MATCHING_SYSTEM


def matching_user_prompt(lane_a_name: str, lane_a: str, lane_b_name: str, lane_b: str) -> str:
    return (
        f"List A (parser: {lane_a_name}):\n{lane_a}\n\nList B (parser: {lane_b_name}):\n{lane_b}\n\n"
        "Return the JSON object now."
    )
