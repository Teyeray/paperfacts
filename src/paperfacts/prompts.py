"""Prompts for extraction and for sample matching.

The templates here are domain-free: every word about the domain -- what a sample is, the example units, the
JSON key paper-level values go under -- is a slot of the profile's :class:`~paperfacts.profile.PromptSlots`,
filled in by :func:`render` in a single pass. What the slots cannot say is computed from the profile first
(the group lists, rule 5 when the profile writes none, rule 8, the subset rule) and inserted as finished text.

Editing a single word here invalidates every cached extraction and the next run re-asks the model: the
system prompts are hashed into ``extractor_key`` by value, and this module's source is hashed too, which is
what covers the user half and the field-table template.

Three rules carry most of the weight, and they are repeated in every prompt here: quote verbatim and leave
conversion to the code, cite only ids that appear in the text shown, and omit anything the paper does not
state.

Two extraction prompts, for the two modes:

- ``extraction_*`` asks for the whole paper at once (document mode).
- ``inventory_*`` and ``field_*`` split that into "which samples exist?" followed by one question per
  field, each shown only the blocks :mod:`paperfacts.passages` retrieved for it (passage mode). The system
  half of the field question is deliberately identical for every field and every lane, so it is a single
  constant in the cache key; what differs goes in the user half.

All prompts are given text carrying ``<!-- source: <id> -->`` markers, and all of them must cite those ids:
they are how a value is traced back to a page and a bounding box.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from paperfacts.fields import FieldSpec
from paperfacts.profile import MARKER, DomainProfile, GroupSpec

# A value stated for part of the series. Left unsaid, the model reports "all films deposited at 100 °C" as one
# value with no sample and the series flag off, and the value reaches none of the samples it names (the gold set
# lost 36 substrate temperatures this way). Shared so the two modes place a subset the same way. Membership must
# be stated, never inferred: a subset whose samples the text does not identify stays unplaced, which is where
# each prompt says such a value goes (CLAUDE.md: never attach a value to a plausible neighbour).
_SUBSET_SCOPE = (
    "A value the paper states for a named subset of the listed samples -- {subset_examples} -- belongs to each"
    " sample in that subset when the excerpts or the sample list say exactly which listed samples form the"
    " subset: then report it once per sample of the subset, each time under that sample's own id, never as one"
    " entry without a sample id. Never guess which samples a subset holds. `applies_to_all_samples` stays reserved"
    " for a value that holds for the whole list; a subset, however large, is reported sample by sample."
)
# Rule 5 of the document prompt when the profile does not write its own: a paper-level value must not be
# attached to a sample. With no paper-level group the key still carries a whole-series value (rule 10), which
# document mode copies onto every sample, so the rule says that instead of naming an empty group list.
_PAPER_LEVEL_RULE = (
    'Paper-level fields (group {paper_groups}) belong to the paper, not to a sample: put them under "{paper_key}"'
    ' only, never under a sample. Use null for "{paper_key}" only if the paper states none of them.'
)
_NO_PAPER_LEVEL_RULE = (
    'No field is paper-level. Use "{paper_key}" only for a whole-series value as rule 10 describes; otherwise set'
    " it to null."
)
_CONDITION_RULE = "For `{name}` always fill `condition` with {rule}."
_NO_CONDITION_RULE = "When a field's line names a condition, always fill `condition` with it."

# The templates are full of literal JSON braces, which the marker pattern never matches (a JSON brace is
# followed by a quote, a space or a newline), so they are filled by render() rather than str.format.
_EXTRACTION_SYSTEM = """You extract {fact_noun} from ONE scientific paper about {domain_subject}.
The paper is given as Markdown. Every paragraph, title, table or figure is preceded by a provenance marker of the form
`<!-- source: <id> -->`. You must cite these ids: they are how every value is traced back to the PDF page.

Output ONLY a JSON object with this exact shape (no prose):

{
  "{paper_key}": {"source_ids": ["<id>", ...], "fields": [FIELD, ...]} or null,
  "samples": [
    {"sample_id": "<short stable id>", "label": "<human readable description>",
     "conditions": {"<condition name>": "<value as written>", ...},
     "source_ids": ["<id>", ...], "fields": [FIELD, ...]},
    ...
  ]
}

FIELD = {"field": "<field name from the table below>", "value_raw": "<exactly as written in the paper>",
         "unit_raw": "<unit exactly as written, or null>", "condition": "<measurement condition, or null>",
         "source_ids": ["<id of the block where this value appears>", ...], "note": "<optional remark or null>",
         "applies_to_all_samples": <true|false>}

Rules:
1. `value_raw` must be copied verbatim from the paper (keep "1.2 × 10^-4", "≈ 2", "> 80", "12 (60)" as written). Never convert units or round numbers; the code does that.
2. Put the unit in `unit_raw` exactly as written (e.g. {unit_examples}). If the number and unit are fused, split them.
   If a table column header or row label carries a power of ten, copy that factor into `unit_raw` together with the quantity symbol and unit, exactly as the header writes it, brackets included (e.g. {scaled_header_examples}), and copy the cell as it is: never apply the factor yourself. A header with a unit and no power of ten gives the unit alone ({plain_header_example}).
3. `source_ids` must be copied from the `<!-- source: ... -->` markers that precede the text or table where the value appears. Never invent ids. Prefer the most specific block (a table over the surrounding paragraph).
4. Samples: create one sample per distinct {sample_unit} that the paper reports results for {sample_examples}. Use the paper's own sample names when it has them; otherwise build `sample_id` from the distinguishing condition (e.g. "{sample_id_example}"). `conditions` holds the {condition_noun} that distinguish samples {condition_examples}, values as written.
   {sample_definition} If {no_samples_clause}, return an empty "samples" list.
5. {paper_level_rule}
6. Only report a field when the paper states it. Never guess, never fill defaults. Omit missing fields.
   {field_scope}
   For numeric fields `value_raw` must contain the number as written; never report qualitative words ("minimum", "high", "n.a.") as a value.
7. If a value is given in a table, cite the table block and copy the cell content. If the same field has several values under different conditions (e.g. {multi_condition_example}), report them as separate FIELD entries with `condition` set.
8. {condition_rules}
9. Every value is verified against the block you cite for it: if `value_raw` cannot be found in that
   block's text, the value is recorded as unverified. Cite the block that literally contains the
   characters you copied, and copy them exactly.
10. `applies_to_all_samples` is true ONLY for a sample-level field (group {sample_groups}) that the paper states once for the whole series -- {whole_series_examples} -- and never ties to one sample. Report such a value once, under "{paper_key}", with `applies_to_all_samples: true`; it is copied onto every sample. Never true for a value tied to only some of the samples, and false everywhere else, including every FIELD under a sample.
    {subset_scope} If they do not say which samples form the subset, leave the value out rather than attach it to samples it may not belong to.

Fields to extract:
{fields}

Return the JSON object only."""

_INVENTORY_SYSTEM = """You are reading ONE scientific paper about {domain_subject}, given as Markdown excerpts.
Every excerpt is preceded by a provenance marker of the form `<!-- source: <id> -->`.

Your only job is to list the {sample_plural} the paper reports results for. Do NOT report any measured values; later questions ask for those.

Output ONLY a JSON object with this exact shape (no prose):

{
  "samples": [
    {"sample_id": "<short stable id>", "label": "<human readable description>",
     "conditions": {"<condition name>": "<value as written>", ...},
     "source_ids": ["<id>", ...]},
    ...
  ],
  "{no_samples_key}": <true|false>
}

Rules:
1. Create one entry per distinct {sample_unit} that the paper reports results for {sample_examples}.
   {sample_definition}
2. Use the paper's own sample names when it has them; otherwise build `sample_id` from the distinguishing condition (e.g. "{sample_id_example}"). Keep every `sample_id` unique.
3. `conditions` holds the {condition_noun} that distinguish the samples {condition_examples}, values exactly as written in the paper.
4. `source_ids` must be copied from the `<!-- source: ... -->` markers of the excerpts the sample is described in. Never invent ids.
5. Only report samples the paper actually reports results for. Never guess and never invent a series: if the paper studies a single {sample_singular}, return exactly one sample; if the excerpts name none, return an empty list.
6. `{no_samples_key}` is true ONLY when the excerpts show that {no_samples_condition}; then "samples" is empty. It is false whenever {samples_present_condition}, and false when the excerpts simply do not say.

Return the JSON object only."""

_FIELD_SYSTEM = """You are reading selected excerpts from ONE scientific paper about {domain_subject}.
Every excerpt is preceded by a provenance marker of the form `<!-- source: <id> -->`. You must cite these ids: they are how every value is traced back to the PDF page.

You are asked about ONE field at a time. The field, and the list of samples this paper reports, are given in the question.

Output ONLY a JSON object with this exact shape (no prose):

{"values": [VALUE, ...]}

VALUE = {"sample_id": "<sample id from the list, or null>", "value_raw": "<exactly as written in the paper>",
         "unit_raw": "<unit exactly as written, or null>", "condition": "<measurement condition, or null>",
         "source_ids": ["<id of the excerpt where this value appears>", ...], "note": "<optional remark or null>",
         "applies_to_all_samples": <true|false>}

Rules:
1. `value_raw` must be copied verbatim from the excerpt (keep "1.2 × 10^-4", "≈ 2", "> 80", "12 (60)" as written). Never convert units or round numbers; the code does that.
2. Put the unit in `unit_raw` exactly as written (e.g. {unit_examples}). If the number and unit are fused, split them.
   If a table column header or row label carries a power of ten, copy that factor into `unit_raw` together with the quantity symbol and unit, exactly as the header writes it, brackets included (e.g. {scaled_header_examples}), and copy the cell as it is: never apply the factor yourself. A header with a unit and no power of ten gives the unit alone ({plain_header_example}).
3. `source_ids` must be copied from the `<!-- source: ... -->` markers shown here. Never invent ids and never cite an excerpt you were not shown. Prefer the most specific excerpt (a table over the surrounding paragraph).
4. `sample_id` must be copied exactly from the sample list in the question. Use null only for a paper-level field, or when the excerpts genuinely do not say which sample the value belongs to. If the list holds exactly one sample, every sample-level value belongs to it.
5. `applies_to_all_samples` is true ONLY when the excerpt states the value holds for every sample in the list -- the whole series, {whole_series_examples}. Then `sample_id` must be null. If the excerpt names one sample, give that id and false. Never true for a value the excerpts tie to only some of the samples; false everywhere else. A collective noun that covers most but not all of the listed samples ({partial_collective_example}) is false: give the individual sample ids the excerpt names.
   {subset_scope} If they do not, report it once with a null `sample_id`.
6. Report only the field you are asked about, and only where the excerpts state it. Never guess, never fill defaults, never carry a value over from another field.
   {field_scope}
   For a numeric field `value_raw` must contain the number as written; never report qualitative words ("minimum", "high", "n.a.") as a value.
7. If the same field has several values -- one per sample, or the same sample under different conditions (e.g. {multi_condition_example}) -- report them as separate entries with `condition` set.
8. {condition_rules}
9. Every value is verified against the excerpt you cite for it: if `value_raw` cannot be found in that excerpt's text, the value is recorded as unverified. Cite the excerpt that literally contains the characters you copied, and copy them exactly.
10. If the excerpts do not state this field at all, return {"values": []}. An empty answer is a correct answer.

Return the JSON object only."""

_FIELD_LINE = "- `{name}` (group: {group}, kind: {kind}{unit}): {description}{condition}{plausible}"
# Told to the model so it checks what it is quoting before it answers; the code drops what still falls outside.
_PLAUSIBLE = (
    " Plausible values are {range}; a number outside that range almost always belongs to {origin}, so check"
    " before reporting it."
)


def render(template: str, values: Mapping[str, str]) -> str:
    """``template`` with every ``{name}`` in ``values`` replaced, in one pass: an inserted value is never scanned
    again, so a slot or a field description that happens to contain ``{fields}`` reaches the model as written.
    A marker with no value is left as it stands."""
    return MARKER.sub(lambda match: values.get(match.group(1), match.group(0)), template)


def quoted_names(groups: Sequence[GroupSpec]) -> str:
    """``"a"``, ``"a" or "b"``, ``"a", "b" or "c"``: how a rule names the groups it is about."""
    names = [f'"{group.name}"' for group in groups]
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} or {names[-1]}"


def paper_level_rule(profile: DomainProfile) -> str:
    """The profile's own rule 5, or the generated one for its paper-level groups."""
    if profile.prompt.paper_level_rule is not None:
        return profile.prompt.paper_level_rule
    key = {"paper_key": profile.prompt.paper_key}
    if not profile.paper_groups:
        return render(_NO_PAPER_LEVEL_RULE, key)
    return render(_PAPER_LEVEL_RULE, {**key, "paper_groups": quoted_names(profile.paper_groups)})


def condition_rules(specs: Sequence[FieldSpec]) -> str:
    """Rule 8: which fields must always carry their measurement condition, and what that condition is."""
    rules = [_CONDITION_RULE.format(name=spec.name, rule=spec.condition_rule) for spec in specs if spec.condition_rule]
    return " ".join(rules) if rules else _NO_CONDITION_RULE


def _values(profile: DomainProfile) -> dict[str, str]:
    """Every slot and computed marker of the system prompts. The computed ones are finished text before the
    templates are rendered, so they too are inserted verbatim."""
    values = {**dataclasses.asdict(profile.prompt), "paper_level_rule": paper_level_rule(profile)}
    values["sample_groups"] = quoted_names(profile.sample_groups)
    values["condition_rules"] = condition_rules(profile.fields)
    values["subset_scope"] = render(_SUBSET_SCOPE, values)
    return values


def render_field_table(specs: Sequence[FieldSpec], implausible_origin: str) -> str:
    """One line per field. ``implausible_origin`` is the profile's :attr:`PromptSlots.implausible_origin`."""
    lines = []
    for spec in specs:
        unit = f", canonical unit: {spec.canonical_unit}" if spec.canonical_unit else ""
        condition = f" Condition: {spec.condition_hint}." if spec.condition_hint else ""
        described = spec.describe_range()
        plausible = _PLAUSIBLE.format(range=described, origin=implausible_origin) if described else ""
        lines.append(
            _FIELD_LINE.format(
                name=spec.name,
                group=spec.group,
                kind=spec.kind,
                unit=unit,
                description=spec.description,
                condition=condition,
                plausible=plausible,
            )
        )
    return "\n".join(lines)


def extraction_system_prompt(profile: DomainProfile) -> str:
    fields = render_field_table(profile.fields, profile.prompt.implausible_origin)
    return render(_EXTRACTION_SYSTEM, {**_values(profile), "fields": fields})


def extraction_user_prompt(markdown: str) -> str:
    return f"Paper (Markdown with provenance markers):\n\n{markdown}\n\nReturn the JSON object now."


def inventory_system_prompt(profile: DomainProfile) -> str:
    return render(_INVENTORY_SYSTEM, _values(profile))


def inventory_user_prompt(markdown: str) -> str:
    return f"Paper excerpts (Markdown with provenance markers):\n\n{markdown}\n\nReturn the JSON object now."


def field_system_prompt(profile: DomainProfile) -> str:
    """One text for every field and both lanes: what varies is the question, not the instructions."""
    return render(_FIELD_SYSTEM, _values(profile))


def field_user_prompt(spec: FieldSpec, sample_list: str, markdown: str, implausible_origin: str) -> str:
    """``sample_list`` is rendered by the caller, which owns the record types; this module stays free of them.
    ``implausible_origin`` is the profile's :attr:`PromptSlots.implausible_origin`."""
    return (
        f"Field to extract:\n{render_field_table((spec,), implausible_origin)}\n\n"
        f"Samples this paper reports:\n{sample_list}\n\n"
        f"Excerpts (Markdown with provenance markers):\n\n{markdown}\n\n"
        "Return the JSON object now."
    )


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


_MATCHING_SYSTEM = """Two independent PDF parsers were run on the SAME paper and a fact extractor produced a list of {sample_plural} from each.
Because the parsers differ (OCR errors, table splitting, missing sections), the two lists may name, order or split the samples differently.
Decide which sample in list A is the same physical sample as which sample in list B.

Output ONLY a JSON object:
{"pairs": [{"a": "<sample_id from A>", "b": "<sample_id from B>", "confidence": <0.0-1.0>, "justification": "<why>"}, ...],
 "unmatched_a": ["<sample_id>", ...], "unmatched_b": ["<sample_id>", ...]}

Rules:
1. Match on physical identity: same {condition_noun} {matching_condition_examples} or the same explicit sample name/number. Field VALUES {matching_value_examples} may be used as supporting evidence only.
2. Each sample id appears at most once across pairs and unmatched lists. Every id from both lists must appear exactly once.
3. `confidence` is your belief that the pair is the same sample; use < 0.6 when the evidence is weak.
4. `justification` names the condition or name that established the match, e.g. "{matching_justification_example}".

Return the JSON object only."""


def matching_system_prompt(profile: DomainProfile) -> str:
    return render(_MATCHING_SYSTEM, _values(profile))


def matching_user_prompt(lane_a_name: str, lane_a: str, lane_b_name: str, lane_b: str) -> str:
    return (
        f"List A (parser: {lane_a_name}):\n{lane_a}\n\nList B (parser: {lane_b_name}):\n{lane_b}\n\n"
        "Return the JSON object now."
    )
