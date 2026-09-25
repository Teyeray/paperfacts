"""Write ``profiles/tco.json`` from what the code states about TCO today.

    PYTHONPATH=src uv run python scripts/extract_tco_profile.py

A one-shot migration tool, run on the checkout whose prompts are still TCO literals:

- ``fields`` and ``condition_keywords`` are copied from ``config.json`` as they are;
- every prompt slot is sliced out of the prompt constants in ``prompts.py`` and ``figures.py`` between two
  anchors that must occur exactly once, so a prompt that no longer reads as expected stops the script instead
  of writing a wrong slot;
- the new field attributes restate the special cases the code makes by field name today (the condition rule
  and its dataset note on transmittance, the chart-readable fields, the fields printed in scientific notation),
  each checked against the source it replaces.

``tests/test_profile_load.py`` checks the result against the running field table.
"""

from __future__ import annotations

import json
import re

from paperfacts import figures, prompts
from paperfacts.config import DEFAULT_REPO_ROOT, config_path
from paperfacts.passages import CONDITION_UNIT

OUTPUT = DEFAULT_REPO_ROOT / "profiles" / "tco.json"
SOURCE = DEFAULT_REPO_ROOT / "src" / "paperfacts"
# The dataset note decide.py writes when a transmittance arrives without its wavelength.
MISSING_CONDITION_NOTE = "原文提取结果未注明透光率波长或波段"
# The fields workbook.py prints in scientific notation, and the test it does so with.
SCIENTIFIC = ("resistance", "resistivity")
SCIENTIFIC_TEST = 'key in {"resistance", "resistivity"}'


def between(text: str, before: str, after: str) -> str:
    """The one run of text on a single line between ``before`` and ``after``."""
    found = re.findall(re.escape(before) + "(.*?)" + re.escape(after), text)
    if len(found) != 1:
        raise SystemExit(f"expected exactly one {before!r} ... {after!r}, found {len(found)}")
    return found[0]


def prompt_slots() -> dict[str, str]:
    extraction = prompts._EXTRACTION_SYSTEM
    inventory = prompts._INVENTORY_SYSTEM
    field = prompts._FIELD_SYSTEM
    matching = prompts._MATCHING_SYSTEM
    return {
        "domain_subject": between(extraction, "ONE scientific paper about ", ".\n"),
        "sample_definition": prompts._SAMPLE_SCOPE,
        "field_scope": prompts._LAYER_SCOPE,
        "fact_noun": between(extraction, "You extract ", " from ONE"),
        "sample_plural": between(inventory, "list the ", " the paper reports"),
        "sample_singular": between(inventory, "if the paper studies a single ", ", return"),
        "sample_unit": between(extraction, "one sample per distinct ", " that the paper reports"),
        "sample_examples": between(extraction, "that the paper reports results for ", ". Use the paper's"),
        "sample_id_example": between(extraction, 'distinguishing condition (e.g. "', '")'),
        "condition_noun": between(extraction, "`conditions` holds the ", " that distinguish samples"),
        "condition_examples": between(extraction, "that distinguish samples ", ", values as written"),
        "unit_examples": between(extraction, "exactly as written (e.g. ", "). If the number"),
        "paper_key": between(extraction, '{\n  "', '": {"source_ids"'),
        "paper_level_rule": between(extraction, "\n5. ", "\n"),
        "no_samples_key": between(inventory, '],\n  "', '": <true|false>'),
        "no_samples_clause": between(extraction, " If ", ', return an empty "samples" list.'),
        "no_samples_condition": between(inventory, "the excerpts show that ", "; then"),
        "samples_present_condition": between(inventory, "It is false whenever ", ", and false when"),
        "subset_examples": between(prompts._SUBSET_SCOPE, "listed samples -- ", " -- belongs"),
        "whole_series_examples": between(extraction, "for the whole series -- ", " -- and never"),
        "partial_collective_example": between(field, "most but not all of the listed samples (", ") is false"),
        "multi_condition_example": between(extraction, "under different conditions (e.g. ", "), report"),
        "matching_condition_examples": between(matching, "same deposition conditions ", " or the same explicit"),
        "matching_value_examples": between(matching, "Field VALUES ", " may be used"),
        "matching_justification_example": between(matching, 'e.g. "', '".'),
    }


def figure_slots() -> dict[str, str]:
    prompt = figures.USER_PROMPT
    return {
        "subject": between(prompt, "a scientific paper about ", ".\n"),
        "property_noun": between(prompt, "Only these ", " are of interest"),
        "chart_definition": between(prompt, "chart: ", ", with one discrete marker per sample."),
        "axis_example": between(prompt, "exactly as the title writes it (e.g. ", "). When the multiplier"),
    }


def main() -> None:
    config = json.loads(config_path({}).read_text(encoding="utf-8"))
    rule_field = between(prompts._EXTRACTION_SYSTEM, "\n8. For `", "` always fill")
    condition_rule = between(prompts._EXTRACTION_SYSTEM, f"8. For `{rule_field}` always fill `condition` with ", ".\n")
    if f'spec.name == "{rule_field}"' not in (SOURCE / "decide.py").read_text(encoding="utf-8"):
        raise SystemExit(f"decide.py no longer names {rule_field!r}")
    if MISSING_CONDITION_NOTE not in (SOURCE / "decide.py").read_text(encoding="utf-8"):
        raise SystemExit("decide.py no longer writes the missing-condition note")
    if SCIENTIFIC_TEST not in (SOURCE / "workbook.py").read_text(encoding="utf-8"):
        raise SystemExit("workbook.py no longer prints these fields in scientific notation")
    readable = {spec.name for spec in figures.figure_fields()}

    fields = []
    for entry in config["fields"]:
        entry = dict(entry)
        if entry["name"] == rule_field:
            entry["condition_rule"] = condition_rule
            entry["missing_condition_note_zh"] = MISSING_CONDITION_NOTE
        if entry["name"] in readable:
            entry["figure_readable"] = True
        if entry["name"] in SCIENTIFIC:
            entry["display_format"] = "scientific"
        fields.append(entry)

    profile = {
        "$comment": (
            "The TCO profile: sputtered transparent conductive oxide films. Written by "
            "scripts/extract_tco_profile.py from config.json and the prompts as they stood; every slot is a "
            "verbatim slice of today's prompts, so editing one changes what the model is asked."
        ),
        "format": 1,
        "name": "tco",
        "title_zh": "透明导电氧化物（TCO）薄膜",
        "maturity": "production",
        "description_zh": "磁控溅射透明导电氧化物薄膜：靶材、沉积工艺与薄膜的电学、光学性能。",
        "groups": [
            {"name": "target", "level": "paper", "label_zh": "靶材"},
            {"name": "process", "level": "sample", "label_zh": "工艺"},
            {"name": "film", "level": "sample", "label_zh": "薄膜"},
        ],
        "prompt": prompt_slots(),
        "figures": figure_slots(),
        "retrieval": {
            "condition_keywords": config["condition_keywords"],
            "condition_unit_pattern": CONDITION_UNIT.pattern,
        },
        "units": {},
        "ui": {
            "paper_level_label_zh": "靶材（论文级）",
            "paper_level_short_zh": "靶材",
            "entity_label_zh": "样品",
            "no_samples_message_zh": "该论文没有自己沉积的 TCO 膜，所以没有样品级数据。",
        },
        "fields": fields,
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
