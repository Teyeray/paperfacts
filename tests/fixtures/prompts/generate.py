"""Record every prompt the pipeline can send, rendered from fixed inputs, as tests/fixtures/prompts/snapshot.json.

The generalisation work must leave the TCO profile's requests byte-identical: the LLM cache is keyed by the
request payload, so one changed byte turns a free replay of the corpus into a paid re-extraction, and the gold
set's cells can move. ``tests/test_prompt_snapshot.py`` compares the live prompts against this file.

Run from the repository root only when a prompt change is intended:

    PYTHONPATH=src uv run python tests/fixtures/prompts/generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from paperfacts import extract, figures, matching, prompts
from paperfacts.config import Settings
from paperfacts.profile_loader import load_profile, profile_path
from paperfacts.records import (
    ExtractionResponse,
    FieldResponse,
    FieldValue,
    InventoryResponse,
    InventorySample,
    SampleRecord,
)

SNAPSHOT = Path(__file__).with_name("snapshot.json")
# The shipped TCO profile, from the built-in settings so the environment cannot change what is rendered.
PROFILE = load_profile(profile_path(Settings()))

# Fixed inputs: the prompts' own text is what is pinned, so the inputs only have to be stable.
SAMPLE_LIST = "- id: S1 | label: ITO at 100 W | conditions: power=100 W\n- id: S2 | label: - | conditions: -"
MARKDOWN = "<!-- source: mineru_p0_b0 -->\nThe film was 120 nm thick.\n\n<!-- source: mineru_p0_b1 -->\nTable 1"
CAPTION = "Fig. 2 (a) Sheet resistance and (b) transmittance of films sputtered at 50-200 W."

# A rejected answer goes back to the model inside the repair prompt as ``str(ValidationError)``, so pydantic's
# wording and the response models' class names, field names and constraints are request bytes too.
INVALID_ANSWERS: dict[str, tuple[type[BaseModel], str]] = {
    "missing_key": (ExtractionResponse, '{"samples": [{"sample_id": "S1", "fields": [{"value_raw": "12"}]}]}'),
    "target_wrong_type": (ExtractionResponse, '{"target": {"fields": "12 nm"}}'),
    "empty_value_raw": (FieldResponse, '{"values": [{"sample_id": "S1", "value_raw": ""}]}'),
    "samples_not_list": (InventoryResponse, '{"samples": {"sample_id": "S1"}}'),
    "not_json": (FieldResponse, '{"values": [{"value_raw": "12"'),
    "no_tco_film_type": (InventoryResponse, '{"samples": [], "no_tco_film": "maybe"}'),
}

# What the field questions and the matching question are shown of the samples: rendered by extract.py and
# matching.py rather than prompts.py, so the templates above do not cover them.
INVENTORY = (
    InventorySample(sample_id="S1", label="ITO at 100 W", conditions={"power": "100 W", "O2": "2 sccm"}),
    InventorySample(sample_id="S2"),
)
MATCHING_SAMPLES = [
    SampleRecord(
        sample_id="ITO-100W",
        label="ITO at 100 W",
        conditions={"power": "100 W"},
        fields=(
            FieldValue(field="sheet_resistance", value_raw="12.5", unit_raw="Ω/sq"),
            FieldValue(field="transmittance", value_raw="> 80", unit_raw="%", condition="550 nm"),
        ),
    ),
    SampleRecord(sample_id="ITO-200W"),
]


def validation_error(model: type[BaseModel], answer: str) -> str:
    try:
        model.model_validate_json(answer)
    except ValidationError as exc:
        return str(exc)
    raise AssertionError(f"{answer!r} validates against {model.__name__}; it was meant not to")


def snapshot() -> dict[str, str]:
    rendered = {
        "extraction_system": prompts.extraction_system_prompt(PROFILE),
        "extraction_user": prompts.extraction_user_prompt(MARKDOWN),
        "inventory_system": prompts.inventory_system_prompt(PROFILE),
        "inventory_user": prompts.inventory_user_prompt(MARKDOWN),
        "field_system": prompts.field_system_prompt(PROFILE),
        "matching_system": prompts.matching_system_prompt(PROFILE),
        "matching_user": prompts.matching_user_prompt("mineru", "- S1", "paddleocr_vl", "- S1"),
        "repair": prompts.repair_prompt("original question", "{bad json", "Expecting value"),
        "field_table": prompts.render_field_table(PROFILE.fields, PROFILE.prompt.implausible_origin),
        "figures_system": figures.SYSTEM_PROMPT,
        "figures_user": figures.user_prompt(CAPTION, PROFILE.figure_fields, PROFILE.figures),
    }
    for spec in PROFILE.fields:
        rendered[f"field_user:{spec.name}"] = prompts.field_user_prompt(
            spec, SAMPLE_LIST, MARKDOWN, PROFILE.prompt.implausible_origin
        )
    for name, (model, answer) in INVALID_ANSWERS.items():
        rendered[f"validation_error:{name}"] = validation_error(model, answer)
    rendered["sample_list"] = extract._render_sample_list(INVENTORY)
    rendered["sample_list:empty"] = extract._render_sample_list(())
    rendered["matching_render"] = matching._render(MATCHING_SAMPLES)
    return rendered


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(snapshot(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT} ({len(snapshot())} prompts)")
