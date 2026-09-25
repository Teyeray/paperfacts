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

from paperfacts import figures, prompts
from paperfacts.fields import FIELD_SPECS

SNAPSHOT = Path(__file__).with_name("snapshot.json")

# Fixed inputs: the prompts' own text is what is pinned, so the inputs only have to be stable.
SAMPLE_LIST = "- id: S1 | label: ITO at 100 W | conditions: power=100 W\n- id: S2 | label: - | conditions: -"
MARKDOWN = "<!-- source: mineru_p0_b0 -->\nThe film was 120 nm thick.\n\n<!-- source: mineru_p0_b1 -->\nTable 1"
CAPTION = "Fig. 2 (a) Sheet resistance and (b) transmittance of films sputtered at 50-200 W."


def snapshot() -> dict[str, str]:
    rendered = {
        "extraction_system": prompts.extraction_system_prompt(),
        "extraction_user": prompts.extraction_user_prompt(MARKDOWN),
        "inventory_system": prompts.inventory_system_prompt(),
        "inventory_user": prompts.inventory_user_prompt(MARKDOWN),
        "field_system": prompts.field_system_prompt(),
        "matching_system": prompts.matching_system_prompt(),
        "matching_user": prompts.matching_user_prompt("mineru", "- S1", "paddleocr_vl", "- S1"),
        "repair": prompts.repair_prompt("original question", "{bad json", "Expecting value"),
        "field_table": prompts.render_field_table(),
        "figures_system": figures.SYSTEM_PROMPT,
        "figures_user": figures.user_prompt(CAPTION, tuple(spec for spec in FIELD_SPECS if spec.group == "film")),
    }
    for spec in FIELD_SPECS:
        rendered[f"field_user:{spec.name}"] = prompts.field_user_prompt(spec, SAMPLE_LIST, MARKDOWN)
    return rendered


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(snapshot(), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT} ({len(snapshot())} prompts)")
