"""Record every request a passage-mode ``run_document`` sends, as tests/fixtures/payloads/b0.json.

The prompt snapshot (``tests/fixtures/prompts/``) pins the templates over fixed inputs. What it cannot pin is
the user half as the pipeline builds it: the rendered Markdown of a real parse, the blocks retrieval picks
for each field, the sample list the inventory produced, and the matching listing built from the lanes'
cleaned samples -- which depends on extract, records, normalize and voting together. The LLM cache is keyed by
the whole request, so this file records every request's ``(system, user)`` and the ``cache_key`` the real
client computes for it, over the two recorded real parses (``mineru_real_sample``, ``paddle_real_sample``)
with the repository's ``config.json``: the inventory, every field question, the matching question, and one
repair each for a field question and for matching. The fixtures contain no figure, so no vision request is
recorded.

It was recorded at B0 (the merged fix branch, before the generalisation work). ``tests/test_payload_pins.py``
compares a fresh run against it and pins the file's digest. **Never re-record it after B0 without a reason
stated in the commit**: a re-record accepts every request that changed, which is exactly what it exists to
catch. When a change is intended:

    PYTHONPATH=src uv run python tests/fixtures/payloads/generate_b0.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

TESTS_DIR = Path(__file__).resolve().parents[2]
if str(TESTS_DIR) not in sys.path:  # run as a script; under pytest ``support`` is importable already
    sys.path.insert(0, str(TESTS_DIR))

from paperfacts import workflow  # noqa: E402
from paperfacts.config import Settings  # noqa: E402
from paperfacts.models import BACKENDS, DocumentInput, RawParseOutput  # noqa: E402
from paperfacts.storage import DataLayout  # noqa: E402
from support.factories import make_blank_pdf  # noqa: E402
from support.llm import FakeLlmClient  # noqa: E402

RECORDING = Path(__file__).with_name("b0.json")
FIXTURES = {
    "mineru": TESTS_DIR / "fixtures" / "mineru_real_sample",
    "paddleocr_vl": TESTS_DIR / "fixtures" / "paddle_real_sample",
}

REPAIR_MARKER = "Your previous answer was not valid JSON"
_SOURCE = re.compile(r"<!-- source: (\S+) -->")
_LISTED_ID = re.compile(r"^- id: (.+?) \| label:", re.MULTILINE)
_FIELD = re.compile(r"\AField to extract:\n- `([^`]+)`")

# The canned answers. Each lane names its samples differently, so matching has to ask the model; the values
# are quoted from the paper so they survive cleaning and reach the matching listing.
INVENTORY = {
    "mineru": [
        ("SnO2:Ta optimized", "optimized O2 flow", {"deposition temperature": "225 °C"}),
        ("SnOx:Ta O2 series", "", {}),
    ],
    "paddleocr_vl": [
        ("SnO2:Ta (0.70 at.% Ta)", "optimized flow conditions", {"temperature": "225°C"}),
        ("O2 flow series", "", {}),
    ],
}
FIELD_VALUES = {
    "resistivity": ("0.3", "Ω cm"),
    "substrate_temperature": ("225", "°C"),
    "working_pressure": ("1.1", "Pa"),
    "transmittance": (">80", "%"),
    "component": ("Sn/Ta target 95:5 wt.%", None),
}
# Its first answer fails validation, so the repair request is recorded too.
REPAIRED_FIELD = "thickness"


def _lane(user: str) -> str:
    return "paddleocr_vl" if "<!-- source: paddleocr_vl_" in user else "mineru"


def respond(system: str, user: str) -> str:
    """The fake model: a deterministic answer keyed on the question, whatever order the pool asks in."""
    repair = REPAIR_MARKER in user
    if user.startswith("Paper excerpts"):
        lane = _lane(user)
        cited = _SOURCE.findall(user)[:1]
        samples = [
            {"sample_id": sid, "label": label, "conditions": conditions, "source_ids": cited}
            for sid, label, conditions in INVENTORY[lane]
        ]
        return json.dumps({"samples": samples, "no_tco_film": False})
    if user.startswith("List A (parser:"):
        if not repair:
            return "pairs: none"
        list_a, list_b = user.split("\n\nList B (parser:", 1)
        ids_a, ids_b = _LISTED_ID.findall(list_a), _LISTED_ID.findall(list_b)
        paired = zip(ids_a, ids_b, strict=False)  # the lists may differ in length; the rest stay unmatched
        pairs = [{"a": a, "b": b, "confidence": 0.9, "justification": "same position"} for a, b in paired]
        return json.dumps({"pairs": pairs, "unmatched_a": ids_a[len(pairs) :], "unmatched_b": ids_b[len(pairs) :]})
    match = _FIELD.match(user)
    if match is None:
        raise AssertionError(f"an unexpected request: {user[:200]!r}")
    field = match.group(1)
    if field == REPAIRED_FIELD and not repair:
        return json.dumps({"values": [{"sample_id": None, "value_raw": ""}]})
    excerpts = user.split("Excerpts (Markdown with provenance markers):", 1)[1]
    cited = _SOURCE.findall(excerpts)[:1]
    if field not in FIELD_VALUES or not cited:
        return json.dumps({"values": []})
    listed = _LISTED_ID.findall(user.split("Excerpts (Markdown", 1)[0])
    value_raw, unit_raw = FIELD_VALUES[field]
    value = {"sample_id": listed[0] if listed and field != "component" else None, "value_raw": value_raw}
    value |= {"unit_raw": unit_raw, "source_ids": cited}
    return json.dumps({"values": [value]})


def _label(user: str) -> str:
    """Which question a request is, stable across runs: the pool's order is not."""
    suffix = ":repair" if REPAIR_MARKER in user else ""
    if user.startswith("Paper excerpts"):
        return f"{_lane(user)}:inventory{suffix}"
    if user.startswith("List A (parser:"):
        return f"matching{suffix}"
    match = _FIELD.match(user)
    return f"{_lane(user)}:field:{match.group(1) if match else '?'}{suffix}"


def _kind(label: str) -> str:
    return "matching" if label.startswith("matching") else label.split(":")[1]


def record(work_dir: Path) -> dict[str, Any]:
    """Run the whole pipeline once over the two recorded parses and return every request it sent."""
    # The repository's config.json, as production reads it: the model, sampling and retrieval settings are
    # all part of the payload. Only the mode is forced, since passage mode is what this file pins.
    environ = {
        "PAPERFACTS_DATA_ROOT": str(work_dir / "data"),
        "PAPERFACTS_LLM_API_KEY": "sk-pin",
        "PAPERFACTS_EXTRACTION_MODE": "passage",
        "PAPERFACTS_FIGURES_ENABLED": "false",
    }
    settings = Settings.from_env(environ)
    pages = RawParseOutput.load(FIXTURES["mineru"], "mineru").meta.pages
    pdf = make_blank_pdf(work_dir / "sample.pdf", [(page.width_pt, page.height_pt) for page in pages])
    document = DocumentInput.from_path(pdf)
    layout = DataLayout(settings.data_root)
    for backend in BACKENDS:
        # A raw output already in place is a parser cache hit: the real parse path runs, no runner does.
        shutil.copytree(FIXTURES[backend], layout.raw_dir(document.document_id, backend))

    # The client the run would build, used only for what it would send and the key it would cache it under.
    real = workflow.build_llm_client(settings)
    real.close()
    fake = FakeLlmClient(
        respond,
        model=real.model,
        temperature=real.temperature,
        max_tokens=real.max_tokens,
        reasoning_effort=real.reasoning_effort,
    )
    with mock.patch.object(workflow, "build_llm_client", lambda _settings: fake):
        workflow.run_document(document, settings, workflow.load_run_profile(settings))

    systems: dict[str, str] = {}
    requests = []
    for call in fake.calls:
        label = _label(call.user)
        kind = _kind(label)
        if systems.setdefault(kind, call.system) != call.system:
            raise AssertionError(f"two different {kind} system prompts in one run")
        payload = real.payload(system=call.system, user=call.user, reasoning_effort=call.reasoning_effort)
        cache_key = real.cache_key(payload, cache_salt=call.cache_salt)
        requests.append({"label": label, "system": kind, "user": call.user, "cache_key": cache_key})
    requests.sort(key=lambda request: (request["label"], request["cache_key"]))
    return {"systems": dict(sorted(systems.items())), "requests": requests}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        recorded = record(Path(tmp))
    RECORDING.write_text(json.dumps(recorded, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {RECORDING} ({len(recorded['requests'])} requests)")
