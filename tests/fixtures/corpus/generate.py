"""Record the corpus's value strings and sample ids, with how a reference commit and this checkout read them.

    PYTHONPATH=src uv run python tests/fixtures/corpus/generate.py <data_root> [--reference origin/main]

``<data_root>`` is a copy of a server's ``data/`` (one directory per document, each with ``facts/*.json``).
The output is two JSON files next to this script, read by ``tests/test_corpus_strings.py``:

- ``values.json``: every distinct (numeric field, ``value_raw``) with ``reference`` -- what
  ``normalize.parse_number`` returned at the reference commit -- and ``expected``, what it returns now.
- ``sample_ids.json``: every lane pair's sample ids with the reference's ``normalize_key`` and today's
  ``records.sample_key``.

Re-run it after an intended change to either function and review the diff of the JSON: that diff is the
whole behavioural change on real data. The test then pins it, so an unintended one fails without a corpus.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import types
from pathlib import Path

from paperfacts.config import Settings
from paperfacts.normalize import parse_number
from paperfacts.profile_loader import load_profile, profile_path
from paperfacts.records import sample_key

HERE = Path(__file__).parent


def reference_module(ref: str) -> types.ModuleType:
    """The reference commit's ``normalize.py``, imported under another name beside this checkout's."""
    source = subprocess.run(
        ["git", "show", f"{ref}:src/paperfacts/normalize.py"], check=True, capture_output=True, text=True
    ).stdout
    module = types.ModuleType("normalize_reference")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{ref}:normalize.py", "exec"), module.__dict__)
    return module


def lane_values(lane: dict) -> list[dict]:
    target = lane.get("target") or {}
    values = list(target.get("fields", []))
    for sample in lane.get("samples", []):
        values.extend(sample.get("fields", []))
    return [*values, *lane.get("unattributed", [])]


def reading(result: tuple[float | None, str | None]) -> list:
    return [result[0], result[1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--reference", default="origin/main")
    args = parser.parse_args()
    reference = reference_module(args.reference)

    fields = load_profile(profile_path(Settings())).by_name
    lanes = sorted(args.data_root.glob("*/facts/*.json"))
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    by_pair: dict[tuple[str, str], dict[str, list[str]]] = collections.defaultdict(dict)
    for path in lanes:
        lane = json.loads(path.read_text(encoding="utf-8"))
        for value in lane_values(lane):
            spec = fields.get(value["field"])
            if spec is not None and spec.kind == "numeric":
                counts[(value["field"], value["value_raw"])] += 1
        backend, key, _ = path.name.split(".")
        by_pair[(path.parent.parent.name, key)][backend] = [sample["sample_id"] for sample in lane.get("samples", [])]

    values = [
        {
            "field": field,
            "value_raw": raw,
            "occurrences": count,
            "reference": reading(reference.parse_number(raw)),
            "expected": reading(parse_number(raw)),
        }
        for (field, raw), count in sorted(counts.items())
    ]
    # Documents are numbered rather than named: the test needs which ids share a lane pair, not which paper.
    documents = {name: index for index, name in enumerate(sorted({document for document, _ in by_pair}))}
    pairs = [
        {
            "document": documents[document],
            "lanes": {
                backend: [
                    {
                        "id": sample_id,
                        "reference": reference.normalize_key(sample_id),
                        "expected": sample_key(sample_id),
                    }
                    for sample_id in ids
                ]
                for backend, ids in sorted(lanes_.items())
            },
        }
        for (document, _), lanes_ in sorted(by_pair.items())
    ]
    write_rows(HERE / "values.json", values)
    write_rows(HERE / "sample_ids.json", pairs)
    ids = sum(len(entries) for pair in pairs for entries in pair["lanes"].values())
    print(f"{len(lanes)} lanes, {len(values)} distinct values, {len(pairs)} lane pairs, {ids} sample ids")


def write_rows(path: Path, rows: list[dict]) -> None:
    """One row per line, so a re-run's diff shows exactly the entries whose reading changed."""
    lines = ",\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(f"[\n{lines}\n]\n", encoding="utf-8")


if __name__ == "__main__":
    main()
