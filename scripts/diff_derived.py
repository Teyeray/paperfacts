"""Compare a library's derived files under two cache keys: the proof that a re-key changed nothing else.

A refactor that moves the cache keys leaves both generations on disk after a re-derivation: every document's
facts, comparison report and dataset under the old keys and under the new ones. This script compares them
file by file, for the whole library rather than the handful of papers the gold set covers, and exits non-zero
when anything differs beyond the fields a re-key is expected to change (the keys themselves, fingerprints,
usage counters).

    python scripts/diff_derived.py --data-root data \\
        --old <extractor_key>.<comparison_key> --new <extractor_key>.<comparison_key> [--json report.json]

Round 2 renames the paper-level record ``target`` -> ``paper`` and the no-samples verdict ``no_tco_film`` ->
``no_samples`` in the stored files. Across that rename the old side is read under the new names first
(``--legacy-names``), so the proof still compares content rather than spelling. By default this happens for a
document exactly when its old lanes use the old names and its new lanes do not, so B1 against B1 is untouched.

Standard library only, so it runs from any checkout against any data root.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKENDS = ("mineru", "paddleocr_vl")
# Top-level fields a re-key changes by definition, or that record cost rather than content.
DEFAULT_IGNORED = frozenset(
    {"extractor_key", "comparison_key", "schema_version", "profile_fingerprint", "usage", "raw_response"}
)
MAX_PATHS = 20
# The round 2 rename, as it shows in each stored file: a lane's top-level keys, a report's paper-level scope, a
# dataset's paper-level quality row id. Nothing else was renamed.
LEGACY_LANE_KEYS = {"target": "paper", "no_tco_film": "no_samples"}
LEGACY_PAPER_ID = "target"
PAPER_ID = "paper"
RENAMED_ROW_IDS = {"comparisons": ("comparisons", "scope"), "datasets": ("quality_rows", "sample_id")}


@dataclass
class FileDiff:
    document: str
    kind: str
    status: str  # identical | differs | missing_old | missing_new | missing_both
    paths: list[str] = field(default_factory=list)


def keys_of(spec: str) -> tuple[str, str]:
    extractor, _, comparison = spec.partition(".")
    if not extractor or not comparison:
        raise SystemExit(f"a key pair is <extractor_key>.<comparison_key>, got {spec!r}")
    return extractor, comparison


def differences(old: Any, new: Any, ignored: frozenset[str], path: str = "") -> Iterator[str]:
    """JSON paths where ``old`` and ``new`` differ. Ignored names are skipped at every depth."""
    if isinstance(old, dict) and isinstance(new, dict):
        for name in sorted(set(old) | set(new)):
            if name in ignored:
                continue
            here = f"{path}.{name}" if path else name
            if name not in old or name not in new:
                yield here
            else:
                yield from differences(old[name], new[name], ignored, here)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            yield f"{path}[len {len(old)}→{len(new)}]"
        for index, (a, b) in enumerate(zip(old, new, strict=False)):
            yield from differences(a, b, ignored, f"{path}[{index}]")
    elif old != new:
        yield path or "<root>"


def current_names(kind: str, data: Any) -> Any:
    """A stored file of ``kind`` (``facts:<backend>``, ``comparisons`` or ``datasets``) under the names after the
    round 2 rename. Only the renamed spots change; a value that merely equals ``"target"`` elsewhere stays."""
    if not isinstance(data, dict):
        return data
    if kind.startswith("facts:"):
        return {LEGACY_LANE_KEYS.get(name, name): value for name, value in data.items()}
    rows, id_key = RENAMED_ROW_IDS[kind]
    if not isinstance(data.get(rows), list):
        return data
    renamed = [
        row | {id_key: PAPER_ID} if isinstance(row, dict) and row.get(id_key) == LEGACY_PAPER_ID else row
        for row in data[rows]
    ]
    return data | {rows: renamed}


def uses_legacy_names(old_lanes: list[Path], new_lanes: list[Path]) -> bool:
    """Whether a document straddles the rename: an old lane has the legacy keys and no new lane still has them."""

    def legacy(path: Path) -> bool:
        return path.is_file() and any(name in json.loads(path.read_text()) for name in LEGACY_LANE_KEYS)

    return any(legacy(path) for path in old_lanes) and not any(legacy(path) for path in new_lanes)


def compare_file(
    document: str, kind: str, old: Path, new: Path, ignored: frozenset[str], *, legacy_names: bool = False
) -> FileDiff:
    if not old.is_file() and not new.is_file():
        return FileDiff(document, kind, "missing_both")
    if not old.is_file():
        return FileDiff(document, kind, "missing_old")
    if not new.is_file():
        return FileDiff(document, kind, "missing_new")
    old_data = json.loads(old.read_text())
    if legacy_names:
        old_data = current_names(kind, old_data)
    paths = list(differences(old_data, json.loads(new.read_text()), ignored))
    return FileDiff(document, kind, "differs" if paths else "identical", paths[:MAX_PATHS])


def compare_library(
    data_root: Path,
    old: tuple[str, str],
    new: tuple[str, str],
    ignored: frozenset[str],
    *,
    legacy_names: bool | None = None,
) -> list[FileDiff]:
    """Every document's derived files, old key against new. ``legacy_names`` None decides per document
    (:func:`uses_legacy_names`); True or False reads every old file, or none, under the new names."""
    results: list[FileDiff] = []
    for document in sorted(path for path in (data_root / "docs").iterdir() if path.is_dir()):
        name = document.name
        old_lanes = [document / "facts" / f"{backend}.{old[0]}.json" for backend in BACKENDS]
        new_lanes = [document / "facts" / f"{backend}.{new[0]}.json" for backend in BACKENDS]
        rename = uses_legacy_names(old_lanes, new_lanes) if legacy_names is None else legacy_names
        for backend, old_lane, new_lane in zip(BACKENDS, old_lanes, new_lanes, strict=True):
            results.append(compare_file(name, f"facts:{backend}", old_lane, new_lane, ignored, legacy_names=rename))
        for kind in ("comparisons", "datasets"):
            results.append(
                compare_file(
                    name,
                    kind,
                    document / kind / f"{old[0]}.{old[1]}.json",
                    document / kind / f"{new[0]}.{new[1]}.json",
                    ignored,
                    legacy_names=rename,
                )
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--old", required=True, help="<extractor_key>.<comparison_key> of the old generation")
    parser.add_argument("--new", required=True, help="<extractor_key>.<comparison_key> of the new generation")
    parser.add_argument("--ignore", action="append", default=[], help="another field name to ignore (repeatable)")
    parser.add_argument("--json", type=Path, help="write every file's result here")
    parser.add_argument(
        "--legacy-names",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="read the old side under the round 2 names (target -> paper, no_tco_film -> no_samples); "
        "default: per document, when its old lanes use the old names and its new lanes do not",
    )
    args = parser.parse_args(argv)

    ignored = DEFAULT_IGNORED | frozenset(args.ignore)
    results = compare_library(
        args.data_root, keys_of(args.old), keys_of(args.new), ignored, legacy_names=args.legacy_names
    )
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
        if result.status in ("differs", "missing_old", "missing_new"):
            print(f"{result.document} {result.kind}: {result.status} {', '.join(result.paths)}")
    print("summary:", ", ".join(f"{status} {count}" for status, count in sorted(counts.items())))
    if args.json:
        args.json.write_text(json.dumps([result.__dict__ for result in results], indent=1), encoding="utf-8")
    changed = sum(counts.get(status, 0) for status in ("differs", "missing_old", "missing_new"))
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
