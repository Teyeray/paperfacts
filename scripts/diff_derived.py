"""Compare a library's derived files under two cache keys: the proof that a re-key changed nothing else.

A refactor that moves the cache keys leaves both generations on disk after a re-derivation: every document's
facts, comparison report and dataset under the old keys and under the new ones. This script compares them
file by file, for the whole library rather than the handful of papers the gold set covers, and exits non-zero
when anything differs beyond the fields a re-key is expected to change (the keys themselves, fingerprints,
usage counters).

    python scripts/diff_derived.py --data-root data \\
        --old <extractor_key>.<comparison_key> --new <extractor_key>.<comparison_key> [--json report.json]

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


def compare_file(document: str, kind: str, old: Path, new: Path, ignored: frozenset[str]) -> FileDiff:
    if not old.is_file() and not new.is_file():
        return FileDiff(document, kind, "missing_both")
    if not old.is_file():
        return FileDiff(document, kind, "missing_old")
    if not new.is_file():
        return FileDiff(document, kind, "missing_new")
    paths = list(differences(json.loads(old.read_text()), json.loads(new.read_text()), ignored))
    return FileDiff(document, kind, "differs" if paths else "identical", paths[:MAX_PATHS])


def compare_library(
    data_root: Path, old: tuple[str, str], new: tuple[str, str], ignored: frozenset[str]
) -> list[FileDiff]:
    results: list[FileDiff] = []
    for document in sorted(path for path in (data_root / "docs").iterdir() if path.is_dir()):
        name = document.name
        for backend in BACKENDS:
            results.append(
                compare_file(
                    name,
                    f"facts:{backend}",
                    document / "facts" / f"{backend}.{old[0]}.json",
                    document / "facts" / f"{backend}.{new[0]}.json",
                    ignored,
                )
            )
        for kind in ("comparisons", "datasets"):
            results.append(
                compare_file(
                    name,
                    kind,
                    document / kind / f"{old[0]}.{old[1]}.json",
                    document / kind / f"{new[0]}.{new[1]}.json",
                    ignored,
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
    args = parser.parse_args(argv)

    ignored = DEFAULT_IGNORED | frozenset(args.ignore)
    results = compare_library(args.data_root, keys_of(args.old), keys_of(args.new), ignored)
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
