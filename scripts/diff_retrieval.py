"""What a code or profile change does to the blocks each question is shown, measured without a model.

Retrieval, the bibliography cut and article-type detection are pure functions of the stored parses, so their
effect on a whole library can be measured for free before any paid re-run. ``snapshot`` records, for every
stored artifact, the blocks the model would be shown -- the informative blocks, the inventory selection and
what the budget trims from it, each field's candidate blocks -- and the detected article type. ``compare``
diffs two snapshots and names the lanes whose questions change, which is the paid set of a re-run.

    uv run python scripts/diff_retrieval.py snapshot --data-root data --out head.json
    # the same code before the change, from a worktree of the base commit, against the same data:
    (cd ../base && PYTHONPATH=src uv run --project ../paperfacts-fix python \\
        ../paperfacts-fix/scripts/diff_retrieval.py snapshot --data-root ../paperfacts-fix/data --out base.json)
    uv run python scripts/diff_retrieval.py compare base.json head.json

Candidate blocks use the ``sample_blocks`` the stored lane's inventory cited. Where the inventory selection
itself changes, a re-run asks a new inventory whose citations may differ, so for those lanes the field
diffs are a lower bound; ``compare`` lists them apart.

Run it from the checkout whose code it measures: the profile and ``config.json`` are that checkout's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from paperfacts import extract, passages
from paperfacts.config import Settings
from paperfacts.keys import ExtractionOptions
from paperfacts.models import BACKENDS, ParsedArtifact
from paperfacts.profile_loader import load_profile, profile_path


def _sample_blocks(docdir: Path, backend: str, lane_key: str | None) -> tuple[frozenset[str], str | None]:
    """The block ids the stored inventory cited for the samples, and the lane file they came from."""
    lanes = sorted((docdir / "facts").glob(f"{backend}.*.json"), key=lambda path: path.stat().st_mtime)
    if lane_key is not None:
        lanes = [path for path in lanes if path.name == f"{backend}.{lane_key}.json"]
    if not lanes:
        return frozenset(), None
    lane = json.loads(lanes[-1].read_text(encoding="utf-8"))
    cited = frozenset(source_id for sample in lane.get("samples", []) for source_id in sample.get("source_ids", []))
    return cited, lanes[-1].name


def snapshot(data_root: Path, lane_key: str | None) -> dict[str, Any]:
    settings = Settings.from_env()
    profile = load_profile(profile_path(settings))
    options = ExtractionOptions.from_settings(settings, profile)
    budget = extract._budget_chars(options)
    detect = getattr(extract, "detect_article_type", None)
    documents: dict[str, Any] = {}
    for docdir in sorted(path for path in (data_root / "docs").iterdir() if path.is_dir()):
        lanes: dict[str, Any] = {}
        for backend in BACKENDS:
            path = docdir / "parsed" / f"{backend}.artifact.json"
            if not path.is_file():
                continue
            artifact = ParsedArtifact.read(path)
            blocks = extract.informative_blocks(artifact.blocks)
            chosen = passages.inventory_blocks(blocks, profile.retrieval)
            kept = passages.fit_budget(chosen, budget_chars=budget)
            cited, lane_file = _sample_blocks(docdir, backend, lane_key)
            fields = {
                spec.name: [
                    block.source_id
                    for block in passages.candidate_blocks(
                        spec, blocks, units=profile.units, limit=options.candidate_limit, sample_blocks=cited
                    )
                ]
                for spec in profile.fields
            }
            lanes[backend] = {
                "informative": [block.source_id for block in blocks],
                "inventory": [block.source_id for block in chosen],
                "inventory_budget_dropped": sorted({b.source_id for b in chosen} - {b.source_id for b in kept}),
                "article_type": None if detect is None else detect(artifact.blocks),
                "lane_file": lane_file,
                "fields": fields,
            }
        documents[docdir.name] = lanes
    return {"profile": profile.name, "candidate_limit": options.candidate_limit, "documents": documents}


def compare(old: dict[str, Any], new: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Lines to print, and the same as data: per lane what moved, and the paid set."""
    lines: list[str] = []
    changed: dict[str, Any] = {}
    for doc in sorted(set(old["documents"]) | set(new["documents"])):
        for backend in BACKENDS:
            a = old["documents"].get(doc, {}).get(backend)
            b = new["documents"].get(doc, {}).get(backend)
            if a is None or b is None:
                continue
            entry: dict[str, Any] = {}
            for part in ("informative", "inventory"):
                if a[part] != b[part]:
                    entry[part] = {
                        "added": sorted(set(b[part]) - set(a[part])),
                        "removed": sorted(set(a[part]) - set(b[part])),
                    }
            if a.get("article_type") != b.get("article_type"):
                entry["article_type"] = [a.get("article_type"), b.get("article_type")]
            fields = {
                name: {
                    "added": sorted(set(b["fields"][name]) - set(ids)),
                    "removed": sorted(set(ids) - set(b["fields"][name])),
                }
                for name, ids in a["fields"].items()
                if name in b["fields"] and b["fields"][name] != ids
            }
            if fields:
                entry["fields"] = fields
            if entry:
                changed[f"{doc}/{backend}"] = entry
    inventory = [lane for lane, entry in changed.items() if "inventory" in entry]
    informative = [lane for lane, entry in changed.items() if "informative" in entry]
    field_only = {
        lane: sorted(entry["fields"]) for lane, entry in changed.items() if "fields" in entry and lane not in inventory
    }
    types = {lane: entry["article_type"] for lane, entry in changed.items() if "article_type" in entry}
    lines.append(f"lanes whose informative blocks changed: {len(informative)}")
    lines += [
        f"  {lane}: +{len(changed[lane]['informative']['added'])} -{len(changed[lane]['informative']['removed'])}"
        for lane in informative
    ]
    lines.append(f"lanes whose inventory changed (all questions re-asked; field diffs a lower bound): {len(inventory)}")
    lines += [f"  {lane}" for lane in inventory]
    lines.append(f"lanes where only field questions changed: {len(field_only)}")
    lines += [f"  {lane}: {', '.join(names)}" for lane, names in field_only.items()]
    lines.append(f"article type changes: {len(types)}")
    lines += [f"  {lane}: {was} -> {now}" for lane, (was, now) in types.items()]
    return lines, {"changed": changed, "inventory_lanes": inventory, "field_only": field_only, "article_types": types}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--data-root", type=Path, required=True)
    snap.add_argument("--out", type=Path, required=True)
    snap.add_argument(
        "--lane-key", help="extractor key of the lanes whose inventory citations to use (default: newest)"
    )
    comp = sub.add_parser("compare")
    comp.add_argument("old", type=Path)
    comp.add_argument("new", type=Path)
    comp.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        args.out.write_text(
            json.dumps(snapshot(args.data_root, args.lane_key), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return 0
    old, new = (json.loads(path.read_text(encoding="utf-8")) for path in (args.old, args.new))
    lines, data = compare(old, new)
    print("\n".join(lines))
    if args.json:
        args.json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
