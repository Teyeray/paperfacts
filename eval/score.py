"""Score PaperFacts datasets against the hand-built gold set in eval/gold/.

Run it in the package's environment: a text field's closed categories are matched by the package's own rule
(``normalize.canonical_category``), so the score can never disagree with the pipeline about what "RF" is.

    uv run python eval/score.py --data-root data --keys <extractor_key>.<comparison_key>
    uv run python eval/score.py --data-root data --out report.md --json report.json   # newest datasets
    uv run python eval/score.py --dataset 80c3b69d570c2b6d=path/to/dataset.json

The field table is the profile's (``--profile``, default ``profiles/tco.json``), parsed and validated by the
package itself, so scoring reads the same fields, tolerances and categories as the run. The gold files keep their
own ids: paper-level cells are under ``"paper"`` and are reported with sample ``"paper"``, whatever the profile
calls its paper-level group. A gold file or a dataset written before round 2 spells that id ``"target"``; both are
still read, the gold file with a note on stderr.

``--keys`` names the dataset file (``datasets/<extractor_key>.<comparison_key>.json``), so the score is of the
run those keys describe. Without it the newest dataset file of each document that was built under this profile
(by the profile fingerprint the dataset records) is scored, which is only right while a library holds datasets
of a single set of keys per profile.

The rules (sample alignment, cell outcomes, what counts toward precision and recall) are documented in
eval/README.md; the code below implements exactly those rules and nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from paperfacts.fields import FieldSpec
from paperfacts.keys import profile_comparison_fingerprint
from paperfacts.normalize import canonical_category
from paperfacts.profile import DomainProfile
from paperfacts.profile_loader import parse_profile

REPO = Path(__file__).resolve().parent.parent
# The gold files' id for the paper-level record, and the sample id the report prints for it.
PAPER = "paper"
# The same id before round 2, in gold files and in datasets' quality rows.
LEGACY_PAPER = "target"
OUTCOMES = ("correct", "soft", "wrong", "missing", "extra", "disputed")


# The package's field description: kind, tolerances, categories and level are what scoring reads.
Spec = FieldSpec


@dataclass(frozen=True)
class Cell:
    doc: str
    sample: str  # gold sample id, "paper", or "(unaligned) <row id>"
    row: str  # dataset sample_id, "" when the dataset has no row for it
    field: str
    outcome: str
    value: object
    gold: str  # short rendering of the gold cells
    detail: str  # quality_rows decision + detail for the dataset cell


def load_scoring_profile(path: Path) -> DomainProfile:
    """The profile in ``path``, as the package parses it. Parsed under its own name rather than the file's: the
    stem rule keeps two profiles' workbooks apart, and scoring writes none, so a renamed copy scores the same."""
    data = json.loads(path.read_text(encoding="utf-8"))
    name = data.get("name") if isinstance(data, dict) else None
    return parse_profile(data, path.with_name(f"{name}.json") if isinstance(name, str) else path)


def load_specs(profile: Path) -> dict[str, Spec]:
    """The profile's field table as scoring needs it: the package's own ``FieldSpec`` by name."""
    return load_scoring_profile(profile).by_name


# ---------------------------------------------------------------------------------------------- value matching


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def value_matches(spec: Spec, got: object, cell: dict) -> bool:
    gold = cell.get("value")
    if gold is None or got is None:
        return False
    if spec.kind == "numeric":
        try:
            return math.isclose(float(got), float(gold), rel_tol=spec.rel_tol, abs_tol=spec.abs_tol)
        except (TypeError, ValueError):
            return False
    # A field with closed categories: 'RF magnetron sputtering' is 'RF'. Text that names no category is
    # compared as text below, exactly as the pipeline falls back.
    wanted = canonical_category(spec.categories, str(gold))
    if wanted is not None:
        return canonical_category(spec.categories, str(got)) == wanted
    if _norm(got) == _norm(gold):
        return True
    return any(re.search(p, str(got), re.I) for p in cell.get("accept", ()))


def is_definite(cell: dict, sample_ambiguous: bool) -> bool:
    return not (sample_ambiguous or cell.get("ambiguous") or cell.get("figure_only") or cell.get("value") is None)


def render(cells: list[dict]) -> str:
    parts = []
    for cell in cells:
        tag = "fig" if cell.get("figure_only") else "amb" if cell.get("ambiguous") or cell.get("value") is None else ""
        cond = f" @{cell['condition']}" if cell.get("condition") else ""
        parts.append(f"{cell.get('value')}{cond}{' [' + tag + ']' if tag else ''} (p{cell.get('page')})")
    return "; ".join(parts)


def classify(spec: Spec, got: object, cells: list[dict], sample_ambiguous: bool) -> str | None:
    """Outcome of one dataset cell against the gold cells for the same sample and field (None = nothing to say)."""
    definite = [c for c in cells if is_definite(c, sample_ambiguous)]
    if got is None:
        return "missing" if definite else None
    if not cells:
        return "extra"
    if any(value_matches(spec, got, c) for c in definite):
        return "correct"
    if any(value_matches(spec, got, c) for c in cells):
        return "soft"
    return "wrong" if definite else "disputed"


# ---------------------------------------------------------------------------------------------- alignment


def gold_cells(gold: dict, sample: dict, field: str) -> list[dict]:
    if field in sample.get("fields", {}):
        return sample["fields"][field]
    if field in sample.get("exclude_series", ()):
        return []
    return gold.get("series", {}).get(field, [])


def row_text(row: dict) -> str:
    return f"{row.get('sample_id', '')} | {row.get('sample_label', '')}"


def is_candidate(specs: dict[str, Spec], sample: dict, row: dict) -> bool:
    match = sample.get("match", {})
    for field, want in match.get("fields", {}).items():
        if not value_matches(specs[field], row.get(field), {"value": want}):
            return False
    text = row_text(row)
    if match.get("label") and not re.search(match["label"], text, re.I | re.S):
        return False
    if match.get("label_not") and re.search(match["label_not"], text, re.I | re.S):
        return False
    return bool(match)


def agreement(specs: dict[str, Spec], gold: dict, sample: dict, row: dict) -> int:
    return sum(
        1
        for name, spec in specs.items()
        if spec.level != "paper" and any(value_matches(spec, row.get(name), c) for c in gold_cells(gold, sample, name))
    )


def align(specs: dict[str, Spec], gold: dict, rows: list[dict]) -> dict[int, int]:
    """gold sample index -> row index. Candidates must pass the sample's `match`; pairs are then taken greedily by
    descending number of agreeing cells, ties broken by gold order, then row order. One-to-one."""
    pairs = [
        (-agreement(specs, gold, sample, row), gi, ri)
        for gi, sample in enumerate(gold["samples"])
        for ri, row in enumerate(rows)
        if is_candidate(specs, sample, row)
    ]
    taken_g, taken_r, result = set(), set(), {}
    for _, gi, ri in sorted(pairs):
        if gi not in taken_g and ri not in taken_r:
            result[gi] = ri
            taken_g.add(gi)
            taken_r.add(ri)
    return result


# ---------------------------------------------------------------------------------------------- scoring


def quality_index(dataset: dict) -> dict[tuple[str, str], dict]:
    return {(q.get("sample_id"), q.get("field")): q for q in dataset.get("quality_rows", [])}


def trace(quality: dict, sample_id: str, field: str) -> str:
    q = quality.get((sample_id, field))
    if not q and sample_id == PAPER:
        q = quality.get((LEGACY_PAPER, field))
    if not q:
        return ""
    bits = [q.get("decision") or "", q.get("conditions") or "", q.get("detail") or "", q.get("source_ids") or ""]
    return " | ".join(b for b in bits if b)


def score_document(specs: dict[str, Spec], gold: dict, dataset: dict) -> list[Cell]:
    doc = gold["doc_id"]
    quality = quality_index(dataset)
    rows = list(dataset.get("sample_rows", []))
    cells: list[Cell] = []

    paper_row = dataset.get("paper_row") or {}
    if PAPER not in gold and LEGACY_PAPER in gold:
        print(f"{doc}: gold file uses the legacy key {LEGACY_PAPER!r}; read as {PAPER!r}", file=sys.stderr)
    paper_gold = gold.get(PAPER, gold.get(LEGACY_PAPER, {}))
    for name, spec in specs.items():
        if spec.level != "paper":
            continue
        gcells = paper_gold.get(name, [])
        got = paper_row.get(name)
        outcome = classify(spec, got, gcells, False)
        if outcome:
            detail = trace(quality, PAPER, name)
            cells.append(Cell(doc, PAPER, PAPER, name, outcome, got, render(gcells), detail))

    sample_fields = [s for s in specs.values() if s.level != "paper"]
    mapping = align(specs, gold, rows)
    for gi, sample in enumerate(gold["samples"]):
        ri = mapping.get(gi)
        row = rows[ri] if ri is not None else {}
        rid = row.get("sample_id", "")
        amb = bool(sample.get("ambiguous"))
        for spec in sample_fields:
            gcells = gold_cells(gold, sample, spec.name)
            outcome = classify(spec, row.get(spec.name), gcells, amb)
            if outcome:
                detail = trace(quality, rid, spec.name) if rid else "no dataset row aligned to this gold sample"
                got = row.get(spec.name)
                cells.append(Cell(doc, sample["id"], rid, spec.name, outcome, got, render(gcells), detail))
    aligned_rows = set(mapping.values())
    for ri, row in enumerate(rows):
        if ri in aligned_rows:
            continue
        rid = row.get("sample_id", "")
        for spec in sample_fields:
            if row.get(spec.name) is not None:
                detail = "row matches no gold sample; " + trace(quality, rid, spec.name)
                cells.append(Cell(doc, f"(unaligned) {rid}", rid, spec.name, "extra", row.get(spec.name), "", detail))
    return cells


def prf(counts: Counter) -> tuple[float | None, float | None]:
    tp_p = counts["correct"] + counts["soft"]
    denom_p = tp_p + counts["wrong"] + counts["extra"]
    denom_r = counts["correct"] + counts["wrong"] + counts["missing"]
    precision = tp_p / denom_p if denom_p else None
    recall = counts["correct"] / denom_r if denom_r else None
    return precision, recall


# ---------------------------------------------------------------------------------------------- IO + report


def _recorded_fingerprint(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("profile_fingerprint")
    except (OSError, ValueError, AttributeError):
        return None


def find_dataset(data_root: Path, doc_id: str, keys: str | None, fingerprint: str | None = None) -> Path:
    """The dataset named by ``keys`` (``<extractor_key>.<comparison_key>``), or the newest one without it.

    Without keys, ``fingerprint`` (the scored profile's comparison fingerprint) leaves out the datasets another
    profile built: a library scored under two profiles would otherwise score one profile's table against the
    other's gold set. A dataset that records no fingerprint predates profiles and is kept."""
    candidates = [p for p in (data_root / "docs").glob(f"{doc_id}*/datasets/*.json")]
    if keys is not None:
        candidates = [p for p in candidates if p.stem == keys]
    elif fingerprint is not None:
        candidates = [p for p in candidates if _recorded_fingerprint(p) in (None, fingerprint)]
    if not candidates:
        which = f"dataset {keys}.json" if keys is not None else "dataset"
        if keys is None and fingerprint is not None:
            which += f" of profile fingerprint {fingerprint[:12]}"
        raise FileNotFoundError(f"no {which} for {doc_id} under {data_root / 'docs'}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")[:300]


def fmt(x: float | None) -> str:
    return "–" if x is None else f"{x:.2f}"


def table(title: str, groups: dict[str, Counter]) -> list[str]:
    out = [f"| {title} | correct | soft | wrong | missing | extra | disputed | precision | recall |", "|" + "---|" * 9]
    for key, counts in groups.items():
        p, r = prf(counts)
        out.append(f"| {key} | " + " | ".join(str(counts[o]) for o in OUTCOMES) + f" | {fmt(p)} | {fmt(r)} |")
    return out


def report(cells: list[Cell], sources: dict[str, str], specs: dict[str, Spec]) -> str:
    total = Counter(c.outcome for c in cells)
    by_field: dict[str, Counter] = {name: Counter() for name in specs}
    by_group: dict[str, Counter] = defaultdict(Counter)
    by_doc: dict[str, Counter] = {doc: Counter() for doc in sources}
    for c in cells:
        by_field[c.field][c.outcome] += 1
        by_group[specs[c.field].group][c.outcome] += 1
        by_doc[c.doc][c.outcome] += 1
    p, r = prf(total)
    # Micro averages are dominated by long series (one paper has 36 rows repeating the same recipe), so the
    # macro average over papers is reported next to them.
    per_doc = [prf(counts) for counts in by_doc.values()]
    macro_p = [x for x, _ in per_doc if x is not None]
    macro_r = [y for _, y in per_doc if y is not None]
    mp = sum(macro_p) / len(macro_p) if macro_p else None
    mr = sum(macro_r) / len(macro_r) if macro_r else None
    lines = ["# PaperFacts gold evaluation", ""]
    lines += [f"Micro: precision {fmt(p)}, recall {fmt(r)}. Macro over papers: precision {fmt(mp)}, recall {fmt(mr)}."]
    lines += ["", "Datasets scored:", "", *(f"- `{d}`: `{s}`" for d, s in sources.items()), ""]
    lines += ["## Overall", "", *table("all", {"all": total}), ""]
    lines += ["## Per field group", "", *table("group", dict(sorted(by_group.items()))), ""]
    lines += ["## Per field", "", *table("field", {k: v for k, v in by_field.items() if sum(v.values())}), ""]
    lines += ["## Per paper", "", *table("paper", dict(sorted(by_doc.items()))), ""]
    lines += ["## Wrong, extra, missing and disputed cells", ""]
    header = ("paper", "gold sample", "dataset row", "field", "outcome", "dataset value", "gold", "quality_rows trace")
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for c in sorted(cells, key=lambda c: (c.doc, c.outcome, c.sample, c.field)):
        if c.outcome in {"correct", "soft"}:
            continue
        values = (c.doc, c.sample, c.row, c.field, c.outcome, c.value, c.gold, c.detail)
        lines.append("| " + " | ".join(_escape(v) for v in values) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=REPO / "eval" / "gold", help="directory of <doc_id>.json gold files")
    ap.add_argument(
        "--profile", type=Path, default=REPO / "profiles" / "tco.json", help="profile JSON with the field table"
    )
    ap.add_argument("--keys", metavar="EK.CK", help="score datasets/<EK.CK>.json (default: the newest dataset)")
    ap.add_argument("--data-root", type=Path, default=REPO / "data", help="PaperFacts data root (contains docs/)")
    ap.add_argument("--dataset", action="append", default=[], metavar="DOC=PATH", help="explicit dataset JSON")
    ap.add_argument("--only", action="append", default=[], metavar="DOC", help="score only these gold doc ids")
    ap.add_argument("--out", type=Path, help="write the markdown report here instead of stdout")
    ap.add_argument("--json", type=Path, help="also write every scored cell as JSON")
    args = ap.parse_args(argv)

    profile = load_scoring_profile(args.profile)
    specs = profile.by_name
    fingerprint = profile_comparison_fingerprint(profile)
    explicit = dict(item.split("=", 1) for item in args.dataset)
    cells: list[Cell] = []
    sources: dict[str, str] = {}
    # Dotfiles are skipped: macOS tar/cp leave "._<name>.json" AppleDouble files beside the real ones.
    for path in sorted(p for p in args.gold.glob("*.json") if not p.name.startswith(".")):
        gold = json.loads(path.read_text(encoding="utf-8"))
        doc = gold["doc_id"]
        if args.only and doc not in args.only:
            continue
        ds_path = Path(explicit[doc]) if doc in explicit else find_dataset(args.data_root, doc, args.keys, fingerprint)
        sources[doc] = str(ds_path)
        cells += score_document(specs, gold, json.loads(ds_path.read_text(encoding="utf-8")))
    text = report(cells, sources, specs)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.json:
        args.json.write_text(
            json.dumps([asdict(c) for c in cells], ensure_ascii=False, indent=1, default=str), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
