"""Move chart readings stored before the per-profile directories into them, once per data root.

Readings used to be stored flat, ``docs/<id>/figures/<figure_key>.json``; they are now kept per profile,
``docs/<id>/figures/<profile>/<figure_key>.json``. Every flat file was read under the TCO profile unless it
records another, and is stamped with that profile as it moves. Without ``--apply`` the plan is only printed.

    uv run python scripts/migrate_figures.py --data-root data            # what would move
    uv run python scripts/migrate_figures.py --data-root data --apply    # move it

Safe to run again: a moved file is gone from the flat directory, and a file whose destination already exists
is left where it is. The TCO profile still reads the flat files until every data root has been migrated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from paperfacts.readings import migrate_legacy_figures
from paperfacts.storage import DataLayout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="PaperFacts data root (contains docs/)")
    parser.add_argument("--apply", action="store_true", help="move the files; without it the plan is printed")
    args = parser.parse_args(argv)
    moves = migrate_legacy_figures(DataLayout(args.data_root), apply=args.apply)
    for source, destination, outcome in moves:
        print(f"{outcome}: {source} -> {destination}")
    movable = sum(1 for *_, outcome in moves if outcome in ("moved", "would move"))
    print(f"{movable} of {len(moves)} flat files {'moved' if args.apply else 'would move'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
