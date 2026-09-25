"""scripts/diff_derived.py: the library-wide proof that a re-key changed nothing but the keys."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diff_derived.py"
spec = importlib.util.spec_from_file_location("diff_derived", SCRIPT)
diff_derived = importlib.util.module_from_spec(spec)
sys.modules["diff_derived"] = diff_derived  # dataclasses look their module up while the class is built
spec.loader.exec_module(diff_derived)

OLD, NEW = ("oldext", "oldcmp"), ("newext", "newcmp")


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def library(tmp_path: Path, *, new_dataset: dict | None = None) -> Path:
    doc = tmp_path / "docs" / "abcd"
    lane = {"extractor_key": "x", "samples": [{"sample_id": "S1", "fields": [{"value_raw": "12"}]}], "usage": {"a": 1}}
    for key in (OLD, NEW):
        for backend in diff_derived.BACKENDS:
            write(doc / "facts" / f"{backend}.{key[0]}.json", lane | {"extractor_key": key[0]})
        write(doc / "comparisons" / f"{key[0]}.{key[1]}.json", {"comparison_key": key[1], "counts": {"agree": 3}})
    dataset = {"sample_rows": [{"sample_id": "S1", "thickness": 100}]}
    write(doc / "datasets" / f"{OLD[0]}.{OLD[1]}.json", dataset)
    write(doc / "datasets" / f"{NEW[0]}.{NEW[1]}.json", new_dataset if new_dataset is not None else dataset)
    return tmp_path


def test_a_re_key_that_changed_only_keys_and_usage_is_identical(tmp_path: Path):
    results = diff_derived.compare_library(library(tmp_path), OLD, NEW, diff_derived.DEFAULT_IGNORED)

    assert {result.status for result in results} == {"identical"}


def test_a_changed_cell_is_reported_with_its_path_and_fails_the_run(tmp_path: Path, capsys):
    root = library(tmp_path, new_dataset={"sample_rows": [{"sample_id": "S1", "thickness": 101}]})

    code = diff_derived.main(["--data-root", str(root), "--old", "oldext.oldcmp", "--new", "newext.newcmp"])

    assert code == 1
    assert "sample_rows[0].thickness" in capsys.readouterr().out


def test_a_file_missing_on_one_side_fails_the_run(tmp_path: Path):
    root = library(tmp_path)
    (root / "docs" / "abcd" / "datasets" / f"{NEW[0]}.{NEW[1]}.json").unlink()

    assert diff_derived.main(["--data-root", str(root), "--old", "oldext.oldcmp", "--new", "newext.newcmp"]) == 1


def test_a_list_that_grew_is_reported(tmp_path: Path):
    paths = list(diff_derived.differences({"a": [1]}, {"a": [1, 2]}, frozenset()))

    assert paths == ["a[len 1→2]"]
