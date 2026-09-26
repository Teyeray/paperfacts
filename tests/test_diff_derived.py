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


# ---- across the round 2 rename (target -> paper, no_tco_film -> no_samples) -------------------------------

B1_FIXTURES = Path(__file__).parent / "fixtures" / "b1_formats"
B1_FILES = {
    "facts:mineru": "lane.json",
    "facts:paddleocr_vl": "lane_paddleocr_vl.json",
    "comparisons": "report.json",
    "datasets": "dataset.json",
}
# The rename spelled out on the file text, independently of the script's own mapping.
RENAMED_TEXT = (
    ('"target":', '"paper":'),
    ('"no_tco_film":', '"no_samples":'),
    ('"scope": "target"', '"scope": "paper"'),
    ('"sample_id": "target"', '"sample_id": "paper"'),
)


def renamed(text: str) -> str:
    for old, new in RENAMED_TEXT:
        text = text.replace(old, new)
    data = json.loads(text)
    if "matching" in data:
        # A report's one matching became the implicit entity's entry of ``matchings``.
        data["matchings"] = {"sample": data.pop("matching")}
        text = json.dumps(data, ensure_ascii=False, indent=2)
    return text


def b1_library(tmp_path: Path, new_text=renamed) -> Path:
    """One document: the B1 fixtures under the old key, and the same files after ``new_text`` under the new."""
    doc = tmp_path / "docs" / "abcd"
    for key, transform in ((OLD, lambda text: text), (NEW, new_text)):
        for kind, name in B1_FILES.items():
            text = transform((B1_FIXTURES / name).read_text(encoding="utf-8"))
            if kind.startswith("facts:"):
                path = doc / "facts" / f"{kind.removeprefix('facts:')}.{key[0]}.json"
            else:
                path = doc / kind / f"{key[0]}.{key[1]}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return tmp_path


def test_b1_against_b1_reads_no_legacy_names_and_is_identical(tmp_path: Path):
    root = b1_library(tmp_path, new_text=lambda text: text)

    results = diff_derived.compare_library(root, OLD, NEW, diff_derived.DEFAULT_IGNORED)

    assert {result.status for result in results} == {"identical"}


def test_b1_against_renamed_files_is_identical_by_default(tmp_path: Path):
    root = b1_library(tmp_path)

    code = diff_derived.main(["--data-root", str(root), "--old", "oldext.oldcmp", "--new", "newext.newcmp"])

    assert code == 0


def test_without_legacy_names_the_rename_itself_is_reported(tmp_path: Path, capsys):
    root = b1_library(tmp_path)
    args = ["--data-root", str(root), "--old", "oldext.oldcmp", "--new", "newext.newcmp", "--no-legacy-names"]

    assert diff_derived.main(args) == 1
    out = capsys.readouterr().out
    assert "facts:mineru: differs no_samples, no_tco_film, paper, target" in out
    assert "comparisons: differs comparisons[0].scope" in out
    assert "datasets: differs quality_rows[0].sample_id" in out


def test_a_changed_paper_value_still_differs_across_the_rename(tmp_path: Path):
    root = b1_library(tmp_path, new_text=lambda text: renamed(text).replace("Sn/Ta target 95:5", "Sn/Ta target 90:10"))

    results = diff_derived.compare_library(root, OLD, NEW, diff_derived.DEFAULT_IGNORED)

    differing = {result.kind: result.paths for result in results if result.status == "differs"}
    assert "paper.fields[0].value_raw" in differing["facts:mineru"]


def test_only_the_renamed_spots_are_rewritten():
    lane = {"target": {"fields": []}, "no_tco_film": True, "samples": [{"sample_id": "target"}]}
    report = {"comparisons": [{"scope": "target"}, {"scope": "sample:target|x", "field": "target"}]}

    assert diff_derived.current_names("facts:mineru", lane) == {
        "paper": {"fields": []},
        "no_samples": True,
        "samples": [{"sample_id": "target"}],
    }
    assert diff_derived.current_names("comparisons", report)["comparisons"] == [
        {"scope": "paper"},
        {"scope": "sample:target|x", "field": "target"},
    ]
    assert diff_derived.current_names("comparisons", report | {"matching": {"pairs": []}})["matchings"] == {
        "sample": {"pairs": []}
    }
    current = {"matchings": {"sample": {}}, "comparisons": []}
    assert diff_derived.current_names("comparisons", current) == current
