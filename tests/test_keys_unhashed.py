"""Presentation and orchestration modules stay out of every cache key.

How a workbook is laid out, where readings are stored or how a run is orchestrated decides nothing a stored
result depends on; hashing their source would rename stored extractions, comparisons or readings for an edit
that changes no answer and no verdict.
"""

from __future__ import annotations

import shutil

from paperfacts import keys
from paperfacts.workbook import data_columns

UNHASHED = {
    "workbook.py",
    "columns.py",
    "readings.py",
    "llm.py",
    "config.py",
    "cli.py",
    "workflow.py",
    "batch.py",
    "ui_copy.py",
    "profile_loader.py",
}


def _hashed_modules(monkeypatch, profile) -> set[str]:
    hashed: set[str] = set()
    real = keys.source_fingerprint

    def recording(*module_files: str) -> str:
        hashed.update(module_files)
        return real(*module_files)

    monkeypatch.setattr(keys, "source_fingerprint", recording)
    # The cached fingerprints are called through __wrapped__ so an earlier test's cache cannot hide a list.
    keys.extraction_code_fingerprint.__wrapped__()
    keys.normalization_fingerprint.__wrapped__()
    keys.comparison_code_fingerprint.__wrapped__()
    keys.retrieval_fingerprint.__wrapped__(profile)
    keys.figure_key(profile, "model", dpi=200, max_pixels=1, max_per_document=1)
    return hashed


def test_the_unhashed_modules_appear_in_no_key_list(monkeypatch, tco_profile):
    hashed = _hashed_modules(monkeypatch, tco_profile)

    assert {"extract.py", "dataset.py", "figures.py", "passages.py"} <= hashed  # the recording saw the lists
    assert not hashed & UNHASHED


def test_the_profile_modules_are_hashed_where_what_they_hold_is_read(monkeypatch, tco_profile):
    # text.py folds and units.py converts what normalize.py and passages.py read; profile.py holds the slot
    # defaults the prompts are built from.
    hashed = _hashed_modules(monkeypatch, tco_profile)

    assert {"text.py", "units.py", "profile.py"} <= hashed


def test_the_kind_rules_are_hashed_with_the_stages_that_dispatch_to_them(monkeypatch, tco_profile):
    # normalize_field, the comparison, the dataset cell and the field line all ask kinds.py.
    real = keys.source_fingerprint
    lists: dict[str, tuple[str, ...]] = {}

    def recording(*module_files: str) -> str:
        lists[module_files[0]] = module_files
        return real(*module_files)

    monkeypatch.setattr(keys, "source_fingerprint", recording)
    keys.extraction_code_fingerprint.__wrapped__()
    keys.normalization_fingerprint.__wrapped__()
    keys.comparison_code_fingerprint.__wrapped__()
    keys.retrieval_fingerprint.__wrapped__(tco_profile)

    assert all("kinds.py" in lists[first] for first in ("extract.py", "normalize.py", "compare.py"))
    # Retrieval reads only which kinds need a digit, from fields.py; it never imports kinds.py.
    assert "fields.py" in lists["passages.py"]


def test_a_workbook_header_edit_leaves_the_comparison_key(monkeypatch, tmp_path, tco_profile):
    # The row headers are display text in workbook.py; were they in dataset.py, renaming one would rename every
    # stored comparison. The package is copied so its sources can be edited without touching the checkout.
    assert ("agree_fields", "双路一致字段数") in data_columns(tco_profile)
    before = keys.comparison_code_fingerprint.__wrapped__()
    package = tmp_path / "paperfacts"
    shutil.copytree(keys._PACKAGE_DIR, package, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(keys, "_PACKAGE_DIR", package)
    workbook = package / "workbook.py"
    workbook.write_text(workbook.read_text(encoding="utf-8").replace("双路一致字段数", "两路一致"), encoding="utf-8")

    assert keys.comparison_code_fingerprint.__wrapped__() == before

    dataset = package / "dataset.py"
    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n# an edit\n", encoding="utf-8")
    assert keys.comparison_code_fingerprint.__wrapped__() != before  # the copy is what is hashed
