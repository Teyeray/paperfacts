"""Presentation and orchestration modules stay out of every cache key.

How a workbook is laid out, where readings are stored or how a run is orchestrated decides nothing a stored
result depends on; hashing their source would rename stored extractions, comparisons or readings for an edit
that changes no answer and no verdict.
"""

from __future__ import annotations

from paperfacts import keys

UNHASHED = {"workbook.py", "readings.py", "llm.py", "config.py", "cli.py", "workflow.py"}


def _hashed_modules(monkeypatch) -> set[str]:
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
    keys.retrieval_fingerprint.__wrapped__()
    keys.figure_key("model", dpi=200, max_pixels=1, max_per_document=1)
    return hashed


def test_the_unhashed_modules_appear_in_no_key_list(monkeypatch):
    hashed = _hashed_modules(monkeypatch)

    assert {"extract.py", "dataset.py", "figures.py", "passages.py"} <= hashed  # the recording saw the lists
    assert not hashed & UNHASHED
