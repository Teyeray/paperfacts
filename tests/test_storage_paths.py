"""The single source of truth for on-disk paths.

Every assertion here is a **directory contract**: if the path rules drift, existing cache
directories get silently bypassed and the expensive parsers get silently rerun.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperfacts.storage.paths import DOC_DIR_ID_LENGTH, DataLayout

DOC_ID = "0123456789abcdef" + "f" * 48  # 16-char prefix padded out to 64


@pytest.fixture
def layout() -> DataLayout:
    return DataLayout(Path("/data"))


def test_doc_dir_uses_the_first_sixteen_hex_characters(layout: DataLayout):
    # Unique enough, and short enough to read in a terminal; the full sha256 lives in artifact / meta.json.
    assert layout.doc_dir(DOC_ID) == Path("/data/docs/0123456789abcdef")
    assert DOC_DIR_ID_LENGTH == 16


def test_two_document_ids_sharing_a_prefix_collide_by_design(layout: DataLayout):
    # Documents current behavior: the directory name only looks at the first 16 chars, so two
    # documents sharing a prefix share a directory.
    other = "0123456789abcdef" + "0" * 48

    assert layout.doc_dir(other) == layout.doc_dir(DOC_ID)


@pytest.mark.parametrize("backend", ["mineru", "paddleocr_vl"])
def test_raw_dir_is_per_document_and_per_backend(layout: DataLayout, backend):
    assert layout.raw_dir(DOC_ID, backend) == Path(f"/data/docs/0123456789abcdef/raw/{backend}")


def test_parsed_artifacts_share_one_directory_and_differ_only_by_filename(layout: DataLayout):
    parsed = layout.parsed_dir(DOC_ID)

    assert parsed == Path("/data/docs/0123456789abcdef/parsed")
    assert layout.markdown_path(DOC_ID, "mineru") == parsed / "mineru.md"
    assert layout.sources_path(DOC_ID, "mineru") == parsed / "mineru.sources.json"
    assert layout.artifact_path(DOC_ID, "mineru") == parsed / "mineru.artifact.json"


def test_the_two_backends_never_write_to_the_same_file(layout: DataLayout):
    files = {
        layout.markdown_path(DOC_ID, "mineru"),
        layout.markdown_path(DOC_ID, "paddleocr_vl"),
        layout.artifact_path(DOC_ID, "mineru"),
        layout.artifact_path(DOC_ID, "paddleocr_vl"),
        layout.sources_path(DOC_ID, "mineru"),
        layout.sources_path(DOC_ID, "paddleocr_vl"),
    }

    assert len(files) == 6


def test_overlay_dir_is_per_document_and_per_backend(layout: DataLayout):
    assert layout.overlay_dir(DOC_ID, "paddleocr_vl") == Path("/data/docs/0123456789abcdef/overlays/paddleocr_vl")


def test_every_derived_path_stays_under_the_data_root(layout: DataLayout):
    derived = [
        layout.doc_dir(DOC_ID),
        layout.raw_dir(DOC_ID, "mineru"),
        layout.parsed_dir(DOC_ID),
        layout.markdown_path(DOC_ID, "mineru"),
        layout.overlay_dir(DOC_ID, "mineru"),
    ]

    assert all(path.is_relative_to(layout.root) for path in derived)


def test_a_relative_root_produces_relative_paths():
    layout = DataLayout(Path("data"))

    assert layout.markdown_path(DOC_ID, "mineru") == Path("data/docs/0123456789abcdef/parsed/mineru.md")


def test_layout_methods_do_not_touch_the_filesystem(tmp_path: Path):
    # Only computes paths, never creates directories: the caller decides when to write to disk.
    layout = DataLayout(tmp_path)

    layout.artifact_path(DOC_ID, "mineru")
    layout.overlay_dir(DOC_ID, "mineru")

    assert list(tmp_path.iterdir()) == []


def test_layout_is_frozen(tmp_path: Path):
    import dataclasses

    layout = DataLayout(tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.root = tmp_path / "elsewhere"


# ---- extraction & alignment (M2) ----------------------------------------------------


def test_extraction_path_is_per_backend_and_per_extractor(layout: DataLayout):
    # extractor_key is part of the filename: changing the prompt / model / schema writes to a new
    # file automatically, leaving the old result around for comparison.
    path = layout.extraction_path(DOC_ID, "mineru", "abc123def456")

    assert path == Path("/data/docs/0123456789abcdef/facts/mineru.abc123def456.json")


def test_the_two_backends_never_share_an_extraction_file(layout: DataLayout):
    key = "abc123def456"

    assert layout.extraction_path(DOC_ID, "mineru", key) != layout.extraction_path(DOC_ID, "paddleocr_vl", key)


def test_a_different_extractor_key_is_a_different_file(layout: DataLayout):
    assert layout.extraction_path(DOC_ID, "mineru", "aaa") != layout.extraction_path(DOC_ID, "mineru", "bbb")


def test_comparison_path_carries_both_keys(layout: DataLayout):
    """The report path carries both the extractor_key and the comparison_key: changing tolerance
    skips re-running the LLM but still forces the comparison to be recomputed.
    """
    path = layout.comparison_path(DOC_ID, "abc123def456", "999888777666")

    assert path == Path("/data/docs/0123456789abcdef/comparisons/abc123def456.999888777666.json")


def test_changing_either_comparison_key_changes_the_file(layout: DataLayout):
    baseline = layout.comparison_path(DOC_ID, "aaa", "bbb")

    assert layout.comparison_path(DOC_ID, "ccc", "bbb") != baseline
    assert layout.comparison_path(DOC_ID, "aaa", "ccc") != baseline


def test_the_llm_cache_is_shared_across_documents(layout: DataLayout):
    # Content-addressed: if the same prompt has already appeared for another paper, it's free the
    # second time, which is why this lives outside doc_dir.
    cache = layout.llm_cache_dir()

    assert cache == Path("/data/llm_cache")
    assert not cache.is_relative_to(layout.doc_dir(DOC_ID))


def test_the_m2_paths_stay_under_the_data_root(layout: DataLayout):
    derived = [
        layout.extraction_path(DOC_ID, "mineru", "key"),
        layout.comparison_path(DOC_ID, "key", "cmp"),
        layout.llm_cache_dir(),
    ]

    assert all(path.is_relative_to(layout.root) for path in derived)


def test_the_m2_paths_do_not_touch_the_filesystem(tmp_path: Path):
    local = DataLayout(tmp_path)

    local.extraction_path(DOC_ID, "mineru", "key")
    local.comparison_path(DOC_ID, "key", "cmp")
    local.llm_cache_dir()

    assert list(tmp_path.iterdir()) == []
