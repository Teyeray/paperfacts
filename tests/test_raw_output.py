"""The ``meta.json`` contract: the sole interface between a runner and the main package.

A runner never imports the main package, so this contract can only be pinned down by
**validation**. This file does three things:

1. runs a fixture of real runner output through :class:`ParserMeta` (a runner field change goes
   red here first);
2. pins down the required fields the main package actually depends on — missing one must
   error, rather than falling back to a default and computing a wrong coordinate;
3. pins down the round trip between ``PageMeta.json_path`` and the JSON key ``"json"``, because
   the HTTP path relies on it to write files isomorphic to a runner's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from paperfacts.models import META_FILENAME, PageMeta, ParserMeta, RawParseOutput, SourceMeta
from support.factories import FIXTURES_DIR

REAL_SAMPLES = {
    "mineru": FIXTURES_DIR / "mineru_real_sample",
    "paddleocr_vl": FIXTURES_DIR / "paddle_real_sample",
}

SHA = "5c10f7a0128f15e06358cdfcb9fd2d36337fe35dafb670e31a00149274414fbd"


def minimal_meta(**overrides: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "parser": "mineru",
        "parser_version": "3.4.5",
        "source": {
            "pdf": "/tmp/paper.pdf",
            "sha256": SHA,
            "page_count": 10,
            "parsed_page_count": 2,
            "page_range": [0, 1],
        },
        "pages": [{"index": 0, "width_pt": 595.0, "height_pt": 842.0}],
        "files": {"content_list": "native/document/auto/document_content_list.json"},
    }
    meta.update(overrides)
    return meta


def write_meta(out_dir: Path, meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / META_FILENAME).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return out_dir


# ---- Real runner output ---------------------------------------------------------------


@pytest.mark.parametrize("backend", sorted(REAL_SAMPLES))
def test_real_runner_meta_json_satisfies_the_contract(backend):
    """A meta.json written by a real runner must pass ParserMeta directly; if a runner changes
    its fields, this goes red first."""
    text = (REAL_SAMPLES[backend] / META_FILENAME).read_text(encoding="utf-8")

    meta = ParserMeta.model_validate_json(text)

    assert meta.parser == backend
    assert meta.source.sha256 == SHA
    assert meta.source.parsed_page_count == 2
    assert meta.pages


def test_real_runner_meta_keeps_its_private_extra_fields():
    # extra="allow": fields a runner cares about itself (backend / framework / render_dpi ...)
    # must not be swallowed by the model.
    mineru = ParserMeta.model_validate_json((REAL_SAMPLES["mineru"] / META_FILENAME).read_text(encoding="utf-8"))
    paddle = ParserMeta.model_validate_json((REAL_SAMPLES["paddleocr_vl"] / META_FILENAME).read_text(encoding="utf-8"))

    assert mineru.model_dump()["backend"] == "pipeline"
    assert paddle.model_dump()["render_dpi"] == 200
    assert paddle.model_dump()["framework"]["paddlepaddle"] == "3.2.1"


def test_real_paddle_meta_exposes_page_files_through_the_json_alias():
    paddle = ParserMeta.model_validate_json((REAL_SAMPLES["paddleocr_vl"] / META_FILENAME).read_text(encoding="utf-8"))

    assert paddle.pages[0].json_path == "pages/page_000.json"
    assert paddle.pages[0].width_px == 1654


# ---- Required fields --------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["sha256", "parsed_page_count", "page_count", "pdf"])
def test_load_rejects_meta_missing_a_required_source_field(tmp_path: Path, missing):
    # Missing even one required field must blow up: a silent default would let downstream code
    # compute a page number/checksum that looks plausible but is wrong.
    source = minimal_meta()["source"]
    del source[missing]
    out_dir = write_meta(tmp_path / "raw", minimal_meta(source=source))

    with pytest.raises(ValueError):
        RawParseOutput.load(out_dir, "mineru")


def test_load_rejects_a_truncated_sha256(tmp_path: Path):
    out_dir = write_meta(tmp_path / "raw", minimal_meta(source={**minimal_meta()["source"], "sha256": "abc"}))

    with pytest.raises(ValueError):
        RawParseOutput.load(out_dir, "mineru")


def test_load_rejects_an_unknown_parser_name(tmp_path: Path):
    out_dir = write_meta(tmp_path / "raw", minimal_meta(parser="tesseract"))

    with pytest.raises(ValueError):
        RawParseOutput.load(out_dir, "mineru")


def test_load_rejects_meta_belonging_to_the_other_parser(tmp_path: Path):
    out_dir = write_meta(tmp_path / "raw", minimal_meta(parser="paddleocr_vl"))

    with pytest.raises(ValueError, match="belongs to parser='paddleocr_vl'"):
        RawParseOutput.load(out_dir, "mineru")


def test_load_raises_file_not_found_when_meta_json_is_missing(tmp_path: Path):
    (tmp_path / "raw").mkdir()

    with pytest.raises(FileNotFoundError, match="parser output is incomplete"):
        RawParseOutput.load(tmp_path / "raw", "mineru")


def test_load_defaults_the_version_to_unknown(tmp_path: Path):
    meta = minimal_meta()
    del meta["parser_version"]
    out_dir = write_meta(tmp_path / "raw", meta)

    assert RawParseOutput.load(out_dir, "mineru").backend_version == "unknown"


def test_load_marks_the_result_as_a_cache_hit_when_asked(tmp_path: Path):
    out_dir = write_meta(tmp_path / "raw", minimal_meta())

    assert RawParseOutput.load(out_dir, "mineru", cache_hit=True).cache_hit is True


# ---- page_offset ----------------------------------------------------------------------


@pytest.mark.parametrize(("page_range", "expected"), [([0, None], 0), ([3, None], 3), ([3, 7], 3), ([None, None], 0)])
def test_page_offset_comes_from_the_start_of_the_page_range(page_range, expected):
    # After MinerU crops with --start-page, page_idx restarts from 0; the adapter must add the
    # offset back.
    source = SourceMeta(pdf="p.pdf", sha256=SHA, page_count=10, parsed_page_count=2, page_range=page_range)

    assert source.page_offset == expected


def test_page_range_defaults_to_the_whole_document():
    source = SourceMeta(pdf="p.pdf", sha256=SHA, page_count=3, parsed_page_count=3)

    assert source.page_range == (0, None)
    assert source.page_offset == 0


# ---- PageMeta's "json" alias -----------------------------------------------------------


def test_page_meta_reads_the_json_key_and_dumps_it_back_under_the_same_name():
    # A runner writes "json"; the main package's attribute is called json_path (avoiding the
    # builtin name). The alias must hold in both directions.
    raw = {
        "index": 0,
        "width_pt": 595.0,
        "height_pt": 842.0,
        "width_px": 1654,
        "height_px": 2174,
        "image": "pages/page_000.png",
        "json": "pages/page_000.json",
        "markdown_dir": "pages/page_000_md",
    }

    page = PageMeta.model_validate(raw)

    assert page.json_path == "pages/page_000.json"
    assert page.model_dump(by_alias=True)["json"] == "pages/page_000.json"
    assert "json_path" not in page.model_dump(by_alias=True)
    assert PageMeta.model_validate(page.model_dump(by_alias=True)) == page


def test_page_meta_also_accepts_the_python_field_name():
    # populate_by_name: more natural for the main package to construct with json_path=... itself.
    page = PageMeta(index=1, width_pt=1.0, height_pt=2.0, json_path="pages/page_001.json")

    assert page.model_dump(by_alias=True)["json"] == "pages/page_001.json"


def test_page_meta_leaves_render_fields_none_for_a_mineru_page():
    # The MinerU runner doesn't render per page, so these fields simply don't exist; None is a
    # valid state, not a defect.
    page = PageMeta(index=0, width_pt=595.0, height_pt=842.0)

    assert (page.width_px, page.height_px, page.json_path, page.image, page.markdown_dir) == (None,) * 5


@pytest.mark.parametrize("field", ["width_px", "height_px"])
def test_page_meta_rejects_non_positive_pixel_sizes(field):
    # 0 would cause division by zero during normalization; a negative number would flip the
    # bbox. Both must be blocked at the contract layer.
    with pytest.raises(ValidationError):
        PageMeta.model_validate({"index": 0, "width_pt": 1.0, "height_pt": 1.0, field: 0})


def test_parser_meta_round_trips_through_json_with_aliases():
    meta = ParserMeta.model_validate(
        minimal_meta(
            parser="paddleocr_vl",
            pages=[{"index": 0, "width_pt": 1.0, "height_pt": 2.0, "width_px": 3, "height_px": 4, "json": "p.json"}],
        )
    )

    restored = ParserMeta.model_validate_json(meta.model_dump_json(by_alias=True))

    assert restored == meta
    assert restored.pages[0].json_path == "p.json"
