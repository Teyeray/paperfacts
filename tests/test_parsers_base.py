"""``Parser`` template method: check cache, clear the directory, produce, write meta.json, validate.

Every parser implementation (subprocess / both HTTP backends) inherits from it, so these
invariants only need verifying once, here:

- a cache hit never calls ``_produce`` (the expensive model runs only once, design doc §22);
- a cache miss always starts from a **clean** directory — leftovers from a previous run never
  mix into this one;
- if ``_produce`` raises, the directory has **no** meta.json afterward — its existence is what
  marks output as complete;
- a cached meta.json with the wrong shape fails loudly (stage="cache") instead of silently
  rerunning and masking a contract change.

Verified with a minimal fake subclass; no subprocess or network involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperfacts.errors import ParserError
from paperfacts.models import META_FILENAME, Backend, DocumentInput, ParserMeta
from paperfacts.parsers import Parser, write_meta


def make_meta(parser: Backend = "mineru", *, version: str = "fake-1.0") -> ParserMeta:
    return ParserMeta.model_validate(
        {
            "parser": parser,
            "parser_version": version,
            "source": {
                "pdf": "/tmp/paper.pdf",
                "sha256": "d" * 64,
                "page_count": 2,
                "parsed_page_count": 2,
                "page_range": [0, None],
            },
            "pages": [{"index": 0, "width_pt": 595.0, "height_pt": 842.0}],
            "files": {"content_list": "native.json"},
        }
    )


class FakeParser(Parser):
    """Minimal subclass: writes one native file and returns a ParserMeta for the base class to persist."""

    backend: Backend = "mineru"

    def __init__(self, *, fail: bool = False, meta: ParserMeta | None = None, write_own_meta: bool = False) -> None:
        self.fail = fail
        self.meta = meta if meta is not None else make_meta()
        self.write_own_meta = write_own_meta
        self.calls: list[Path] = []

    def _produce(self, document: DocumentInput, out_dir: Path) -> ParserMeta | None:
        self.calls.append(out_dir)
        (out_dir / "native.json").write_text("[]", encoding="utf-8")
        if self.fail:
            raise ParserError(self.backend, "run", "fake parser failed on purpose")
        if self.write_own_meta:
            # Mimics the runner subprocess: meta.json is already written externally, so
            # _produce returns None.
            write_meta(out_dir, self.meta)
            return None
        return self.meta


# ---- Happy path -----------------------------------------------------------------------


def test_produce_result_is_written_as_meta_json_and_loaded_back(tmp_path: Path, document: DocumentInput):
    parser = FakeParser()
    out_dir = tmp_path / "raw"

    raw = parser.parse(document, out_dir)

    assert (out_dir / META_FILENAME).is_file()
    assert raw.backend == "mineru"
    assert raw.backend_version == "fake-1.0"
    assert raw.cache_hit is False
    assert raw.out_dir == out_dir
    # _produce writes into a staging sibling; only a complete run is renamed onto out_dir.
    assert parser.calls != [out_dir]
    assert parser.calls[0].parent == out_dir.parent
    assert parser.calls[0].name.startswith(f".{out_dir.name}.new.")


def test_returning_none_means_meta_json_was_written_by_the_subclass(tmp_path: Path, document: DocumentInput):
    # This is the path the subprocess parser takes: the runner writes meta.json itself, and the
    # base class only validates and loads it.
    parser = FakeParser(write_own_meta=True)

    raw = parser.parse(document, tmp_path / "raw")

    assert raw.backend_version == "fake-1.0"


def test_output_directory_is_created_including_missing_parents(tmp_path: Path, document: DocumentInput):
    out_dir = tmp_path / "a" / "b" / "raw"

    FakeParser().parse(document, out_dir)

    assert (out_dir / "native.json").is_file()


def test_meta_json_is_written_with_aliases(tmp_path: Path, document: DocumentInput):
    # Matches the runner's structure: written to disk as "json", not json_path.
    meta = ParserMeta.model_validate(
        {
            **make_meta("paddleocr_vl").model_dump(),
            "pages": [{"index": 0, "width_pt": 1.0, "height_pt": 2.0, "width_px": 3, "height_px": 4, "json": "p.json"}],
        }
    )

    class PaddleFake(FakeParser):
        backend: Backend = "paddleocr_vl"

    out_dir = tmp_path / "raw"
    PaddleFake(meta=meta).parse(document, out_dir)

    text = (out_dir / META_FILENAME).read_text(encoding="utf-8")
    assert '"json"' in text
    assert "json_path" not in text


# ---- Cache semantics -----------------------------------------------------------------------


def test_second_parse_hits_the_cache_and_does_not_call_produce(tmp_path: Path, document: DocumentInput):
    parser = FakeParser()
    out_dir = tmp_path / "raw"

    first = parser.parse(document, out_dir)
    second = parser.parse(document, out_dir)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.meta == first.meta
    assert len(parser.calls) == 1


def test_force_bypasses_the_cache_and_calls_produce_again(tmp_path: Path, document: DocumentInput):
    parser = FakeParser()
    out_dir = tmp_path / "raw"
    parser.parse(document, out_dir)

    raw = parser.parse(document, out_dir, force=True)

    assert raw.cache_hit is False
    assert len(parser.calls) == 2


def test_a_cache_miss_starts_from_a_clean_directory(tmp_path: Path, document: DocumentInput):
    # Files left over from a run that stopped halfway must be cleared, or this run's result
    # would point at the previous run's native output.
    out_dir = tmp_path / "raw"
    out_dir.mkdir(parents=True)
    stale = out_dir / "leftover.json"
    stale.write_text("{}", encoding="utf-8")

    FakeParser().parse(document, out_dir)

    assert not stale.exists()


def test_force_wipes_the_directory_before_rerunning(tmp_path: Path, document: DocumentInput):
    parser = FakeParser()
    out_dir = tmp_path / "raw"
    parser.parse(document, out_dir)
    stale = out_dir / "leftover_from_last_run.json"
    stale.write_text("{}", encoding="utf-8")

    parser.parse(document, out_dir, force=True)

    assert not stale.exists()
    assert (out_dir / META_FILENAME).is_file()


# ---- Failure paths -----------------------------------------------------------------------


def test_a_failing_produce_leaves_nothing_behind(tmp_path: Path, document: DocumentInput):
    # The other half of "meta.json's existence means output is complete". The partial output is
    # thrown away with its staging directory: keeping it would mean a failed forced rerun had
    # already destroyed the previous good parse.
    out_dir = tmp_path / "raw"

    with pytest.raises(ParserError):
        FakeParser(fail=True).parse(document, out_dir)

    assert not out_dir.exists()
    assert list(tmp_path.glob(".raw*")) == []


def test_a_failed_run_is_not_mistaken_for_a_cache_hit_next_time(tmp_path: Path, document: DocumentInput):
    out_dir = tmp_path / "raw"
    with pytest.raises(ParserError):
        FakeParser(fail=True).parse(document, out_dir)

    parser = FakeParser()
    raw = parser.parse(document, out_dir)

    assert raw.cache_hit is False
    assert len(parser.calls) == 1


def test_a_subclass_that_writes_nothing_fails_at_the_output_stage(tmp_path: Path, document: DocumentInput):
    class SilentParser(Parser):
        backend: Backend = "mineru"

        def _produce(self, document: DocumentInput, out_dir: Path) -> ParserMeta | None:
            return None

    with pytest.raises(ParserError) as excinfo:
        SilentParser().parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "output"
    assert "parser output is incomplete" in excinfo.value.detail


def test_meta_written_for_another_backend_fails_at_the_output_stage(tmp_path: Path, document: DocumentInput):
    with pytest.raises(ParserError) as excinfo:
        FakeParser(meta=make_meta("paddleocr_vl")).parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "output"
    assert "belongs to parser=" in excinfo.value.detail


def test_an_invalid_cached_meta_fails_loudly_instead_of_silently_rerunning(tmp_path: Path, document: DocumentInput):
    # A meta.json left by an older runner version, with a shape that no longer matches: this is a
    # contract change, and silently rerunning would mask it.
    out_dir = tmp_path / "raw"
    out_dir.mkdir(parents=True)
    (out_dir / META_FILENAME).write_text(json.dumps({"parser": "mineru"}), encoding="utf-8")
    parser = FakeParser()

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, out_dir)

    assert excinfo.value.stage == "cache"
    assert "--force" in excinfo.value.detail
    assert parser.calls == []


def test_force_recovers_from_an_invalid_cached_meta(tmp_path: Path, document: DocumentInput):
    out_dir = tmp_path / "raw"
    out_dir.mkdir(parents=True)
    (out_dir / META_FILENAME).write_text(json.dumps({"parser": "mineru"}), encoding="utf-8")

    raw = FakeParser().parse(document, out_dir, force=True)

    assert raw.cache_hit is False


def test_the_base_class_requires_subclasses_to_implement_produce(tmp_path: Path, document: DocumentInput):
    class Incomplete(Parser):
        backend: Backend = "mineru"

    with pytest.raises(NotImplementedError):
        Incomplete().parse(document, tmp_path / "raw")


def test_parser_error_message_names_backend_and_stage():
    error = ParserError("paddleocr_vl", "http", "connection refused")

    assert str(error) == "[paddleocr_vl] http failed: connection refused"
    assert isinstance(error, RuntimeError)
