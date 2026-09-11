"""Subprocess parser: build the command, wait for the exit code, read meta.json.

Uses a pure-Python fake runner in place of the real MinerU / PaddleOCR runner, so these cases
don't need torch / paddle / model weights. The fake runner appends a line to a counter file on
every execution, turning "did it actually run" into something assertable as fact instead of
guessed from a mock's call count.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from paperfacts.models import META_FILENAME, DocumentInput
from paperfacts.parsers.base import ParserError
from paperfacts.parsers.subprocess_parser import (
    DEFAULT_COMMAND_PREFIX,
    RUNNER_SCRIPTS,
    STDERR_TAIL_LINES,
    SubprocessParser,
    default_runner_script,
)

# Fake runner: follows the same contract as runners/*.py (--pdf / --out, writes meta.json last).
# Extra debug switches let the same script act out three kinds of failure: "fail / skip meta / timeout".
FAKE_RUNNER = """
import argparse, hashlib, json, os, pathlib, sys, time

parser = argparse.ArgumentParser()
parser.add_argument("--pdf", required=True, type=pathlib.Path)
parser.add_argument("--out", required=True, type=pathlib.Path)
parser.add_argument("--counter", required=True, type=pathlib.Path)
parser.add_argument("--dpi", type=int, default=200)
parser.add_argument("--parser-name", default="mineru")
parser.add_argument("--fail", action="store_true")
parser.add_argument("--no-meta", action="store_true")
parser.add_argument("--sleep", type=float, default=0.0)
args = parser.parse_args()

with args.counter.open("a", encoding="utf-8") as fh:
    fh.write("run\\n")

if args.sleep:
    time.sleep(args.sleep)
if args.fail:
    for i in range(60):
        print(f"noise line {i}", file=sys.stderr)
    print("BOOM: fake runner exploded", file=sys.stderr)
    sys.exit(3)

args.out.mkdir(parents=True, exist_ok=True)
(args.out / "native.txt").write_text("native output", encoding="utf-8")
if args.no_meta:
    sys.exit(0)

meta = {
    "parser": args.parser_name,
    "parser_version": "fake-9.9",
    "dpi": args.dpi,
    "tag": os.environ.get("FAKE_RUNNER_TAG"),
    "source": {
        "pdf": str(args.pdf),
        "sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "page_count": 2,
        "parsed_page_count": 2,
        "page_range": [0, None],
    },
}
(args.out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
print(json.dumps({"ok": True}))
"""


@pytest.fixture
def fake_runner(tmp_path: Path) -> Path:
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER, encoding="utf-8")
    return script


@pytest.fixture
def counter(tmp_path: Path) -> Path:
    """The fake runner's execution counter file; kept outside out_dir so force's directory wipe never deletes it."""
    return tmp_path / "runs.log"


def run_count(counter: Path) -> int:
    return len(counter.read_text(encoding="utf-8").splitlines()) if counter.exists() else 0


def make_parser(script: Path, counter: Path, *args: str, **kwargs) -> SubprocessParser:
    return SubprocessParser(
        "mineru",
        script,
        command_prefix=(sys.executable,),
        extra_args=("--counter", str(counter), *args),
        **kwargs,
    )


# ---- Command construction -----------------------------------------------------------------------


def test_command_puts_pdf_and_out_after_the_script_and_extra_args_last(
    tmp_path: Path, document: DocumentInput, fake_runner: Path
):
    parser = SubprocessParser("paddleocr_vl", fake_runner, command_prefix=("uv", "run"), extra_args=("--dpi", "300"))

    cmd = parser.command(document, tmp_path / "out")

    assert cmd == [
        "uv",
        "run",
        str(fake_runner),
        "--pdf",
        str(document.pdf_path),
        "--out",
        str(tmp_path / "out"),
        "--dpi",
        "300",
    ]


def test_default_command_prefix_uses_uv_locked_script_mode():
    # --locked makes a stale lockfile fail outright instead of silently resolving a different set
    # of dependencies.
    assert DEFAULT_COMMAND_PREFIX == ("uv", "run", "--locked", "--script")


def test_default_runner_script_maps_each_backend_to_its_script(tmp_path: Path):
    assert default_runner_script(tmp_path, "mineru") == tmp_path / "runners/mineru_runner.py"
    assert default_runner_script(tmp_path, "paddleocr_vl") == tmp_path / "runners/paddle_runner.py"
    assert set(RUNNER_SCRIPTS) == {"mineru", "paddleocr_vl"}


# ---- Happy path -----------------------------------------------------------------------


def test_parse_runs_the_script_and_loads_the_meta_it_wrote(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    out_dir = tmp_path / "raw" / "mineru"

    raw = make_parser(fake_runner, counter).parse(document, out_dir)

    assert raw.backend == "mineru"
    assert raw.backend_version == "fake-9.9"
    assert raw.cache_hit is False
    assert run_count(counter) == 1
    assert (out_dir / "native.txt").is_file()


def test_parse_creates_the_output_directory_before_launching(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    out_dir = tmp_path / "deep" / "nested" / "mineru"

    make_parser(fake_runner, counter).parse(document, out_dir)

    assert (out_dir / META_FILENAME).is_file()


def test_extra_env_is_merged_on_top_of_the_process_environment(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    parser = make_parser(fake_runner, counter, env={"FAKE_RUNNER_TAG": "from-settings"})

    raw = parser.parse(document, tmp_path / "raw")

    assert raw.meta.model_extra["tag"] == "from-settings"


# ---- Cache --------------------------------------------------------------------------


def test_second_parse_hits_the_cache_and_does_not_rerun_the_script(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    # This is the core guarantee from design doc §22: the expensive model runs only once.
    parser = make_parser(fake_runner, counter)
    out_dir = tmp_path / "raw"

    first = parser.parse(document, out_dir)
    second = parser.parse(document, out_dir)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.meta == first.meta
    assert run_count(counter) == 1


def test_force_wipes_the_directory_and_reruns(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    # A forced rerun must clear the previous run's leftovers, or old pages/files would mix into
    # the new result.
    parser = make_parser(fake_runner, counter)
    out_dir = tmp_path / "raw"
    parser.parse(document, out_dir)
    stale = out_dir / "leftover_from_last_run.json"
    stale.write_text("{}", encoding="utf-8")

    raw = parser.parse(document, out_dir, force=True)

    assert run_count(counter) == 2
    assert raw.cache_hit is False
    assert not stale.exists()


def test_force_works_when_the_directory_does_not_exist_yet(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    raw = make_parser(fake_runner, counter).parse(document, tmp_path / "never-created", force=True)

    assert raw.cache_hit is False


def test_a_cache_miss_with_leftovers_from_a_failed_run_starts_from_a_clean_directory(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    # The invariant is "really running means starting from a clean directory", not just --force:
    # leftovers from a run that failed halfway (no meta.json written) pollute this run just the
    # same (e.g. letting mineru_runner's glob match a previous run's content_list).
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    stale = out_dir / "leftover_from_failed_run.json"
    stale.write_text("{}", encoding="utf-8")

    raw = make_parser(fake_runner, counter).parse(document, out_dir)

    assert raw.cache_hit is False
    assert not stale.exists()
    assert run_count(counter) == 1


def test_an_invalid_cached_meta_fails_at_the_cache_stage_instead_of_silently_rerunning(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    # An existing meta.json with the wrong shape (likely from an older runner) is a contract
    # change: it must fail explicitly and point at --force.
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    (out_dir / META_FILENAME).write_text('{"parser": "mineru"}', encoding="utf-8")

    with pytest.raises(ParserError) as excinfo:
        make_parser(fake_runner, counter).parse(document, out_dir)

    assert excinfo.value.stage == "cache"
    assert "--force" in excinfo.value.detail
    assert run_count(counter) == 0


# ---- Failure paths -----------------------------------------------------------------------


def test_missing_runner_script_fails_at_the_launch_stage(tmp_path: Path, document: DocumentInput, counter: Path):
    parser = make_parser(tmp_path / "does_not_exist.py", counter)

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "launch"
    assert excinfo.value.backend == "mineru"
    assert "runner script not found" in excinfo.value.detail
    assert run_count(counter) == 0


def test_missing_interpreter_fails_at_the_launch_stage(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    parser = SubprocessParser("mineru", fake_runner, command_prefix=("paperfacts-no-such-binary",))

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "launch"
    assert "executable not found" in excinfo.value.detail


def test_non_zero_exit_code_fails_at_the_run_stage_and_keeps_the_stderr_tail(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    # stderr only makes it into the exception on failure, and only its last few lines: enough to
    # diagnose without flooding the terminal.
    parser = make_parser(fake_runner, counter, "--fail")

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "run"
    assert "exit code 3" in excinfo.value.detail
    assert "BOOM: fake runner exploded" in excinfo.value.detail
    assert "noise line 0" not in excinfo.value.detail  # early noise gets truncated
    assert len(excinfo.value.detail.splitlines()) <= 42


def test_script_that_forgets_meta_json_fails_at_the_output_stage(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    # Exit code 0 but no meta.json: the output is incomplete and must never count as a
    # successful parse.
    parser = make_parser(fake_runner, counter, "--no-meta")

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "output"
    assert "parser output is incomplete" in excinfo.value.detail


def test_meta_written_by_another_parser_fails_at_the_output_stage(
    tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path
):
    parser = make_parser(fake_runner, counter, "--parser-name", "paddleocr_vl")

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "output"
    assert "belongs to parser=" in excinfo.value.detail


def test_timeout_fails_at_the_timeout_stage(tmp_path: Path, document: DocumentInput, fake_runner: Path, counter: Path):
    parser = make_parser(fake_runner, counter, "--sleep", "10", timeout_s=0.5)

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "timeout"
    assert "did not finish within 0.5s" in excinfo.value.detail


def test_parser_error_message_names_backend_and_stage():
    error = ParserError("paddleocr_vl", "http", "connection refused")

    assert str(error) == "[paddleocr_vl] http failed: connection refused"
    assert isinstance(error, RuntimeError)


def test_a_directory_is_not_accepted_as_a_runner_script(tmp_path: Path, document: DocumentInput, counter: Path):
    # is_file(), not exists(): a same-named directory must also fail at the launch stage.
    directory = tmp_path / "runner_dir"
    directory.mkdir()
    directory.chmod(directory.stat().st_mode | stat.S_IXUSR)

    with pytest.raises(ParserError) as excinfo:
        make_parser(directory, counter).parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "launch"


def test_stderr_tail_decodes_bytes_and_keeps_only_the_last_lines():
    # subprocess.TimeoutExpired gives bytes when text=False; this defensive branch must decode it.
    from paperfacts.parsers.subprocess_parser import _tail

    payload = "\n".join(f"line {i}" for i in range(100)).encode("utf-8")

    tail = _tail(payload)

    assert tail.splitlines()[0] == f"line {100 - STDERR_TAIL_LINES}"
    assert tail.splitlines()[-1] == "line 99"


def test_stderr_tail_reports_empty_output_explicitly():
    from paperfacts.parsers.subprocess_parser import _tail

    assert _tail(None) == "(empty)"
    assert _tail("") == "(empty)"


def test_stderr_tail_replaces_undecodable_bytes_instead_of_raising():
    from paperfacts.parsers.subprocess_parser import _tail

    assert "\ufffd" in _tail(b"\xff\xfe not utf-8")
