"""Which profile a command runs under: loaded once per entry point, from ``--profile``, else from the settings
(``PAPERFACTS_PROFILE`` or config.json's ``profile``), and handed to everything the command calls."""

from __future__ import annotations

import contextlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from typer.testing import CliRunner

from paperfacts.cli import app
from paperfacts.config import Settings
from paperfacts.errors import ConfigError, ParserError
from paperfacts.keys import comparison_key_for, extractor_key_for, figure_key_for
from paperfacts.profile import DomainProfile
from paperfacts.workflow import load_run_profile, run_document
from support.profiles import SHIPPED_PROFILE_PATH, profile_data

runner = CliRunner()


@pytest.fixture
def demo_path(tmp_path: Path) -> Path:
    """A synthetic profile on disk; its file name must be its ``name``."""
    path = tmp_path / "profiles" / "demo.json"
    path.parent.mkdir()
    path.write_text(json.dumps(profile_data(), ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def run_profiles(monkeypatch) -> list[DomainProfile]:
    """The profile every ``run_document`` the CLI starts is handed."""
    seen: list[DomainProfile] = []

    def fake_run_document(document: Any, settings: Settings, profile: DomainProfile, **kwargs: Any) -> None:
        seen.append(profile)
        raise ParserError("mineru", "run", "stop here, the profile has been observed")

    monkeypatch.setattr("paperfacts.cli.run_document", fake_run_document)
    return seen


@pytest.fixture
def served(monkeypatch) -> list[FastAPI]:
    apps: list[FastAPI] = []
    monkeypatch.setattr(uvicorn, "run", lambda application, **kwargs: apps.append(application))
    return apps


def test_run_uses_the_profile_the_environment_names(
    monkeypatch, run_profiles: list[DomainProfile], demo_path: Path, two_page_pdf: Path, tmp_path: Path
):
    monkeypatch.setenv("PAPERFACTS_PROFILE", str(demo_path))

    runner.invoke(app, ["run", str(two_page_pdf), "--data-root", str(tmp_path / "data")])

    assert [profile.name for profile in run_profiles] == ["demo"]


def test_the_profile_flag_overrides_the_environment(
    monkeypatch, run_profiles: list[DomainProfile], demo_path: Path, two_page_pdf: Path, tmp_path: Path
):
    monkeypatch.setenv("PAPERFACTS_PROFILE", str(demo_path))

    runner.invoke(
        app, ["run", str(two_page_pdf), "--data-root", str(tmp_path / "data"), "--profile", str(SHIPPED_PROFILE_PATH)]
    )

    assert [profile.name for profile in run_profiles] == ["tco"]


def test_a_profile_that_is_not_there_fails_before_any_work(
    run_profiles: list[DomainProfile], two_page_pdf: Path, tmp_path: Path
):
    result = runner.invoke(
        app, ["run", str(two_page_pdf), "--data-root", str(tmp_path / "data"), "--profile", "no_such_profile"]
    )

    assert result.exit_code == 1
    assert "no profile at" in result.output
    assert run_profiles == []


def test_serve_builds_the_library_under_the_profile_the_environment_names(
    monkeypatch, served: list[FastAPI], demo_path: Path, tmp_path: Path, tco_profile: DomainProfile
):
    monkeypatch.setenv("PAPERFACTS_PROFILE", str(demo_path))

    result = runner.invoke(app, ["serve", "--data-root", str(tmp_path / "data")])

    assert result.exit_code == 0, result.output
    assert "profile=demo" in result.output
    library = served[0].state.library
    settings = served[0].state.settings
    assert library.profile.name == "demo"
    # The browser looks for files under the demo keys, which are not TCO's.
    assert library.extractor_key == extractor_key_for(settings, library.profile)
    assert library.extractor_key != extractor_key_for(settings, tco_profile)
    assert library.comparison_key != comparison_key_for(settings, tco_profile)
    assert library.figure_key != figure_key_for(settings, tco_profile)


def _stop(seen: list[DomainProfile]):
    def record(*args: Any, **kwargs: Any) -> None:
        # extract and compare are handed the profile inside their options.
        profiles = (getattr(arg, "profile", arg) for arg in args)
        seen.extend(profile for profile in profiles if isinstance(profile, DomainProfile))
        raise ParserError("mineru", "run", "stop here, the profile has been observed")

    return record


@pytest.mark.parametrize("command", ["batch", "export", "extract", "compare"])
def test_every_command_runs_under_the_profile_flag(
    monkeypatch, command: str, demo_path: Path, two_page_pdf: Path, tmp_path: Path
):
    seen: list[DomainProfile] = []
    monkeypatch.setattr("paperfacts.cli.run_batch", _stop(seen))
    monkeypatch.setattr("paperfacts.cli.extract_document", _stop(seen))
    monkeypatch.setattr("paperfacts.cli.compare_document", _stop(seen))
    monkeypatch.setattr(
        "paperfacts.cli.build_llm_client", lambda settings: contextlib.nullcontext(SimpleNamespace(model="fake-model"))
    )
    source = two_page_pdf.parent if command in {"batch", "export"} else two_page_pdf

    runner.invoke(app, [command, str(source), "--data-root", str(tmp_path / "data"), "--profile", str(demo_path)])

    assert [profile.name for profile in seen] == ["demo"]


def test_the_app_hands_its_jobs_the_profile_its_library_uses(monkeypatch, tmp_path: Path, demo_path: Path):
    from paperfacts.web import app as web_app

    handed: list[DomainProfile] = []
    real = web_app.pipeline_runner

    def recording(settings: Settings, registry: Any) -> Any:
        handed.append(registry.default.profile)
        return real(settings, registry)

    monkeypatch.setattr(web_app, "pipeline_runner", recording)
    settings = Settings(data_root=tmp_path / "data", repo_root=tmp_path, profile=str(demo_path))

    application = web_app.create_app(settings)

    assert handed == [application.state.library.profile]
    assert handed[0] is application.state.library.profile


# ---- A profile's name decides its workbook's name ----------------------------------------------------------


def _write_profile(path: Path, changes: dict[str, Any] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = profile_data({"name": path.stem, **(changes or {})})
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_profile_by_path_may_not_shadow_a_different_repository_profile_of_its_name(tmp_path: Path):
    shipped = _write_profile(tmp_path / "repo" / "profiles" / "demo.json")
    elsewhere = _write_profile(tmp_path / "mine" / "demo.json", {"prompt.domain_subject": "something else"})

    with pytest.raises(ConfigError) as refused:
        load_run_profile(Settings(repo_root=tmp_path / "repo", profile=str(elsewhere)))

    assert str(shipped.resolve()) in str(refused.value) and str(elsewhere.resolve()) in str(refused.value)


def test_a_copy_of_a_repository_profile_is_accepted(tmp_path: Path):
    _write_profile(tmp_path / "repo" / "profiles" / "demo.json")
    copy = _write_profile(tmp_path / "mine" / "demo.json")

    assert load_run_profile(Settings(repo_root=tmp_path / "repo", profile=str(copy))).name == "demo"


def test_a_copy_differing_only_in_display_text_is_refused(tmp_path: Path):
    # Same content hash, but a workbook titled from either file would carry the other's display text.
    _write_profile(tmp_path / "repo" / "profiles" / "demo.json")
    copy = _write_profile(tmp_path / "mine" / "demo.json", {"title_zh": "另一个标题"})

    with pytest.raises(ConfigError, match="differ"):
        load_run_profile(Settings(repo_root=tmp_path / "repo", profile=str(copy)))


def test_a_relative_path_to_a_copy_is_accepted(monkeypatch, tmp_path: Path):
    _write_profile(tmp_path / "repo" / "profiles" / "demo.json")
    copy = _write_profile(tmp_path / "mine" / "demo.json")
    monkeypatch.chdir(tmp_path)

    profile = load_run_profile(Settings(repo_root=tmp_path / "repo", profile="mine/demo.json"))

    assert profile.source == copy.resolve()


def test_a_symlink_to_the_repository_file_is_that_file(tmp_path: Path):
    shipped = _write_profile(tmp_path / "repo" / "profiles" / "demo.json")
    link = tmp_path / "mine" / "demo.json"
    link.parent.mkdir()
    link.symlink_to(shipped)

    profile = load_run_profile(Settings(repo_root=tmp_path / "repo", profile=str(link)))

    assert profile.source == shipped.resolve()


def test_a_bare_name_is_the_repository_file(tmp_path: Path):
    shipped = _write_profile(tmp_path / "repo" / "profiles" / "demo.json")

    assert load_run_profile(Settings(repo_root=tmp_path / "repo", profile="demo")).source == shipped.resolve()


def test_the_legacy_export_name_is_reserved(tmp_path: Path):
    path = _write_profile(tmp_path / "paperfacts.json")

    with pytest.raises(ConfigError, match="reserved"):
        load_run_profile(Settings(repo_root=tmp_path, profile=str(path)))


def test_run_document_takes_no_default_profile():
    # Every entry point loads its profile once with load_run_profile and passes it; a silent fallback here would
    # skip the reserved-name and shadow checks that loader makes.
    assert inspect.signature(run_document).parameters["profile"].default is inspect.Parameter.empty


# `paperfacts prompts` output for the shipped profiles, recorded on main before the rendering moved from cli.py to
# profile_view.py; the web's prompt preview is the same function, so this pins both.
GOLDEN_PROMPTS = Path(__file__).parent / "fixtures" / "cli_prompts"


@pytest.mark.parametrize("golden", sorted(GOLDEN_PROMPTS.glob("*.txt")), ids=lambda path: path.stem)
def test_prompts_output_is_the_recorded_golden(golden: Path):
    name, _, field = golden.stem.partition("--field-")
    args = ["prompts", "--profile", str(SHIPPED_PROFILE_PATH.parent / f"{name}.json")]
    if field:
        args += ["--field", field]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert result.stdout == golden.read_text(encoding="utf-8")


def test_an_unknown_field_is_the_recorded_golden_message_and_exit_code():
    golden = json.loads((GOLDEN_PROMPTS / "unknown-field.json").read_text(encoding="utf-8"))

    result = runner.invoke(app, ["prompts", "--profile", str(SHIPPED_PROFILE_PATH), "--field", "colour"])

    assert {"exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr} == golden


def test_every_shipped_profile_has_a_prompts_golden():
    shipped = {path.stem for path in SHIPPED_PROFILE_PATH.parent.glob("*.json")}
    assert shipped == {path.stem for path in GOLDEN_PROMPTS.glob("*.txt") if "--field-" not in path.stem}
