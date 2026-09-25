"""Which profile a command runs under: loaded once per entry point, from ``--profile``, else from the settings
(``PAPERFACTS_PROFILE`` or config.json's ``profile``), and handed to everything the command calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from typer.testing import CliRunner

from paperfacts.cli import app
from paperfacts.config import Settings
from paperfacts.errors import ParserError
from paperfacts.keys import comparison_key_for, extractor_key_for, figure_key_for
from paperfacts.profile import DomainProfile
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
