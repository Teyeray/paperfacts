"""``paperfacts serve``: the wiring from the CLI to uvicorn.

The command itself only does three things — assemble Settings, build the app, and hand the listen
parameters to uvicorn — so here ``uvicorn.run`` is swapped for a fake function that records its
calls: not a single port ever actually gets bound.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from typer.testing import CliRunner

from paperfacts.cli import app

runner = CliRunner()


@pytest.fixture
def uvicorn_calls(monkeypatch) -> list[dict[str, Any]]:
    """Record the arguments of every ``uvicorn.run`` call, and return the list of calls."""
    calls: list[dict[str, Any]] = []

    def fake_run(application: FastAPI, **kwargs: Any) -> None:
        calls.append({"app": application, **kwargs})

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return calls


def test_serve_listens_on_the_requested_address(uvicorn_calls: list[dict[str, Any]], tmp_path: Path):
    result = runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "9123", "--data-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert len(uvicorn_calls) == 1
    assert (uvicorn_calls[0]["host"], uvicorn_calls[0]["port"]) == ("0.0.0.0", 9123)
    assert isinstance(uvicorn_calls[0]["app"], FastAPI)


def test_serve_defaults_to_localhost(uvicorn_calls: list[dict[str, Any]], tmp_path: Path):
    # listens on localhost only by default: exposing the interface to the LAN must be an explicit choice.
    result = runner.invoke(app, ["serve", "--data-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (uvicorn_calls[0]["host"], uvicorn_calls[0]["port"]) == ("127.0.0.1", 8000)


def test_serve_hands_uvicorn_an_app_wired_to_the_requested_data_root(
    uvicorn_calls: list[dict[str, Any]], tmp_path: Path
):
    data_root = tmp_path / "elsewhere"

    result = runner.invoke(app, ["serve", "--data-root", str(data_root)])

    assert result.exit_code == 0, result.output
    assert uvicorn_calls[0]["app"].state.settings.data_root == data_root
    assert uvicorn_calls[0]["app"].state.library.layout.root == data_root


def test_serve_prints_the_url_and_where_the_data_lives(uvicorn_calls: list[dict[str, Any]], tmp_path: Path):
    result = runner.invoke(app, ["serve", "--port", "9000", "--data-root", str(tmp_path)])

    assert "http://127.0.0.1:9000" in result.output
    assert str(tmp_path) in result.output


@pytest.mark.parametrize(("flags", "log_level"), [([], "warning"), (["--verbose"], "info")])
def test_the_verbose_flag_decides_how_chatty_uvicorn_is(
    uvicorn_calls: list[dict[str, Any]], tmp_path: Path, flags: list[str], log_level: str
):
    result = runner.invoke(app, ["serve", "--data-root", str(tmp_path), *flags])

    assert result.exit_code == 0, result.output
    assert uvicorn_calls[0]["log_level"] == log_level


def test_serve_falls_back_to_the_data_root_environment_variable(
    uvicorn_calls: list[dict[str, Any]], tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("PAPERFACTS_DATA_ROOT", str(tmp_path / "from-env"))

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0, result.output
    assert uvicorn_calls[0]["app"].state.settings.data_root == tmp_path / "from-env"


def test_serve_is_listed_in_the_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.output


def test_serve_refuses_offline_replay(uvicorn_calls: list[dict[str, Any]], tmp_path: Path, monkeypatch):
    # A server under replay would fail every upload and pile its misses into a record no job reports.
    monkeypatch.setenv("PAPERFACTS_LLM_OFFLINE", "1")

    result = runner.invoke(app, ["serve", "--data-root", str(tmp_path)])

    assert result.exit_code == 1
    assert "offline replay is for `run` and `batch`" in result.output
    assert uvicorn_calls == []
