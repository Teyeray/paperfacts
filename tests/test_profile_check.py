"""Checking a pasted profile (``POST /api/profile-check``): untrusted text, so it is size-capped as it arrives,
checked in a limited child process, never cached, stored or used as a path, and refused cross-origin like any
request that could spend server time.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import json
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from paperfacts import profile_loader
from paperfacts.config import Settings
from paperfacts.errors import ConfigError, ProfileCheckError
from paperfacts.profile_check import MAX_CHECK_BYTES, check_profile, nesting_exceeds, run_check
from paperfacts.profile_loader import MAX_SPELLING_LENGTH, parse_profile
from paperfacts.profile_view import profile_definition, prompt_sections
from paperfacts.web.app import create_app
from paperfacts.web.jobs import JobManager
from paperfacts.workflow import stage_names
from support.profiles import SHIPPED_PROFILE_PATH, shipped_profile
from support.web import RecordingRunner


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    profiles = tmp_path / "repo" / "profiles"
    profiles.mkdir(parents=True)
    for name in ("tco", "catalysis"):
        shutil.copy(SHIPPED_PROFILE_PATH.parent / f"{name}.json", profiles / f"{name}.json")
    return tmp_path / "repo"


@pytest.fixture
def settings(repo: Path) -> Settings:
    return Settings(data_root=repo.parent / "data", repo_root=repo, profile="tco", llm_model="fake-model")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings, jobs=JobManager(RecordingRunner(), stage_names()))) as test_client:
        yield test_client


def tco_data() -> dict[str, Any]:
    return json.loads(SHIPPED_PROFILE_PATH.read_text(encoding="utf-8"))


def post(client: TestClient, body: bytes | str | dict[str, Any], **kwargs: Any) -> Any:
    content = json.dumps(body, ensure_ascii=False) if isinstance(body, dict) else body
    return client.post("/api/profile-check", content=content, **kwargs)


# ---- the answer ------------------------------------------------------------------------------------------


def test_a_valid_profile_gets_its_definition_and_the_prompts_paperfacts_prompts_prints(client: TestClient):
    response = post(client, SHIPPED_PROFILE_PATH.read_bytes())

    assert response.status_code == 200
    body = response.json()
    profile = shipped_profile()
    assert body["ok"] is True and body["errors"] == []
    assert body["definition"] == json.loads(json.dumps(profile_definition(profile), ensure_ascii=False))
    assert body["prompts"] == [{"title": title, "text": text} for title, text in prompt_sections(profile)]
    assert body["same_name"] == {"name": "tco", "same_content_hash": True}


def test_a_display_only_edit_keeps_the_content_hash_and_a_prompt_edit_does_not(client: TestClient):
    display = tco_data()
    display["title_zh"] = "改过的标题"
    prompt = tco_data()
    prompt["fields"][0]["description"] += " (edited)"

    assert post(client, display).json()["same_name"] == {"name": "tco", "same_content_hash": True}
    assert post(client, prompt).json()["same_name"] == {"name": "tco", "same_content_hash": False}


def test_one_field_s_question_is_previewed_on_request(client: TestClient):
    name = shipped_profile().fields[0].name
    body = client.post(f"/api/profile-check?field={name}", content=SHIPPED_PROFILE_PATH.read_bytes()).json()

    assert body["prompts"] == [
        {"title": title, "text": text} for title, text in prompt_sections(shipped_profile(), name)
    ]


def test_every_validation_error_is_its_own_line_named_by_the_pasted_name_never_a_path(client: TestClient):
    data = tco_data()
    data["name"] = "draft"
    data["format"] = 2
    data["fields"][0]["kind"] = "nonsense"

    body = post(client, data).json()

    assert body["ok"] is False and body["definition"] is None and body["prompts"] is None
    assert len(body["errors"]) >= 2
    assert all(line.startswith("draft.json") for line in body["errors"])
    assert not any(str(Path.cwd()) in line or str(SHIPPED_PROFILE_PATH.parent) in line for line in body["errors"])


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b"{", "not valid JSON"),
        (b"[1, 2]", "must hold a JSON object"),
        (b"\xff\xfe", "not UTF-8"),
        (b'{"name": "x", "format": NaN}', "NaN"),
        (b'{"name": "x", "format": Infinity}', "Infinity"),
        (b'{"name": "x", "format": 1e999}', "finite"),
        (b"[" * 100_000, "nests"),
        (b'{"a": ' * 70 + b"1" + b"}" * 70, "nests"),
    ],
)
def test_malformed_text_is_an_error_line_not_a_server_error(client: TestClient, body: bytes, expected: str):
    response = post(client, body)

    assert response.status_code == 200
    result = response.json()
    assert result["ok"] is False
    assert any(expected in line for line in result["errors"])


def test_the_reserved_name_is_refused_like_the_cli_refuses_it():
    data = tco_data()
    data["name"] = "paperfacts"

    result = check_profile(json.dumps(data).encode(), served={}, extraction_mode="passage")

    assert result["ok"] is False
    assert result["errors"] == ["paperfacts.json: the profile name 'paperfacts' is reserved for old exports"]


def test_a_backtracking_retrieval_pattern_is_reported(client: TestClient):
    data = tco_data()
    data["retrieval"]["condition_unit_pattern"] = "(a+)+"

    errors = post(client, data).json()["errors"]

    assert any("condition_unit_pattern" in line for line in errors)


def test_an_entity_profile_under_document_mode_is_valid_with_a_note(repo: Path):
    catalysis = (repo / "profiles" / "catalysis.json").read_bytes()

    result = check_profile(catalysis, served={}, extraction_mode="document")

    assert result["ok"] is True
    assert any("passage mode" in note for note in result["notes"])
    assert any("catalysis.json" in note and "extraction.mode" in note for note in result["notes"])


def test_nesting_is_measured_outside_strings():
    assert not nesting_exceeds('{"a": "[[[[[[[[[[[["}', 3)
    assert not nesting_exceeds('{"a": "\\"[[[[[[["}', 3)
    assert nesting_exceeds('{"a": [[[1]]]}', 3)


# ---- what the text may cost ------------------------------------------------------------------------------


def test_a_body_declaring_more_than_the_cap_is_refused_unread(client: TestClient):
    response = post(client, b"x", headers={"content-length": str(MAX_CHECK_BYTES + 1)})

    assert response.status_code == 413


def test_a_chunked_body_is_refused_once_its_bytes_pass_the_cap(client: TestClient):
    def chunks() -> Iterator[bytes]:
        for _ in range(MAX_CHECK_BYTES // 4096 + 2):
            yield b" " * 4096

    response = client.post("/api/profile-check", content=chunks())

    assert response.status_code == 413


def test_a_long_unit_spelling_is_refused_before_it_is_folded(client: TestClient):
    """``text._HTML_SUB`` backtracks cubically in a run of spaces: "<sub>" + 8000 spaces took 87 s in-process."""
    data = tco_data()
    data["units"] = {"ohm": {"aliases": {"ohm": 1, "<sub>" + " " * 8000: 2}}}
    started = time.monotonic()

    body = post(client, data).json()

    assert time.monotonic() - started < 5
    assert any(f"longer than {MAX_SPELLING_LENGTH}" in line for line in body["errors"])


def test_a_check_that_runs_out_of_time_is_killed_and_answered_422(client: TestClient, monkeypatch):
    monkeypatch.setattr("paperfacts.web.app.run_check", functools.partial(run_check, timeout=0.001))

    response = post(client, SHIPPED_PROFILE_PATH.read_bytes())

    assert response.status_code == 422
    assert "检查超时" in response.json()["detail"]


def test_a_child_that_fails_is_not_a_result(monkeypatch):
    monkeypatch.setattr("paperfacts.profile_check._CHILD", "import sys; sys.exit(3)")

    with pytest.raises(ProfileCheckError, match="检查未能完成"):
        run_check(b"{}", served={}, extraction_mode="passage")


def test_more_than_two_checks_at_once_are_refused(client: TestClient):
    slots = client.app.state.check_slots
    assert slots.acquire(blocking=False) and slots.acquire(blocking=False)
    try:
        response = post(client, SHIPPED_PROFILE_PATH.read_bytes())
    finally:
        slots.release()
        slots.release()

    assert response.status_code == 429
    assert post(client, SHIPPED_PROFILE_PATH.read_bytes()).status_code == 200


def _cache_sizes() -> dict[str, int]:
    """The size of every functools cache in the package, by qualified name."""
    sizes: dict[str, int] = {}
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("paperfacts") or module is None:
            continue
        candidates = list(vars(module).items())
        for owner_name, owner in list(candidates):
            if isinstance(owner, type) and owner.__module__ == module_name:
                candidates += [(f"{owner_name}.{name}", value) for name, value in vars(owner).items()]
        for name, value in candidates:
            value = value.__func__ if isinstance(value, staticmethod | classmethod) else value
            if hasattr(value, "cache_info") and getattr(value, "__module__", None) == module_name:
                sizes[f"{module_name}.{name}"] = value.cache_info().currsize
    return sizes


def distinct_unit_profile(n: int) -> dict[str, Any]:
    """A valid profile with a unit of its own that a field is measured in, so validating it compiles that unit's
    retrieval pattern (``units._compiled``, an unbounded cache) -- in whichever process validates it."""
    data = tco_data()
    data["units"] = {f"u{n}": {"aliases": {f"u{n}": 1, f"x{n}u": 2}}}
    numeric = next(entry for entry in data["fields"] if entry.get("canonical_unit"))
    numeric["canonical_unit"] = f"u{n}"
    numeric.pop("valid_range", None)
    return data


def test_checks_grow_no_cache_in_the_server_process(client: TestClient):
    client.get("/api/profiles")
    before = _cache_sizes()
    sha_before = dict(profile_loader._FILE_SHA256)
    assert "paperfacts.units._compiled" in before and "paperfacts.keys.extraction_code_fingerprint" in before

    for n in range(10):
        assert post(client, distinct_unit_profile(n)).json()["ok"] is True
        post(client, SHIPPED_PROFILE_PATH.read_bytes())

    assert _cache_sizes() == before
    assert profile_loader._FILE_SHA256 == sha_before


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns) for path in root.rglob("*")}


def test_a_check_writes_nothing(client: TestClient, settings: Settings):
    watched = (settings.data_root, settings.repo_root / "profiles")
    before = [_tree(root) for root in watched]
    data = tco_data()
    data["name"] = "brand_new"

    assert post(client, data).json()["ok"] is True
    post(client, b"{")

    assert [_tree(root) for root in watched] == before
    assert not (settings.repo_root / "profiles" / "brand_new.json").exists()


# ---- the edge ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("headers", [{"Origin": "https://evil.example"}, {"Sec-Fetch-Site": "cross-site"}])
def test_a_cross_origin_check_is_refused(client: TestClient, headers: dict[str, str]):
    assert post(client, SHIPPED_PROFILE_PATH.read_bytes(), headers=headers).status_code == 403


def test_a_check_needs_the_login(settings: Settings):
    app = create_app(
        dataclasses.replace(settings, web_password="pw"), jobs=JobManager(RecordingRunner(), stage_names())
    )
    with TestClient(app) as client:
        assert post(client, SHIPPED_PROFILE_PATH.read_bytes()).status_code == 401
        assert post(client, SHIPPED_PROFILE_PATH.read_bytes(), auth=("paperfacts", "pw")).status_code != 401


# ---- the loader's own bounds -----------------------------------------------------------------------------


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), float("-inf")])
def test_the_loader_refuses_a_non_finite_valid_range(bound: float):
    data = tco_data()
    numeric = next(entry for entry in data["fields"] if entry.get("canonical_unit"))
    numeric["valid_range"] = {"max": bound}

    with pytest.raises(ConfigError, match=r"valid_range\.max must be a finite number"):
        parse_profile(data, Path("tco.json"))


@pytest.mark.parametrize("key", ["rel_tol", "abs_tol"])
def test_the_loader_refuses_a_non_finite_tolerance(key: str):
    data = tco_data()
    numeric = next(entry for entry in data["fields"] if entry.get("canonical_unit"))
    numeric[key] = float("nan")

    with pytest.raises(ConfigError, match=f"{key} must be a finite number"):
        parse_profile(data, Path("tco.json"))


@pytest.mark.parametrize(
    "units",
    [
        {"u" * (MAX_SPELLING_LENGTH + 1): {"aliases": {"u" * (MAX_SPELLING_LENGTH + 1): 1}}},
        {"u": {"aliases": {"u": 1, "v" * (MAX_SPELLING_LENGTH + 1): 2}}},
    ],
)
def test_the_loader_caps_unit_spellings(units: dict[str, Any]):
    data = tco_data()
    data["units"] = units

    with pytest.raises(ConfigError, match=str(MAX_SPELLING_LENGTH)):
        parse_profile(data, Path("tco.json"))


def test_the_loader_caps_ignored_suffixes():
    data = copy.deepcopy(tco_data())
    data["ignored_unit_suffixes"] = ["x" * (MAX_SPELLING_LENGTH + 1)]

    with pytest.raises(ConfigError, match="ignored_unit_suffixes"):
        parse_profile(data, Path("tco.json"))


def test_every_shipped_profile_is_within_the_spelling_cap():
    for path in sorted(SHIPPED_PROFILE_PATH.parent.glob("*.json")):
        parse_profile(json.loads(path.read_text(encoding="utf-8")), path)
