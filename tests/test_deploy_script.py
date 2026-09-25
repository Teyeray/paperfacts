"""``scripts/deploy.sh`` reads its settings the way the service does.

The script only runs on the Linux host, so its helpers are lifted out of it by name and run in bash against a
temporary ``.env`` and ``config.json``. A value the script reads differently from ``config.py`` makes it check
the wrong lane or log in as the wrong user.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "deploy.sh"
TEXT = SCRIPT.read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def function(name: str) -> str:
    """One top-level function of the script, from its ``name() {`` line to the closing brace in column 0."""
    match = re.search(rf"^{name}\(\) {{.*?^}}$", TEXT, re.MULTILINE | re.DOTALL)
    assert match is not None, f"deploy.sh has no function {name}"
    return match.group(0)


def setting(tmp_path: Path, env: str, config: dict, key: str, path: str) -> str:
    (tmp_path / ".env").write_text(env, encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    helpers = "\n".join(function(name) for name in ("env_value", "config_value", "setting"))
    result = subprocess.run(
        ["bash", "-c", f'{helpers}\nsetting "$1" "$2"', "deploy", key, path],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_a_quoted_value_in_the_env_file_is_read_without_its_quotes(tmp_path: Path):
    value = setting(
        tmp_path,
        'PAPERFACTS_PADDLE_VL_BACKEND="vllm-server"\n',
        {"parsers": {"paddle_vl_backend": None}},
        "PAPERFACTS_PADDLE_VL_BACKEND",
        "parsers.paddle_vl_backend",
    )

    assert value == "vllm-server"


def test_a_setting_absent_from_the_env_file_comes_from_config_json(tmp_path: Path):
    value = setting(
        tmp_path,
        "PAPERFACTS_WEB_PASSWORD=secret\n",
        {"parsers": {"paddle_vl_server_url": "http://127.0.0.1:8110/v1"}},
        "PAPERFACTS_PADDLE_VL_SERVER_URL",
        "parsers.paddle_vl_server_url",
    )

    assert value == "http://127.0.0.1:8110/v1"


def test_the_env_file_overrides_config_json(tmp_path: Path):
    # As in config.py: a PAPERFACTS_WEB_USERNAME override is the user the service checks.
    value = setting(
        tmp_path,
        "PAPERFACTS_WEB_USERNAME='alice'\n",
        {"web": {"username": "paperfacts"}},
        "PAPERFACTS_WEB_USERNAME",
        "web.username",
    )

    assert value == "alice"


def test_every_env_file_read_goes_through_the_dotenv_helper():
    by_hand = [line for line in TEXT.splitlines() if ".env" in line and re.search(r"\b(sed|grep|awk|cut)\b", line)]

    assert by_hand == []
    assert 'WEB_USERNAME="$(setting PAPERFACTS_WEB_USERNAME web.username)"' in TEXT


def test_the_web_user_falls_back_to_the_services_default():
    # config.py serves as "paperfacts" when neither .env nor config.json names a user; an empty user here
    # would answer every call with 401.
    assert '[ -n "$WEB_USERNAME" ] || WEB_USERNAME=paperfacts' in TEXT
