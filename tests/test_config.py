"""Runtime configuration: environment variables only, prefixed ``PAPERFACTS_``.

Misreading configuration tends to fail quietly -- an ``*_url`` gets mistaken for "set" and a development
machine that should run a subprocess instead tries to reach a service that does not exist. So every detail
is pinned here: the prefix, what an empty string means, trailing slashes, defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paperfacts.config import ENV_PREFIX, Settings
from paperfacts.errors import ConfigError


def test_defaults_point_at_local_data_and_subprocess_runners():
    settings = Settings.from_env({})

    assert settings.data_root == Path("data")
    assert settings.uv_bin == "uv"
    assert settings.mineru_url is None
    assert settings.paddle_url is None
    assert settings.paddle_render_dpi == 200


def test_repo_root_defaults_to_the_repository_containing_the_package():
    settings = Settings.from_env({})

    assert (settings.repo_root / "src" / "paperfacts" / "config.py").is_file()


def test_every_setting_is_read_from_the_prefixed_variable():
    env = {
        f"{ENV_PREFIX}DATA_ROOT": "/srv/pf-data",
        f"{ENV_PREFIX}REPO_ROOT": "/srv/pf-repo",
        f"{ENV_PREFIX}UV_BIN": "/opt/bin/uv",
        f"{ENV_PREFIX}MINERU_URL": "http://gpu01:8000",
        f"{ENV_PREFIX}PADDLE_URL": "http://gpu01:8080",
        f"{ENV_PREFIX}PADDLE_RENDER_DPI": "144",
    }

    settings = Settings.from_env(env)

    assert settings.data_root == Path("/srv/pf-data")
    assert settings.repo_root == Path("/srv/pf-repo")
    assert settings.uv_bin == "/opt/bin/uv"
    assert settings.mineru_url == "http://gpu01:8000"
    assert settings.paddle_url == "http://gpu01:8080"
    assert settings.paddle_render_dpi == 144


def test_unprefixed_variables_are_ignored():
    # Guards against a generic name like DATA_ROOT accidentally hijacking the configuration.
    settings = Settings.from_env({"DATA_ROOT": "/wrong", "MINERU_URL": "http://wrong"})

    assert settings.data_root == Path("data")
    assert settings.mineru_url is None


@pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
def test_a_blank_value_counts_as_unset(raw):
    # A blank string must count as "unset", or `export PAPERFACTS_MINERU_URL=` would send the main
    # package off to connect to an empty address.
    settings = Settings.from_env({f"{ENV_PREFIX}MINERU_URL": raw, f"{ENV_PREFIX}DATA_ROOT": raw})

    assert settings.mineru_url is None
    assert settings.data_root == Path("data")


def test_urls_pass_through_untouched_except_for_surrounding_whitespace():
    # Normalising a trailing slash is the HTTP parser's job (the real boundary); Settings only reads the
    # environment variable and trims whitespace.
    settings = Settings.from_env({f"{ENV_PREFIX}MINERU_URL": "  http://gpu01:8000/  "})

    assert settings.mineru_url == "http://gpu01:8000/"


def test_timeouts_have_generous_defaults_and_are_read_as_floats():
    defaults = Settings.from_env({})
    custom = Settings.from_env({f"{ENV_PREFIX}SUBPROCESS_TIMEOUT_S": "120", f"{ENV_PREFIX}HTTP_TIMEOUT_S": "30.5"})

    assert defaults.subprocess_timeout_s >= 1800 and defaults.http_timeout_s >= 600
    assert (custom.subprocess_timeout_s, custom.http_timeout_s) == (120.0, 30.5)


def test_render_dpi_is_parsed_as_an_integer():
    settings = Settings.from_env({f"{ENV_PREFIX}PADDLE_RENDER_DPI": " 300 "})

    assert settings.paddle_render_dpi == 300
    assert isinstance(settings.paddle_render_dpi, int)


def test_a_non_numeric_render_dpi_fails_loudly():
    # Silently falling back to 200 would leave the pixel coordinates inconsistent with the runner side,
    # which is an invisible mis-drawn box, not a loud failure.
    with pytest.raises(ConfigError, match="PADDLE_RENDER_DPI"):
        Settings.from_env({f"{ENV_PREFIX}PADDLE_RENDER_DPI": "high"})


def test_from_env_reads_the_real_process_environment_when_no_mapping_is_given(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}DATA_ROOT", "/from-os-environ")

    assert Settings.from_env().data_root == Path("/from-os-environ")


def test_settings_is_frozen():
    import dataclasses

    settings = Settings.from_env({})

    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.data_root = Path("/elsewhere")


# ---- LLM configuration (M2) ----------------------------------------------------------


def test_llm_defaults_point_at_deepseek():
    settings = Settings.from_env({})

    assert settings.llm_base_url == "https://api.deepseek.com"
    assert settings.llm_model == "deepseek-chat"
    assert settings.llm_timeout_s >= 60


def test_the_llm_base_url_loses_its_trailing_slash():
    # The client appends "/chat/completions"; without stripping the slash that becomes ".../chat/completions"
    # with a doubled slash.
    settings = Settings.from_env({f"{ENV_PREFIX}LLM_BASE_URL": "https://api.example.com/v1/"})

    assert settings.llm_base_url == "https://api.example.com/v1"


def test_the_llm_model_and_timeout_are_read_from_the_environment():
    env = {f"{ENV_PREFIX}LLM_MODEL": "deepseek-reasoner", f"{ENV_PREFIX}LLM_TIMEOUT_S": "45.5"}

    settings = Settings.from_env(env)

    assert settings.llm_model == "deepseek-reasoner"
    assert settings.llm_timeout_s == 45.5


def test_a_non_numeric_llm_timeout_fails_loudly():
    with pytest.raises(ConfigError, match="LLM_TIMEOUT_S"):
        Settings.from_env({f"{ENV_PREFIX}LLM_TIMEOUT_S": "soon"})


# ---- Context budget and self-consistency passes (M2) ----------------------------------


def test_the_llm_context_tokens_default_matches_the_extractor_default():
    # Settings and the extractor must agree on the default context window, or a caller who never touches
    # this setting still silently gets a different budget than extract_lane()'s own default.
    from paperfacts.extraction.extractor import DEFAULT_CONTEXT_TOKENS

    settings = Settings.from_env({})

    assert settings.llm_context_tokens == DEFAULT_CONTEXT_TOKENS


def test_the_llm_context_tokens_are_read_from_the_environment():
    settings = Settings.from_env({f"{ENV_PREFIX}LLM_CONTEXT_TOKENS": "32000"})

    assert settings.llm_context_tokens == 32000
    assert isinstance(settings.llm_context_tokens, int)


def test_a_non_numeric_llm_context_tokens_fails_loudly():
    # A silent fallback here would let an oversized paper through the budget guard instead of stopping
    # before an LLM call is made.
    with pytest.raises(ConfigError, match="LLM_CONTEXT_TOKENS"):
        Settings.from_env({f"{ENV_PREFIX}LLM_CONTEXT_TOKENS": "lots"})


def test_the_extraction_passes_default_to_one():
    # One pass = one LLM call per lane; self-consistency voting is opt-in, never a surprise cost.
    settings = Settings.from_env({})

    assert settings.extraction_passes == 1


def test_the_extraction_passes_are_read_from_the_environment():
    settings = Settings.from_env({f"{ENV_PREFIX}EXTRACTION_PASSES": "3"})

    assert settings.extraction_passes == 3
    assert isinstance(settings.extraction_passes, int)


def test_a_non_numeric_extraction_passes_fails_loudly():
    with pytest.raises(ConfigError, match="EXTRACTION_PASSES"):
        Settings.from_env({f"{ENV_PREFIX}EXTRACTION_PASSES": "many"})


# ---- Key source (M2) ------------------------------------------------------------------
# from_env is a pure function: it only reads environ, never stats the key file. Every case here points
# REPO_ROOT at tmp_path, so none of them accidentally reads the real deepseek_api_key in the repo root.


def env_with_repo(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {f"{ENV_PREFIX}REPO_ROOT": str(tmp_path), **extra}


def test_the_prefixed_variable_wins_over_everything(tmp_path: Path):
    env = env_with_repo(tmp_path, **{f"{ENV_PREFIX}LLM_API_KEY": "sk-prefixed", "DEEPSEEK_API_KEY": "sk-deepseek"})
    (tmp_path / "deepseek_api_key").write_text("sk-from-file", encoding="utf-8")

    assert Settings.from_env(env).require_llm_api_key() == "sk-prefixed"


def test_deepseek_api_key_is_the_second_choice(tmp_path: Path):
    env = env_with_repo(tmp_path, DEEPSEEK_API_KEY="sk-deepseek")
    (tmp_path / "deepseek_api_key").write_text("sk-from-file", encoding="utf-8")

    assert Settings.from_env(env).require_llm_api_key() == "sk-deepseek"


def test_the_key_file_variable_is_the_third_choice(tmp_path: Path):
    key_file = tmp_path / "custom.key"
    key_file.write_text("sk-from-custom-file\n", encoding="utf-8")
    (tmp_path / "deepseek_api_key").write_text("sk-from-default-file", encoding="utf-8")
    env = env_with_repo(tmp_path, **{f"{ENV_PREFIX}LLM_API_KEY_FILE": str(key_file)})

    settings = Settings.from_env(env)

    assert settings.llm_api_key_file == key_file
    assert settings.require_llm_api_key() == "sk-from-custom-file"


def test_the_repo_root_key_file_is_the_last_resort(tmp_path: Path):
    (tmp_path / "deepseek_api_key").write_text("  sk-from-default-file  \n", encoding="utf-8")

    settings = Settings.from_env(env_with_repo(tmp_path))

    assert settings.llm_api_key_file == tmp_path / "deepseek_api_key"
    assert settings.require_llm_api_key() == "sk-from-default-file"


def test_from_env_never_touches_the_key_file(tmp_path: Path):
    """``from_env`` only records the path, never reads its content -- commands like parse / overlay that
    never touch the LLM should not stat the key file, and unit tests should not behave differently just
    because the machine they run on happens to have a key file lying around.
    """
    (tmp_path / "deepseek_api_key").write_text("sk-from-file", encoding="utf-8")

    settings = Settings.from_env(env_with_repo(tmp_path))

    assert settings.llm_api_key is None
    assert settings.llm_api_key_file == tmp_path / "deepseek_api_key"


def test_require_llm_api_key_says_where_to_put_the_key_when_there_is_none(tmp_path: Path):
    # Letting a request go out with an empty key and hit a 401 is one of the hardest failures to trace.
    settings = Settings.from_env(env_with_repo(tmp_path))

    with pytest.raises(ConfigError, match="PAPERFACTS_LLM_API_KEY / DEEPSEEK_API_KEY") as excinfo:
        settings.require_llm_api_key()

    assert str(tmp_path / "deepseek_api_key") in str(excinfo.value)


def test_an_empty_key_file_counts_as_no_key(tmp_path: Path):
    (tmp_path / "deepseek_api_key").write_text("   \n", encoding="utf-8")

    with pytest.raises(ConfigError):
        Settings.from_env(env_with_repo(tmp_path)).require_llm_api_key()


def test_the_api_key_stays_out_of_the_repr():
    # No logger.debug("%s", settings) call should ever end up writing sk-... to a log.
    settings = Settings(llm_api_key="sk-super-secret")

    assert "sk-super-secret" not in repr(settings)


# ---- PaddleOCR-VL's external VLM service (M2) ------------------------------------------


def test_the_paddle_vl_variables_are_read():
    env = {
        f"{ENV_PREFIX}PADDLE_VL_BACKEND": "mlx",
        f"{ENV_PREFIX}PADDLE_VL_SERVER_URL": "  http://127.0.0.1:8000/v1  ",
        f"{ENV_PREFIX}PADDLE_VL_MODEL_NAME": "PaddleOCR-VL",
    }

    settings = Settings.from_env(env)

    assert settings.paddle_vl_backend == "mlx"
    assert settings.paddle_vl_server_url == "http://127.0.0.1:8000/v1"
    assert settings.paddle_vl_model_name == "PaddleOCR-VL"


def test_the_paddle_vl_variables_default_to_unset():
    settings = Settings.from_env({})

    assert settings.paddle_vl_backend is None
    assert settings.paddle_vl_server_url is None
    assert settings.paddle_vl_model_name is None
