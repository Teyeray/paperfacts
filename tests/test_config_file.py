"""``config.json``: the file that holds everything about a run except the secrets.

Two failure modes are worth more than the rest, and both are silent. A mistyped key that falls back to a
default runs a different pipeline than the one asked for and files the results under that pipeline's cache
key, where nothing downstream can notice. And the shipped file drifting away from the built-in constants
would renumber every cache key at once, because :mod:`paperfacts.keys` treats those constants as "nobody
edited anything". So every error here is pinned to name the key and the file, and the shipped file is
checked against the baselines it is supposed to mirror.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from paperfacts.config import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_LLM_REASONING_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAY_DPI,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_S,
    DEFAULT_TEMPERATURE,
    ENV_CONFIG_PATH,
    ENV_PREFIX,
    ConfigDocument,
    Settings,
    config_path,
    load_config,
)
from paperfacts.errors import ConfigError
from paperfacts.fields import FIELD_SPECS, load_condition_keywords, load_field_specs

SHIPPED = config_path({})

# A field entry with only the keys that have no default, used to check what the optional ones fall back to.
MINIMAL_FIELD: dict[str, Any] = {
    "name": "thickness",
    "group": "film",
    "kind": "numeric",
    "description": "Film thickness.",
    "keywords": ["thickness"],
}


def document(data: Mapping[str, Any]) -> ConfigDocument:
    """A document that was never on disk; the path is only ever used in error messages."""
    return ConfigDocument(data=data, path=Path("config.json"))


def write_config(path: Path, changes: Mapping[str, Any] | None = None) -> Path:
    """The shipped configuration with some dotted keys replaced, written where a test can point at it.

    Starting from the real file rather than a hand-built one means a test keeps working when a setting is
    added, and keeps testing the shape the package actually ships.
    """
    data = json.loads(SHIPPED.read_text(encoding="utf-8"))
    for dotted, value in (changes or {}).items():
        *parents, leaf = dotted.split(".")
        node = data
        for part in parents:
            node = node[part]
        node[leaf] = value
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def env_for(path: Path, **extra: str) -> dict[str, str]:
    return {ENV_CONFIG_PATH: str(path), **extra}


# ---- Reading one setting ---------------------------------------------------------------


def test_a_dotted_key_reads_a_nested_value():
    config = document({"llm": {"model": "deepseek-chat"}})

    assert config.get("llm.model", str) == "deepseek-chat"


def test_a_missing_key_names_itself():
    config = document({"llm": {"model": "deepseek-chat"}})

    with pytest.raises(ConfigError, match=re.escape("llm.timeout_s")):
        config.get("llm.timeout_s", float)


def test_a_key_under_a_missing_section_names_the_whole_path():
    config = document({})

    with pytest.raises(ConfigError, match=re.escape("server.page_dpi.default")):
        config.get("server.page_dpi.default", int)


def test_a_wrong_type_names_the_key_the_expected_type_and_the_value():
    # The three things needed to fix it without opening the source: where, what was wanted, what was found.
    config = document({"server": {"port": "8000"}})

    with pytest.raises(ConfigError) as excinfo:
        config.get("server.port", int)

    message = str(excinfo.value)
    assert "server.port" in message and "int" in message and "'8000'" in message


def test_an_integer_is_accepted_where_a_float_is_wanted():
    # 300 is a perfectly ordinary way to write 300.0, and JSON has one number type anyway.
    config = document({"llm": {"timeout_s": 300}})

    value = config.get("llm.timeout_s", float)

    assert value == 300.0
    assert isinstance(value, float)


def test_a_boolean_is_not_accepted_where_an_integer_is_wanted():
    # bool is a subclass of int in Python, so an isinstance check would quietly accept `true` as a port.
    config = document({"server": {"port": True}})

    with pytest.raises(ConfigError, match=re.escape("server.port")):
        config.get("server.port", int)


def test_a_boolean_is_not_accepted_where_a_float_is_wanted():
    config = document({"llm": {"temperature": False}})

    with pytest.raises(ConfigError, match=re.escape("llm.temperature")):
        config.get("llm.temperature", float)


def test_null_reads_as_not_configured():
    # This is how "no parser service here, run the subprocess" is written.
    config = document({"parsers": {"mineru_url": None}})

    assert config.text_or_none("parsers.mineru_url") is None


def test_a_configured_url_comes_back_without_its_surrounding_whitespace():
    config = document({"parsers": {"mineru_url": "  http://gpu01:8000  "}})

    assert config.text_or_none("parsers.mineru_url") == "http://gpu01:8000"


def test_a_blank_string_counts_as_not_configured():
    config = document({"parsers": {"mineru_url": "   "}})

    assert config.text_or_none("parsers.mineru_url") is None


def test_a_non_string_where_a_url_belongs_names_the_key():
    config = document({"parsers": {"mineru_url": 8000}})

    with pytest.raises(ConfigError, match=re.escape("parsers.mineru_url")):
        config.text_or_none("parsers.mineru_url")


def test_a_list_setting_comes_back_as_a_list():
    config = document({"condition_keywords": ["power", "pressure"]})

    assert config.entries("condition_keywords") == ["power", "pressure"]


def test_a_list_setting_that_is_not_a_list_names_the_key_and_what_was_found():
    config = document({"condition_keywords": "power"})

    with pytest.raises(ConfigError) as excinfo:
        config.entries("condition_keywords")

    assert "condition_keywords" in str(excinfo.value) and "str" in str(excinfo.value)


# ---- Finding and parsing the file ------------------------------------------------------


def test_a_missing_configuration_file_names_the_path_and_the_variable_that_moves_it(tmp_path: Path):
    missing = tmp_path / "config.json"

    with pytest.raises(ConfigError) as excinfo:
        load_config(missing)

    assert str(missing) in str(excinfo.value)
    assert ENV_CONFIG_PATH in str(excinfo.value)


def test_invalid_json_names_the_file(tmp_path: Path):
    broken = tmp_path / "config.json"
    broken.write_text('{"llm": {"model": "deepseek-chat",}}', encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_config(broken)

    assert str(broken) in str(excinfo.value)


def test_a_top_level_array_is_refused(tmp_path: Path):
    # A list would make every dotted lookup fail one at a time; saying so once, up front, is kinder.
    array = tmp_path / "config.json"
    array.write_text('["llm"]', encoding="utf-8")

    with pytest.raises(ConfigError, match="JSON object"):
        load_config(array)


def test_the_configuration_path_follows_the_environment_variable(tmp_path: Path):
    elsewhere = tmp_path / "mounted.json"

    assert config_path({ENV_CONFIG_PATH: str(elsewhere)}) == elsewhere


def test_the_configuration_path_defaults_to_the_repository_root():
    # Next to the repository, not the working directory: running paperfacts from anywhere reads one file.
    path = config_path({})

    assert path.name == "config.json"
    assert path.is_file()
    assert (path.parent / "src" / "paperfacts" / "config.py").is_file()


# ---- Layering: constants, then the file, then the environment --------------------------


def test_settings_take_their_values_from_the_configuration_file(tmp_path: Path):
    path = write_config(
        tmp_path / "config.json",
        {
            "data_root": "/srv/facts",
            "llm.model": "some-other-model",
            "server.port": 9123,
            "extraction.mode": "document",
            "extraction.candidate_limit": 3,
            "parsers.paddle_render_dpi": 144,
            "parsers.mineru_url": "http://gpu01:8000",
            "overlay.dpi": 96,
        },
    )

    settings = Settings.from_env(env_for(path))

    assert settings.data_root == Path("/srv/facts")
    assert settings.llm_model == "some-other-model"
    assert settings.server_port == 9123
    assert settings.extraction_mode == "document"
    assert settings.candidate_limit == 3
    assert settings.paddle_render_dpi == 144
    assert settings.mineru_url == "http://gpu01:8000"
    assert settings.overlay_dpi == 96


def test_an_environment_variable_wins_over_the_file(tmp_path: Path):
    # This is how one machine points at its own services without editing a file everyone shares.
    path = write_config(tmp_path / "config.json", {"llm.model": "from-the-file", "server.port": 9123})

    settings = Settings.from_env(
        env_for(path, **{f"{ENV_PREFIX}LLM_MODEL": "from-the-environment", f"{ENV_PREFIX}SERVER_PORT": "7777"})
    )

    assert settings.llm_model == "from-the-environment"
    assert settings.server_port == 7777


def test_the_upload_limit_is_written_in_megabytes_and_read_in_bytes(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"server.max_upload_mb": 5})

    assert Settings.from_env(env_for(path)).max_upload_bytes == 5 * 1024 * 1024


def test_an_unknown_extraction_mode_in_the_file_names_the_modes_and_the_file(tmp_path: Path):
    # Falling back to the default would run a different extractor than the one asked for and file the
    # results under that extractor's key, which nothing downstream could detect.
    path = write_config(tmp_path / "config.json", {"extraction.mode": "paragraph"})

    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env(env_for(path))

    message = str(excinfo.value)
    assert "document, passage" in message and str(path) in message


def test_the_page_dpi_bounds_come_from_the_file(tmp_path: Path):
    path = write_config(
        tmp_path / "config.json",
        {"server.page_dpi.default": 120, "server.page_dpi.min": 60, "server.page_dpi.max": 200},
    )

    settings = Settings.from_env(env_for(path))

    assert (settings.page_dpi, settings.page_dpi_min, settings.page_dpi_max) == (120, 60, 200)


def test_the_sampling_settings_come_from_the_file(tmp_path: Path):
    # These reach the model, so they belong to the run rather than to the machine running it.
    path = write_config(
        tmp_path / "config.json",
        {"llm.temperature": 0.7, "llm.max_tokens": 4096, "llm.retry_attempts": 2, "llm.retry_backoff_s": 0.5},
    )

    settings = Settings.from_env(env_for(path))

    assert settings.llm_temperature == 0.7
    assert settings.llm_max_tokens == 4096
    assert settings.llm_retry_attempts == 2
    assert settings.llm_retry_backoff_s == 0.5


def test_the_reasoning_effort_comes_from_the_file(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"llm.reasoning_effort": "low"})

    assert Settings.from_env(env_for(path)).llm_reasoning_effort == "low"


def test_a_null_reasoning_effort_leaves_the_parameter_out(tmp_path: Path):
    # null is how the file says "send no reasoning_effort at all", which is the built-in baseline.
    path = write_config(tmp_path / "config.json", {"llm.reasoning_effort": None})

    assert Settings.from_env(env_for(path)).llm_reasoning_effort is DEFAULT_LLM_REASONING_EFFORT


def test_an_unknown_reasoning_effort_names_the_ones_that_exist(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"llm.reasoning_effort": "maximum"})

    with pytest.raises(ConfigError, match=re.escape("llm.reasoning_effort is 'maximum'")) as caught:
        Settings.from_env(env_for(path))
    assert "none, low, medium, high" in str(caught.value)


def test_the_environment_overrides_the_reasoning_effort(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"llm.reasoning_effort": "none"})

    settings = Settings.from_env(env_for(path, PAPERFACTS_LLM_REASONING_EFFORT="high"))

    assert settings.llm_reasoning_effort == "high"


def test_an_empty_reasoning_effort_variable_leaves_the_file_alone(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"llm.reasoning_effort": "low"})

    settings = Settings.from_env(env_for(path, PAPERFACTS_LLM_REASONING_EFFORT=""))

    assert settings.llm_reasoning_effort == "low"


def test_a_missing_setting_in_the_file_fails_loudly(tmp_path: Path):
    # Not a silent fallback: a file that has lost a key is a file someone edited by hand, and the value
    # they deleted is more likely to be wrong than absent.
    data = json.loads(SHIPPED.read_text(encoding="utf-8"))
    del data["llm"]["model"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError, match=re.escape("llm.model")):
        Settings.from_env(env_for(path))


# ---- The field table -------------------------------------------------------------------


def test_a_minimal_field_entry_fills_in_the_optional_keys():
    specs = load_field_specs(document({"fields": [MINIMAL_FIELD]}))

    assert len(specs) == 1
    spec = specs[0]
    assert (spec.name, spec.group, spec.kind) == ("thickness", "film", "numeric")
    assert spec.keywords == ("thickness",)
    assert spec.canonical_unit is None
    assert (spec.rel_tol, spec.abs_tol) == (0.0, 0.0)
    assert spec.condition_hint is None
    assert spec.bare_number == "reject"


def test_every_optional_key_is_read_when_it_is_given():
    entry = MINIMAL_FIELD | {
        "canonical_unit": "nm",
        "rel_tol": 0.05,
        "abs_tol": 1,
        "condition_hint": "wavelength",
        "bare_number": "assume_canonical",
    }

    spec = load_field_specs(document({"fields": [entry]}))[0]

    assert spec.canonical_unit == "nm"
    assert (spec.rel_tol, spec.abs_tol) == (0.05, 1.0)
    assert spec.condition_hint == "wavelength"
    assert spec.bare_number == "assume_canonical"


def test_an_unknown_key_in_a_field_names_the_field_and_the_valid_keys():
    # The likeliest edit is a typo, and "keyword" for "keywords" would otherwise extract nothing.
    entry = MINIMAL_FIELD | {"keyword": ["thickness"]}

    with pytest.raises(ConfigError) as excinfo:
        load_field_specs(document({"fields": [entry]}))

    message = str(excinfo.value)
    assert "thickness" in message and "keyword" in message and "keywords" in message


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        pytest.param({"group": "films"}, "group", id="group"),
        pytest.param({"kind": "number"}, "kind", id="kind"),
        pytest.param({"bare_number": "percent"}, "bare_number", id="bare-number"),
        pytest.param({"bare_number": None}, "bare_number", id="bare-number-null"),
        pytest.param({"rel_tol": "0.05"}, "rel_tol", id="tolerance-as-text"),
        pytest.param({"abs_tol": True}, "abs_tol", id="tolerance-as-boolean"),
        pytest.param({"keywords": "thickness"}, "keywords", id="keywords-not-a-list"),
        pytest.param({"keywords": ["thickness", ""]}, "keywords", id="keywords-with-an-empty-string"),
        pytest.param({"keywords": ["thickness", 7]}, "keywords", id="keywords-with-a-number"),
        pytest.param({"description": "   "}, "description", id="blank-description"),
        pytest.param({"canonical_unit": 5}, "canonical_unit", id="unit-not-a-string"),
    ],
)
def test_an_invalid_field_value_names_the_field_and_the_key(change: dict[str, Any], expected: str):
    entry = MINIMAL_FIELD | change

    with pytest.raises(ConfigError) as excinfo:
        load_field_specs(document({"fields": [entry]}))

    message = str(excinfo.value)
    assert "thickness" in message and expected in message


@pytest.mark.parametrize("name", [None, "", "   ", 7], ids=["missing", "empty", "blank", "not-a-string"])
def test_a_field_without_a_usable_name_is_refused(name: Any):
    entry = MINIMAL_FIELD | {"name": name}
    if name is None:
        del entry["name"]

    with pytest.raises(ConfigError, match="name"):
        load_field_specs(document({"fields": [entry]}))


def test_a_field_without_a_description_is_refused():
    entry = {key: value for key, value in MINIMAL_FIELD.items() if key != "description"}

    with pytest.raises(ConfigError, match="description"):
        load_field_specs(document({"fields": [entry]}))


def test_a_field_that_is_not_an_object_names_its_position():
    with pytest.raises(ConfigError, match=r"fields\[1\]"):
        load_field_specs(document({"fields": [MINIMAL_FIELD, "thickness"]}))


def test_an_empty_field_table_is_refused():
    # An empty table would extract nothing at all, and would do it without a word.
    with pytest.raises(ConfigError, match="nothing to extract"):
        load_field_specs(document({"fields": []}))


def test_two_fields_with_the_same_name_are_refused():
    # FIELD_BY_NAME would keep the last one, so the earlier entry would be configured but never used.
    with pytest.raises(ConfigError, match="thickness"):
        load_field_specs(document({"fields": [MINIMAL_FIELD, MINIMAL_FIELD | {"canonical_unit": "nm"}]}))


@pytest.mark.parametrize("words", [["power", ""], ["power", 7], ["power", None]], ids=["empty", "number", "null"])
def test_condition_keywords_must_all_be_non_empty_strings(words: list[Any]):
    with pytest.raises(ConfigError, match="condition_keywords"):
        load_condition_keywords(document({"condition_keywords": words}))


def test_condition_keywords_come_back_in_the_order_they_were_written():
    words = load_condition_keywords(document({"condition_keywords": ["power", "pressure", "flow rate"]}))

    assert words == ("power", "pressure", "flow rate")


# ---- The shipped file ------------------------------------------------------------------


def test_the_shipped_configuration_is_the_field_table_the_package_exposes():
    # Catches an edit that leaves the file loadable but no longer describing what the package extracts.
    specs = load_field_specs(load_config(SHIPPED))

    assert specs == FIELD_SPECS


def test_the_shipped_configuration_mirrors_the_built_in_baselines():
    """:mod:`paperfacts.keys` mixes these into a cache key only when they differ from the constants below,
    meaning "nobody edited anything". A shipped file that drifted from them would silently rename every
    stored extraction on disk.
    """
    data = json.loads(SHIPPED.read_text(encoding="utf-8"))

    assert data["llm"]["temperature"] == DEFAULT_TEMPERATURE
    assert data["llm"]["max_tokens"] == DEFAULT_MAX_TOKENS
    # The one deliberate exception: the baseline omits `reasoning_effort` (what the code shipped with
    # before the parameter existed), while the file turns reasoning off, because the extraction task is
    # quote-and-cite and the hidden reasoning costs minutes per field question.
    assert data["llm"]["reasoning_effort"] == "none"
    assert DEFAULT_LLM_REASONING_EFFORT is None
    assert data["llm"]["retry_attempts"] == DEFAULT_RETRY_ATTEMPTS
    assert data["llm"]["retry_backoff_s"] == DEFAULT_RETRY_BACKOFF_S
    assert data["extraction"]["candidate_limit"] == DEFAULT_CANDIDATE_LIMIT
    assert data["overlay"]["dpi"] == DEFAULT_OVERLAY_DPI


def test_the_shipped_configuration_agrees_with_the_dataclass_defaults():
    # Every setting the file feeds should read back as the value the code would have used without it, so
    # that reading the constants in config.py tells the truth about an unedited checkout.
    baseline = Settings()
    configured = Settings.from_env({})
    assert configured.llm_reasoning_effort == "none"

    fed_by_the_file = [
        name
        for name in (field.name for field in dataclasses.fields(Settings))
        # repo_root follows the checkout and the two key fields are environment-only, by design.
        # llm_reasoning_effort is the deliberate exception above: the file turns reasoning off, the
        # built-in baseline leaves the parameter out.
        if name not in {"repo_root", "llm_api_key", "llm_api_key_file", "llm_reasoning_effort"}
    ]
    assert [getattr(configured, name) for name in fed_by_the_file] == [
        getattr(baseline, name) for name in fed_by_the_file
    ]
