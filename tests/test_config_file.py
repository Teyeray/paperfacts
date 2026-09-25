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

from paperfacts import keys
from paperfacts.config import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_LLM_CONCURRENCY,
    DEFAULT_LLM_INVENTORY_REASONING_EFFORT,
    DEFAULT_LLM_MAX_IN_FLIGHT,
    DEFAULT_LLM_REASONING_EFFORT,
    DEFAULT_MAX_PARALLEL_DOCUMENTS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAY_DPI,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_S,
    DEFAULT_TEMPERATURE,
    ENV_CONFIG_PATH,
    ENV_PREFIX,
    INHERIT,
    ConfigDocument,
    Settings,
    config_path,
    load_config,
)
from paperfacts.errors import ConfigError
from paperfacts.fields import FieldRole, FieldSpec
from paperfacts.profile import parse_profile
from support.profiles import SHIPPED_PROFILE_PATH, make_profile

SHIPPED = config_path({})

# A field entry with only the keys that have no default, used to check what the optional ones fall back to.
MINIMAL_FIELD: dict[str, Any] = {
    "name": "thickness",
    "group": "film",
    "kind": "numeric",
    "description": "Film thickness.",
    "keywords": ["thickness"],
}


def load_fields(entries: list[Any]) -> tuple[FieldSpec, ...]:
    """``entries`` as the shipped profile's field table, validated the way every field table is: config.json no
    longer has one of its own that anything reads, and the entry rules are the profile's."""
    # No chart slots: none of these entries is figure_readable, and a profile has slots only when one is.
    data = json.loads(SHIPPED_PROFILE_PATH.read_text(encoding="utf-8")) | {"fields": entries, "figures": None}
    return parse_profile(data, Path("profiles/tco.json")).fields


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


def test_the_inventory_reasoning_effort_comes_from_the_file(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"llm.inventory_reasoning_effort": "none"})

    assert Settings.from_env(env_for(path)).llm_inventory_reasoning_effort == "none"


def test_a_null_inventory_reasoning_effort_inherits_the_general_one(tmp_path: Path):
    # null is the baseline, spelled INHERIT in code: the inventory question is sent with whatever
    # llm.reasoning_effort says. The word "inherit" says the same thing out loud.
    path = write_config(tmp_path / "config.json", {"llm.inventory_reasoning_effort": None})

    settings = Settings.from_env(env_for(path))

    assert settings.llm_inventory_reasoning_effort is DEFAULT_LLM_INVENTORY_REASONING_EFFORT is INHERIT
    spelled = write_config(tmp_path / "spelled.json", {"llm.inventory_reasoning_effort": "inherit"})
    assert Settings.from_env(env_for(spelled)).llm_inventory_reasoning_effort is INHERIT


def test_an_omitted_inventory_reasoning_effort_sends_no_parameter(tmp_path: Path):
    # The one meaning null cannot carry: send that question with no reasoning_effort at all, while the
    # client keeps sending its own on every other question.
    path = write_config(tmp_path / "config.json", {"llm.inventory_reasoning_effort": "omit"})

    assert Settings.from_env(env_for(path)).llm_inventory_reasoning_effort is None


def test_an_unknown_inventory_reasoning_effort_names_its_own_key(tmp_path: Path):
    # The error has to name the key that is wrong, not the neighbouring one that shares the parser.
    path = write_config(tmp_path / "config.json", {"llm.inventory_reasoning_effort": "maximum"})

    with pytest.raises(ConfigError, match=re.escape("llm.inventory_reasoning_effort is 'maximum'")) as caught:
        Settings.from_env(env_for(path))
    assert "PAPERFACTS_LLM_INVENTORY_REASONING_EFFORT" in str(caught.value)


def test_the_environment_overrides_the_inventory_reasoning_effort(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"llm.inventory_reasoning_effort": None})

    settings = Settings.from_env(env_for(path, PAPERFACTS_LLM_INVENTORY_REASONING_EFFORT="low"))

    assert settings.llm_inventory_reasoning_effort == "low"


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
    specs = load_fields([MINIMAL_FIELD])

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

    spec = load_fields([entry])[0]

    assert spec.canonical_unit == "nm"
    assert (spec.rel_tol, spec.abs_tol) == (0.05, 1.0)
    assert spec.condition_hint == "wavelength"
    assert spec.bare_number == "assume_canonical"


def test_a_field_label_is_optional_and_read_when_it_is_given():
    assert load_fields([MINIMAL_FIELD])[0].label == ""

    spec = load_fields([MINIMAL_FIELD | {"label": "厚度"}])[0]

    assert spec.label == "厚度"


@pytest.mark.parametrize("bad", ["", "  ", 5, None])
def test_a_label_that_is_not_a_non_empty_string_is_refused(bad):
    with pytest.raises(ConfigError, match="label"):
        load_fields([MINIMAL_FIELD | {"label": bad}])


def test_every_shipped_field_has_a_chinese_label(tco_profile):
    assert all(spec.label for spec in tco_profile.fields)


def keys_under(profile, specs) -> tuple[str, str, str]:
    """The document- and passage-mode extractor keys and the comparison key of ``profile`` with its field table
    replaced by ``specs``."""
    edited = dataclasses.replace(profile, fields=specs)
    return (
        keys.extractor_key(keys.ExtractionOptions(edited, "a-model", mode="document")),
        keys.extractor_key(keys.ExtractionOptions(edited, "a-model", mode="passage")),
        keys.comparison_key_for(Settings(), edited),
    )


def test_a_label_changes_neither_cache_key(tco_profile):
    # It is a column header, nothing more: adding or editing one must not re-extract or re-compare a paper.
    plain = load_fields([MINIMAL_FIELD])
    labelled = load_fields([MINIMAL_FIELD | {"label": "厚度"}])

    assert keys_under(tco_profile, plain) == keys_under(tco_profile, labelled)


def test_a_chinese_description_is_optional_and_read_when_it_is_given():
    assert load_fields([MINIMAL_FIELD])[0].description_zh == ""

    spec = load_fields([MINIMAL_FIELD | {"description_zh": "薄膜厚度"}])[0]

    assert spec.description_zh == "薄膜厚度"


@pytest.mark.parametrize("bad", ["", "  ", 5, None])
def test_a_chinese_description_that_is_not_a_non_empty_string_is_refused(bad):
    with pytest.raises(ConfigError, match="description_zh"):
        load_fields([MINIMAL_FIELD | {"description_zh": bad}])


def test_every_shipped_field_has_a_chinese_description(tco_profile):
    assert all(spec.description_zh for spec in tco_profile.fields)


def test_a_chinese_description_changes_neither_cache_key(tco_profile):
    # Like the label: it reaches a tooltip and a spreadsheet sheet, never a prompt and never a verdict.
    plain = load_fields([MINIMAL_FIELD])
    described = load_fields([MINIMAL_FIELD | {"description_zh": "薄膜厚度"}])

    assert keys_under(tco_profile, plain) == keys_under(tco_profile, described)


def test_an_unknown_key_in_a_field_names_the_field_and_the_valid_keys():
    # The likeliest edit is a typo, and "keyword" for "keywords" would otherwise extract nothing.
    entry = MINIMAL_FIELD | {"keyword": ["thickness"]}

    with pytest.raises(ConfigError) as excinfo:
        load_fields([entry])

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
        pytest.param({"rel_tol": -0.05}, "rel_tol", id="negative-relative-tolerance"),
        pytest.param({"abs_tol": -1}, "abs_tol", id="negative-absolute-tolerance"),
        pytest.param(
            {"canonical_unit": "nm", "bare_number": "percent_or_fraction"}, "bare_number", id="fraction-on-nm"
        ),
        pytest.param({"bare_number": "percent_or_fraction"}, "bare_number", id="fraction-without-a-unit"),
    ],
)
def test_an_invalid_field_value_names_the_field_and_the_key(change: dict[str, Any], expected: str):
    entry = MINIMAL_FIELD | change

    with pytest.raises(ConfigError) as excinfo:
        load_fields([entry])

    message = str(excinfo.value)
    assert "thickness" in message and expected in message


@pytest.mark.parametrize("name", [None, "", "   ", 7], ids=["missing", "empty", "blank", "not-a-string"])
def test_a_field_without_a_usable_name_is_refused(name: Any):
    entry = MINIMAL_FIELD | {"name": name}
    if name is None:
        del entry["name"]

    with pytest.raises(ConfigError, match="name"):
        load_fields([entry])


def test_a_field_without_a_description_is_refused():
    entry = {key: value for key, value in MINIMAL_FIELD.items() if key != "description"}

    with pytest.raises(ConfigError, match="description"):
        load_fields([entry])


def test_a_field_that_is_not_an_object_names_its_position():
    with pytest.raises(ConfigError, match=r"fields\[1\]"):
        load_fields([MINIMAL_FIELD, "thickness"])


def test_an_empty_field_table_is_refused():
    # An empty table would extract nothing at all, and would do it without a word.
    with pytest.raises(ConfigError, match="at least one field"):
        load_fields([])


def test_two_fields_with_the_same_name_are_refused():
    # A table by name would keep the last one, so the earlier entry would be configured but never used.
    with pytest.raises(ConfigError, match="thickness"):
        load_fields([MINIMAL_FIELD, MINIMAL_FIELD | {"canonical_unit": "nm"}])


@pytest.mark.parametrize("words", [["power", ""], ["power", 7], ["power", None]], ids=["empty", "number", "null"])
def test_condition_keywords_must_all_be_non_empty_strings(words: list[Any]):
    with pytest.raises(ConfigError, match="condition_keywords"):
        make_profile({"retrieval.condition_keywords": words})


def test_condition_keywords_come_back_in_the_order_they_were_written():
    words = make_profile({"retrieval.condition_keywords": ["power", "pressure", "flow rate"]})

    assert words.retrieval.condition_keywords == ("power", "pressure", "flow rate")


# ---- The shipped file ------------------------------------------------------------------


def test_the_shipped_configuration_mirrors_the_built_in_baselines():
    """:mod:`paperfacts.keys` mixes these into a cache key only when they differ from the constants below,
    meaning "nobody edited anything". A shipped file that drifted from them would silently rename every
    stored extraction on disk.
    """
    data = json.loads(SHIPPED.read_text(encoding="utf-8"))

    assert data["llm"]["temperature"] == DEFAULT_TEMPERATURE
    assert data["llm"]["max_tokens"] == DEFAULT_MAX_TOKENS
    # Shipped unset on purpose: turning reasoning off was measured to lose a third of the extracted values
    # (.omc/research/reasoning-effort.md), and an unedited checkout must keep its cache keys.
    assert data["llm"]["reasoning_effort"] is DEFAULT_LLM_REASONING_EFFORT is None
    # Shipped unset too: the inventory question inherits the general effort until somebody asks otherwise.
    assert data["llm"]["inventory_reasoning_effort"] is None
    assert DEFAULT_LLM_INVENTORY_REASONING_EFFORT is INHERIT
    assert data["llm"]["concurrency"] == DEFAULT_LLM_CONCURRENCY
    assert data["llm"]["max_in_flight"] == DEFAULT_LLM_MAX_IN_FLIGHT
    assert data["web"]["max_parallel_documents"] == DEFAULT_MAX_PARALLEL_DOCUMENTS
    assert data["llm"]["retry_attempts"] == DEFAULT_RETRY_ATTEMPTS
    assert data["llm"]["retry_backoff_s"] == DEFAULT_RETRY_BACKOFF_S
    assert data["extraction"]["candidate_limit"] == DEFAULT_CANDIDATE_LIMIT
    assert data["overlay"]["dpi"] == DEFAULT_OVERLAY_DPI


def test_the_shipped_configuration_agrees_with_the_dataclass_defaults():
    # Every setting the file feeds should read back as the value the code would have used without it, so
    # that reading the constants in config.py tells the truth about an unedited checkout.
    baseline = Settings()
    configured = Settings.from_env({})
    assert configured.llm_reasoning_effort is DEFAULT_LLM_REASONING_EFFORT

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


# ---- llm.concurrency: how many questions wait at once, never what they say ----------------


def test_the_concurrency_comes_from_the_file(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"llm.concurrency": 9})

    assert Settings.from_env(env_for(path)).llm_concurrency == 9


def test_the_concurrency_environment_variable_wins_over_the_file(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"llm.concurrency": 9})

    settings = Settings.from_env(env_for(path, PAPERFACTS_LLM_CONCURRENCY="2"))

    assert settings.llm_concurrency == 2


def test_a_concurrency_below_one_names_the_key_and_the_file(tmp_path: Path):
    # Zero questions in flight would hang forever inside the pool rather than fail here.
    path = write_config(tmp_path / "config.json", {"llm.concurrency": 0})

    with pytest.raises(ConfigError, match=r"llm\.concurrency must be at least 1"):
        Settings.from_env(env_for(path))


def test_a_file_without_a_concurrency_key_names_the_key_and_the_file(tmp_path: Path):
    """Every setting is declared in config.json, so a missing key is a hand-edited file, not a default."""
    path = write_config(tmp_path / "config.json", {})
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["llm"]["concurrency"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ConfigError, match=r"missing setting 'llm\.concurrency'"):
        Settings.from_env(env_for(path))


def test_a_non_integer_concurrency_still_names_the_key(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"llm.concurrency": "four"})

    with pytest.raises(ConfigError, match=r"llm\.concurrency must be int"):
        Settings.from_env(env_for(path))


# ---- llm.max_in_flight and web.max_parallel_documents: scheduling knobs, validated like counts ----


@pytest.mark.parametrize(
    ("dotted", "variable", "attribute"),
    [
        ("llm.max_in_flight", "LLM_MAX_IN_FLIGHT", "llm_max_in_flight"),
        ("web.max_parallel_documents", "WEB_MAX_PARALLEL_DOCUMENTS", "max_parallel_documents"),
    ],
)
def test_a_parallelism_limit_comes_from_the_file_and_the_environment_wins(
    tmp_path: Path, dotted: str, variable: str, attribute: str
):
    path = write_config(tmp_path / "config.json", {dotted: 5})

    assert getattr(Settings.from_env(env_for(path)), attribute) == 5
    assert getattr(Settings.from_env(env_for(path, **{f"{ENV_PREFIX}{variable}": "2"})), attribute) == 2


@pytest.mark.parametrize("dotted", ["llm.max_in_flight", "web.max_parallel_documents"])
def test_a_parallelism_limit_below_one_names_the_key(tmp_path: Path, dotted: str):
    # Zero would not mean "unlimited": every request, or every job, would wait forever.
    path = write_config(tmp_path / "config.json", {dotted: 0})

    with pytest.raises(ConfigError, match=rf"{re.escape(dotted)} must be at least 1"):
        Settings.from_env(env_for(path))


# ---- valid_range -------------------------------------------------------------------------

RANGED_FIELD: dict[str, Any] = MINIMAL_FIELD | {"canonical_unit": "nm"}


def test_a_field_without_a_range_accepts_every_value():
    spec = load_fields([RANGED_FIELD])[0]

    assert spec.valid_range == (None, None)
    assert spec.describe_range() is None
    assert spec.in_range(1e9)


@pytest.mark.parametrize(
    ("bounds", "described", "inside", "outside"),
    [
        pytest.param({"max": 500}, "at most 500 nm", 500.0, 501.0, id="ceiling"),
        pytest.param({"min": 60}, "at least 60 nm", 60.0, 59.9, id="floor"),
        pytest.param({"min": 1, "max": 2.5}, "between 1 and 2.5 nm", 2.0, 3.0, id="both"),
    ],
)
def test_a_range_is_read_in_the_canonical_unit_with_either_end_open(bounds, described, inside, outside):
    spec = load_fields([RANGED_FIELD | {"valid_range": bounds}])[0]

    assert spec.describe_range() == described
    assert spec.in_range(inside)
    assert not spec.in_range(outside)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(RANGED_FIELD | {"valid_range": [0, 500]}, "object", id="not-an-object"),
        pytest.param(RANGED_FIELD | {"valid_range": {"maximum": 500}}, "object", id="unknown-bound"),
        pytest.param(RANGED_FIELD | {"valid_range": {}}, "at least one", id="empty"),
        pytest.param(RANGED_FIELD | {"valid_range": {"max": "500"}}, "valid_range.max", id="bound-as-text"),
        pytest.param(RANGED_FIELD | {"valid_range": {"max": True}}, "valid_range.max", id="bound-as-boolean"),
        pytest.param(RANGED_FIELD | {"valid_range": {"min": 5, "max": 5}}, "below", id="empty-interval"),
        pytest.param(MINIMAL_FIELD | {"valid_range": {"max": 5}}, "canonical_unit", id="no-unit"),
    ],
)
def test_a_malformed_range_names_the_field_and_the_problem(entry, expected):
    with pytest.raises(ConfigError, match=expected) as excinfo:
        load_fields([entry])

    assert "thickness" in str(excinfo.value)


def test_a_range_moves_both_cache_keys_and_its_absence_moves_neither(tco_profile):
    # It changes what the model is told and which values survive, so it must re-extract; but a table that
    # declares no range has to keep the keys it had before ranges existed.
    plain = load_fields([RANGED_FIELD])
    ranged = load_fields([RANGED_FIELD | {"valid_range": {"max": 500}}])

    everything = (FieldRole.PROMPT, FieldRole.CLEANING, FieldRole.VERDICT)
    assert "valid_range" not in keys._field_material(plain[0], *everything)
    assert "valid_range" in keys._field_material(ranged[0], *everything)
    document_key, passage_key, comparison = keys_under(tco_profile, plain)
    ranged_document, ranged_passage, ranged_comparison = keys_under(tco_profile, ranged)
    assert ranged_document != document_key and ranged_passage != passage_key and ranged_comparison != comparison


# Every FieldSpec cell the extraction key hashes: what the model is told, and what cleaning its answer reads.
EXTRACTION_CELLS = {
    "name",
    "group",
    "kind",
    "description",
    "canonical_unit",
    "condition_hint",
    "bare_number",
    "valid_range",
    "level",
    "condition_rule",
    "range_policy",
}


def test_every_field_cell_is_classified_for_the_cache_keys():
    # A cell nobody classified lands in no key or the wrong one. Its roles (fields.py) decide; the extraction
    # key hashes the PROMPT and CLEANING ones.
    assert set(keys.attributes_with(FieldRole.PROMPT, FieldRole.CLEANING)) == EXTRACTION_CELLS


def test_a_tolerance_moves_only_the_comparison_key(tco_profile):
    # A tolerance decides whether two quoted values agree; the model is never told it and no cleaning rule
    # reads it, so editing one must leave every stored extraction where it is.
    plain = load_fields([RANGED_FIELD])
    tolerant = load_fields([RANGED_FIELD | {"rel_tol": 0.1, "abs_tol": 2}])

    document_key, passage_key, comparison = keys_under(tco_profile, plain)
    document_after, passage_after, comparison_after = keys_under(tco_profile, tolerant)
    assert (document_key, passage_key) == (document_after, passage_after)
    assert comparison != comparison_after


@pytest.mark.parametrize("mode", ["document", "passage"])
def test_the_range_sentence_the_model_reads_is_part_of_the_extractor_key(monkeypatch, mode, tco_profile):
    # The "Plausible values are ..." line is written by FieldSpec.describe_range and reaches every question,
    # in passage mode through the field question's user half, which is not hashed by value there. The
    # rendered field table is in the key by value (the document prompt carries it), so a change in how
    # the range is worded re-keys both modes, not just a change of the range itself.

    before = keys.extractor_key(keys.ExtractionOptions(tco_profile, "a-model", mode=mode))
    monkeypatch.setattr(FieldSpec, "describe_range", lambda self: "no more than a little")

    assert keys.extractor_key(keys.ExtractionOptions(tco_profile, "a-model", mode=mode)) != before


def test_fields_py_is_part_of_the_extraction_code_fingerprint(monkeypatch):
    # It also decides which fields are sample-level, which gates which questions are asked at all.
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(keys, "source_fingerprint", lambda *files: seen.append(files) or "x")
    keys.extraction_code_fingerprint.cache_clear()
    try:
        keys.extraction_code_fingerprint()
    finally:
        keys.extraction_code_fingerprint.cache_clear()

    assert "fields.py" in seen[0]


# ---- condition_preference -----------------------------------------------------------------


def test_a_condition_preference_is_read_in_order():
    spec = load_fields([RANGED_FIELD | {"condition_preference": ["400-800", "550"]}])[0]

    assert spec.condition_preference == ("400-800", "550")


@pytest.mark.parametrize("bad", ["400-800", [""], ["visible"], [550]])
def test_a_condition_preference_must_be_a_list_of_entries_naming_numbers(bad):
    with pytest.raises(ConfigError, match="condition_preference"):
        load_fields([RANGED_FIELD | {"condition_preference": bad}])


def test_a_condition_preference_moves_only_the_comparison_key(tco_profile):
    # It picks a dataset cell among values already extracted; the model never hears of it.
    plain = load_fields([RANGED_FIELD])
    preferring = load_fields([RANGED_FIELD | {"condition_preference": ["550"]}])

    (document_key, passage_key, comparison), after = keys_under(tco_profile, plain), keys_under(tco_profile, preferring)
    assert (document_key, passage_key) == after[:2]
    assert comparison != after[2]


# ---- figures: the opt-in chart-reading stage ------------------------------------------------


def test_figure_reading_ships_switched_off_with_the_measured_model():
    # Opt-in: about a minute of a vision model per chart is not something an upload should pay by surprise.
    settings = Settings.from_env({})

    assert settings.figures_enabled is False
    assert settings.figures_model == "qwen3.7-plus"
    assert settings.figures_timeout_s == 300.0
    assert settings.figures_max_per_document == 12


def test_the_figure_settings_come_from_the_file(tmp_path: Path):
    path = write_config(
        tmp_path / "config.json",
        {
            "figures.enabled": True,
            "figures.model": "qwen3.6-plus",
            "figures.max_per_document": 3,
            "figures.dpi": 150,
            "figures.max_pixels": 1000,
            "figures.timeout_s": 60,
        },
    )

    settings = Settings.from_env(env_for(path))

    assert settings.figures_enabled is True
    assert settings.figures_model == "qwen3.6-plus"
    assert settings.figures_max_per_document == 3
    assert settings.figures_dpi == 150
    assert settings.figures_max_pixels == 1000
    assert settings.figures_timeout_s == 60.0


@pytest.mark.parametrize(
    ("raw", "expected"), [("true", True), ("1", True), ("ON", True), ("false", False), ("0", False)]
)
def test_the_environment_switches_figure_reading(tmp_path: Path, raw: str, expected: bool):
    path = write_config(tmp_path / "config.json", {"figures.enabled": not expected})

    assert Settings.from_env(env_for(path, PAPERFACTS_FIGURES_ENABLED=raw)).figures_enabled is expected


def test_an_unreadable_figures_switch_names_the_variable(tmp_path: Path):
    path = write_config(tmp_path / "config.json")

    with pytest.raises(ConfigError, match="PAPERFACTS_FIGURES_ENABLED"):
        Settings.from_env(env_for(path, PAPERFACTS_FIGURES_ENABLED="ture"))


def test_the_environment_overrides_the_figure_model_and_limits(tmp_path: Path):
    path = write_config(tmp_path / "config.json")

    settings = Settings.from_env(
        env_for(
            path,
            PAPERFACTS_FIGURES_MODEL="other-vl",
            PAPERFACTS_FIGURES_MAX_PER_DOCUMENT="2",
            PAPERFACTS_FIGURES_DPI="100",
            PAPERFACTS_FIGURES_MAX_PIXELS="5000",
            PAPERFACTS_FIGURES_TIMEOUT_S="30",
        )
    )

    assert (settings.figures_model, settings.figures_max_per_document, settings.figures_dpi) == ("other-vl", 2, 100)
    assert (settings.figures_max_pixels, settings.figures_timeout_s) == (5000, 30.0)


def test_a_figure_limit_below_one_names_the_key(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"figures.max_per_document": 0})

    with pytest.raises(ConfigError, match=r"figures\.max_per_document must be at least 1"):
        Settings.from_env(env_for(path))


def test_a_string_where_the_figures_switch_belongs_names_the_key(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"figures.enabled": "yes"})

    with pytest.raises(ConfigError, match=r"figures\.enabled must be bool"):
        Settings.from_env(env_for(path))


def test_a_figures_timeout_of_zero_names_the_key(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"figures.timeout_s": 0})

    with pytest.raises(ConfigError, match=r"figures\.timeout_s must be positive"):
        Settings.from_env(env_for(path))


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_an_ambiguous_match_confidence_outside_0_to_1_names_the_key(tmp_path: Path, value: float):
    # A confidence outside [0, 1] would count every pairing, or none, as low confidence.
    path = write_config(tmp_path / "config.json", {"comparison.ambiguous_match_confidence": value})

    with pytest.raises(ConfigError, match=r"comparison\.ambiguous_match_confidence must be between 0 and 1"):
        Settings.from_env(env_for(path))


# ---- Ranges and cross-checks on the settings -------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        pytest.param({"llm.context_tokens": 0}, "llm.context_tokens", id="context-zero"),
        pytest.param({"llm.max_tokens": 0}, "llm.max_tokens", id="max-tokens-zero"),
        pytest.param(
            {"llm.context_tokens": 60000, "llm.max_tokens": 65536}, "llm.context_tokens", id="reply-fills-window"
        ),
        pytest.param({"llm.temperature": -0.1}, "llm.temperature", id="temperature-negative"),
        pytest.param({"llm.temperature": 2.5}, "llm.temperature", id="temperature-too-high"),
        pytest.param({"llm.timeout_s": 0}, "llm.timeout_s", id="llm-timeout-zero"),
        pytest.param({"parsers.http_timeout_s": -1}, "parsers.http_timeout_s", id="http-timeout-negative"),
        pytest.param({"parsers.subprocess_timeout_s": 0}, "parsers.subprocess_timeout_s", id="subprocess-timeout"),
        pytest.param({"llm.retry_backoff_s": -1}, "llm.retry_backoff_s", id="backoff-negative"),
        pytest.param({"server.port": 0}, "server.port", id="port-zero"),
        pytest.param({"server.port": 70000}, "server.port", id="port-too-high"),
        pytest.param({"server.max_upload_mb": 0}, "server.max_upload_mb", id="upload-zero"),
        pytest.param({"server.page_dpi.min": 0}, "server.page_dpi.min", id="dpi-min-zero"),
        pytest.param({"server.page_dpi.default": 300}, "server.page_dpi.default", id="dpi-above-max"),
        pytest.param({"server.page_dpi.default": 20}, "server.page_dpi.default", id="dpi-below-min"),
        pytest.param({"overlay.dpi": 0}, "overlay.dpi", id="overlay-dpi-zero"),
        pytest.param({"parsers.paddle_render_dpi": 0}, "parsers.paddle_render_dpi", id="render-dpi-zero"),
    ],
)
def test_a_setting_outside_its_range_names_the_key_and_the_file(tmp_path: Path, changes, expected):
    path = write_config(tmp_path / "config.json", changes)

    with pytest.raises(ConfigError) as excinfo:
        Settings.from_env(env_for(path))

    assert expected in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_the_reply_budget_error_names_both_keys(tmp_path: Path):
    path = write_config(tmp_path / "config.json", {"llm.context_tokens": 60000, "llm.max_tokens": 65536})

    with pytest.raises(ConfigError, match=r"llm\.max_tokens .* llm\.context_tokens"):
        Settings.from_env(env_for(path))


def test_the_shipped_settings_pass_every_range_check(tmp_path: Path):
    Settings.from_env(env_for(write_config(tmp_path / "config.json")))


def test_a_canonical_unit_with_no_converter_names_the_field_and_the_file():
    with pytest.raises(ConfigError, match=r"tco\.json: field 'thickness': canonical_unit 'furlong'"):
        load_fields([MINIMAL_FIELD | {"canonical_unit": "furlong"}])


def test_offline_replay_is_off_by_default_and_the_environment_can_turn_it_on(tmp_path: Path):
    path = write_config(tmp_path / "config.json")

    assert Settings.from_env(env_for(path)).llm_offline is False
    assert Settings.from_env(env_for(path, PAPERFACTS_LLM_OFFLINE="true")).llm_offline is True


def test_offline_replay_moves_no_cache_key(tmp_path: Path, tco_profile):
    # It decides whether a request is sent, never what is asked, so stored results keep their names.
    path = write_config(tmp_path / "config.json")
    online = Settings.from_env(env_for(path))
    offline = Settings.from_env(env_for(path, PAPERFACTS_LLM_OFFLINE="1"))

    assert keys.extractor_key_for(online, tco_profile) == keys.extractor_key_for(offline, tco_profile)
