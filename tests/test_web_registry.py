"""The profiles one server serves: the default from the settings, every other ``profiles/*.json`` that loads, and
the ones that do not, listed with errors that name the file and never the directory the server keeps it in."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from paperfacts.config import Settings
from paperfacts.errors import ConfigError
from paperfacts.web.registry import ProfileRegistry
from support.profiles import SHIPPED_PROFILE_PATH, entity_profile_data, make_entity_profile, make_profile, profile_data


def _repo(tmp_path: Path) -> Path:
    """A repository whose profiles/ holds two shipped profiles, a broken file, the reserved name and a profile
    with entity types."""
    profiles = tmp_path / "repo" / "profiles"
    profiles.mkdir(parents=True)
    for name in ("tco", "battery_cathode"):
        shutil.copy(SHIPPED_PROFILE_PATH.parent / f"{name}.json", profiles / f"{name}.json")
    (profiles / "broken.json").write_text("{", encoding="utf-8")
    (profiles / "paperfacts.json").write_text(
        json.dumps(profile_data({"name": "paperfacts"}), ensure_ascii=False), encoding="utf-8"
    )
    wear = entity_profile_data()
    wear["name"] = "wear"
    (profiles / "wear.json").write_text(json.dumps(wear, ensure_ascii=False), encoding="utf-8")
    return tmp_path / "repo"


def _settings(repo: Path, **overrides) -> Settings:
    return Settings(data_root=repo.parent / "data", repo_root=repo, profile="tco", llm_model="fake", **overrides)


def test_every_loadable_profile_is_served_the_default_first(tmp_path: Path):
    registry = ProfileRegistry.build(_settings(_repo(tmp_path)))

    assert registry.default.name == "tco"
    assert list(registry.served) == ["tco", "battery_cathode", "wear"]
    assert registry.get(None) is registry.default
    assert registry.get("wear").library.profile is registry.get("wear").profile
    with pytest.raises(KeyError):
        registry.get("nosuch")


def test_a_file_that_does_not_load_is_listed_with_errors_naming_the_file_only(tmp_path: Path):
    registry = ProfileRegistry.build(_settings(_repo(tmp_path)))

    assert [(invalid.name, invalid.file) for invalid in registry.invalid] == [
        ("broken", "broken.json"),
        ("paperfacts", "paperfacts.json"),
    ]
    reserved = registry.invalid_named("paperfacts")
    assert reserved is not None and any("reserved" in error for error in reserved.errors)
    for invalid in registry.invalid:
        assert invalid.errors
        assert all(str(tmp_path) not in error and invalid.file in error for error in invalid.errors)


def test_a_profile_the_mode_cannot_ask_is_served_but_not_runnable(tmp_path: Path):
    registry = ProfileRegistry.build(_settings(_repo(tmp_path), extraction_mode="document"))

    wear = registry.get("wear")
    assert (registry.default.runnable, wear.runnable) == (True, False)
    assert wear.library is None  # its keys cannot be computed under document mode
    assert registry.profiles_done("0123456789abcdef") == () and registry.finished_documents(wear) == []
    assert wear.not_runnable is not None and "passage" in wear.not_runnable
    assert "wear.json" in wear.not_runnable and str(tmp_path) not in wear.not_runnable


def test_a_default_the_mode_cannot_ask_is_fatal(tmp_path: Path):
    settings = _settings(_repo(tmp_path), extraction_mode="document")

    with pytest.raises(ConfigError, match="passage"):
        ProfileRegistry.build(settings, profile=make_entity_profile())


def test_a_default_that_does_not_load_is_fatal(tmp_path: Path):
    repo = _repo(tmp_path)

    with pytest.raises(ConfigError):
        ProfileRegistry.build(Settings(data_root=tmp_path / "data", repo_root=repo, profile="broken"))


def test_in_memory_profiles_are_served_as_given(tmp_path: Path, tco_profile):
    repo = _repo(tmp_path)
    settings = _settings(repo)

    alone = ProfileRegistry.build(settings, profile=tco_profile, profiles=())
    beside = ProfileRegistry.build(settings, profile=tco_profile, profiles=(make_entity_profile(),))

    assert (list(alone.served), alone.invalid) == (["tco"], ())
    assert list(beside.served) == ["tco", "demo"]
    assert not beside.changed_on_disk(beside.get("demo"))  # built in memory: no file to drift from


def test_a_name_that_is_not_an_identifier_is_refused(tmp_path: Path, tco_profile):
    import dataclasses

    odd = dataclasses.replace(make_entity_profile(), name='x"; filename="evil')

    with pytest.raises(ConfigError, match="must match"):
        ProfileRegistry.build(_settings(_repo(tmp_path)), profile=tco_profile, profiles=(odd,))


def test_drift_is_per_profile(tmp_path: Path):
    repo = _repo(tmp_path)
    registry = ProfileRegistry.build(_settings(repo))
    battery = repo / "profiles" / "battery_cathode.json"

    battery.write_bytes(battery.read_bytes() + b"\n")

    assert registry.changed_on_disk(registry.get("battery_cathode"))
    assert not registry.changed_on_disk(registry.default)


def test_a_link_to_a_served_profile_is_listed_not_served_twice(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "profiles" / "alias.json").symlink_to(repo / "profiles" / "battery_cathode.json")

    registry = ProfileRegistry.build(_settings(repo))

    assert list(registry.served) == ["tco", "battery_cathode", "wear"]
    alias = registry.invalid_named("alias")
    assert alias is not None and "battery_cathode" in alias.errors[0] and str(tmp_path) not in alias.errors[0]


def test_a_link_to_the_default_is_listed_too(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "profiles" / "alias.json").symlink_to(repo / "profiles" / "tco.json")

    registry = ProfileRegistry.build(_settings(repo))

    assert "alias" not in registry.served and registry.invalid_named("alias") is not None


def test_web_profiles_narrows_what_is_served_beside_the_default(tmp_path: Path):
    registry = ProfileRegistry.build(_settings(_repo(tmp_path), web_profiles=("battery_cathode", "nosuch")))

    assert list(registry.served) == ["tco", "battery_cathode"]
    # Files not asked for are not even read; a name asked for that has no file is listed.
    assert [(invalid.name, invalid.errors) for invalid in registry.invalid] == [
        ("nosuch", ("nosuch.json: no such profile (web.profiles)",))
    ]


def test_an_empty_web_profiles_serves_the_default_alone(tmp_path: Path):
    registry = ProfileRegistry.build(_settings(_repo(tmp_path), web_profiles=()))

    assert (list(registry.served), registry.invalid) == (["tco"], ())


def test_a_deeply_nested_extra_profile_does_not_stop_the_server(tmp_path: Path):
    """A file this malformed overflows json.loads' own recursion before the loader ever gets a chance to
    raise ConfigError; one broken extra file must not take the whole server down with it."""
    repo = _repo(tmp_path)
    (repo / "profiles" / "deep.json").write_text("[" * 200_000, encoding="utf-8")

    registry = ProfileRegistry.build(_settings(repo))

    assert list(registry.served) == ["tco", "battery_cathode", "wear"]
    deep = registry.invalid_named("deep")
    assert deep is not None
    assert deep.errors and "could not be loaded" in deep.errors[0] and "RecursionError" in deep.errors[0]


def test_a_broken_symlink_outside_profiles_does_not_leak_its_absolute_path(tmp_path: Path):
    repo = _repo(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{", encoding="utf-8")  # invalid JSON, loaded through the link
    (repo / "profiles" / "alias.json").symlink_to(outside)

    registry = ProfileRegistry.build(_settings(repo))

    alias = registry.invalid_named("alias")
    assert alias is not None and alias.errors
    assert all(str(tmp_path) not in error and str(outside) not in error for error in alias.errors)
    assert any("alias.json" in error for error in alias.errors)


def test_the_same_profile_named_twice_in_the_profiles_seam_is_deduped(tmp_path: Path, tco_profile):
    settings = _settings(_repo(tmp_path))

    registry = ProfileRegistry.build(settings, profile=tco_profile, profiles=(tco_profile,))

    assert list(registry.served) == ["tco"]


def test_a_different_profile_under_the_defaults_name_is_the_duplicate_name_error(tmp_path: Path, tco_profile):
    """An extra profile that merely shares the default's name, but not its content, must not silently lose to
    the default (or win over it): it falls through to the duplicate-name error, and that error names only the
    names that actually collide."""
    settings = _settings(_repo(tmp_path))
    imposter = make_profile({"name": "tco"})
    other = make_entity_profile()  # named "demo": does not collide with anything

    with pytest.raises(ConfigError, match=r"^two served profiles share a name: tco$"):
        ProfileRegistry.build(settings, profile=tco_profile, profiles=(imposter, other))
