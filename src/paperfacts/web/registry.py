"""Every profile one server serves over its data root, read once when the app is built.

The default is the profile the settings select (``--profile`` / ``PAPERFACTS_PROFILE`` / config.json): it answers
every request that names no profile, so a URL from before there were several stays what it was, and a server
without it cannot start. Every other ``profiles/*.json`` is served beside it when it loads, and listed with its
errors when it does not; one broken file never stops the others. Results need no separating: they are already
named by keys that follow the profile's content, workbooks and chart readings by the profile's name.

What reaches a browser names a profile's file, never where the server keeps it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from paperfacts.config import Settings
from paperfacts.errors import ConfigError
from paperfacts.profile import DomainProfile
from paperfacts.profile_loader import IDENTIFIER, PROFILES_DIRNAME, loaded_file_sha256, profile_path
from paperfacts.web.documents import Library
from paperfacts.workflow import check_mode, load_run_profile

logger = logging.getLogger(__name__)


def profile_origin(settings: Settings, profile: DomainProfile) -> Path:
    """The path ``profile`` is named by, unresolved: the settings' own when it leads to the file the profile was
    loaded from, so a symlink retargeted later is followed again; otherwise that file itself."""
    named = profile_path(settings)
    return named if named.resolve() == profile.source else profile.source


def profile_file_changed(profile: DomainProfile, origin: Path) -> bool:
    """Whether the file ``origin`` leads to now holds other bytes than ``profile`` was loaded from. Every byte
    counts, display text included: a server keeps showing the text it started with. Raises OSError when the file
    cannot be read; a profile built in memory has no file, and never changed."""
    loaded = loaded_file_sha256(profile)
    if loaded is None:
        return False
    return hashlib.sha256(origin.resolve().read_bytes()).hexdigest() != loaded


@dataclass(frozen=True)
class ServedProfile:
    """One profile the server answers for, with the library whose keys follow it."""

    profile: DomainProfile
    # The path it was named by, for the drift check.
    origin: Path
    # None when the configured extraction mode cannot ask this profile (workflow.check_mode): its keys cannot even
    # be computed under these settings, so it has no results here and nothing is queued under it, but it is still
    # listed and described. ``not_runnable`` says why.
    library: Library | None
    not_runnable: str | None = None

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def runnable(self) -> bool:
        return self.library is not None


@dataclass(frozen=True)
class InvalidProfile:
    """A file under ``profiles/`` that did not load, and why."""

    name: str
    file: str
    errors: tuple[str, ...]


class ProfileRegistry:
    """The served profiles, the default first and the rest by name, and the files that did not load."""

    def __init__(self, served: Sequence[ServedProfile], invalid: Sequence[InvalidProfile] = ()) -> None:
        names = [entry.name for entry in served]
        colliding = sorted(name for name, count in Counter(names).items() if count > 1)
        if colliding:
            raise ConfigError(f"two served profiles share a name: {', '.join(colliding)}")
        for name in names:
            if not IDENTIFIER.fullmatch(name):
                # The name goes unquoted into a Content-Disposition header; the loader enforces this, a profile
                # built in memory need not have been through it.
                raise ConfigError(f"profile name {name!r} must match {IDENTIFIER.pattern}")
        self.default = served[0]
        if self.default.library is None:
            raise ConfigError(self.default.not_runnable or f"the default profile {self.default.name} cannot run")
        # The profile-free reads (a document's existence, its parse, its page images, an upload) go through it.
        self.library: Library = self.default.library
        self.served = {entry.name: entry for entry in served}
        self.invalid = tuple(invalid)

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        profile: DomainProfile | None = None,
        profiles: Sequence[DomainProfile] | None = None,
    ) -> ProfileRegistry:
        """The default is ``profile``, else the one ``settings`` selects, loaded to run: a failure, a mode that
        cannot ask it included, is fatal, since the server could not answer an old URL. The others are ``profiles``
        when given, as built, else the ``profiles/*.json`` that ``settings.web_profiles`` names (every one when
        None), loaded with the checks a run makes, where a failure only lists the file as invalid."""
        if profile is None:
            profile = load_run_profile(settings)
        else:
            check_mode(profile, settings)
        default = _served(settings, profile, profile_origin(settings, profile))
        others: list[ServedProfile] = []
        invalid: list[InvalidProfile] = []
        if profiles is not None:
            # An extra profile that is the default under another name is dropped only when it is truly the
            # same content (the default handed back in its own ``profiles``); one with the default's name but
            # different content is kept, so __init__'s duplicate-name check catches it instead of one of the
            # two silently winning.
            others = [
                _served(settings, extra, extra.source)
                for extra in profiles
                if extra.name != default.name or extra.content_hash != profile.content_hash
            ]
        else:
            others, invalid = _scan(settings, default.name)
        others.sort(key=lambda entry: entry.name)
        registry = cls([default, *others], invalid)
        logger.info(
            "serving profiles %s (default %s)%s",
            ", ".join(f"{entry.name} {entry.profile.content_hash[:12]}" for entry in registry.served.values()),
            default.name,
            f"; invalid: {', '.join(entry.file for entry in invalid)}" if invalid else "",
        )
        return registry

    def get(self, name: str | None) -> ServedProfile:
        """The profile named ``name``, the default for None; ``KeyError`` for a name not served."""
        return self.default if name is None else self.served[name]

    def invalid_named(self, name: str) -> InvalidProfile | None:
        return next((entry for entry in self.invalid if entry.name == name), None)

    def profiles_done(self, document_id: str) -> tuple[str, ...]:
        """The profiles this document is finished under: its dataset is stored under their current keys. Two
        profiles of identical content share keys, so both are named, and both show the results; only the
        per-document workbook, named after the profile a run was made under, is missing for the other until a run
        under it writes one."""
        return tuple(
            name
            for name, entry in self.served.items()
            if entry.library is not None and entry.library.finished(document_id)
        )

    def finished_documents(self, served: ServedProfile) -> list[str]:
        library = served.library
        if library is None:
            return []
        return [document_id for document_id in library.document_ids() if library.finished(document_id)]

    def changed_on_disk(self, served: ServedProfile) -> bool:
        try:
            return profile_file_changed(served.profile, served.origin)
        except OSError:
            return True  # a file that cannot be read is no longer the one being served


def _scan(settings: Settings, default: str) -> tuple[list[ServedProfile], list[InvalidProfile]]:
    """Every file under ``profiles/`` that the settings ask to serve, loaded, or listed with why it is not."""
    directory = settings.repo_root / PROFILES_DIRNAME
    wanted = None if settings.web_profiles is None else set(settings.web_profiles) - {default}
    served: list[ServedProfile] = []
    invalid: list[InvalidProfile] = []
    names = {default}
    for path in sorted(directory.glob("*.json")):
        if path.stem == default or (wanted is not None and path.stem not in wanted):
            continue
        if not IDENTIFIER.fullmatch(path.stem):
            invalid.append(InvalidProfile(path.stem, path.name, (f"{path.name}: the file name is not a profile name",)))
            continue
        named = dataclasses.replace(settings, profile=path.stem)
        try:
            loaded = load_run_profile(named, to_run=False)
        except ConfigError as exc:
            logger.warning("not serving profile %s: %s", path.name, exc)
            errors = tuple(_strip(line, directory, path) for line in str(exc).splitlines())
            invalid.append(InvalidProfile(path.stem, path.name, errors))
            continue
        except Exception as exc:
            # A file this malformed can raise something the loader was never written to catch (a deeply
            # nested bracket run overflows json.loads' own recursion, and can then overflow the loader's
            # `{value!r}` error formatting too). One broken extra file must not take the others -- or the
            # whole server -- down with it; the default profile is not guarded this way, so it still fails
            # loudly.
            logger.exception("not serving profile %s", path.name)
            invalid.append(
                InvalidProfile(path.stem, path.name, (f"{path.name}: could not be loaded ({type(exc).__name__})",))
            )
            continue
        if loaded.name != path.stem or loaded.name in names:
            # A link to another profile's file loads as that profile (the loader resolves it and checks the name
            # against the target): served under its own file name, one name would answer for two entries.
            reason = (
                f"{path.name}: loads as the profile {loaded.name!r} ({loaded.source.name}); serve that file instead"
            )
            logger.warning("not serving %s", reason)
            invalid.append(InvalidProfile(path.stem, path.name, (reason,)))
            continue
        names.add(loaded.name)
        try:
            served.append(_served(settings, loaded, profile_origin(named, loaded)))
        except Exception as exc:
            # Same guard around building the profile's Library: an extra profile's content can be malformed
            # in a way check_mode does not catch and the Library construction chokes on.
            logger.exception("not serving profile %s (library)", path.name)
            invalid.append(
                InvalidProfile(path.stem, path.name, (f"{path.name}: could not be loaded ({type(exc).__name__})",))
            )
    for missing in sorted((wanted or set()) - {entry.name for entry in served} - {entry.name for entry in invalid}):
        invalid.append(InvalidProfile(missing, f"{missing}.json", (f"{missing}.json: no such profile (web.profiles)",)))
    return served, invalid


def _served(settings: Settings, profile: DomainProfile, origin: Path) -> ServedProfile:
    try:
        check_mode(profile, settings)
    except ConfigError as exc:
        # Shown to the browser: the file's name, not where the server keeps it.
        reason = str(exc).replace(str(profile.source), profile.source.name)
        return ServedProfile(profile, origin, None, not_runnable=reason)
    return ServedProfile(profile, origin, Library(settings, profile))


def _strip(line: str, directory: Path, path: Path) -> str:
    """A loader error line with the profiles directory taken off every path it names, and ``path``'s own
    resolved target swapped for the name it is served under -- a symlink under ``profiles/`` can point
    anywhere, and a loader error naming that target must not leak an absolute path outside it."""
    for prefix in {str(directory.resolve()), str(directory)}:
        line = line.replace(prefix + "/", "")
    resolved = str(path.resolve())
    if resolved in line:
        line = line.replace(resolved, path.name)
    return line
