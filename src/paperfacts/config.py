"""Runtime configuration: ``config.json`` at the repository root, overridden by ``PAPERFACTS_*`` variables.

Three layers, each winning over the one before it:

1. the constants in this module, which are what the code shipped with;
2. ``config.json`` -- the file to edit. Everything that is not a secret lives there: the server address, the
   model and its endpoint, the parser services, and the field table itself (read by :mod:`paperfacts.fields`);
3. the environment, including anything ``.env`` puts there. This is how one machine points at its own parser
   services, and the only place a secret belongs: the API key is never written to ``config.json``.

An empty ``*_url`` means run the script in ``runners/`` as a subprocess (a workstation); a non-empty one
means call a long-running service (a GPU server). :mod:`paperfacts.workflow` picks the implementation from
that, so nothing above it has to know which environment it is in.

``from_env(environ)`` with an explicit mapping touches neither ``.env`` nor the API key file, so a unit test
behaves the same on a machine that happens to have a key lying around.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast, get_args

from paperfacts.errors import ConfigError

ENV_PREFIX = "PAPERFACTS_"
# The editable configuration file, and the variable that moves it (a container mounting it elsewhere).
CONFIG_FILENAME = "config.json"
ENV_CONFIG_PATH = f"{ENV_PREFIX}CONFIG"
ENV_FILENAME = ".env"

# parents[2] of src/paperfacts/config.py is the repository root. The project is run from a checkout with
# `uv run`; set PAPERFACTS_REPO_ROOT to point at runners/ if it is ever installed as a wheel elsewhere.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
# DPI used to rasterise pages for PaddleOCR-VL. The subprocess and HTTP paths must agree or the pixel
# coordinates they report are not comparable.
DEFAULT_RENDER_DPI = 200
# A first subprocess run downloads model weights and inference on a laptop is slow, so allow an hour. A
# server keeps its models resident, where fifteen minutes per paper is generous.
DEFAULT_SUBPROCESS_TIMEOUT_S = 3600.0
DEFAULT_HTTP_TIMEOUT_S = 900.0
# Any OpenAI-compatible endpoint works. This deployment runs against an Alibaba Cloud Model Studio
# workspace in its OpenAI-compatible ("compatible-mode") mode; plain DeepSeek was the earlier default.
DEFAULT_LLM_BASE_URL = "https://ws-1q3kj1umgcqeng4f.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "deepseek-v4.1-flash"
# v4.1-flash reasons before it answers and the reasoning is billed against max_tokens, so a request takes
# longer than one to a plain chat model; 600s is what the gateway needs for a full paper.
DEFAULT_LLM_TIMEOUT_S = 600.0
# Measured against this workspace's endpoint: a 178k-token prompt is accepted, so the window is at least
# that. The budget the code plans against is this minus DEFAULT_MAX_TOKENS, which leaves headroom on both
# sides instead of silently truncating the prompt or the answer that has to fit beside it.
DEFAULT_LLM_CONTEXT_TOKENS = 200_000
# The key stays on the machine: environment first, then this file in the repository root (gitignored).
DEFAULT_API_KEY_FILENAME = "deepseek_api_key"
# Built-in baselines for the knobs that change what the model is asked. config.json ships with these exact
# values; paperfacts.keys treats them as the shape a cache key had before anybody edited anything, so the
# stored facts of an unmodified checkout keep their filenames.
DEFAULT_TEMPERATURE = 0.0
# Room for the reasoning plus the JSON answer it precedes: 8192 was enough for a non-reasoning model and
# is not for this one.
DEFAULT_MAX_TOKENS = 65536
# Reasoning effort, as the OpenAI-shaped `reasoning_effort` request parameter. The baseline is None, which
# means the parameter is omitted entirely: that is what every request looked like before this setting
# existed, so an unedited checkout keeps its cache keys. See _parse_reasoning_effort for the accepted values.
DEFAULT_LLM_REASONING_EFFORT: str | None = None
# Passage mode's inventory question ("which samples does this paper have?") reasons an order of magnitude
# longer than the per-field questions that follow it -- measured at 11k-17k hidden tokens per lane, about
# 70% of a run's completion tokens. This gives that one request its own effort. The baseline is None, which
# means "inherit llm.reasoning_effort", i.e. exactly what every request looked like before this setting.
DEFAULT_LLM_INVENTORY_REASONING_EFFORT: str | None = None
REASONING_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high")
DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_BACKOFF_S = 2.0
# How many of one lane's per-field questions may be in flight at once. Purely a scheduling knob: every
# request is byte-identical to the one the sequential loop would have sent, so it is deliberately absent
# from extractor_key and comparison_key. 1 restores the strictly sequential behaviour.
DEFAULT_LLM_CONCURRENCY = 4
DEFAULT_CANDIDATE_LIMIT = 8
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
# The web interface's login. The username is configuration; the password is a secret and lives only in
# the environment (.env), like the API key -- a default password in a committed file would be worse than
# none, because it looks like protection.
DEFAULT_WEB_USERNAME = "paperfacts"
DEFAULT_MAX_UPLOAD_MB = 200
DEFAULT_PAGE_DPI = 110
DEFAULT_OVERLAY_DPI = 150

# How the model is asked for the facts: the whole paper in one question, or one question per field over the
# blocks retrieved for it (see :mod:`paperfacts.extract`). Passage mode is the default because on the three
# papers in ``data/docs`` it found 88 values where whole-document mode found 48, halved nothing, and left a
# lower share of them ungrounded; it costs about three times the prompt tokens. The measurement is in
# ``.omc/research/extraction-modes.md``.
type ExtractionMode = Literal["document", "passage"]
EXTRACTION_MODES: tuple[str, ...] = get_args(ExtractionMode.__value__)
DEFAULT_EXTRACTION_MODE: ExtractionMode = "passage"


def config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Where the configuration file lives. Found next to the repository root, not the working directory, so
    running ``paperfacts`` from anywhere reads the same file."""
    env = os.environ if environ is None else environ
    override = (env.get(ENV_CONFIG_PATH) or "").strip()
    return Path(override) if override else DEFAULT_REPO_ROOT / CONFIG_FILENAME


@dataclass(frozen=True)
class ConfigDocument:
    """``config.json``, read and type-checked.

    Every error names the key and the file. A typo in a configuration file is the one mistake a user is
    guaranteed to make, and "must be a number, got \'8000\'" is the difference between a fix and a hunt.
    """

    data: Mapping[str, Any]
    path: Path

    def _node(self, dotted: str) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise ConfigError(f"{self.path}: missing setting {dotted!r}")
            node = node[part]
        return node

    def get[T](self, dotted: str, kind: type[T]) -> T:
        node = self._node(dotted)
        if kind is float and type(node) is int:
            return cast(T, float(node))  # 300 is a perfectly good way to write 300.0
        # `type(...) is` rather than isinstance: bool is a subclass of int, and `true` is not a port number.
        if type(node) is not kind:
            raise ConfigError(f"{self.path}: {dotted} must be {kind.__name__}, got {node!r}")
        return cast(T, node)

    def has(self, dotted: str) -> bool:
        """Whether the file carries this key at all. Absent means "use the built-in baseline"."""
        try:
            self._node(dotted)
        except ConfigError:
            return False
        return True

    def get_or[T](self, dotted: str, kind: type[T], default: T) -> T:
        """A setting whose key a configuration file written before it existed does not carry yet.

        Absent means "use the built-in baseline". Present still has to be the right type, so a typo is
        an error naming the key rather than a silent fallback.
        """
        return self.get(dotted, kind) if self.has(dotted) else default

    def text_or_none(self, dotted: str) -> str | None:
        """A string setting that may be ``null``, which is how "not configured" is written for a URL."""
        value = self._node(dotted)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConfigError(f"{self.path}: {dotted} must be a string or null, got {value!r}")
        return value.strip() or None

    def text_or_none_if_absent(self, dotted: str) -> str | None:
        """:meth:`text_or_none` for a key a file written before this setting existed may not carry:
        absent reads as ``null`` rather than as an error, the same contract as :meth:`get_or`."""
        return self.text_or_none(dotted) if self.has(dotted) else None

    def entries(self, dotted: str) -> list[Any]:
        value = self._node(dotted)
        if not isinstance(value, list):
            raise ConfigError(f"{self.path}: {dotted} must be a list, got {type(value).__name__}")
        return value


@cache
def load_config(path: Path) -> ConfigDocument:
    """Read and parse the configuration file. Cached per path: it is static for the life of a process."""
    if not path.is_file():
        raise ConfigError(
            f"no configuration file at {path}. It is committed at the repository root; copy it back, or "
            f"point {ENV_CONFIG_PATH} at another one."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must hold a JSON object, got {type(data).__name__}")
    return ConfigDocument(data=data, path=path)


def configuration(environ: Mapping[str, str] | None = None) -> ConfigDocument:
    """The configuration file this process should read."""
    return load_config(config_path(environ))


@cache
def load_env_file() -> None:
    """Put ``.env`` into the environment, once, without overwriting what is already there.

    An exported variable or a systemd unit still wins over the file, which is what makes ``.env`` a
    development convenience rather than a second source of truth.
    """
    from dotenv import load_dotenv

    load_dotenv(DEFAULT_REPO_ROOT / ENV_FILENAME, override=False)


def _warn_if_fields_came_from_elsewhere(path: Path) -> None:
    """Warn when these settings and the field table were read from two different files.

    ``paperfacts.fields`` binds the table once, at import, against the real environment. Passing ``from_env``
    a mapping that names a different file therefore yields settings from one file and a field schema from
    another -- harmless in a test that means it, confusing anywhere else, so it is said out loud.
    """
    fields_module = sys.modules.get("paperfacts.fields")
    loaded = getattr(fields_module, "_CONFIG", None)
    if loaded is not None and loaded.path != path:
        logging.getLogger(__name__).warning(
            "settings read from %s but the field table was loaded from %s", path, loaded.path
        )


@dataclass(frozen=True)
class Settings:
    data_root: Path = Path("data")
    repo_root: Path = DEFAULT_REPO_ROOT
    uv_bin: str = "uv"
    mineru_url: str | None = None
    paddle_url: str | None = None
    paddle_render_dpi: int = DEFAULT_RENDER_DPI
    subprocess_timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S
    http_timeout_s: float = DEFAULT_HTTP_TIMEOUT_S
    # In subprocess mode, hand PaddleOCR-VL's VLM stage to an external server (mlx-vlm-server on Apple
    # silicon, vllm-server on Linux) instead of running it in-process.
    paddle_vl_backend: str | None = None
    paddle_vl_server_url: str | None = None
    paddle_vl_model_name: str | None = None
    # The key is kept out of repr: no stray logger.debug("%s", settings) should ever write sk-... to a log.
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str = DEFAULT_LLM_MODEL
    llm_api_key: str | None = field(default=None, repr=False)
    llm_api_key_file: Path | None = None
    llm_timeout_s: float = DEFAULT_LLM_TIMEOUT_S
    llm_context_tokens: int = DEFAULT_LLM_CONTEXT_TOKENS
    llm_temperature: float = DEFAULT_TEMPERATURE
    llm_max_tokens: int = DEFAULT_MAX_TOKENS
    # None leaves `reasoning_effort` out of the request; a value asks the endpoint for that much hidden
    # reasoning before the answer.
    llm_reasoning_effort: str | None = DEFAULT_LLM_REASONING_EFFORT
    # None inherits llm_reasoning_effort; a value overrides it for passage mode's inventory question only.
    llm_inventory_reasoning_effort: str | None = DEFAULT_LLM_INVENTORY_REASONING_EFFORT
    # Per-field questions in flight per lane. The two lanes themselves always run as a pair, so the peak
    # number of open requests is twice this. It changes nothing about what is asked, only when.
    llm_concurrency: int = DEFAULT_LLM_CONCURRENCY
    llm_retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    llm_retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S
    # Extract each lane this many times and keep what a majority of passes agree on. Costs one LLM call
    # per pass, so it stays at 1 unless a run explicitly asks for more.
    extraction_passes: int = 1
    extraction_mode: ExtractionMode = DEFAULT_EXTRACTION_MODE
    # How many blocks one passage-mode question may carry; ignored in document mode.
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    server_host: str = DEFAULT_SERVER_HOST
    server_port: int = DEFAULT_SERVER_PORT
    # Login for the web interface. An empty password means the app is open, which is the right default for
    # a laptop and the wrong one for a tunnel: see README, "The web interface".
    web_username: str = DEFAULT_WEB_USERNAME
    web_password: str | None = field(default=None, repr=False)
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    page_dpi: int = DEFAULT_PAGE_DPI
    page_dpi_min: int = 50
    page_dpi_max: int = 220
    overlay_dpi: int = DEFAULT_OVERLAY_DPI

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Read ``PAPERFACTS_*``. An empty string counts as unset; a bad number names the variable.

        Key resolution order: ``PAPERFACTS_LLM_API_KEY``, then ``DEEPSEEK_API_KEY``. With neither, record
        where the key file should be (``PAPERFACTS_LLM_API_KEY_FILE``, else ``deepseek_api_key`` in the
        repository root) and read it only when it is actually needed.
        """
        env = os.environ if environ is None else environ
        if environ is None:
            load_env_file()  # only when reading the real environment; an explicit mapping is the whole truth

        def get(name: str) -> str | None:
            return (env.get(ENV_PREFIX + name) or "").strip() or None

        def number[T: (int, float)](name: str, default: T, kind: type[T]) -> T:
            return _parse_number(name, get(name), default, kind)

        file = configuration(env)
        _warn_if_fields_came_from_elsewhere(file.path)
        repo_root = Path(get("REPO_ROOT") or DEFAULT_REPO_ROOT)
        key_file = get("LLM_API_KEY_FILE")
        return cls(
            data_root=Path(get("DATA_ROOT") or file.get("data_root", str)),
            repo_root=repo_root,
            uv_bin=get("UV_BIN") or file.get("parsers.uv_bin", str),
            mineru_url=get("MINERU_URL") or file.text_or_none("parsers.mineru_url"),
            paddle_url=get("PADDLE_URL") or file.text_or_none("parsers.paddle_url"),
            paddle_render_dpi=number("PADDLE_RENDER_DPI", file.get("parsers.paddle_render_dpi", int), int),
            subprocess_timeout_s=number("SUBPROCESS_TIMEOUT_S", file.get("parsers.subprocess_timeout_s", float), float),
            http_timeout_s=number("HTTP_TIMEOUT_S", file.get("parsers.http_timeout_s", float), float),
            paddle_vl_backend=get("PADDLE_VL_BACKEND") or file.text_or_none("parsers.paddle_vl_backend"),
            paddle_vl_server_url=get("PADDLE_VL_SERVER_URL") or file.text_or_none("parsers.paddle_vl_server_url"),
            paddle_vl_model_name=get("PADDLE_VL_MODEL_NAME") or file.text_or_none("parsers.paddle_vl_model_name"),
            llm_base_url=(get("LLM_BASE_URL") or file.get("llm.base_url", str)).rstrip("/"),
            llm_model=get("LLM_MODEL") or file.get("llm.model", str),
            llm_api_key=get("LLM_API_KEY") or (env.get("DEEPSEEK_API_KEY") or "").strip() or None,
            llm_api_key_file=Path(key_file) if key_file else repo_root / DEFAULT_API_KEY_FILENAME,
            llm_timeout_s=number("LLM_TIMEOUT_S", file.get("llm.timeout_s", float), float),
            llm_context_tokens=number("LLM_CONTEXT_TOKENS", file.get("llm.context_tokens", int), int),
            llm_temperature=number("LLM_TEMPERATURE", file.get("llm.temperature", float), float),
            llm_max_tokens=number("LLM_MAX_TOKENS", file.get("llm.max_tokens", int), int),
            llm_reasoning_effort=_parse_reasoning_effort(
                get("LLM_REASONING_EFFORT") or file.text_or_none("llm.reasoning_effort"), file.path
            ),
            llm_inventory_reasoning_effort=_parse_reasoning_effort(
                get("LLM_INVENTORY_REASONING_EFFORT") or file.text_or_none_if_absent("llm.inventory_reasoning_effort"),
                file.path,
                dotted="llm.inventory_reasoning_effort",
                variable="LLM_INVENTORY_REASONING_EFFORT",
            ),
            llm_concurrency=_positive(
                number("LLM_CONCURRENCY", file.get_or("llm.concurrency", int, DEFAULT_LLM_CONCURRENCY), int),
                "llm.concurrency",
                file.path,
            ),
            llm_retry_attempts=_positive(
                number("LLM_RETRY_ATTEMPTS", file.get("llm.retry_attempts", int), int), "llm.retry_attempts", file.path
            ),
            llm_retry_backoff_s=number("LLM_RETRY_BACKOFF_S", file.get("llm.retry_backoff_s", float), float),
            extraction_passes=_positive(
                number("EXTRACTION_PASSES", file.get("extraction.passes", int), int), "extraction.passes", file.path
            ),
            extraction_mode=_parse_mode(get("EXTRACTION_MODE") or file.get("extraction.mode", str), file.path),
            candidate_limit=_positive(
                number("CANDIDATE_LIMIT", file.get("extraction.candidate_limit", int), int),
                "extraction.candidate_limit",
                file.path,
            ),
            server_host=get("SERVER_HOST") or file.get("server.host", str),
            server_port=number("SERVER_PORT", file.get("server.port", int), int),
            web_username=get("WEB_USERNAME") or file.get("web.username", str),
            web_password=get("WEB_PASSWORD"),
            max_upload_bytes=number("MAX_UPLOAD_MB", file.get("server.max_upload_mb", int), int) * 1024 * 1024,
            page_dpi=number("PAGE_DPI", file.get("server.page_dpi.default", int), int),
            page_dpi_min=number("PAGE_DPI_MIN", file.get("server.page_dpi.min", int), int),
            page_dpi_max=number("PAGE_DPI_MAX", file.get("server.page_dpi.max", int), int),
            overlay_dpi=number("OVERLAY_DPI", file.get("overlay.dpi", int), int),
        )

    def require_llm_api_key(self) -> str:
        """Resolve the key at the moment it is needed: environment first, then the key file."""
        if self.llm_api_key:
            return self.llm_api_key
        if self.llm_api_key_file is not None and self.llm_api_key_file.is_file():
            key = self.llm_api_key_file.read_text(encoding="utf-8").strip()
            if key:
                return key
        key_file = self.llm_api_key_file or self.repo_root / DEFAULT_API_KEY_FILENAME
        raise ConfigError(
            f"no LLM API key: put PAPERFACTS_LLM_API_KEY in {DEFAULT_REPO_ROOT / ENV_FILENAME} "
            f"(copy .env.example), export PAPERFACTS_LLM_API_KEY or DEEPSEEK_API_KEY, "
            f"or write the key to {key_file}. All three are gitignored; config.json cannot hold a key."
        )


def _positive(value: int, dotted: str, source: Path) -> int:
    """A count that must be at least one. Zero passes, zero retries or zero candidate blocks all fail deep
    inside a loop with an error that names neither the setting nor the file it came from."""
    if value < 1:
        raise ConfigError(
            f"{dotted} must be at least 1, got {value} (set in {source} or the matching {ENV_PREFIX} variable)"
        )
    return value


def _parse_mode(raw: str, source: Path) -> ExtractionMode:
    """An unknown mode names the ones that exist rather than silently falling back to the default."""
    if raw not in EXTRACTION_MODES:
        modes = ", ".join(EXTRACTION_MODES)
        raise ConfigError(
            f"extraction mode is {raw!r}, expected one of {modes} (set in {source} or {ENV_PREFIX}EXTRACTION_MODE)"
        )
    return cast(ExtractionMode, raw)


def _parse_reasoning_effort(
    raw: str | None,
    source: Path,
    *,
    dotted: str = "llm.reasoning_effort",
    variable: str = "LLM_REASONING_EFFORT",
) -> str | None:
    """``null`` (or an unset variable) means "omit the parameter"; anything else must be one we know the
    endpoint accepts, named here rather than discovered as a 400 halfway through a paper."""
    if raw is None:
        return None
    if raw not in REASONING_EFFORTS:
        efforts = ", ".join(REASONING_EFFORTS)
        raise ConfigError(
            f"{dotted} is {raw!r}, expected null or one of {efforts} (set in {source} or {ENV_PREFIX}{variable})"
        )
    return raw


def _parse_number[T: (int, float)](name: str, raw: str | None, default: T, kind: type[T]) -> T:
    if raw is None:
        return default
    try:
        return kind(raw)
    except ValueError as exc:
        raise ConfigError(f"environment variable {ENV_PREFIX}{name} is not a valid {kind.__name__}: {raw!r}") from exc
