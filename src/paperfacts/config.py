"""Runtime configuration: ``config.json`` at the repository root, overridden by ``PAPERFACTS_*`` variables.

Three layers, each winning over the one before it:

1. the constants in this module, which are what the code shipped with;
2. ``config.json`` -- the file to edit. Everything that is not a secret lives there: the server address, the
   model and its endpoint, the parser services, and which domain profile to run (``profile``; the field table
   is the profile's, in ``profiles/<name>.json``);
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
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
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
# What the OpenAI-shaped `reasoning_effort` request parameter may say. "none" is a value the endpoint
# accepts and is sent as such; not sending the parameter at all is a different thing, spelled None.
ReasoningEffort = Literal["none", "low", "medium", "high"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = get_args(ReasoningEffort)


class Inherit(Enum):
    """The one member is a sentinel: "whatever the client is already asking for".

    Three meanings used to share one ``None``: omit the parameter, inherit the client's, and "no override
    given". Spelling the third as its own value is what lets the inventory question say "omit" while the
    client still sends an effort on every other question.
    """

    INHERIT = "inherit"


INHERIT = Inherit.INHERIT
# The inventory override: an effort to send, None to send no parameter at all, INHERIT to leave the
# request exactly as the client would have built it.
InventoryReasoningEffort = ReasoningEffort | None | Inherit
# The baseline is None, which means the parameter is omitted entirely: that is what every request looked
# like before this setting existed, so an unedited checkout keeps its cache keys.
DEFAULT_LLM_REASONING_EFFORT: ReasoningEffort | None = None
# Passage mode's inventory question ("which samples does this paper have?") reasons an order of magnitude
# longer than the per-field questions that follow it -- measured at 11k-17k hidden tokens per lane, about
# 70% of a run's completion tokens. This gives that one request its own effort. The baseline is INHERIT,
# which leaves it exactly as it was before this setting existed.
DEFAULT_LLM_INVENTORY_REASONING_EFFORT: InventoryReasoningEffort = INHERIT
DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_BACKOFF_S = 2.0
# How many of one lane's per-field questions may be in flight at once. Purely a scheduling knob: every
# request is byte-identical to the one the sequential loop would have sent, so it is deliberately absent
# from extractor_key and comparison_key. 1 restores the strictly sequential behaviour.
DEFAULT_LLM_CONCURRENCY = 4
# How many model requests (text and vision, every lane of every document) may be on the wire at once in
# one process. The per-lane `llm.concurrency` alone multiplies with the lanes, the figures stage and the
# documents running beside each other; this is the one number that bounds the product, so a corpus run
# stays under the Model Studio workspace's rate limit instead of spending its retries on 429s. 8 is two
# documents' worth of lanes at the default `llm.concurrency`. Scheduling only, so absent from every key.
DEFAULT_LLM_MAX_IN_FLIGHT = 8
DEFAULT_CANDIDATE_LIMIT = 8
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
# The web interface's login. The username is configuration; the password is a secret and lives only in
# the environment (.env), like the API key -- a default password in a committed file would be worse than
# none, because it looks like protection.
DEFAULT_WEB_USERNAME = "paperfacts"
DEFAULT_MAX_UPLOAD_MB = 200
# How many documents the web job queue and `paperfacts batch` process at once. Parsing is still one paper
# per parser at a time (the single GPU), so extra documents mostly overlap their model waits; three keeps
# the in-flight limit above busy without letting a queue of papers pile up behind it.
DEFAULT_MAX_PARALLEL_DOCUMENTS = 3
DEFAULT_PAGE_DPI = 110
DEFAULT_OVERLAY_DPI = 150
# The domain profile: a bare name is profiles/<name>.json under the repository root (paperfacts.profile).
DEFAULT_PROFILE = "tco"
# Sample-pairing confidence below this counts as low confidence: the fact is still compared, but the report
# counts it separately so a reviewer can look at it. File-only (comparison.ambiguous_match_confidence).
DEFAULT_AMBIGUOUS_MATCH_CONFIDENCE = 0.6

# How the model is asked for the facts: the whole paper in one question, or one question per field over the
# blocks retrieved for it (see :mod:`paperfacts.extract`). Passage mode is the default because on the three
# papers in ``data/docs`` it found 88 values where whole-document mode found 48, halved nothing, and left a
# lower share of them ungrounded; it costs about three times the prompt tokens. The measurement is in
# ``.omc/research/extraction-modes.md``.
type ExtractionMode = Literal["document", "passage"]
EXTRACTION_MODES: tuple[str, ...] = get_args(ExtractionMode.__value__)
DEFAULT_EXTRACTION_MODE: ExtractionMode = "passage"

# Figure reading (:mod:`paperfacts.figures`): a vision model reads property-vs-condition charts. Off by
# default because it is an offline batch stage: about a minute per chart with the model below. The model,
# the one-retry budget and the timeout come from the measurement in .omc/research/figure-reading-accuracy.md
# (qwen3.7-plus: median error 2.2 %, p90 13.6 %; qwen3-vl-plus had a p90 of 150 % and is not usable).
DEFAULT_FIGURES_ENABLED = False
DEFAULT_FIGURES_MODEL = "qwen3.7-plus"
DEFAULT_FIGURES_MAX_PER_DOCUMENT = 12
# Crops are rendered at the DPI PaddleOCR-VL pages are, which is dense enough for small tick labels.
DEFAULT_FIGURES_DPI = DEFAULT_RENDER_DPI
# Bounds a crop's area so the endpoint never resizes it with an algorithm we do not control.
DEFAULT_FIGURES_MAX_PIXELS = 2_000_000
# One chart took up to 134 s in the measurement, and one request to a sibling model hung for 271 s.
DEFAULT_FIGURES_TIMEOUT_S = 300.0
# Visual validation (:mod:`paperfacts.validate`): a vision model transcribes the region a value was cited
# from, and the code decides whether the value occurs in that transcription. Off by default, so a checkout
# that never configured a VLM keeps every filename it has -- a disabled stage writes nothing and stamps
# nothing.
DEFAULT_VLM_ENABLED = False
# The same workspace serves a vision model through the same compatible-mode address, so one key and one
# endpoint cover both models until a self-hosted server takes over.
DEFAULT_VLM_BASE_URL = DEFAULT_LLM_BASE_URL
# An open-weight vision model, so a hosted pilot and a self-hosted route can run the *same* weights and
# their readings stay comparable. Chosen for independence from both parsers rather than leaderboard rank:
# one lane's parser is a VLM of its own family, and the OCR specialists share a layout model with the other.
DEFAULT_VLM_MODEL = "qwen3-vl-32b-instruct"
DEFAULT_VLM_TIMEOUT_S = 300.0
DEFAULT_VLM_TEMPERATURE = 0.0
# A transcription of one block or one table; far less than an extraction answer.
DEFAULT_VLM_MAX_TOKENS = 4096
# The region is rendered at the same DPI one parser renders pages at, so the model reads pixels of that
# density and its boxes line up with the overlays.
DEFAULT_VLM_CROP_DPI = DEFAULT_RENDER_DPI
# Page fraction added around the cited blocks: enough to catch a descender or a table rule the parser's box
# clipped, not enough to pull in a neighbouring paragraph.
DEFAULT_VLM_CROP_PADDING = 0.01
# Above this the crop is shrunk before it is sent. Hosted endpoints resize large images anyway; doing it
# here keeps the decision, and its cost to small digits, visible and testable.
DEFAULT_VLM_CROP_MAX_PIXELS = 2_000_000
# Which values are shown to the model. "disputed": everything the two lanes could not settle between them --
# conflicts, ambiguities, one-sided values, and any value grounding could not locate. "tables": the disputed
# set plus every value cited from a table block, agreed or not -- a table is where a parser's layout model
# fails silently (a shifted column reads as a clean number) and where the extractor sees only what the parser
# gave it, so a value from a table is checked against the pixels even when both lanes agree; prose earns a
# check only when disputed. "all": every value in both lanes, which is what measuring the "both lanes agree
# and both are wrong" rate needs.
type ValidationPolicy = Literal["disputed", "tables", "all"]
VALIDATION_POLICIES: tuple[str, ...] = get_args(ValidationPolicy.__value__)
DEFAULT_VLM_POLICY: ValidationPolicy = "tables"
# Blocks before and after the cited block (same page, reading order) included in the crop -- the sliding
# window. A table's caption sits in the block before it and its footnote in the block after; a value the
# parser split across a block boundary lives in both. One on each side is what those cases need; a wider
# window mostly adds prose the model has to read past.
DEFAULT_VLM_CONTEXT_BLOCKS = 1
# Fill blanks from tables: for the fields a lane's sample lacks, the model's transcription of the tables that
# sample was cited from is handed to the extraction model, and a value it quotes is kept only when it grounds
# in that transcription. Off, the stage only validates.
DEFAULT_VLM_FILL_BLANKS = True
# Vision requests in flight at once. Scheduling only, like llm.concurrency: absent from every key.
DEFAULT_VLM_CONCURRENCY = 4
# The supervisor stage (paperfacts.supervisor): off by default, and at the LLM's endpoint and key unless the
# file names another (a null base_url follows llm.base_url), so a hosted pilot needs no second account. The
# model is a small instruct model: it reads one passage and returns one score, which a 7B-class model does
# well and quickly.
DEFAULT_SUPERVISOR_ENABLED = False
DEFAULT_SUPERVISOR_BASE_URL = DEFAULT_LLM_BASE_URL
DEFAULT_SUPERVISOR_MODEL = "qwen2.5-7b-instruct"
DEFAULT_SUPERVISOR_MIN_CONFIDENCE = 0.6
DEFAULT_SUPERVISOR_VOTE_THRESHOLD = 0.6
DEFAULT_SUPERVISOR_TIMEOUT_S = 30.0
# Keys config.json held until the domain moved into a profile (profiles/<name>.json).
MOVED_TO_PROFILE = ("fields", "condition_keywords")
_TRUE_WORDS = {"1", "true", "yes", "on"}
_FALSE_WORDS = {"0", "false", "no", "off"}


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

    def text_or_none(self, dotted: str) -> str | None:
        """A string setting that may be ``null``, which is how "not configured" is written for a URL."""
        value = self._node(dotted)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ConfigError(f"{self.path}: {dotted} must be a string or null, got {value!r}")
        return value.strip() or None

    def names_or_none(self, dotted: str) -> tuple[str, ...] | None:
        """A list of names that may be ``null``, which is how "all of them" is written."""
        value = self._node(dotted)
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            raise ConfigError(f"{self.path}: {dotted} must be a list of names or null, got {value!r}")
        return tuple(item.strip() for item in value)

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
    for key in MOVED_TO_PROFILE:
        if key in data:
            # Loud rather than ignored: a table left here looks live while every run reads the profile's, so an
            # edit to it would silently do nothing.
            # Imported here: profile_loader.py imports this module. Its rule, so a profile given as a path is named
            # as one.
            from paperfacts.profile_loader import profile_path

            profile = data["profile"] if isinstance(data.get("profile"), str) else "<name>"
            raise ConfigError(
                f"{path}: {key} moved to {profile_path(Settings(profile=profile))}; delete {key!r} from {path} and "
                "edit it there"
            )
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


@dataclass(frozen=True)
class Settings:
    data_root: Path = Path("data")
    repo_root: Path = DEFAULT_REPO_ROOT
    # A profile name, or a path to a profile file (anything with a "/" or ending in ".json").
    profile: str = DEFAULT_PROFILE
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
    llm_reasoning_effort: ReasoningEffort | None = DEFAULT_LLM_REASONING_EFFORT
    # INHERIT reuses llm_reasoning_effort, None omits the parameter, a value overrides it -- for passage
    # mode's inventory question only.
    llm_inventory_reasoning_effort: InventoryReasoningEffort = DEFAULT_LLM_INVENTORY_REASONING_EFFORT
    # Per-field questions in flight per lane. The two lanes themselves always run as a pair, so one paper's
    # peak is twice this; llm_max_in_flight caps the total. It changes nothing about what is asked, only when.
    llm_concurrency: int = DEFAULT_LLM_CONCURRENCY
    # The process-wide ceiling on requests in flight, over every lane, stage and document together.
    llm_max_in_flight: int = DEFAULT_LLM_MAX_IN_FLIGHT
    llm_retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    llm_retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S
    # Replay only: every model request must be answered from the LLM cache or fail. Not part of any key:
    # it changes whether a request is sent, never what is asked.
    llm_offline: bool = False
    # Extract each lane this many times and keep what a majority of passes agree on. Costs one LLM call
    # per pass, so it stays at 1 unless a run explicitly asks for more.
    extraction_passes: int = 1
    extraction_mode: ExtractionMode = DEFAULT_EXTRACTION_MODE
    # Caps the blocks a passage-mode question sees on a unit match alone; blocks naming the field by a
    # keyword always come. Ignored in document mode.
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    server_host: str = DEFAULT_SERVER_HOST
    server_port: int = DEFAULT_SERVER_PORT
    # Login for the web interface. An empty password means the app is open, which is the right default for
    # a laptop and the wrong one for a tunnel: see README, "The web interface".
    web_username: str = DEFAULT_WEB_USERNAME
    web_password: str | None = field(default=None, repr=False)
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    # Documents processed at once, by the web job queue and by `batch` unless it is given --jobs.
    max_parallel_documents: int = DEFAULT_MAX_PARALLEL_DOCUMENTS
    # The profiles under profiles/ a server serves beside its default, by name; None serves every one that loads.
    # A deployment that does not want the example profiles runnable (each run costs tokens) names its own here.
    web_profiles: tuple[str, ...] | None = None
    page_dpi: int = DEFAULT_PAGE_DPI
    page_dpi_min: int = 50
    page_dpi_max: int = 220
    overlay_dpi: int = DEFAULT_OVERLAY_DPI
    # The figures stage. Its endpoint and key are the LLM's: one Model Studio workspace serves both.
    figures_enabled: bool = DEFAULT_FIGURES_ENABLED
    figures_model: str = DEFAULT_FIGURES_MODEL
    figures_max_per_document: int = DEFAULT_FIGURES_MAX_PER_DOCUMENT
    figures_dpi: int = DEFAULT_FIGURES_DPI
    figures_max_pixels: int = DEFAULT_FIGURES_MAX_PIXELS
    figures_timeout_s: float = DEFAULT_FIGURES_TIMEOUT_S
    # Visual validation. Its endpoint and key are the LLM's, as the figures stage's are.
    vlm_enabled: bool = DEFAULT_VLM_ENABLED
    vlm_base_url: str = DEFAULT_VLM_BASE_URL
    vlm_model: str = DEFAULT_VLM_MODEL
    vlm_timeout_s: float = DEFAULT_VLM_TIMEOUT_S
    vlm_temperature: float = DEFAULT_VLM_TEMPERATURE
    vlm_max_tokens: int = DEFAULT_VLM_MAX_TOKENS
    vlm_crop_dpi: int = DEFAULT_VLM_CROP_DPI
    vlm_crop_padding: float = DEFAULT_VLM_CROP_PADDING
    vlm_crop_max_pixels: int = DEFAULT_VLM_CROP_MAX_PIXELS
    vlm_policy: ValidationPolicy = DEFAULT_VLM_POLICY
    vlm_context_blocks: int = DEFAULT_VLM_CONTEXT_BLOCKS
    vlm_fill_blanks: bool = DEFAULT_VLM_FILL_BLANKS
    vlm_concurrency: int = DEFAULT_VLM_CONCURRENCY
    # The supervisor stage. ``supervisor_api_key_env`` names an environment variable holding its key; None
    # means the LLM's key, as the figures and vlm stages use. ``supervisor_api_key`` is that variable's value,
    # read once with the rest of the environment.
    supervisor_enabled: bool = DEFAULT_SUPERVISOR_ENABLED
    supervisor_base_url: str = DEFAULT_SUPERVISOR_BASE_URL
    supervisor_model: str = DEFAULT_SUPERVISOR_MODEL
    supervisor_api_key_env: str | None = None
    supervisor_api_key: str | None = None
    supervisor_min_confidence: float = DEFAULT_SUPERVISOR_MIN_CONFIDENCE
    supervisor_vote_threshold: float = DEFAULT_SUPERVISOR_VOTE_THRESHOLD
    supervisor_timeout_s: float = DEFAULT_SUPERVISOR_TIMEOUT_S
    ambiguous_match_confidence: float = DEFAULT_AMBIGUOUS_MATCH_CONFIDENCE

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
        repo_root = Path(get("REPO_ROOT") or DEFAULT_REPO_ROOT)
        key_file = get("LLM_API_KEY_FILE")
        supervisor_key_env = get("SUPERVISOR_API_KEY_ENV") or file.text_or_none("supervisor.api_key_env")
        llm_base_url = (get("LLM_BASE_URL") or file.get("llm.base_url", str)).rstrip("/")
        settings = cls(
            data_root=Path(get("DATA_ROOT") or file.get("data_root", str)),
            repo_root=repo_root,
            profile=get("PROFILE") or file.get("profile", str),
            uv_bin=get("UV_BIN") or file.get("parsers.uv_bin", str),
            mineru_url=get("MINERU_URL") or file.text_or_none("parsers.mineru_url"),
            paddle_url=get("PADDLE_URL") or file.text_or_none("parsers.paddle_url"),
            paddle_render_dpi=number("PADDLE_RENDER_DPI", file.get("parsers.paddle_render_dpi", int), int),
            subprocess_timeout_s=number("SUBPROCESS_TIMEOUT_S", file.get("parsers.subprocess_timeout_s", float), float),
            http_timeout_s=number("HTTP_TIMEOUT_S", file.get("parsers.http_timeout_s", float), float),
            paddle_vl_backend=get("PADDLE_VL_BACKEND") or file.text_or_none("parsers.paddle_vl_backend"),
            paddle_vl_server_url=get("PADDLE_VL_SERVER_URL") or file.text_or_none("parsers.paddle_vl_server_url"),
            paddle_vl_model_name=get("PADDLE_VL_MODEL_NAME") or file.text_or_none("parsers.paddle_vl_model_name"),
            llm_base_url=llm_base_url,
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
            llm_inventory_reasoning_effort=_parse_inventory_reasoning_effort(
                get("LLM_INVENTORY_REASONING_EFFORT") or file.text_or_none("llm.inventory_reasoning_effort"),
                file.path,
            ),
            llm_concurrency=_positive(
                number("LLM_CONCURRENCY", file.get("llm.concurrency", int), int),
                "llm.concurrency",
                file.path,
            ),
            llm_max_in_flight=_positive(
                number("LLM_MAX_IN_FLIGHT", file.get("llm.max_in_flight", int), int), "llm.max_in_flight", file.path
            ),
            llm_retry_attempts=_positive(
                number("LLM_RETRY_ATTEMPTS", file.get("llm.retry_attempts", int), int), "llm.retry_attempts", file.path
            ),
            llm_retry_backoff_s=number("LLM_RETRY_BACKOFF_S", file.get("llm.retry_backoff_s", float), float),
            llm_offline=_parse_bool("LLM_OFFLINE", get("LLM_OFFLINE"), file.get("llm.offline", bool)),
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
            max_parallel_documents=_positive(
                number("WEB_MAX_PARALLEL_DOCUMENTS", file.get("web.max_parallel_documents", int), int),
                "web.max_parallel_documents",
                file.path,
            ),
            web_profiles=_parse_names(get("WEB_PROFILES"), file.names_or_none("web.profiles")),
            max_upload_bytes=number("MAX_UPLOAD_MB", file.get("server.max_upload_mb", int), int) * 1024 * 1024,
            page_dpi=number("PAGE_DPI", file.get("server.page_dpi.default", int), int),
            page_dpi_min=number("PAGE_DPI_MIN", file.get("server.page_dpi.min", int), int),
            page_dpi_max=number("PAGE_DPI_MAX", file.get("server.page_dpi.max", int), int),
            overlay_dpi=number("OVERLAY_DPI", file.get("overlay.dpi", int), int),
            figures_enabled=_parse_bool("FIGURES_ENABLED", get("FIGURES_ENABLED"), file.get("figures.enabled", bool)),
            figures_model=get("FIGURES_MODEL") or file.get("figures.model", str),
            figures_max_per_document=_positive(
                number("FIGURES_MAX_PER_DOCUMENT", file.get("figures.max_per_document", int), int),
                "figures.max_per_document",
                file.path,
            ),
            figures_dpi=_positive(number("FIGURES_DPI", file.get("figures.dpi", int), int), "figures.dpi", file.path),
            figures_max_pixels=_positive(
                number("FIGURES_MAX_PIXELS", file.get("figures.max_pixels", int), int),
                "figures.max_pixels",
                file.path,
            ),
            figures_timeout_s=_positive_seconds(
                number("FIGURES_TIMEOUT_S", file.get("figures.timeout_s", float), float), "figures.timeout_s", file.path
            ),
            vlm_enabled=_parse_bool("VLM_ENABLED", get("VLM_ENABLED"), file.get("vlm.enabled", bool)),
            vlm_base_url=get("VLM_BASE_URL") or file.get("vlm.base_url", str),
            vlm_model=get("VLM_MODEL") or file.get("vlm.model", str),
            vlm_timeout_s=_positive_seconds(
                number("VLM_TIMEOUT_S", file.get("vlm.timeout_s", float), float), "vlm.timeout_s", file.path
            ),
            vlm_temperature=number("VLM_TEMPERATURE", file.get("vlm.temperature", float), float),
            vlm_max_tokens=_positive(
                number("VLM_MAX_TOKENS", file.get("vlm.max_tokens", int), int), "vlm.max_tokens", file.path
            ),
            vlm_crop_dpi=_positive(
                number("VLM_CROP_DPI", file.get("vlm.crop_dpi", int), int), "vlm.crop_dpi", file.path
            ),
            vlm_crop_padding=_fraction(
                number("VLM_CROP_PADDING", file.get("vlm.crop_padding", float), float), "vlm.crop_padding", file.path
            ),
            vlm_crop_max_pixels=_positive(
                number("VLM_CROP_MAX_PIXELS", file.get("vlm.crop_max_pixels", int), int),
                "vlm.crop_max_pixels",
                file.path,
            ),
            vlm_policy=_parse_policy(get("VLM_POLICY") or file.get("vlm.policy", str), file.path),
            vlm_context_blocks=_non_negative(
                number("VLM_CONTEXT_BLOCKS", file.get("vlm.context_blocks", int), int),
                "vlm.context_blocks",
                file.path,
            ),
            vlm_fill_blanks=_parse_bool("VLM_FILL_BLANKS", get("VLM_FILL_BLANKS"), file.get("vlm.fill_blanks", bool)),
            vlm_concurrency=_positive(
                number("VLM_CONCURRENCY", file.get("vlm.concurrency", int), int), "vlm.concurrency", file.path
            ),
            supervisor_enabled=_parse_bool(
                "SUPERVISOR_ENABLED", get("SUPERVISOR_ENABLED"), file.get("supervisor.enabled", bool)
            ),
            # Null in the file means the LLM's endpoint, whatever it was set to, so the LLM's key (the default
            # when no key variable is named) is never sent to another provider by a stale URL.
            supervisor_base_url=(
                get("SUPERVISOR_BASE_URL") or file.text_or_none("supervisor.base_url") or llm_base_url
            ).rstrip("/"),
            supervisor_model=get("SUPERVISOR_MODEL") or file.get("supervisor.model", str),
            supervisor_api_key_env=supervisor_key_env,
            supervisor_api_key=((env.get(supervisor_key_env) or "").strip() or None) if supervisor_key_env else None,
            supervisor_min_confidence=number(
                "SUPERVISOR_MIN_CONFIDENCE", file.get("supervisor.min_confidence", float), float
            ),
            supervisor_vote_threshold=number(
                "SUPERVISOR_VOTE_THRESHOLD", file.get("supervisor.vote_threshold", float), float
            ),
            supervisor_timeout_s=_positive_seconds(
                number("SUPERVISOR_TIMEOUT_S", file.get("supervisor.timeout_s", float), float),
                "supervisor.timeout_s",
                file.path,
            ),
            # File-only: a verdict threshold is not something to flip per invocation.
            ambiguous_match_confidence=file.get("comparison.ambiguous_match_confidence", float),
        )
        _check_ranges(settings, file.path)
        return settings

    def require_supervisor_api_key(self) -> str:
        """The supervisor's key: the variable ``supervisor.api_key_env`` names, or the LLM's key when none is
        named. A named variable that is unset is an error here, not a silent fall-back to the wrong account."""
        if self.supervisor_api_key_env is None:
            return self.require_llm_api_key()
        if self.supervisor_api_key:
            return self.supervisor_api_key
        raise ConfigError(
            f"no supervisor API key: supervisor.api_key_env names {self.supervisor_api_key_env}, which is not set "
            f"(set it, or set supervisor.api_key_env to null to use the LLM's key)"
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


def _check_ranges(settings: Settings, source: Path) -> None:
    """Refuse a value outside what its setting can mean, naming the key and the file.

    Each of these otherwise fails far from its cause: a zero timeout reads as a broken endpoint, a port of
    0 as a bind error, and ``max_tokens >= context_tokens`` as a ContextBudgetError blaming every paper for
    being too long.
    """
    mb = settings.max_upload_bytes // (1024 * 1024)
    rules: list[tuple[bool, str]] = [
        (
            settings.paddle_render_dpi >= 1,
            f"parsers.paddle_render_dpi must be at least 1, got {settings.paddle_render_dpi}",
        ),
        (
            settings.subprocess_timeout_s > 0,
            f"parsers.subprocess_timeout_s must be positive, got {settings.subprocess_timeout_s}",
        ),
        (settings.http_timeout_s > 0, f"parsers.http_timeout_s must be positive, got {settings.http_timeout_s}"),
        (settings.llm_timeout_s > 0, f"llm.timeout_s must be positive, got {settings.llm_timeout_s}"),
        (settings.llm_context_tokens >= 1, f"llm.context_tokens must be at least 1, got {settings.llm_context_tokens}"),
        (settings.llm_max_tokens >= 1, f"llm.max_tokens must be at least 1, got {settings.llm_max_tokens}"),
        (
            settings.llm_max_tokens < settings.llm_context_tokens,
            f"llm.max_tokens ({settings.llm_max_tokens}) must be below llm.context_tokens "
            f"({settings.llm_context_tokens}): the reply is reserved out of the context window, and no prompt "
            "would be left",
        ),
        (
            0 <= settings.llm_temperature <= 2,
            f"llm.temperature must be between 0 and 2, got {settings.llm_temperature}",
        ),
        (
            settings.llm_retry_backoff_s >= 0,
            f"llm.retry_backoff_s must not be negative, got {settings.llm_retry_backoff_s}",
        ),
        (1 <= settings.server_port <= 65535, f"server.port must be between 1 and 65535, got {settings.server_port}"),
        (settings.max_upload_bytes >= 1024 * 1024, f"server.max_upload_mb must be at least 1, got {mb}"),
        (settings.page_dpi_min >= 1, f"server.page_dpi.min must be at least 1, got {settings.page_dpi_min}"),
        (
            settings.page_dpi_min <= settings.page_dpi <= settings.page_dpi_max,
            f"server.page_dpi.default ({settings.page_dpi}) must lie between server.page_dpi.min "
            f"({settings.page_dpi_min}) and server.page_dpi.max ({settings.page_dpi_max})",
        ),
        (settings.overlay_dpi >= 1, f"overlay.dpi must be at least 1, got {settings.overlay_dpi}"),
        (
            0 <= settings.ambiguous_match_confidence <= 1,
            f"comparison.ambiguous_match_confidence must be between 0 and 1, got {settings.ambiguous_match_confidence}",
        ),
        (
            0 <= settings.supervisor_min_confidence <= settings.supervisor_vote_threshold <= 1,
            "supervisor.min_confidence and supervisor.vote_threshold must satisfy 0 <= min_confidence <= "
            f"vote_threshold <= 1, got {settings.supervisor_min_confidence} and {settings.supervisor_vote_threshold}",
        ),
    ]
    for ok, message in rules:
        if not ok:
            raise ConfigError(f"{message} (set in {source} or the matching {ENV_PREFIX} variable)")


def _positive(value: int, dotted: str, source: Path) -> int:
    """A count that must be at least one. Zero passes, zero retries or zero candidate blocks all fail deep
    inside a loop with an error that names neither the setting nor the file it came from."""
    if value < 1:
        raise ConfigError(
            f"{dotted} must be at least 1, got {value} (set in {source} or the matching {ENV_PREFIX} variable)"
        )
    return value


def _positive_seconds(value: float, dotted: str, source: Path) -> float:
    """A timeout of zero makes every request fail at once, which reads as a broken endpoint, not a setting."""
    if value <= 0:
        raise ConfigError(
            f"{dotted} must be positive, got {value} (set in {source} or the matching {ENV_PREFIX} variable)"
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


def _parse_policy(raw: str, source: Path) -> ValidationPolicy:
    """An unknown policy names the ones that exist. Falling back to the default here would quietly check a
    different set of values than the one asked for, and the stored report would not say so."""
    if raw not in VALIDATION_POLICIES:
        policies = ", ".join(VALIDATION_POLICIES)
        raise ConfigError(
            f"vlm.policy is {raw!r}, expected one of {policies} (set in {source} or {ENV_PREFIX}VLM_POLICY)"
        )
    return cast(ValidationPolicy, raw)


def _fraction(value: float, dotted: str, source: Path) -> float:
    """A page fraction. Negative padding would shrink the crop inside the box the parser found, which is the
    one thing padding exists to prevent."""
    if not 0.0 <= value < 1.0:
        raise ConfigError(
            f"{dotted} must be at least 0 and below 1, got {value} "
            f"(set in {source} or the matching {ENV_PREFIX} variable)"
        )
    return value


def _non_negative(value: int, dotted: str, source: Path) -> int:
    """A count that may be zero: no context blocks is a meaningful choice (the cited blocks alone)."""
    if value < 0:
        raise ConfigError(
            f"{dotted} must be at least 0, got {value} (set in {source} or the matching {ENV_PREFIX} variable)"
        )
    return value


def _parse_reasoning_effort(raw: str | None, source: Path) -> ReasoningEffort | None:
    """``null`` (or an unset variable) means "omit the parameter"; anything else must be one we know the
    endpoint accepts, named here rather than discovered as a 400 halfway through a paper."""
    if raw is None:
        return None
    return _known_effort(raw, source, dotted="llm.reasoning_effort", variable="LLM_REASONING_EFFORT", extra="null")


def _parse_inventory_reasoning_effort(raw: str | None, source: Path) -> InventoryReasoningEffort:
    """The override has three outcomes, so it has three spellings. ``null`` (the shipped value) and the
    word ``inherit`` both mean "send the request the client would have sent"; ``omit`` is the explicit way
    to ask for no ``reasoning_effort`` parameter on this one question while the client still sends one on
    the others; anything else is the effort to send.
    """
    if raw is None or raw == INHERIT.value:
        return INHERIT
    if raw == "omit":
        return None
    return _known_effort(
        raw,
        source,
        dotted="llm.inventory_reasoning_effort",
        variable="LLM_INVENTORY_REASONING_EFFORT",
        extra="null, inherit, omit",
    )


def _known_effort(raw: str, source: Path, *, dotted: str, variable: str, extra: str) -> ReasoningEffort:
    if raw not in REASONING_EFFORTS:
        efforts = ", ".join(REASONING_EFFORTS)
        raise ConfigError(
            f"{dotted} is {raw!r}, expected {extra} or one of {efforts} (set in {source} or {ENV_PREFIX}{variable})"
        )
    return cast(ReasoningEffort, raw)


def _parse_names(raw: str | None, default: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """A comma-separated list from the environment; unset (or only commas) leaves ``default`` -- the file's
    value, always parsed and validated whether or not the environment goes on to override it, so a malformed
    ``web.profiles`` is loud even behind an override. ``"-"`` is the one way this variable can spell "no extra
    profiles" (only the default): an empty string already means unset, so it cannot."""
    if raw is not None and raw.strip() == "-":
        return ()
    names = tuple(part.strip() for part in (raw or "").split(",") if part.strip())
    return names or default


def _parse_bool(name: str, raw: str | None, default: bool) -> bool:
    """An on/off variable. Anything but the usual spellings is refused: "ture" silently reading as off would
    skip a stage somebody asked for."""
    if raw is None:
        return default
    word = raw.lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    raise ConfigError(f"environment variable {ENV_PREFIX}{name} must be true or false, got {raw!r}")


def _parse_number[T: (int, float)](name: str, raw: str | None, default: T, kind: type[T]) -> T:
    if raw is None:
        return default
    try:
        return kind(raw)
    except ValueError as exc:
        raise ConfigError(f"environment variable {ENV_PREFIX}{name} is not a valid {kind.__name__}: {raw!r}") from exc
