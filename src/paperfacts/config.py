"""Runtime configuration, entirely from ``PAPERFACTS_*`` environment variables. There is no config file.

Only three things need configuring: where data lives, where the parsers are, and how to reach the LLM.
An empty ``*_url`` means run the script in ``runners/`` as a subprocess (a workstation); a non-empty one
means call a long-running service (a GPU server). :mod:`paperfacts.workflow` picks the implementation from
that, so nothing above it has to know which environment it is in.

``from_env`` only reads ``environ``. The API key file is not touched until
:meth:`Settings.require_llm_api_key` is called, so commands that never reach the LLM never stat it, and
unit tests do not behave differently on a machine that happens to have a key lying around.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from paperfacts.errors import ConfigError
from paperfacts.extraction.extractor import DEFAULT_CONTEXT_TOKENS

ENV_PREFIX = "PAPERFACTS_"

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
# Any OpenAI-compatible endpoint works. DeepSeek is the default: cheap, long context, JSON mode.
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-chat"
DEFAULT_LLM_TIMEOUT_S = 300.0
# The key stays on the machine: environment first, then this file in the repository root (gitignored).
DEFAULT_API_KEY_FILENAME = "deepseek_api_key"


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
    llm_context_tokens: int = DEFAULT_CONTEXT_TOKENS
    # Extract each lane this many times and keep what a majority of passes agree on. Costs one LLM call
    # per pass, so it stays at 1 unless a run explicitly asks for more.
    extraction_passes: int = 1

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Read ``PAPERFACTS_*``. An empty string counts as unset; a bad number names the variable.

        Key resolution order: ``PAPERFACTS_LLM_API_KEY``, then ``DEEPSEEK_API_KEY``. With neither, record
        where the key file should be (``PAPERFACTS_LLM_API_KEY_FILE``, else ``deepseek_api_key`` in the
        repository root) and read it only when it is actually needed.
        """
        env = os.environ if environ is None else environ

        def get(name: str) -> str | None:
            return (env.get(ENV_PREFIX + name) or "").strip() or None

        repo_root = Path(get("REPO_ROOT") or DEFAULT_REPO_ROOT)
        key_file = get("LLM_API_KEY_FILE")
        return cls(
            data_root=Path(get("DATA_ROOT") or "data"),
            repo_root=repo_root,
            uv_bin=get("UV_BIN") or "uv",
            mineru_url=get("MINERU_URL"),
            paddle_url=get("PADDLE_URL"),
            paddle_render_dpi=_parse_number("PADDLE_RENDER_DPI", get("PADDLE_RENDER_DPI"), DEFAULT_RENDER_DPI, int),
            subprocess_timeout_s=_parse_number(
                "SUBPROCESS_TIMEOUT_S", get("SUBPROCESS_TIMEOUT_S"), DEFAULT_SUBPROCESS_TIMEOUT_S, float
            ),
            http_timeout_s=_parse_number("HTTP_TIMEOUT_S", get("HTTP_TIMEOUT_S"), DEFAULT_HTTP_TIMEOUT_S, float),
            paddle_vl_backend=get("PADDLE_VL_BACKEND"),
            paddle_vl_server_url=get("PADDLE_VL_SERVER_URL"),
            paddle_vl_model_name=get("PADDLE_VL_MODEL_NAME"),
            llm_base_url=(get("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).rstrip("/"),
            llm_model=get("LLM_MODEL") or DEFAULT_LLM_MODEL,
            llm_api_key=get("LLM_API_KEY") or (env.get("DEEPSEEK_API_KEY") or "").strip() or None,
            llm_api_key_file=Path(key_file) if key_file else repo_root / DEFAULT_API_KEY_FILENAME,
            llm_timeout_s=_parse_number("LLM_TIMEOUT_S", get("LLM_TIMEOUT_S"), DEFAULT_LLM_TIMEOUT_S, float),
            llm_context_tokens=_parse_number(
                "LLM_CONTEXT_TOKENS", get("LLM_CONTEXT_TOKENS"), DEFAULT_CONTEXT_TOKENS, int
            ),
            extraction_passes=_parse_number("EXTRACTION_PASSES", get("EXTRACTION_PASSES"), 1, int),
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
            "no LLM API key: set PAPERFACTS_LLM_API_KEY / DEEPSEEK_API_KEY, "
            f"or put the key in {key_file} (that path is gitignored)"
        )


def _parse_number[T: (int, float)](name: str, raw: str | None, default: T, kind: type[T]) -> T:
    if raw is None:
        return default
    try:
        return kind(raw)
    except ValueError as exc:
        raise ConfigError(f"environment variable {ENV_PREFIX}{name} is not a valid {kind.__name__}: {raw!r}") from exc
