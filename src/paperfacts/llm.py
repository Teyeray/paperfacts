"""LLM client: OpenAI-compatible chat completions (DeepSeek by default), JSON mode, cache, retries.

The cache key is ``sha256(base_url + the entire request payload)``, so an identical request is paid for
once and waited for once. Hashing the whole payload rather than a hand-picked subset means "we forgot to
put that parameter in the cache key" cannot happen: change the prompt, the model, or the temperature and
the key changes with it.

``refresh=True`` skips reading the cache but still writes it -- that is how ``--force`` genuinely re-asks
the model instead of replaying an answer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import httpx
from pydantic import BaseModel, ValidationError

from paperfacts.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_S,
    DEFAULT_TEMPERATURE,
)
from paperfacts.errors import LlmError, LlmResponseError

logger = logging.getLogger(__name__)

# Which HTTP statuses are worth trying again: the ones that mean "later", never the ones that mean "wrong".
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class LlmResult:
    text: str
    usage: dict[str, int]
    cached: bool


class LlmClient(Protocol):
    """Extraction and sample matching depend on this one method: system + user in, JSON text out.

    The three attributes are part of the protocol because they decide what the model is asked, and so belong
    in the cache key of anything derived from the answer. Reading them off the client rather than passing
    them alongside it is what keeps the key honest: it names what was actually sent.
    """

    model: str
    temperature: float
    max_tokens: int

    def complete_json(self, *, system: str, user: str, refresh: bool = False, cache_salt: str = "") -> LlmResult: ...


class OpenAICompatibleClient:
    """Talks to ``POST {base_url}/chat/completions``."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_s: float,
        cache_dir: Path | None = None,
        client: httpx.Client | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.cache_dir = cache_dir
        self.client = client or httpx.Client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts
        self.retry_backoff_s = retry_backoff_s
        self._sleep = sleep  # injectable so tests do not actually sleep

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete_json(self, *, system: str, user: str, refresh: bool = False, cache_salt: str = "") -> LlmResult:
        payload = self.payload(system=system, user=user)
        key = self.cache_key(payload, cache_salt=cache_salt)
        if not refresh:
            cached = self._read_cache(key)
            if cached is not None:
                return cached

        data = self._post_with_retry(payload)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"response has no choices[0].message.content: {str(data)[:300]}") from exc
        if not text or not text.strip():
            raise LlmError("the model returned empty content (usually max_tokens truncation in JSON mode)")
        usage = {k: int(v) for k, v in (data.get("usage") or {}).items() if isinstance(v, int | float)}
        result = LlmResult(text=text, usage=usage, cached=False)
        self._write_cache(key, result)
        return result

    def payload(self, *, system: str, user: str) -> dict[str, Any]:
        """The complete request body. The cache key hashes this, so no parameter can escape the key."""
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # JSON mode: the reply is exactly one JSON object. DeepSeek requires the word "json" to appear
            # in the prompt for this to be accepted; prompts.py guarantees that.
            "response_format": {"type": "json_object"},
        }

    def cache_key(self, payload: dict[str, Any], *, cache_salt: str = "") -> str:
        """Key for this request.

        ``cache_salt`` distinguishes repeated identical requests -- self-consistency passes send the exact
        same payload and want a separate answer each time. It is deliberately not part of the payload, so
        the bytes on the wire stay identical and replaying a run is still free. An empty salt is omitted
        from the material so keys written before salts existed still resolve.
        """
        material: dict[str, Any] = {"base_url": self.base_url, "payload": payload}
        if cache_salt:
            material["salt"] = cache_salt
        return hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        for attempt in range(1, self.retry_attempts + 1):
            error: LlmError
            try:
                response = self.client.post(url, json=payload, headers=headers, timeout=self.timeout_s)
            except httpx.HTTPError as exc:
                error = LlmError(f"{type(exc).__name__}: {exc}")
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise LlmError(f"response is not JSON: {response.text[:300]}") from exc
                error = LlmError(f"HTTP {response.status_code}: {response.text[:300]}")
                if response.status_code not in RETRY_STATUS:
                    raise error
            if attempt == self.retry_attempts:
                raise error
            delay = self.retry_backoff_s * 2 ** (attempt - 1)
            logger.warning("llm retry %d/%d in %.0fs: %s", attempt, self.retry_attempts, delay, error)
            self._sleep(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _cache_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> LlmResult | None:
        path = self._cache_path(key)
        if path is None or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return LlmResult(text=data["text"], usage=dict(data.get("usage") or {}), cached=True)
        except (ValueError, KeyError, TypeError):
            logger.warning("llm cache entry unreadable, ignoring: %s", path)
            return None

    def _write_cache(self, key: str, result: LlmResult) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model": self.model, "text": result.text, "usage": result.usage}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def complete_validated[M: BaseModel](
    client: LlmClient,
    model_cls: type[M],
    *,
    system: str,
    user: str,
    repair: Callable[[str, str], str],
    refresh: bool = False,
    cache_salt: str = "",
) -> tuple[M, str, dict[str, int]]:
    """Ask for JSON that validates against ``model_cls``, giving the model one chance to fix itself.

    An invalid first answer is sent back with its validation error; only a second failure gives up. The
    caller supplies ``repair(previous_text, error)`` to build the follow-up prompt. Returns the parsed
    model, the raw text, and the summed token usage of both calls.
    """
    first = client.complete_json(system=system, user=user, refresh=refresh, cache_salt=cache_salt)
    usage = dict(first.usage)
    try:
        return model_cls.model_validate_json(first.text), first.text, usage
    except ValidationError as exc:
        error = str(exc)
    logger.warning("%s response invalid, asking for a repair: %s", model_cls.__name__, error[:300])
    second = client.complete_json(system=system, user=repair(first.text, error), refresh=refresh, cache_salt=cache_salt)
    for key, value in second.usage.items():
        usage[key] = usage.get(key, 0) + value
    try:
        return model_cls.model_validate_json(second.text), second.text, usage
    except ValidationError as exc:
        raise LlmResponseError(
            f"the model twice failed to produce valid {model_cls.__name__} JSON: {str(exc)[:500]}"
        ) from exc
