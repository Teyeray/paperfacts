"""LLM client: OpenAI-compatible chat completions (DeepSeek by default), JSON mode, cache, retries.

The cache key is ``sha256(base_url + the entire request payload)``, so an identical request is paid for
once and waited for once. Hashing the whole payload rather than a hand-picked subset means "we forgot to
put that parameter in the cache key" cannot happen: change the prompt, the model, or the temperature and
the key changes with it.

``refresh=True`` skips reading the cache but still writes it -- that is how ``--force`` genuinely re-asks
the model instead of replaying an answer.

Only an answer the caller accepts is cached (``accept``; :func:`complete_validated` passes its schema check).
A cached answer that fails the check is a miss and is asked again, and a fresh one that fails it is returned
but never written: one bad reply must cost one repair, not become the answer of every later run. A reply cut
off at ``max_tokens`` is never an answer and raises.

Vision requests (:meth:`OpenAICompatibleClient.complete_vision`) carry one PNG as an OpenAI-style
``image_url`` content part. Their cache key hashes the request with the image replaced by its sha256: the
same bytes on the wire are still the same key, but a megabyte of base64 never lands in the key material or
in the cache entry.

Every request that reaches the network first takes a slot from :data:`IN_FLIGHT`, one limit shared by every
client in the process (see :class:`InFlightLimit`). A cache hit never touches it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import httpx
from pydantic import BaseModel, ValidationError

from paperfacts.config import (
    DEFAULT_LLM_MAX_IN_FLIGHT,
    DEFAULT_LLM_REASONING_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_S,
    DEFAULT_TEMPERATURE,
    INHERIT,
    Inherit,
    ReasoningEffort,
)
from paperfacts.errors import LlmError, LlmResponseError
from paperfacts.storage import write_text_atomic

logger = logging.getLogger(__name__)

# Which HTTP statuses are worth trying again: the ones that mean "later", never the ones that mean "wrong".
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

# The longest ``Retry-After`` honoured. A server asking for an hour would park a worker thread for that hour;
# a longer request is cut to this, and the attempt budget decides whether the call is given up.
MAX_RETRY_AFTER_S = 120.0

# Decides whether an answer's text is worth caching (and replaying from the cache).
Accept = Callable[[str], bool]


class InFlightLimit:
    """At most ``limit`` model requests on the wire at once, across every client that shares this object.

    The pools nest -- documents, then the two lanes and the figures stage, then one lane's field questions --
    and each level's own limit multiplies with the others; only a limit taken around the request itself
    bounds the product. A slot is held for exactly one HTTP call and never while waiting on anything else
    (another future, a retry's backoff), so no thread can hold a slot that some other slot-holder waits for,
    and the nesting cannot deadlock.

    A condition over a counter rather than a ``Semaphore``: the limit comes from the settings, which are read
    after this module has made the shared object, and a semaphore's size cannot change once it exists.
    """

    def __init__(self, limit: int) -> None:
        self._limit = _at_least_one(limit)
        self._active = 0
        self._changed = threading.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    def set_limit(self, limit: int) -> None:
        """A raised limit wakes the waiters at once; a lowered one lets the requests already out finish."""
        with self._changed:
            self._limit = _at_least_one(limit)
            self._changed.notify_all()

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._changed:
            while self._active >= self._limit:
                self._changed.wait()
            self._active += 1
        try:
            yield
        finally:
            with self._changed:
                self._active -= 1
                self._changed.notify()


def _at_least_one(limit: int) -> int:
    if limit < 1:
        raise ValueError(f"an in-flight limit must be at least 1, got {limit}")
    return limit


# The process-wide limit. Module-level because the thing it protects -- the endpoint's rate limit -- is
# shared by every client this process builds, whichever document, lane or stage built it.
IN_FLIGHT = InFlightLimit(DEFAULT_LLM_MAX_IN_FLIGHT)


def set_max_in_flight(limit: int) -> None:
    """Size the process-wide limit. Called once, where a process reads its settings (``create_app``, each
    CLI command), not by every client built: a client built from other settings -- a test, a one-off
    helper -- must not resize the limit under documents that are already running."""
    IN_FLIGHT.set_limit(limit)


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
    reasoning_effort: ReasoningEffort | None

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        refresh: bool = False,
        cache_salt: str = "",
        reasoning_effort: ReasoningEffort | Inherit | None = INHERIT,
        accept: Accept | None = None,
    ) -> LlmResult: ...


class VisionClient(Protocol):
    """What figure reading needs from a vision model: one image plus a question in, text out.

    Deliberately a separate protocol from :class:`LlmClient`. Extraction must never be handed an image (the
    two lanes are compared on the *text* the parsers produced, and an image would be a third source), and
    figure reading must never be handed a text-only client. ``model`` is recorded with the readings; the
    sampling that goes into ``figure_key`` is fixed where the stage's client is built.
    """

    model: str

    def complete_vision(self, *, system: str, user: str, image_png: bytes, refresh: bool = False) -> LlmResult: ...


class OpenAICompatibleClient:
    """Talks to ``POST {base_url}/chat/completions``; text-only JSON requests and single-image vision requests."""

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
        reasoning_effort: ReasoningEffort | None = DEFAULT_LLM_REASONING_EFFORT,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        sleep: Callable[[float], None] = time.sleep,
        in_flight: InFlightLimit = IN_FLIGHT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.cache_dir = cache_dir
        self.client = client or httpx.Client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.retry_attempts = retry_attempts
        self.retry_backoff_s = retry_backoff_s
        self._sleep = sleep  # injectable so tests do not actually sleep
        self.in_flight = in_flight

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        refresh: bool = False,
        cache_salt: str = "",
        reasoning_effort: ReasoningEffort | Inherit | None = INHERIT,
        accept: Accept | None = None,
    ) -> LlmResult:
        """One JSON answer. ``accept`` gates the cache both ways: a cached answer it rejects is asked again,
        a fresh one it rejects is returned to the caller (who repairs it) but not written."""
        payload = self.payload(system=system, user=user, reasoning_effort=reasoning_effort)
        key = self.cache_key(payload, cache_salt=cache_salt)
        if not refresh:
            cached = self._read_cache(key)
            if cached is not None:
                if accept is None or accept(cached.text):
                    return cached
                logger.warning("llm cache entry %s fails validation; asking again", key[:16])

        data = self._post_with_retry(payload)
        try:
            finish_reason = data["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            finish_reason = None  # _content says what is wrong with the shape
        if finish_reason == "length":
            # Truncated JSON is not an answer; caching it would replay the torn reply on every later run. A
            # response error, like an answer that failed validation twice: matching then records a failed
            # matching for this run instead of failing the paper, and extraction a failed field question.
            # Checked before the content, because a reply cut off before its first character is empty.
            raise LlmResponseError("the model's reply was cut off at max_tokens")
        text = _content(data, "the model")
        result = LlmResult(text=text, usage=_flat_usage(data.get("usage")), cached=False)
        if accept is None or accept(text):
            self._write_cache(key, result)
        return result

    def payload(
        self, *, system: str, user: str, reasoning_effort: ReasoningEffort | Inherit | None = INHERIT
    ) -> dict[str, Any]:
        """The complete request body. The cache key hashes this, so no parameter can escape the key.

        ``reasoning_effort`` overrides the client's own setting for this one request: ``INHERIT`` keeps it,
        ``None`` sends no such parameter at all, a value sends that one.
        A question that reasons far longer than its neighbours can therefore be given its own effort
        without changing any other request's bytes, and so without invalidating their cached answers.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # JSON mode: the reply is exactly one JSON object. DeepSeek requires the word "json" to appear
            # in the prompt for this to be accepted; prompts.py guarantees that.
            "response_format": {"type": "json_object"},
        }
        # Omitted rather than sent as null when unset: the bytes on the wire stay what they were before this
        # parameter existed, so every cached answer still resolves.
        effort = self.reasoning_effort if reasoning_effort is INHERIT else reasoning_effort
        if effort is not None:
            body["reasoning_effort"] = effort
        return body

    def complete_vision(self, *, system: str, user: str, image_png: bytes, refresh: bool = False) -> LlmResult:
        """Ask about one PNG. No JSON mode: not every vision endpoint accepts ``response_format``, and the
        figures stage parses the reply leniently anyway."""
        if not image_png:
            raise ValueError("complete_vision needs a non-empty PNG")
        payload = self.vision_payload(system=system, user=user, image_png=image_png)
        key = self.vision_cache_key(payload, image_png)
        if not refresh:
            cached = self._read_cache(key)
            if cached is not None:
                return cached
        data = self._post_with_retry(payload)
        text = _content(data, "the vision model")
        if data["choices"][0].get("finish_reason") == "length":
            # A reply cut off at max_tokens is not an answer. Caching it would serve the same torn JSON to
            # every later run; raising leaves the panel to be asked again.
            raise LlmError("the vision model's reply was cut off at max_tokens")
        result = LlmResult(text=text, usage=_flat_usage(data.get("usage")), cached=False)
        self._write_cache(key, result)
        return result

    def vision_payload(self, *, system: str, user: str, image_png: bytes) -> dict[str, Any]:
        """The multimodal request body: the image first, then the question, as the OpenAI vision API and its
        Model Studio imitation both accept it."""
        data_url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": user},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def vision_cache_key(self, payload: dict[str, Any], image_png: bytes) -> str:
        """The request hashed with the image stood in for by its digest.

        Hashing the base64 itself would be correct but wasteful: the key material would be the size of the
        image. The digest is exact -- the same PNG bytes give the same key, one different pixel another.
        """
        digest = hashlib.sha256(image_png).hexdigest()
        stripped = json.loads(json.dumps(payload))
        for message in stripped["messages"]:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        part["image_url"] = {"sha256": digest}
        material = {"base_url": self.base_url, "payload": stripped, "vision": True}
        return hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

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
            retry_after: float | None = None
            try:
                # The slot covers the call alone: a backoff below sleeps without one, so a request waiting
                # out a 429 never keeps another from going out.
                with self.in_flight.slot():
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
                retry_after = _retry_after(response)
            if attempt == self.retry_attempts:
                raise error
            delay = self.retry_backoff_s * 2 ** (attempt - 1)
            if retry_after is not None:
                # A 429 that says "try again in 30 s" is a promise, not a suggestion: backing off less
                # than the server asked just spends another attempt of the same fixed budget.
                delay = max(delay, retry_after)
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
        payload = json.dumps(
            {"model": self.model, "text": result.text, "usage": result.usage}, ensure_ascii=False, indent=2
        )
        # Atomic like every other on-disk write: a crash mid-write must not leave a torn entry. The
        # reader would tolerate one, but never creating it is cheaper than healing it.
        write_text_atomic(path, payload)


def _content(data: dict[str, Any], who: str) -> str:
    """The reply text of a chat completion, or an :class:`LlmError` for anything that is not one."""
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError(f"response has no choices[0].message.content: {str(data)[:300]}") from exc
    if not isinstance(text, str):
        # Some providers answer with a list of content parts; that is not the JSON text this client asked for.
        raise LlmError(f"{who} returned non-text content: {str(text)[:300]}")
    if not text.strip():
        raise LlmError(f"{who} returned empty content (usually max_tokens truncation in JSON mode)")
    return text


def _flat_usage(usage: Any) -> dict[str, int]:
    """The response's usage block as flat integers.

    Reasoning endpoints report the hidden reasoning tokens one level down, in
    ``completion_tokens_details.reasoning_tokens``; that number is most of what a slow question costs, so
    it is lifted to the top level as ``reasoning_tokens`` where the lane log and the page can show it.
    """
    flat: dict[str, int] = {}
    for key, value in (usage or {}).items():
        if isinstance(value, int | float):
            flat[key] = int(value)
        elif key == "completion_tokens_details" and isinstance(value, dict):
            reasoning = value.get("reasoning_tokens")
            if isinstance(reasoning, int | float):
                flat["reasoning_tokens"] = int(reasoning)
    return flat


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds the server asked us to wait, when it said so as a plain number.

    The HTTP-date form is rare in practice and the exponential backoff already errs long, so an
    unparseable header is simply ignored rather than interpreted. A wait longer than
    :data:`MAX_RETRY_AFTER_S` is capped there: a worker thread is not parked for an hour on a server's word.
    """
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return min(seconds, MAX_RETRY_AFTER_S) if seconds > 0 else None


def complete_validated[M: BaseModel](
    client: LlmClient,
    model_cls: type[M],
    *,
    system: str,
    user: str,
    repair: Callable[[str, str], str],
    refresh: bool = False,
    cache_salt: str = "",
    reasoning_effort: ReasoningEffort | Inherit | None = INHERIT,
) -> tuple[M, str, dict[str, int]]:
    """Ask for JSON that validates against ``model_cls``, giving the model one chance to fix itself.

    An invalid first answer is sent back with its validation error; only a second failure gives up. The
    caller supplies ``repair(previous_text, error)`` to build the follow-up prompt. Returns the parsed
    model, the raw text, and the summed token usage of both calls.

    Both requests pass the schema as ``accept``, so only the answer that validated is cached, and an invalid
    answer already in the cache (written before this rule) is asked again rather than replayed.
    """

    def accept(text: str) -> bool:
        try:
            model_cls.model_validate_json(text)
        except ValidationError:
            return False
        return True

    first = client.complete_json(
        system=system,
        user=user,
        refresh=refresh,
        cache_salt=cache_salt,
        reasoning_effort=reasoning_effort,
        accept=accept,
    )
    usage = dict(first.usage)
    try:
        return model_cls.model_validate_json(first.text), first.text, usage
    except ValidationError as exc:
        error = str(exc)
    logger.warning("%s response invalid, asking for a repair: %s", model_cls.__name__, error[:300])
    # The repair asks the same question again, so it gets the same effort: a retry must not silently
    # become a more expensive request than the one it is fixing.
    second = client.complete_json(
        system=system,
        user=repair(first.text, error),
        refresh=refresh,
        cache_salt=cache_salt,
        reasoning_effort=reasoning_effort,
        accept=accept,
    )
    for key, value in second.usage.items():
        usage[key] = usage.get(key, 0) + value
    try:
        return model_cls.model_validate_json(second.text), second.text, usage
    except ValidationError as exc:
        raise LlmResponseError(
            f"the model twice failed to produce valid {model_cls.__name__} JSON: {str(exc)[:500]}"
        ) from exc
