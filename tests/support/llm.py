"""A fake LLM client. Every extraction and matching unit test runs on it; no real call is ever made.

The extraction layer depends on exactly one method of the
:class:`paperfacts.llm.LlmClient` protocol, so "never call the real model" reduces to passing
this object in.

It doubles as an assertion surface: every (system, user) pair is recorded, which lets tests assert on the
prompt text itself -- that both lanes used the same system prompt, that a repair request carried the
previous error, that a matching prompt contained the sample conditions -- instead of mocking internals.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from paperfacts.config import DEFAULT_LLM_REASONING_EFFORT, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from paperfacts.llm import LlmResult

# Fake token counts. The numbers mean nothing; they only prove usage is recorded and summed.
DEFAULT_USAGE: dict[str, int] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}

# One canned response: plain text gets the default usage, or pass an LlmResult to control it.
Response = str | LlmResult
Responder = Callable[[str, str], Response]


@dataclass(frozen=True)
class LlmCall:
    """The arguments of one ``complete_json`` call."""

    system: str
    user: str
    refresh: bool = False
    cache_salt: str = ""


class FakeLlmClient:
    """An :class:`LlmClient` that returns canned text in order.

    ``responses`` is either a sequence consumed in call order, or a ``(system, user) -> text`` function
    for cases where the reply depends on the request, such as only returning valid JSON on the retry.
    """

    def __init__(
        self,
        responses: Sequence[Response] | Responder,
        *,
        model: str = "fake-model",
        usage: dict[str, int] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        reasoning_effort: str | None = DEFAULT_LLM_REASONING_EFFORT,
    ) -> None:
        self.model = model
        # Part of the LlmClient protocol: what the real client would send, and therefore what the cache key
        # for an extraction records. The defaults match an unedited config.json.
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.calls: list[LlmCall] = []
        self.closed = False
        # Passage mode may ask several field questions at once, so the recorder itself has to survive
        # concurrent callers. The lock covers only the bookkeeping; the responder runs outside it, which is
        # what lets a test's responder sleep and actually overlap.
        self._lock = threading.Lock()
        self._usage = dict(DEFAULT_USAGE if usage is None else usage)
        self._responder: Responder | None = responses if callable(responses) else None
        self._queue: list[Response] = [] if callable(responses) else list(responses)

    # ---- LlmClient protocol ----------------------------------------------------------

    def complete_json(self, *, system: str, user: str, refresh: bool = False, cache_salt: str = "") -> LlmResult:
        with self._lock:
            self.calls.append(LlmCall(system=system, user=user, refresh=refresh, cache_salt=cache_salt))
            index = len(self.calls) - 1
        if self._responder is not None:
            return self._as_result(self._responder(system, user))
        # A queue is answered by position, so it is only meaningful when calls are made one at a time --
        # which is why the concurrency tests below use a responder keyed on the prompt instead.
        if index >= len(self._queue):
            raise AssertionError(f"FakeLlmClient got call {index + 1} but only {len(self._queue)} responses queued")
        return self._as_result(self._queue[index])

    # ---- Context manager: callers use `with build_llm_client(settings) as client` -----

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeLlmClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- Views used by assertions -----------------------------------------------------

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def refreshes(self) -> list[bool]:
        return [call.refresh for call in self.calls]

    @property
    def systems(self) -> list[str]:
        return [call.system for call in self.calls]

    @property
    def users(self) -> list[str]:
        return [call.user for call in self.calls]

    def _as_result(self, response: Response) -> LlmResult:
        if isinstance(response, LlmResult):
            return response
        return LlmResult(text=response, usage=dict(self._usage), cached=False)
