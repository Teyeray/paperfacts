"""Every exception the package raises on purpose, in one place.

The CLI catches :class:`PaperFactsError` (plus ``FileNotFoundError``) and turns it into a single red line
with exit code 1. Anything else -- a programming error, a third-party ValueError -- keeps its traceback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paperfacts.models import Backend


class PaperFactsError(RuntimeError):
    """An expected failure that should be reported to the user as a single line."""


class ConfigError(PaperFactsError):
    """Configuration is missing or invalid (for example, no LLM key)."""


class ParserError(PaperFactsError):
    """A parser run failed. ``stage`` names where: launch / run / timeout / http / output / cache."""

    def __init__(self, backend: Backend, stage: str, detail: str) -> None:
        self.backend = backend
        self.stage = stage
        self.detail = detail
        super().__init__(f"[{backend}] {stage} failed: {detail}")


class LlmError(PaperFactsError):
    """An LLM call failed: network, HTTP status, or response shape."""


class LlmResponseError(LlmError):
    """The model failed twice to produce JSON matching the required schema."""


class LlmOfflineMiss(LlmError):
    """``llm.offline`` is on and the request has no cached answer: nothing was sent."""


class ContextBudgetError(PaperFactsError):
    """The paper does not fit in the model's context window."""


class Cancelled(PaperFactsError):
    """A run was asked to stop before it finished: the paper it served failed elsewhere, or a batch was
    interrupted. Whatever it had already paid for is in the caches, so the next run picks it up for free."""
