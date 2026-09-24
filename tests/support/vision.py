"""A fake vision client: figure-reading tests hand this in and no image ever leaves the process.

It satisfies :class:`paperfacts.llm.VisionClient` and records every request, so a test can assert on the
prompt a panel was asked with and on the exact PNG bytes it was shown.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from paperfacts.errors import LlmError
from paperfacts.llm import LlmResult

USAGE: dict[str, int] = {"prompt_tokens": 900, "completion_tokens": 400, "total_tokens": 1300}

# An answer: text, a JSON-able mapping, or an exception to raise (a failed request).
Answer = str | Mapping[str, Any] | LlmError
Responder = Callable[[str, bytes], Answer]


@dataclass(frozen=True)
class VisionCall:
    system: str
    user: str
    image_png: bytes
    refresh: bool


class FakeVisionClient:
    """Answers by ``responder(user_prompt, image_png)``, or with one fixed answer for every panel."""

    def __init__(
        self,
        answer: Answer | Responder,
        *,
        model: str = "fake-vl",
    ) -> None:
        self.model = model
        self.calls: list[VisionCall] = []
        self.closed = False
        self._lock = threading.Lock()
        self._responder: Responder = answer if callable(answer) else (lambda user, image: answer)

    def complete_vision(self, *, system: str, user: str, image_png: bytes, refresh: bool = False) -> LlmResult:
        with self._lock:
            self.calls.append(VisionCall(system=system, user=user, image_png=image_png, refresh=refresh))
        answer = self._responder(user, image_png)
        if isinstance(answer, LlmError):
            raise answer
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return LlmResult(text=text, usage=dict(USAGE), cached=False)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> FakeVisionClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def chart_answer(
    *,
    field: str = "sheet_resistance",
    unit: str = "Ω/sq",
    scale: str = "linear",
    points: tuple[tuple[Any, float], ...] = ((100, 25.0), (200, 40.0)),
    series: tuple[str, ...] = ("Rs",),
    confidence: float = 0.9,
) -> dict[str, Any]:
    """A well-formed property-vs-condition answer: every series on the one left axis."""
    return {
        "chart_type": "property_vs_condition",
        "x_axis": {"quantity": "O2 flow", "unit": "sccm", "scale": "linear"},
        "y_axes": [{"id": "left", "field": field, "quantity": "Rs", "unit": unit, "scale": scale, "broken": False}],
        "series": [{"label": label, "y_axis": "left", "marker": "square"} for label in series],
        "points": [
            {"series": label, "x": x, "x_on_tick": True, "y": y, "y_error": None, "confidence": confidence}
            for label in series
            for x, y in points
        ],
    }


NOT_A_CHART: dict[str, Any] = {"chart_type": "not_property_vs_condition", "reason": "a transmittance spectrum"}
