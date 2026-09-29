"""Supervisor verification: scoring lane extractions to detect hallucinations and guide retries.

The supervisor is a separate, configurable model (typically smaller/local) that scores each extracted value
against its cited passages. It runs lazily — only on disagreements and borderline agreements — to target
the ~40% of extractions where verification matters most.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, ConfigDict, Field

from paperfacts.errors import SupervisorError

if TYPE_CHECKING:
    from paperfacts.config import Settings
    from paperfacts.fields import FieldSpec
    from paperfacts.records import FieldValue

logger = logging.getLogger(__name__)


class SupervisorResult(BaseModel):
    """One supervisor scoring result."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0, description="confidence that the value is correct")
    flag: str = Field(description="correct | plausible | unit_mismatch | value_not_in_passage")
    critique: str = Field(default="", description="detailed reasoning when score < threshold; empty otherwise")


@dataclass(frozen=True)
class SupervisorClient:
    """OpenAI-compatible client for supervisor scoring. Separate from the extraction client."""

    base_url: str
    api_key: str
    model: str
    timeout_s: float = 30.0

    def score_value(
        self,
        field: FieldSpec,
        value: FieldValue,
        passage: str,
        *,
        request_critique: bool = False,
    ) -> SupervisorResult:
        """Score one value against its cited passage. ``request_critique`` asks for detailed reasoning."""
        system_prompt = _supervisor_system_prompt()
        user_prompt = _supervisor_user_prompt(field, value, passage, request_critique=request_critique)

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise SupervisorError("timeout", f"request timed out after {self.timeout_s}s") from exc
        except httpx.HTTPStatusError as exc:
            raise SupervisorError("http", f"HTTP {exc.response.status_code}: {exc.response.text[:500]}") from exc
        except httpx.RequestError as exc:
            raise SupervisorError("http", str(exc)) from exc

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return SupervisorResult.model_validate(parsed)
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
            raise SupervisorError("response", f"malformed response: {exc}") from exc


def should_supervise(
    lanes_agree: bool,
    value_a: FieldValue | None,
    value_b: FieldValue | None,
    spec: FieldSpec,
) -> bool:
    """Lazy trigger: only supervise on disagreement or borderline agreement.

    Returns True when:
    1. Lanes disagree
    2. Lanes agree but the value is within 10% of a plausible range boundary
    """
    if not lanes_agree:
        return True

    # Both lanes agree — check if the value is borderline
    if value_a is None or value_a.value is None or spec.valid_range is None:
        return False

    val = value_a.value
    low, high = spec.valid_range

    # Within 10% of either boundary
    if low is not None:
        margin = abs(high - low) * 0.1 if high is not None else abs(low) * 0.1
        if val <= low + margin:
            return True

    if high is not None:
        margin = abs(high - low) * 0.1 if low is not None else abs(high) * 0.1
        if val >= high - margin:
            return True

    return False


def _supervisor_system_prompt() -> str:
    """Domain-free supervisor prompt. Returns structured JSON only."""
    return """You verify extracted values against cited passages from a scientific paper.

Output ONLY a JSON object:
{
  "score": <0.0-1.0>,
  "flag": "<correct|plausible|unit_mismatch|value_not_in_passage>",
  "critique": "<detailed reasoning if score < 0.6, else empty string>"
}

Rules:
1. "correct": value and unit appear in the passage exactly as extracted
2. "plausible": value is present but unit differs, or minor OCR variation
3. "unit_mismatch": value is correct but unit is wrong or missing
4. "value_not_in_passage": the cited text does not contain this value
5. Score 1.0 for correct, 0.7-0.9 for plausible, 0.3-0.6 for unit issues, 0.0-0.2 when value is absent
6. Critique is required when score < 0.6; explain what is wrong and what the passage actually says

Return the JSON object only."""


def _supervisor_user_prompt(
    field: FieldSpec,
    value: FieldValue,
    passage: str,
    *,
    request_critique: bool = False,
) -> str:
    """Construct user prompt for supervisor scoring."""
    critique_note = "\nProvide detailed critique in the 'critique' field." if request_critique else ""
    return f"""Field: {field.name}
Description: {field.description}

Extracted value: {value.value_raw}
Extracted unit: {value.unit_raw or "(none)"}

Cited passage:
{passage}
{critique_note}

Return the JSON object now."""


def build_supervisor_client(settings: Settings) -> SupervisorClient | None:
    """Build supervisor client from settings, or None if disabled or provider is 'anthropic' (uses main client)."""
    if not settings.supervisor_enabled:
        return None

    if settings.supervisor_provider == "anthropic":
        # Falls back to main llm.complete path — caller handles this
        return None

    # OpenAI-compatible mode
    api_key = os.environ.get(settings.supervisor_api_key_env, "")
    if not api_key:
        logger.warning(
            "supervisor enabled with openai_compat provider but %s not set; supervisor disabled",
            settings.supervisor_api_key_env,
        )
        return None

    return SupervisorClient(
        base_url=settings.supervisor_base_url,
        api_key=api_key,
        model=settings.supervisor_model,
        timeout_s=settings.supervisor_timeout_s,
    )
