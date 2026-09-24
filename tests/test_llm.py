"""The OpenAI-compatible LLM client: request shape, error handling, retries, content-addressed cache.

Everything runs on ``httpx.MockTransport``: not one real network request goes out, and ``sleep`` is
injected away too, so the retry tests do not actually wait 14 seconds.

The two most expensive things at this layer are **money** and **time**, so the boundaries of caching and
retrying are pinned down one by one: one missed cache means paying again, one extra retry of a 4xx just
sends the same bad request four times instead of one.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from paperfacts.config import DEFAULT_RETRY_ATTEMPTS as RETRY_ATTEMPTS
from paperfacts.config import DEFAULT_RETRY_BACKOFF_S as RETRY_BACKOFF_S
from paperfacts.errors import LlmError
from paperfacts.llm import OpenAICompatibleClient
from support.http import make_client, recording_client

BASE_URL = "https://api.example.com/v1"
API_KEY = "sk-test-key"


def cache_key_of(llm: OpenAICompatibleClient, system: str, user: str) -> str:
    return llm.cache_key(llm.payload(system=system, user=user))


def chat_response(content: str = '{"samples": []}', usage: dict | None = None) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage if usage is not None else {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


class FakeSleep:
    """Records the backoff durations without actually sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def make_llm(
    handler=None, *, cache_dir: Path | None = None, sleep: FakeSleep | None = None, model: str = "deepseek-chat"
) -> OpenAICompatibleClient:
    client = make_client(handler or (lambda request: httpx.Response(200, json=chat_response())))
    return OpenAICompatibleClient(
        BASE_URL,
        API_KEY,
        model,
        timeout_s=30.0,
        cache_dir=cache_dir,
        client=client,
        sleep=sleep or FakeSleep(),
    )


# ---- Request shape -------------------------------------------------------------------


def test_the_request_goes_to_the_chat_completions_endpoint():
    client, requests = recording_client(lambda request: httpx.Response(200, json=chat_response()))
    llm = OpenAICompatibleClient(BASE_URL, API_KEY, "m", timeout_s=5.0, client=client)

    llm.complete_json(system="S", user="U")

    assert str(requests[0].url) == f"{BASE_URL}/chat/completions"


def test_a_trailing_slash_on_the_base_url_does_not_double_up():
    client, requests = recording_client(lambda request: httpx.Response(200, json=chat_response()))
    llm = OpenAICompatibleClient(BASE_URL + "/", API_KEY, "m", timeout_s=5.0, client=client)

    llm.complete_json(system="S", user="U")

    assert str(requests[0].url) == f"{BASE_URL}/chat/completions"


def test_the_payload_carries_the_model_the_two_messages_and_json_mode():
    client, requests = recording_client(lambda request: httpx.Response(200, json=chat_response()))
    llm = OpenAICompatibleClient(BASE_URL, API_KEY, "deepseek-chat", timeout_s=5.0, client=client)

    llm.complete_json(system="SYSTEM TEXT", user="USER TEXT")
    payload = json.loads(requests[0].content)

    assert payload["model"] == "deepseek-chat"
    assert payload["messages"] == [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": "USER TEXT"},
    ]
    # JSON mode is the only guarantee that "the model outputs exactly one JSON object"; losing it turns
    # extraction into an intermittent failure.
    assert payload["response_format"] == {"type": "json_object"}


def test_the_api_key_travels_as_a_bearer_token():
    client, requests = recording_client(lambda request: httpx.Response(200, json=chat_response()))
    llm = OpenAICompatibleClient(BASE_URL, API_KEY, "m", timeout_s=5.0, client=client)

    llm.complete_json(system="S", user="U")

    assert requests[0].headers["authorization"] == f"Bearer {API_KEY}"
    assert requests[0].headers["content-type"] == "application/json"


# ---- Successful response --------------------------------------------------------------


def test_a_successful_call_returns_the_content_and_the_usage():
    llm = make_llm(lambda request: httpx.Response(200, json=chat_response('{"ok": 1}', {"total_tokens": 7})))

    result = llm.complete_json(system="S", user="U")

    assert result.text == '{"ok": 1}'
    assert result.usage == {"total_tokens": 7}
    assert result.cached is False


def test_a_missing_usage_block_is_not_an_error():
    llm = make_llm(lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}))

    assert llm.complete_json(system="S", user="U").usage == {}


def test_reasoning_tokens_are_lifted_out_of_the_details_block():
    usage = {"total_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 40, "text_tokens": 8}}
    llm = make_llm(lambda request: httpx.Response(200, json=chat_response(usage=usage)))

    assert llm.complete_json(system="S", user="U").usage == {"total_tokens": 50, "reasoning_tokens": 40}


def test_non_numeric_usage_entries_are_skipped():
    llm = make_llm(lambda request: httpx.Response(200, json=chat_response(usage={"total_tokens": 5, "model": "x"})))

    assert llm.complete_json(system="S", user="U").usage == {"total_tokens": 5}


# ---- Bad response ----------------------------------------------------------------------


@pytest.mark.parametrize("content", ["", "   "])
def test_empty_content_is_an_error_rather_than_an_empty_extraction(content):
    # Empty content is usually max_tokens truncation; treating it as "nothing was found" would silently
    # drop every fact in the paper.
    llm = make_llm(lambda request: httpx.Response(200, json=chat_response(content)))

    with pytest.raises(LlmError, match="empty content"):
        llm.complete_json(system="S", user="U")


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{"message": {}}]}, {"choices": "nope"}])
def test_a_response_without_the_expected_shape_is_an_error(payload):
    llm = make_llm(lambda request: httpx.Response(200, json=payload))

    with pytest.raises(LlmError, match="choices"):
        llm.complete_json(system="S", user="U")


def test_a_body_that_is_not_json_is_an_error():
    llm = make_llm(lambda request: httpx.Response(200, content=b"<html>gateway</html>"))

    with pytest.raises(LlmError, match="not JSON"):
        llm.complete_json(system="S", user="U")


# ---- Retries ----------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 503, 500, 408])
def test_a_retryable_status_is_retried_and_can_then_succeed(status):
    sleep = FakeSleep()
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(status, text="slow down")
        return httpx.Response(200, json=chat_response('{"ok": 1}'))

    llm = make_llm(handler, sleep=sleep)

    assert llm.complete_json(system="S", user="U").text == '{"ok": 1}'
    assert len(attempts) == 2
    assert sleep.delays == [RETRY_BACKOFF_S]


def test_the_backoff_grows_exponentially_between_attempts():
    sleep = FakeSleep()
    llm = make_llm(lambda request: httpx.Response(429, text="rate limited"), sleep=sleep)

    with pytest.raises(LlmError, match="429"):
        llm.complete_json(system="S", user="U")

    # No sleep after the final failure, so there are RETRY_ATTEMPTS - 1 backoffs.
    assert sleep.delays == [RETRY_BACKOFF_S * 2**i for i in range(RETRY_ATTEMPTS - 1)]


def test_a_numeric_retry_after_header_extends_the_backoff():
    # A rate limit that names its own deadline is a promise: waiting less just spends another attempt
    # of the same fixed budget on a request the server already refused.
    sleep = FakeSleep()
    llm = make_llm(lambda request: httpx.Response(429, text="rate limited", headers={"Retry-After": "30"}), sleep=sleep)

    with pytest.raises(LlmError, match="429"):
        llm.complete_json(system="S", user="U")

    assert sleep.delays == [30.0] * (RETRY_ATTEMPTS - 1)


def test_a_retry_after_shorter_than_the_backoff_does_not_shorten_it():
    sleep = FakeSleep()
    llm = make_llm(
        lambda request: httpx.Response(429, text="rate limited", headers={"Retry-After": "0.5"}), sleep=sleep
    )

    with pytest.raises(LlmError, match="429"):
        llm.complete_json(system="S", user="U")

    assert sleep.delays == [RETRY_BACKOFF_S * 2**i for i in range(RETRY_ATTEMPTS - 1)]


def test_a_non_numeric_retry_after_header_is_ignored():
    # The HTTP-date form is rare; interpreting it wrongly would be worse than the exponential backoff
    # that already errs long.
    sleep = FakeSleep()
    llm = make_llm(
        lambda request: httpx.Response(503, text="down", headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
        sleep=sleep,
    )

    with pytest.raises(LlmError, match="503"):
        llm.complete_json(system="S", user="U")

    assert sleep.delays == [RETRY_BACKOFF_S * 2**i for i in range(RETRY_ATTEMPTS - 1)]


def test_every_attempt_is_actually_sent():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, text="down")

    llm = make_llm(handler)

    with pytest.raises(LlmError):
        llm.complete_json(system="S", user="U")

    assert len(requests) == RETRY_ATTEMPTS


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_client_error_is_not_retried(status):
    # A 4xx is a problem with our own request; retrying it four times just sends the same bad request
    # four times.
    sleep = FakeSleep()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, text="bad request")

    llm = make_llm(handler, sleep=sleep)

    with pytest.raises(LlmError, match=str(status)):
        llm.complete_json(system="S", user="U")

    assert len(requests) == 1
    assert sleep.delays == []


def test_a_transport_error_is_retried_like_a_retryable_status():
    sleep = FakeSleep()
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=chat_response())

    llm = make_llm(handler, sleep=sleep)

    llm.complete_json(system="S", user="U")

    assert len(attempts) == 2
    assert sleep.delays == [RETRY_BACKOFF_S]


def test_a_transport_error_on_every_attempt_surfaces_as_an_llm_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    llm = make_llm(handler)

    with pytest.raises(LlmError, match="ConnectError"):
        llm.complete_json(system="S", user="U")


# ---- Cache ------------------------------------------------------------------------------


def test_an_identical_request_is_served_from_disk_without_any_http_call(tmp_path: Path):
    # The same prompt is paid for once and waited for once -- this is design doc §22's idempotency
    # landing at the extraction layer.
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=chat_response('{"cached": 1}'))

    llm = make_llm(handler, cache_dir=tmp_path / "llm_cache")

    first = llm.complete_json(system="S", user="U")
    second = llm.complete_json(system="S", user="U")

    assert len(requests) == 1
    assert first.cached is False and second.cached is True
    assert second.text == first.text == '{"cached": 1}'
    assert second.usage == first.usage


def test_the_cache_file_is_named_after_the_cache_key(tmp_path: Path):
    cache_dir = tmp_path / "llm_cache"
    llm = make_llm(cache_dir=cache_dir)

    llm.complete_json(system="S", user="U")

    assert [p.name for p in cache_dir.iterdir()] == [f"{cache_key_of(llm, 'S', 'U')}.json"]


@pytest.mark.parametrize(
    ("system", "user", "model"),
    [("OTHER", "U", "deepseek-chat"), ("S", "OTHER", "deepseek-chat"), ("S", "U", "other-model")],
)
def test_changing_the_model_or_either_prompt_changes_the_cache_key(system, user, model):
    # A single changed word in the prompt must force re-extraction, or an improved prompt would still
    # show the old result.
    baseline = cache_key_of(make_llm(), "S", "U")

    assert baseline != cache_key_of(make_llm(model=model), system, user)


def test_the_cache_key_covers_the_whole_payload_not_just_the_prompts():
    """cache key = sha256(base_url + the whole payload), so "we forgot to fold some parameter into the
    key" cannot happen structurally."""
    base = make_llm()
    hotter = OpenAICompatibleClient(BASE_URL, API_KEY, base.model, timeout_s=30.0, temperature=0.7)
    longer = OpenAICompatibleClient(BASE_URL, API_KEY, base.model, timeout_s=30.0, max_tokens=99)

    assert cache_key_of(base, "S", "U") != cache_key_of(hotter, "S", "U")
    assert cache_key_of(base, "S", "U") != cache_key_of(longer, "S", "U")


def test_the_payload_omits_the_reasoning_effort_when_it_is_unset():
    # Unset means "send what was sent before this parameter existed", so every cached answer still resolves.
    assert "reasoning_effort" not in make_llm().payload(system="S", user="U")


def test_the_payload_carries_the_reasoning_effort_when_it_is_set():
    llm = OpenAICompatibleClient(BASE_URL, API_KEY, "deepseek-chat", timeout_s=30.0, reasoning_effort="none")

    assert llm.payload(system="S", user="U")["reasoning_effort"] == "none"


def test_a_request_can_override_the_clients_reasoning_effort():
    # One question that reasons far longer than its neighbours can be given its own effort without
    # touching what any other request sends.
    llm = OpenAICompatibleClient(BASE_URL, API_KEY, "deepseek-chat", timeout_s=30.0, reasoning_effort="high")

    assert llm.payload(system="S", user="U", reasoning_effort="none")["reasoning_effort"] == "none"
    assert llm.payload(system="S", user="U")["reasoning_effort"] == "high"


def test_an_override_can_add_the_parameter_to_a_client_that_omits_it():
    llm = make_llm()

    assert "reasoning_effort" not in llm.payload(system="S", user="U")
    assert llm.payload(system="S", user="U", reasoning_effort="none")["reasoning_effort"] == "none"


def test_the_cache_key_covers_the_reasoning_effort():
    base = make_llm()
    quiet = OpenAICompatibleClient(BASE_URL, API_KEY, base.model, timeout_s=30.0, reasoning_effort="none")

    assert cache_key_of(base, "S", "U") != cache_key_of(quiet, "S", "U")


def test_the_cache_key_covers_the_base_url():
    # The same prompt sent to a different service is a different request; sharing a cache would pass off
    # service A's answer as service B's.
    other = OpenAICompatibleClient("https://api.other.com/v1", API_KEY, "deepseek-chat", timeout_s=30.0)

    assert cache_key_of(make_llm(), "S", "U") != cache_key_of(other, "S", "U")


def test_the_api_key_is_not_part_of_the_cache_key():
    # Rotating the key should not invalidate the whole cache -- the key does not affect the answer.
    rotated = OpenAICompatibleClient(BASE_URL, "sk-rotated", "deepseek-chat", timeout_s=30.0)

    assert cache_key_of(make_llm(), "S", "U") == cache_key_of(rotated, "S", "U")


# ---- cache_salt (self-consistency passes) ----------------------------------------------


def test_an_empty_cache_salt_is_indistinguishable_from_no_salt_at_all():
    # Self-consistency passes did not exist when the first cache entries were written, so those keys were
    # computed with no salt argument at all. An explicit empty salt must resolve to the exact same key, or
    # every cache entry written before this feature existed would silently miss.
    llm = make_llm()
    payload = llm.payload(system="S", user="U")

    assert llm.cache_key(payload) == llm.cache_key(payload, cache_salt="")


def test_a_non_empty_cache_salt_changes_the_key():
    llm = make_llm()
    payload = llm.payload(system="S", user="U")

    assert llm.cache_key(payload) != llm.cache_key(payload, cache_salt="pass-1")


def test_different_non_empty_salts_produce_different_keys():
    # Each self-consistency pass needs its own cache slot, or pass 2 would just replay pass 1's answer.
    llm = make_llm()
    payload = llm.payload(system="S", user="U")

    assert llm.cache_key(payload, cache_salt="pass-1") != llm.cache_key(payload, cache_salt="pass-2")


# ---- refresh: --force really re-asks the model ------------------------------------------


def test_refresh_skips_reading_the_cache_but_still_writes_it(tmp_path: Path):
    cache_dir = tmp_path / "llm_cache"
    answers = iter(['{"first": 1}', '{"second": 2}'])
    llm = make_llm(lambda request: httpx.Response(200, json=chat_response(next(answers))), cache_dir=cache_dir)

    first = llm.complete_json(system="S", user="U")
    refreshed = llm.complete_json(system="S", user="U", refresh=True)
    after = llm.complete_json(system="S", user="U")

    assert first.text == '{"first": 1}'
    assert refreshed.text == '{"second": 2}' and refreshed.cached is False
    # The refreshed result overwrote the cache, so the next plain call gets the new answer.
    assert after.text == '{"second": 2}' and after.cached is True


def test_without_refresh_the_cache_wins(tmp_path: Path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=chat_response())

    llm = make_llm(handler, cache_dir=tmp_path / "llm_cache")

    llm.complete_json(system="S", user="U")
    llm.complete_json(system="S", user="U", refresh=False)

    assert len(requests) == 1


# ---- Lifecycle ----------------------------------------------------------------------------


def test_the_client_closes_its_http_client_on_exit():
    # The CLI uses `with build_llm_client(settings) as client`; failing to close the pool leaks sockets
    # over a long-running batch job.
    http_client = make_client(lambda request: httpx.Response(200, json=chat_response()))
    llm = OpenAICompatibleClient(BASE_URL, API_KEY, "m", timeout_s=5.0, client=http_client)

    with llm as entered:
        assert entered is llm

    assert http_client.is_closed


def test_a_corrupt_cache_entry_is_ignored_and_the_request_is_made_again(tmp_path: Path):
    cache_dir = tmp_path / "llm_cache"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=chat_response('{"fresh": 1}'))

    llm = make_llm(handler, cache_dir=cache_dir)
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{cache_key_of(llm, 'S', 'U')}.json").write_text("{ truncated", encoding="utf-8")

    result = llm.complete_json(system="S", user="U")

    assert len(requests) == 1
    assert result.text == '{"fresh": 1}'
    assert result.cached is False


def test_a_cache_entry_without_a_text_key_is_ignored(tmp_path: Path):
    cache_dir = tmp_path / "llm_cache"
    llm = make_llm(cache_dir=cache_dir)
    cache_dir.mkdir(parents=True)
    (cache_dir / f"{cache_key_of(llm, 'S', 'U')}.json").write_text('{"usage": {}}', encoding="utf-8")

    assert llm.complete_json(system="S", user="U").cached is False


def test_the_cache_entry_records_the_model_alongside_the_text(tmp_path: Path):
    cache_dir = tmp_path / "llm_cache"
    llm = make_llm(cache_dir=cache_dir, model="deepseek-chat")

    llm.complete_json(system="S", user="U")
    entry = json.loads(next(cache_dir.iterdir()).read_text(encoding="utf-8"))

    assert entry["model"] == "deepseek-chat"
    assert entry["text"] == '{"samples": []}'


def test_the_cache_directory_holds_no_temp_files_after_a_write(tmp_path: Path):
    # The entry is written atomically (unique temp file + replace), so a crash mid-write can never leave
    # a torn entry behind -- only the finished one is ever visible, and no temp name survives.
    cache_dir = tmp_path / "llm_cache"
    llm = make_llm(cache_dir=cache_dir)

    llm.complete_json(system="S", user="U")

    names = sorted(path.name for path in cache_dir.iterdir())
    assert names == [f"{cache_key_of(llm, 'S', 'U')}.json"]


def test_without_a_cache_directory_nothing_is_written(tmp_path: Path):
    llm = make_llm(cache_dir=None)

    llm.complete_json(system="S", user="U")
    llm.complete_json(system="S", user="U")

    assert list(tmp_path.iterdir()) == []


def test_a_failed_call_leaves_no_cache_entry(tmp_path: Path):
    cache_dir = tmp_path / "llm_cache"
    llm = make_llm(lambda request: httpx.Response(400, text="bad"), cache_dir=cache_dir)

    with pytest.raises(LlmError):
        llm.complete_json(system="S", user="U")

    assert not cache_dir.exists()
