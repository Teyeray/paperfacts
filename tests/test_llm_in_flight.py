"""The process-wide ceiling on model requests in flight.

Several documents, their two lanes, the figures stage and each lane's field questions all nest pools inside
pools; the endpoint's rate limit sees only the product. These tests pin the three things that make one
shared limit safe: it is never exceeded, a cache hit never waits for it, and a slot is held for the HTTP call
alone -- never across a backoff or a wait on another future, which is what keeps the nesting deadlock-free.

Timing is staged with events and a polled counter, never with sleeps. Every call that could wait on the
limit runs on a daemon thread (support.threads) behind a timed wait, so a leaked slot fails the test instead
of hanging the suite.
"""

from __future__ import annotations

import threading
from pathlib import Path

import httpx
import pytest

from paperfacts.errors import LlmError
from paperfacts.llm import IN_FLIGHT, InFlightLimit, OpenAICompatibleClient, set_max_in_flight
from support.http import make_client
from support.threads import submit_daemon
from support.web import WAIT_TIMEOUT_S, wait_until

BASE_URL = "https://api.example.com/v1"
PNG = b"\x89PNG\r\n\x1a\n-not-really-an-image"


def chat_response(content: str = '{"samples": []}') -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": {"total_tokens": 1}}


class GatedEndpoint:
    """A fake endpoint that counts the requests inside it and holds each one until ``release`` is set."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.active = 0
        self.peak = 0
        self.served = 0
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if not self.release.wait(timeout=WAIT_TIMEOUT_S):
                raise AssertionError("the test never released the endpoint")
            return httpx.Response(200, json=chat_response())
        finally:
            with self._lock:
                self.active -= 1
                self.served += 1


def client_for(
    handler, limit: InFlightLimit, *, cache_dir: Path | None = None, sleep=lambda seconds: None
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        BASE_URL,
        "sk-test",
        "m",
        timeout_s=5.0,
        client=make_client(handler),
        cache_dir=cache_dir,
        in_flight=limit,
        sleep=sleep,
    )


def test_the_number_of_requests_on_the_wire_never_exceeds_the_limit():
    endpoint = GatedEndpoint()
    limit = InFlightLimit(3)
    llm = client_for(endpoint, limit)
    futures = [submit_daemon(llm.complete_json, system="S", user=f"question {i}") for i in range(10)]
    # Three are let in and then everybody else queues on the limit, not on the endpoint.
    wait_until(lambda: endpoint.active == 3, what="three requests to reach the endpoint")
    endpoint.release.set()
    for future in futures:
        future.result(timeout=WAIT_TIMEOUT_S)

    assert endpoint.served == 10
    assert endpoint.peak == 3


def test_text_and_vision_requests_share_one_limit():
    endpoint = GatedEndpoint()
    limit = InFlightLimit(2)
    text, vision = client_for(endpoint, limit), client_for(endpoint, limit)
    futures = [submit_daemon(text.complete_json, system="S", user=f"q{i}") for i in range(3)]
    futures += [submit_daemon(vision.complete_vision, system="S", user=f"chart {i}", image_png=PNG) for i in range(3)]
    wait_until(lambda: endpoint.active == 2, what="two requests to reach the endpoint")
    endpoint.release.set()
    for future in futures:
        future.result(timeout=WAIT_TIMEOUT_S)

    assert endpoint.peak == 2


def test_a_cache_hit_does_not_wait_for_a_slot(tmp_path: Path):
    """A resumed corpus run is mostly cache hits; queueing them behind live requests would make a
    replay as slow as the run it replays."""
    endpoint = GatedEndpoint()
    limit = InFlightLimit(1)
    llm = client_for(endpoint, limit, cache_dir=tmp_path)
    endpoint.release.set()
    llm.complete_json(system="S", user="already answered")  # now on disk
    endpoint.release.clear()

    live = submit_daemon(llm.complete_json, system="S", user="a new question")
    wait_until(lambda: endpoint.active == 1, what="the live request to take the only slot")
    try:
        # Answered while the only slot is still taken: had it queued, this would time out.
        cached = submit_daemon(llm.complete_json, system="S", user="already answered").result(timeout=WAIT_TIMEOUT_S)
    finally:
        endpoint.release.set()
    live.result(timeout=WAIT_TIMEOUT_S)

    assert cached.cached is True
    assert endpoint.served == 2  # the first answer and the live one; the replay never reached the endpoint


def test_a_backoff_sleeps_without_holding_a_slot():
    """A request waiting out a 429 must not keep the only slot, or every other request stalls behind a
    server that asked for patience."""
    limit = InFlightLimit(1)
    attempts: list[int] = []

    def throttled_once(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(429 if len(attempts) == 1 else 200, json=chat_response())

    other = client_for(lambda request: httpx.Response(200, json=chat_response()), limit)
    during_backoff: list[bool] = []

    def sleep(seconds: float) -> None:
        # Another request must get the slot while this one backs off. Run it on its own thread with a
        # timeout, so a regression (the slot still held) fails here instead of deadlocking the test.
        meanwhile = submit_daemon(other.complete_json, system="S", user="meanwhile")
        during_backoff.append(meanwhile.result(timeout=WAIT_TIMEOUT_S).cached is False)

    llm = client_for(throttled_once, limit, sleep=sleep)
    llm.complete_json(system="S", user="throttled")

    assert during_backoff == [True]
    assert len(attempts) == 2


def test_nested_pools_that_wait_on_each_other_finish_under_a_limit_of_one():
    """documents -> lanes -> field questions, each level waiting on the futures of the next: with the slot
    taken only around the request, even a limit of one cannot deadlock."""
    limit = InFlightLimit(1)
    llm = client_for(lambda request: httpx.Response(200, json=chat_response()), limit)

    # Every level is a thread per task that its parent blocks on, as the pools are in production.
    def lane(document: int, lane_id: int) -> int:
        futures = [
            submit_daemon(llm.complete_json, system="S", user=f"d{document} l{lane_id} f{field}") for field in range(4)
        ]
        return sum(1 for future in futures if future.result(timeout=WAIT_TIMEOUT_S))

    def document(index: int) -> int:
        lanes = [submit_daemon(lane, index, lane_id) for lane_id in range(2)]
        return sum(future.result(timeout=WAIT_TIMEOUT_S) for future in lanes)

    documents = [submit_daemon(document, index) for index in range(3)]
    answered = [future.result(timeout=WAIT_TIMEOUT_S) for future in documents]

    assert answered == [8, 8, 8]


def test_a_raised_limit_lets_waiting_requests_through_at_once():
    endpoint = GatedEndpoint()
    limit = InFlightLimit(1)
    llm = client_for(endpoint, limit)
    futures = [submit_daemon(llm.complete_json, system="S", user=f"q{i}") for i in range(3)]
    wait_until(lambda: endpoint.active == 1, what="the first request")
    limit.set_limit(3)
    wait_until(lambda: endpoint.active == 3, what="the queued requests to be let in")
    endpoint.release.set()
    for future in futures:
        future.result(timeout=WAIT_TIMEOUT_S)


def test_a_failed_request_gives_its_slot_back():
    limit = InFlightLimit(1)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LlmError):
        client_for(unreachable, limit).complete_json(system="S", user="lost")
    # With the slot leaked, the next request would wait forever for it.
    answered = submit_daemon(
        client_for(lambda request: httpx.Response(200, json=chat_response()), limit).complete_json,
        system="S",
        user="next",
    )
    assert answered.result(timeout=WAIT_TIMEOUT_S).cached is False


@pytest.mark.parametrize("bad", [0, -1])
def test_a_limit_below_one_is_refused(bad: int):
    with pytest.raises(ValueError, match="at least 1"):
        InFlightLimit(bad)


def test_clients_built_from_the_settings_share_the_process_wide_limit_without_resizing_it():
    from paperfacts.config import Settings
    from paperfacts.workflow import build_llm_client, build_vision_client

    previous = IN_FLIGHT.limit
    settings = Settings(llm_api_key="sk-test", llm_max_in_flight=previous + 5)
    with build_llm_client(settings) as text, build_vision_client(settings) as vision:
        assert text.in_flight is vision.in_flight is IN_FLIGHT
    assert IN_FLIGHT.limit == previous  # only an entry point sizes it


def test_the_entry_points_size_the_limit_from_their_settings(monkeypatch, tmp_path: Path):
    from typer.testing import CliRunner

    from paperfacts.cli import app
    from paperfacts.config import Settings
    from paperfacts.web.app import create_app

    previous = IN_FLIGHT.limit
    try:
        create_app(Settings(data_root=tmp_path / "data", llm_max_in_flight=5))
        assert IN_FLIGHT.limit == 5

        monkeypatch.setenv("PAPERFACTS_LLM_MAX_IN_FLIGHT", "6")
        CliRunner().invoke(app, ["fields"])  # a command that reads no settings leaves it alone
        assert IN_FLIGHT.limit == 5
        (tmp_path / "papers").mkdir()
        CliRunner().invoke(app, ["export", str(tmp_path / "papers"), "--data-root", str(tmp_path / "data")])
        assert IN_FLIGHT.limit == 6
    finally:
        set_max_in_flight(previous)
