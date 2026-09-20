"""The vision half of the OpenAI-compatible client: one PNG plus a question in, text out.

Two things are pinned here. The request has the shape every OpenAI-style vision endpoint (vLLM, Model
Studio) accepts -- the image as a ``data:`` URL in an ``image_url`` content part, the question after it --
and the cache key stands the image in by its digest: two identical crops share one entry, one changed pixel
gets another, and a megabyte of base64 is neither hashed nor written to disk.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from paperfacts.errors import LlmError
from paperfacts.llm import OpenAICompatibleClient
from support.http import make_client, recording_client

BASE_URL = "https://vision.example.com/v1"
# A real (tiny) PNG, so anything that decodes the image downstream has something to decode.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
OTHER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)


def reply(content: str = '{"transcription": "Rs = 12.5 Ω/sq", "legible": true}') -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 800, "completion_tokens": 20, "total_tokens": 820},
    }


def make_vlm(handler=None, *, cache_dir: Path | None = None, model: str = "qwen3-vl") -> OpenAICompatibleClient:
    client = make_client(handler or (lambda request: httpx.Response(200, json=reply())))
    return OpenAICompatibleClient(BASE_URL, "sk-vision", model, timeout_s=30.0, cache_dir=cache_dir, client=client)


# ---- Request shape ---------------------------------------------------------------------------------------


def test_the_image_travels_as_a_data_url_content_part_before_the_question():
    client, requests = recording_client(lambda request: httpx.Response(200, json=reply()))
    vlm = OpenAICompatibleClient(BASE_URL, "k", "qwen3-vl", timeout_s=5.0, client=client)

    vlm.complete_vision(system="transcribe", user="the region", image_png=PNG)

    body = json.loads(requests[0].content)
    assert body["model"] == "qwen3-vl"
    assert body["messages"][0] == {"role": "system", "content": "transcribe"}
    user = body["messages"][1]
    assert user["role"] == "user"
    assert [part["type"] for part in user["content"]] == ["image_url", "text"]
    assert user["content"][0]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert user["content"][1]["text"] == "the region"


def test_a_vision_request_does_not_ask_for_json_mode():
    # Not every vision endpoint accepts response_format; the reply is parsed leniently instead.
    client, requests = recording_client(lambda request: httpx.Response(200, json=reply()))
    vlm = OpenAICompatibleClient(BASE_URL, "k", "m", timeout_s=5.0, client=client)

    vlm.complete_vision(system="s", user="u", image_png=PNG)

    assert "response_format" not in json.loads(requests[0].content)


def test_the_sampling_settings_reach_the_vision_request():
    client, requests = recording_client(lambda request: httpx.Response(200, json=reply()))
    vlm = OpenAICompatibleClient(BASE_URL, "k", "m", timeout_s=5.0, client=client, temperature=0.0, max_tokens=4096)

    vlm.complete_vision(system="s", user="u", image_png=PNG)

    body = json.loads(requests[0].content)
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 4096


def test_an_empty_image_is_refused_before_any_request_is_made():
    client, requests = recording_client(lambda request: httpx.Response(200, json=reply()))
    vlm = OpenAICompatibleClient(BASE_URL, "k", "m", timeout_s=5.0, client=client)

    with pytest.raises(ValueError):
        vlm.complete_vision(system="s", user="u", image_png=b"")
    assert requests == []


# ---- Result --------------------------------------------------------------------------------------------------


def test_a_successful_call_returns_the_text_and_the_usage():
    vlm = make_vlm()

    result = vlm.complete_vision(system="s", user="u", image_png=PNG)

    assert json.loads(result.text)["transcription"] == "Rs = 12.5 Ω/sq"
    assert result.usage["total_tokens"] == 820
    assert result.cached is False


def test_empty_content_is_an_error():
    vlm = make_vlm(lambda request: httpx.Response(200, json=reply("")))

    with pytest.raises(LlmError):
        vlm.complete_vision(system="s", user="u", image_png=PNG)


def test_a_client_error_surfaces_as_an_llm_error():
    vlm = make_vlm(lambda request: httpx.Response(404, text="model not found"))

    with pytest.raises(LlmError, match="404"):
        vlm.complete_vision(system="s", user="u", image_png=PNG)


# ---- The cache key --------------------------------------------------------------------------------------------


def key_of(vlm: OpenAICompatibleClient, *, system: str = "s", user: str = "u", image: bytes = PNG) -> str:
    return vlm.vision_cache_key(vlm.vision_payload(system=system, user=user, image_png=image), image)


def test_the_cache_key_material_carries_the_image_digest_not_the_base64():
    vlm = make_vlm()
    payload = vlm.vision_payload(system="s", user="u", image_png=PNG)

    key = vlm.vision_cache_key(payload, PNG)

    # The key is the hash of material in which the data URL was replaced by the sha256 of the bytes; the
    # same substitution here must produce the same key, and the base64 must not be in that material.
    stripped = json.loads(json.dumps(payload))
    stripped["messages"][1]["content"][0]["image_url"] = {"sha256": hashlib.sha256(PNG).hexdigest()}
    material = {"base_url": BASE_URL, "payload": stripped, "vision": True}
    assert key == hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert base64.b64encode(PNG).decode() not in json.dumps(material)


def test_the_same_image_and_prompts_give_the_same_key():
    vlm = make_vlm()
    assert key_of(vlm) == key_of(vlm)


@pytest.mark.parametrize(
    ("system", "user", "image", "model"),
    [("S2", "u", PNG, "m"), ("s", "U2", PNG, "m"), ("s", "u", OTHER_PNG, "m"), ("s", "u", PNG, "other-model")],
)
def test_changing_the_image_either_prompt_or_the_model_changes_the_key(system, user, image, model):
    baseline = key_of(make_vlm(model="m"))
    changed = key_of(make_vlm(model=model), system=system, user=user, image=image)
    assert changed != baseline


def test_a_vision_key_never_collides_with_a_text_key_for_the_same_prompts():
    # The text client hashes {"base_url", "payload"}; the vision key adds a "vision" marker, so even a
    # pathological payload cannot make the two request kinds share a cache entry.
    vlm = make_vlm()
    text_key = vlm.cache_key(vlm.payload(system="s", user="u"))
    assert key_of(vlm) != text_key


# ---- The cache on disk ------------------------------------------------------------------------------------------


def test_an_identical_vision_request_is_served_from_disk_without_any_http_call(tmp_path: Path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=reply())

    vlm = make_vlm(handler, cache_dir=tmp_path)
    first = vlm.complete_vision(system="s", user="u", image_png=PNG)
    second = vlm.complete_vision(system="s", user="u", image_png=PNG)

    assert calls == 1
    assert first.text == second.text
    assert (first.cached, second.cached) == (False, True)


def test_the_cache_entry_stores_the_answer_and_never_the_image(tmp_path: Path):
    vlm = make_vlm(cache_dir=tmp_path)
    vlm.complete_vision(system="s", user="u", image_png=PNG)

    (entry,) = tmp_path.glob("*.json")
    stored = entry.read_text(encoding="utf-8")
    assert json.loads(stored)["text"] == reply()["choices"][0]["message"]["content"]
    assert base64.b64encode(PNG).decode() not in stored
    assert entry.stem == key_of(vlm)


def test_a_different_crop_is_a_different_entry(tmp_path: Path):
    vlm = make_vlm(cache_dir=tmp_path)
    vlm.complete_vision(system="s", user="u", image_png=PNG)
    vlm.complete_vision(system="s", user="u", image_png=OTHER_PNG)

    assert len(list(tmp_path.glob("*.json"))) == 2


def test_refresh_re_asks_but_still_writes_the_cache(tmp_path: Path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=reply(f'{{"transcription": "call {calls}", "legible": true}}'))

    vlm = make_vlm(handler, cache_dir=tmp_path)
    vlm.complete_vision(system="s", user="u", image_png=PNG)
    refreshed = vlm.complete_vision(system="s", user="u", image_png=PNG, refresh=True)
    replayed = vlm.complete_vision(system="s", user="u", image_png=PNG)

    assert calls == 2
    assert json.loads(refreshed.text)["transcription"] == "call 2"
    assert replayed.text == refreshed.text and replayed.cached
