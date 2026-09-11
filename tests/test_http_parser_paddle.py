"""PaddleOCR-VL HTTP parser: sends our own rendered PNG per page, via ``httpx.MockTransport`` instead of a real service.

Same key contract as the runner: **we render the page images ourselves**, so we know the exact
pixel size and write it into ``meta.pages[]``; that's what the adapter divides by when
normalizing ``block_bbox``. Handing the whole PDF to the service would leave the size a guess.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from paperfacts.models import META_FILENAME, DocumentInput, ParserMeta
from paperfacts.parsers.base import ParserError
from paperfacts.parsers.http_parser import PaddleHttpParser
from paperfacts.storage import raw_layout
from support.http import make_client, recording_client

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

PRUNED_RESULT: dict[str, Any] = {
    "parsing_res_list": [
        {
            "block_id": 0,
            "block_order": 0,
            "block_label": "text",
            "block_bbox": [10, 20, 300, 200],
            "block_content": "hello",
        }
    ]
}


def paddle_response(**result_overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"prunedResult": PRUNED_RESULT, "markdown": {"text": "page markdown"}}
    result.update(result_overrides)
    return {"result": {"layoutParsingResults": [result]}}


@pytest.fixture
def paddle() -> tuple[PaddleHttpParser, list[httpx.Request]]:
    client, requests = recording_client(lambda request: httpx.Response(200, json=paddle_response()))
    return PaddleHttpParser("http://paddle.internal:8080/", client=client, render_dpi=150), requests


@pytest.fixture
def paddle_parser(paddle: tuple[PaddleHttpParser, list[httpx.Request]]) -> PaddleHttpParser:
    return paddle[0]


@pytest.fixture
def paddle_requests(paddle: tuple[PaddleHttpParser, list[httpx.Request]]) -> list[httpx.Request]:
    return paddle[1]


# ---- Request shape -----------------------------------------------------------------------


def test_base_url_drops_the_trailing_slash(paddle_parser: PaddleHttpParser):
    assert paddle_parser.base_url == "http://paddle.internal:8080"


def test_one_post_per_page_to_the_layout_parsing_endpoint(
    tmp_path: Path, document: DocumentInput, paddle_parser: PaddleHttpParser, paddle_requests: list[httpx.Request]
):
    paddle_parser.parse(document, tmp_path / "raw")

    assert len(paddle_requests) == 2  # a two-page PDF -> two requests
    assert {str(r.url) for r in paddle_requests} == {"http://paddle.internal:8080/layout-parsing"}


def test_request_body_sends_a_base64_png_with_file_type_one(
    tmp_path: Path, document: DocumentInput, paddle_parser: PaddleHttpParser, paddle_requests: list[httpx.Request]
):
    # fileType=1 means "image"; we always send our own rendered single-page PNG, which is how we
    # know the exact pixel size.
    paddle_parser.parse(document, tmp_path / "raw")

    payload = json.loads(paddle_requests[0].content)

    assert payload["fileType"] == 1
    assert payload["visualize"] is False
    assert base64.b64decode(payload["file"])[: len(PNG_MAGIC)] == PNG_MAGIC


def test_each_request_carries_the_image_of_its_own_page(
    tmp_path: Path, document: DocumentInput, paddle_parser: PaddleHttpParser, paddle_requests: list[httpx.Request]
):
    # The two pages differ in size: sending the wrong page would show up as both requests
    # carrying the same image.
    out_dir = tmp_path / "raw"

    paddle_parser.parse(document, out_dir)

    sent = [base64.b64decode(json.loads(r.content)["file"]) for r in paddle_requests]
    pages_dir = raw_layout.paddle_pages_dir(out_dir)
    assert sent[0] == raw_layout.paddle_page_image(pages_dir, 0).read_bytes()
    assert sent[1] == raw_layout.paddle_page_image(pages_dir, 1).read_bytes()
    assert sent[0] != sent[1]


def test_timeout_is_forwarded_to_every_request(tmp_path: Path, document: DocumentInput):
    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200, json=paddle_response())

    PaddleHttpParser("http://svc", client=make_client(handler), timeout_s=7.0).parse(document, tmp_path / "raw")

    assert seen and all(t == {"connect": 7.0, "pool": 7.0, "read": 7.0, "write": 7.0} for t in seen)


# ---- On-disk layout -----------------------------------------------------------------------


def test_native_files_land_where_the_runner_would_have_put_them(
    tmp_path: Path, document: DocumentInput, paddle_parser: PaddleHttpParser
):
    out_dir = tmp_path / "raw"

    paddle_parser.parse(document, out_dir)

    pages_dir = raw_layout.paddle_pages_dir(out_dir)
    assert json.loads(raw_layout.paddle_page_json(pages_dir, 0).read_text(encoding="utf-8")) == PRUNED_RESULT
    markdown_dir = raw_layout.paddle_page_markdown_dir(pages_dir, 0)
    assert raw_layout.paddle_page_markdown(markdown_dir, 0).read_text(encoding="utf-8") == "page markdown"
    assert raw_layout.paddle_page_image(pages_dir, 1).is_file()


def test_meta_pages_record_the_real_rendered_pixel_size(
    tmp_path: Path, document: DocumentInput, paddle_parser: PaddleHttpParser
):
    # The adapter divides by these two numbers when normalizing block_bbox; they must come from
    # the real image, not a DPI formula.
    out_dir = tmp_path / "raw"

    raw = paddle_parser.parse(document, out_dir)

    page0 = raw.meta.pages[0]
    with Image.open(raw_layout.paddle_page_image(raw_layout.paddle_pages_dir(out_dir), 0)) as image:
        assert (page0.width_px, page0.height_px) == image.size
    assert (page0.width_pt, page0.height_pt) == (595.0, 842.0)
    assert page0.json_path == "pages/page_000.json"
    assert page0.image == "pages/page_000.png"
    assert page0.markdown_dir == "pages/page_000_md"


def test_meta_marks_the_version_as_unknown_because_the_service_does_not_report_it(
    tmp_path: Path, document: DocumentInput, paddle_parser: PaddleHttpParser
):
    raw = paddle_parser.parse(document, tmp_path / "raw")

    assert raw.backend_version == "unknown"
    assert raw.meta.model_dump()["vl_backend"] == "http"
    assert raw.meta.model_dump()["render_dpi"] == 150
    assert raw.meta.files == {"pages_dir": raw_layout.PADDLE_PAGES_DIRNAME}


def test_written_meta_json_validates_against_the_contract_with_the_json_alias(
    tmp_path: Path, document: DocumentInput, paddle_parser: PaddleHttpParser
):
    # Must be written to disk with the runner's "json" key, not the Python side's json_path.
    out_dir = tmp_path / "raw"

    paddle_parser.parse(document, out_dir)

    text = (out_dir / META_FILENAME).read_text(encoding="utf-8")
    assert '"json"' in text
    assert "json_path" not in text
    ParserMeta.model_validate_json(text)


def test_the_written_output_is_consumable_by_the_paddle_adapter(
    tmp_path: Path, document: DocumentInput, paddle_parser: PaddleHttpParser, geometry
):
    from paperfacts.adapters import convert

    raw = paddle_parser.parse(document, tmp_path / "raw")

    artifact = convert(raw, document, geometry)

    assert sorted({block.page for block in artifact.blocks}) == [0, 1]
    assert all(0.0 <= b.bbox.x1 < b.bbox.x2 <= 1.0 for b in artifact.blocks)


# ---- Cache --------------------------------------------------------------------------


def test_cache_hit_sends_no_request(
    tmp_path: Path, document: DocumentInput, paddle_parser: PaddleHttpParser, paddle_requests: list[httpx.Request]
):
    out_dir = tmp_path / "raw"
    paddle_parser.parse(document, out_dir)

    second = paddle_parser.parse(document, out_dir)

    assert second.cache_hit is True
    assert len(paddle_requests) == 2  # still just the first run's two pages, nothing new


def test_force_wipes_pages_left_over_from_a_longer_previous_run(
    tmp_path: Path, document: DocumentInput, paddle_parser: PaddleHttpParser
):
    # If a previous run parsed more pages, the leftover page_00X.* files would make the
    # directory disagree with meta.pages.
    out_dir = tmp_path / "raw"
    paddle_parser.parse(document, out_dir)
    stale = raw_layout.paddle_page_json(raw_layout.paddle_pages_dir(out_dir), 9)
    stale.write_text("{}", encoding="utf-8")

    paddle_parser.parse(document, out_dir, force=True)

    assert not stale.exists()


# ---- Failure paths -----------------------------------------------------------------------


def test_non_200_response_names_the_failing_page(tmp_path: Path, document: DocumentInput):
    parser = PaddleHttpParser(
        "http://svc", client=make_client(lambda r: httpx.Response(500, text="cuda oom")), render_dpi=100
    )

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "http"
    assert "page 0 HTTP 500" in excinfo.value.detail


def test_unexpected_response_shape_raises_a_parser_error(tmp_path: Path, document: DocumentInput):
    payload = {"result": {"layoutParsingResults": []}}
    parser = PaddleHttpParser(
        "http://svc", client=make_client(lambda r: httpx.Response(200, json=payload)), render_dpi=100
    )

    with pytest.raises(ParserError, match="unexpected shape"):
        parser.parse(document, tmp_path / "raw")


def test_transport_level_failure_names_the_failing_page(tmp_path: Path, document: DocumentInput):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    parser = PaddleHttpParser("http://svc", client=make_client(handler), render_dpi=100)

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, tmp_path / "raw")

    assert "page 0" in excinfo.value.detail
    assert "ReadTimeout" in excinfo.value.detail


def test_missing_markdown_in_the_response_writes_an_empty_page_markdown(tmp_path: Path, document: DocumentInput):
    payload = {"result": {"layoutParsingResults": [{"prunedResult": PRUNED_RESULT}]}}
    out_dir = tmp_path / "raw"
    parser = PaddleHttpParser(
        "http://svc", client=make_client(lambda r: httpx.Response(200, json=payload)), render_dpi=100
    )

    parser.parse(document, out_dir)

    markdown_dir = raw_layout.paddle_page_markdown_dir(raw_layout.paddle_pages_dir(out_dir), 0)
    assert raw_layout.paddle_page_markdown(markdown_dir, 0).read_text(encoding="utf-8") == ""


def test_a_page_failing_midway_leaves_no_meta_json_behind(tmp_path: Path, document: DocumentInput):
    # Page 0 succeeds, page 1 fails: the directory holds half a set of native output, but there
    # must never be a meta.json.
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=paddle_response())
        return httpx.Response(500, text="cuda oom")

    out_dir = tmp_path / "raw"
    parser = PaddleHttpParser("http://svc", client=make_client(handler), render_dpi=100)

    with pytest.raises(ParserError):
        parser.parse(document, out_dir)

    assert not (out_dir / META_FILENAME).exists()
    assert raw_layout.paddle_page_json(raw_layout.paddle_pages_dir(out_dir), 0).is_file()
