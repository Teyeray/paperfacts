"""MinerU HTTP parser: swap in ``httpx.MockTransport`` for the real service; never sends a real request.

The most important contract at this layer: **the HTTP implementation produces the same directory
layout and meta.json as the runner script**, so adapters can't tell whether native output came
from a subprocess or was pulled from the service. So these cases assert both on request shape
(what the service requires) and on what lands on disk (what the adapter requires).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from paperfacts import parsers
from paperfacts.errors import ParserError
from paperfacts.models import META_FILENAME, DocumentGeometry, DocumentInput, ParserMeta
from paperfacts.parsers import DEFAULT_TIMEOUT_S, MinerUHttpParser
from support.http import make_client, multipart_fields, multipart_files, recording_client

CONTENT_LIST = json.dumps([{"type": "text", "page_idx": 0, "bbox": [0, 0, 500, 100], "text": "hello"}])
MIDDLE_JSON = json.dumps({"pdf_info": []})
MARKDOWN = "# native markdown from the service"

# The form fields mineru-api's /file_parse requires; we pin each one down individually, because
# omitting one (e.g. return_content_list) means the service won't send back the one file the
# adapter actually consumes.
EXPECTED_FORM_FIELDS = {
    "backend": "pipeline",
    "parse_method": "auto",
    "lang_list": "en",
    "formula_enable": "true",
    "table_enable": "true",
    "return_md": "true",
    "return_middle_json": "true",
    "return_content_list": "true",
    "return_images": "false",
    "response_format_zip": "false",
}


def mineru_response(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": "pipeline",
        "version": "3.4.5",
        "results": {"document.pdf": {"md_content": MARKDOWN, "middle_json": MIDDLE_JSON, "content_list": CONTENT_LIST}},
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def mineru() -> tuple[MinerUHttpParser, list[httpx.Request]]:
    client, requests = recording_client(lambda request: httpx.Response(200, json=mineru_response()))
    return MinerUHttpParser("http://mineru.internal:8000/", client=client), requests


@pytest.fixture
def mineru_parser(mineru: tuple[MinerUHttpParser, list[httpx.Request]]) -> MinerUHttpParser:
    return mineru[0]


@pytest.fixture
def mineru_requests(mineru: tuple[MinerUHttpParser, list[httpx.Request]]) -> list[httpx.Request]:
    return mineru[1]


# ---- Basic configuration -----------------------------------------------------------------------


def test_base_url_drops_the_trailing_slash():
    # Otherwise the joined URL becomes //file_parse, which some reverse proxies 404 on.
    parser = MinerUHttpParser("http://svc:8000/", client=make_client(lambda r: httpx.Response(200)))

    assert parser.base_url == "http://svc:8000"


def test_default_timeout_is_generous_enough_for_a_full_paper():
    # A several-dozen-page paper can take minutes on the pipeline backend; the exact number can
    # move, but "generous" is the contract.
    assert DEFAULT_TIMEOUT_S >= 600


def test_timeout_is_forwarded_to_every_request(document: DocumentInput, tmp_path: Path):
    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200, json=mineru_response())

    MinerUHttpParser("http://svc", client=make_client(handler), timeout_s=12.5).parse(document, tmp_path / "raw")

    assert seen and all(t == {"connect": 12.5, "pool": 12.5, "read": 12.5, "write": 12.5} for t in seen)


# ---- Request shape -----------------------------------------------------------------------


def test_request_targets_the_file_parse_endpoint_with_multipart(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser, mineru_requests: list[httpx.Request]
):
    mineru_parser.parse(document, tmp_path / "raw")

    request = mineru_requests[0]

    assert request.method == "POST"
    assert str(request.url) == "http://mineru.internal:8000/file_parse"
    assert request.headers["content-type"].startswith("multipart/form-data")


def test_every_documented_form_field_is_sent_with_the_documented_value(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser, mineru_requests: list[httpx.Request]
):
    # Parse the whole multipart body and compare field-by-field, instead of slicing raw bytes.
    mineru_parser.parse(document, tmp_path / "raw")

    assert multipart_fields(mineru_requests[0]) == EXPECTED_FORM_FIELDS


def test_the_pdf_is_sent_as_a_single_file_part_named_files(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser, mineru_requests: list[httpx.Request]
):
    # The filename is pinned to document.pdf: spaces or non-ASCII characters in the paper's
    # original name would trip MinerU's filename rules.
    mineru_parser.parse(document, tmp_path / "raw")

    files = multipart_files(mineru_requests[0])

    assert set(files) == {"files"}
    assert files["files"].filename == f"{parsers.MINERU_NATIVE_STEM}.pdf"
    assert files["files"].content_type == "application/pdf"
    assert files["files"].content == document.pdf_path.read_bytes()


def test_the_language_hint_is_configurable(tmp_path: Path, document: DocumentInput):
    client, requests = recording_client(lambda request: httpx.Response(200, json=mineru_response()))

    MinerUHttpParser("http://svc", client=client, lang="ch").parse(document, tmp_path / "raw")

    assert multipart_fields(requests[0])["lang_list"] == "ch"


# ---- On-disk layout -----------------------------------------------------------------------


def test_native_files_land_where_the_runner_would_have_put_them(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser
):
    out_dir = tmp_path / "raw"

    raw = mineru_parser.parse(document, out_dir)

    native = parsers.mineru_native_dir(out_dir)
    assert (native / f"{parsers.MINERU_NATIVE_STEM}_content_list.json").read_text(encoding="utf-8") == CONTENT_LIST
    assert (native / f"{parsers.MINERU_NATIVE_STEM}_middle.json").read_text(encoding="utf-8") == MIDDLE_JSON
    assert (native / f"{parsers.MINERU_NATIVE_STEM}.md").read_text(encoding="utf-8") == MARKDOWN
    assert raw.meta.files["content_list"] == "native/document/auto/document_content_list.json"


def test_meta_records_the_version_reported_by_the_service(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser
):
    raw = mineru_parser.parse(document, tmp_path / "raw")

    assert raw.backend_version == "3.4.5"
    assert raw.meta.parser == "mineru"
    assert raw.meta.runner["base_url"] == "http://mineru.internal:8000"


def test_meta_keeps_the_runner_private_backend_field(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser
):
    # ParserMeta is extra="allow": runner-private fields must survive untouched, or the two paths'
    # meta.json would no longer be structurally identical.
    raw = mineru_parser.parse(document, tmp_path / "raw")

    assert raw.meta.model_dump()["backend"] == "pipeline"


def test_meta_page_geometry_comes_from_the_local_pdf(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser, geometry: DocumentGeometry
):
    # The service doesn't report page sizes; the main package must read them itself with
    # pypdfium2 so they agree with the cropping side.
    raw = mineru_parser.parse(document, tmp_path / "raw")

    assert raw.meta.source.page_count == geometry.page_count
    assert raw.meta.source.parsed_page_count == geometry.page_count
    assert raw.meta.source.page_range == (0, None)
    assert raw.meta.source.sha256 == document.sha256
    assert [(p.index, p.width_pt, p.height_pt) for p in raw.meta.pages] == [
        (p.index, p.width_pt, p.height_pt) for p in geometry.pages
    ]


def test_written_meta_json_validates_against_the_contract(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser
):
    out_dir = tmp_path / "raw"

    mineru_parser.parse(document, out_dir)

    ParserMeta.model_validate_json((out_dir / META_FILENAME).read_text(encoding="utf-8"))


def test_the_written_output_is_consumable_by_the_mineru_adapter(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser, geometry: DocumentGeometry
):
    from paperfacts.adapters import convert

    raw = mineru_parser.parse(document, tmp_path / "raw")

    assert [block.content for block in convert(raw, document, geometry).blocks] == ["hello"]


# ---- Cache --------------------------------------------------------------------------


def test_cache_hit_skips_the_request_entirely(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser, mineru_requests: list[httpx.Request]
):
    out_dir = tmp_path / "raw"
    mineru_parser.parse(document, out_dir)

    second = mineru_parser.parse(document, out_dir)

    assert second.cache_hit is True
    assert len(mineru_requests) == 1


def test_force_reruns_even_with_a_cached_meta(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser, mineru_requests: list[httpx.Request]
):
    out_dir = tmp_path / "raw"
    mineru_parser.parse(document, out_dir)

    mineru_parser.parse(document, out_dir, force=True)

    assert len(mineru_requests) == 2


def test_force_wipes_stale_files_from_the_previous_run(
    tmp_path: Path, document: DocumentInput, mineru_parser: MinerUHttpParser
):
    # A leftover file from the previous run mixing into this one would make the adapter read
    # native output that belongs to the old run.
    out_dir = tmp_path / "raw"
    mineru_parser.parse(document, out_dir)
    stale = out_dir / "leftover_from_last_run.json"
    stale.write_text("{}", encoding="utf-8")

    mineru_parser.parse(document, out_dir, force=True)

    assert not stale.exists()


# ---- Failure paths -----------------------------------------------------------------------


def test_non_200_response_raises_a_parser_error_carrying_the_body(tmp_path: Path, document: DocumentInput):
    parser = MinerUHttpParser(
        "http://svc", client=make_client(lambda r: httpx.Response(503, text="model server is warming up"))
    )

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "http"
    assert "HTTP 503" in excinfo.value.detail
    assert "warming up" in excinfo.value.detail


@pytest.mark.parametrize("results", [{}, {"a.pdf": {}, "b.pdf": {}}])
def test_a_result_count_other_than_one_raises_a_parser_error(tmp_path: Path, document: DocumentInput, results):
    # We only ever send one file, so a result count other than 1 means the API's semantics
    # changed and it isn't safe to guess.
    parser = MinerUHttpParser(
        "http://svc", client=make_client(lambda r: httpx.Response(200, json=mineru_response(results=results)))
    )

    with pytest.raises(ParserError, match="expected exactly 1 result"):
        parser.parse(document, tmp_path / "raw")


def test_a_result_missing_content_list_raises_a_parser_error(tmp_path: Path, document: DocumentInput):
    payload = mineru_response(results={"document.pdf": {"middle_json": MIDDLE_JSON}})
    parser = MinerUHttpParser("http://svc", client=make_client(lambda r: httpx.Response(200, json=payload)))

    with pytest.raises(ParserError, match="response missing content_list"):
        parser.parse(document, tmp_path / "raw")


def test_a_result_without_markdown_still_succeeds(tmp_path: Path, document: DocumentInput):
    # md_content is optional: the adapter doesn't consume it, so missing it shouldn't fail the
    # whole parse.
    payload = mineru_response(results={"document.pdf": {"content_list": CONTENT_LIST, "middle_json": MIDDLE_JSON}})
    parser = MinerUHttpParser("http://svc", client=make_client(lambda r: httpx.Response(200, json=payload)))

    raw = parser.parse(document, tmp_path / "raw")

    assert "markdown" not in raw.meta.files


def test_transport_level_failure_is_wrapped_in_a_parser_error(tmp_path: Path, document: DocumentInput):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    parser = MinerUHttpParser("http://svc", client=make_client(handler))

    with pytest.raises(ParserError) as excinfo:
        parser.parse(document, tmp_path / "raw")

    assert excinfo.value.stage == "http"
    assert "ConnectError" in excinfo.value.detail


def test_a_failed_run_leaves_no_meta_json_behind(tmp_path: Path, document: DocumentInput):
    # meta.json's existence marks output as complete; leaving one behind after a failure would
    # make the next run misjudge it as a cache hit.
    out_dir = tmp_path / "raw"
    parser = MinerUHttpParser("http://svc", client=make_client(lambda r: httpx.Response(500, text="boom")))

    with pytest.raises(ParserError):
        parser.parse(document, out_dir)

    assert not (out_dir / META_FILENAME).exists()
