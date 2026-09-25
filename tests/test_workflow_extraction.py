"""The M2 workflow: orchestration of extraction and comparison, where things land on disk, caching and
``--force``.

The value of this layer is "never pay twice": extraction and comparison each have their own on-disk cache,
and each key carries everything that should invalidate it. So every case here counts LLM calls -- a cache
that leaks through is invisible in behaviour and only shows up on the bill. The LLM is always
:class:`support.llm.FakeLlmClient`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from paperfacts.compare import ComparisonReport
from paperfacts.config import Settings
from paperfacts.errors import ConfigError
from paperfacts.extract import extract_lane
from paperfacts.keys import ExtractionOptions, comparison_key, extractor_key, extractor_key_for
from paperfacts.llm import OpenAICompatibleClient
from paperfacts.models import BACKENDS, Backend, DocumentInput
from paperfacts.parsers import SubprocessParser
from paperfacts.prompts import inventory_system_prompt
from paperfacts.records import FailedQuestion, LaneExtraction
from paperfacts.storage import DataLayout
from paperfacts.workflow import (
    BACKEND_A,
    BACKEND_B,
    build_llm_client,
    build_parser,
    compare_document,
    export_document,
    extract_document,
)
from support.extraction import make_artifact
from support.factories import make_block
from support.llm import FakeLlmClient

# ---- build_parser: PaddleOCR-VL's external VLM service parameters ---------------------


def test_the_vl_options_are_passed_to_the_paddle_runner(tmp_path: Path):
    # mlx-vlm-server on Mac and vllm-server on Linux accelerate the VLM stage; if this wiring breaks it
    # falls back silently to in-process inference, which is ten times slower.
    settings = Settings(
        repo_root=tmp_path,
        paddle_vl_backend="mlx",
        paddle_vl_server_url="http://127.0.0.1:8000/v1",
        paddle_vl_model_name="PaddleOCR-VL",
    )

    parser = build_parser("paddleocr_vl", settings)

    assert isinstance(parser, SubprocessParser)
    assert parser.extra_args == (
        "--dpi",
        "200",
        "--vl-backend",
        "mlx",
        "--vl-server-url",
        "http://127.0.0.1:8000/v1",
        "--vl-model-name",
        "PaddleOCR-VL",
    )


def test_no_vl_options_are_passed_when_none_are_configured(tmp_path: Path):
    parser = build_parser("paddleocr_vl", Settings(repo_root=tmp_path))

    assert isinstance(parser, SubprocessParser)
    assert parser.extra_args == ("--dpi", "200")


def test_the_vl_options_are_independent_of_each_other(tmp_path: Path):
    settings = Settings(repo_root=tmp_path, paddle_vl_server_url="http://127.0.0.1:8000/v1")

    parser = build_parser("paddleocr_vl", settings)

    assert isinstance(parser, SubprocessParser)
    assert parser.extra_args == ("--dpi", "200", "--vl-server-url", "http://127.0.0.1:8000/v1")


def test_the_vl_options_do_not_reach_the_mineru_runner(tmp_path: Path):
    settings = Settings(repo_root=tmp_path, paddle_vl_backend="mlx")

    parser = build_parser("mineru", settings)

    assert isinstance(parser, SubprocessParser)
    assert parser.extra_args == ()


# ---- build_llm_client -------------------------------------------------------------------


def test_building_a_client_without_a_key_fails_with_a_pointer_to_the_key_file(tmp_path: Path):
    # Letting a request go out with an empty key and hit a 401 is one of the hardest failures to trace;
    # stopping here and naming the key file avoids that entirely.
    settings = Settings(repo_root=tmp_path, llm_api_key=None, llm_api_key_file=tmp_path / "deepseek_api_key")

    with pytest.raises(ConfigError, match="deepseek_api_key"):
        build_llm_client(settings)


def test_the_client_is_wired_from_the_settings(tmp_path: Path):
    settings = Settings(
        data_root=tmp_path / "data",
        repo_root=tmp_path,
        llm_base_url="https://api.example.com/v1",
        llm_model="some-model",
        llm_api_key="sk-test",
        llm_timeout_s=42.0,
    )

    with build_llm_client(settings) as client:
        assert isinstance(client, OpenAICompatibleClient)
        assert client.base_url == "https://api.example.com/v1"
        assert client.model == "some-model"
        assert client.timeout_s == 42.0
        # The cache is shared across documents: the same prompt seen on another paper is not paid for again.
        assert client.cache_dir == DataLayout(settings.data_root).llm_cache_dir()


# ---- Fixtures -----------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Document mode explicitly: these cases queue one canned answer per lane and count the calls, which is
    # the shape of a whole-document extraction. Passage mode has its own file.
    return Settings(data_root=tmp_path / "data", repo_root=tmp_path, llm_api_key="sk-test", extraction_mode="document")


@pytest.fixture
def parsed(settings: Settings, document: DocumentInput) -> dict[Backend, str]:
    """Write both lane artifacts to disk directly (without running a parser), returning each lane's first
    block's source_id."""
    layout = DataLayout(settings.data_root)
    first_ids: dict[Backend, str] = {}
    for backend in BACKENDS:
        blocks = (
            make_block(page=0, order=0, backend=backend, document_id=document.document_id, content="Sample A"),
            make_block(page=0, order=1, backend=backend, document_id=document.document_id, content="Rs = 12.5"),
        )
        artifact = make_artifact(blocks, backend=backend, document_id=document.document_id)
        artifact.write(layout.artifact_path(document.document_id, backend))
        first_ids[backend] = blocks[0].source_id
    return first_ids


def extraction_json(sample_id: str = "A", value: str = "12.5", source_id: str | None = None) -> str:
    field = {"field": "sheet_resistance", "value_raw": value, "unit_raw": "Ω/sq"}
    if source_id:
        field["source_ids"] = [source_id]
    return json.dumps({"target": None, "samples": [{"sample_id": sample_id, "fields": [field]}]})


# ---- extract_document -----------------------------------------------------------------


def test_extract_document_calls_the_model_once_and_writes_the_result(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    client = FakeLlmClient([extraction_json()])

    lane = extract_document(document, "mineru", settings, client)

    assert client.call_count == 1
    path = DataLayout(settings.data_root).extraction_path(
        document.document_id, "mineru", extractor_key(ExtractionOptions(client.model, mode="document"))
    )
    assert path.is_file()
    assert lane.sample("A") is not None


def test_the_returned_lane_is_normalized_but_the_file_on_disk_is_not(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    """What is written to disk is the verbatim-level result; what is returned is normalized.

    That way changing a normalization rule never needs a fresh LLM call -- only changing the prompt,
    model or field schema changes extractor_key.
    """
    from paperfacts.records import LaneExtraction

    client = FakeLlmClient([extraction_json(value="1.25", source_id=None)])

    lane = extract_document(document, "mineru", settings, client)
    path = DataLayout(settings.data_root).extraction_path(
        document.document_id, "mineru", extractor_key(ExtractionOptions(client.model, mode="document"))
    )
    on_disk = LaneExtraction.read(path)

    assert lane.sample("A").get("sheet_resistance").value == 1.25
    assert lane.sample("A").get("sheet_resistance").unit == "Ω/sq"
    assert on_disk.sample("A").get("sheet_resistance").value is None


def test_a_second_call_hits_the_cache_and_does_not_call_the_model(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    client = FakeLlmClient([extraction_json()])

    first = extract_document(document, "mineru", settings, client)
    second = extract_document(document, "mineru", settings, client)

    assert client.call_count == 1
    assert second == first


def test_force_re_asks_the_model_and_bypasses_the_llm_cache_too(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    client = FakeLlmClient([extraction_json(value="12.5"), extraction_json(value="99")])

    extract_document(document, "mineru", settings, client)
    refreshed = extract_document(document, "mineru", settings, client, force=True)

    assert client.call_count == 2
    # Both the on-disk cache and the LLM cache must be skipped, or --force would just replay the old answer.
    assert client.refreshes == [False, True]
    assert refreshed.sample("A").get("sheet_resistance").value_raw == "99"


def test_each_backend_has_its_own_cache_entry(settings: Settings, document: DocumentInput, parsed: dict[Backend, str]):
    client = FakeLlmClient([extraction_json(), extraction_json()])

    extract_document(document, "mineru", settings, client)
    extract_document(document, "paddleocr_vl", settings, client)

    layout = DataLayout(settings.data_root)
    key = extractor_key(ExtractionOptions(client.model, mode="document"))
    assert layout.extraction_path(document.document_id, "mineru", key).is_file()
    assert layout.extraction_path(document.document_id, "paddleocr_vl", key).is_file()
    assert client.call_count == 2


def test_changing_the_model_invalidates_the_extraction_cache(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    extract_document(document, "mineru", settings, FakeLlmClient([extraction_json()], model="model-a"))
    other = FakeLlmClient([extraction_json()], model="model-b")

    extract_document(document, "mineru", settings, other)

    assert other.call_count == 1


def _reparse(settings: Settings, document: DocumentInput, backend: Backend, content: str = "Rs = 99") -> None:
    """Replace the stored artifact with a different parse of the same paper: same ids, other blocks."""
    blocks = (
        make_block(page=0, order=0, backend=backend, document_id=document.document_id, content="Sample A"),
        make_block(page=0, order=1, backend=backend, document_id=document.document_id, content=content),
    )
    make_artifact(blocks, backend=backend, document_id=document.document_id).write(
        DataLayout(settings.data_root).artifact_path(document.document_id, backend)
    )


def test_the_lane_records_the_parse_it_came_from(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    from paperfacts.records import LaneExtraction
    from paperfacts.workflow import load_artifact

    client = FakeLlmClient([extraction_json()])
    extract_document(document, "mineru", settings, client)
    path = DataLayout(settings.data_root).extraction_path(
        document.document_id, "mineru", extractor_key(ExtractionOptions(client.model, mode="document"))
    )

    assert LaneExtraction.read(path).artifact_sha256 == load_artifact(document, "mineru", settings).content_hash()


def test_a_lane_from_another_parse_is_re_derived_not_served(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    # The review's scenario: `parse --force` gave new blocks under the same positional ids; the old lane's
    # citations would now point at whatever block has that ordinal.
    client = FakeLlmClient([extraction_json(value="12.5"), extraction_json(value="99")])
    extract_document(document, "mineru", settings, client)
    _reparse(settings, document, "mineru")

    lane = extract_document(document, "mineru", settings, client)

    assert client.call_count == 2
    assert lane.sample("A").get("sheet_resistance").value_raw == "99"


def test_a_lane_file_without_a_recorded_parse_still_reads(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    from paperfacts.records import LaneExtraction

    client = FakeLlmClient([extraction_json()])
    extract_document(document, "mineru", settings, client)
    path = DataLayout(settings.data_root).extraction_path(
        document.document_id, "mineru", extractor_key(ExtractionOptions(client.model, mode="document"))
    )
    legacy = LaneExtraction.read(path).model_copy(update={"artifact_sha256": None})
    legacy.write(path)

    extract_document(document, "mineru", settings, client)

    assert client.call_count == 1  # unknown is not a mismatch


def test_a_comparison_of_other_parses_is_compared_again(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    client = FakeLlmClient([extraction_json(), extraction_json(), extraction_json(value="99")])
    first = compare_document(document, settings, client)
    _reparse(settings, document, BACKEND_B)

    second = compare_document(document, settings, client)

    assert client.call_count == 3  # lane B re-derived; lane A and the matching (exact ids) cost nothing
    assert second.artifact_sha256_a == first.artifact_sha256_a
    assert second.artifact_sha256_b != first.artifact_sha256_b
    assert second.counts.conflict == 1


def test_extracting_before_parsing_says_to_run_parse_first(settings: Settings, document: DocumentInput):
    client = FakeLlmClient([])

    with pytest.raises(FileNotFoundError, match="paperfacts parse"):
        extract_document(document, "mineru", settings, client)

    assert client.call_count == 0


def test_the_source_ids_are_validated_against_the_artifact_on_disk(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    client = FakeLlmClient([extraction_json(source_id="mineru_p9_b9")])

    lane = extract_document(document, "mineru", settings, client)

    assert lane.invalid_source_ids == ("mineru_p9_b9",)


# ---- compare_document -----------------------------------------------------------------


def test_compare_document_extracts_both_lanes_then_matches_and_writes_the_report(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    # Both lanes use the same sample_id -> exact match, no need for a third model call.
    client = FakeLlmClient([extraction_json(), extraction_json()])

    report = compare_document(document, settings, client)

    assert client.call_count == 2
    path = DataLayout(settings.data_root).comparison_path(
        document.document_id, extractor_key(ExtractionOptions(client.model, mode="document")), comparison_key()
    )
    assert path.is_file()
    assert report.backend_a == BACKEND_A and report.backend_b == BACKEND_B
    assert report.counts.samples_matched == 1


def test_the_matching_model_is_called_when_the_sample_ids_differ(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    client = FakeLlmClient(
        [
            extraction_json(sample_id="A1"),
            extraction_json(sample_id="B1"),
            json.dumps({"pairs": [{"a": "A1", "b": "B1", "confidence": 0.9, "justification": "same"}]}),
        ]
    )

    report = compare_document(document, settings, client)

    assert client.call_count == 3
    assert report.counts.samples_matched == 1


def test_a_second_comparison_hits_the_cache_and_calls_nothing(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    client = FakeLlmClient([extraction_json(), extraction_json()])

    first = compare_document(document, settings, client)
    second = compare_document(document, settings, client)

    assert client.call_count == 2
    assert second == first


def test_force_redoes_the_comparison_without_re_extracting(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    """``--force`` on compare only redoes matching and comparison: extraction has its own cache and its
    own ``--force``.

    Otherwise every time someone wants to take another look at the comparison result, both lanes would
    have to be re-extracted, which is the most expensive step.
    """
    client = FakeLlmClient([extraction_json(sample_id="A1"), extraction_json(sample_id="B1"), *[json.dumps({})] * 2])

    compare_document(document, settings, client)
    calls_after_first = client.call_count

    compare_document(document, settings, client, force=True)

    # Only one extra matching call; both extractions went through their on-disk cache.
    assert client.call_count == calls_after_first + 1
    assert client.refreshes[-1] is True


def test_a_stored_lane_with_an_unanswered_question_is_extracted_again(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    # Its invalid answers were never cached, so extracting again re-asks only that question; every other
    # request replays from the LLM cache (this fake has none, so it simply answers twice).
    client = FakeLlmClient([extraction_json(), extraction_json()])
    extract_document(document, "mineru", settings, client)
    path = DataLayout(settings.data_root).extraction_path(
        document.document_id, "mineru", extractor_key(ExtractionOptions(client.model, mode="document"))
    )
    incomplete = LaneExtraction.read(path).model_copy(
        update={"failed_questions": (FailedQuestion(field="thickness", detail="cut off at max_tokens"),)}
    )
    incomplete.write(path)

    lane = extract_document(document, "mineru", settings, client)

    assert client.call_count == 2
    assert lane.failed_questions == ()
    assert LaneExtraction.read(path).failed_questions == ()


def test_a_comparison_of_a_lane_with_an_unanswered_question_is_not_stored(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    good = json.dumps({"pairs": [{"a": "A1", "b": "B1", "confidence": 0.9, "justification": "same"}]})
    client = FakeLlmClient([extraction_json(sample_id="A1"), extraction_json(sample_id="B1"), good])
    lanes = {backend: extract_document(document, backend, settings, client) for backend in BACKENDS}
    lanes[BACKEND_A] = lanes[BACKEND_A].model_copy(
        update={"failed_questions": (FailedQuestion(field="thickness", detail="cut off at max_tokens"),)}
    )

    report = compare_document(document, settings, client, lanes=lanes)

    path = DataLayout(settings.data_root).comparison_path(
        document.document_id, report.extractor_key, report.comparison_key
    )
    assert not path.is_file()


def test_an_offline_export_stores_the_comparison_it_rebuilt(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    # export recompares with regrounded values; left unstored, /report kept showing the old verdicts beside
    # a table built from the new ones.
    good = json.dumps({"pairs": [{"a": "A1", "b": "B1", "confidence": 0.9, "justification": "same"}]})
    client = FakeLlmClient([extraction_json(sample_id="A1"), extraction_json(sample_id="B1"), good])
    report = compare_document(document, settings, client)
    path = DataLayout(settings.data_root).comparison_path(
        document.document_id, report.extractor_key, report.comparison_key
    )
    report.model_copy(update={"comparisons": ()}).write(path)

    export_document(document, dataclasses.replace(settings, llm_model=client.model))

    assert ComparisonReport.read(path).comparisons == report.comparisons


def test_an_offline_export_of_a_lane_with_an_unanswered_question_is_refused(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    # The comparison on disk came from a complete run; the lane stored since is not complete. A table built
    # from it would mark the paper finished with that question never asked again.
    good = json.dumps({"pairs": [{"a": "A1", "b": "B1", "confidence": 0.9, "justification": "same"}]})
    client = FakeLlmClient([extraction_json(sample_id="A1"), extraction_json(sample_id="B1"), good])
    compare_document(document, settings, client)
    path = DataLayout(settings.data_root).extraction_path(
        document.document_id, BACKEND_A, extractor_key(ExtractionOptions(client.model, mode="document"))
    )
    LaneExtraction.read(path).model_copy(
        update={"failed_questions": (FailedQuestion(field="thickness", detail="cut off at max_tokens"),)}
    ).write(path)

    with pytest.raises(FileNotFoundError, match="no valid answer to mineru:thickness"):
        export_document(document, dataclasses.replace(settings, llm_model=client.model))


def test_a_failed_matching_is_reported_but_not_stored_so_the_next_run_asks_again(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    # Stored, a matching the model botched twice would blank the paper's sample cells on every later run.
    good = json.dumps({"pairs": [{"a": "A1", "b": "B1", "confidence": 0.9, "justification": "same"}]})
    client = FakeLlmClient(
        [extraction_json(sample_id="A1"), extraction_json(sample_id="B1"), '{"pairs": 1}', '{"pairs": 2}', good]
    )

    failed = compare_document(document, settings, client)
    path = DataLayout(settings.data_root).comparison_path(
        document.document_id, failed.extractor_key, failed.comparison_key
    )

    assert failed.matching.failed
    assert not path.is_file()

    retried = compare_document(document, settings, client)

    assert client.call_count == 5  # the lanes came from their cache; only matching was asked again
    assert not retried.matching.failed
    assert path.is_file()


def test_the_report_path_carries_both_the_extractor_and_the_comparison_key(
    settings: Settings, document: DocumentInput, parsed: dict[Backend, str]
):
    # Changing a tolerance only changes comparison_key (no LLM re-run); changing the prompt is what
    # changes extractor_key.
    client = FakeLlmClient([extraction_json(), extraction_json()])

    report = compare_document(document, settings, client)
    path = DataLayout(settings.data_root).comparison_path(
        document.document_id, report.extractor_key, report.comparison_key
    )

    assert path.is_file()
    assert ComparisonReport.read(path) == report


def test_comparing_before_parsing_says_to_run_parse_first(settings: Settings, document: DocumentInput):
    with pytest.raises(FileNotFoundError, match="paperfacts parse"):
        compare_document(document, settings, FakeLlmClient([]))


def test_the_two_lane_assumption_is_stated_explicitly():
    # M2's comparison is strictly two lanes head to head; adding a third parser needs compare_lanes
    # redesigned.
    assert (BACKEND_A, BACKEND_B) == BACKENDS


# ---- The key a lane is stored under is the key it is looked up by ------------------------------------

# Every option that reaches ExtractionOptions moved off its baseline, so an option one path forgot shows up
# as two different keys. context_tokens is the one that once went missing from the writer's key.
EDITED_OPTIONS = {
    "extraction_mode": "passage",
    "extraction_passes": 2,
    "llm_temperature": 0.3,
    "llm_max_tokens": 30_000,
    "llm_reasoning_effort": "low",
    "llm_inventory_reasoning_effort": "none",
    "candidate_limit": 5,
    "llm_context_tokens": 150_000,
}


def test_from_settings_reads_every_extraction_option(tmp_path: Path):
    options = ExtractionOptions.from_settings(Settings(repo_root=tmp_path, llm_model="edited", **EDITED_OPTIONS))

    for option in dataclasses.fields(ExtractionOptions):
        if option.name != "mode":
            assert getattr(options, option.name) != option.default, option.name


def test_a_lane_extracted_with_edited_settings_carries_the_key_the_reader_looks_up(
    tmp_path: Path, document: DocumentInput
):
    settings = Settings(data_root=tmp_path / "data", repo_root=tmp_path, llm_api_key="sk-test", **EDITED_OPTIONS)
    layout = DataLayout(settings.data_root)
    blocks = (make_block(page=0, order=0, document_id=document.document_id, content="Sample A at 100 sccm"),)
    make_artifact(blocks, document_id=document.document_id).write(layout.artifact_path(document.document_id, "mineru"))

    def respond(system: str, user: str) -> str:
        if system == inventory_system_prompt():
            return json.dumps({"samples": [{"sample_id": "A"}]})
        return json.dumps({"values": []})

    # A stand-in for the client build_llm_client makes from these settings; the next test pins that the real
    # one sends exactly these values, and extract_lane refuses a client that does not.
    client = FakeLlmClient(
        respond,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_effort=settings.llm_reasoning_effort,
    )

    extract_document(document, "mineru", settings, client)

    key = extractor_key_for(settings)
    stored = LaneExtraction.read(layout.extraction_path(document.document_id, "mineru", key))
    assert stored.extractor_key == key
    assert key != extractor_key(ExtractionOptions(settings.llm_model, mode="passage"))


def test_the_client_built_from_settings_asks_with_the_options_built_from_them(tmp_path: Path):
    # The writer's key comes from ExtractionOptions.from_settings, the requests from build_llm_client; if the
    # two ever read a setting differently, the key would describe requests that were never sent.
    settings = Settings(data_root=tmp_path / "data", repo_root=tmp_path, llm_api_key="sk-test", **EDITED_OPTIONS)
    options = ExtractionOptions.from_settings(settings)

    with build_llm_client(settings) as client:
        sent = (client.model, client.temperature, client.max_tokens, client.reasoning_effort)

    assert sent == (options.model, options.temperature, options.max_tokens, options.reasoning_effort)


def test_extract_lane_refuses_options_that_do_not_describe_the_client():
    client = FakeLlmClient([], temperature=0.7)

    with pytest.raises(ValueError, match="extraction options describe"):
        extract_lane(make_artifact(), client, ExtractionOptions(client.model, mode="document"))
