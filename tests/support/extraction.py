"""Object builders for the M2 layers: a ParsedArtifact with provenance markers, and extraction records
(fields / samples / one lane's result).

Kept separate from :mod:`support.factories` (the parser-native-output layer): what is built here is the
shape that comes *after* extraction, so tests of normalisation, sample matching and field comparison can
all draw their raw material from here instead of each hand-assembling a dozen required fields.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from paperfacts.config import Settings
from paperfacts.keys import ComparisonOptions, ExtractionOptions, profile_extraction_fingerprint
from paperfacts.models import Backend, PageGeometry, ParsedArtifact, SourceBlock
from paperfacts.records import FieldValue, LaneExtraction, SampleRecord, TargetRecord
from support.factories import DOC_ID, make_block
from support.profiles import shipped_profile

# Placeholder extractor_key for extraction-layer tests: the real value is computed by extractor_key();
# the comparison layer only requires both lanes to carry the same one.
DEFAULT_EXTRACTOR_KEY = "0123456789ab"
DEFAULT_MODEL = "fake-model"


def make_artifact(
    blocks: Sequence[SourceBlock] | None = None,
    *,
    backend: Backend = "mineru",
    document_id: str = DOC_ID,
    backend_version: str | None = "3.4.5",
) -> ParsedArtifact:
    """A minimal but self-consistent ParsedArtifact."""
    if blocks is None:
        blocks = (
            make_block(page=0, order=0, backend=backend, document_id=document_id, content="Sample A was deposited."),
            make_block(
                page=0,
                order=1,
                type="table",
                backend=backend,
                document_id=document_id,
                content="<table><tr><td>Rs</td><td>12.5 Ω/sq</td></tr></table>",
            ),
        )
    return ParsedArtifact(
        document_id=document_id,
        backend=backend,
        backend_version=backend_version,
        pages=(PageGeometry(index=0, width_pt=595.0, height_pt=842.0),),
        blocks=tuple(blocks),
    )


def make_field(
    field: str,
    value_raw: str,
    *,
    unit_raw: str | None = None,
    condition: str | None = None,
    source_ids: Sequence[str] = (),
    value: float | None = None,
    unit: str | None = None,
    normalization_note: str | None = None,
) -> FieldValue:
    return FieldValue(
        field=field,
        value_raw=value_raw,
        unit_raw=unit_raw,
        condition=condition,
        source_ids=tuple(source_ids),
        value=value,
        unit=unit,
        normalization_note=normalization_note,
    )


def make_sample(
    sample_id: str,
    fields: Iterable[FieldValue] = (),
    *,
    label: str = "",
    conditions: Mapping[str, str] | None = None,
    source_ids: Sequence[str] = (),
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        label=label,
        conditions=dict(conditions or {}),
        source_ids=tuple(source_ids),
        fields=tuple(fields),
    )


def make_lane(
    *,
    backend: Backend = "mineru",
    samples: Iterable[SampleRecord] = (),
    target: TargetRecord | None = None,
    document_id: str = DOC_ID,
    extractor_key: str = DEFAULT_EXTRACTOR_KEY,
    model: str = DEFAULT_MODEL,
    invalid_source_ids: Sequence[str] = (),
    dropped: Sequence[str] = (),
    unattributed: Iterable[FieldValue] = (),
    usage: Mapping[str, int] | None = None,
    raw_response: str = "",
) -> LaneExtraction:
    return LaneExtraction(
        document_id=document_id,
        backend=backend,
        extractor_key=extractor_key,
        model=model,
        profile_fingerprint=profile_extraction_fingerprint(shipped_profile()),
        target=target,
        samples=tuple(samples),
        invalid_source_ids=tuple(invalid_source_ids),
        dropped=tuple(dropped),
        unattributed=tuple(unattributed),
        usage=dict(usage or {}),
        raw_response=raw_response,
    )


def lane_options(client=None, *, mode, **overrides) -> ExtractionOptions:
    """The options a caller builds for ``client``: its sampling settings, plus the extraction mode and any
    other option a test moves off its default. Without a client, those of a default ``FakeLlmClient``. The
    profile is the shipped one unless the test passes its own."""
    overrides.setdefault("profile", shipped_profile())
    if client is None:
        return ExtractionOptions(model=DEFAULT_MODEL, mode=mode, **overrides)
    return ExtractionOptions(
        model=client.model,
        mode=mode,
        temperature=client.temperature,
        max_tokens=client.max_tokens,
        reasoning_effort=client.reasoning_effort,
        **overrides,
    )


def comparison_options() -> ComparisonOptions:
    """The options a comparison of :func:`make_lane` lanes runs under: the shipped profile's."""
    return ComparisonOptions.from_settings(Settings(), shipped_profile())
