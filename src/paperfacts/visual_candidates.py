"""Deterministic visual-evidence candidates; no model calls or fact adjudication."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from paperfacts.compare import ComparisonReport
from paperfacts.models import NormalizedBBox, ParsedArtifact, SourceBlock
from paperfacts.passages import keyword_hits
from paperfacts.pdf import NativePage
from paperfacts.profile import DomainProfile
from paperfacts.records import FieldValue, LaneExtraction
from paperfacts.text import normalize_text
from paperfacts.validate import region_of_blocks

CandidateSource = Literal["risk", "independent"]
SelectionStrategy = Literal["balanced", "risk_only"]
FULL_PAGE = NormalizedBBox(x1=0, y1=0, x2=1, y2=1)
_CAPTION = re.compile(r"\b(?:fig(?:ure)?s?\.?|tables?)\s*(?:[a-z]?\d+|[ivx]+)\b", re.IGNORECASE)
_COMPLEX_TABLE = re.compile(
    r"(?:[×✕x]|\\times)\s*10|10\s*\^|10[⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹]|<sup\b|[†‡*]|\bfootnotes?\b|\[[a-z]\]",
    re.IGNORECASE,
)
_CONTINUED = re.compile(r"\bcontinued\b|\bcontinuation\b", re.IGNORECASE)


class VisualCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int = Field(ge=0)
    bbox: NormalizedBBox
    reasons: tuple[str, ...]
    sources: tuple[CandidateSource, ...]
    selected_by: CandidateSource | None = None
    zoom_boxes: tuple[NormalizedBBox, ...] = ()
    source_ids: tuple[str, ...] = ()
    context_ids: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()


class CandidateSelection(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidates: tuple[VisualCandidate, ...]
    issues: tuple[str, ...]
    unselected_pages: tuple[int, ...]
    strategy: SelectionStrategy


def select_candidates(
    *,
    pages: Sequence[NativePage],
    artifacts: Mapping[str, ParsedArtifact],
    comparison: ComparisonReport,
    lanes: Sequence[LaneExtraction],
    profile: DomainProfile,
    limit: int = 4,
    strategy: SelectionStrategy = "balanced",
) -> CandidateSelection:
    """Reserve half the budget for discovery independently of A/B's sample inventory or page coverage.

    ``issues`` and ``unselected_pages`` describe selection coverage only; a selected page has not yet
    been read, and neither list establishes scientific completeness.
    """
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if strategy not in ("balanced", "risk_only"):
        raise ValueError("strategy must be balanced or risk_only")
    page_numbers = {page.page for page in pages}
    if len(page_numbers) != len(pages) or any(page < 0 for page in page_numbers):
        raise ValueError("native pages must have unique, non-negative page indices")
    if any(item.document_id != comparison.document_id for item in (*artifacts.values(), *lanes)):
        raise ValueError("artifacts, lanes and comparison must belong to the same document")
    issues: set[str] = set()
    candidates = _risk_candidates(artifacts, comparison, lanes, profile, page_numbers, issues)
    condition_keywords = tuple(
        sorted({keyword for entity in profile.entities for keyword in entity.retrieval.condition_keywords})
    )
    for page in sorted(pages, key=lambda item: item.page):
        text = normalize_text(page.text).lower()
        fields = tuple(sorted(spec.name for spec in profile.fields if keyword_hits(spec.keywords, text)))
        reasons = []
        if not text.strip():
            reasons.append("no_native_text")
            issues.add(f"no_native_text:page={page.page}")
        if fields:
            reasons.append("native_keyword")
        if keyword_hits(condition_keywords, text):
            reasons.append("native_condition")
        if _CAPTION.search(text):
            reasons.append("native_caption")
        if not reasons:
            issues.add(f"no_native_candidate:page={page.page}")
        elif strategy == "balanced":
            candidates.append(
                VisualCandidate(
                    page=page.page, bbox=FULL_PAGE, reasons=tuple(reasons), sources=("independent",), fields=fields
                )
            )
        else:
            issues.add(f"strategy_excluded:page={page.page}")
    merged = _merge_candidates(candidates)
    selected = _allocate(merged, limit, strategy)
    selected_regions = {(candidate.page, candidate.bbox) for candidate in selected}
    for candidate in merged:
        if (candidate.page, candidate.bbox) not in selected_regions:
            issues.add(f"candidate_limit:page={candidate.page}:bbox={_box_key(candidate.bbox)}")
    return CandidateSelection(
        candidates=tuple(selected),
        issues=tuple(sorted(issues)),
        unselected_pages=tuple(sorted(page_numbers - {candidate.page for candidate in selected})),
        strategy=strategy,
    )


def _risk_candidates(
    artifacts: Mapping[str, ParsedArtifact],
    comparison: ComparisonReport,
    lanes: Sequence[LaneExtraction],
    profile: DomainProfile,
    page_numbers: set[int],
    issues: set[str],
) -> list[VisualCandidate]:
    candidates: list[VisualCandidate] = []
    for row in comparison.comparisons:
        for backend, value in ((comparison.backend_a, row.a), (comparison.backend_b, row.b)):
            if value is None:
                continue
            reasons = []
            if row.status in ("conflict", "ambiguous"):
                reasons.append(row.status)
            elif row.status == "missing" and (row.a is None) != (row.b is None):
                reasons.append("missing")
            if not value.grounded:
                reasons.append("ungrounded")
            artifact = artifacts.get(backend)
            blocks = {block.source_id: block for block in artifact.blocks} if artifact else {}
            if row.status == "agree" and any(
                source_id in blocks
                and blocks[source_id].type == "table"
                and _COMPLEX_TABLE.search(blocks[source_id].content)
                for source_id in value.source_ids
            ):
                reasons.append("complex_table")
            if reasons:
                candidates.extend(_locate(value, backend, reasons, artifacts, profile, page_numbers, issues))
    # Unattributed values can be absent from the comparison yet still expose a grounding failure.
    for lane in lanes:
        for value in lane.ungrounded():
            candidates.extend(_locate(value, lane.backend, ["ungrounded"], artifacts, profile, page_numbers, issues))
    return candidates


def _locate(
    value: FieldValue,
    backend: str,
    reasons: list[str],
    artifacts: Mapping[str, ParsedArtifact],
    profile: DomainProfile,
    page_numbers: set[int],
    issues: set[str],
) -> list[VisualCandidate]:
    label = f"backend={backend}:field={value.field}"
    if value.field not in profile.by_name:
        issues.add(f"unknown_field:{label}")
        return []
    artifact = artifacts.get(backend)
    indexed = {block.source_id: block for block in artifact.blocks} if artifact else {}
    if not value.source_ids:
        issues.add(f"missing_source:{label}")
    blocks_by_page: dict[int, list[SourceBlock]] = {}
    incomplete = False
    for source_id in sorted(set(value.source_ids)):
        block = indexed.get(source_id)
        if block is None:
            issues.add(f"missing_source:{label}:source={source_id}")
            incomplete = True
        elif block.page not in page_numbers:
            issues.add(f"invalid_source_page:{label}:source={source_id}:page={block.page}")
            incomplete = True
        else:
            blocks_by_page.setdefault(block.page, []).append(block)
            incomplete = incomplete or bool(_CONTINUED.search(block.content))
    if incomplete:
        reasons = [*reasons, "incomplete_source"]
        issues.add(f"incomplete_source:{label}")
    multipage = len(blocks_by_page) > 1
    if multipage:
        issues.add(f"multipage_source:{label}:pages={tuple(sorted(blocks_by_page))}")
        reasons = [*reasons, "multipage_source"]
    candidates = []
    if artifact is None:
        return candidates
    ordered = artifact.model_copy(
        update={"blocks": tuple(sorted(artifact.blocks, key=lambda block: (block.page, block.order, block.source_id)))}
    )
    for page, blocks in sorted(blocks_by_page.items()):
        regions = [region_of_blocks([block], ordered) for block in blocks if block.type in ("table", "figure")]
        cited_ids = {block.source_id for block in blocks}
        context_ids = {source_id for region in regions for source_id in region.context_ids} - cited_ids
        context_incomplete = any(_CONTINUED.search(indexed[source_id].content) for source_id in context_ids)
        page_reasons = set(reasons)
        if context_incomplete:
            page_reasons.add("incomplete_source")
            issues.add(f"incomplete_source:{label}:page={page}")
        zoom_boxes = (
            ()
            if incomplete or multipage or context_incomplete
            else tuple(sorted({region.bbox for region in regions}, key=_box_key))
        )
        # Captions and prose do not identify a complete visual region. Multiple citations need their context.
        bbox = zoom_boxes[0] if len(blocks) == 1 and len(zoom_boxes) == 1 else FULL_PAGE
        candidates.append(
            VisualCandidate(
                page=page,
                bbox=bbox,
                reasons=tuple(sorted(page_reasons)),
                sources=("risk",),
                zoom_boxes=zoom_boxes,
                source_ids=tuple(sorted(block.source_id for block in blocks)),
                context_ids=tuple(sorted(context_ids, key=lambda source_id: (indexed[source_id].order, source_id))),
                fields=(value.field,),
            )
        )
    return candidates


def _box_key(box: NormalizedBBox) -> tuple[float, float, float, float]:
    return box.x1, box.y1, box.x2, box.y2


def _position(candidate: VisualCandidate) -> tuple:
    return candidate.page, *_box_key(candidate.bbox)


def _overlap(left: VisualCandidate, right: VisualCandidate) -> bool:
    a, b = left.bbox, right.bbox
    return left.page == right.page and max(a.x1, b.x1) < min(a.x2, b.x2) and max(a.y1, b.y1) < min(a.y2, b.y2)


def _merge_candidates(candidates: Sequence[VisualCandidate]) -> list[VisualCandidate]:
    merged: list[VisualCandidate] = []
    for candidate in sorted(candidates, key=_position):
        while (overlap := next((item for item in merged if _overlap(item, candidate)), None)) is not None:
            merged.remove(overlap)
            candidate = candidate.model_copy(
                update={
                    "bbox": candidate.bbox.union(overlap.bbox),
                    "reasons": tuple(sorted(set(candidate.reasons + overlap.reasons))),
                    "sources": tuple(
                        source for source in ("risk", "independent") if source in candidate.sources + overlap.sources
                    ),
                    "zoom_boxes": tuple(sorted(set(candidate.zoom_boxes + overlap.zoom_boxes), key=_box_key)),
                    "source_ids": tuple(sorted(set(candidate.source_ids + overlap.source_ids))),
                    "context_ids": tuple(sorted(set(candidate.context_ids + overlap.context_ids))),
                    "fields": tuple(sorted(set(candidate.fields + overlap.fields))),
                }
            )
        merged.append(candidate)
    return sorted(merged, key=_position)


def _priority(candidate: VisualCandidate) -> int:
    if {"conflict", "ambiguous", "ungrounded"}.intersection(candidate.reasons):
        return 0
    return 1 if "missing" in candidate.reasons else 2


def _allocate(candidates: Sequence[VisualCandidate], limit: int, strategy: SelectionStrategy) -> list[VisualCandidate]:
    pools = {
        "risk": sorted(
            (item for item in candidates if "risk" in item.sources), key=lambda item: (_priority(item), _position(item))
        ),
        "independent": sorted((item for item in candidates if "independent" in item.sources), key=_position),
    }
    quotas = {
        "risk": limit if strategy == "risk_only" else limit // 2,
        "independent": 0 if strategy == "risk_only" else limit - limit // 2,
    }
    selected: list[VisualCandidate] = []
    regions: set[tuple] = set()

    def take(source: CandidateSource) -> bool:
        candidate = next((item for item in pools[source] if _position(item) not in regions), None)
        if candidate is None:
            return False
        regions.add(_position(candidate))
        selected.append(candidate.model_copy(update={"selected_by": source}))
        return True

    for index in range(max(quotas.values())):
        for source in ("risk", "independent"):
            if index < quotas[source]:
                take(source)
    # Borrow only after each pool has had its reserved opportunities; dual-source regions count once.
    while len(selected) < limit:
        added = False
        for source in ("risk", "independent"):
            if len(selected) < limit and (source == "risk" or strategy == "balanced"):
                added = take(source) or added
        if not added:
            break
    return selected
