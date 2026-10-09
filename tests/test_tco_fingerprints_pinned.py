"""The TCO profile's key material is pinned (round 2 spec §7, gate b).

Round 2 adds field attributes, kinds, entities and prompt slots, each omitted from the key material at its
default, so a code change must not move the profile material by a byte: a moved fingerprint means a new
attribute reached the material at its default, which would re-key production for nothing (and, for the
extraction fingerprint, rename every stored lane). Only the *code* fingerprints may move, so none is pinned, and
``retrieval_fingerprint`` -- which hashes the retrieval modules' source beside the profile -- is pinned with that
code part held constant.

``B1`` is what the profile material was when ``tests/fixtures/b1_formats`` was written, and stays so: those files
record it. ``CURRENT`` is the live profile's, and moves only with a deliberate edit of ``profiles/tco.json``: the
"Ω cm^-2" aliases of Ω/sq and ``substrate_temperature.named_values`` (units reach all three fingerprints and the
retrieval part, named values the extraction and comparison ones), transmittance's spectrum declaration (the
figure fingerprint only), and ``component`` as a strict list with its description reworded (``cardinality`` is PROMPT
and VERDICT, ``strict_list`` VERDICT, the description PROMPT and FIGURE: the extraction and comparison fingerprints
move; the figure part does not, because ``component`` is not ``figure_readable``, and retrieval is untouched).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paperfacts import keys
from paperfacts.keys import figure_profile_fingerprint, profile_comparison_fingerprint, profile_extraction_fingerprint
from paperfacts.profile import DomainProfile
from support.profiles import SHIPPED_PROFILE_PATH

B1 = {
    profile_extraction_fingerprint: "1a1957c0a3b8",
    profile_comparison_fingerprint: "5ec489a6749b",
    figure_profile_fingerprint: "623d87026f22",
}
CURRENT = {
    profile_extraction_fingerprint: "66f4dcb016c5",
    profile_comparison_fingerprint: "a170e426444d",
    figure_profile_fingerprint: "d5a2ab6a6ede",
}
# retrieval_fingerprint(tco) with every source fingerprint replaced by CODE_STANDIN: the profile's part only. B1's
# was "71fd3107c43c"; the Ω/sq aliases widen the unit's retrieval pattern.
CURRENT_RETRIEVAL_PROFILE_PART = "cbd2a932b484"
CODE_STANDIN = "code"
# The file itself: the pins are only meaningful over this exact profile.
TCO_JSON_SHA256 = "3c7797a1fa7b59827c192778cb59d8afea0b161193cc8c6b3bf378211bb702be"
# DomainProfile.content_hash covers every non-display attribute *at default too*, so unlike the fingerprints it
# moves whenever FieldSpec, GroupSpec or PromptSlots gains an attribute (spec §7). It names no stored file (it
# serves __hash__, /api/health and the CLI listing), so a step that adds an attribute updates this pin, and says
# so in its commit; the fingerprints above must not move with it. Moved in S5a by FieldSpec.entity and
# PromptSlots.sample_list_heading, both at their defaults in TCO; in S6 by FieldSpec.references, None in TCO; by
# FieldSpec.named_values, () in TCO; then by TCO's own edit: the Ω/sq aliases, substrate_temperature's named values
# and the article_type_hint slot; then by FieldSpec.figure_spectrum_axis and figure_spectrum_points and TCO's
# transmittance spectrum declaration; then by FieldSpec.strict_list and TCO's component becoming a strict list, with
# its description reworded.
TCO_CONTENT_HASH = "5527e90f7190bdeed85331bf45139f7f28a2b0c7bb0ace034c57c8cd810feecf"


def test_the_pinned_profile_file_is_unchanged():
    assert hashlib.sha256(SHIPPED_PROFILE_PATH.read_bytes()).hexdigest() == TCO_JSON_SHA256


@pytest.mark.parametrize("fingerprint", list(CURRENT), ids=lambda fingerprint: fingerprint.__name__)
def test_a_tco_profile_fingerprint_keeps_its_pinned_value(tco_profile: DomainProfile, fingerprint):
    assert fingerprint(tco_profile) == CURRENT[fingerprint]


def test_the_profile_part_of_the_tco_retrieval_fingerprint_keeps_its_pinned_value(
    tco_profile: DomainProfile, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(keys, "source_fingerprint", lambda *modules: CODE_STANDIN)

    # Uncached: the cached value was computed over the real source.
    assert keys.retrieval_fingerprint.__wrapped__(tco_profile) == CURRENT_RETRIEVAL_PROFILE_PART


def test_the_b1_fixtures_were_written_under_the_pinned_fingerprints():
    fixtures = Path(__file__).parent / "fixtures" / "b1_formats"

    def recorded(name: str) -> str:
        return json.loads((fixtures / name).read_text(encoding="utf-8"))["profile_fingerprint"]

    assert recorded("lane.json") == B1[profile_extraction_fingerprint]
    assert recorded("report.json") == recorded("dataset.json") == B1[profile_comparison_fingerprint]


def test_the_tco_content_hash_keeps_its_pinned_value(tco_profile: DomainProfile):
    assert tco_profile.content_hash == TCO_CONTENT_HASH
