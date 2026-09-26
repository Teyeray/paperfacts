"""The TCO profile's key material is pinned to its B1 values (round 2 spec §7, gate b).

Round 2 adds field attributes, kinds, entities and prompt slots, each omitted from the key material at its
default. TCO uses none of them, so its profile material must not move by a byte: a moved fingerprint means a new
attribute reached the material at its default, which would re-key production for nothing (and, for the
extraction fingerprint, rename every stored lane). Only the *code* fingerprints may move, so none is pinned, and
``retrieval_fingerprint`` -- which hashes the retrieval modules' source beside the profile -- is pinned with that
code part held constant.
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
# retrieval_fingerprint(tco) with every source fingerprint replaced by CODE_STANDIN: the profile's part only.
B1_RETRIEVAL_PROFILE_PART = "71fd3107c43c"
CODE_STANDIN = "code"
# The file itself: the pins are only meaningful over this exact profile.
TCO_JSON_SHA256 = "2ef95bccbea7100ecc3d8bcdd9f9a7801852974682a8cbaf3571744a241d49c4"
# DomainProfile.content_hash covers every non-display attribute *at default too*, so unlike the fingerprints it
# moves whenever FieldSpec, GroupSpec or PromptSlots gains an attribute (spec §7). It names no stored file (it
# serves __hash__, /api/health and the CLI listing), so a step that adds an attribute updates this pin, and says
# so in its commit; the fingerprints above must not move with it.
TCO_CONTENT_HASH = "923c2519b56f724c48b12e37d34e3d4f91f06227477a438d5ac068e2314f5555"


def test_the_pinned_profile_file_is_unchanged():
    assert hashlib.sha256(SHIPPED_PROFILE_PATH.read_bytes()).hexdigest() == TCO_JSON_SHA256


@pytest.mark.parametrize("fingerprint", list(B1), ids=lambda fingerprint: fingerprint.__name__)
def test_a_tco_profile_fingerprint_keeps_its_b1_value(tco_profile: DomainProfile, fingerprint):
    assert fingerprint(tco_profile) == B1[fingerprint]


def test_the_profile_part_of_the_tco_retrieval_fingerprint_keeps_its_b1_value(
    tco_profile: DomainProfile, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(keys, "source_fingerprint", lambda *modules: CODE_STANDIN)

    # Uncached: the cached value was computed over the real source.
    assert keys.retrieval_fingerprint.__wrapped__(tco_profile) == B1_RETRIEVAL_PROFILE_PART


def test_the_b1_fixtures_were_written_under_the_pinned_fingerprints():
    fixtures = Path(__file__).parent / "fixtures" / "b1_formats"

    def recorded(name: str) -> str:
        return json.loads((fixtures / name).read_text(encoding="utf-8"))["profile_fingerprint"]

    assert recorded("lane.json") == B1[profile_extraction_fingerprint]
    assert recorded("report.json") == recorded("dataset.json") == B1[profile_comparison_fingerprint]


def test_the_tco_content_hash_keeps_its_b1_value(tco_profile: DomainProfile):
    assert tco_profile.content_hash == TCO_CONTENT_HASH
