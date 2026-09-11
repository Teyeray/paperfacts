"""Content fingerprints of source files, used to key caches on the code that produced them.

Several caches must be invalidated when the logic behind them changes: the extraction cache when the
document renderer or the cleaning rules change, the comparison cache when a normalisation rule changes.
Hashing the source is more reliable than a hand-maintained version number, because nobody has to remember
to bump it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Hex characters kept from a fingerprint. Long enough that a collision is not a practical concern, short
# enough that the resulting filenames stay readable.
FINGERPRINT_LENGTH = 12


def content_fingerprint(material: str) -> str:
    """Fingerprint of an already-serialised description of whatever the cache depends on."""
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:FINGERPRINT_LENGTH]


def source_fingerprint(*paths: Path) -> str:
    """Fingerprint of the given source files, in the order given."""
    return content_fingerprint("\n".join(path.read_text(encoding="utf-8") for path in paths))
