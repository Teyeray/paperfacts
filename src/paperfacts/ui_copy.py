"""The Chinese display copy a profile may set: what the web page and the workbook call its paper-level record and
its samples.

Kept out of :mod:`paperfacts.profile` on purpose. That module's source is hashed into the cache keys, and copy
decides no answer and no verdict, so editing a default here must never rename a stored extraction or comparison.
No key list hashes this module (``tests/test_keys_unhashed.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

# What an article type (extract.detect_article_type) is called on the page and in the workbook. Domain-free: a
# review is a review in every field of research.
ARTICLE_TYPE_ZH: dict[str, str] = {"review": "综述"}


def article_type_zh(article_type: str | None) -> str | None:
    """The Chinese label of ``article_type``, the type itself when it has none, None for an ordinary paper."""
    return None if article_type is None else ARTICLE_TYPE_ZH.get(article_type, article_type)


@dataclass(frozen=True)
class UiCopy:
    """A profile's ``ui`` object. The defaults are domain-free; a profile names its own entities."""

    # A paper-level record wherever it is shown in full: the column header, a fact's scope.
    paper_level_label_zh: str = "论文级"
    # The same where there is room for a word only: the sample list's first entry.
    paper_level_short_zh: str = "论文级"
    # What one sample is called.
    entity_label_zh: str = "样品"
    # Shown in place of the sample table when the inventory found no in-scope sample.
    no_samples_message_zh: str = "该论文没有范围内的样品，所以没有样品级数据。"
