"""The page's generic copy. ``state.js`` mirrors ``ui_copy.UiCopy``'s defaults so the page can still name things
when ``/api/profile`` fails; the mirror must not drift, and nothing else on the page may spell the entity itself."""

from __future__ import annotations

import dataclasses
import re

from paperfacts.ui_copy import UiCopy
from test_domain_free import STATIC, WEB_FILES, web_text

MIRROR = re.compile(r"const GENERIC_UI_COPY = \{(.*?)\};", re.DOTALL)
ENTRY = re.compile(r'^\s*(\w+): "([^"]*)",?\s*$', re.MULTILINE)


def test_the_state_js_mirror_equals_the_python_defaults():
    block = MIRROR.search((STATIC / "state.js").read_text(encoding="utf-8"))

    assert block is not None
    assert dict(ENTRY.findall(block.group(1))) == dataclasses.asdict(UiCopy())


def test_only_the_mirror_spells_the_default_entity():
    # A literal "样品" anywhere else would stay "样品" under a profile that calls its entity something else.
    spelled = {
        path.name: text.count("样品") for path in WEB_FILES if (text := MIRROR.sub("", web_text(path))).count("样品")
    }

    assert spelled == {}
