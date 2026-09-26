"""AC-10: the package names no domain. Every domain word lives in ``profiles/``; the code only carries what a
profile hands it.

Two readings, one per kind of source. Python is read through its AST, so a docstring or a comment may still say
"TCO" when it explains *why* (the code's behaviour never depends on it); every other string constant counts.
The web assets are read after their comments are stripped, for the same reason: what the browser shows is the
strings, not the comments.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import paperfacts

SOURCE = Path(paperfacts.__file__).parent
STATIC = SOURCE / "web" / "static"
PY_FILES = sorted(SOURCE.rglob("*.py"))
WEB_FILES = sorted(path for path in STATIC.iterdir() if path.suffix in {".js", ".html", ".css"})

# The TCO profile's words (spec section 6). "ITO" and "FTO" are matched as whole upper-case words, so they do
# not fire inside "EDITOR" or a lower-case word; "tco" in any case and wherever no letter touches it, so an
# identifier such as ``tco_x`` counts too; the rest are matched in any case.
FORBIDDEN = (
    re.compile(r"(?<![A-Za-z])tco(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"\bITO\b"),
    re.compile(r"\bFTO\b"),
    re.compile(r"sputter", re.IGNORECASE),
    re.compile(r"perovskite", re.IGNORECASE),
    re.compile(r"transparent", re.IGNORECASE),
    re.compile(r"\bfilms?\b", re.IGNORECASE),
    re.compile(r"transmittance", re.IGNORECASE),
    re.compile("靶材"),
    re.compile("薄膜"),
    re.compile("溅射"),
    # The catalysis example's words (profiles/catalysis.json).
    re.compile(r"catalys", re.IGNORECASE),
    re.compile(r"methanol", re.IGNORECASE),
    re.compile("催化"),
)
# Exact phrases allowed per file (keyed by the path under the package) with the most times each may occur, so a
# new occurrence of an allowed phrase fails just like a new word. Each is a legacy name read from stored files,
# is the default profile's name, or is not the domain word.
# - records.py ``"no_tco_film"``: the read alias of ``LaneExtraction.no_samples``, the name lanes written before
#   round 2 store it under. The model-facing key of that name is the TCO profile's ``no_samples_key``.
# - config.py ``DEFAULT_PROFILE = "tco"``, readings.py ``LEGACY_PROFILE = "tco"``: the shipped profile's name
#   (``profiles/tco.json``), the only literal spelling of it in the package. It is a file name, not copy.
# - app.css ``transparent``: the CSS colour keyword.
ALLOWED: dict[str, dict[str, int]] = {
    "records.py": {'"no_tco_film"': 1},
    "config.py": {'"tco"': 1},
    "readings.py": {'"tco"': 1},
    "web/static/app.css": {"transparent": 12},
}
# The internal names round 2 retired (spec §1.8), matched in the whole source, identifiers included: the
# paper-level record is ``paper`` / ``PaperRecord`` and a report holds ``matchings`` per entity. ``event.target``
# is the DOM's.
RETIRED = (
    re.compile(r"TargetRecord"),
    re.compile(r"(?<!event)\.target\b"),
    re.compile(r"report\.matching\b"),
)


def allowed_for(path: Path) -> dict[str, int]:
    return ALLOWED.get(path.relative_to(SOURCE).as_posix(), {})


def hits(text: str, allowed: dict[str, int]) -> list[str]:
    """The forbidden words left in ``text`` once each allowed phrase is taken out, plus every allowed phrase that
    occurs more often than its count."""
    over = {f"{phrase} (more than {count})" for phrase, count in allowed.items() if text.count(phrase) > count}
    for phrase in allowed:
        text = text.replace(phrase, "")
    return sorted(over | {match.group(0) for pattern in FORBIDDEN for match in pattern.finditer(text)})


def python_strings(path: Path) -> str:
    """Every string constant of a module except its docstrings, one per line and in double quotes, so an allowed
    phrase can be a whole constant (``"tco"``) without also allowing it inside a longer one."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return "\n".join(
        f'"{node.value}"'
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    )


# The characters after which a "/" starts a regular expression literal rather than a division.
_REGEX_AFTER = set("(,=:[!&|?{};+-*%<>~^") | {""}


def strip_js_comments(source: str) -> str:
    """``source`` without its ``//`` and ``/* */`` comments. Strings, template literals and regular expression
    literals are copied through untouched, so a "//" inside a URL or a quote inside a regex does not derail it."""
    out: list[str] = []
    i, n = 0, len(source)
    last = ""  # the last significant character copied, to tell a regex from a division
    while i < n:
        char, pair = source[i], source[i : i + 2]
        if pair == "//":
            end = source.find("\n", i)
            i = n if end < 0 else end
            continue
        if pair == "/*":
            end = source.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if char in "\"'`" or (char == "/" and last in _REGEX_AFTER):
            start, in_class = i, False
            i += 1
            while i < n:
                current = source[i]
                if current == "\\":
                    i += 2
                    continue
                if char == "/" and current == "[":
                    in_class = True
                elif char == "/" and current == "]":
                    in_class = False
                elif current == char and not in_class:
                    break
                i += 1
            out.append(source[start : i + 1])
            i += 1
            last = char
            continue
        out.append(char)
        if not char.isspace():
            last = char
        i += 1
    return "".join(out)


def web_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".js":
        return strip_js_comments(text)
    if path.suffix == ".css":
        return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


@pytest.mark.parametrize("path", PY_FILES, ids=lambda path: str(path.relative_to(SOURCE)))
def test_no_python_string_names_the_domain(path: Path):
    assert hits(python_strings(path), allowed_for(path)) == []


@pytest.mark.parametrize("path", WEB_FILES, ids=lambda path: path.name)
def test_no_web_asset_names_the_domain(path: Path):
    assert hits(web_text(path), allowed_for(path)) == []


@pytest.mark.parametrize("path", [*PY_FILES, *WEB_FILES], ids=lambda path: str(path.relative_to(SOURCE)))
def test_no_source_uses_a_retired_internal_name(path: Path):
    text = path.read_text(encoding="utf-8")

    assert sorted({match.group(0) for pattern in RETIRED for match in pattern.finditer(text)}) == []


def test_the_retired_names_are_caught_and_the_dom_s_event_target_is_not():
    text = "lane.target\nx = TargetRecord()\nreport.matching.pairs\nevent.target.closest('a')\nreport.matchings"

    assert sorted(match.group(0) for pattern in RETIRED for match in pattern.finditer(text)) == [
        ".target",
        "TargetRecord",
        "report.matching",
    ]


def test_the_scan_sees_through_comments_but_not_into_strings():
    source = '// TCO\nconst a = "http://x"; /* 靶材 */ const b = /["\']/g; const c = "薄膜";'

    stripped = strip_js_comments(source)

    assert hits(stripped, {}) == ["薄膜"]
    assert '"http://x"' in stripped and "/[\"']/g" in stripped


def test_a_docstring_may_name_the_domain_but_a_string_may_not(tmp_path: Path):
    module = tmp_path / "m.py"
    module.write_text('"""About TCO films."""\n\nLABEL = "靶材"\n', encoding="utf-8")

    assert hits(python_strings(module), {}) == ["靶材"]


def test_the_patterns_catch_case_plurals_and_identifiers():
    assert hits("Tco\nthin films\nFilm\ntco_layer\nfilmic", {}) == ["Film", "Tco", "films", "tco"]


def test_an_allowed_phrase_is_counted_and_a_whole_constant_only_as_itself(tmp_path: Path):
    module = tmp_path / "m.py"
    module.write_text('A = "tco"\nB = "tco profile"\n', encoding="utf-8")

    assert hits(python_strings(module), {'"tco"': 1}) == ["tco"]
    assert hits('"tco"\n"tco"', {'"tco"': 1}) == ['"tco" (more than 1)']


def test_the_allowlist_only_names_files_that_exist():
    names = {path.relative_to(SOURCE).as_posix() for path in (*PY_FILES, *WEB_FILES)}

    assert set(ALLOWED) <= names
