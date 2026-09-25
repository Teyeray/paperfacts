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

# The TCO profile's words (spec section 6). The acronyms are matched as whole upper-case words, so "ITO" does
# not fire inside "EDITOR"; the rest are matched in any case.
FORBIDDEN = (
    re.compile(r"\bTCO\b"),
    re.compile(r"\bITO\b"),
    re.compile(r"\bFTO\b"),
    re.compile(r"sputter", re.IGNORECASE),
    re.compile(r"perovskite", re.IGNORECASE),
    re.compile(r"transparent", re.IGNORECASE),
    re.compile(r" film", re.IGNORECASE),
    re.compile(r"transmittance", re.IGNORECASE),
    re.compile("靶材"),
    re.compile("薄膜"),
    re.compile("溅射"),
)
# Exact phrases allowed per file; each goes with the deferred ``target``/``no_tco_film`` rename (spec section 0)
# or is not the domain word at all.
# - ``no_tco_film``: the persisted inventory flag, an internal name this migration keeps.
# - records.py ``no TCO film``: the descriptions of that same flag. records.py is hashed into extractor_key, so
#   rewording them would re-key every stored extraction; they change together with the rename.
# - app.css ``transparent``: the CSS colour keyword.
ALLOWED: dict[str, tuple[str, ...]] = {
    "records.py": ("no_tco_film", "no TCO film"),
    "extract.py": ("no_tco_film",),
    "state.js": ("no_tco_film",),
    "app.css": ("transparent",),
}


def hits(text: str, allowed: tuple[str, ...]) -> list[str]:
    for word in allowed:
        text = text.replace(word, "")
    return sorted({match.group(0) for pattern in FORBIDDEN for match in pattern.finditer(text)})


def python_strings(path: Path) -> str:
    """Every string constant of a module except its docstrings, one per line."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return "\n".join(
        node.value
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
    assert hits(python_strings(path), ALLOWED.get(path.name, ())) == []


@pytest.mark.parametrize("path", WEB_FILES, ids=lambda path: path.name)
def test_no_web_asset_names_the_domain(path: Path):
    assert hits(web_text(path), ALLOWED.get(path.name, ())) == []


def test_the_scan_sees_through_comments_but_not_into_strings():
    source = '// TCO\nconst a = "http://x"; /* 靶材 */ const b = /["\']/g; const c = "薄膜";'

    stripped = strip_js_comments(source)

    assert hits(stripped, ()) == ["薄膜"]
    assert '"http://x"' in stripped and "/[\"']/g" in stripped


def test_a_docstring_may_name_the_domain_but_a_string_may_not(tmp_path: Path):
    module = tmp_path / "m.py"
    module.write_text('"""About TCO films."""\n\nLABEL = "靶材"\n', encoding="utf-8")

    assert hits(python_strings(module), ()) == ["靶材"]


def test_the_allowlist_only_names_files_that_exist():
    names = {path.name for path in (*PY_FILES, *WEB_FILES)}

    assert set(ALLOWED) <= names
