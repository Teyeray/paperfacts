"""AC-11: the package holds no domain table of its own; every stage is handed the profile it runs under.

Two checks, one per way a table could come back. Importing every module and looking at what it binds catches a
field table or a profile however it is spelled; reading the source catches the names the migration removed and a
read of ``config.json``'s field table, which would put a second source of fields beside the profile.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from collections.abc import Iterator
from pathlib import Path

import pytest

import paperfacts
from paperfacts.fields import FieldSpec
from paperfacts.profile import DomainProfile, RetrievalSpec

SOURCE = Path(paperfacts.__file__).parent
MODULES = sorted(info.name for info in pkgutil.walk_packages(paperfacts.__path__, prefix="paperfacts."))
FILES = sorted(SOURCE.rglob("*.py"))

# The module-level names the profile replaced (spec section 2.1). None may be bound again, whatever its value.
REMOVED = {
    "FIELD_SPECS",
    "FIELD_BY_NAME",
    "TARGET_FIELDS",
    "SAMPLE_FIELDS",
    "CONDITION_KEYWORDS",
    "FIELDS_SOURCE",
    "AMBIGUOUS_MATCH_CONFIDENCE",
    "_CONFIG",
    "CONDITION_UNIT",
    "UNIT_PATTERNS",
    "DEFAULT_RETRIEVAL",
    "_DATA_COLUMNS",
    "CONFIG_GROUP_LEVELS",
}
# config.json keys that held the domain until the profile did. The file still carries them until S9, and no
# code may read them.
DOMAIN_CONFIG_KEYS = {"fields", "condition_keywords"}
# The calls that load a configuration or a profile: at module level they would bind one at import.
LOADERS = {"configuration", "load_config", "load_profile", "_load_resolved", "parse_profile"}


def holds_domain(value: object) -> bool:
    if isinstance(value, FieldSpec | DomainProfile | RetrievalSpec):
        return True
    if isinstance(value, tuple | list | set | frozenset):
        return any(holds_domain(item) for item in value)
    if isinstance(value, dict):
        return any(holds_domain(item) for item in value.values())
    return False


@pytest.mark.parametrize("name", MODULES)
def test_no_module_binds_a_field_or_a_profile_at_import(name):
    module = importlib.import_module(name)

    bound = [attribute for attribute, value in vars(module).items() if holds_domain(value)]

    assert bound == [], f"{name} binds domain state at import: {', '.join(bound)}"


def module_level(tree: ast.Module) -> Iterator[ast.stmt]:
    """Every statement that runs at import: the module body, and the bodies of its top-level if/try blocks."""
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, ast.If | ast.Try):
            pending += [*node.body, *node.orelse, *getattr(node, "finalbody", []), *getattr(node, "handlers", [])]


def bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.Assign):
        return {name.id for target in node.targets for name in ast.walk(target) if isinstance(name, ast.Name)}
    if isinstance(node, ast.AnnAssign | ast.AugAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    if isinstance(node, ast.ImportFrom | ast.Import):
        return {alias.asname or alias.name for alias in node.names}
    return set()


def called(node: ast.AST) -> set[str]:
    names = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            func = call.func
            names.add(func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else "")
    return names


@pytest.mark.parametrize("path", FILES, ids=lambda path: str(path.relative_to(SOURCE)))
def test_no_module_binds_a_removed_name_or_loads_at_import(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in module_level(tree):
        assert not bound_names(node) & REMOVED, f"{path.name}:{node.lineno} binds {bound_names(node) & REMOVED}"
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue  # a body runs when called, and a decorator only wraps
        loads = called(node) & LOADERS
        assert not loads, f"{path.name}:{node.lineno} calls {', '.join(sorted(loads))} at import"


def reads_config_key(tree: ast.Module) -> list[int]:
    """Lines that ask a configuration document for a domain key: ``document.entries("fields")`` or
    ``document.get("condition_keywords", ...)``, the two ways ``config.json``'s table was read."""
    return [
        call.lineno
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"entries", "get", "text_or_none"}
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value in DOMAIN_CONFIG_KEYS
    ]


@pytest.mark.parametrize("path", FILES, ids=lambda path: str(path.relative_to(SOURCE)))
def test_no_module_reads_the_config_json_field_table(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if path.name == "profile.py":
        # The profile's own file has keys of the same name; reading them is the point.
        return

    assert reads_config_key(tree) == []


def test_the_checks_see_a_global_table_and_a_config_read():
    # A guard that could never fail proves nothing: both shapes the migration removed are caught.
    tree = ast.parse(
        "from paperfacts.config import configuration\n"
        "_CONFIG = configuration()\n"
        "words = _CONFIG.entries('condition_keywords')\n"
    )

    statements = list(module_level(tree))
    assert any(bound_names(node) & REMOVED for node in statements)
    assert any(called(node) & LOADERS for node in statements)
    assert reads_config_key(tree) == [3]
    assert holds_domain({"x": (FieldSpec(name="x", group="g", kind="text", description="d", keywords=()),)})
