"""Spec §9: 'grep the whole agent/ package for task_image and fail if found.'

Implemented via `ast` rather than a literal text grep: several modules legitimately *document*
the §0.6 constraint in docstrings/comments (e.g. "info['task_image'] is the target image PIL
object, never read it") — a naive text grep would flag that documentation forever. What actually
matters is whether any *code* (a subscript or a .get()/.pop() call) reads the key, which AST lets
us check precisely.

The one sanctioned exception is questioner.py's __init__, which pops and immediately discards
`info["task_image"]` (spec §0.6 requires exactly this: "add an assertion in __init__ that
pops/discards it"). Every other code-level reference — anywhere in agent/ — is a violation.
"""

import ast
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
_SANCTIONED_FILE = "questioner.py"
_KEY = "task_image"


def _is_task_image_string(node) -> bool:
    return isinstance(node, ast.Constant) and node.value == _KEY


def _find_references(tree: ast.AST):
    """Yields (lineno, is_sanctioned_pop) for every code-level reference to the "task_image" key."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _is_task_image_string(node.slice):
            yield node.lineno, False
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("get", "pop") and node.args and _is_task_image_string(node.args[0]):
                is_sanctioned_pop = node.func.attr == "pop" and len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value is None
                yield node.lineno, is_sanctioned_pop


def test_no_unsanctioned_task_image_references():
    violations = []
    for path in sorted(AGENT_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for lineno, is_sanctioned_pop in _find_references(tree):
            if path.name == _SANCTIONED_FILE and is_sanctioned_pop:
                continue
            violations.append(f"{path.name}:{lineno}")

    assert not violations, "Unsanctioned code-level reference(s) to info['task_image'] found:\n" + "\n".join(violations)


def test_sanctioned_discard_line_still_present():
    """Guards the other direction: if someone removes the discard entirely, this should fail
    loudly rather than silently starting to leak task_image through self.info.
    """
    tree = ast.parse((AGENT_DIR / _SANCTIONED_FILE).read_text())
    assert any(is_pop for _, is_pop in _find_references(tree)), (
        "questioner.py no longer discards info['task_image'] — spec §0.6 requires this."
    )
