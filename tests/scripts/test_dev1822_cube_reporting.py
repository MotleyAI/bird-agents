"""DEV-1822: cascade_for_combo recognises the `cube` mode slot + CLI choice."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cascade_for_combo.py"


def _load():
    spec = importlib.util.spec_from_file_location("cascade_for_combo", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.mark.parametrize("name,expected", [
    ("20260826-claudes-cube-abc123.json", "cube"),
    ("20260826t1000_claude_sdk_cube.json", "cube"),
    ("local_20260826t1000_claude_sdk_cube.json", "cube"),
    ("20260826-claudes-raw-abc123.json", "raw"),
    ("20260826-claudes-slayer-abc123.json", "slayer"),
])
def test_mode_from_filename_recognises_cube(name, expected):
    assert _load().mode_from_filename(name) == expected


def test_mode_choices_include_cube():
    tree = ast.parse(_SCRIPT.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--mode"):
            for kw in node.keywords:
                if kw.arg == "choices" and isinstance(kw.value, (ast.Tuple, ast.List)):
                    found = {e.value for e in kw.value.elts if isinstance(e, ast.Constant)}
    assert {"raw", "slayer", "cube"}.issubset(found)
