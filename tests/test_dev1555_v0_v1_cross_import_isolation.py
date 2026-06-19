"""DEV-1555 v0/v1 split — no cross-imports between v0 and v1 packages.

The four v0 packages must never import from the four v1 packages, and
vice versa. Otherwise a "v0" run silently inherits v1 behaviour (or v1
silently inherits v0), and the A/B comparison this PR is designed to
support becomes meaningless.

This is the regression test for Codex review Finding #6 — directory
inheritance is preserved only if child modules use SAME-SIDE imports.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import bird_interact_agents


_V0_PKGS = (
    "claude_sdk_otf",
    "claude_sdk_otf_raw",
    "claude_sdk_otf_ainteract",
    "claude_sdk_otf_ainteract_raw",
)

_V1_PKGS = (
    "claude_sdk_otf_v1",
    "claude_sdk_otf_raw_v1",
    "claude_sdk_otf_ainteract_v1",
    "claude_sdk_otf_ainteract_raw_v1",
)


def _agents_root() -> Path:
    """Filesystem path to ``bird_interact_agents/agents/``."""
    return Path(bird_interact_agents.__file__).parent / "agents"


def _walk_py_files(dir_path: Path):
    for p in dir_path.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def _read(p: Path) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


_V0_NAME_PATTERN = "|".join(re.escape(n) for n in _V0_PKGS)
_V1_NAME_PATTERN = "|".join(re.escape(n) for n in _V1_PKGS)


# A reference to a v0 package from inside a v1 file looks like
#     `from bird_interact_agents.agents.<v0_name>(.…)` or
#     `import bird_interact_agents.agents.<v0_name>(.…)` or
#     a string mention `"bird_interact_agents.agents.<v0_name>(.…)"`.
#
# The trick: each v0 name is a strict PREFIX of the corresponding v1 name
# (`claude_sdk_otf` vs `claude_sdk_otf_v1`). The regex must enforce that
# the match is NOT followed by `_v1` — i.e. the next character after the
# v0 name is `.`, end-of-word, quote, or whitespace.

_V0_REF_FROM_V1 = re.compile(
    rf"bird_interact_agents\.agents\.({_V0_NAME_PATTERN})(?!_v1)\b"
)
_V1_REF_FROM_V0 = re.compile(
    rf"bird_interact_agents\.agents\.({_V1_NAME_PATTERN})\b"
)


# ---------------------------------------------------------------------------
# AST-based import scan — catches absolute, relative, and `import-as`
# forms that a string regex over the source bytes can miss.
# ---------------------------------------------------------------------------


def _resolve_relative(module: str | None, level: int, current_pkg: str) -> str:
    """Resolve a `from .x.y import z` to its absolute module path.

    ``current_pkg`` is the dotted package of the file holding the import
    statement (e.g. ``bird_interact_agents.agents.claude_sdk_otf_v1``).
    """
    if level == 0:
        return module or ""
    parts = current_pkg.split(".")
    # `level=1` means relative to current package's parent of the file.
    # In Python, `from . import x` inside `pkg.module` resolves to
    # `pkg.x`. So drop `level - 1` trailing components if module is None,
    # or drop `level` components and append the module otherwise.
    if level - 1 > 0:
        parts = parts[: -(level - 1)] if level - 1 <= len(parts) else []
    base = ".".join(parts)
    if module:
        return f"{base}.{module}"
    return base


def _imported_modules(tree: ast.Module, current_pkg: str) -> set[str]:
    """All fully-resolved module paths referenced by import statements."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative(
                node.module, node.level or 0, current_pkg
            )
            if resolved:
                out.add(resolved)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
    return out


def _module_belongs_to_any(module: str, package_names: tuple[str, ...]) -> bool:
    """True if ``module`` is under ``bird_interact_agents.agents.<pkg>(.…)``."""
    for pkg in package_names:
        prefix = f"bird_interact_agents.agents.{pkg}"
        if module == prefix or module.startswith(prefix + "."):
            return True
    return False


def _current_pkg_for_file(fp: Path) -> str:
    """Map ``…/src/bird_interact_agents/agents/.../foo.py`` → its package dotted name."""
    parts = fp.with_suffix("").parts
    if "bird_interact_agents" not in parts:
        return ""
    idx = parts.index("bird_interact_agents")
    # The package the FILE lives in is the parent dotted path
    # (drop the filename, which is `parts[-1]`).
    dotted = ".".join(parts[idx:-1])
    return dotted


@pytest.mark.parametrize("v1_pkg", _V1_PKGS)
def test_v1_package_does_not_import_v0_packages(v1_pkg: str):
    """No v1 file under ``agents/<v1_pkg>/`` IMPORTS any v0 package name.

    Uses AST to walk every ``import`` and ``from … import`` statement,
    resolving relative imports via the file's containing package. Catches
    absolute imports (``from bird_interact_agents.agents.claude_sdk_otf
    import …``), relative imports (``from ..claude_sdk_otf import …``),
    and ``import bird_interact_agents.agents.claude_sdk_otf`` forms.
    """
    pkg_dir = _agents_root() / v1_pkg
    if not pkg_dir.is_dir():
        pytest.fail(
            f"v1 package directory missing: {pkg_dir} — implementation "
            "step `Move this branch's claude_sdk_otf*/ into *_v1/ dirs` "
            "is not done yet."
        )
    offenders: list[tuple[Path, str]] = []
    for fp in _walk_py_files(pkg_dir):
        try:
            tree = ast.parse(_read(fp))
        except SyntaxError:
            continue
        current_pkg = _current_pkg_for_file(fp)
        for mod in _imported_modules(tree, current_pkg):
            if _module_belongs_to_any(mod, _V0_PKGS):
                offenders.append(
                    (fp.relative_to(_agents_root()), mod)
                )
    assert not offenders, (
        f"v1 package {v1_pkg} imports v0 packages: "
        + ", ".join(f"{f}: {ref}" for f, ref in offenders)
    )


@pytest.mark.parametrize("v0_pkg", _V0_PKGS)
def test_v0_package_does_not_import_v1_packages(v0_pkg: str):
    """No v0 file under ``agents/<v0_pkg>/`` IMPORTS any v1 package name.

    Same AST-based approach as the v1 side.
    """
    pkg_dir = _agents_root() / v0_pkg
    if not pkg_dir.is_dir():
        pytest.fail(
            f"v0 package directory missing: {pkg_dir} — implementation "
            "step `Restore four claude_sdk_otf*/ v0 dirs from origin/main` "
            "is not done yet."
        )
    offenders: list[tuple[Path, str]] = []
    for fp in _walk_py_files(pkg_dir):
        try:
            tree = ast.parse(_read(fp))
        except SyntaxError:
            continue
        current_pkg = _current_pkg_for_file(fp)
        for mod in _imported_modules(tree, current_pkg):
            if _module_belongs_to_any(mod, _V1_PKGS):
                offenders.append(
                    (fp.relative_to(_agents_root()), mod)
                )
    assert not offenders, (
        f"v0 package {v0_pkg} imports v1 packages: "
        + ", ".join(f"{f}: {ref}" for f, ref in offenders)
    )


# ---------------------------------------------------------------------------
# Belt-and-braces: source-string scan catches dynamic / string-form
# references (e.g. importlib.import_module("…claude_sdk_otf")) that an
# AST import scan would miss. False positives are possible from
# docstrings/comments — accept that as a tradeoff.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v1_pkg", _V1_PKGS)
def test_v1_source_does_not_string_reference_v0(v1_pkg: str):
    """String / docstring scan: no v0 fully-qualified name in v1 source.

    This catches ``importlib.import_module("bird_interact_agents.agents.
    claude_sdk_otf")`` and similar dynamic forms the AST import scan
    skips. Accepts the (small) docstring false-positive risk; if a v1
    docstring genuinely needs to mention a v0 module name, use the
    short form ``claude_sdk_otf`` (without the package prefix) so this
    test stays quiet.
    """
    pkg_dir = _agents_root() / v1_pkg
    if not pkg_dir.is_dir():
        pytest.skip(f"v1 dir missing: {pkg_dir}")
    offenders: list[tuple[Path, str]] = []
    for fp in _walk_py_files(pkg_dir):
        src = _read(fp)
        for m in _V0_REF_FROM_V1.finditer(src):
            offenders.append((fp.relative_to(_agents_root()), m.group(0)))
    assert not offenders, (
        f"v1 package {v1_pkg} string-references v0 fully-qualified names: "
        + ", ".join(f"{f}: {ref}" for f, ref in offenders)
    )


@pytest.mark.parametrize("v0_pkg", _V0_PKGS)
def test_v0_source_does_not_string_reference_v1(v0_pkg: str):
    """String / docstring scan: no v1 fully-qualified name in v0 source."""
    pkg_dir = _agents_root() / v0_pkg
    if not pkg_dir.is_dir():
        pytest.skip(f"v0 dir missing: {pkg_dir}")
    offenders: list[tuple[Path, str]] = []
    for fp in _walk_py_files(pkg_dir):
        src = _read(fp)
        for m in _V1_REF_FROM_V0.finditer(src):
            offenders.append((fp.relative_to(_agents_root()), m.group(0)))
    assert not offenders, (
        f"v0 package {v0_pkg} string-references v1 fully-qualified names: "
        + ", ".join(f"{f}: {ref}" for f, ref in offenders)
    )


def test_all_four_v0_dirs_present():
    """All four v0 package directories exist as expected by the dispatcher."""
    root = _agents_root()
    missing = [p for p in _V0_PKGS if not (root / p).is_dir()]
    assert not missing, (
        f"Missing v0 package directories: {missing}. "
        "Implementation must restore them from origin/main."
    )


def test_all_four_v1_dirs_present():
    """All four v1 package directories exist as expected by the dispatcher."""
    root = _agents_root()
    missing = [p for p in _V1_PKGS if not (root / p).is_dir()]
    assert not missing, (
        f"Missing v1 package directories: {missing}. "
        "Implementation must move this branch's `claude_sdk_otf*/` contents "
        "into the `_v1/` dirs."
    )
