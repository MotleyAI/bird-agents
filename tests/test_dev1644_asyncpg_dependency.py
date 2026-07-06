"""DEV-1644 Fix 2: SLayer maps ``postgres -> postgresql+asyncpg``
(``slayer/sql/client.py``), so its async postgres query path imports
``asyncpg``. It was declared nowhere in ``pyproject``/``uv.lock``, so every
SLayer query on a postgres benchmark (livesqlbench-*, bird-interact-*) threw
``No module named 'asyncpg'`` and the agent burned turns routing around it.

``asyncpg`` must be declared in every extra that installs the SLayer-postgres
runtime, and it must be present in the resolved lockfile (Codex F5 — a
pyproject declaration alone doesn't guarantee an installable env).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_EXTRAS = ("slayer", "all", "postgres", "cloud")


def _extras() -> dict[str, list[str]]:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["optional-dependencies"]


def _asyncpg_req(reqs: list[str]) -> Requirement | None:
    for raw in reqs:
        req = Requirement(raw)
        if req.name.lower() == "asyncpg":
            return req
    return None


def test_asyncpg_declared_with_min_floor_in_required_extras():
    """Codex round-2 F3: the plan pins asyncpg>=0.30, so a bare `asyncpg` or a
    lower/exact pin must NOT pass — parse the specifier, don't prefix-match."""
    extras = _extras()
    for name in _REQUIRED_EXTRAS:
        assert name in extras, f"extra {name!r} missing from pyproject"
        req = _asyncpg_req(extras[name])
        assert req is not None, f"asyncpg not declared in the {name!r} extra"
        # The specifier must forbid anything below 0.30 (i.e. 0.29 excluded).
        assert not req.specifier.contains("0.29"), (
            f"{name!r} extra allows asyncpg<0.30 (spec={req.specifier})"
        )
        assert req.specifier.contains("0.30"), (
            f"{name!r} extra excludes asyncpg 0.30 (spec={req.specifier})"
        )


def test_asyncpg_present_in_lockfile():
    lock = (_REPO_ROOT / "uv.lock").read_text()
    assert 'name = "asyncpg"' in lock, (
        "asyncpg missing from uv.lock — re-run `uv lock` after declaring it"
    )


def test_asyncpg_importable():
    """The dependency must actually resolve in the synced env (the SLayer
    postgres query path does a bare ``import asyncpg``)."""
    import importlib

    assert importlib.import_module("asyncpg") is not None
