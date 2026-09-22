"""Tests for ``paths.reports_root()`` — worktree-safe by construction.

Mirrors the contract of ``tests/test_cloud_paths_unchanged.py`` /
``tests/test_paths.py``: any worktree run resolves to the MAIN checkout's
``reports/`` so submission artifacts land in one place.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


# Helpers borrowed verbatim from tests/test_paths.py
def _init_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(repo_dir)], check=True
    )
    (repo_dir / "README.md").write_text("test\n")
    subprocess.run(
        ["git", "-C", str(repo_dir), "add", "README.md"], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "user.name=Test",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        check=True,
    )


@pytest.fixture(autouse=True)
def _isolate_paths(monkeypatch):
    from bird_interact_agents import paths

    paths._main_checkout_root_cached.cache_clear()
    monkeypatch.delenv("BIRD_REPORTS_ROOT", raising=False)
    yield
    paths._main_checkout_root_cached.cache_clear()


def test_reports_root_anchored_at_main_checkout(tmp_path, monkeypatch):
    """`reports_root()` resolves to <main-checkout>/reports/ even when the
    helper is called from a worktree."""
    from bird_interact_agents import paths

    main = tmp_path / "main"
    _init_repo(main)
    subprocess.run(
        [
            "git",
            "-C",
            str(main),
            "worktree",
            "add",
            "-b",
            "feature-x",
            str(tmp_path / "wt"),
        ],
        check=True,
    )

    # Pretend the importing module lives inside the worktree.
    monkeypatch.setattr(paths, "_LOOKUP_DIR", tmp_path / "wt")
    paths._main_checkout_root_cached.cache_clear()

    assert paths.reports_root() == main / "reports"


def test_reports_root_env_override_honored(tmp_path, monkeypatch):
    from bird_interact_agents import paths

    override = tmp_path / "custom-reports"
    monkeypatch.setenv("BIRD_REPORTS_ROOT", str(override))
    assert paths.reports_root() == override


def test_reports_root_default_creates_directory(tmp_path, monkeypatch):
    """The default reports root is created lazily (mirrors runs_root)."""
    from bird_interact_agents import paths

    main = tmp_path / "main"
    _init_repo(main)
    monkeypatch.setattr(paths, "_LOOKUP_DIR", main)
    paths._main_checkout_root_cached.cache_clear()

    root = paths.reports_root()
    assert root.exists()
    assert root.is_dir()
