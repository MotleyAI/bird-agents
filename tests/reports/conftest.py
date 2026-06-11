"""Pytest fixtures for the bird_interact_agents.reports test suite (DEV-1553).

The builder helpers live in ``tests/reports/_fixtures.py`` so tests can
import them directly without going through pytest's fixture system.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.reports._fixtures import stage_run


# ---------------------------------------------------------------------------
# Deterministic offline tokenizer
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_count_tokens(monkeypatch):
    """Monkeypatch reports.tokens.count_tokens with a deterministic char-based fake.

    Returns ``max(1, len(s) // 4)`` so tests can land exactly on the
    250/1000 thresholds by sizing the string. Never contacts Anthropic.
    """

    def _fake(s: str, *, model: str = "claude-haiku-4-5-20251001") -> int:
        return max(1, len(s) // 4)

    from bird_interact_agents.reports import tokens as _tokens

    monkeypatch.setattr(_tokens, "count_tokens", _fake)
    return _fake


# ---------------------------------------------------------------------------
# Stage a synthetic run on disk + rewire the paths roots
# ---------------------------------------------------------------------------


@pytest.fixture
def stage(tmp_path: Path, monkeypatch):
    """Return a callable that stages a run and rewires paths.runs_root /
    paths.results_root to the staged tmp directory."""
    from bird_interact_agents import paths

    def _do(**kwargs) -> tuple[Path, Path]:
        runs_root, results_root = stage_run(tmp_path, **kwargs)
        monkeypatch.setattr(paths, "runs_root", lambda: runs_root)
        monkeypatch.setattr(paths, "results_root", lambda: results_root)
        return runs_root, results_root

    return _do
