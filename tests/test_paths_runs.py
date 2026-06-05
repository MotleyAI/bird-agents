"""DEV-1533: paths.runs_root() worktree-safety regression.

Mirrors the contract used by annotations_root() / audited_gold_root():
the helper MUST anchor at the main checkout, and BIRD_RUNS_ROOT must
be honoured as the env override.
"""
from __future__ import annotations


def test_runs_root_anchors_at_main_checkout(monkeypatch, tmp_path):
    import bird_interact_agents.paths as p
    monkeypatch.setattr(p, "main_checkout_root", lambda: tmp_path)
    assert p.runs_root() == tmp_path / "runs"


def test_runs_root_not_under_worktrees():
    import bird_interact_agents.paths as p
    root = p.runs_root()
    assert ".worktrees" not in root.parts


def test_bird_runs_root_env_override(monkeypatch, tmp_path):
    import bird_interact_agents.paths as p
    override = tmp_path / "custom_runs"
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(override))
    # Clear the functools.cache so the env var is re-read.
    # runs_root() doesn't use the cached _main_checkout_root_cached — it's
    # a plain function — so just calling it is sufficient.
    result = p.runs_root()
    assert result == override
    assert override.exists()
