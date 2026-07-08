"""DEV-1649: run_edited_models_archive path shape + worktree-safety.

Mirrors test_annotation_io_runs / test_paths_runs: the helper MUST anchor at
the main checkout (never the worktree) and honour BIRD_RUNS_ROOT.
"""

from __future__ import annotations


def test_run_edited_models_archive_shape(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod

    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    monkeypatch.delenv("BIRD_RUNS_ROOT", raising=False)
    from bird_interact_agents.eval.annotation_io import run_edited_models_archive

    p = run_edited_models_archive(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1",
    )
    assert p == (
        tmp_path / "runs" / "mini-interact" / "alien" / "alien_1"
        / "edited_models.tar.gz"
    )


def test_run_edited_models_archive_not_under_worktrees(monkeypatch):
    import bird_interact_agents.paths as paths_mod

    monkeypatch.delenv("BIRD_RUNS_ROOT", raising=False)
    from bird_interact_agents.eval.annotation_io import run_edited_models_archive

    p = run_edited_models_archive(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1",
    )
    assert ".worktrees" not in p.parts


def test_run_edited_models_archive_env_override(monkeypatch, tmp_path):
    override = tmp_path / "custom_runs"
    monkeypatch.setenv("BIRD_RUNS_ROOT", str(override))
    from bird_interact_agents.eval.annotation_io import run_edited_models_archive

    p = run_edited_models_archive(
        benchmark="mini-interact", selected_database="alien",
        instance_id="alien_1",
    )
    assert str(p).startswith(str(override))
    assert p.name == "edited_models.tar.gz"
