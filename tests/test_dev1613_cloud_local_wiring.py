"""DEV-1613: the cloud (ray_app) and local (run.py) inline graders both
forward the run's ``agent_model`` to ``grade_one_submission`` so it can
build the N5 judge. The judge object itself is unit-tested in
``test_dev1613_judge_helper_and_resilience``; here we pin the plumbing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests import test_run_local_inline_grader as base

# DEV-1640: these tests pin the LOCAL in-process per-task wiring / grading by
# monkeypatching agents + graders + loaders, which a spawned worker process
# cannot see. The process pool is now the default, so route run_evaluation
# through the retained legacy single-loop path (identical per-task wiring).
@pytest.fixture(autouse=True)
def _dev1640_force_legacy_inprocess(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_LOCAL_INPROCESS", "1")


# ---------------------------------------------------------------------------
# Local (run.py) — _grade_local_row must forward agent_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_grader_receives_agent_model(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    rows = [{"instance_id": "alien_1", "selected_database": "alien",
             "sol_sql": ["SELECT 1"], "amb_user_query": "q1"}]
    base._patch_loader_returns(monkeypatch, rows)
    monkeypatch.setattr(run_mod, "_maybe_force_wipe_otf", lambda **kw: None)
    base._stub_runner_factory(monkeypatch, {
        "alien_1": {
            "instance_id": "alien_1", "database": "alien",
            "phase1_passed": False, "phase2_passed": False,
            "total_reward": 0.0, "submitted_sql": "SELECT 1",
            "trajectory": [], "usage": {},
        },
    })

    captured: dict = {}

    def _stub_grader(*, task_data, **kw):
        captured.update(kw)
        out_dir = Path(kw["rows_dir"]) / task_data["instance_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "submission_annotation.json"
        p.write_text("{}")  # invalid-as-annotation; aggregator degrades gracefully
        return p

    monkeypatch.setattr(run_mod, "grade_one_submission", _stub_grader)

    await run_mod.run_evaluation(
        framework="claude_sdk_otf_ainteract", query_mode="slayer",
        mode="a-interact", data_path="ignored",
        data_dir=str(tmp_path / "ignored_data_dir"),
        output_path=str(tmp_path / "eval.json"),
        concurrency=1, limit=None,
        agent_model="anthropic/claude-opus-4-7",
        strict=False, prompt_cache=False, max_depth=1,
        slayer_storage_root=str(tmp_path / "slayer_models"),
        slayer_setup="on-the-fly", reasoning_effort=None,
        use_audited_gold_sql=False, dataset="mini-interact",
        filter_ids=None,
    )

    assert captured.get("agent_model") == "anthropic/claude-opus-4-7"


# ---------------------------------------------------------------------------
# Cloud (ray_app._run_one_in_actor) — the grading call must forward agent_model
# ---------------------------------------------------------------------------


def test_cloud_actor_forwards_agent_model_to_grader(monkeypatch, tmp_path):
    from bird_interact_agents.cloud import ray_app

    cfg = {
        "dataset": "mini-interact",
        "framework": "pydantic_ai",
        "query_mode": "raw",
        "mode": "a-interact",
        "agent_model": "anthropic/claude-opus-4-7",
        "user_sim_model": "anthropic/claude-sonnet-4-6",
        "patience": 3,
        "strict": False,
        "use_audited_gold_sql": False,
        "prompt_cache": False,
        "max_depth": 1,
        "reasoning_effort": None,
        "slayer_setup": "on-the-fly",
        "pre_encoded_source": None,
        "slayer_storage_root": None,
        "data_dir": str(tmp_path / "data"),
    }

    async def _fake_run_one_task_async(**kwargs):  # noqa: ANN003
        return {
            "instance_id": "alien_1",
            "database": "alien",
            "selected_database": "alien",
            "submitted_sql": "SELECT 2",
            "phase1_passed": False,
            "phase2_passed": False,
            "total_reward": 0.0,
            "duration_s": 1.0,
            "predicted_row_count": 0,
            "usage": {},
        }

    captured: dict = {}

    def _capture_grader(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        d = Path(kwargs["rows_dir"]) / kwargs["instance_id"]
        d.mkdir(parents=True, exist_ok=True)
        p = d / "submission_annotation.json"
        p.write_text("{}")
        return p

    def _noop(*a, **kw):  # noqa: ANN003
        return None

    monkeypatch.setattr(ray_app, "_run_one_task_async", _fake_run_one_task_async)
    monkeypatch.setattr(ray_app, "_grade_one_submission", _capture_grader)
    monkeypatch.setattr(ray_app._gcs, "write_submission_annotation", _noop)
    monkeypatch.setattr(ray_app._gcs, "write_row", _noop)
    monkeypatch.setattr(ray_app._gcs, "write_log", _noop)
    # DEV-1640: the upload-back triple moved into GcsStore.upload_back, which
    # calls the upload_back MODULE functions — patch those directly.
    from bird_interact_agents.cloud import upload_back as _ub
    monkeypatch.setattr(_ub, "upload_per_task_debug", _noop)
    monkeypatch.setattr(_ub, "upload_per_task_setup_sessions", _noop)
    monkeypatch.setattr(_ub, "upload_otf_reference_delta", _noop)

    ray_app._run_one_in_actor(
        task_data={"instance_id": "alien_1", "selected_database": "alien",
                   "db_file_path": "/dev/null"},
        cfg=cfg,
        run_id="r1",
        attempt=1,
        store=ray_app.GcsStore(object()),
    )

    assert captured.get("agent_model") == "anthropic/claude-opus-4-7"
