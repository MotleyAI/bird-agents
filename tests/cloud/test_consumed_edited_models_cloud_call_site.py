"""DEV-1778: the CLOUD grade call-site (`ray_app._run_one_in_actor`) forwards
the row's `consumed_edited_models` onto the written annotation — end-to-end
through `run_pool(local_only=True)` with a stubbed runner (Codex #5)."""
from __future__ import annotations

RUN_ID = "20260811T0000-otf-slayer-dev1778"
_CONSUMED = {"db": "alien", "instance_id": "alien_1", "store_fp": "cd" * 32}


def test_cloud_call_site_stamps_consumed_on_annotation(monkeypatch, fake_gcs_bucket):
    from bird_interact_agents.cloud import gcs, ray_app

    client, _store = fake_gcs_bucket
    monkeypatch.setattr(ray_app, "default_gcs_client", lambda: client)
    monkeypatch.setattr(ray_app, "_maybe_build_cached_runner", lambda _cfg: None)

    async def fake_run_one_task(task_data, **_kw):
        return {
            "instance_id": task_data["instance_id"],
            "database": "alien", "selected_database": "alien",
            "submitted_sql": "SELECT 1",
            "phase1_passed": False, "phase2_passed": False, "total_reward": 0.0,
            "duration_s": 0.01, "error": None,
            # Stamped by the finalize hook after a successful apply.
            "consumed_edited_models": _CONSUMED,
        }

    monkeypatch.setattr("bird_interact_agents.run.run_one_task", fake_run_one_task)

    # Raw framework avoids the slayer-setup download; the grade call-site that
    # stamps consumed_edited_models is framework-agnostic (it reads the row).
    ray_app.run_pool(
        run_id=RUN_ID,
        instance_ids=["alien_1"],
        framework="pydantic_ai",
        query_mode="raw",
        mode="c-interact",
        agent_model="anthropic/claude-haiku-4-5-20251001",
        num_actors=1,
        attempt=1,
        task_data_by_id={
            "alien_1": {"instance_id": "alien_1", "selected_database": "alien"},
        },
        dataset="mini-interact",
        local_only=True,
    )

    ann = gcs.read_submission_annotation(RUN_ID, "alien_1", client=client)
    # Whether grading succeeds or falls back to the fail-everything writer, the
    # consumed provenance must be on the annotation.
    assert ann["consumed_edited_models"] == _CONSUMED
