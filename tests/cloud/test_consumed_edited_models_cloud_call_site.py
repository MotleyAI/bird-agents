"""DEV-1778: the CLOUD grade call-site (`ray_app._run_one_in_actor`) forwards
the row's `consumed_edited_models` into the grader on BOTH the success path
(`_grade_one_submission`) and the fail-everything path
(`write_failed_submission_annotation`) — driven end-to-end through
`run_pool(local_only=True)` with a stubbed runner (Codex #5)."""
from __future__ import annotations

from pathlib import Path

RUN_ID = "20260811T0000-otf-slayer-dev1778"
_CONSUMED = {"db": "alien", "instance_id": "alien_1", "store_fp": "cd" * 32}


def _min_ann_path(rows_dir, iid) -> Path:
    """Write a minimal valid annotation the grade block can upload."""
    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification, SubmissionAnnotation, SubmissionEvaluation,
        SubmissionMetadata,
    )
    ann = SubmissionAnnotation(
        instance_id=iid, selected_database="alien",
        task_annotation_ref=f"annotations/mini-interact/alien/{iid}.task.json",
        annotated_by="stub", annotated_at="2026-08-11",
        submission=SubmissionMetadata(
            cloud_run_id=RUN_ID, trajectory_path=f"rows/{iid}/attempt-1.json",
        ),
        evaluation=SubmissionEvaluation(
            phase1_against_original_gold="fail",
            phase1_against_audited_primary="fail",
            phase1_against_any_audited_variant="fail",
            verdict="agent_miss",
        ),
        failure_classification=FailureClassification(
            primary="agent_miss", agent_at_fault=True, remediation_target="agent",
        ),
    )
    d = Path(rows_dir) / iid
    d.mkdir(parents=True, exist_ok=True)
    out = d / "submission_annotation.json"
    out.write_text(ann.model_dump_json(exclude_none=False) + "\n")
    return out


def _run(monkeypatch, client):
    from bird_interact_agents.cloud import ray_app

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
    # forwards consumed_edited_models is framework-agnostic (it reads the row).
    ray_app.run_pool(
        run_id=RUN_ID, instance_ids=["alien_1"], framework="pydantic_ai",
        query_mode="raw", mode="c-interact",
        agent_model="anthropic/claude-haiku-4-5-20251001", num_actors=1, attempt=1,
        task_data_by_id={
            "alien_1": {"instance_id": "alien_1", "selected_database": "alien"},
        },
        dataset="mini-interact", local_only=True,
    )


def test_cloud_success_path_forwards_consumed(monkeypatch, fake_gcs_bucket):
    """Deterministic success path: `_grade_one_submission` RECEIVES the row's
    consumed record (not merely satisfied by the fallback writer)."""
    from bird_interact_agents.cloud import ray_app

    client, _store = fake_gcs_bucket
    captured: dict = {}

    def _stub_grade(**kw):
        captured["consumed"] = kw.get("consumed_edited_models")
        return _min_ann_path(kw["rows_dir"], kw["task_data"]["instance_id"])

    monkeypatch.setattr(ray_app, "_grade_one_submission", _stub_grade)
    _run(monkeypatch, client)
    assert captured["consumed"] == _CONSUMED


def test_cloud_failure_path_forwards_consumed(monkeypatch, fake_gcs_bucket):
    """Deterministic failure path: when grading raises, the fallback
    `write_failed_submission_annotation` still RECEIVES the consumed record."""
    from bird_interact_agents.cloud import ray_app

    client, _store = fake_gcs_bucket
    captured: dict = {}

    def _raise(**kw):
        raise RuntimeError("grader boom")

    def _stub_failed(**kw):
        captured["consumed"] = kw.get("consumed_edited_models")
        return _min_ann_path(kw["rows_dir"], kw["instance_id"])

    monkeypatch.setattr(ray_app, "_grade_one_submission", _raise)
    monkeypatch.setattr(ray_app, "write_failed_submission_annotation", _stub_failed)
    _run(monkeypatch, client)
    assert captured["consumed"] == _CONSUMED
