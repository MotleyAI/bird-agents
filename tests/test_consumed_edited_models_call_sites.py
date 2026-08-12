"""DEV-1778: the LOCAL grade call-sites (`run._grade_local_row`) forward the
row's `consumed_edited_models` into the grader — on the success path AND onto
the FAILED annotation when the grader raises after a successful apply (Codex
#5). End-to-end through `run_evaluation` (the call-sites are nested closures)."""
from __future__ import annotations

from pathlib import Path

import pytest

_CONSUMED = {"db": "alien", "instance_id": "alien_1", "store_fp": "ab" * 32}


@pytest.fixture(autouse=True)
def _force_legacy_inprocess(monkeypatch):
    # The process pool can't see monkeypatches; force the in-process path.
    monkeypatch.setenv("BIRD_INTERACT_LOCAL_INPROCESS", "1")


def _write_min_annotation(rows_dir, iid, run_id):
    """Write a minimal valid annotation to rows/ + runs/ so the downstream
    cascade aggregator in ``run_evaluation`` doesn't choke."""
    from bird_interact_agents.eval.annotation_io import (
        run_annotation_path, write_run_annotation,
    )
    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification, SubmissionAnnotation, SubmissionEvaluation,
        SubmissionMetadata,
    )
    ann = SubmissionAnnotation(
        instance_id=iid, selected_database="alien",
        task_annotation_ref=f"annotations/mini-interact/alien/{iid}.task.json",
        annotated_by="stub", annotated_at="2026-08-11",
        submission=SubmissionMetadata(
            cloud_run_id=run_id, trajectory_path=f"rows/{iid}/attempt-1.json",
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
    write_run_annotation(ann, run_annotation_path(
        benchmark="mini-interact", selected_database="alien",
        instance_id=iid, run_id=run_id,
    ))
    return out


def _common_setup(monkeypatch, tmp_path):
    import bird_interact_agents.paths as paths_mod
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **k: [
        {"instance_id": "alien_1", "selected_database": "alien",
         "sol_sql": ["SELECT 1"], "amb_user_query": "q"},
    ])
    monkeypatch.setattr(run_mod, "_maybe_force_wipe_otf", lambda **k: None)

    async def _runner(td, data_dir, patience, user_sim_model):  # noqa: ARG001
        return {
            "instance_id": "alien_1", "database": "alien",
            "phase1_passed": False, "phase2_passed": False, "total_reward": 0.0,
            "submitted_sql": "SELECT 1", "trajectory": [], "usage": {},
            # This is what the finalize hook stamps onto the row after apply.
            "consumed_edited_models": _CONSUMED,
        }

    monkeypatch.setattr(run_mod, "_make_runner", lambda **k: _runner)
    return run_mod


async def _run(run_mod, tmp_path):
    await run_mod.run_evaluation(
        framework="claude_sdk_otf_ainteract", query_mode="slayer",
        mode="a-interact", data_path="ignored", data_dir=str(tmp_path / "d"),
        output_path=str(tmp_path / "eval.json"), concurrency=1, limit=None,
        agent_model="anthropic/claude-haiku-4-5-20251001", strict=False,
        prompt_cache=False, max_depth=1,
        slayer_storage_root=str(tmp_path / "sm"), slayer_setup="on-the-fly",
        reasoning_effort=None, use_audited_gold_sql=False,
        dataset="mini-interact", filter_ids=None,
    )


@pytest.mark.asyncio
async def test_local_success_call_site_forwards_consumed(monkeypatch, tmp_path):
    run_mod = _common_setup(monkeypatch, tmp_path)
    captured: dict = {}

    def _stub(*, task_data, submitted_sql, rows_dir, run_id, benchmark, db_path, **kw):  # noqa: ARG001
        captured["consumed"] = kw.get("consumed_edited_models")
        return _write_min_annotation(rows_dir, task_data["instance_id"], run_id)

    monkeypatch.setattr(run_mod, "grade_one_submission", _stub)
    await _run(run_mod, tmp_path)
    assert captured["consumed"] == _CONSUMED


@pytest.mark.asyncio
async def test_local_failed_annotation_retains_consumed(monkeypatch, tmp_path):
    run_mod = _common_setup(monkeypatch, tmp_path)

    def _raise(**kw):  # grader blows up AFTER a successful apply
        raise RuntimeError("grader boom")

    captured: dict = {}

    def _stub_failed(**kw):
        captured["consumed"] = kw.get("consumed_edited_models")
        return _write_min_annotation(kw["rows_dir"], kw["instance_id"], kw["run_id"])

    monkeypatch.setattr(run_mod, "grade_one_submission", _raise)
    monkeypatch.setattr(run_mod, "write_failed_submission_annotation", _stub_failed)
    await _run(run_mod, tmp_path)
    assert captured["consumed"] == _CONSUMED
