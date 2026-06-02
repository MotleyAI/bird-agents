"""DEV-1515 round-3 follow-up: local `run.run_evaluation` MUST invoke
the inline grader after every task so the `cascading_phase1` block
lands in `eval.json`. Pre-fix the local runner just persisted each
agent result row into `results.db` and left the rows-dir aggregation
gated on `submission_annotation.json` files that were never written —
so local audited / annotated runs silently lost the N1-N9 cascade
metrics.

This test pins the wiring: a synthetic 2-task run with a stub runner
and a stub inline grader, then assert (1) the grader was called for
EVERY task and (2) ``eval.json`` carries the ``cascading_phase1``
block produced by the rows aggregator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _patch_loader_returns(monkeypatch, rows: list[dict]) -> None:
    """Same shape as the DEV-1510 wiring test — patch the loader so
    `run_evaluation` reaches the runner-call loop without real data."""
    monkeypatch.setattr(
        "bird_interact_agents.harness.load_benchmark_tasks",
        lambda *a, **kw: rows,
    )
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **kw: rows)


def _stub_runner_factory(monkeypatch, results_by_inst: dict[str, dict]):
    """Replace `_make_runner` with a factory that returns an async
    runner. The runner looks each task up by `instance_id` and returns
    the canned result row."""

    async def _stub_runner(td: dict, data_dir: str, patience: int, user_sim_model: str):
        return dict(results_by_inst[td["instance_id"]])

    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(run_mod, "_make_runner", lambda **kw: _stub_runner)


@pytest.mark.asyncio
async def test_local_run_invokes_inline_grader_per_task(monkeypatch, tmp_path):
    """End-to-end: 2 fake tasks → inline grader called twice → the
    rows-dir aggregator emits ``cascading_phase1`` in ``eval.json``."""
    import bird_interact_agents.run as run_mod

    rows = [
        {
            "instance_id": "alien_1",
            "selected_database": "alien",
            "sol_sql": ["SELECT 1"],
            "amb_user_query": "q1",
        },
        {
            "instance_id": "alien_2",
            "selected_database": "alien",
            "sol_sql": ["SELECT 2"],
            "amb_user_query": "q2",
        },
    ]
    _patch_loader_returns(monkeypatch, rows)
    monkeypatch.setattr(run_mod, "_maybe_force_wipe_otf", lambda **kw: None)
    _stub_runner_factory(monkeypatch, {
        "alien_1": {
            "instance_id": "alien_1",
            "database": "alien",
            "phase1_passed": True,
            "phase2_passed": False,
            "total_reward": 1.0,
            "submitted_sql": "SELECT 1",
            "trajectory": [],
            "usage": {"n_agent_turns": 3, "n_ask_user_calls": 1},
        },
        "alien_2": {
            "instance_id": "alien_2",
            "database": "alien",
            "phase1_passed": False,
            "phase2_passed": False,
            "total_reward": 0.0,
            "submitted_sql": "SELECT 2",
            "trajectory": [],
            "usage": {"n_agent_turns": 1, "n_ask_user_calls": 0},
        },
    })

    # Capture the grader calls — replace ``grade_one_submission`` at the
    # spot ``run.py`` looks it up (module-level import) with a stub that
    # writes a minimal ``submission_annotation.json``. The shape only has
    # to satisfy ``emit_cascading_eval_json``'s loader; the cascade body
    # itself is unit-tested elsewhere.
    calls: list[dict[str, Any]] = []

    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification,
        SubmissionAnnotation,
        SubmissionEvaluation,
        SubmissionMetadata,
        UserSimInteraction,
    )

    def _stub_grader(*, task_data, submitted_sql, rows_dir, run_id, benchmark, db_path, **kw):
        calls.append({
            "instance_id": task_data["instance_id"],
            "submitted_sql": submitted_sql,
            "benchmark": benchmark,
            "rows_dir": rows_dir,
            "run_id": run_id,
        })
        out_dir = Path(rows_dir) / task_data["instance_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        passed = task_data["instance_id"] == "alien_1"
        ann = SubmissionAnnotation(
            instance_id=task_data["instance_id"],
            selected_database=task_data["selected_database"],
            task_annotation_ref=(
                f"annotations/mini_interact/alien/"
                f"{task_data['instance_id']}.task.json"
            ),
            annotated_by="auto-inline-grader",
            annotated_at="2026-06-02T00:00:00+00:00",
            submission=SubmissionMetadata(
                cloud_run_id=run_id,
                trajectory_path=f"rows/{task_data['instance_id']}/attempt-1.json",
            ),
            evaluation=SubmissionEvaluation(
                phase1_against_original_gold="pass" if passed else "fail",
                phase1_against_audited_primary="pass" if passed else "fail",
                phase1_against_any_audited_variant="pass" if passed else "fail",
                phase1_against_variants=[],
                correct_up_to_tie_order=passed,
                novel_reading_judgment=None,
                correct_under_numeric_epsilon=passed,
                correct_under_trailing_whitespace=passed,
                correct_under_column_order=passed,
                correct_under_case_fold=passed,
                numeric_epsilon=1e-6,
                verdict="correct" if passed else "invalid",
                matched_variant_id="primary" if passed else None,
                rationale="",
                miss_diagnostics=None,
            ),
            failure_classification=FailureClassification(
                primary="no_fail" if passed else "agent_miss",
                agent_at_fault=not passed,
                remediation_target="other" if passed else "agent",
                details="stub",
            ),
            decision_point=None,
            user_sim_interaction=UserSimInteraction(
                n_asks=kw.get("n_ask_user_calls") or 0,
            ),
        )
        path = out_dir / "submission_annotation.json"
        path.write_text(ann.model_dump_json(indent=2, exclude_none=False) + "\n")
        return path

    monkeypatch.setattr(run_mod, "grade_one_submission", _stub_grader)

    output_path = tmp_path / "eval.json"
    metrics = await run_mod.run_evaluation(
        framework="claude_sdk_otf_ainteract",
        query_mode="slayer",
        mode="a-interact",
        data_path="ignored",
        data_dir=str(tmp_path / "ignored_data_dir"),
        output_path=str(output_path),
        concurrency=1,
        limit=None,
        agent_model="anthropic/claude-haiku-4-5-20251001",
        strict=False,
        prompt_cache=False,
        max_depth=1,
        slayer_storage_root=str(tmp_path / "slayer_models"),
        slayer_setup="on-the-fly",
        reasoning_effort=None,
        use_audited_gold_sql=False,
        dataset="mini-interact",
        gold_file=None,
        filter_ids=None,
    )

    # The grader fired for EVERY task — load-bearing assertion: the
    # local runner now invokes the inline grader per the
    # _run_with_sem _grade_local_row hook.
    instances = sorted(c["instance_id"] for c in calls)
    assert instances == ["alien_1", "alien_2"], (
        f"inline grader should be called once per task; got: {instances}"
    )
    # Benchmark token is the canonical underscore form.
    assert all(c["benchmark"] == "mini_interact" for c in calls), (
        f"benchmark must be canonicalized to underscore form; got: {calls}"
    )
    # Per-row ``submission_annotation.json`` files were written.
    rows_dir = output_path.parent / "rows"
    for inst in ("alien_1", "alien_2"):
        assert (rows_dir / inst / "submission_annotation.json").exists(), (
            f"submission_annotation.json missing for {inst}"
        )
    # And the aggregator emitted the ``cascading_phase1`` block into
    # eval.json — the whole point of this fix.
    final_eval = json.loads(output_path.read_text())
    assert "cascading_phase1" in final_eval, (
        f"cascading_phase1 missing from eval.json; keys={sorted(final_eval)}"
    )
    cp = final_eval["cascading_phase1"]
    assert cp["n_dual_eval_tasks"] == 2, f"got cp={cp}"
    # alien_1 passed every tier (n1..n9) in the stub annotation;
    # alien_2 failed everything. Each tier should therefore count 1.
    assert cp["counts"]["n3"] == 1, f"got cp={cp}"
    assert cp["rates"]["n3"] == 0.5, f"got cp={cp}"


@pytest.mark.asyncio
async def test_local_run_grader_failure_writes_fail_everything_annotation(
    monkeypatch, tmp_path,
):
    """If the inline grader raises on one instance, the loop MUST
    continue AND a fail-everything ``submission_annotation.json`` MUST
    be written for the broken instance so the aggregator's denominator
    stays at ``len(tasks)``. Pre-fix the broken instance was silently
    dropped, inflating ``cascading_phase1.rates`` (Codex round-4 finding
    on ``run.py:1007-1012``)."""
    import bird_interact_agents.run as run_mod

    rows = [
        {"instance_id": "alien_1", "selected_database": "alien",
         "sol_sql": ["SELECT 1"], "amb_user_query": "q1"},
        {"instance_id": "alien_2", "selected_database": "alien",
         "sol_sql": ["SELECT 2"], "amb_user_query": "q2"},
    ]
    _patch_loader_returns(monkeypatch, rows)
    monkeypatch.setattr(run_mod, "_maybe_force_wipe_otf", lambda **kw: None)
    _stub_runner_factory(monkeypatch, {
        "alien_1": {
            "instance_id": "alien_1", "database": "alien",
            "phase1_passed": True, "phase2_passed": False,
            "total_reward": 1.0, "submitted_sql": "SELECT 1",
            "trajectory": [], "usage": {},
        },
        "alien_2": {
            "instance_id": "alien_2", "database": "alien",
            "phase1_passed": False, "phase2_passed": False,
            "total_reward": 0.0, "submitted_sql": "SELECT 2",
            "trajectory": [], "usage": {},
        },
    })

    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification,
        SubmissionAnnotation,
        SubmissionEvaluation,
        SubmissionMetadata,
        UserSimInteraction,
    )

    def _stub_grader_with_explosion(*, task_data, **kw):
        if task_data["instance_id"] == "alien_1":
            raise RuntimeError("boom")
        # Second task writes a real annotation so the aggregator
        # has something parseable to read.
        rows_dir = Path(kw["rows_dir"])
        run_id = kw["run_id"]
        out_dir = rows_dir / task_data["instance_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        ann = SubmissionAnnotation(
            instance_id=task_data["instance_id"],
            selected_database=task_data["selected_database"],
            task_annotation_ref=(
                f"annotations/mini_interact/alien/"
                f"{task_data['instance_id']}.task.json"
            ),
            annotated_by="auto-inline-grader",
            annotated_at="2026-06-02T00:00:00+00:00",
            submission=SubmissionMetadata(
                cloud_run_id=run_id,
                trajectory_path=f"rows/{task_data['instance_id']}/attempt-1.json",
            ),
            evaluation=SubmissionEvaluation(
                phase1_against_original_gold="fail",
                phase1_against_audited_primary="fail",
                phase1_against_any_audited_variant="fail",
                phase1_against_variants=[],
                correct_up_to_tie_order=False,
                novel_reading_judgment=None,
                correct_under_numeric_epsilon=False,
                correct_under_trailing_whitespace=False,
                correct_under_column_order=False,
                correct_under_case_fold=False,
                numeric_epsilon=1e-6,
                verdict="invalid",
                matched_variant_id=None,
                rationale="",
                miss_diagnostics=None,
            ),
            failure_classification=FailureClassification(
                primary="agent_miss",
                agent_at_fault=True,
                remediation_target="agent",
                details="stub",
            ),
            decision_point=None,
            user_sim_interaction=UserSimInteraction(),
        )
        path = out_dir / "submission_annotation.json"
        path.write_text(ann.model_dump_json(indent=2, exclude_none=False) + "\n")
        return path

    monkeypatch.setattr(
        run_mod, "grade_one_submission", _stub_grader_with_explosion,
    )

    output_path = tmp_path / "eval.json"
    metrics = await run_mod.run_evaluation(
        framework="claude_sdk_otf_ainteract",
        query_mode="slayer",
        mode="a-interact",
        data_path="ignored",
        data_dir=str(tmp_path / "ignored_data_dir"),
        output_path=str(output_path),
        concurrency=1,
        limit=None,
        agent_model="anthropic/claude-haiku-4-5-20251001",
        strict=False,
        prompt_cache=False,
        max_depth=1,
        slayer_storage_root=str(tmp_path / "slayer_models"),
        slayer_setup="on-the-fly",
        reasoning_effort=None,
        use_audited_gold_sql=False,
        dataset="mini-interact",
        gold_file=None,
        filter_ids=None,
    )

    # Both tasks ran to completion despite alien_1 grader raising —
    # the loop kept going, total_tasks==2.
    assert metrics["total_tasks"] == 2
    # Both annotations are on disk: alien_2 from the stub success path,
    # alien_1 from the fail-everything fallback the runner wrote when
    # the stub raised. Without that fallback the aggregator's
    # denominator would have been 1, not 2.
    rows_dir = output_path.parent / "rows"
    for inst in ("alien_1", "alien_2"):
        assert (rows_dir / inst / "submission_annotation.json").exists(), (
            f"submission_annotation.json missing for {inst} after "
            f"grader failure — fail-everything fallback didn't fire"
        )
    cp = metrics["cascading_phase1"]
    assert cp["n_dual_eval_tasks"] == 2, (
        f"denominator should stay at 2 even when alien_1's grader "
        f"raised; got cp={cp}"
    )
    # Both alien_1 (fail-everything from the fallback) and alien_2
    # (the stub's deliberate fail-everything) count as 0 at every tier.
    for tier in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"):
        assert cp["counts"][tier] == 0, (
            f"every cascade tier should count 0 (both rows fail); "
            f"got cp={cp}"
        )
    # alien_1's annotation is the fail-everything shape: primary
    # ``other`` (the cascade was never actually run on alien_1).
    alien1 = json.loads(
        (rows_dir / "alien_1" / "submission_annotation.json").read_text(),
    )
    assert alien1["failure_classification"]["primary"] == "other"
    assert alien1["evaluation"]["verdict"] == "invalid"
    assert "grader raised" in alien1["failure_classification"]["details"]


@pytest.mark.asyncio
async def test_local_run_no_submitted_sql_writes_fail_everything_annotation(
    monkeypatch, tmp_path,
):
    """If the agent crashes before submit (no ``submitted_sql`` on the
    result row), the local runner MUST still write a fail-everything
    annotation so the cascade denominator stays honest. Pre-fix the
    no-submit row was silently dropped from
    ``cascading_phase1.n_dual_eval_tasks``."""
    import bird_interact_agents.run as run_mod

    rows = [
        {"instance_id": "alien_1", "selected_database": "alien",
         "sol_sql": ["SELECT 1"], "amb_user_query": "q1"},
        {"instance_id": "alien_2", "selected_database": "alien",
         "sol_sql": ["SELECT 2"], "amb_user_query": "q2"},
    ]
    _patch_loader_returns(monkeypatch, rows)
    monkeypatch.setattr(run_mod, "_maybe_force_wipe_otf", lambda **kw: None)
    _stub_runner_factory(monkeypatch, {
        # alien_1 returns a row WITHOUT submitted_sql — simulating an
        # agent crash before reaching the submit step.
        "alien_1": {
            "instance_id": "alien_1", "database": "alien",
            "phase1_passed": False, "phase2_passed": False,
            "total_reward": 0.0, "submitted_sql": None,
            "trajectory": [], "usage": {},
        },
        "alien_2": {
            "instance_id": "alien_2", "database": "alien",
            "phase1_passed": False, "phase2_passed": False,
            "total_reward": 0.0, "submitted_sql": "SELECT 2",
            "trajectory": [], "usage": {},
        },
    })

    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification,
        SubmissionAnnotation,
        SubmissionEvaluation,
        SubmissionMetadata,
        UserSimInteraction,
    )

    def _stub_grader_for_alien_2(*, task_data, **kw):
        # Only called for alien_2 — alien_1 is short-circuited by the
        # no-submitted_sql check before reaching the grader.
        out_dir = Path(kw["rows_dir"]) / task_data["instance_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        ann = SubmissionAnnotation(
            instance_id=task_data["instance_id"],
            selected_database=task_data["selected_database"],
            task_annotation_ref=(
                f"annotations/mini_interact/alien/"
                f"{task_data['instance_id']}.task.json"
            ),
            annotated_by="auto-inline-grader",
            annotated_at="2026-06-02T00:00:00+00:00",
            submission=SubmissionMetadata(
                cloud_run_id=kw["run_id"],
                trajectory_path=f"rows/{task_data['instance_id']}/attempt-1.json",
            ),
            evaluation=SubmissionEvaluation(
                phase1_against_original_gold="fail",
                phase1_against_audited_primary="fail",
                phase1_against_any_audited_variant="fail",
                phase1_against_variants=[],
                correct_up_to_tie_order=False,
                novel_reading_judgment=None,
                correct_under_numeric_epsilon=False,
                correct_under_trailing_whitespace=False,
                correct_under_column_order=False,
                correct_under_case_fold=False,
                numeric_epsilon=1e-6,
                verdict="invalid",
                matched_variant_id=None,
                rationale="",
                miss_diagnostics=None,
            ),
            failure_classification=FailureClassification(
                primary="agent_miss",
                agent_at_fault=True,
                remediation_target="agent",
                details="stub",
            ),
            decision_point=None,
            user_sim_interaction=UserSimInteraction(),
        )
        path = out_dir / "submission_annotation.json"
        path.write_text(ann.model_dump_json(indent=2, exclude_none=False) + "\n")
        return path

    monkeypatch.setattr(
        run_mod, "grade_one_submission", _stub_grader_for_alien_2,
    )

    output_path = tmp_path / "eval.json"
    metrics = await run_mod.run_evaluation(
        framework="claude_sdk_otf_ainteract",
        query_mode="slayer",
        mode="a-interact",
        data_path="ignored",
        data_dir=str(tmp_path / "ignored_data_dir"),
        output_path=str(output_path),
        concurrency=1,
        limit=None,
        agent_model="anthropic/claude-haiku-4-5-20251001",
        strict=False,
        prompt_cache=False,
        max_depth=1,
        slayer_storage_root=str(tmp_path / "slayer_models"),
        slayer_setup="on-the-fly",
        reasoning_effort=None,
        use_audited_gold_sql=False,
        dataset="mini-interact",
        gold_file=None,
        filter_ids=None,
    )

    rows_dir = output_path.parent / "rows"
    for inst in ("alien_1", "alien_2"):
        assert (rows_dir / inst / "submission_annotation.json").exists(), (
            f"submission_annotation.json missing for {inst} — "
            f"no-submitted_sql fallback didn't fire"
        )
    cp = metrics["cascading_phase1"]
    assert cp["n_dual_eval_tasks"] == 2, (
        f"denominator should stay at 2 even when alien_1 had no "
        f"submitted_sql; got cp={cp}"
    )
    alien1 = json.loads(
        (rows_dir / "alien_1" / "submission_annotation.json").read_text(),
    )
    assert alien1["failure_classification"]["primary"] == "other"
    assert "no submitted_sql" in alien1["failure_classification"]["details"]
