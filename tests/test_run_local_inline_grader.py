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

# DEV-1640: these tests pin the LOCAL in-process per-task wiring / grading by
# monkeypatching agents + graders + loaders, which a spawned worker process
# cannot see. The process pool is now the default, so route run_evaluation
# through the retained legacy single-loop path (identical per-task wiring).
@pytest.fixture(autouse=True)
def _dev1640_force_legacy_inprocess(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_LOCAL_INPROCESS", "1")


def _also_write_to_runs(ann, *, benchmark: str, run_id: str) -> None:
    """Write ``ann`` to the runs/ golden store (DEV-1533).

    Called from every grader stub that manually constructs a
    SubmissionAnnotation — the real ``grade_and_write`` / ``_write_to_runs``
    are bypassed when the stub patches ``grade_one_submission`` directly.
    Relies on ``main_checkout_root`` being patched to ``tmp_path`` by the
    enclosing test so the write goes to the isolated temp directory.
    """
    from bird_interact_agents.eval.annotation_io import (
        run_annotation_path, write_run_annotation,
    )
    dest = run_annotation_path(
        benchmark=benchmark,
        selected_database=ann.selected_database,
        instance_id=ann.instance_id,
        run_id=run_id,
    )
    write_run_annotation(ann, dest)


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
    import bird_interact_agents.paths as paths_mod
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

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
            "db_path": db_path,
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
                verdict="correct" if passed else "agent_miss",
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
        _also_write_to_runs(ann, benchmark=benchmark, run_id=run_id)
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
        filter_ids=None,
    )

    # The grader fired for EVERY task — load-bearing assertion: the
    # local runner now invokes the inline grader per the
    # _run_with_sem _grade_local_row hook.
    instances = sorted(c["instance_id"] for c in calls)
    assert instances == ["alien_1", "alien_2"], (
        f"inline grader should be called once per task; got: {instances}"
    )
    # Benchmark token is the canonical hyphenated form.
    assert all(c["benchmark"] == "mini-interact" for c in calls), (
        f"benchmark must be the canonical hyphenated form; got: {calls}"
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
    # Codex r7: ``db_path`` MUST be rooted at the caller-provided
    # ``data_dir`` (the same sqlite the agent ran against) — NOT
    # ``paths.benchmark_data_root``. Without this guard a local run
    # pointed at a tmp / alternate checkout / env-overridden data dir
    # would grade against the global sqlite, and the cascade verdict
    # could disagree with the agent's submission for purely path-routing
    # reasons.
    expected_data_dir = (tmp_path / "ignored_data_dir").resolve()
    for c in calls:
        db_path = Path(c["db_path"]).resolve()
        assert expected_data_dir in db_path.parents, (
            f"db_path must be rooted at data_dir={expected_data_dir!r}; "
            f"got db_path={c['db_path']!r}"
        )
        assert db_path.name == "alien.sqlite", (
            f"db_path leaf must be <db>.sqlite; got {c['db_path']!r}"
        )


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
    import bird_interact_agents.paths as paths_mod
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

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
        benchmark = kw["benchmark"]
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
                verdict="agent_miss",
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
        _also_write_to_runs(ann, benchmark=benchmark, run_id=run_id)
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
    assert alien1["evaluation"]["verdict"] == "eval_failed"
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
    import bird_interact_agents.paths as paths_mod
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

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
        run_id = kw["run_id"]
        benchmark = kw["benchmark"]
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
                verdict="agent_miss",
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
        _also_write_to_runs(ann, benchmark=benchmark, run_id=run_id)
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


@pytest.mark.asyncio
async def test_local_run_wipes_stale_rows_when_no_filter(monkeypatch, tmp_path):
    """Codex r9: reusing the same ``output_dir`` without ``filter_ids``
    MUST wipe ``rows/`` first, so the aggregator only sees the current
    run's annotations. Pre-fix a stale ``rows/old_iid/...`` from a prior
    pass would inflate ``cascading_phase1.n_dual_eval_tasks``."""
    import bird_interact_agents.paths as paths_mod
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    output_path = tmp_path / "eval.json"
    rows_dir = output_path.parent / "rows"
    # Plant a stale annotation from a prior run under a DIFFERENT iid
    # than the current task set will produce.
    stale_dir = rows_dir / "stale_old_iid"
    stale_dir.mkdir(parents=True)
    (stale_dir / "submission_annotation.json").write_text(
        '{"this": "should be wiped"}',
    )

    rows = [
        {"instance_id": "alien_1", "selected_database": "alien",
         "sol_sql": ["SELECT 1"], "amb_user_query": "q1"},
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
    })
    # No-op grader so cascading_phase1 is built off the (sole) row.
    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification, SubmissionAnnotation, SubmissionEvaluation,
        SubmissionMetadata, UserSimInteraction,
    )

    def _stub(*, task_data, **kw):
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
                verdict="agent_miss",
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
        _also_write_to_runs(ann, benchmark=kw["benchmark"], run_id=kw["run_id"])
        return path
    monkeypatch.setattr(run_mod, "grade_one_submission", _stub)

    metrics = await run_mod.run_evaluation(
        framework="claude_sdk_otf_ainteract", query_mode="slayer",
        mode="a-interact", data_path="ignored",
        data_dir=str(tmp_path / "ignored_data_dir"),
        output_path=str(output_path),
        concurrency=1, limit=None,
        agent_model="anthropic/claude-haiku-4-5-20251001",
        strict=False, prompt_cache=False, max_depth=1,
        slayer_storage_root=str(tmp_path / "slayer_models"),
        slayer_setup="on-the-fly", reasoning_effort=None,
        use_audited_gold_sql=False, dataset="mini-interact",
        filter_ids=None,
    )
    assert not stale_dir.exists(), (
        "rows/stale_old_iid should have been wiped before the run"
    )
    cp = metrics["cascading_phase1"]
    assert cp["n_dual_eval_tasks"] == 1, (
        f"denominator must reflect ONLY current tasks (1), not the "
        f"union with stale entries; got cp={cp}"
    )


@pytest.mark.asyncio
async def test_local_run_filter_ids_preserves_unrelated_rows(
    monkeypatch, tmp_path,
):
    """Symmetric case: filtered reruns must NOT wipe rows from a prior
    full run if those instances aren't in the current task set. Only
    the subdirs we're about to overwrite get reset."""
    import bird_interact_agents.paths as paths_mod
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    output_path = tmp_path / "eval.json"
    rows_dir = output_path.parent / "rows"
    # Unrelated prior-run annotation that the filtered rerun should
    # preserve verbatim.
    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification, SubmissionAnnotation, SubmissionEvaluation,
        SubmissionMetadata, UserSimInteraction,
    )
    unrelated = rows_dir / "alien_99"
    unrelated.mkdir(parents=True)

    def _make_ann(iid: str, marker: str) -> SubmissionAnnotation:
        return SubmissionAnnotation(
            instance_id=iid,
            selected_database="alien",
            task_annotation_ref=f"annotations/mini_interact/alien/{iid}.task.json",
            annotated_by=marker,
            annotated_at="2026-06-02T00:00:00+00:00",
            submission=SubmissionMetadata(
                cloud_run_id="r1",
                trajectory_path=f"rows/{iid}/attempt-1.json",
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
                verdict="agent_miss",
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
    (unrelated / "submission_annotation.json").write_text(
        _make_ann("alien_99", "unrelated-survivor")
        .model_dump_json(indent=2, exclude_none=False) + "\n",
    )

    rows = [
        {"instance_id": "alien_1", "selected_database": "alien",
         "sol_sql": ["SELECT 1"], "amb_user_query": "q1"},
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
    })

    def _stub(*, task_data, **kw):
        out_dir = Path(kw["rows_dir"]) / task_data["instance_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        ann = _make_ann(task_data["instance_id"], "fresh")
        (out_dir / "submission_annotation.json").write_text(
            ann.model_dump_json(indent=2, exclude_none=False) + "\n",
        )
        _also_write_to_runs(ann, benchmark=kw["benchmark"], run_id=kw["run_id"])
        return out_dir / "submission_annotation.json"
    monkeypatch.setattr(run_mod, "grade_one_submission", _stub)

    metrics = await run_mod.run_evaluation(
        framework="claude_sdk_otf_ainteract", query_mode="slayer",
        mode="a-interact", data_path="ignored",
        data_dir=str(tmp_path / "ignored_data_dir"),
        output_path=str(output_path),
        concurrency=1, limit=None,
        agent_model="anthropic/claude-haiku-4-5-20251001",
        strict=False, prompt_cache=False, max_depth=1,
        slayer_storage_root=str(tmp_path / "slayer_models"),
        slayer_setup="on-the-fly", reasoning_effort=None,
        use_audited_gold_sql=False, dataset="mini-interact",
        # Filter: only alien_1 — alien_99's prior annotation should survive.
        filter_ids=["alien_1"],
    )
    assert (unrelated / "submission_annotation.json").exists(), (
        "filtered rerun must NOT wipe rows for instances outside the "
        "filter set; alien_99's prior annotation was deleted"
    )
    surviving = json.loads(
        (unrelated / "submission_annotation.json").read_text(),
    )
    assert surviving["annotated_by"] == "unrelated-survivor"
    # Codex r11: published metrics MUST describe ONLY the current run's
    # row set, NOT the union with preserved prior rows. Otherwise
    # ``cascading_phase1.n_dual_eval_tasks`` could exceed
    # ``total_tasks`` and the rewritten ``phase1_count`` /
    # ``phase1_rate`` would become uninterpretable.
    cp = metrics["cascading_phase1"]
    assert metrics["total_tasks"] == 1, f"got {metrics['total_tasks']}"
    assert cp["n_dual_eval_tasks"] == 1, (
        f"cascade denominator must be scoped to the current run "
        f"(filter_ids=[alien_1]) — got {cp['n_dual_eval_tasks']}; "
        f"alien_99's preserved annotation must NOT pollute it"
    )
    assert metrics["total_tasks"] == cp["n_dual_eval_tasks"], (
        "total_tasks and cascading_phase1.n_dual_eval_tasks MUST agree"
    )


@pytest.mark.asyncio
async def test_autopsy_and_task_annotation_stripped_from_local_eval_json(
    monkeypatch, tmp_path,
):
    """C4: _autopsy and _task_annotation are internal pipeline keys consumed
    by the inline grader. They must NOT appear in the published eval.json
    results list. Pre-fix: run.py serialised them via json.dump(default=str),
    turning Pydantic objects into ugly stringified repr — polluting the
    public result schema."""
    import bird_interact_agents.run as run_mod

    rows = [
        {
            "instance_id": "alien_1",
            "selected_database": "alien",
            "sol_sql": ["SELECT 1"],
            "amb_user_query": "q1",
        },
    ]
    _patch_loader_returns(monkeypatch, rows)
    monkeypatch.setattr(run_mod, "_maybe_force_wipe_otf", lambda **kw: None)

    class _FakePydanticObj:
        """Stands in for a real Pydantic model — not JSON-serialisable natively."""
        def __str__(self):
            return "LEAKED_PYDANTIC_OBJECT"

    # Runner returns result rows WITH the internal pipeline keys.
    async def _stub_runner(td, data_dir, patience, user_sim_model):
        return {
            "instance_id": td["instance_id"],
            "database": td.get("selected_database", ""),
            "phase1_passed": True,
            "phase2_passed": False,
            "total_reward": 1.0,
            "submitted_sql": "SELECT 1",
            "trajectory": [],
            "usage": {},
            "_autopsy": _FakePydanticObj(),
            "_task_annotation": _FakePydanticObj(),
        }

    monkeypatch.setattr(run_mod, "_make_runner", lambda **kw: _stub_runner)

    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification, SubmissionAnnotation, SubmissionEvaluation,
        SubmissionMetadata, UserSimInteraction,
    )

    def _stub_grader(*, task_data, **kw):
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
                phase1_against_original_gold="pass",
                phase1_against_audited_primary="pass",
                phase1_against_any_audited_variant="pass",
                phase1_against_variants=[],
                correct_up_to_tie_order=True,
                novel_reading_judgment=None,
                correct_under_numeric_epsilon=True,
                correct_under_trailing_whitespace=True,
                correct_under_column_order=True,
                correct_under_case_fold=True,
                numeric_epsilon=1e-6,
                verdict="correct",
                matched_variant_id="primary",
                rationale="",
                miss_diagnostics=None,
            ),
            failure_classification=FailureClassification(
                primary="no_fail",
                agent_at_fault=False,
                remediation_target="other",
                details="stub",
            ),
            decision_point=None,
            user_sim_interaction=UserSimInteraction(),
        )
        path = out_dir / "submission_annotation.json"
        path.write_text(ann.model_dump_json(indent=2, exclude_none=False) + "\n")
        return path

    monkeypatch.setattr(run_mod, "grade_one_submission", _stub_grader)

    output_path = tmp_path / "eval.json"
    await run_mod.run_evaluation(
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
        filter_ids=None,
    )

    final_eval = json.loads(output_path.read_text())
    for r in final_eval.get("results", []):
        assert "_autopsy" not in r, (
            f"_autopsy leaked into eval.json result row for "
            f"{r.get('instance_id')}: {r.get('_autopsy')!r}"
        )
        assert "_task_annotation" not in r, (
            f"_task_annotation leaked into eval.json result row for "
            f"{r.get('instance_id')}: {r.get('_task_annotation')!r}"
        )


@pytest.mark.asyncio
async def test_local_cascade_emission_failure_surfaces_on_metrics(
    monkeypatch, tmp_path, caplog,
):
    """Codex (DEV-1533 third pass): when ``emit_cascading_eval_json``
    raises during local-run finalization, the failure MUST land on the
    metrics dict as ``cascading_phase1_error`` AND a warning log line so
    the operator sees that the PR's primary metric block is missing.
    Pre-fix the bare ``except: pass`` silently dropped the failure."""
    import logging
    import bird_interact_agents.paths as paths_mod
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    rows = [
        {"instance_id": "alien_1", "selected_database": "alien",
         "sol_sql": ["SELECT 1"], "amb_user_query": "q1"},
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
            "usage": {"n_agent_turns": 1, "n_ask_user_calls": 0},
        },
    })

    # Stub grader is a no-op — the test only cares about the post-loop
    # emit call.
    def _stub_grader(*, task_data, **kw):
        out_dir = Path(kw["rows_dir"]) / task_data["instance_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "submission_annotation.json"
        path.write_text("{}")
        return path

    monkeypatch.setattr(run_mod, "grade_one_submission", _stub_grader)

    # Force the cascade emission to blow up.
    def _boom(*args, **kwargs):
        raise RuntimeError("boom: simulated aggregation failure")

    monkeypatch.setattr(run_mod, "emit_cascading_eval_json", _boom)

    output_path = tmp_path / "eval.json"
    with caplog.at_level(logging.WARNING, logger="bird_interact_agents.run"):
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
            filter_ids=None,
        )

    assert metrics.get("cascading_phase1_error", "").startswith("boom"), (
        f"cascading_phase1_error must be populated when the emission "
        f"raises; got metrics keys={sorted(metrics)}"
    )
    assert any(
        "cascading_phase1 aggregation failed" in rec.message
        for rec in caplog.records
    ), f"expected warning log; got {[r.message for r in caplog.records]}"
