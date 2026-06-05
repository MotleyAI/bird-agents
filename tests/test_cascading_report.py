"""DEV-1515/1533: cascading-report aggregator + eval.json writer.

Pins:
* eval.json carries a single ``cascading_phase1`` block with N1..N9
  counts, rates, deltas, and n_dual_eval_tasks.
* ``phase1_count`` / ``phase1_rate`` stay (basic back-compat) but map
  to N1 of the cascade.
* The legacy dual-eval block is removed.
* Aggregator reads from ``runs/<benchmark>/<db>/<inst>/<run_id>.json``
  (DEV-1533 golden store) via BIRD_RUNS_ROOT override in tests.
* Each task maps to exactly one partition tier in ``cascading_partition``.
"""
from __future__ import annotations

import json

import pytest


_BENCHMARK = "mini-interact"
_RUN_ID = "test-run-001"
_DB = "alien"


def _make_annotation_json(
    *,
    instance_id: str,
    selected_database: str,
    n1: bool, n2: bool, n3: bool, n4: bool, n5: bool,
    n6: bool, n7: bool, n8: bool, n9: bool = False,
    verdict: str = "correct",
    original_gold_annotated_correct: bool = True,
    rationale: str = "",
    annotated_at: str = "2026-06-01T10:00:00+00:00",
) -> dict:
    """Build the JSON shape produced by tolerant_grader → SubmissionAnnotation."""
    return {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": instance_id,
        "selected_database": selected_database,
        "task_annotation_ref": (
            f"annotations/{_BENCHMARK}/{selected_database}/"
            f"{instance_id}.task.json"
        ),
        "annotated_by": "auto",
        "annotated_at": annotated_at,
        "submission": {
            "cloud_run_id": _RUN_ID,
            "trajectory_path": f"rows/{instance_id}/attempt-1.json",
            "submitted_sql_path": None,
            "predicted_row_count": 1,
            "duration_s": 1.0,
            "cost_usd_agent": 0.0,
            "cost_usd_user_sim": 0.0,
            "n_agent_turns": 1,
            "n_ask_user_calls": 0,
        },
        "evaluation": {
            "phase1_against_original_gold": "pass" if n1 else "fail",
            "phase1_against_audited_primary": "pass" if n2 else "fail",
            "phase1_against_any_audited_variant": "pass" if n3 else "fail",
            "phase1_against_variants": [],
            "correct_up_to_tie_order": n4,
            "novel_reading_judgment": "pass" if n5 and not n4 else None,
            "correct_under_numeric_epsilon": n6,
            "correct_under_trailing_whitespace": n7,
            "correct_under_column_order": n8,
            "correct_under_case_fold": n9,
            "numeric_epsilon": 1e-6,
            "verdict": verdict,
            "matched_variant_id": "primary" if n3 else None,
            "rationale": rationale,
        },
        "failure_classification": {
            "primary": "other",
            "secondary": [],
            "agent_at_fault": False,
            "remediation_target": "other",
            "remediation_text": "",
            "details": "",
        },
        "decision_point": None,
        "user_sim_interaction": {
            "n_asks": 0, "key_responses": [],
            "disclosed_resolutions": [], "undisclosed_resolutions": [],
        },
        "original_gold_annotated_correct": original_gold_annotated_correct,
    }


def _write_run_annotation(tmp_path, annotation: dict, run_id: str = _RUN_ID) -> None:
    """Write annotation to runs/<benchmark>/<db>/<inst>/<run_id>.json."""
    db = annotation["selected_database"]
    iid = annotation["instance_id"]
    dest = tmp_path / _BENCHMARK / db / iid / f"{run_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(annotation))


# ---------------------------------------------------------------------------
# Aggregator — builds cascading_phase1 block from runs/ annotations
# ---------------------------------------------------------------------------


def test_aggregator_emits_cascading_phase1_block(monkeypatch, tmp_path):
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    annotations = [
        _make_annotation_json(
            instance_id="alien_1", selected_database=_DB,
            n1=True, n2=True, n3=True, n4=True, n5=True,
            n6=True, n7=True, n8=True,
        ),
        _make_annotation_json(
            instance_id="alien_2", selected_database=_DB,
            n1=False, n2=False, n3=False, n4=True, n5=True,
            n6=True, n7=True, n8=True,
        ),
        _make_annotation_json(
            instance_id="alien_3", selected_database=_DB,
            n1=False, n2=False, n3=False, n4=False, n5=False,
            n6=False, n7=False, n8=False,
        ),
    ]
    for ann in annotations:
        _write_run_annotation(tmp_path / "runs", ann)

    block = aggregate_cascading_phase1(_BENCHMARK, _RUN_ID)
    assert block["n_dual_eval_tasks"] == 3
    counts = block["counts"]
    assert counts["n1"] == 1
    assert counts["n2"] == 1
    assert counts["n3"] == 1
    assert counts["n4"] == 2
    assert counts["n5"] == 2
    assert counts["n6"] == 2
    assert counts["n7"] == 2
    assert counts["n8"] == 2

    rates = block["rates"]
    assert rates["n1"] == pytest.approx(1 / 3)
    assert rates["n8"] == pytest.approx(2 / 3)

    deltas = block["deltas"]
    assert deltas["n2"] == 0
    assert deltas["n3"] == 0
    assert deltas["n4"] == 1  # alien_2 added at N4


def test_aggregator_instance_filter_scopes_to_current_run(monkeypatch, tmp_path):
    """Filtered local reruns: instance_filter scopes the cascade to only
    the current run's instances."""
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    annotations = [
        _make_annotation_json(
            instance_id="alien_1", selected_database=_DB,
            n1=True, n2=True, n3=True, n4=True, n5=True,
            n6=True, n7=True, n8=True,
        ),
        _make_annotation_json(
            instance_id="alien_2_stale", selected_database=_DB,
            n1=True, n2=True, n3=True, n4=True, n5=True,
            n6=True, n7=True, n8=True,
        ),
        _make_annotation_json(
            instance_id="alien_3", selected_database=_DB,
            n1=False, n2=False, n3=False, n4=False, n5=False,
            n6=False, n7=False, n8=False,
        ),
    ]
    for ann in annotations:
        _write_run_annotation(tmp_path / "runs", ann)

    block_all = aggregate_cascading_phase1(_BENCHMARK, _RUN_ID)
    assert block_all["n_dual_eval_tasks"] == 3
    assert block_all["counts"]["n1"] == 2

    block_filtered = aggregate_cascading_phase1(
        _BENCHMARK, _RUN_ID,
        instance_filter={"alien_1", "alien_3"},
    )
    assert block_filtered["n_dual_eval_tasks"] == 2
    assert block_filtered["counts"]["n1"] == 1


def test_aggregator_enforces_monotonicity_on_tampered_row(monkeypatch, tmp_path):
    """Aggregator must enforce monotone: a row with N5=True but N6=False
    is repaired so N6+ also become True."""
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    ann = _make_annotation_json(
        instance_id="x_1", selected_database="x",
        n1=False, n2=False, n3=False, n4=False, n5=True,
        n6=False,  # VIOLATES monotone
        n7=False, n8=False,
    )
    _write_run_annotation(tmp_path / "runs", ann)

    block = aggregate_cascading_phase1(_BENCHMARK, _RUN_ID)
    counts = block["counts"]
    assert counts["n5"] == 1
    assert counts["n6"] == 1
    assert counts["n7"] == 1
    assert counts["n8"] == 1


def test_aggregator_surfaces_n9_case_fold(monkeypatch, tmp_path):
    """N9 case-fold must be counted; was previously dropped."""
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    ann = _make_annotation_json(
        instance_id="alien_1", selected_database=_DB,
        n1=False, n2=False, n3=False, n4=False, n5=False,
        n6=False, n7=False, n8=False, n9=True,
    )
    _write_run_annotation(tmp_path / "runs", ann)

    block = aggregate_cascading_phase1(_BENCHMARK, _RUN_ID)
    assert block["counts"]["n9"] == 1
    assert block["rates"]["n9"] == pytest.approx(1.0)
    assert "n9" in block["deltas"]
    for k in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8"):
        assert block["counts"][k] == 0


def test_aggregator_empty_runs_returns_zero_counts(monkeypatch, tmp_path):
    """When no run annotations exist for the run_id, aggregator returns zeros."""
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    block = aggregate_cascading_phase1(_BENCHMARK, _RUN_ID)
    assert block["n_dual_eval_tasks"] == 0
    assert all(v == 0 for v in block["counts"].values())


# ---------------------------------------------------------------------------
# eval.json shape — cascading_phase1 present; legacy block removed
# ---------------------------------------------------------------------------


def test_eval_json_contains_cascading_phase1_after_run(monkeypatch, tmp_path):
    """End-to-end: write runs, call emit_cascading_eval_json, check eval.json."""
    from bird_interact_agents.eval.cascading_report import emit_cascading_eval_json
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    for inst, n1 in (("pass_1", True), ("fail_1", False)):
        ann = _make_annotation_json(
            instance_id=inst, selected_database="x",
            n1=n1, n2=n1, n3=n1, n4=n1, n5=n1,
            n6=n1, n7=n1, n8=n1,
            verdict="correct" if n1 else "agent_miss",
        )
        _write_run_annotation(tmp_path / "runs", ann)

    out = tmp_path / "eval.json"
    emit_cascading_eval_json(
        _BENCHMARK, _RUN_ID, out,
        base_metrics={"phase1_count": 999, "phase1_rate": 0.42},
    )
    metrics = json.loads(out.read_text())
    assert "cascading_phase1" in metrics
    assert metrics["cascading_phase1"]["counts"]["n1"] == 1
    assert metrics["phase1_count"] == 1
    assert metrics["phase1_count"] == metrics["cascading_phase1"]["counts"]["n1"]
    assert metrics["phase1_rate"] == pytest.approx(0.5)
    for k in (
        "phase1_count_audited", "phase1_count_original",
        "phase1_rate_audited", "phase1_rate_original",
        "n_dual_eval_tasks",
    ):
        assert k not in metrics


# ---------------------------------------------------------------------------
# Partition tests (DEV-1533)
# ---------------------------------------------------------------------------


def test_partition_l1_correct_original(monkeypatch, tmp_path):
    """N1=True + original_gold_annotated_correct=True → L1."""
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    ann = _make_annotation_json(
        instance_id="alien_1", selected_database=_DB,
        n1=True, n2=True, n3=True, n4=True, n5=True,
        n6=True, n7=True, n8=True,
        original_gold_annotated_correct=True,
    )
    _write_run_annotation(tmp_path / "runs", ann)

    block = aggregate_cascading_phase1(_BENCHMARK, _RUN_ID)
    p = block["cascading_partition"]
    assert p["tiers"]["l1_correct_original"]["count"] == 1
    assert p["tiers"]["l2_wrong_original"]["count"] == 0
    assert p["pass_count"] == 1


def test_partition_l2_wrong_original_not_counted_as_pass(monkeypatch, tmp_path):
    """N1=True + original_gold_annotated_correct=False + N2=False → L2 (diagnostic)."""
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    # N1=True means agent matched original (which is wrong);
    # N2/N3 are False because original != audited and agent didn't match audited.
    ann = _make_annotation_json(
        instance_id="alien_1", selected_database=_DB,
        n1=True, n2=False, n3=False, n4=False, n5=False,
        n6=False, n7=False, n8=False,
        original_gold_annotated_correct=False,
        verdict="agent_miss",
    )
    _write_run_annotation(tmp_path / "runs", ann)

    block = aggregate_cascading_phase1(_BENCHMARK, _RUN_ID)
    p = block["cascading_partition"]
    assert p["tiers"]["l2_wrong_original"]["count"] == 1
    assert p["tiers"]["l1_correct_original"]["count"] == 0
    assert p["pass_count"] == 0, "L2 must NOT count as a pass"


def test_partition_l3_audited_primary(monkeypatch, tmp_path):
    """N2=True + N1=False → L3 (matched audited primary, not original)."""
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    ann = _make_annotation_json(
        instance_id="alien_1", selected_database=_DB,
        n1=False, n2=True, n3=True, n4=True, n5=True,
        n6=True, n7=True, n8=True,
        original_gold_annotated_correct=False,
        verdict="correct",
    )
    _write_run_annotation(tmp_path / "runs", ann)

    block = aggregate_cascading_phase1(_BENCHMARK, _RUN_ID)
    p = block["cascading_partition"]
    assert p["tiers"]["l3_audited_primary"]["count"] == 1
    assert p["tiers"]["l1_correct_original"]["count"] == 0
    assert p["pass_count"] == 1


def test_partition_l11_fail(monkeypatch, tmp_path):
    """All-fail annotation → L11."""
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    ann = _make_annotation_json(
        instance_id="alien_1", selected_database=_DB,
        n1=False, n2=False, n3=False, n4=False, n5=False,
        n6=False, n7=False, n8=False,
        verdict="agent_miss",
    )
    _write_run_annotation(tmp_path / "runs", ann)

    block = aggregate_cascading_phase1(_BENCHMARK, _RUN_ID)
    p = block["cascading_partition"]
    assert p["tiers"]["l11_fail"]["count"] == 1
    assert p["pass_count"] == 0


def test_partition_cumsum_sums_to_n(monkeypatch, tmp_path):
    """The final cumsum of partition tiers must equal n_tasks."""
    from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1
    import bird_interact_agents.paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)

    annotations = [
        _make_annotation_json(
            instance_id="alien_1", selected_database=_DB,
            n1=True, n2=True, n3=True, n4=True, n5=True,
            n6=True, n7=True, n8=True,
        ),
        _make_annotation_json(
            instance_id="alien_2", selected_database=_DB,
            n1=True, n2=False, n3=False, n4=False, n5=False,
            n6=False, n7=False, n8=False,
            original_gold_annotated_correct=False,
            verdict="agent_miss",
        ),
        _make_annotation_json(
            instance_id="alien_3", selected_database=_DB,
            n1=False, n2=False, n3=False, n4=False, n5=False,
            n6=False, n7=False, n8=False,
            verdict="agent_miss",
        ),
    ]
    for ann in annotations:
        _write_run_annotation(tmp_path / "runs", ann)

    block = aggregate_cascading_phase1(_BENCHMARK, _RUN_ID)
    p = block["cascading_partition"]
    total = sum(t["count"] for t in p["tiers"].values())
    assert total == 3, "Partition tiers must sum to n_tasks"
    # The last tier's cumsum must equal n_tasks
    last_tier = p["tiers"]["l11_fail"]
    assert last_tier["cumsum"] == 3
