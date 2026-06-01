"""DEV-1515: cascading-report aggregator + legacy-field removal.

Pins:
* eval.json carries a single ``cascading_phase1`` block with N1..N8
  counts, rates, deltas, and n_dual_eval_tasks.
* ``phase1_count`` / ``phase1_rate`` stay (basic back-compat) but map
  to N1 of the cascade.
* The legacy dual-eval block (``n_dual_eval_tasks`` standalone,
  ``phase1_rate_audited``, ``phase1_rate_original`` etc.) is removed.
* Aggregator reads per-row ``submission_annotation.json`` files; if
  any row is missing the file, the aggregator raises (no silent
  under-counts).
"""
from __future__ import annotations

import json

import pytest


def _make_submission_annotation_json(
    *,
    instance_id: str,
    selected_database: str,
    n1: bool, n2: bool, n3: bool, n4: bool, n5: bool,
    n6: bool, n7: bool, n8: bool, n9: bool = False,
    verdict: str = "correct",
) -> dict:
    """Build the JSON shape produced by tolerant_grader → SubmissionAnnotation."""
    return {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": instance_id,
        "selected_database": selected_database,
        "task_annotation_ref": (
            f"annotations/mini-interact/{selected_database}/"
            f"{instance_id}.task.json"
        ),
        "annotated_by": "auto",
        "annotated_at": "2026-05-31",
        "submission": {
            "cloud_run_id": "test-run",
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
            "rationale": "",
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
    }


# ---------------------------------------------------------------------------
# Aggregator — builds cascading_phase1 block from per-row SubmissionAnnotation
# ---------------------------------------------------------------------------


def test_aggregator_emits_cascading_phase1_block(tmp_path):
    from bird_interact_agents.eval.cascading_report import (
        aggregate_cascading_phase1,
    )

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    # 3 rows: full pass, only-N4, all-fail.
    annotations = [
        _make_submission_annotation_json(
            instance_id="alien_1", selected_database="alien",
            n1=True, n2=True, n3=True, n4=True, n5=True,
            n6=True, n7=True, n8=True,
        ),
        _make_submission_annotation_json(
            instance_id="alien_2", selected_database="alien",
            n1=False, n2=False, n3=False, n4=True, n5=True,
            n6=True, n7=True, n8=True,
        ),
        _make_submission_annotation_json(
            instance_id="alien_3", selected_database="alien",
            n1=False, n2=False, n3=False, n4=False, n5=False,
            n6=False, n7=False, n8=False,
        ),
    ]
    for ann in annotations:
        d = rows_dir / ann["instance_id"]
        d.mkdir()
        (d / "submission_annotation.json").write_text(json.dumps(ann))

    block = aggregate_cascading_phase1(rows_dir)
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


def test_aggregator_enforces_monotonicity_on_tampered_row(tmp_path):
    """The aggregator MUST enforce monotonicity. We deliberately feed a
    violating row (N5=True, N6=False) — a "later level is more strict
    than earlier" pattern that breaks the cascade. The aggregator must
    repair this so the published `cascading_phase1` counts respect
    N1 ≤ N2 ≤ ... ≤ N8.
    """
    from bird_interact_agents.eval.cascading_report import (
        aggregate_cascading_phase1,
    )

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    ann = _make_submission_annotation_json(
        instance_id="x_1", selected_database="x",
        n1=False, n2=False, n3=False, n4=False, n5=True,
        n6=False,  # VIOLATES monotone: passes at N5 must also pass at N6.
        n7=False, n8=False,
    )
    d = rows_dir / "x_1"
    d.mkdir()
    (d / "submission_annotation.json").write_text(json.dumps(ann))

    block = aggregate_cascading_phase1(rows_dir)
    counts = block["counts"]
    # After enforcement, every level from N5 onward inherits the pass.
    assert counts["n5"] == 1
    assert counts["n6"] == 1, (
        "monotone enforcement: N5 pass must propagate to N6"
    )
    assert counts["n7"] == 1
    assert counts["n8"] == 1


def test_aggregator_surfaces_n9_case_fold(tmp_path):
    """Regression for DEV-1515 follow-up: N9 was added to
    ``tolerant_grader._CASCADE_ORDER`` but the aggregator's
    ``_per_row_cascade_bools`` was hardcoded to N1..N8, leaving
    ``counts['n9']`` stuck at 0 (and no ``deltas['n9']``) even when
    the per-row annotation reported a case-fold-only pass."""
    from bird_interact_agents.eval.cascading_report import (
        aggregate_cascading_phase1,
    )

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    # One row: case-fold-only pass — every earlier level fails.
    ann = _make_submission_annotation_json(
        instance_id="alien_1", selected_database="alien",
        n1=False, n2=False, n3=False, n4=False, n5=False,
        n6=False, n7=False, n8=False, n9=True,
    )
    d = rows_dir / "alien_1"
    d.mkdir()
    (d / "submission_annotation.json").write_text(json.dumps(ann))

    block = aggregate_cascading_phase1(rows_dir)
    assert block["counts"]["n9"] == 1, (
        "n9_case_fold must increment counts['n9'] — it was previously "
        "dropped because _per_row_cascade_bools hardcoded n1..n8 only"
    )
    assert block["rates"]["n9"] == pytest.approx(1.0)
    assert "n9" in block["deltas"], (
        "deltas must extend to n9, not stop at n8"
    )
    # Monotone enforcement walks N1→N9, so a case-fold-only pass leaves
    # every stricter level (N1..N8) at 0 and only N9 at 1. The point of
    # this regression test is that N9 is SURFACED at all — pre-fix it
    # was stuck at 0 because the aggregator hardcoded N1..N8.
    for k in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8"):
        assert block["counts"][k] == 0, (
            f"only n9 was True in the raw row; {k} must remain 0"
        )


def test_aggregator_raises_when_row_missing_submission_annotation(tmp_path):
    """If any per-row dir is missing submission_annotation.json, the
    aggregator must raise — silent under-count is forbidden."""
    from bird_interact_agents.eval.cascading_report import (
        aggregate_cascading_phase1,
    )

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    (rows_dir / "alien_1").mkdir()  # empty — no submission_annotation.json

    with pytest.raises(FileNotFoundError):
        aggregate_cascading_phase1(rows_dir)


# ---------------------------------------------------------------------------
# eval.json shape — `cascading_phase1` present; legacy block removed
# ---------------------------------------------------------------------------


def test_eval_json_contains_cascading_phase1_after_run(tmp_path):
    """End-to-end: a synthetic local run emits eval.json with the new
    block AND has dropped the legacy dual-eval keys."""
    from bird_interact_agents.eval.cascading_report import (
        emit_cascading_eval_json,
    )

    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    # 2 instances: one passes at N1, one fails everything.
    for inst, n1 in (("pass_1", True), ("fail_1", False)):
        ann = _make_submission_annotation_json(
            instance_id=inst, selected_database="x",
            n1=n1, n2=n1, n3=n1, n4=n1, n5=n1,
            n6=n1, n7=n1, n8=n1,
        )
        d = rows_dir / inst
        d.mkdir()
        (d / "submission_annotation.json").write_text(json.dumps(ann))

    # Deliberately wrong base metrics — emit_cascading_eval_json must
    # REWRITE phase1_count/phase1_rate from the freshly-computed N1,
    # not blindly carry the stale base value forward.
    out = tmp_path / "eval.json"
    emit_cascading_eval_json(
        rows_dir, out,
        base_metrics={"phase1_count": 999, "phase1_rate": 0.42},
    )
    metrics = json.loads(out.read_text())
    assert "cascading_phase1" in metrics
    assert metrics["cascading_phase1"]["counts"]["n1"] == 1
    # phase1_count is REWRITTEN from cascade N1 (back-compat alias);
    # it must match counts["n1"], NOT the stale base value.
    assert metrics["phase1_count"] == 1
    assert metrics["phase1_count"] == metrics["cascading_phase1"]["counts"]["n1"]
    # And phase1_rate is rewritten too: 1/2 = 0.5.
    assert metrics["phase1_rate"] == pytest.approx(0.5)
    # Legacy dual-eval keys MUST be absent.
    for k in (
        "phase1_count_audited", "phase1_count_original",
        "phase1_rate_audited", "phase1_rate_original",
        "n_dual_eval_tasks",   # moved INTO cascading_phase1
    ):
        assert k not in metrics, (
            f"legacy key {k} must be removed from eval.json top level"
        )
