"""DEV-1515: shared inline grader that both the cloud worker
(``ray_app.py``) and the local runner (``run.py``) invoke per task.

The function ``grade_and_write`` runs ``tolerant_grader.grade_submission``
and persists the resulting ``SubmissionAnnotation`` to
``<rows_dir>/<instance_id>/submission_annotation.json``. The cloud
``fetch`` path later merges that file into
``<main_checkout>/annotations/<benchmark>/<db>/<instance>.submission.<run-id>.json``.

The pre-DEV-1515 raw per-gold pass-fail bools are NOT emitted anywhere
— all per-task verdicts live in the SubmissionAnnotation's ``evaluation``
block.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Callable, List, Optional

from bird_interact_agents.eval.annotation_schema import (
    FailureClassification,
    PhaseVerdict,
    SubmissionAnnotation,
    SubmissionEvaluation,
    SubmissionMetadata,
    TaskAnnotation,
    UserSimInteraction,
)
from bird_interact_agents.eval.tolerant_grader import (
    CascadeVerdict,
    grade_submission,
)


_AUTO_ANNOTATOR = "auto-inline-grader"


def _verdict_to_phase(b: bool) -> PhaseVerdict:
    return "pass" if b else "fail"


def _build_submission_annotation(
    *,
    task_annotation: TaskAnnotation,
    cascade: CascadeVerdict,
    benchmark: str,
    run_id: str,
    trajectory_path: str,
    predicted_row_count: Optional[int],
    duration_s: Optional[float],
    cost_usd_agent: Optional[float],
    cost_usd_user_sim: Optional[float],
    n_agent_turns: Optional[int],
    n_ask_user_calls: Optional[int],
    user_sim_interaction: Optional[UserSimInteraction] = None,
    epsilon: float = 1e-6,
) -> SubmissionAnnotation:
    """Map the in-memory CascadeVerdict → on-disk SubmissionAnnotation."""
    if cascade.n3_any_audited_variant:
        verdict_label = "correct"
    elif cascade.n5_llm_judge or cascade.n4_tie_order:
        verdict_label = "valid_interpretation"
    elif (
        cascade.n6_numeric_epsilon
        or cascade.n7_trailing_whitespace
        or cascade.n8_column_order
    ):
        verdict_label = "valid_interpretation"
    else:
        verdict_label = "invalid"

    ev = SubmissionEvaluation(
        phase1_against_original_gold=_verdict_to_phase(cascade.n1_original_gold),
        phase1_against_audited_primary=_verdict_to_phase(cascade.n2_audited_primary),
        phase1_against_any_audited_variant=_verdict_to_phase(
            cascade.n3_any_audited_variant
        ),
        phase1_against_variants=list(cascade.variant_matches),
        correct_up_to_tie_order=cascade.n4_tie_order,
        novel_reading_judgment=cascade.novel_reading_judgment,
        correct_under_numeric_epsilon=cascade.n6_numeric_epsilon,
        correct_under_trailing_whitespace=cascade.n7_trailing_whitespace,
        correct_under_column_order=cascade.n8_column_order,
        numeric_epsilon=epsilon,
        verdict=verdict_label,  # type: ignore[arg-type]
        matched_variant_id=cascade.matched_variant_id,
        rationale="",
    )

    task_ann_ref = (
        f"annotations/{benchmark}/"
        f"{task_annotation.selected_database}/"
        f"{task_annotation.instance_id}.task.json"
    )
    return SubmissionAnnotation(
        instance_id=task_annotation.instance_id,
        selected_database=task_annotation.selected_database,
        task_annotation_ref=task_ann_ref,
        annotated_by=_AUTO_ANNOTATOR,
        annotated_at=_dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0).isoformat(),
        submission=SubmissionMetadata(
            cloud_run_id=run_id,
            trajectory_path=trajectory_path,
            predicted_row_count=predicted_row_count,
            duration_s=duration_s,
            cost_usd_agent=cost_usd_agent,
            cost_usd_user_sim=cost_usd_user_sim,
            n_agent_turns=n_agent_turns,
            n_ask_user_calls=n_ask_user_calls,
        ),
        evaluation=ev,
        failure_classification=FailureClassification(
            primary="other",
            agent_at_fault=not cascade.n3_any_audited_variant,
            remediation_target="other",
            details="auto-generated; human review pending",
        ),
        decision_point=None,
        user_sim_interaction=(
            user_sim_interaction or UserSimInteraction()
        ),
    )


def grade_and_write(
    *,
    rows_dir: Path,
    instance_id: str,
    benchmark: str,
    run_id: str,
    task_annotation: TaskAnnotation,
    audited_gold_rows: List[dict],
    original_sol_sql: List[str],
    submitted_sql: str,
    db_path: Path,
    conn: Any = None,
    executor: Optional[Callable[..., Any]] = None,
    trajectory_path: str,
    cost_usd_agent: Optional[float] = None,
    cost_usd_user_sim: Optional[float] = None,
    duration_s: Optional[float] = None,
    n_agent_turns: Optional[int] = None,
    n_ask_user_calls: Optional[int] = None,
    predicted_row_count: Optional[int] = None,
    user_sim_interaction: Optional[UserSimInteraction] = None,
    llm_judge: Any = None,
    epsilon: float = 1e-6,
) -> Path:
    """Run the tolerant grader and write the SubmissionAnnotation to
    ``<rows_dir>/<instance_id>/submission_annotation.json``."""
    cascade = grade_submission(
        task_annotation=task_annotation,
        audited_gold_rows=audited_gold_rows,
        original_sol_sql=original_sol_sql,
        submitted_sql=submitted_sql,
        db_path=db_path,
        conn=conn,
        executor=executor,
        llm_judge=llm_judge,
        epsilon=epsilon,
    )
    ann = _build_submission_annotation(
        task_annotation=task_annotation,
        cascade=cascade,
        benchmark=benchmark,
        run_id=run_id,
        trajectory_path=trajectory_path,
        predicted_row_count=predicted_row_count,
        duration_s=duration_s,
        cost_usd_agent=cost_usd_agent,
        cost_usd_user_sim=cost_usd_user_sim,
        n_agent_turns=n_agent_turns,
        n_ask_user_calls=n_ask_user_calls,
        user_sim_interaction=user_sim_interaction,
        epsilon=epsilon,
    )
    out_dir = Path(rows_dir) / instance_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission_annotation.json"
    out_path.write_text(ann.model_dump_json(indent=2, exclude_none=False) + "\n")
    return out_path
