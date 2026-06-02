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
import json
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
from bird_interact_agents.eval.implicit_annotation import (
    implicit_task_annotation,
)
from bird_interact_agents.eval.tolerant_grader import (
    CascadeVerdict,
    grade_submission,
)


_AUTO_ANNOTATOR = "auto-inline-grader"


def _verdict_to_phase(b: bool) -> PhaseVerdict:
    return "pass" if b else "fail"


def _failure_details_for(
    cascade: CascadeVerdict, primary: str,
) -> str:
    """Return a one-line free-form human summary for
    ``FailureClassification.details``. Downstream consumers MUST go
    through ``cascade.miss_diagnostics.miss_patterns`` for structured
    signals; this string is for humans only."""
    if primary == "agent_miss" and cascade.miss_diagnostics is not None:
        md = cascade.miss_diagnostics
        return (
            f"strict miss vs best_variant={md.best_variant_id!r}; "
            f"patterns={md.miss_patterns}"
        )
    if primary == "other":
        return (
            "Strict miss across all cascade tiers; human review "
            "pending — pick the specific failure class."
        )
    return (
        "Auto-classified from cascade verdict; no human review "
        "needed for no_fail / cascade-tier categories."
    )


def verdict_label_from_cascade(cascade: CascadeVerdict) -> str:
    """Map a cascade verdict → the ``SubmissionEvaluation.verdict`` label.

    Shared by ``_build_submission_annotation`` (cloud + local runners) and
    ``annotate._eval_from_cascade`` (skeleton CLI + regrade CLI). Keep
    them in sync — every cascade tier that flips the headline verdict
    should appear here:

    * ``n3_any_audited_variant`` → ``"correct"`` (strict set-equal pass)
    * ``n4_tie_order`` / ``n5_llm_judge`` / ``n6_numeric_epsilon`` /
      ``n7_trailing_whitespace`` / ``n8_column_order`` / ``n9_case_fold`` →
      ``"valid_interpretation"`` (cascade-tier acceptance — the row is
      not strictly identical to the gold but matches under a named
      tolerance / under the LLM judge for an ``insufficient`` task)
    * otherwise → ``"invalid"``
    """
    if cascade.n3_any_audited_variant:
        return "correct"
    if (
        cascade.n4_tie_order
        or cascade.n5_llm_judge
        or cascade.n6_numeric_epsilon
        or cascade.n7_trailing_whitespace
        or cascade.n8_column_order
        or cascade.n9_case_fold
    ):
        return "valid_interpretation"
    return "invalid"


def _auto_failure_class(cascade: CascadeVerdict) -> tuple[str, bool, str]:
    """Pick the (primary, agent_at_fault, remediation_target) triple
    purely from the cascade. ``no_fail`` for N3 passes; each lower
    cascade tier maps to its own bucket so consumers know exactly which
    grader-tolerance dimension flipped the verdict. ``other`` is the
    only catch-all that requires human review.

    N5 (LLM judge) is treated as its own outcome (``novel_reading_accepted``)
    because the judge fires ONLY when ``metadata_sufficiency.verdict``
    is ``"insufficient"`` — so the agent is being accepted under a
    valid-novel-reading exception, not under a tolerance-of-the-gold."""
    if cascade.n3_any_audited_variant:
        return ("no_fail", False, "other")
    if cascade.n4_tie_order:
        return ("row_order", False, "grader")
    if cascade.n5_llm_judge and cascade.novel_reading_judgment == "pass":
        # Strict cascade missed; the judge accepted a novel reading
        # because the metadata couldn't pin a single answer.
        return ("novel_reading_accepted", False, "kb")
    if cascade.n6_numeric_epsilon:
        return ("numerical_precision", False, "grader")
    if cascade.n7_trailing_whitespace:
        return ("trailing_whitespace", False, "grader")
    if cascade.n8_column_order:
        return ("column_order", False, "grader")
    if cascade.n9_case_fold:
        return ("case_sensitivity", False, "grader")
    # Genuine strict miss across every cascade tier — agent miss. The
    # rich diagnostic detail lives on ``ev.miss_diagnostics`` (DEV-1515
    # session-4); ``FailureClassification.details`` carries a one-line
    # human summary derived from ``miss_diagnostics.miss_patterns``.
    return ("agent_miss", True, "agent")


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
    verdict_label = verdict_label_from_cascade(cascade)

    auto_primary, auto_at_fault, auto_remediation = _auto_failure_class(cascade)

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
        correct_under_case_fold=cascade.n9_case_fold,
        numeric_epsilon=epsilon,
        verdict=verdict_label,  # type: ignore[arg-type]
        matched_variant_id=cascade.matched_variant_id,
        rationale="",
        miss_diagnostics=cascade.miss_diagnostics,
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
            primary=auto_primary,  # type: ignore[arg-type]
            agent_at_fault=auto_at_fault,
            remediation_target=auto_remediation,  # type: ignore[arg-type]
            details=_failure_details_for(cascade, auto_primary),
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
    # Resolve the user-sim signal for `grade_submission`. Interactive
    # benchmarks (mini-interact a-interact) pass the int count of
    # `ask_user` calls so the `never_asked_user` diagnostic can fire
    # when the count is zero; one-shot benchmarks (livesqlbench) pass
    # None so the flag stays out of `miss_patterns`. Prefer the
    # already-parsed `user_sim_interaction.n_asks` over the raw
    # `n_ask_user_calls` since the former encodes the parsing rule.
    from bird_interact_agents.benchmark import get_benchmark
    try:
        _bench = get_benchmark(benchmark)
        _is_interactive = not _bench.one_shot
    except Exception:  # noqa: BLE001 — unknown benchmark token
        _is_interactive = False
    _user_sim_n_asks: Optional[int]
    if _is_interactive:
        if user_sim_interaction is not None:
            _user_sim_n_asks = user_sim_interaction.n_asks
        elif n_ask_user_calls is not None:
            _user_sim_n_asks = n_ask_user_calls
        else:
            _user_sim_n_asks = 0
    else:
        _user_sim_n_asks = None

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
        user_sim_n_asks=_user_sim_n_asks,
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


def load_task_annotation_or_implicit(
    *, instance_id: str, selected_database: str, benchmark: str,
    amb_user_query: str = "",
) -> TaskAnnotation:
    """Try to read ``<paths.annotations_root()>/<benchmark>/<db>/<inst>.task.json``;
    if missing, fall back to the in-memory implicit default. NEVER writes
    a synthesized stub to disk.

    Shared by the cloud worker (``ray_app.py``) and the local runner
    (``run.py``) so both paths produce the same TaskAnnotation for the
    same instance — keeping the per-row ``submission_annotation.json``
    comparable across local + cloud runs.
    """
    from bird_interact_agents.eval.annotation_io import (
        read_task_annotation, task_annotation_path,
    )

    # Leave ``repo_root`` unset so ``annotation_io._annotations_root``
    # honours ``BIRD_ANNOTATIONS_ROOT`` (the default already anchors at
    # ``paths.main_checkout_root()`` via ``paths.annotations_root()``,
    # so the production path is unchanged).
    p = task_annotation_path(
        benchmark=benchmark, selected_database=selected_database,
        instance_id=instance_id,
    )
    if p.exists():
        return read_task_annotation(p)
    return implicit_task_annotation(
        instance_id=instance_id,
        selected_database=selected_database,
        benchmark=benchmark,
        amb_user_query=amb_user_query,
    )


def load_audited_gold_rows_for(
    *, benchmark: str, instance_id: str,
) -> list[dict]:
    """Load every audited-gold row for ``instance_id`` from the
    consolidated JSONL. Empty list when no rows exist (graceful default
    — see ``implicit_task_annotation``)."""
    from bird_interact_agents import paths
    from bird_interact_agents.benchmark import get_benchmark

    try:
        bench = get_benchmark(benchmark.replace("-", "_"))
    except Exception:  # noqa: BLE001
        return []
    if getattr(bench, "audited_gold_layout", None) != "single_file":
        return []
    consolidated = paths.audited_gold_root() / f"{bench.name}_audited.jsonl"
    if not consolidated.exists():
        return []
    out: list[dict] = []
    for line in consolidated.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("instance_id") == instance_id:
            out.append(row)
    return out


def grade_one_submission(
    *,
    task_data: dict,
    submitted_sql: str,
    rows_dir: Path,
    run_id: str,
    benchmark: str,
    db_path: Path,
    conn: Any = None,
    cost_usd_agent: Optional[float] = None,
    cost_usd_user_sim: Optional[float] = None,
    duration_s: Optional[float] = None,
    n_agent_turns: Optional[int] = None,
    n_ask_user_calls: Optional[int] = None,
    predicted_row_count: Optional[int] = None,
    user_sim_interaction: Optional[UserSimInteraction] = None,
) -> Path:
    """Inline-grade one submission and write the per-row
    ``submission_annotation.json``. Idempotent at the per-(task, run)
    level — both the cloud fetch path and the local rows aggregator are
    no-overwrite at the destination.

    Shared between cloud (``cloud.ray_app``) and local (``run``) so the
    ``cascading_phase1`` block in ``eval.json`` is populated regardless
    of where the run was launched.
    """
    instance_id = task_data["instance_id"]
    selected_database = task_data["selected_database"]
    ann = load_task_annotation_or_implicit(
        instance_id=instance_id,
        selected_database=selected_database,
        benchmark=benchmark,
        amb_user_query=task_data.get("amb_user_query", ""),
    )
    audited_rows = load_audited_gold_rows_for(
        benchmark=benchmark, instance_id=instance_id,
    )
    return grade_and_write(
        rows_dir=rows_dir,
        instance_id=instance_id,
        benchmark=benchmark,
        run_id=run_id,
        task_annotation=ann,
        audited_gold_rows=audited_rows,
        original_sol_sql=list(
            task_data.get("original_sol_sql")
            or task_data.get("sol_sql")
            or [],
        ),
        submitted_sql=submitted_sql,
        db_path=db_path,
        conn=conn,
        trajectory_path=f"rows/{instance_id}/attempt-1.json",
        cost_usd_agent=cost_usd_agent,
        cost_usd_user_sim=cost_usd_user_sim,
        duration_s=duration_s,
        n_agent_turns=n_agent_turns,
        n_ask_user_calls=n_ask_user_calls,
        predicted_row_count=predicted_row_count,
        user_sim_interaction=user_sim_interaction,
    )
