"""DEV-1515: annotation-skeleton generator (task-side only).

Builds the per-task JSON skeletons mechanically from each benchmark
data row. Human-judgment fields are left as ``PENDING_HUMAN_REVIEW``
sentinels; ``--task-mode`` controls overwrite semantics.

Per-submission skeletons are built by ``scripts/dev1515_convert_runs.py``
from a completed run's per-row attempts (it has the trajectory + usage
plumbing this module does not).

Usage::

    python -m bird_interact_agents.eval.annotate \\
        --benchmark mini_interact \\
        [--instance-ids ...] \\
        [--task-mode {init,refresh,force-all}] \\
        [--dry-run]

``--benchmark`` accepts both the dash form (``mini-interact``) and the
underscore form (``mini_interact``); ``annotation_io`` normalizes to
the canonical underscore form on every path-build so writes from
either form land in the same on-disk tree.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from bird_interact_agents import paths
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.eval.annotation_io import (
    read_submission_annotation,
    read_task_annotation,
    run_annotation_path,
    task_annotation_path,
    write_run_annotation,
    write_task_annotation,
)
from bird_interact_agents.eval.annotation_schema import (
    AuditedGoldRef,
    FailureClassification,
    GoldVariantRef,
    MaskedTerm,
    MetadataSufficiency,
    Provenance,
    SubmissionAnnotation,
    SubmissionEvaluation,
    SubmissionMetadata,
    TaskAnnotation,
    UserSimInteraction,
    UserSimResponseSummary,
)
from bird_interact_agents.eval.grade_in_place import _auto_failure_class
from bird_interact_agents.eval.regrade import _latest_attempt_file

PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"


def _resolve_db_sqlite_path(benchmark_data_root: Path, db: str) -> Path:
    """Return the SQLite DB path for ``db``, falling back to ``_template.sqlite``.

    LiveSQLBench uses ``{db}_template.sqlite``; mini-interact uses ``{db}.sqlite``.
    Mirrors the fallback logic in regrade.py."""
    db_dir = benchmark_data_root / db
    primary = db_dir / f"{db}.sqlite"
    if not primary.exists():
        alt = db_dir / f"{db}_template.sqlite"
        if alt.exists():
            return alt
    return primary


# ---------------------------------------------------------------------------
# Mechanical builders — pure, no I/O
# ---------------------------------------------------------------------------


def _benchmark_task_jsonl_name(benchmark: str) -> str:
    try:
        return get_benchmark(benchmark).data_file
    except ValueError:
        return f"{benchmark}.jsonl"


def _masked_terms_from(task_row: dict) -> list[MaskedTerm]:
    amb = (task_row.get("user_query_ambiguity") or {}).get("critical_ambiguity", [])
    out: list[MaskedTerm] = []
    for entry in amb or []:
        me = entry.get("metadata_evidence") or []
        out.append(MaskedTerm(
            term=entry.get("term", ""),
            type=entry.get("type", "intent_ambiguity"),
            is_mask=bool(entry.get("is_mask", True)),
            metadata_evidence=me if isinstance(me, list) else [me],
        ))
    return out


def generate_task_annotation(
    *,
    task_row: dict,
    benchmark: str,
    annotated_at: Optional[str] = None,
) -> TaskAnnotation:
    """Mechanically-filled TaskAnnotation skeleton; human-judgment fields
    carry ``PENDING_HUMAN_REVIEW`` sentinels."""
    if annotated_at is None:
        annotated_at = _dt.datetime.now(_dt.timezone.utc).replace(
            microsecond=0,
        ).isoformat()
    return TaskAnnotation(
        instance_id=task_row["instance_id"],
        selected_database=task_row["selected_database"],
        annotated_by="auto-skeleton",
        annotated_at=annotated_at,
        amb_user_query=task_row.get("amb_user_query", ""),
        external_knowledge=list(task_row.get("external_knowledge", []) or []),
        masked_terms=_masked_terms_from(task_row),
        metadata_sufficiency=MetadataSufficiency(
            verdict="ambiguous",  # safe non-judgemental default
            rationale=PENDING_HUMAN_REVIEW,
            evidence_sources_consulted=[],
        ),
        original_gold_is_correct=False,
        gold_variants=[],
        evaluator_prompt=None,
        provenance=Provenance(
            task_jsonl_path=_benchmark_task_jsonl_name(benchmark),
            task_jsonl_instance_id=task_row["instance_id"],
        ),
    )


def _user_sim_interaction_from_trajectory(traj) -> UserSimInteraction:
    """Build the ``UserSimInteraction`` summary from a trajectory.

    Several agent flavors (``pydantic_ai_otf_encode``,
    ``pydantic_ai_recursive``) emit ``trajectory`` as a DICT shape
    (``{"final_output_excerpt": ..., "agents": [...]}``) instead of
    the per-turn-step list this helper was originally designed for.
    Without the type check below, ``list(dict_traj)`` at the call
    site would yield the dict's KEYS (strings); iterating and calling
    ``item.get("role")`` then raises ``AttributeError`` on a ``str``
    and the whole grader-fallback / skeleton-build path crashes after
    the cascade has already computed cleanly (Codex r10).

    Contract: any non-list-of-dicts shape degrades to a zero-asks
    ``UserSimInteraction()`` default. The per-step path stays the
    same for the genuine list-of-dicts case.
    """
    if not isinstance(traj, list):
        return UserSimInteraction()
    n_asks = 0
    responses: list[UserSimResponseSummary] = []
    for i, item in enumerate(traj):
        if not isinstance(item, dict):
            # Mixed shape (or list of strings from ``list(dict)``) —
            # skip non-dict entries instead of raising on ``.get()``.
            continue
        if item.get("role") == "tool_call" and item.get("name") == "ask_user":
            n_asks += 1
        elif item.get("role") in ("user_sim", "user") and i > 0:
            prev = traj[i - 1]
            if (
                isinstance(prev, dict)
                and prev.get("role") == "tool_call"
                and prev.get("name") == "ask_user"
            ):
                # Recently followed an ask — record short summary.
                txt = str(item.get("content") or "")
                responses.append(UserSimResponseSummary(
                    trajectory_idx=i,
                    summary=(txt[:80] + "…") if len(txt) > 80 else txt,
                ))
    return UserSimInteraction(
        n_asks=n_asks,
        key_responses=responses,
        disclosed_resolutions=[],
        undisclosed_resolutions=[],
    )


def _skeleton_failure_classification(cascade: Any) -> FailureClassification:
    """Auto-classify from the cascade; ``other`` for strict misses still
    needing human review."""
    primary, at_fault, remediation = _auto_failure_class(cascade)
    if primary == "other":
        details = PENDING_HUMAN_REVIEW
    else:
        details = (
            "Auto-classified from cascade verdict; no human review "
            "needed for no_fail / cascade-tier categories."
        )
    return FailureClassification(
        primary=primary,  # type: ignore[arg-type]
        agent_at_fault=at_fault,
        remediation_target=remediation,  # type: ignore[arg-type]
        details=details,
    )


def _eval_from_cascade(cascade: Any, epsilon: float = 1e-6) -> SubmissionEvaluation:
    """Convert a tolerant_grader CascadeVerdict into a
    SubmissionEvaluation persistence shape. Uses
    :func:`grade_in_place.verdict_label_from_cascade` so this helper and
    the cloud/local-runner builder cannot drift apart — without the
    shared mapping, N4/N5/N6/N7/N8 cascade-tier passes would land here
    with ``verdict="invalid"`` while the inline grader's annotations
    carried ``verdict="valid_interpretation"``."""
    from bird_interact_agents.eval.grade_in_place import (
        verdict_label_from_cascade,
    )

    return SubmissionEvaluation(
        phase1_against_original_gold="pass" if cascade.n1_original_gold else "fail",
        phase1_against_audited_primary="pass" if cascade.n2_audited_primary else "fail",
        phase1_against_any_audited_variant=(
            "pass" if cascade.n3_any_audited_variant else "fail"
        ),
        phase1_against_variants=list(cascade.variant_matches),
        correct_up_to_tie_order=cascade.n4_tie_order,
        novel_reading_judgment=cascade.novel_reading_judgment,
        correct_under_numeric_epsilon=cascade.n6_numeric_epsilon,
        correct_under_trailing_whitespace=cascade.n7_trailing_whitespace,
        correct_under_column_order=cascade.n8_column_order,
        correct_under_case_fold=cascade.n9_case_fold,
        numeric_epsilon=epsilon,
        verdict=verdict_label_from_cascade(cascade),  # type: ignore[arg-type]
        matched_variant_id=cascade.matched_variant_id,
        rationale="",
        miss_diagnostics=getattr(cascade, "miss_diagnostics", None),
    )


def generate_submission_annotation(
    *,
    rows_dir: Path,
    instance_id: str,
    selected_database: str,
    benchmark: str,
    run_id: str,
    task_row: dict,
    grader: Callable[..., Any],
    epsilon: float = 1e-6,
    annotated_at: Optional[str] = None,
) -> SubmissionAnnotation:
    """Mechanically-filled SubmissionAnnotation skeleton.

    ``grader`` is a callable that returns a CascadeVerdict given the
    submitted SQL + task row context. Tests pass a stub; production
    callers pass ``tolerant_grader.grade_submission``."""
    sub_dir = Path(rows_dir) / instance_id
    attempt_path = _latest_attempt_file(sub_dir) or (sub_dir / "attempt-1.json")
    attempt = json.loads(attempt_path.read_text())
    submitted_sql = attempt.get("submitted_sql", "")
    # Don't wrap with ``list(...)`` — dict trajectories from
    # pydantic_ai_otf_encode etc. would coerce to a list of keys.
    # ``_user_sim_interaction_from_trajectory`` handles non-list
    # shapes defensively (Codex r10).
    traj = attempt.get("trajectory") or []
    usage = attempt.get("usage", {}) or {}

    cascade = grader(
        instance_id=instance_id,
        submitted_sql=submitted_sql,
        task_row=task_row,
    )
    ev = _eval_from_cascade(cascade, epsilon=epsilon)
    if annotated_at is None:
        annotated_at = _dt.datetime.now(_dt.timezone.utc).replace(
            microsecond=0,
        ).isoformat()

    return SubmissionAnnotation(
        instance_id=instance_id,
        selected_database=selected_database,
        task_annotation_ref=(
            f"annotations/{benchmark}/{selected_database}/"
            f"{instance_id}.task.json"
        ),
        annotated_by="auto-skeleton",
        annotated_at=annotated_at,
        submission=SubmissionMetadata(
            cloud_run_id=run_id,
            trajectory_path=str(attempt_path),
            predicted_row_count=attempt.get("predicted_row_count"),
            duration_s=attempt.get("duration_s"),
            cost_usd_agent=usage.get("cost_usd_agent"),
            cost_usd_user_sim=usage.get("cost_usd_user_sim"),
            n_agent_turns=usage.get("n_agent_turns"),
            n_ask_user_calls=usage.get("n_ask_user_calls"),
        ),
        evaluation=ev,
        failure_classification=_skeleton_failure_classification(cascade),
        decision_point=None,
        user_sim_interaction=_user_sim_interaction_from_trajectory(traj),
    )


# ---------------------------------------------------------------------------
# Mode-aware writers
# ---------------------------------------------------------------------------


def _refresh_mechanical(existing: TaskAnnotation, fresh: TaskAnnotation) -> TaskAnnotation:
    """Preserve any non-sentinel human-judgment field on ``existing``;
    overwrite mechanical fields from ``fresh``."""
    out = existing.model_copy(deep=True)
    out.amb_user_query = fresh.amb_user_query
    out.external_knowledge = list(fresh.external_knowledge)
    out.masked_terms = list(fresh.masked_terms)
    out.provenance = fresh.provenance
    out.annotated_at = fresh.annotated_at
    # Sentinels mean "still pending" — re-import the sentinel only if
    # the user hasn't yet authored a real value.
    if (
        existing.metadata_sufficiency.rationale == PENDING_HUMAN_REVIEW
        and existing.metadata_sufficiency.verdict == fresh.metadata_sufficiency.verdict
        and existing.metadata_sufficiency.evidence_sources_consulted
            == fresh.metadata_sufficiency.evidence_sources_consulted
    ):
        out.metadata_sufficiency = fresh.metadata_sufficiency
    return out


def write_task_skeleton(
    *,
    task_row: dict,
    benchmark: str,
    mode: str = "init",  # init | refresh | force-all
    dry_run: bool = False,
    repo_root: Optional[Path] = None,
) -> Optional[Path]:
    """Returns the destination path. ``None`` ⇒ skipped (e.g. file
    already exists in init mode)."""
    repo_root = repo_root or paths.main_checkout_root()
    fresh = generate_task_annotation(task_row=task_row, benchmark=benchmark)
    dest = task_annotation_path(
        benchmark=benchmark, selected_database=task_row["selected_database"],
        instance_id=task_row["instance_id"], repo_root=repo_root,
    )
    if mode == "init" and dest.exists():
        return None
    if mode == "refresh" and dest.exists():
        existing = read_task_annotation(dest)
        merged = _refresh_mechanical(existing, fresh)
        if not dry_run:
            write_task_annotation(merged, dest)
        return dest
    # force-all (or init when file missing).
    if not dry_run:
        write_task_annotation(fresh, dest)
    return dest


def write_submission_skeleton(
    *,
    rows_dir: Path,
    instance_id: str,
    selected_database: str,
    benchmark: str,
    run_id: str,
    task_row: dict,
    grader: Callable[..., Any],
    mode: str = "overwrite",  # overwrite | init
    dry_run: bool = False,
    repo_root: Optional[Path] = None,
) -> Optional[Path]:
    repo_root = repo_root or paths.main_checkout_root()
    dest = run_annotation_path(
        benchmark=benchmark, selected_database=selected_database,
        instance_id=instance_id, run_id=run_id, repo_root=repo_root,
    )
    if mode == "init" and dest.exists():
        return None
    ann = generate_submission_annotation(
        rows_dir=rows_dir, instance_id=instance_id,
        selected_database=selected_database, benchmark=benchmark,
        run_id=run_id, task_row=task_row, grader=grader,
    )
    if not dry_run:
        write_run_annotation(ann, dest)
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_task_rows(*, benchmark: str, instance_ids: Optional[Iterable[str]]) -> list[dict]:
    """Load task rows for the given instance IDs from the benchmark JSONL."""
    bench = get_benchmark(benchmark)
    data_path = paths.benchmark_data_root(bench) / bench.data_file
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    if instance_ids:
        wanted = set(instance_ids)
        rows = [r for r in rows if r.get("instance_id") in wanted]
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate / refresh per-task annotation skeletons.",
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--instance-ids", default=None,
        help="Comma-separated subset; default = every instance in the "
             "benchmark data file.",
    )
    parser.add_argument(
        "--task-mode", choices=("init", "refresh", "force-all"),
        default="init",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-id", default=None,
        help="Cloud run ID. When provided, also generate per-submission "
             "annotation skeletons from the run's attempt files.",
    )
    parser.add_argument(
        "--submission-mode", choices=("overwrite", "init"),
        default="overwrite",
        help="Overwrite-semantics for submission skeletons (default: overwrite).",
    )
    args = parser.parse_args(argv)

    benchmark = get_benchmark(args.benchmark).name
    instance_ids = (
        [s.strip() for s in args.instance_ids.split(",")]
        if args.instance_ids else None
    )
    rows = _load_task_rows(benchmark=benchmark, instance_ids=instance_ids)
    for row in rows:
        write_task_skeleton(
            task_row=row, benchmark=benchmark,
            mode=args.task_mode, dry_run=args.dry_run,
        )
    print(f"task skeletons: {len(rows)} processed in mode={args.task_mode}")

    if args.run_id:
        from bird_interact_agents.eval.grade_in_place import (
            load_audited_gold_rows_for,
            load_task_annotation_or_implicit,
        )
        from bird_interact_agents.eval.tolerant_grader import grade_submission

        run_dir = paths.results_root() / "cloud" / args.run_id
        rows_dir = run_dir / "rows"
        if not rows_dir.exists():
            print(f"rows_dir not found: {rows_dir}")
            return 1
        n_sub = 0
        n_skipped = 0
        for row in rows:
            iid = row["instance_id"]
            if not (rows_dir / iid).is_dir():
                n_skipped += 1
                continue
            db = row["selected_database"]
            db_path = _resolve_db_sqlite_path(
                paths.benchmark_data_root(benchmark), db
            )
            ann = load_task_annotation_or_implicit(
                instance_id=iid, selected_database=db,
                benchmark=benchmark, amb_user_query=row.get("amb_user_query", ""),
            )
            audited = load_audited_gold_rows_for(benchmark=benchmark, instance_id=iid)

            def _grader(  # noqa: E731
                *,
                instance_id: str,
                submitted_sql: str,
                task_row: dict,
                _ann=ann,
                _audited=audited,
                _row=row,
                _db_path=db_path,
                **_kw,
            ):
                return grade_submission(
                    task_annotation=_ann,
                    audited_gold_rows=_audited,
                    original_sol_sql=_row.get("sol_sql"),
                    submitted_sql=submitted_sql,
                    db_path=_db_path,
                    conn=None,
                )

            dest = write_submission_skeleton(
                rows_dir=rows_dir,
                instance_id=iid,
                selected_database=db,
                benchmark=benchmark,
                run_id=args.run_id,
                task_row=row,
                grader=_grader,
                mode=args.submission_mode,
                dry_run=args.dry_run,
            )
            if dest is not None:
                n_sub += 1
        skip_str = f", {n_skipped} skipped (no attempt dir)" if n_skipped else ""
        print(f"submission skeletons: {n_sub} written in mode={args.submission_mode}{skip_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
