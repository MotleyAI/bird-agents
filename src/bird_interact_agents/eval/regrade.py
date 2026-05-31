"""DEV-1515: offline re-grade CLI.

Re-runs the tolerant grader over a completed run's saved artefacts,
overwriting the per-(instance, run) SubmissionAnnotation in the main
checkout. Distinct from ``driver.fetch``'s merge — that one is
no-overwrite; this one is opt-in OVERWRITE for the explicit "re-grade"
workflow (after adding a new variant or editing the evaluator_prompt).

Run::

    python -m bird_interact_agents.eval.regrade \\
        --run-id <id> --benchmark mini-interact \\
        [--instance-ids ...] \\
        [--force-llm-judge]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_io import (
    submission_annotation_path,
    write_submission_annotation,
)
from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation
from bird_interact_agents.eval.cascading_report import emit_cascading_eval_json


class RegradeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    benchmark: str
    regraded: int = 0
    skipped: int = 0
    regraded_instances: list[str] = Field(default_factory=list)


def _attempt_rows_dir(run_dir: Path) -> Path:
    return run_dir / "rows"


def clear_llm_judge_cache(
    *,
    cache_path: Path,
    instance_ids: Iterable[str],
) -> None:
    """Drop cache entries whose embedded ``instance_id`` matches any of
    ``instance_ids``. Entries for other instances are preserved."""
    if not cache_path.exists():
        return
    cache = json.loads(cache_path.read_text())
    wanted = set(instance_ids)
    new = {
        k: v for k, v in cache.items()
        if v.get("instance_id") not in wanted
    }
    cache_path.write_text(json.dumps(new, indent=2))


def regrade_run(
    *,
    run_id: str,
    benchmark: str,
    run_dir: Path,
    instance_ids: Optional[List[str]] = None,
    force_llm_judge: bool = False,
    grader: Callable[..., Any],
    repo_root: Optional[Path] = None,
) -> RegradeReport:
    """Walk per-task attempt JSONs and re-grade each. Writes a fresh
    ``eval_regraded.json`` alongside the historical ``eval.json``; the
    historical file is NOT touched."""
    repo_root = repo_root or paths.main_checkout_root()
    rows_dir = _attempt_rows_dir(run_dir)
    report = RegradeReport(run_id=run_id, benchmark=benchmark)
    if not rows_dir.exists():
        return report

    filter_set = set(instance_ids) if instance_ids else None
    if force_llm_judge and filter_set:
        clear_llm_judge_cache(
            cache_path=run_dir / "llm_judge_cache.json",
            instance_ids=filter_set,
        )

    fresh_rows_dir = run_dir / "regrade_rows"
    fresh_rows_dir.mkdir(parents=True, exist_ok=True)

    for sub in sorted(p for p in rows_dir.iterdir() if p.is_dir()):
        instance_id = sub.name
        if filter_set is not None and instance_id not in filter_set:
            report.skipped += 1
            continue
        attempt = sub / "attempt-1.json"
        if not attempt.exists():
            report.skipped += 1
            continue
        attempt_data = json.loads(attempt.read_text())
        submitted_sql = attempt_data.get("submitted_sql", "")
        selected_database = attempt_data.get("selected_database", "")
        cascade = grader(
            instance_id=instance_id,
            submitted_sql=submitted_sql,
            task_row=attempt_data,
        )
        # Build a fresh SubmissionAnnotation from the cascade.
        from bird_interact_agents.eval.annotate import (
            _eval_from_cascade,
            _user_sim_interaction_from_trajectory,
        )
        from bird_interact_agents.eval.annotation_schema import (
            FailureClassification,
            SubmissionMetadata,
        )
        usage = attempt_data.get("usage", {}) or {}
        ann = SubmissionAnnotation(
            instance_id=instance_id,
            selected_database=selected_database,
            task_annotation_ref=(
                f"annotations/{benchmark}/{selected_database}/"
                f"{instance_id}.task.json"
            ),
            annotated_by="auto-regrade",
            annotated_at="",  # populated below
            submission=SubmissionMetadata(
                cloud_run_id=run_id,
                trajectory_path=str(attempt),
                predicted_row_count=attempt_data.get("predicted_row_count"),
                duration_s=attempt_data.get("duration_s"),
                cost_usd_agent=usage.get("cost_usd_agent"),
                cost_usd_user_sim=usage.get("cost_usd_user_sim"),
                n_agent_turns=usage.get("n_agent_turns"),
                n_ask_user_calls=usage.get("n_ask_user_calls"),
            ),
            evaluation=_eval_from_cascade(cascade),
            failure_classification=FailureClassification(
                primary="other",
                agent_at_fault=not cascade.n3_any_audited_variant,
                remediation_target="other",
            ),
            user_sim_interaction=_user_sim_interaction_from_trajectory(
                list(attempt_data.get("trajectory", []) or []),
            ),
        )
        import datetime as _dt
        ann.annotated_at = _dt.datetime.now(_dt.timezone.utc).replace(
            microsecond=0,
        ).isoformat()

        # OVERWRITE the per-(instance, run) annotation in the main checkout.
        dest = submission_annotation_path(
            benchmark=benchmark, selected_database=selected_database,
            instance_id=instance_id, run_id=run_id, repo_root=repo_root,
        )
        write_submission_annotation(ann, dest)

        # Stash the per-row file in a fresh dir so the cascading_phase1
        # block in eval_regraded.json is built from THIS regrade pass, not
        # the historical run's annotations.
        fresh_sub = fresh_rows_dir / instance_id
        fresh_sub.mkdir(parents=True, exist_ok=True)
        (fresh_sub / "submission_annotation.json").write_text(
            ann.model_dump_json(indent=2, exclude_none=False) + "\n",
        )

        report.regraded += 1
        report.regraded_instances.append(instance_id)

    # Emit eval_regraded.json — NEVER touch eval.json.
    emit_cascading_eval_json(
        fresh_rows_dir, run_dir / "eval_regraded.json",
        base_metrics={},
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-grade a completed run's per-task artefacts.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--instance-ids", default=None)
    parser.add_argument("--force-llm-judge", action="store_true")
    args = parser.parse_args(argv)

    instance_ids = (
        [s.strip() for s in args.instance_ids.split(",")]
        if args.instance_ids else None
    )
    from bird_interact_agents.eval.tolerant_grader import grade_submission
    run_dir = paths.main_checkout_root() / "results" / "cloud" / args.run_id

    def _grader(*, instance_id: str, submitted_sql: str, task_row: dict, **_kw):
        # Minimal end-to-end wiring — production callers pre-build the
        # implicit annotation + audited gold rows themselves.
        from bird_interact_agents.eval.implicit_annotation import (
            implicit_task_annotation,
        )
        from bird_interact_agents.cloud.ray_app import (
            _load_audited_gold_rows_for, _load_task_annotation_or_implicit,
        )
        ann = _load_task_annotation_or_implicit(
            instance_id=instance_id,
            selected_database=task_row.get("selected_database", ""),
            benchmark=args.benchmark,
            amb_user_query=task_row.get("amb_user_query", ""),
        )
        audited = _load_audited_gold_rows_for(
            benchmark=args.benchmark, instance_id=instance_id,
        )
        return grade_submission(
            task_annotation=ann,
            audited_gold_rows=audited,
            original_sol_sql=list(
                task_row.get("original_sol_sql") or task_row.get("sol_sql") or [],
            ),
            submitted_sql=submitted_sql,
            db_path=Path("/dev/null"),
            conn=None,
        )

    report = regrade_run(
        run_id=args.run_id, benchmark=args.benchmark, run_dir=run_dir,
        instance_ids=instance_ids, force_llm_judge=args.force_llm_judge,
        grader=_grader,
    )
    print(f"regrade: {report.regraded} instances rewritten, "
          f"{report.skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
