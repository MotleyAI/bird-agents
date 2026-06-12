#!/usr/bin/env python3
"""Backfill cascade N1 for mini-interact stored runs using upstream's
``ex_base``-equivalent grader.

The pre-fix N1 ("phase1_against_original_gold") was bag equality on
``repr(cell)``. Upstream mini-interact's grader applies 2-dp Decimal/
float rounding + date normalisation + ``set()``-dedup comparison via
``test_case_default`` + ``ex_base`` + ``preprocess_results``. Result: ~6
slayer + ~3 raw mini-interact cases that pass upstream were being
demoted to N6 epsilon (or worse) in our cascade.

This script walks every stored result under
``paths.runs_root() / 'mini-interact' / <db> / <iid> / <run_id>.json``
and, for each one:

* loads ``submitted_sql`` from the JSON and ``sol_sql`` / ``conditions``
  from the per-task annotation (falling back to the canonical
  ``mini_interact.jsonl`` row when no annotation exists);
* skips conservatively if either side mutates the DB (``INSERT`` /
  ``UPDATE`` / ``DELETE`` / ``CREATE`` / ``DROP`` / ``ALTER`` /
  ``TRUNCATE`` / ``REPLACE`` at statement start) — the pristine
  backfill DB diverges from the inline-grader's post-mutation state
  for those tasks;
* resolves the SQLite db_path via
  ``paths.benchmark_data_root('mini-interact') / <db> / f"{<db>}.sqlite"``;
* re-runs the FULL cascade (via ``tolerant_grader.grade_submission``,
  which now dispatches N1 to upstream ``ex_base`` for mini-interact)
  so every field downstream of N1 (audited tiers, tolerance booleans,
  verdict, ``failure_classification``) gets recomputed consistently —
  NOT a single-field patch (Codex round-1 finding #1);
* rewrites the JSON in place when fields change.

The script is mini-interact only (livesqlbench backfill needs Postgres
+ per-task DB load; deferred to a separate follow-up).

Usage::

    uv run python scripts/regrade_n1_ex_base.py            # apply changes
    uv run python scripts/regrade_n1_ex_base.py --dry-run  # print would-flips only

Exits 0 when all tasks were processed without script-level errors
(state-sensitive / missing-input skips are normal, not failures).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from bird_interact_agents import paths
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.eval.annotation_schema import (
    SubmissionAnnotation,
    TaskAnnotation,
)
from bird_interact_agents.eval.grade_in_place import (
    _auto_failure_class,
    _build_submission_annotation,
    _verdict_to_phase,
    load_audited_gold_rows_for,
    load_task_annotation_or_implicit,
    normalize_sol_sql,
)
from bird_interact_agents.eval.tolerant_grader import (
    grade_submission,
)
from bird_interact_agents.eval.upstream_ex_base import is_mutation_sql

logger = logging.getLogger("regrade_n1_ex_base")

_BENCHMARK = "mini-interact"


@dataclass
class Report:
    processed: int = 0
    regraded_flipped: int = 0
    regraded_unchanged: int = 0
    would_flip: int = 0
    skipped_state_sensitive: int = 0
    skipped_missing_inputs: int = 0
    errors: int = 0
    flipped_iids: list[str] = field(default_factory=list)

    def to_one_line(self) -> str:
        return (
            f"processed={self.processed} "
            f"regraded_flipped={self.regraded_flipped} "
            f"regraded_unchanged={self.regraded_unchanged} "
            f"would_flip={self.would_flip} "
            f"skipped_state_sensitive={self.skipped_state_sensitive} "
            f"skipped_missing_inputs={self.skipped_missing_inputs} "
            f"errors={self.errors}"
        )


def _load_task_annotation_or_jsonl(
    instance_id: str, selected_database: str,
) -> tuple[Optional[TaskAnnotation], list[str], Optional[dict]]:
    """Load ``(task_annotation, sol_sql, conditions)`` from the per-task
    annotation when present, else from the canonical jsonl row.

    Returns ``(None, [], None)`` when neither source carries this task —
    the caller skips with ``skipped_missing_inputs``.
    """
    annotations_root = paths.annotations_root()
    ann_path = (
        annotations_root / _BENCHMARK / selected_database
        / f"{instance_id}.task.json"
    )
    sol_sql: list[str] = []
    conditions: Optional[dict] = None
    task_ann: Optional[TaskAnnotation] = None

    if ann_path.is_file():
        payload = json.loads(ann_path.read_text())
        # sol_sql / conditions live on the task json next to the
        # TaskAnnotation fields; TaskAnnotation uses `extra="forbid"`, so
        # peel them off before validating.
        sol_sql = normalize_sol_sql(payload.get("sol_sql"))
        conditions = payload.get("conditions")
        validation_payload = {
            k: v for k, v in payload.items()
            if k not in ("sol_sql", "conditions")
        }
        try:
            task_ann = TaskAnnotation.model_validate(validation_payload)
        except Exception:  # noqa: BLE001
            task_ann = None

    if not sol_sql:
        jsonl_path = os.environ.get("BIRD_MINI_INTERACT_DATA_PATH")
        candidates: list[Path] = []
        if jsonl_path:
            candidates.append(Path(jsonl_path))
        # Default canonical mini_interact.jsonl location.
        candidates.append(paths.benchmark_data_root(_BENCHMARK) / "mini_interact.jsonl")
        for cand in candidates:
            if not cand.is_file():
                continue
            for line in cand.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("instance_id") == instance_id:
                    sol_sql = normalize_sol_sql(row.get("sol_sql"))
                    if conditions is None:
                        conditions = row.get("conditions")
                    break
            if sol_sql:
                break

    if task_ann is None:
        try:
            task_ann = load_task_annotation_or_implicit(
                instance_id=instance_id,
                selected_database=selected_database,
                benchmark=_BENCHMARK,
                amb_user_query="",
            )
        except Exception:  # noqa: BLE001
            task_ann = None

    return task_ann, sol_sql, conditions


def _resolve_db_path(selected_database: str) -> Path:
    """Codex round-1 finding #4: the SQLite layout is
    ``benchmark_data_root('mini-interact') / <db> / <db>.sqlite``."""
    return (
        paths.benchmark_data_root(_BENCHMARK)
        / selected_database
        / f"{selected_database}.sqlite"
    )


def _evaluation_diffs(before: SubmissionAnnotation, after: SubmissionAnnotation) -> bool:
    """True iff at least one persisted field changed (``evaluation`` or
    ``failure_classification``). Identity-on-content; bytes-equal JSON
    elsewhere on idempotent re-run."""
    return (
        before.evaluation.model_dump() != after.evaluation.model_dump()
        or before.failure_classification.model_dump()
            != after.failure_classification.model_dump()
    )


def _process_one(
    result_path: Path, *, dry_run: bool, report: Report,
) -> None:
    payload = json.loads(result_path.read_text())
    try:
        before = SubmissionAnnotation.model_validate(payload)
    except Exception:  # noqa: BLE001
        logger.exception("Could not parse SubmissionAnnotation at %s", result_path)
        report.errors += 1
        return

    instance_id = before.instance_id
    selected_database = before.selected_database
    submitted_sql = before.submitted_sql or ""

    task_ann, sol_sql, conditions = _load_task_annotation_or_jsonl(
        instance_id, selected_database,
    )
    if task_ann is None or not sol_sql or not submitted_sql:
        logger.info(
            "[skip missing_inputs] %s/%s",
            selected_database, instance_id,
        )
        report.skipped_missing_inputs += 1
        return

    # Mutation-bearing skip (Codex round-1 finding #5).
    all_sqls = [submitted_sql, *sol_sql]
    if any(is_mutation_sql(s) for s in all_sqls):
        logger.info(
            "[skip state_sensitive] %s/%s (mutation-bearing SQL)",
            selected_database, instance_id,
        )
        report.skipped_state_sensitive += 1
        return

    db_path = _resolve_db_path(selected_database)
    if not db_path.is_file():
        logger.info(
            "[skip missing_inputs] %s/%s — db_path %s not found",
            selected_database, instance_id, db_path,
        )
        report.skipped_missing_inputs += 1
        return

    try:
        audited_rows = load_audited_gold_rows_for(
            benchmark=_BENCHMARK,
            instance_id=instance_id,
        )
    except Exception:  # noqa: BLE001
        audited_rows = []

    benchmark_obj = get_benchmark(_BENCHMARK)
    try:
        cascade = grade_submission(
            task_annotation=task_ann,
            audited_gold_rows=audited_rows,
            original_sol_sql=sol_sql,
            submitted_sql=submitted_sql,
            db_path=db_path,
            benchmark=benchmark_obj,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "grade_submission raised for %s/%s", selected_database, instance_id,
        )
        report.errors += 1
        return

    after = _build_submission_annotation(
        task_annotation=task_ann,
        cascade=cascade,
        benchmark=_BENCHMARK,
        run_id=before.submission.cloud_run_id,
        trajectory_path=before.submission.trajectory_path,
        predicted_row_count=before.submission.predicted_row_count,
        duration_s=before.submission.duration_s,
        cost_usd_agent=before.submission.cost_usd_agent,
        cost_usd_user_sim=before.submission.cost_usd_user_sim,
        n_agent_turns=before.submission.n_agent_turns,
        n_ask_user_calls=before.submission.n_ask_user_calls,
        submitted_sql=submitted_sql,
        user_sim_interaction=before.user_sim_interaction,
        config=before.submission.config,
    )
    # Preserve `annotated_by` / `annotated_at` from the original
    # annotation so the regrade isn't mistakenly attributed to the
    # auto-annotator. We only changed the evaluation block.
    after = after.model_copy(update={
        "annotated_by": before.annotated_by,
        "annotated_at": before.annotated_at,
        "autopsy": before.autopsy,
        "decision_point": before.decision_point,
    })

    # Every task we successfully re-graded counts as processed,
    # whether or not the result flipped — the report line's
    # processed counter then matches the size of the work-list minus
    # the skip / error buckets. Without this, idempotent re-runs
    # report processed=0 even though every JSON was loaded, graded,
    # and compared. (CodeRabbit round 2.)
    report.processed += 1

    flipped = _evaluation_diffs(before, after)
    if not flipped:
        report.regraded_unchanged += 1
        return

    if dry_run:
        report.would_flip += 1
        n1_before = before.evaluation.phase1_against_original_gold
        n1_after = after.evaluation.phase1_against_original_gold
        primary_before = before.failure_classification.primary
        primary_after = after.failure_classification.primary
        logger.info(
            "[would_flip] %s/%s n1 %s→%s primary %s→%s",
            selected_database, instance_id,
            n1_before, n1_after, primary_before, primary_after,
        )
        return

    report.regraded_flipped += 1
    report.flipped_iids.append(f"{selected_database}/{instance_id}")
    # Rewrite the JSON in place.
    out = after.model_dump(mode="json")
    result_path.write_text(json.dumps(out, indent=2) + "\n")
    logger.info(
        "[regraded] %s/%s n1=%s primary=%s",
        selected_database, instance_id,
        after.evaluation.phase1_against_original_gold,
        after.failure_classification.primary,
    )


def regrade(
    *, runs_root: Optional[Path] = None, dry_run: bool = False,
) -> Report:
    """Top-level entrypoint: walk runs_root and process each
    mini-interact result JSON. Returns a final :class:`Report`."""
    if runs_root is None:
        runs_root = paths.runs_root()
    benchmark_root = runs_root / _BENCHMARK
    report = Report()
    if not benchmark_root.is_dir():
        logger.warning("[regrade] no mini-interact root at %s", benchmark_root)
        return report

    # Discover result JSONs. The skip-on-non-mini-interact contract is
    # already enforced by only walking mini-interact's subtree (other
    # benchmarks' subtrees are never visited).
    for f in sorted(benchmark_root.glob("*/*/*.json")):
        if f.name.endswith(".trajectory.json"):
            continue
        try:
            _process_one(f, dry_run=dry_run, report=report)
        except Exception:  # noqa: BLE001
            logger.exception("[regrade] unexpected error on %s", f)
            report.errors += 1
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill mini-interact cascade N1 to upstream ex_base "
            "semantics (DEV-1550 follow-up)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--runs-root", default=None,
        help="Override runs/ root (default: $BIRD_RUNS_ROOT or paths.runs_root()).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="[%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, args.log_level),
    )

    runs_root = Path(args.runs_root) if args.runs_root else None
    report = regrade(runs_root=runs_root, dry_run=args.dry_run)
    print(report.to_one_line())
    if report.flipped_iids:
        print(f"flipped {len(report.flipped_iids)} tasks:")
        for iid in report.flipped_iids:
            print(f"  {iid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
