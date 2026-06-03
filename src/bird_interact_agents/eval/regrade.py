"""DEV-1515: offline re-grade CLI.

Re-runs the tolerant grader over a completed run's saved artefacts,
overwriting the per-(instance, run) SubmissionAnnotation in the main
checkout. Distinct from ``driver.fetch``'s merge — that one is
no-overwrite; this one is opt-in OVERWRITE for the explicit "re-grade"
workflow (after adding a new variant or editing the evaluator_prompt).

Run::

    python -m bird_interact_agents.eval.regrade \\
        --run-id <id> --benchmark mini-interact \\
        [--instance-ids ...]

The N5 LLM-judge runs automatically when a task's
``metadata_sufficiency.verdict == "insufficient"``; it uses the agent's
own model (read from ``<run_dir>/manifest.json``) so the judge re-asks
the same model whether its submission is a defensible novel reading
of the ambiguous task. ``CachedLLMJudge`` persists verdicts to
``<run_dir>/llm_judge_cache.json`` keyed on model name + content
hashes, so re-grades reuse decisions and a model change naturally
invalidates entries (no separate cache-clear flag).
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_io import (
    submission_annotation_path,
    write_submission_annotation,
)
from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation
from bird_interact_agents.eval.cascading_report import emit_cascading_eval_json
from bird_interact_agents.eval.grade_in_place import normalize_sol_sql


class RegradeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    benchmark: str
    regraded: int = 0
    skipped: int = 0
    regraded_instances: list[str] = Field(default_factory=list)


def _attempt_rows_dir(run_dir: Path) -> Path:
    return run_dir / "rows"


_ATTEMPT_FILE_RE = re.compile(r"attempt-(\d+)\.json")


def _latest_attempt_file(sub: Path) -> Path | None:
    """Return the highest-numbered ``attempt-N.json`` in ``sub`` or
    None when no attempt file exists.

    Pre-fix the regrade CLI hardcoded ``attempt-1.json`` (Codex r10), so
    a resubmit's attempt-2 was either silently skipped (instance had ONLY
    attempt-2) or, worse, overwritten by stale attempt-1 data — even
    though cloud collation already treats the max attempt as canonical
    and the round-8 fetch merge compares attempt numbers."""
    if not sub.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for p in sub.iterdir():
        if not p.is_file():
            continue
        m = _ATTEMPT_FILE_RE.match(p.name)
        if m is None:
            continue
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        if best is None or n > best[0]:
            best = (n, p)
    return best[1] if best else None


def _build_original_sql_index(benchmark: str) -> dict[str, list[str]]:
    """Map ``instance_id`` → list-of-SQL-strings for the benchmark's
    original gold. mini_interact carries ``sol_sql`` inline on each task
    row in ``mini_interact.jsonl``; livesqlbench ships an empty
    ``sol_sql`` on the public ``livesqlbench_data_sqlite.jsonl`` and the
    real list lives on the gated gold sidecar (env override
    ``BIRD_LIVESQLBENCH_GOLD_FILE``). Look it up once at CLI startup so
    the per-row grader doesn't repeatedly parse a multi-megabyte JSONL.
    Empty rows fall back to ``[]`` so the cascade's N1 just doesn't fire
    for instances whose source row genuinely has no gold (rather than
    crashing).
    """
    from bird_interact_agents.benchmark import get_benchmark

    out: dict[str, list[str]] = {}
    data_file = paths.benchmark_data_file(benchmark)
    if data_file.exists():
        with data_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = r.get("instance_id")
                sol = normalize_sol_sql(r.get("sol_sql"))
                # ``normalize_sol_sql`` returns ``[]`` for None / empty /
                # missing, ``[s]`` for a string, and ``list(value)`` for
                # a list — so a string-shaped ``sol_sql`` is no longer
                # silently dropped at index-build time (Codex r6).
                if iid and sol:
                    out[iid] = sol
    # Merge in livesqlbench's gated sidecar if available.
    bench = get_benchmark(benchmark)
    if bench.gold_required:
        gold_path: Optional[Path] = None
        # DEV-1525: gated gold lives at paths.gated_gold_root(benchmark=) /
        # <gt_sidecar>.jsonl; a BIRD_GATED_GOLD_ROOT env var overrides the
        # parent dir (benchmark subdir still appended).
        gated_root = paths.gated_gold_root(benchmark=benchmark)
        for candidate in (
            gated_root / "livesqlbench_sqlite_gt_kg_testcases_0528.jsonl",
            # Backwards compat: sidecar directly inside the data root.
            paths.benchmark_data_root(benchmark)
            / "livesqlbench_sqlite_gt_kg_testcases_0528.jsonl",
        ):
            if candidate.exists():
                gold_path = candidate
                break
        if gold_path and gold_path.exists():
            with gold_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    iid = r.get("instance_id")
                    sol = normalize_sol_sql(r.get("sol_sql"))
                    if iid and sol:
                        out[iid] = sol
    return out


def regrade_run(
    *,
    run_id: str,
    benchmark: str,
    run_dir: Path,
    instance_ids: Optional[List[str]] = None,
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

    # Reset the fresh-rows scratch dir so a partial regrade doesn't
    # leak stale per-instance rows from a previous pass into
    # ``eval_regraded.json``. When filtering by instance_ids, scope
    # the reset to those subdirs so unrelated instances from a prior
    # full regrade survive (and continue contributing to the report).
    fresh_rows_dir = run_dir / "regrade_rows"
    if filter_set is not None and fresh_rows_dir.exists():
        for sub in list(fresh_rows_dir.iterdir()):
            if sub.is_dir() and sub.name in filter_set:
                shutil.rmtree(sub, ignore_errors=True)
    elif filter_set is None:
        shutil.rmtree(fresh_rows_dir, ignore_errors=True)
    fresh_rows_dir.mkdir(parents=True, exist_ok=True)

    for sub in sorted(p for p in rows_dir.iterdir() if p.is_dir()):
        instance_id = sub.name
        if filter_set is not None and instance_id not in filter_set:
            report.skipped += 1
            continue
        attempt = _latest_attempt_file(sub)
        if attempt is None:
            report.skipped += 1
            continue
        attempt_data = json.loads(attempt.read_text())
        submitted_sql = attempt_data.get("submitted_sql", "")
        # The cloud worker writes the per-DB token to ``database``;
        # ``selected_database`` is only present on the source data row,
        # so fall back if the attempt file is missing it.
        selected_database = (
            attempt_data.get("selected_database")
            or attempt_data.get("database")
            or ""
        )
        attempt_data["selected_database"] = selected_database
        try:
            cascade = grader(
                instance_id=instance_id,
                submitted_sql=submitted_sql,
                task_row=attempt_data,
            )
        except Exception as exc:  # noqa: BLE001 — operational CLI
            print(
                f"  skip {instance_id}: grader raised "
                f"{type(exc).__name__}: {exc}"
            )
            report.skipped += 1
            continue
        # Build a fresh SubmissionAnnotation from the cascade.
        from bird_interact_agents.eval.annotate import (
            _eval_from_cascade,
            _skeleton_failure_classification,
            _user_sim_interaction_from_trajectory,
        )
        from bird_interact_agents.eval.annotation_schema import (
            SubmissionMetadata,
        )
        usage = attempt_data.get("usage", {}) or {}
        annotated_at = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )
        ann = SubmissionAnnotation(
            instance_id=instance_id,
            selected_database=selected_database,
            task_annotation_ref=(
                f"annotations/{benchmark}/{selected_database}/"
                f"{instance_id}.task.json"
            ),
            annotated_by="auto-regrade",
            annotated_at=annotated_at,
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
            failure_classification=_skeleton_failure_classification(cascade),
            # Pass the raw trajectory; ``_user_sim_interaction_from_trajectory``
            # defends against dict-shaped trajectories (Codex r10).
            user_sim_interaction=_user_sim_interaction_from_trajectory(
                attempt_data.get("trajectory") or [],
            ),
        )

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
    args = parser.parse_args(argv)

    instance_ids = (
        [s.strip() for s in args.instance_ids.split(",")]
        if args.instance_ids else None
    )
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.eval.annotate import (
        _user_sim_interaction_from_trajectory,
    )
    from bird_interact_agents.eval.tolerant_grader import (
        CachedLLMJudge,
        LiteLLMJudge,
        grade_submission,
        make_executor,
    )
    run_dir = paths.results_root() / "cloud" / args.run_id

    # Index the benchmark's source data once. mini_interact ships sol_sql
    # inline; livesqlbench's gated sidecar carries sol_sql under
    # ``--gold-file`` and the public data file ships it empty. Both routes
    # land under ``instance_id`` so the lookup is identical at call time.
    original_sql_by_inst = _build_original_sql_index(args.benchmark)

    # Resolve interactive-vs-one-shot ONCE for this run — drives the
    # ``user_sim_n_asks`` plumbing on each grader call so the
    # ``never_asked_user`` diagnostic fires on interactive benchmarks
    # where the agent never queried the user-sim.
    _bench = get_benchmark(args.benchmark)
    args.benchmark = _bench.name  # canonicalize so annotation refs use underscore form
    _bench_is_interactive = not _bench.one_shot
    _is_postgres = getattr(_bench, "db_backend", "sqlite") == "postgres"
    _executor = make_executor(_bench) if _is_postgres else None

    # Codex r13: build the LLM judge ONCE from the run's recorded
    # ``agent_model`` (read from ``<run_dir>/manifest.json``). The N5
    # judge uses the same model the agent did — re-asking the model
    # whether its own SQL is a defensible novel reading of the
    # ambiguous task. ``CachedLLMJudge`` persists verdicts to
    # ``llm_judge_cache.json`` and keys on the model name; changing
    # the agent's model on a resubmit naturally invalidates entries
    # (no separate ``--force-llm-judge`` flag needed).
    _manifest_path = run_dir / "manifest.json"
    if _manifest_path.exists():
        _agent_model = json.loads(_manifest_path.read_text()).get(
            "agent_model",
        )
    else:
        _agent_model = None
    if _agent_model:
        _llm_judge: Optional[CachedLLMJudge] = CachedLLMJudge(
            inner=LiteLLMJudge(model=_agent_model),
            cache_path=run_dir / "llm_judge_cache.json",
        )
    else:
        _llm_judge = None

    def _grader(*, instance_id: str, submitted_sql: str, task_row: dict, **_kw):
        # Minimal end-to-end wiring — production callers pre-build the
        # implicit annotation + audited gold rows themselves.
        from bird_interact_agents.eval.grade_in_place import (
            load_audited_gold_rows_for as _load_audited_gold_rows_for,
            load_task_annotation_or_implicit as _load_task_annotation_or_implicit,
        )
        selected_database = task_row.get("selected_database", "")
        if not selected_database:
            raise ValueError(
                f"{instance_id}: attempt JSON missing selected_database / "
                f"database key — cannot resolve sqlite path."
            )
        # For postgres, db_path is used only as a db-name carrier (executor
        # uses db_path.stem). For SQLite, root at the data root for the
        # benchmark; ``benchmark_data_root`` accepts canonical and hyphenated
        # alias forms, rejecting unknown tokens loudly.  The _template.sqlite
        # fallback supports per-task-isolated copies that use the template name.
        if _is_postgres:
            db_path = Path(selected_database)
        else:
            _db_dir = paths.benchmark_data_root(args.benchmark) / selected_database
            db_path = _db_dir / f"{selected_database}.sqlite"
            if not db_path.exists():
                _alt = _db_dir / f"{selected_database}_template.sqlite"
                if _alt.exists():
                    db_path = _alt
        ann = _load_task_annotation_or_implicit(
            instance_id=instance_id,
            selected_database=selected_database,
            benchmark=args.benchmark,
            amb_user_query=task_row.get("amb_user_query", ""),
        )
        audited = _load_audited_gold_rows_for(
            benchmark=args.benchmark, instance_id=instance_id,
        )
        # N1 requires the original gold SQL; the attempt JSON doesn't
        # carry it (it lives on the source data row / gated gold sidecar).
        # ``normalize_sol_sql`` wraps a bare string in a list so the
        # grader doesn't see ``["S", "E", "L", "E", "C", "T", ...]``
        # when the source row carries ``sol_sql`` as a single string.
        original_sol_sql = normalize_sol_sql(
            task_row.get("original_sol_sql")
            or task_row.get("sol_sql")
            or original_sql_by_inst.get(instance_id),
        )
        # Compute the user-sim signal from the attempt's trajectory so
        # the ``never_asked_user`` diagnostic fires properly on
        # interactive runs where the agent never asked. One-shot
        # benchmarks pass None so the flag stays out of miss_patterns.
        if _bench_is_interactive:
            _traj = task_row.get("trajectory") or []
            _user_sim_n_asks: Optional[int] = (
                _user_sim_interaction_from_trajectory(_traj).n_asks
            )
        else:
            _user_sim_n_asks = None
        return grade_submission(
            task_annotation=ann,
            audited_gold_rows=audited,
            original_sol_sql=original_sol_sql,
            submitted_sql=submitted_sql,
            llm_judge=_llm_judge,
            db_path=db_path,
            conn=None,
            executor=_executor,
            benchmark=_bench,
            user_sim_n_asks=_user_sim_n_asks,
        )

    report = regrade_run(
        run_id=args.run_id, benchmark=args.benchmark, run_dir=run_dir,
        instance_ids=instance_ids,
        grader=_grader,
    )
    print(f"regrade: {report.regraded} instances rewritten, "
          f"{report.skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
