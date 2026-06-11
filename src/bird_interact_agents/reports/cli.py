"""``bird-interact-cloud submission`` subcommand entry point."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.reports import coverage as _coverage
from bird_interact_agents.reports import budget as _budget
from bird_interact_agents.reports.converter import (
    build_submission_row,
    cross_check_results_db_sql,
)
from bird_interact_agents.reports.leakage import count_leakage
from bird_interact_agents.reports.output import (
    ManifestPlan,
    write_submission,
)
from bird_interact_agents.reports.selection import load_selection
from bird_interact_agents.reports.sources import resolve_sources


_SUPPORTED_BENCHMARKS = ("bird-interact-lite-exp", "bird-interact-full", "mini-interact")
_BENCHMARK_TO_SPLIT = {
    "bird-interact-lite-exp": "lite",
    "bird-interact-full": "full",
    "mini-interact": "mini-interact",
}


def _slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def _selection_tag(selection_entries: list[tuple[str, str]]) -> str:
    canonical = json.dumps(
        sorted(selection_entries), separators=(",", ":")
    ).encode()
    return "selection-" + hashlib.sha256(canonical).hexdigest()[:10]


def _read_patience_for_instance(
    instance_dir: Path, run_id: str
) -> tuple[int | None, str]:
    """Look for a ``patience`` field in the per-instance submission-
    annotation sidecar. Returns (patience, source)."""
    p = instance_dir / f"{run_id}.json"
    if not p.is_file():
        return (None, "default")
    try:
        obj = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return (None, "default")
    # The harness writes patience inside `submission` for some agents;
    # fall through to `None` if the field is absent.
    pat = obj.get("patience")
    if pat is None and isinstance(obj.get("submission"), dict):
        pat = obj["submission"].get("patience")
    if pat is None:
        return (None, "default")
    return (int(pat), f"runs:{p.name}")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "submission",
        help="Generate a BIRD-INTERACT-1.0 a-Interact submission directory.",
    )
    sp.add_argument("--team-name", required=True)
    sp.add_argument("--method-name", required=True)
    sp.add_argument(
        "--benchmark",
        required=True,
        choices=_SUPPORTED_BENCHMARKS,
    )
    sel = sp.add_mutually_exclusive_group(required=True)
    sel.add_argument("--run-id")
    sel.add_argument("--selection", type=Path)
    sp.add_argument("--allow-partial", action="store_true")
    sp.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Fallback patience value when the per-instance json lacks one.",
    )
    sp.add_argument("--report-tag")
    sp.add_argument("--out", type=Path)
    sp.add_argument(
        "--no-thinking",
        dest="include_thinking",
        action="store_false",
        default=True,
        help="Strip thinking blocks from the `response` field of every step.",
    )
    sp.add_argument(
        "--check-leakage",
        action="store_true",
        help="Add per-instance gold-SQL substring count to manifest.leakage_check.",
    )


def run_submission(args: argparse.Namespace) -> int:
    benchmark: str = args.benchmark
    setting = "a-Interact"
    split = _BENCHMARK_TO_SPLIT[benchmark]

    # ---- Selection ------------------------------------------------
    if args.selection:
        selection = load_selection(args.selection)
        selection_mode = "selection-file"
        source_run_ids = sorted({rid for _, rid in selection})
        default_tag = _selection_tag(selection)
    else:
        # --run-id path: list every instance in that run's results.db.
        results_db = (
            paths.results_root()
            / benchmark
            / "cloud"
            / args.run_id
            / "results.db"
        )
        if not results_db.is_file():
            sys.stderr.write(
                f"error: results.db not found at {results_db}\n"
            )
            return 2
        import sqlite3

        con = sqlite3.connect(results_db)
        con.row_factory = sqlite3.Row
        try:
            iids = [
                row["instance_id"]
                for row in con.execute(
                    "SELECT instance_id FROM task_results WHERE run_id = ?",
                    (args.run_id,),
                )
            ]
        finally:
            con.close()
        selection = [(iid, args.run_id) for iid in iids]
        selection_mode = "run-id"
        source_run_ids = [args.run_id]
        default_tag = args.run_id

    tag = args.report_tag or default_tag

    # ---- Resolve sources (errors out on missing/stub trajectories)
    sources = resolve_sources(selection=selection, benchmark=benchmark)

    # ---- Coverage check ------------------------------------------
    try:
        _coverage.assert_coverage_ok(
            benchmark=benchmark,
            present_instance_ids=set(sources.keys()),
            allow_partial=args.allow_partial,
        )
    except (
        _coverage.IncompleteCoverageError,
        _coverage.UnknownInstanceError,
    ) as e:
        sys.stderr.write(f"error: {e}\n")
        raise SystemExit(2) from e

    # ---- Build rows ---------------------------------------------
    rows = []
    instance_manifest_entries: list[dict] = []
    patience_resolution: list[dict] = []
    warnings_by_instance: list[dict] = []
    leakage_entries: list[dict] = []

    for inst_id, src in sources.items():
        instance_dir = src.trajectory_path.parent
        per_inst_patience, source_label = _read_patience_for_instance(
            instance_dir, src.run_id
        )
        patience = per_inst_patience if per_inst_patience is not None else args.patience
        patience_resolution.append(
            {
                "instance_id": inst_id,
                "patience": patience,
                "source": source_label if per_inst_patience is not None else "default",
            }
        )

        task_data = _budget.lookup_task_data(benchmark, inst_id)
        row = build_submission_row(
            trajectory_obj=src.trajectory_obj,
            framework=src.framework,
            agent_model=src.agent_model,
            user_sim_model=src.user_sim_model,
            task_data=task_data,
            patience=patience,
            include_thinking=args.include_thinking,
        )
        rows.append(row)

        instance_manifest_entries.append(
            {
                "instance_id": inst_id,
                "run_id": src.run_id,
                "framework": src.framework,
                "agent_model": src.agent_model,
                "user_sim_model": src.user_sim_model,
                "trajectory_path": str(src.trajectory_path),
                "results_db_path": str(src.results_db_path),
                "phase1_passed": bool(src.task_results_row.get("phase1_passed")),
                "phase2_passed": bool(src.task_results_row.get("phase2_passed")),
                "error": src.task_results_row.get("error"),
            }
        )

        # Cross-check warning vs results.db.submitted_sql (last submit only).
        db_sql = src.task_results_row.get("submitted_sql") or ""
        warns = cross_check_results_db_sql(
            row=row, results_db_submitted_sql=db_sql
        )
        if warns:
            warnings_by_instance.append({"instance_id": inst_id, "warnings": warns})

        # --check-leakage (optional)
        if args.check_leakage:
            prompts = [e.prompt for e in row.prompt_flow]
            gold = src.task_results_row.get("ground_truth_sql") or src.trajectory_obj.get(
                "ground_truth_sql"
            )
            n = count_leakage(prompts=prompts, ground_truth_sql=gold)
            leakage_entries.append({"instance_id": inst_id, "leak_count": n})

    # ---- Pick output directory ----------------------------------
    team_slug = _slugify(args.team_name)
    method_slug = _slugify(args.method_name)
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = (
            paths.reports_root()
            / benchmark
            / setting
            / f"{team_slug}__{method_slug}__{tag}"
        )

    leakage_block = (
        {"min_substring": 12, "per_instance": leakage_entries}
        if args.check_leakage
        else None
    )
    plan = ManifestPlan(
        benchmark=benchmark,
        setting=setting,
        split=split,
        team=args.team_name,
        method=args.method_name,
        tag=tag,
        selection_mode=selection_mode,
        source_run_ids=source_run_ids,
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        instances=instance_manifest_entries,
        patience_resolution=patience_resolution,
        leakage_check=leakage_block,
        warnings_by_instance=warnings_by_instance,
    )
    write_submission(rows=rows, plan=plan, out_dir=out_dir)
    return 0
