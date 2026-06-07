#!/usr/bin/env python3
"""DEV-1541 backfill: regenerate missing or errored autopsies on the
per-(instance, run) annotation store.

Usage
-----

    uv run python scripts/regenerate_autopsies.py \
        --benchmark <name> \
        [--run-id <id>] [--instance-ids a,b,c] \
        [--model anthropic/...] \
        [--dry-run]

Exit status
-----------

* ``0`` — no IO errors. Autopsy-side errors (e.g. context_overflow)
  are recorded in the annotation but do not flip the exit code, so a
  cron driver doesn't loop on transient model failures.
* ``1`` — at least one IO error (annotation unreadable, trajectory
  missing, write failed).

The library implementation lives at
``bird_interact_agents.eval.regenerate_autopsies``; this file is a
thin argparse wrapper so the standard ``uv run python scripts/...``
entry point keeps working.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate missing / errored autopsies (DEV-1541).",
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--instance-ids",
        default=None,
        help="Comma-separated list of instance_ids to limit work to.",
    )
    parser.add_argument(
        "--model", default="anthropic/claude-haiku-4-5-20251001",
        help="Model string for the autopsy LLM call (default: haiku-4-5).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the work-list without invoking the autopsy LLM.",
    )
    parser.add_argument(
        "--runs-root", default=None,
        help="Override the runs/ root directory. "
             "Default: $BIRD_RUNS_ROOT or <main-checkout>/runs.",
    )
    return parser.parse_args(argv)


def _resolve_runs_root(arg: Optional[str]) -> Path:
    """DEV-1541 r2 (CodeRabbit): always defer to the project's
    ``paths.runs_root()`` helper for the default. The earlier manual
    reconstruction (``results_root().parent / "runs"``) violated the
    project rule "every gitignored input/output lives in the MAIN
    checkout, NEVER in the worktree" — the only correct path is the
    helper, which anchors at the main checkout via git's common dir."""
    if arg is not None:
        return Path(arg).expanduser()
    env = os.environ.get("BIRD_RUNS_ROOT")
    if env:
        return Path(env).expanduser()
    from bird_interact_agents import paths
    return paths.runs_root()


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    from bird_interact_agents.eval.regenerate_autopsies import regenerate

    instance_ids = (
        {s.strip() for s in args.instance_ids.split(",") if s.strip()}
        if args.instance_ids else None
    )
    runs_root = _resolve_runs_root(args.runs_root)
    report = regenerate(
        runs_root=runs_root,
        benchmark=args.benchmark,
        model=args.model,
        dry_run=args.dry_run,
        run_id=args.run_id,
        instance_ids=instance_ids,
    )
    print(
        f"work_items={report.work_items} "
        f"regenerated={report.regenerated} "
        f"autopsy_errors={report.autopsy_errors} "
        f"io_errors={report.io_errors}"
    )
    return report.exit_code()


if __name__ == "__main__":
    sys.exit(main())
