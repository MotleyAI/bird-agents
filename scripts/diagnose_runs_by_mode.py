"""Diagnose a benchmark x query-mode by picking the latest run per task.

Walks ``runs/<benchmark>/<db>/<inst>/<run_id>.json``, filters to run_ids whose
slug matches the requested query-mode (``slayer`` -> ``-slayer-``, ``raw`` ->
``-raw-``), then keeps the latest annotation per ``(db, instance_id)`` by
``annotated_at``. Reports the same cascade + autopsy view used at fetch time.

Usage::

    uv run python scripts/diagnose_runs_by_mode.py \
        --benchmark livesqlbench-base-lite-sqlite --mode slayer

``--mode`` is the query-mode (slayer / raw / any). The agent-mode
(a-interact / one-shot) is NOT filtered — pass ``--agent-mode`` if you want
to restrict it via the run manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_io import read_submission_annotation
from bird_interact_agents.eval.cascading_report import _aggregate_from_annotations


_QUERY_MODE_SLUGS = {
    "slayer": "-slayer-",
    "raw": "-raw-",
    "any": None,
}


def _agent_mode_from_manifest(benchmark: str, run_id: str) -> str | None:
    manifest_fp = (
        paths.results_root() / benchmark / "cloud" / run_id / "manifest.json"
    )
    if not manifest_fp.exists():
        return None
    try:
        return json.loads(manifest_fp.read_text()).get("mode")
    except (OSError, ValueError):
        return None


def _walk_latest(
    benchmark: str,
    query_mode: str,
    agent_mode: str | None,
) -> dict[tuple[str, str], tuple[str, object, Path]]:
    """Latest annotation per (db, instance_id) filtered by mode(s)."""
    runs_root = paths.runs_root() / benchmark
    if not runs_root.exists():
        return {}
    slug_needle = _QUERY_MODE_SLUGS[query_mode]
    agent_mode_cache: dict[str, str | None] = {}
    best: dict[tuple[str, str], tuple[str, str, object, Path]] = {}
    for fp in sorted(runs_root.rglob("*.json")):
        if fp.name.endswith(".trajectory.json"):
            continue
        try:
            rel = fp.relative_to(runs_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        db, iid, run_file = parts
        run_id = run_file[:-5]
        if slug_needle is not None and slug_needle not in run_id:
            continue
        if agent_mode is not None:
            cached = agent_mode_cache.get(run_id, "__miss__")
            if cached == "__miss__":
                cached = _agent_mode_from_manifest(benchmark, run_id)
                agent_mode_cache[run_id] = cached
            if cached != agent_mode:
                continue
        try:
            ann = read_submission_annotation(fp)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] skipping unreadable {fp}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        annotated_at = getattr(ann, "annotated_at", "") or ""
        key = (db, iid)
        cur = best.get(key)
        if cur is None or annotated_at > cur[1]:
            best[key] = (run_id, annotated_at, ann, fp)
    return {k: (v[0], v[2], v[3]) for k, v in best.items()}


def _autopsy_pattern(ann) -> str | None:
    auto = getattr(ann, "autopsy", None)
    if auto is None:
        return None
    if hasattr(auto, "model_dump"):
        auto = auto.model_dump()
    if not isinstance(auto, dict):
        return None
    analysis = auto.get("analysis") or {}
    return analysis.get("pattern") or "unclassified"


def _verdict(ann) -> str | None:
    ev = getattr(ann, "evaluation", None)
    if ev is None:
        return None
    if hasattr(ev, "verdict"):
        return ev.verdict
    if isinstance(ev, dict):
        return ev.get("verdict")
    return None


def report(
    benchmark: str,
    query_mode: str,
    agent_mode: str | None,
    instance_filter: set[str] | None,
) -> int:
    latest = _walk_latest(benchmark, query_mode, agent_mode)
    if not latest:
        print(
            f"No annotations found under runs/{benchmark}/ for "
            f"query_mode={query_mode!r} agent_mode={agent_mode!r}.",
            file=sys.stderr,
        )
        return 1

    pairs = []
    used_run_ids = Counter()
    passes: list[str] = []
    fails: list[tuple[str, str]] = []
    eval_failed: list[str] = []
    autopsy_patterns = Counter()
    per_db_pass = Counter()
    per_db_total = Counter()

    for (db, iid), (run_id, ann, fp) in latest.items():
        if instance_filter is not None and iid not in instance_filter:
            continue
        pairs.append((fp, ann))
        used_run_ids[run_id] += 1
        verdict = _verdict(ann) or "?"
        per_db_total[db] += 1
        if verdict == "correct":
            passes.append(iid)
            per_db_pass[db] += 1
        elif verdict == "eval_failed":
            eval_failed.append(iid)
        else:
            fails.append((iid, verdict))
            pat = _autopsy_pattern(ann)
            if pat:
                autopsy_patterns[pat] += 1

    block = _aggregate_from_annotations(pairs)
    cp = block

    print(f"=== {benchmark} | query_mode={query_mode}"
          f"{f' agent_mode={agent_mode}' if agent_mode else ''}"
          f" (latest run per task) ===")
    print(f"  n_dual_eval_tasks: {cp.get('n_dual_eval_tasks')}")
    counts = cp.get("counts", {})
    rates = cp.get("rates", {})
    print("  cascade N1..N9:")
    for k in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"):
        c = counts.get(k, 0)
        r = rates.get(k, 0.0)
        print(f"    {k}: {c:>3}  ({r * 100:5.1f}%)")
    print("  partition (mutually-exclusive tiers):")
    for tier, info in cp.get("cascading_partition", {}).get("tiers", {}).items():
        cnt = info.get("count", 0)
        if cnt:
            print(f"    {tier}: count={cnt:>3}  cumsum={info.get('cumsum', 0)}")

    print()
    print("  pass/fail summary:")
    print(f"    pass:        {len(passes):>3}")
    print(f"    fail (real): {len(fails):>3}")
    print(f"    eval_failed: {len(eval_failed):>3}")
    if eval_failed:
        print(f"    eval_failed ids: {eval_failed}")

    if autopsy_patterns:
        print()
        total_auto = sum(autopsy_patterns.values())
        print(f"  autopsy themes ({total_auto} fails with autopsy):")
        for pat, n in autopsy_patterns.most_common():
            print(f"    {pat:40s} : {n}")

    print()
    print("  per-db pass rate:")
    for db in sorted(per_db_total.keys()):
        print(f"    {db:30s} : {per_db_pass[db]:>3} / {per_db_total[db]:>3}")

    print()
    print("  source runs (count of tasks contributed):")
    for run_id, n in used_run_ids.most_common():
        print(f"    {run_id}  -> {n}")

    return 0


def _instance_filter(arg: str | None) -> set[str] | None:
    if not arg:
        return None
    if arg.startswith("@"):
        fp = Path(arg[1:])
        return {
            s.strip() for s in fp.read_text().split() if s.strip()
        }
    return {s.strip() for s in arg.split(",") if s.strip()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True)
    p.add_argument(
        "--mode", choices=("slayer", "raw", "any"), default="slayer",
        help="Query-mode filter (slayer / raw / any). Matches against the "
             "run_id slug.",
    )
    p.add_argument(
        "--agent-mode", default=None, choices=(None, "a-interact", "one-shot",
                                                "c-interact"),
        help="Optional agent-mode filter (reads each run's manifest.json).",
    )
    p.add_argument(
        "--instance-ids", default=None,
        help="Optional restriction to this comma-separated id set; or "
             "@path/to/file to read newline/space-separated ids.",
    )
    ns = p.parse_args(argv)
    return report(
        benchmark=ns.benchmark,
        query_mode=ns.mode,
        agent_mode=ns.agent_mode,
        instance_filter=_instance_filter(ns.instance_ids),
    )


if __name__ == "__main__":
    sys.exit(main())
