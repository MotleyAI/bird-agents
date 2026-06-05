"""DEV-1515/1533: print aggregate cascade summary from runs/ golden store.

Reads the latest run per instance for each benchmark and emits:
- The partition cascade (each task maps to exactly one tier)
- Cumulative counts and rates
- Per-failure-class tally

Usage::

    uv run python scripts/dev1515_cascade_summary.py [--benchmark BENCHMARK]

``BENCHMARK`` defaults to ``mini-interact``.

Run after annotated runs have been fetched / grade_and_write has populated
``runs/<benchmark>/``.
"""
from __future__ import annotations

import argparse

from bird_interact_agents.eval.cascading_report import (
    aggregate_cascading_latest,
    _PARTITION_DISPLAY,
    _PARTITION_LABELS,
    _PARTITION_IS_PASS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", default="mini-interact",
        help="Benchmark canonical name (default: mini-interact)",
    )
    args = parser.parse_args()
    benchmark: str = args.benchmark

    block = aggregate_cascading_latest(benchmark)
    n = block["n_dual_eval_tasks"]

    print("=" * 90)
    print(f"Cascade summary — benchmark={benchmark!r}  n={n} (latest run per instance)")
    print("=" * 90)

    if n == 0:
        print("No run annotations found in runs/. Nothing to report.")
        return

    # Partition table
    print()
    print("Partition (each task maps to exactly ONE line):")
    print(f"  {'tier':<54}  {'count':>8}  {'%':>6}  {'cumsum':>8}  {'cum%':>6}")
    print("  " + "-" * 90)
    p = block.get("cascading_partition", {})
    for label in _PARTITION_LABELS:
        if label not in p.get("tiers", {}):
            continue
        tier = p["tiers"][label]
        count = tier["count"]
        rate = tier["rate"] * 100
        cumsum = tier["cumsum"]
        cumsum_rate = tier["cumsum_rate"] * 100
        display = _PARTITION_DISPLAY[label]
        is_pass = _PARTITION_IS_PASS[label]
        tag = "" if is_pass else "  [diagnostic]" if label == "l2_wrong_original" else "  [fail]"
        print(
            f"  {display:<54}  {count:>8}  {rate:>5.1f}%  {cumsum:>8}  {cumsum_rate:>5.1f}%{tag}"
        )

    print()
    counts = block["counts"]
    print("Legacy cumulative N1..N9 (for backward compat):")
    _PHASE1_LABELS = (
        ("n1", "original gold passes"),
        ("n2", "audited primary passes  [n2 - n1 = audited adds over original]"),
        ("n3", "any audited variant matches"),
        ("n4", "correct up to tie order"),
        ("n5", "llm_judge (insufficient tasks)"),
        ("n6", "numeric epsilon"),
        ("n7", "trailing whitespace"),
        ("n8", "column order"),
        ("n9", "case fold"),
    )
    deltas = block["deltas"]
    prev = None
    for key, label in _PHASE1_LABELS:
        tot = counts[key]
        rate = (tot / n * 100.0) if n else 0.0
        delta = f"  (+{deltas[key]})" if prev is not None else ""
        print(f"  {label:<56}  {tot}/{n}  {rate:>5.1f}%{delta}")
        prev = tot

    print()
    print(f"Pass rate (L1 + L3-L10, excl. L2 diagnostic): "
          f"{p.get('pass_count', 0)}/{n} = "
          f"{p.get('pass_rate', 0.0)*100:.1f}%")


if __name__ == "__main__":
    main()
