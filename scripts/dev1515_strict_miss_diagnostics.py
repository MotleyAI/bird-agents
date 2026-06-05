"""DEV-1515/1533: strict-miss diagnostics from runs/ golden store.

Reads the latest run per (benchmark, instance_id) from runs/<benchmark>/,
filters to cascade-fail submissions, and prints:

1. Per-instance breakdown (rowset shape, column shape, SQL signals, flags).
2. Per-flag tally (flags are NOT mutually exclusive).
3. Instance × flag matrix.

Usage::

    uv run python scripts/dev1515_strict_miss_diagnostics.py [--benchmark BENCHMARK]

``BENCHMARK`` may be specified multiple times. Defaults to
``mini-interact`` and ``livesqlbench``.
"""
from __future__ import annotations

import argparse
from collections import Counter

from bird_interact_agents.eval.annotation_io import latest_run_per_instance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", action="append", dest="benchmarks",
        default=None,
        help="Benchmark name (may be repeated; default: mini-interact livesqlbench)",
    )
    args = parser.parse_args()
    benchmarks: list[str] = args.benchmarks or ["mini-interact", "livesqlbench"]

    submissions: list[tuple[str, dict, dict]] = []
    for bench in benchmarks:
        latest = latest_run_per_instance(benchmark=bench)
        for (db, iid), (_run_id, ann) in sorted(latest.items()):
            ev = ann.evaluation
            md = ev.miss_diagnostics
            if md is None:
                continue
            # Disambiguate same-iid across benchmarks.
            label = f"{iid}@{bench}" if len(benchmarks) > 1 else iid
            submissions.append((label, ann.model_dump(), md.model_dump()))

    submissions.sort(key=lambda r: r[0])

    print("=" * 110)
    print(
        f"Strict-miss diagnostics — "
        f"{len(submissions)} instances (cascade-fail), "
        f"benchmarks: {', '.join(benchmarks)}"
    )
    print("=" * 110)

    if not submissions:
        print("No cascade-fail instances found. Nothing to diagnose.")
        return

    print()
    print("Per-instance breakdown")
    print(
        f"  {'instance':<42} {'rows(a/g)':>10}  {'cols(a/g)':>10}  "
        f"{'rel':<20}  {'tbls':<5}  {'agg(a/g)':<10}  flags"
    )
    print("  " + "-" * 108)
    for iid, _ann, md in submissions:
        rows = f"{md['agent_row_count']}/{md['best_variant_row_count']}"
        cols = f"{md.get('agent_column_count', '?')}/{md.get('best_variant_column_count', '?')}"
        rel = md.get("rowset_relation_to_best", "?")
        tbls = (
            "match" if md.get("table_set_match") is True
            else "diff" if md.get("table_set_match") is False
            else "?"
        )
        a_agg = md.get("agent_has_aggregate")
        b_agg = md.get("best_variant_has_aggregate")
        agg = f"{_b(a_agg)}/{_b(b_agg)}"
        flags = ", ".join(md.get("miss_patterns", [])) or "—"
        print(
            f"  {iid:<42} {rows:>10}  {cols:>10}  {rel:<20}  {tbls:<5}  "
            f"{agg:<10}  {flags}"
        )

    counter: Counter[str] = Counter()
    interactive_count = 0
    for _iid, _ann, md in submissions:
        if md.get("user_sim_n_asks") is not None:
            interactive_count += 1
        for f in md.get("miss_patterns", []):
            counter[f] += 1

    print()
    print("Per-flag tally (each instance can carry multiple flags):")
    for flag, count in counter.most_common():
        note = (
            f"   (interactive-only — {interactive_count} of "
            f"{len(submissions)} cascade-fail rows are interactive)"
        ) if flag == "never_asked_user" else ""
        print(f"  {flag:<32} {count}{note}")

    all_flags = sorted(counter)
    if all_flags:
        print()
        print("Instance x flag matrix (x = flag fired):")
        header = "  " + " " * 42 + "  ".join(
            f"{f[:10]:>10}" for f in all_flags
        )
        print(header)
        for iid, _ann, md in submissions:
            cells = "  ".join(
                f"{'x':>10}" if f in md.get("miss_patterns", [])
                else f"{'.':>10}"
                for f in all_flags
            )
            print(f"  {iid:<42}{cells}")


def _b(v: object) -> str:
    if v is True:
        return "T"
    if v is False:
        return "F"
    return "?"


if __name__ == "__main__":
    main()
