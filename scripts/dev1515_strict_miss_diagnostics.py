"""DEV-1515 session-4 — strict-miss diagnostics summary.

Walks the latest-run annotation per instance (mini-interact + livesqlbench),
filters to cascade-fail submissions (those whose grader produced a
``miss_diagnostics`` block), and prints:

1. Per-instance row showing rowset shape, column shape, table-set
   match, group_by signals, and the full flag list.
2. Per-flag tally — flags are NOT mutually exclusive so the totals
   overlap.
3. An instance × flag matrix so co-occurring flags are easy to scan.

Run:

    env -u SSH_AUTH_SOCK uv run python scripts/dev1515_strict_miss_diagnostics.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from bird_interact_agents import paths

# Walk per-benchmark trees. Same instance_id can exist in BOTH
# mini-interact and livesqlbench (e.g. ``credit_4`` lives in both
# benchmark sets), so dedup key MUST be ``(benchmark, instance_id)``,
# not just ``instance_id`` — otherwise the latest-wins collapse
# silently hides mini-interact data when livesqlbench has the same
# instance_id with a later run_id.
ROOTS = {
    "mini_interact": (
        paths.annotations_root() / "mini_interact",
    ),
    "livesqlbench": (
        paths.annotations_root() / "livesqlbench",
    ),
}

SUB_RE = re.compile(r"^(.+)\.submission\.(.+)\.json$")


def _walk_latest_submissions() -> dict[tuple[str, str], dict]:
    """Return {(benchmark, instance_id): latest submission JSON}.
    Latest = lex-max of the run_id suffix; works because run_ids are
    timestamped ``YYYYMMDDtHHMM…``."""
    by_key: dict[tuple[str, str], tuple[str, dict]] = {}
    for bench, roots in ROOTS.items():
        for root in roots:
            if not root.exists():
                continue
            for p in root.glob("*/*.submission.*.json"):
                m = SUB_RE.match(p.name)
                if not m:
                    continue
                iid, run_id = m.group(1), m.group(2)
                key = (bench, iid)
                cur = by_key.get(key)
                if cur is None or run_id > cur[0]:
                    by_key[key] = (run_id, json.loads(p.read_text()))
    return {key: v for key, (_run, v) in by_key.items()}


def main() -> None:
    submissions = _walk_latest_submissions()
    strict_misses: list[tuple[str, dict, dict]] = []
    for (bench, iid), ann in submissions.items():
        ev = ann.get("evaluation", {})
        md = ev.get("miss_diagnostics")
        if md is None:
            continue
        # Display label combines bench + iid so the per-instance table
        # disambiguates same-iid-different-benchmark rows
        # (e.g. `credit_4@mini_interact` vs `credit_4@livesqlbench`).
        label = (
            iid if bench == "mini_interact" and iid not in {
                s.split("@")[0] for s, _, _ in strict_misses
            }
            else f"{iid}@{bench}"
        )
        strict_misses.append((label, ann, md))
    strict_misses.sort()

    print("=" * 110)
    print(
        f"DEV-1515 strict-miss diagnostics — "
        f"{len(strict_misses)} instances (cascade-fail)"
    )
    print("=" * 110)
    print()

    if not strict_misses:
        print("No cascade-fail instances on disk. Nothing to diagnose.")
        return

    # ---- Per-instance table ----------------------------------------------
    print("Per-instance breakdown")
    print(
        f"  {'instance':<42} {'rows(a/g)':>10}  {'cols(a/g)':>10}  "
        f"{'rel':<20}  {'tbls':<5}  {'agg(a/g)':<10}  flags"
    )
    print("  " + "-" * 108)
    for iid, _ann, md in strict_misses:
        rows = f"{md['agent_row_count']}/{md['best_variant_row_count']}"
        cols = f"{md['agent_column_count']}/{md['best_variant_column_count']}"
        rel = md["rowset_relation_to_best"]
        tbls = (
            "match" if md.get("table_set_match") is True
            else "diff" if md.get("table_set_match") is False
            else "?"
        )
        a_agg = md.get("agent_has_aggregate")
        b_agg = md.get("best_variant_has_aggregate")
        agg = f"{_b(a_agg)}/{_b(b_agg)}"
        flags = ", ".join(md["miss_patterns"]) or "—"
        print(
            f"  {iid:<42} {rows:>10}  {cols:>10}  {rel:<20}  {tbls:<5}  "
            f"{agg:<10}  {flags}"
        )

    # ---- Per-flag tally --------------------------------------------------
    counter: Counter[str] = Counter()
    interactive_count = 0
    for _iid, _ann, md in strict_misses:
        if md.get("user_sim_n_asks") is not None:
            interactive_count += 1
        for f in md["miss_patterns"]:
            counter[f] += 1

    print()
    print("Per-flag tally (each instance can carry multiple flags):")
    for flag, count in counter.most_common():
        if flag == "never_asked_user":
            note = (
                f"   (interactive-only — {interactive_count} of "
                f"{len(strict_misses)} cascade-fail rows are interactive)"
            )
        else:
            note = ""
        print(f"  {flag:<32} {count}{note}")

    # ---- Instance × flag matrix -----------------------------------------
    all_flags = sorted(counter)
    if all_flags:
        print()
        print("Instance × flag matrix (x = flag fired):")
        header = "  " + " " * 42 + "  ".join(
            f"{f[:10]:>10}" for f in all_flags
        )
        print(header)
        for iid, _ann, md in strict_misses:
            cells = "  ".join(
                f"{'x':>10}" if f in md["miss_patterns"]
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
