"""DEV-1515: print aggregate cascade summary over the 53 annotated instances.

Walks `annotations/mini-interact/*/<inst>.submission.<run>.json` for the
two May-31 runs and emits the phase1 block (counts + deltas + rates)
plus a small per-failure-class tally.

Run after ``scripts/dev1515_convert_runs.py``.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_io import read_submission_annotation
from bird_interact_agents.eval.tolerant_grader import (
    _CASCADE_ORDER,
    enforce_monotone_cascade,
)


RUNS = (
    "20260531t1008-claudes-slayer-890419",
    "20260531t1343-claudes-slayer-b39bfc",
)
BENCHMARK = "mini-interact"

# Most-stringent → most-lenient labels for the phase1 view.
_PHASE1_LABELS = (
    ("n1", "original gold passes"),
    ("n2", "Δ + audited primary variant passes"),
    ("n3", "Δ + any audited variant matches"),
    ("n4", "Δ + correct up to tie order (row_order)"),
    ("n5", "Δ + llm_judge accepts insufficient-only (novel_reading)"),
    ("n6", "Δ + correct under numeric epsilon (numerical_precision)"),
    ("n7", "Δ + correct under trailing whitespace"),
    ("n8", "Δ + correct under column order"),
)


def _short(field: str) -> str:
    return field.split("_", 1)[0]


def _row_cascade(p: Path) -> dict[str, bool]:
    ann = read_submission_annotation(p)
    ev = ann.evaluation
    raw = {
        "n1_original_gold": ev.phase1_against_original_gold == "pass",
        "n2_audited_primary": ev.phase1_against_audited_primary == "pass",
        "n3_any_audited_variant": (
            ev.phase1_against_any_audited_variant == "pass"
        ),
        "n4_tie_order": ev.correct_up_to_tie_order,
        "n5_llm_judge": (
            ev.phase1_against_any_audited_variant == "pass"
            or ev.correct_up_to_tie_order
            or ev.novel_reading_judgment == "pass"
        ),
        "n6_numeric_epsilon": ev.correct_under_numeric_epsilon,
        "n7_trailing_whitespace": ev.correct_under_trailing_whitespace,
        "n8_column_order": ev.correct_under_column_order,
    }
    return enforce_monotone_cascade(raw)


def _gather_files() -> list[Path]:
    annroot = paths.annotations_root() / BENCHMARK
    out: list[Path] = []
    for run in RUNS:
        for p in sorted(annroot.glob(f"*/*.submission.{run}.json")):
            out.append(p)
    return out


def _is_audit_unchanged(instance_id: str, audit_rows: dict) -> bool:
    row = audit_rows.get(instance_id)
    if row is None:
        return False
    orig = [" ".join((s or "").split()) for s in (row.get("original_sol_sql") or [])]
    aud = [" ".join((s or "").split()) for s in (row.get("audited_sol_sql") or [])]
    return orig == aud


def _aggregate(paths_: list[Path], audit_rows: dict) -> dict:
    """Split U (audit unchanged) vs C (audit changed) at each cascade tier."""
    counts_u = {_short(f): 0 for f in _CASCADE_ORDER}
    counts_c = {_short(f): 0 for f in _CASCADE_ORDER}
    n_u = n_c = 0
    per_class: Counter[str] = Counter()
    per_db_n1: defaultdict = defaultdict(lambda: [0, 0])  # [n1_pass, total]
    for p in paths_:
        ann = read_submission_annotation(p)
        verdicts = _row_cascade(p)
        unchanged = _is_audit_unchanged(ann.instance_id, audit_rows)
        if unchanged:
            n_u += 1
            target = counts_u
        else:
            n_c += 1
            target = counts_c
        for f, v in verdicts.items():
            if v:
                target[_short(f)] += 1
        per_class[ann.failure_classification.primary] += 1
        per_db_n1[ann.selected_database][1] += 1
        if verdicts["n1_original_gold"]:
            per_db_n1[ann.selected_database][0] += 1
    return {
        "n_u": n_u,
        "n_c": n_c,
        "counts_u": counts_u,
        "counts_c": counts_c,
        "failure_classes": dict(per_class),
        "per_db_n1": dict(per_db_n1),
    }


def _load_audit_rows() -> dict:
    p = paths.audited_gold_root() / "mini_interact_audited.jsonl"
    rows = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows[d["instance_id"]] = d
    return rows


def main() -> None:
    files = _gather_files()
    audit_rows = _load_audit_rows()
    block = _aggregate(files, audit_rows)
    nu, nc, n = block["n_u"], block["n_c"], block["n_u"] + block["n_c"]

    print("=" * 90)
    print(f"DEV-1515 cascade summary over {n} (instance, run) pairs")
    print(f"  runs: {', '.join(RUNS)}")
    print(f"  U-instances (audit unchanged): {nu}   "
          f"C-instances (audit changed): {nc}")
    print("=" * 90)
    print()
    print("phase1 cascade  — U(audit-unchanged) vs C(audit-changed) split")
    print(f"{'tier':52s}   {'U':>6s}    {'C':>6s}    {'tot':>6s}")
    print("-" * 90)
    prev_total = None
    for key, label in _PHASE1_LABELS:
        u = block["counts_u"][key]
        c = block["counts_c"][key]
        tot = u + c
        delta = "" if prev_total is None else (
            f"  (+{tot - prev_total})" if tot - prev_total > 0
            else f"  (+0)"
        )
        u_str = f"{u}/{nu}"
        c_str = f"{c}/{nc}"
        tot_str = f"{tot}/{n}"
        rate = (tot / n * 100.0) if n else 0.0
        print(f"  {label:50s}   {u_str:>6s}    {c_str:>6s}    "
              f"{tot_str:>6s}  {rate:>5.1f}%{delta}")
        prev_total = tot
    print()
    print("Decomposition of N3 passes:")
    n1_u = block["counts_u"]["n1"]
    n1_c = block["counts_c"]["n1"]
    n3_u = block["counts_u"]["n3"]
    n3_c = block["counts_c"]["n3"]
    print(f"  U-pass at strict N1 (gold was right, agent right):       {n1_u}")
    print(f"  C-cosmetic-pass at strict N1 (audit cosmetic, agent ok): {n1_c}")
    print(f"  C-fix-pass at N2/N3 (audit fixed buggy gold):            {n3_c - n1_c}")
    print(f"  U-fail (gold was right, agent wrong):                    {nu - n1_u}")
    print(f"  C-fail (audit didn't help, agent wrong):                 {nc - n3_c}")
    print()
    print("Per-DB N1 (original-gold pass) tally:")
    print(f"  {'database':36s}  {'pass':>5s} / {'tot':<3s}  {'%':>5s}")
    print("-" * 60)
    for db, (passed, total) in sorted(block["per_db_n1"].items()):
        rate = (passed / total * 100.0) if total else 0.0
        print(f"  {db:36s}  {passed:>5d} / {total:<3d}  {rate:>4.1f}%")
    print()
    print("Failure-classification primary tally:")
    for cls, c in sorted(block["failure_classes"].items(), key=lambda kv: -kv[1]):
        print(f"  {cls:36s}  {c}")


if __name__ == "__main__":
    main()
