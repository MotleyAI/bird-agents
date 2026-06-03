"""DEV-1515: print aggregate cascade summary over the 53 annotated instances.

Walks `annotations/mini_interact/*/<inst>.submission.<run>.json` for the
two May-31 runs and emits the phase1 monotone cascade (each tier is the
cumulative pass count up to and including that tier), plus a small
per-failure-class tally.

Run after ``scripts/dev1515_convert_runs.py``.
"""
from __future__ import annotations

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
BENCHMARK = "mini_interact"

# Most-stringent → most-lenient labels for the phase1 view.
_PHASE1_LABELS = (
    ("n1", "original gold passes"),
    ("n2", "audited primary variant passes"),
    ("n3", "any audited variant matches"),
    ("n4", "correct up to tie order (row_order)"),
    ("n5", "llm_judge accepts insufficient-only (novel_reading)"),
    ("n6", "correct under numeric epsilon (numerical_precision)"),
    ("n7", "correct under trailing whitespace"),
    ("n8", "correct under column order"),
    ("n9", "correct under case fold (case_sensitivity)"),
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
        "n9_case_fold": ev.correct_under_case_fold,
    }
    return enforce_monotone_cascade(raw)


def _gather_files() -> list[Path]:
    annroot = paths.annotations_root() / BENCHMARK
    out: list[Path] = []
    for run in RUNS:
        for p in sorted(annroot.glob(f"*/*.submission.{run}.json")):
            out.append(p)
    return out


def _aggregate(paths_: list[Path]) -> dict:
    counts = {_short(f): 0 for f in _CASCADE_ORDER}
    n = 0
    per_class: Counter[str] = Counter()
    per_db_n1: defaultdict = defaultdict(lambda: [0, 0])  # [n1_pass, total]
    for p in paths_:
        ann = read_submission_annotation(p)
        verdicts = _row_cascade(p)
        n += 1
        for f, v in verdicts.items():
            if v:
                counts[_short(f)] += 1
        per_class[ann.failure_classification.primary] += 1
        per_db_n1[ann.selected_database][1] += 1
        if verdicts["n1_original_gold"]:
            per_db_n1[ann.selected_database][0] += 1
    return {
        "n": n,
        "counts": counts,
        "failure_classes": dict(per_class),
        "per_db_n1": dict(per_db_n1),
    }


def main() -> None:
    files = _gather_files()
    block = _aggregate(files)
    n = block["n"]

    print("=" * 90)
    print(f"DEV-1515 cascade summary over {n} (instance, run) pairs")
    print(f"  runs: {', '.join(RUNS)}")
    print("=" * 90)
    print()
    print("phase1 monotone cascade  (cumulative passes up to and including each tier)")
    print(f"{'tier':56s}   {'pass':>8s}    {'rate':>6s}    delta")
    print("-" * 90)
    prev_total = None
    for key, label in _PHASE1_LABELS:
        tot = block["counts"][key]
        rate = (tot / n * 100.0) if n else 0.0
        delta = "" if prev_total is None else f"  (+{tot - prev_total})"
        tot_str = f"{tot}/{n}"
        print(f"  {label:54s}   {tot_str:>8s}    {rate:>5.1f}%{delta}")
        prev_total = tot
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
