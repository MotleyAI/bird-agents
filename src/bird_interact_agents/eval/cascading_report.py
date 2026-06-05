"""DEV-1515/1533: cascading-phase1 aggregator + eval.json writer.

Reads from ``runs/<benchmark>/<db>/<instance_id>/<run_id>.json`` (the
golden per-(task, run) store introduced in DEV-1533) rather than the
temporary ``rows_dir/<inst>/submission_annotation.json`` files.

Two aggregation modes:

* ``aggregate_cascading_phase1(benchmark, run_id)`` — aggregate only the
  rows that belong to a specific run. Used by
  ``emit_cascading_eval_json`` to produce ``eval.json`` at end of run.

* ``aggregate_cascading_latest(benchmark)`` — for each (db, instance_id)
  pair pick the latest run (by ``annotated_at``), then aggregate. Used
  by standalone reporting scripts.

Both modes produce:

1. A ``cascading_phase1`` legacy-compat block (cumulative n1..n9 counts)
   for backward compat with existing ``eval.json`` consumers.
2. A ``cascading_partition`` block — mutually exclusive per-line counts
   (each task maps to exactly ONE line) plus cumulative display sums.

The partition lines:

    L1:  original gold annotated CORRECT, agent matched it           [pass]
    L2:  original gold annotated WRONG,   agent matched original     [diagnostic]
    L3:  original gold failed/wrong → agent matched AUDITED PRIMARY  [pass]
    L4:  original gold failed/wrong → agent matched OTHER variant    [pass]
    L5:  nothing above, correct up to tie order (N4)                 [pass]
    L6:  LLM judge accepts (N5)                                      [pass]
    L7:  numeric epsilon (N6)                                        [pass]
    L8:  trailing whitespace (N7)                                    [pass]
    L9:  column order (N8)                                           [pass]
    L10: case fold (N9)                                              [pass]
    L11: all failed                                                  [fail]

``phase1_count`` / ``phase1_rate`` back-compat aliases map to n1 count
of the cumulative block (unchanged from DEV-1515).

The legacy ``_aggregate_cascading_phase1_legacy(rows_dir)`` function is
kept private for in-run emission when the runs/ store is not yet
populated (e.g. first write in a filtered rerun).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from bird_interact_agents import paths as _paths
from bird_interact_agents.eval.annotation_io import (
    iter_run_annotations,
    latest_run_per_instance,
    read_submission_annotation,
)
from bird_interact_agents.eval.tolerant_grader import (
    _CASCADE_ORDER, enforce_monotone_cascade,
)


# ---------------------------------------------------------------------------
# Cascade bools from a stored SubmissionAnnotation
# ---------------------------------------------------------------------------


def _annotation_cascade_bools(ann) -> dict[str, bool]:
    """Extract raw N1..N9 bools from a SubmissionAnnotation and apply
    monotone enforcement.

    For DEV-1533: when ``original_gold_annotated_correct=False`` the
    monotone enforcement must not propagate N1→N2+, matching the grader
    behaviour fixed in DEV-1533.
    """
    ev = ann.evaluation
    raw = {
        "n1_original_gold": ev.phase1_against_original_gold == "pass",
        "n2_audited_primary": ev.phase1_against_audited_primary == "pass",
        "n3_any_audited_variant": ev.phase1_against_any_audited_variant == "pass",
        "n4_tie_order": ev.correct_up_to_tie_order,
        "n5_llm_judge": ev.novel_reading_judgment == "pass",
        "n6_numeric_epsilon": ev.correct_under_numeric_epsilon,
        "n7_trailing_whitespace": ev.correct_under_trailing_whitespace,
        "n8_column_order": ev.correct_under_column_order,
        "n9_case_fold": ev.correct_under_case_fold,
    }
    original_gold_is_correct = getattr(ann, "original_gold_annotated_correct", None)
    if original_gold_is_correct is None:
        original_gold_is_correct = True
    return enforce_monotone_cascade(
        raw, original_gold_is_correct=original_gold_is_correct,
    )


# ---------------------------------------------------------------------------
# Legacy helper (reads from rows_dir temp files — used as fallback)
# ---------------------------------------------------------------------------


def _per_row_cascade_bools_legacy(annotation_dir: Path) -> dict[str, bool]:
    """Load a single ``<rows_dir>/<inst>/submission_annotation.json``."""
    p = annotation_dir / "submission_annotation.json"
    if not p.exists():
        raise FileNotFoundError(
            f"submission_annotation.json missing under {annotation_dir} "
            "— cascading-phase1 aggregator requires every per-task row "
            "to carry a grader-written annotation",
        )
    ann = read_submission_annotation(p)
    return _annotation_cascade_bools(ann)


def _aggregate_cascading_phase1_legacy(
    rows_dir: Path,
    *,
    instance_filter: "set[str] | None" = None,
) -> dict:
    """Walk per-task ``submission_annotation.json`` files in ``rows_dir``.
    Kept private; used only when the runs/ store is not yet available."""
    rows_dir = Path(rows_dir)
    counts = {short_for(f): 0 for f in _CASCADE_ORDER}
    n = 0
    if rows_dir.exists():
        for sub in sorted(p for p in rows_dir.iterdir() if p.is_dir()):
            if instance_filter is not None and sub.name not in instance_filter:
                continue
            verdicts = _per_row_cascade_bools_legacy(sub)
            n += 1
            for f, v in verdicts.items():
                if v:
                    counts[short_for(f)] += 1
    return _build_result(n, counts)


# ---------------------------------------------------------------------------
# Partition assignment
# ---------------------------------------------------------------------------


_PARTITION_LABELS = (
    "l1_correct_original",
    "l2_wrong_original",
    "l3_audited_primary",
    "l4_other_audited_variant",
    "l5_tie_order",
    "l6_llm_judge",
    "l7_numeric_epsilon",
    "l8_trailing_whitespace",
    "l9_column_order",
    "l10_case_fold",
    "l11_fail",
)

_PARTITION_DISPLAY = {
    "l1_correct_original": "original gold correct → agent matched it",
    "l2_wrong_original": "original gold wrong  → agent coincidentally matched it",
    "l3_audited_primary": "original failed/wrong → matched audited primary",
    "l4_other_audited_variant": "original failed/wrong → matched other audited variant",
    "l5_tie_order": "correct up to tie order (N4)",
    "l6_llm_judge": "LLM judge accepts novel reading (N5)",
    "l7_numeric_epsilon": "numeric epsilon (N6)",
    "l8_trailing_whitespace": "trailing whitespace (N7)",
    "l9_column_order": "column order (N8)",
    "l10_case_fold": "case fold (N9)",
    "l11_fail": "all cascade tiers failed",
}

# L2 is diagnostic — does not count toward the headline pass rate.
_PARTITION_IS_PASS = {
    "l1_correct_original": True,
    "l2_wrong_original": False,
    "l3_audited_primary": True,
    "l4_other_audited_variant": True,
    "l5_tie_order": True,
    "l6_llm_judge": True,
    "l7_numeric_epsilon": True,
    "l8_trailing_whitespace": True,
    "l9_column_order": True,
    "l10_case_fold": True,
    "l11_fail": False,
}


def _assign_partition_tier(ann) -> str:
    """Return the single partition label for this annotation."""
    ev = ann.evaluation
    original_gold_is_correct = getattr(ann, "original_gold_annotated_correct", None)
    if original_gold_is_correct is None:
        original_gold_is_correct = True

    n1 = ev.phase1_against_original_gold == "pass"
    n2 = ev.phase1_against_audited_primary == "pass"
    n3 = ev.phase1_against_any_audited_variant == "pass"
    n4 = ev.correct_up_to_tie_order
    n5 = ev.novel_reading_judgment == "pass"
    n6 = ev.correct_under_numeric_epsilon
    n7 = ev.correct_under_trailing_whitespace
    n8 = ev.correct_under_column_order
    n9 = ev.correct_under_case_fold

    # Harness-confirmed rows: route based purely on original_gold_is_correct.
    if getattr(ev, "rationale", "") == "harness_confirmed":
        return "l1_correct_original" if original_gold_is_correct else "l3_audited_primary"

    if original_gold_is_correct and n1:
        return "l1_correct_original"

    # Genuine N2/N3 pass (original gold failed/wrong; agent matched audited).
    if n2:
        return "l3_audited_primary"
    if n3:
        return "l4_other_audited_variant"

    # N1=True but original is wrong and N2/N3 failed → agent matched wrong gold.
    if not original_gold_is_correct and n1:
        return "l2_wrong_original"

    if n4:
        return "l5_tie_order"
    if n5:
        return "l6_llm_judge"
    if n6:
        return "l7_numeric_epsilon"
    if n7:
        return "l8_trailing_whitespace"
    if n8:
        return "l9_column_order"
    if n9:
        return "l10_case_fold"
    return "l11_fail"


def _build_partition_block(
    partition_counts: "dict[str, int]",
    n: int,
) -> dict:
    """Build the cascading_partition block with exclusive + cumulative counts."""
    cumsum = 0
    cumsum_pass = 0
    rows = {}
    for label in _PARTITION_LABELS:
        count = partition_counts.get(label, 0)
        cumsum += count
        is_pass = _PARTITION_IS_PASS[label]
        if is_pass:
            cumsum_pass += count
        rows[label] = {
            "count": count,
            "rate": count / n if n else 0.0,
            "cumsum": cumsum,
            "cumsum_rate": cumsum / n if n else 0.0,
            "display": _PARTITION_DISPLAY[label],
            "is_pass": is_pass,
        }
    return {
        "n_tasks": n,
        "tiers": rows,
        "pass_count": cumsum_pass,
        "pass_rate": cumsum_pass / n if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Result builder shared by all aggregation paths
# ---------------------------------------------------------------------------


def short_for(field: str) -> str:
    """``"n1_original_gold"`` → ``"n1"``."""
    return field.split("_", 1)[0]


def _build_result(n: int, counts: "dict[str, int]", partition_counts: "Optional[dict[str,int]]" = None) -> dict:
    rates = {k: (v / n) if n else 0.0 for k, v in counts.items()}
    deltas: dict[str, int] = {}
    prev: Optional[int] = None
    for k in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"):
        if prev is None:
            deltas[k] = 0
        else:
            deltas[k] = counts[k] - prev
        prev = counts[k]
    result: dict = {
        "n_dual_eval_tasks": n,
        "counts": counts,
        "rates": rates,
        "deltas": deltas,
    }
    if partition_counts is not None:
        result["cascading_partition"] = _build_partition_block(partition_counts, n)
    return result


# ---------------------------------------------------------------------------
# New API: reads from runs/
# ---------------------------------------------------------------------------


def _aggregate_from_annotations(
    annotations: "Iterable[tuple]",
    *,
    instance_filter: "set[str] | None" = None,
) -> dict:
    """Core aggregator — accepts an iterable of ``(path, annotation)`` pairs."""
    counts = {short_for(f): 0 for f in _CASCADE_ORDER}
    partition_counts: dict[str, int] = {k: 0 for k in _PARTITION_LABELS}
    n = 0
    for _path, ann in annotations:
        if instance_filter is not None and ann.instance_id not in instance_filter:
            continue
        verdicts = _annotation_cascade_bools(ann)
        n += 1
        for f, v in verdicts.items():
            if v:
                counts[short_for(f)] += 1
        tier = _assign_partition_tier(ann)
        partition_counts[tier] = partition_counts.get(tier, 0) + 1
    return _build_result(n, counts, partition_counts)


def aggregate_cascading_phase1(
    benchmark: str,
    run_id: str,
    *,
    instance_filter: "set[str] | None" = None,
) -> dict:
    """Aggregate cascade metrics for a specific run from ``runs/``.

    ``instance_filter``: when set, only instances whose ``instance_id`` is
    in the filter are counted (same Codex r11 semantics as the legacy
    ``rows_dir`` version — filtered reruns exclude stale prior annotations
    from the published metrics).
    """
    annotations = iter_run_annotations(benchmark=benchmark, run_id=run_id)
    return _aggregate_from_annotations(annotations, instance_filter=instance_filter)


def aggregate_cascading_latest(benchmark: str) -> dict:
    """Aggregate cascade metrics using the latest run per task.

    "Latest" is determined by ``annotated_at`` field (max per
    ``(db, instance_id)`` pair) — safe for local run_ids that are not
    timestamp-formatted.
    """
    latest = latest_run_per_instance(benchmark=benchmark)
    pairs = [(_paths.runs_root(), ann) for (_run_id, ann) in latest.values()]
    return _aggregate_from_annotations(pairs)


# ---------------------------------------------------------------------------
# eval.json emission
# ---------------------------------------------------------------------------


_LEGACY_KEYS_TO_DROP = (
    "phase1_count_audited",
    "phase1_count_original",
    "phase1_rate_audited",
    "phase1_rate_original",
    "n_dual_eval_tasks",  # moved into cascading_phase1
)


def emit_cascading_eval_json(
    benchmark: str,
    run_id: str,
    out_path: Path,
    base_metrics: "dict | None" = None,
    *,
    instance_filter: "set[str] | None" = None,
) -> dict:
    """Merge ``base_metrics`` with the freshly-computed cascading block
    and write to ``out_path``.

    Reads run annotations from ``runs/<benchmark>/``. If no run
    annotations are found for this ``run_id``, the cascading_phase1 block
    will have zero counts.

    The legacy dual-eval keys are explicitly dropped; ``phase1_count`` /
    ``phase1_rate`` are REWRITTEN from N1.

    Returns the resulting metrics dict.
    """
    block = aggregate_cascading_phase1(
        benchmark, run_id, instance_filter=instance_filter,
    )
    out = dict(base_metrics or {})
    for k in _LEGACY_KEYS_TO_DROP:
        out.pop(k, None)
    out["cascading_phase1"] = block
    out["phase1_count"] = block["counts"]["n1"]
    out["phase1_rate"] = block["rates"]["n1"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    return out
