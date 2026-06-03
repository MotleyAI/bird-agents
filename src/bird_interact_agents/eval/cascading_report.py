"""DEV-1515: cascading-phase1 aggregator + eval.json writer.

Replaces the legacy dual-eval block (``phase1_count_audited`` etc.) with
a single ``cascading_phase1`` dict carrying N1..N8 counts, rates,
deltas, and ``n_dual_eval_tasks``.

The aggregator walks ``<rows_dir>/<instance_id>/submission_annotation.json``
for each per-task row and sums the cascade verdicts. Missing per-row
annotation files raise — silent under-count is forbidden.

Back-compat: ``phase1_count`` and ``phase1_rate`` in the published
``eval.json`` map to the cascade's ``n1`` count + rate, REWRITTEN from
the recomputed cascade (not carried forward from base_metrics).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from bird_interact_agents.eval.annotation_io import read_submission_annotation
from bird_interact_agents.eval.tolerant_grader import (
    _CASCADE_ORDER, enforce_monotone_cascade,
)


def _per_row_cascade_bools(annotation_dir: Path) -> dict[str, bool]:
    """Load a single ``<rows_dir>/<inst>/submission_annotation.json``
    and return the monotone-enforced raw N1..N8 bools."""
    p = annotation_dir / "submission_annotation.json"
    if not p.exists():
        raise FileNotFoundError(
            f"submission_annotation.json missing under {annotation_dir} "
            "— cascading-phase1 aggregator requires every per-task row "
            "to carry a grader-written annotation",
        )
    ann = read_submission_annotation(p)
    ev = ann.evaluation
    raw = {
        "n1_original_gold": ev.phase1_against_original_gold == "pass",
        "n2_audited_primary": ev.phase1_against_audited_primary == "pass",
        "n3_any_audited_variant": (
            ev.phase1_against_any_audited_variant == "pass"
        ),
        "n4_tie_order": ev.correct_up_to_tie_order,
        "n5_llm_judge": ev.novel_reading_judgment == "pass",
        "n6_numeric_epsilon": ev.correct_under_numeric_epsilon,
        "n7_trailing_whitespace": ev.correct_under_trailing_whitespace,
        "n8_column_order": ev.correct_under_column_order,
        "n9_case_fold": ev.correct_under_case_fold,
    }
    return enforce_monotone_cascade(raw)


def aggregate_cascading_phase1(rows_dir: Path) -> dict:
    """Walk per-task ``submission_annotation.json`` files and return the
    cascading_phase1 block.

    Output shape::

      {
        "n_dual_eval_tasks": N,
        "counts": {"n1": ..., "n8": ...},
        "rates":  {"n1": 0.xx, ...},
        "deltas": {"n2": ..., "n3": ..., ...},
      }
    """
    rows_dir = Path(rows_dir)
    counts = {short_for(f): 0 for f in _CASCADE_ORDER}
    n = 0
    if rows_dir.exists():
        for sub in sorted(p for p in rows_dir.iterdir() if p.is_dir()):
            verdicts = _per_row_cascade_bools(sub)
            n += 1
            for f, v in verdicts.items():
                if v:
                    counts[short_for(f)] += 1
    rates = {
        k: (v / n) if n else 0.0 for k, v in counts.items()
    }
    deltas: dict[str, int] = {}
    prev: int | None = None
    for k in ("n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"):
        if prev is None:
            deltas[k] = 0
        else:
            deltas[k] = counts[k] - prev
        prev = counts[k]
    return {
        "n_dual_eval_tasks": n,
        "counts": counts,
        "rates": rates,
        "deltas": deltas,
    }


def short_for(field: str) -> str:
    """``"n1_original_gold"`` → ``"n1"``."""
    return field.split("_", 1)[0]


_LEGACY_KEYS_TO_DROP = (
    "phase1_count_audited",
    "phase1_count_original",
    "phase1_rate_audited",
    "phase1_rate_original",
    "n_dual_eval_tasks",  # moved into cascading_phase1
)


def emit_cascading_eval_json(
    rows_dir: Path,
    out_path: Path,
    base_metrics: dict | None = None,
) -> dict:
    """Merge ``base_metrics`` with the freshly-computed cascading block
    and write to ``out_path``. The legacy dual-eval keys are explicitly
    dropped; ``phase1_count`` / ``phase1_rate`` are REWRITTEN from N1.

    Returns the resulting metrics dict (for inline use)."""
    block = aggregate_cascading_phase1(Path(rows_dir))
    out = dict(base_metrics or {})
    for k in _LEGACY_KEYS_TO_DROP:
        out.pop(k, None)
    out["cascading_phase1"] = block
    # Back-compat aliases — rewritten from the freshly-computed cascade,
    # NOT carried forward from base_metrics.
    out["phase1_count"] = block["counts"]["n1"]
    out["phase1_rate"] = block["rates"]["n1"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    return out
