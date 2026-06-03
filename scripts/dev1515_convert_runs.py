"""DEV-1515: convert the two May-31 mini-interact runs to the new
annotation schema.

* `20260531t1008-claudes-slayer-890419` (15 households)
* `20260531t1343-claudes-slayer-b39bfc`  (38 alien/credit/etc.)

For each of the 53 (instance, run) pairs:

1. Build `<inst>.task.json` (run-independent). When the audited gold is
   IDENTICAL to the original gold (modulo whitespace), pre-fill
   `metadata_sufficiency.verdict = "sufficient"`,
   `original_gold_is_correct = True`, and `gold_variants = []` — per the
   user's "treat unchanged audit as correct" shortcut. Otherwise leave
   `metadata_sufficiency` + `failure_classification` sentinel-filled for
   subagent / human fill-in but pre-populate `gold_variants` with the
   primary audited reference.
2. Build `<inst>.submission.<run>.json` by running the full
   `tolerant_grader.grade_submission` cascade against the local sqlite
   DB at `<mini-interact>/<db>/<db>.sqlite`. N5 (LLM judge) is left at
   False here — it only fires for verdict="insufficient" tasks at grade
   time anyway.
3. Persist both via `annotation_io.write_*`.

Re-runnable: every call OVERWRITES the destination file (the user can
re-run after editing the task file). No `gold_variants` is preserved
across runs for changed-audit tasks — the next refresh will reset it.
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path
from typing import Optional

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_io import (
    write_submission_annotation,
    write_task_annotation,
    submission_annotation_path,
    task_annotation_path,
)
from bird_interact_agents.eval.annotation_schema import (
    AuditedGoldRef,
    GoldVariantRef,
    MaskedTerm,
    MetadataSufficiency,
    Provenance,
    TaskAnnotation,
)
from bird_interact_agents.eval.grade_in_place import (
    _build_submission_annotation,
    normalize_sol_sql,
)
from bird_interact_agents.eval.annotate import (
    _user_sim_interaction_from_trajectory,
)
from bird_interact_agents.eval.tolerant_grader import (
    default_executor,
    grade_submission,
)


BENCHMARK = "mini-interact"
AUDIT_FILE_REL = "audited_gold/mini_interact_audited.jsonl"

RUNS = {
    "20260531t1008-claudes-slayer-890419": "households",
    "20260531t1343-claudes-slayer-b39bfc": "mixed",
}

PENDING = "PENDING_HUMAN_REVIEW"


def _norm(sqls):
    return [" ".join((s or "").split()) for s in (sqls or [])]


def _masked_terms_from(task_row: dict) -> list[MaskedTerm]:
    amb = (task_row.get("user_query_ambiguity") or {}).get("critical_ambiguity") or []
    out: list[MaskedTerm] = []
    for entry in amb:
        out.append(MaskedTerm(
            term=entry.get("term", ""),
            type=entry.get("type", "intent_ambiguity"),
            is_mask=bool(entry.get("is_mask", True)),
            metadata_evidence=list(entry.get("metadata_evidence", []) or []),
        ))
    return out


def _build_task_annotation(
    *,
    task_row: dict,
    audit_row: dict | None,
    audit_unchanged: bool,
    annotated_at: str,
) -> TaskAnnotation:
    """Build the per-task annotation with the user's unchanged-audit shortcut."""
    iid = task_row["instance_id"]
    db = task_row["selected_database"]

    if audit_unchanged:
        ms = MetadataSufficiency(
            verdict="sufficient",
            rationale=(
                "Audited gold is identical to the original sol_sql modulo "
                "whitespace — the audit found nothing to change, so the "
                "task's published metadata is sufficient and the original "
                "gold is the canonical answer (DEV-1515 user-shortcut)."
            ),
            evidence_sources_consulted=[
                "audited_gold/mini_interact_audited.jsonl",
            ],
        )
        gold_variants: list[GoldVariantRef] = []
        original_gold_is_correct = True
        evaluator_prompt = None
    else:
        ms = MetadataSufficiency(
            verdict="ambiguous",
            rationale=PENDING,
            evidence_sources_consulted=[],
        )
        _vid = audit_row.get("variant_id", "primary") if audit_row else "primary"
        gold_variants = [
            GoldVariantRef(
                variant_id=_vid,
                interpretation=PENDING,
                primary=True,
                anchored_in=[],
                audited_gold_ref=AuditedGoldRef(
                    file=AUDIT_FILE_REL,
                    instance_id=iid,
                    variant_id=_vid,
                ),
                notes=(
                    "Audit changed the gold relative to the original — "
                    "see audited_gold for the diff; human review pending."
                ),
            ),
        ]
        original_gold_is_correct = False
        evaluator_prompt = None

    return TaskAnnotation(
        instance_id=iid,
        selected_database=db,
        annotated_by="dev1515-convert-runs",
        annotated_at=annotated_at,
        amb_user_query=task_row.get("amb_user_query", ""),
        external_knowledge=list(task_row.get("external_knowledge", []) or []),
        masked_terms=_masked_terms_from(task_row),
        metadata_sufficiency=ms,
        original_gold_is_correct=original_gold_is_correct,
        gold_variants=gold_variants,
        evaluator_prompt=evaluator_prompt,
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id=iid,
        ),
    )


def _pick_primary(audit_rows: list[dict]) -> dict | None:
    """Return the primary variant from a list of audit rows, or None if
    the list is empty. Falls back to the first row when no row carries
    ``primary: true`` (single-variant rows pre-DEV-1515 omit the field;
    the default is ``True``)."""
    if not audit_rows:
        return None
    for r in audit_rows:
        if r.get("primary", True):
            return r
    return audit_rows[0]


def _process_one(
    *,
    run_id: str,
    instance_id: str,
    task_row: dict,
    audit_rows: list[dict],
    audit_unchanged: bool,
    rows_dir: Path,
    mini_root: Path,
    annotations_root: Path,
    annotated_at: str,
) -> dict:
    """Return a small per-instance dict with cascade booleans + paths."""
    db = task_row["selected_database"]
    db_path = mini_root / db / f"{db}.sqlite"
    attempt_path = rows_dir / instance_id / "attempt-1.json"
    attempt = json.loads(attempt_path.read_text())
    submitted_sql = attempt.get("submitted_sql", "")
    usage = attempt.get("usage", {}) or {}

    audit_row = _pick_primary(audit_rows)

    # 1) Task annotation.
    task_ann = _build_task_annotation(
        task_row=task_row,
        audit_row=audit_row,
        audit_unchanged=audit_unchanged,
        annotated_at=annotated_at,
    )
    task_dest = task_annotation_path(
        benchmark=BENCHMARK,
        selected_database=db,
        instance_id=instance_id,
        repo_root=annotations_root.parent,
    )
    write_task_annotation(task_ann, task_dest)

    # 2) Cascade via grade_submission. Multi-variant audits are passed
    # as the full list so N3 ("any audited variant") sees every
    # alternate — collapsing to the primary alone would miscompute
    # cascade outcomes for ambiguous tasks.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    try:
        cascade = grade_submission(
            task_annotation=task_ann,
            audited_gold_rows=audit_rows,
            # Codex r13: ``normalize_sol_sql`` handles the bare-string
            # shape (some mini-interact rows carry ``sol_sql`` as a
            # single string). A bare ``list(...)`` would char-split it
            # into ``["S", "E", "L", "E", "C", "T", ...]`` and the
            # grader would mis-execute → N1 always False.
            original_sol_sql=normalize_sol_sql(task_row.get("sol_sql")),
            submitted_sql=submitted_sql,
            db_path=db_path,
            conn=conn,
            executor=default_executor,
            llm_judge=None,  # N5 stays False for now
        )
    finally:
        conn.close()

    # 3) Submission annotation.
    traj = list(attempt.get("trajectory") or [])
    user_sim = _user_sim_interaction_from_trajectory(traj)
    ann = _build_submission_annotation(
        task_annotation=task_ann,
        cascade=cascade,
        benchmark=BENCHMARK,
        run_id=run_id,
        trajectory_path=str(attempt_path),
        predicted_row_count=attempt.get("predicted_row_count"),
        duration_s=attempt.get("duration_s"),
        cost_usd_agent=usage.get("cost_usd_agent"),
        cost_usd_user_sim=usage.get("cost_usd_user_sim"),
        n_agent_turns=usage.get("n_agent_turns"),
        n_ask_user_calls=usage.get("n_ask_user_calls"),
        user_sim_interaction=user_sim,
    )
    sub_dest = submission_annotation_path(
        benchmark=BENCHMARK,
        selected_database=db,
        instance_id=instance_id,
        run_id=run_id,
        repo_root=annotations_root.parent,
    )
    write_submission_annotation(ann, sub_dest)

    return {
        "instance_id": instance_id,
        "db": db,
        "run_id": run_id,
        "audit_unchanged": audit_unchanged,
        "n1": cascade.n1_original_gold,
        "n2": cascade.n2_audited_primary,
        "n3": cascade.n3_any_audited_variant,
        "n4": cascade.n4_tie_order,
        "n5": cascade.n5_llm_judge,
        "n6": cascade.n6_numeric_epsilon,
        "n7": cascade.n7_trailing_whitespace,
        "n8": cascade.n8_column_order,
        "n9": cascade.n9_case_fold,
        "task_path": str(task_dest),
        "sub_path": str(sub_dest),
    }


def main() -> None:
    mini_root = paths.mini_interact_root()
    annotations_root = paths.annotations_root()
    results_cloud = paths.results_root() / "cloud"

    # Load mini-interact rows + audited gold.
    rows = {}
    for line in (mini_root / "mini_interact.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows[d["instance_id"]] = d

    # Collect every variant per instance_id from per-DB sidecar files.
    # Multi-variant audits (DEV-1515 source_conflict tasks) ship N rows
    # sharing instance_id; collapsing to a single dict per instance silently
    # drops alternates and makes downstream N3 miscompute.
    audit_rows: dict[str, list[dict]] = {}
    for audit_file in sorted(
        paths.audited_gold_root().glob("*/*_audited.jsonl")
    ):
        for line in audit_file.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            audit_rows.setdefault(d["instance_id"], []).append(d)

    annotated_at = (
        _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    )

    results: list[dict] = []
    for run_id, _label in RUNS.items():
        rows_dir = results_cloud / run_id / "rows"
        inst_ids = sorted(p.name for p in rows_dir.iterdir() if p.is_dir())
        print(f"\n[{run_id}] {len(inst_ids)} instances")
        for iid in inst_ids:
            task_row = rows[iid]
            variants = audit_rows[iid]
            primary = _pick_primary(variants)
            assert primary is not None  # rows[iid] non-empty by construction
            n_orig = _norm(primary.get("original_sol_sql"))
            n_aud = _norm(primary.get("audited_sol_sql"))
            unchanged = n_orig == n_aud
            try:
                rec = _process_one(
                    run_id=run_id,
                    instance_id=iid,
                    task_row=task_row,
                    audit_rows=variants,
                    audit_unchanged=unchanged,
                    rows_dir=rows_dir,
                    mini_root=mini_root,
                    annotations_root=annotations_root,
                    annotated_at=annotated_at,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {iid:40s} FAILED: {type(exc).__name__}: {exc}")
                raise
            else:
                tag = "U" if unchanged else "C"
                print(
                    f"  [{tag}] {iid:40s}  "
                    f"N1={int(rec['n1'])} N2={int(rec['n2'])} N3={int(rec['n3'])} "
                    f"N4={int(rec['n4'])} N6={int(rec['n6'])} "
                    f"N7={int(rec['n7'])} N8={int(rec['n8'])}"
                )
                results.append(rec)

    # Save a small index for the summary step.
    out_index = annotations_root / "_dev1515_convert_index.json"
    out_index.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {len(results)} (task, submission) pairs.")
    print(f"Index: {out_index}")


if __name__ == "__main__":
    main()
