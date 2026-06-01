"""DEV-1515: convert any livesqlbench DB's tasks to the new annotation
schema. Mirrors `dev1515_convert_runs.py` but for livesqlbench's one-shot
data layout:

* `query` (NL only) instead of `amb_user_query` (NL + masked snippet).
* No `user_query_ambiguity.critical_ambiguity`; no masked snippets.
* No user_sim — `user_sim_interaction.n_asks` is always 0.
* `sol_sql` + `external_knowledge` live in a separate gated sidecar,
  joined on `instance_id`.

Inputs:
* `livesqlbench-base-lite-sqlite/livesqlbench_data_sqlite.jsonl` —
  main task rows; filtered to `--db`.
* `livesqlbench-base-lite-sqlite/livesqlbench_sqlite_gt_kg_testcases_0528.jsonl`
  — gold sidecar (`instance_id`, `sol_sql`, `external_knowledge`,
  `test_cases`).
* `audited_gold/livesqlbench_audited.jsonl` — audited sidecar.
* (optional) `results/cloud/<run-id>/rows/<inst>/attempt-1.json` — when
  `--run-id` is passed, submission annotations are written for every
  instance that has a run row.

Outputs:
* `annotations/livesqlbench/<db>/<inst>.task.json` for every instance in
  the DB.
* `annotations/livesqlbench/<db>/<inst>.submission.<run-id>.json` when
  a corresponding `attempt-1.json` exists under `--run-id`.

Usage:
    uv run python scripts/dev1515_convert_livesqlbench.py \\
        --db museum [--run-id 20260531t1013-claudes-slayer-48eb0f]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sqlite3
from pathlib import Path

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
    MetadataSufficiency,
    Provenance,
    TaskAnnotation,
)
from bird_interact_agents.eval.grade_in_place import _build_submission_annotation
from bird_interact_agents.eval.annotate import (
    _user_sim_interaction_from_trajectory,
)
from bird_interact_agents.eval.tolerant_grader import (
    default_executor,
    grade_submission,
)


BENCHMARK = "livesqlbench"
AUDIT_FILE_REL = "audited_gold/livesqlbench_audited.jsonl"

PENDING = "PENDING_HUMAN_REVIEW"

LIVESQL_ROOT = Path("/home/james/Dropbox/SLayer/livesqlbench-base-lite-sqlite")


def _norm(sqls):
    return [" ".join((s or "").split()) for s in (sqls or [])]


def _load_data_rows(db: str) -> dict:
    p = LIVESQL_ROOT / "livesqlbench_data_sqlite.jsonl"
    rows = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("selected_database") == db:
            rows[d["instance_id"]] = d
    return rows


def _load_gold_rows() -> dict:
    p = LIVESQL_ROOT / "livesqlbench_sqlite_gt_kg_testcases_0528.jsonl"
    rows = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows[d["instance_id"]] = d
    return rows


def _load_audit_rows() -> dict:
    p = paths.audited_gold_root() / "livesqlbench_audited.jsonl"
    rows = {}
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows[d["instance_id"]] = d
    return rows


def _build_task_annotation(
    *,
    instance_id: str,
    db: str,
    data_row: dict,
    gold_row: dict,
    audit_row: dict | None,
    audit_unchanged: bool,
    annotated_at: str,
) -> TaskAnnotation:
    is_management_deferral = (
        audit_row is not None
        and audit_row.get("audit_status") == "unrecoverable"
        and any(
            c.get("clause_kind") == "management_category"
            for c in (audit_row.get("changes") or [])
        )
    )
    if is_management_deferral:
        # Management-category tasks (CREATE/ALTER/DELETE/CREATE INDEX
        # etc.) are deferred from the row-count audit per the shared
        # contract — eval depends on side-effects + custom test_case
        # comparators, not row equality. The published metadata is fine
        # for the task itself; the audit just doesn't certify the SQL.
        ms = MetadataSufficiency(
            verdict="sufficient",
            rationale=(
                "Management-category task: the audit-gold-sql skill "
                "deferred this row from the row-count audit because eval "
                "uses test_case Python comparators rather than row "
                "equality. The published metadata sufficiently anchors "
                "the user query; canonicity of the SQL is checked by "
                "the test_cases, not by this annotation."
            ),
            evidence_sources_consulted=["audited_gold/livesqlbench_audited.jsonl"],
        )
        gold_variants: list[GoldVariantRef] = []
        original_gold_is_correct = True
        evaluator_prompt = None
    elif audit_row is not None and audit_unchanged:
        ms = MetadataSufficiency(
            verdict="sufficient",
            rationale=(
                "Audited gold is identical to the original `sol_sql` modulo "
                "whitespace — the audit found nothing to change, so the "
                "task's published metadata is sufficient and the original "
                "gold is the canonical answer (DEV-1515 user-shortcut)."
            ),
            evidence_sources_consulted=["audited_gold/livesqlbench_audited.jsonl"],
        )
        gold_variants = []
        original_gold_is_correct = True
        evaluator_prompt = None
    elif audit_row is not None and not audit_unchanged:
        ms = MetadataSufficiency(
            verdict="ambiguous",
            rationale=PENDING,
            evidence_sources_consulted=[],
        )
        gold_variants = [
            GoldVariantRef(
                variant_id="primary",
                interpretation=PENDING,
                primary=True,
                anchored_in=[],
                audited_gold_ref=AuditedGoldRef(
                    file=AUDIT_FILE_REL,
                    instance_id=instance_id,
                    variant_id="primary",
                ),
                notes=(
                    "Audit changed the gold relative to the original — "
                    "see audited_gold for the diff; human review pending."
                ),
            ),
        ]
        original_gold_is_correct = False
        evaluator_prompt = None
    else:
        # No audit row — treat as sufficient by default; human can revise.
        ms = MetadataSufficiency(
            verdict="sufficient",
            rationale=(
                "No audit row in audited_gold/livesqlbench_audited.jsonl. "
                "Default verdict assumes the original gold is canonical; "
                "revisit if the agent's failure pattern shows real "
                "metadata ambiguity."
            ),
            evidence_sources_consulted=["audited_gold/livesqlbench_audited.jsonl (missing)"],
        )
        gold_variants = []
        original_gold_is_correct = True
        evaluator_prompt = None

    return TaskAnnotation(
        instance_id=instance_id,
        selected_database=db,
        annotated_by=f"dev1515-convert-livesqlbench-{db}",
        annotated_at=annotated_at,
        amb_user_query=data_row.get("query", ""),  # livesqlbench: NL-only `query`
        external_knowledge=list(gold_row.get("external_knowledge", []) or []),
        masked_terms=[],  # no masking layer in livesqlbench
        metadata_sufficiency=ms,
        original_gold_is_correct=original_gold_is_correct,
        gold_variants=gold_variants,
        evaluator_prompt=evaluator_prompt,
        provenance=Provenance(
            task_jsonl_path="livesqlbench_data_sqlite.jsonl",
            task_jsonl_instance_id=instance_id,
        ),
    )


def _process_one(
    *,
    instance_id: str,
    db: str,
    run_id: str | None,
    data_row: dict,
    gold_row: dict,
    audit_row: dict | None,
    audit_unchanged: bool,
    rows_dir: Path | None,
    annotations_root: Path,
    annotated_at: str,
) -> dict:
    db_path = LIVESQL_ROOT / db / f"{db}.sqlite"
    task_ann = _build_task_annotation(
        instance_id=instance_id,
        db=db,
        data_row=data_row,
        gold_row=gold_row,
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

    sub_dest = None
    rec_extra = {"has_submission": False}
    if (
        run_id is not None
        and rows_dir is not None
        and (rows_dir / instance_id / "attempt-1.json").exists()
    ):
        attempt_path = rows_dir / instance_id / "attempt-1.json"
        attempt = json.loads(attempt_path.read_text())
        submitted_sql = attempt.get("submitted_sql", "")
        usage = attempt.get("usage", {}) or {}

        audit_arg = [audit_row] if audit_row is not None else []
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
        try:
            cascade = grade_submission(
                task_annotation=task_ann,
                audited_gold_rows=audit_arg,
                original_sol_sql=list(gold_row.get("sol_sql") or []),
                submitted_sql=submitted_sql,
                db_path=db_path,
                conn=conn,
                executor=default_executor,
                llm_judge=None,
            )
        finally:
            conn.close()

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
        rec_extra = {
            "has_submission": True,
            "n1": cascade.n1_original_gold,
            "n2": cascade.n2_audited_primary,
            "n3": cascade.n3_any_audited_variant,
            "n4": cascade.n4_tie_order,
            "n5": cascade.n5_llm_judge,
            "n6": cascade.n6_numeric_epsilon,
            "n7": cascade.n7_trailing_whitespace,
            "n8": cascade.n8_column_order,
            "sub_path": str(sub_dest),
        }
    return {
        "instance_id": instance_id,
        "audit_unchanged": audit_unchanged,
        "audit_present": audit_row is not None,
        "task_path": str(task_dest),
        **rec_extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        required=True,
        help="livesqlbench `selected_database` to convert (e.g. museum, credit, mental)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="optional cloud run id; when set, writes submission "
             "annotations for instances with attempt-1.json under "
             "results/cloud/<run-id>/rows/.",
    )
    args = parser.parse_args()
    db = args.db
    run_id = args.run_id

    annotations_root = paths.annotations_root()
    results_cloud = paths.results_root() / "cloud"

    data_rows = _load_data_rows(db)
    gold_rows = _load_gold_rows()
    audit_rows = _load_audit_rows()

    instance_ids = sorted(data_rows.keys())
    print(f"{db} instances in data: {len(instance_ids)}")
    print(f"{db} audit rows present: "
          f"{sum(1 for i in instance_ids if i in audit_rows)}")

    run_rows_dir = (results_cloud / run_id / "rows") if run_id else None
    if run_rows_dir is not None:
        print(f"run rows dir exists: {run_rows_dir.exists()}")
        if run_rows_dir.exists():
            present = sorted(p.name for p in run_rows_dir.iterdir()
                             if p.is_dir() and p.name.startswith(f"{db}_"))
            print(f"{db} instances with submissions: {len(present)}")
        else:
            present = []
    else:
        present = []

    annotated_at = (
        _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    )

    results = []
    for iid in instance_ids:
        if iid not in gold_rows:
            print(f"  {iid:30s}  SKIP: no gold row")
            continue
        audit = audit_rows.get(iid)
        unchanged = False
        if audit is not None:
            orig = _norm(audit.get("original_sol_sql"))
            aud = _norm(audit.get("audited_sol_sql"))
            unchanged = (orig == aud)
        rows_dir = (
            run_rows_dir
            if (run_rows_dir is not None and run_rows_dir.exists() and iid in present)
            else None
        )
        try:
            rec = _process_one(
                instance_id=iid,
                db=db,
                run_id=run_id,
                data_row=data_rows[iid],
                gold_row=gold_rows[iid],
                audit_row=audit,
                audit_unchanged=unchanged,
                rows_dir=rows_dir,
                annotations_root=annotations_root,
                annotated_at=annotated_at,
            )
        except Exception as exc:
            print(f"  {iid:30s}  FAILED: {type(exc).__name__}: {exc}")
            raise
        else:
            audit_tag = "U" if unchanged else ("C" if audit else "-")
            sub_tag = "S" if rec["has_submission"] else " "
            cascade_str = ""
            if rec["has_submission"]:
                cascade_str = (
                    f"  N1={int(rec['n1'])} N3={int(rec['n3'])} "
                    f"N6={int(rec['n6'])}"
                )
            print(f"  [{audit_tag}{sub_tag}] {iid:30s}{cascade_str}")
            results.append(rec)

    out_index = annotations_root / f"_dev1515_convert_livesqlbench_{db}_index.json"
    out_index.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {len(results)} task annotations "
          f"+ {sum(1 for r in results if r['has_submission'])} submission annotations.")
    print(f"Index: {out_index}")


if __name__ == "__main__":
    main()
