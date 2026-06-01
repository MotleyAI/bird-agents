"""DEV-1515 review-followup: a multi-statement gold list (`CREATE TEMP …`
+ final `SELECT`) must share its SQLite connection so the setup state
survives to the final comparison statement.

Before the fix in `_multi_sql_execute`, `default_executor` opened and
closed a fresh `sqlite3.Connection` per call when `conn=None`, so the
TEMP table created by the first statement vanished before the second
statement ran, and the final `SELECT` raised
`sqlite3.OperationalError: no such table: temp_setup`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "scratch.sqlite"
    con = sqlite3.connect(str(db))
    try:
        con.execute("CREATE TABLE base (id INTEGER, label TEXT)")
        con.executemany(
            "INSERT INTO base (id, label) VALUES (?, ?)",
            [(1, "a"), (2, "b"), (3, "c")],
        )
        con.commit()
    finally:
        con.close()
    return db


def _audited_row(*, audited_sol_sql: list[str]) -> dict:
    return {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "benchmark": "mini_interact",
        "audit_status": "edited",
        "original_sol_sql": ["SELECT id FROM base WHERE id = 1"],
        "audited_sol_sql": audited_sol_sql,
        "variant_id": "primary",
        "primary": True,
        "changes": [],
        "reasoning_summary": "",
        "skill_version": "audit-gold-sql/1.0",
        "audited_at": "2026-05-30T00:00:00+00:00",
    }


def _task_annotation():
    from bird_interact_agents.eval import (
        AuditedGoldRef,
        GoldVariantRef,
        MetadataSufficiency,
        TaskAnnotation,
    )
    from bird_interact_agents.eval.annotation_schema import Provenance

    return TaskAnnotation(
        instance_id="alien_1",
        selected_database="alien",
        annotated_by="test",
        annotated_at="2026-05-31",
        amb_user_query="x",
        metadata_sufficiency=MetadataSufficiency(
            verdict="sufficient", rationale="r",
        ),
        gold_variants=[
            GoldVariantRef(
                variant_id="primary",
                interpretation="x",
                primary=True,
                audited_gold_ref=AuditedGoldRef(
                    file="audited_gold/mini_interact_audited.jsonl",
                    instance_id="alien_1",
                    variant_id="primary",
                ),
            ),
        ],
        evaluator_prompt=None,
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id="alien_1",
        ),
    )


def test_multi_sql_gold_shares_temp_state_with_conn_none(tmp_path: Path):
    """Audited gold = [CREATE TEMP setup, SELECT from temp]. The two
    statements MUST run against the same SQLite connection or the SELECT
    will fail with `no such table`. Caller passes ``conn=None`` so the
    grader's default executor is exercised end-to-end."""
    from bird_interact_agents.eval.tolerant_grader import grade_submission

    db = _make_db(tmp_path)
    audited = _audited_row(
        audited_sol_sql=[
            "CREATE TEMP TABLE temp_setup AS SELECT id FROM base WHERE id < 3",
            "SELECT id FROM temp_setup ORDER BY id",
        ],
    )
    cascade = grade_submission(
        task_annotation=_task_annotation(),
        audited_gold_rows=[audited],
        original_sol_sql=["SELECT id FROM base WHERE id < 3 ORDER BY id"],
        submitted_sql="SELECT id FROM base WHERE id IN (1, 2) ORDER BY id",
        db_path=db,
        conn=None,
    )
    # The audited primary's two-statement list produces rows (1,) (2,);
    # the submission produces the same rowset. n2/n3 must pass — they
    # can only pass if the TEMP table created by the first statement
    # survived to the second.
    assert cascade.n2_audited_primary is True, (
        "n2 should pass — audited primary's CREATE TEMP + SELECT yields "
        "{1,2}, matching the submission. If TEMP state was lost between "
        "statements, the SELECT raised and this would be False."
    )
    assert cascade.n3_any_audited_variant is True
    # n1 (original gold) also runs against the same DB and should match.
    assert cascade.n1_original_gold is True
