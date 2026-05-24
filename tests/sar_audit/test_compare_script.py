"""Comparison script: SAR-audit vs in-house audit, per-instance diff."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird_interact_agents.sar_audit import compare


def _inhouse_row(instance_id: str, sql: str, sample_row=None):
    return {
        "instance_id": instance_id,
        "selected_database": "credit",
        "audit_status": "clean",
        "original_sol_sql": [sql],
        "audited_sol_sql": [sql],
        "audited_sample_row": sample_row,
        "changes": [],
        "reasoning_summary": "x",
        "skill_version": "audit-gold-sql/1.0",
        "audited_at": "2026-05-14T12:00:00+00:00",
    }


def _sar_row(instance_id: str, original_sql: str, audited_sql: str, status: str, sample_row=None):
    return {
        "instance_id": instance_id,
        "selected_database": "credit",
        "audit_status": status,
        "original_sol_sql": [original_sql],
        "audited_sol_sql": [audited_sql],
        "audited_sample_row": sample_row,
        "audited_sample_row_status": "ok" if sample_row is not None else "empty",
        "audited_sample_row_error": None,
        "changes": [] if status == "clean" else [
            {
                "clause_kind": "sar_revision",
                "source": "sar_agent",
                "original": original_sql,
                "replacement": audited_sql,
                "why_unjustified": "x",
                "justified_by": [],
            }
        ],
        "reasoning_summary": "x",
        "skill_version": "sar-agent/1.0",
        "audited_at": "2026-05-21T12:00:00+00:00",
        "sar_correctness_flag": status == "clean",
        "sar_ambiguity_flag": status == "ambiguous",
        "revised_question": None,
        "step_count": 1,
        "cost_usd": 0.0,
        "audit_model_requested": "claude-opus-4-7",
        "audit_model_actual": "claude-opus-4-7-20260121",
        "raw_trajectory": None,
    }


def test_compare_matching_sql_and_results(
    tmp_path: Path, fake_db, write_jsonl
):
    inhouse = tmp_path / "audited_gold" / "credit" / "credit_audited.jsonl"
    sar = tmp_path / "sar_audited_gold" / "credit" / "credit_sar_audited.jsonl"
    write_jsonl(
        inhouse,
        [_inhouse_row("credit_1", "SELECT x FROM t ORDER BY x LIMIT 1", sample_row=[1])],
    )
    write_jsonl(
        sar,
        [
            _sar_row(
                "credit_1",
                "SELECT x FROM t ORDER BY x LIMIT 1",
                "SELECT x FROM t ORDER BY x LIMIT 1",
                "clean",
                sample_row=[1],
            )
        ],
    )

    out = compare.compare_db(
        db="credit",
        inhouse_path=inhouse,
        sar_path=sar,
        db_path=fake_db,
    )
    assert len(out.per_instance) == 1
    row = out.per_instance[0]
    assert row.instance_id == "credit_1"
    assert row.inhouse_status == "clean"
    assert row.sar_status == "clean"
    assert row.sql_match == "yes"
    assert row.result_match == "yes"
    assert row.sample_row_match == "yes"


def test_compare_differing_sql_same_result(
    tmp_path: Path, fake_db, write_jsonl
):
    """Two SQL strings that aren't syntactically equal but execute to the
    same result set must show sql_match=no but result_match=yes."""
    inhouse = tmp_path / "audited_gold" / "credit" / "credit_audited.jsonl"
    sar = tmp_path / "sar_audited_gold" / "credit" / "credit_sar_audited.jsonl"
    write_jsonl(
        inhouse,
        [_inhouse_row("credit_1", "SELECT x FROM t ORDER BY x ASC LIMIT 1", sample_row=[1])],
    )
    write_jsonl(
        sar,
        [
            _sar_row(
                "credit_1",
                "SELECT x FROM t ORDER BY x ASC LIMIT 1",
                "SELECT x FROM t ORDER BY x LIMIT 1",  # ASC is implicit — semantic equivalent
                "edited",
                sample_row=[1],
            )
        ],
    )

    out = compare.compare_db(db="credit", inhouse_path=inhouse, sar_path=sar, db_path=fake_db)
    row = out.per_instance[0]
    assert row.sql_match == "no"
    assert row.result_match == "yes"


def test_compare_missing_instance_in_sar(tmp_path: Path, fake_db, write_jsonl):
    """Instance present in in-house but missing in SAR → n/a values."""
    inhouse = tmp_path / "audited_gold" / "credit" / "credit_audited.jsonl"
    sar = tmp_path / "sar_audited_gold" / "credit" / "credit_sar_audited.jsonl"
    write_jsonl(inhouse, [_inhouse_row("credit_99", "SELECT 1", sample_row=[1])])
    write_jsonl(sar, [])

    out = compare.compare_db(db="credit", inhouse_path=inhouse, sar_path=sar, db_path=fake_db)
    row = next(r for r in out.per_instance if r.instance_id == "credit_99")
    assert row.sar_status == "(missing)"
    assert row.sql_match == "n/a"
    assert row.result_match == "n/a"


def test_compare_missing_instance_in_inhouse(tmp_path: Path, fake_db, write_jsonl):
    """Instance present in SAR but missing in in-house → inhouse_status='(missing)'."""
    inhouse = tmp_path / "audited_gold" / "credit" / "credit_audited.jsonl"
    sar = tmp_path / "sar_audited_gold" / "credit" / "credit_sar_audited.jsonl"
    write_jsonl(inhouse, [])
    write_jsonl(
        sar,
        [
            _sar_row(
                "credit_99",
                "SELECT x FROM t LIMIT 1",
                "SELECT x FROM t LIMIT 1",
                "clean",
                sample_row=[1],
            )
        ],
    )

    out = compare.compare_db(db="credit", inhouse_path=inhouse, sar_path=sar, db_path=fake_db)
    row = next(r for r in out.per_instance if r.instance_id == "credit_99")
    assert row.inhouse_status == "(missing)"
    assert row.sql_match == "n/a"
    assert row.result_match == "n/a"


def test_compare_result_match_handles_mixed_type_rows(tmp_path: Path, write_jsonl):
    """`(None, 1)`-style rows must not raise TypeError in result-match.

    Previously `_result_match` used `sorted(rows)` which raises on rows
    containing `None` mixed with comparable scalars. We pick two
    textually-distinct but semantically-equivalent SQLs so
    `_sql_match` returns "no" and the buggy `_result_match` path is
    actually exercised (identical strings would short-circuit before
    reaching `_result_match`).
    """
    import sqlite3 as _sqlite3

    db_path = tmp_path / "mixed.sqlite"
    con = _sqlite3.connect(db_path)
    con.execute("CREATE TABLE t (a INTEGER, b INTEGER)")
    con.execute("INSERT INTO t VALUES (NULL, 1)")
    con.execute("INSERT INTO t VALUES (2, NULL)")
    con.commit()
    con.close()

    inhouse = tmp_path / "audited_gold" / "credit" / "credit_audited.jsonl"
    sar = tmp_path / "sar_audited_gold" / "credit" / "credit_sar_audited.jsonl"
    sql_in = "SELECT a, b FROM t ORDER BY rowid"
    sql_sar = "SELECT a, b FROM t WHERE 1 = 1 ORDER BY rowid"
    write_jsonl(inhouse, [_inhouse_row("credit_1", sql_in, sample_row=[None, 1])])
    write_jsonl(
        sar,
        [_sar_row("credit_1", sql_in, sql_sar, "clean", sample_row=[None, 1])],
    )
    out = compare.compare_db(db="credit", inhouse_path=inhouse, sar_path=sar, db_path=db_path)
    row = out.per_instance[0]
    # Texts differ → sql_match is "no", forcing _result_match to run.
    assert row.sql_match == "no"
    # _result_match no longer crashes on (None, int) rows and reports equal.
    assert row.result_match == "yes"


def test_compare_result_match_order_sensitive(tmp_path: Path, write_jsonl):
    """ORDER BY differences must surface as result_match=no, not silently equal."""
    import sqlite3 as _sqlite3

    db_path = tmp_path / "ord.sqlite"
    con = _sqlite3.connect(db_path)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()

    inhouse = tmp_path / "audited_gold" / "credit" / "credit_audited.jsonl"
    sar = tmp_path / "sar_audited_gold" / "credit" / "credit_sar_audited.jsonl"
    # Same rows, different order — must NOT compare equal.
    write_jsonl(
        inhouse,
        [_inhouse_row("credit_1", "SELECT x FROM t ORDER BY x ASC", sample_row=[1])],
    )
    write_jsonl(
        sar,
        [
            _sar_row(
                "credit_1",
                "SELECT x FROM t ORDER BY x ASC",
                "SELECT x FROM t ORDER BY x DESC",
                "edited",
                sample_row=[3],
            )
        ],
    )
    out = compare.compare_db(db="credit", inhouse_path=inhouse, sar_path=sar, db_path=db_path)
    row = out.per_instance[0]
    assert row.result_match == "no"


def test_compare_parse_error_surfaces_as_column(tmp_path: Path, fake_db, write_jsonl):
    """If sqlglot can't parse one side, the column reports parse_error."""
    inhouse = tmp_path / "audited_gold" / "credit" / "credit_audited.jsonl"
    sar = tmp_path / "sar_audited_gold" / "credit" / "credit_sar_audited.jsonl"
    write_jsonl(
        inhouse, [_inhouse_row("credit_1", "SELECT x FROM t ORDER BY x LIMIT 1", sample_row=[1])]
    )
    write_jsonl(
        sar,
        [
            _sar_row(
                "credit_1",
                "SELECT x FROM t ORDER BY x LIMIT 1",
                "##THIS IS NOT SQL##",
                "edited",
                sample_row=[1],
            )
        ],
    )
    out = compare.compare_db(db="credit", inhouse_path=inhouse, sar_path=sar, db_path=fake_db)
    row = out.per_instance[0]
    assert row.sql_match == "parse_error"
    # The failing side and the error text are captured in the payload.
    assert row.sql_match_error is not None
    assert row.sql_match_failing_side in {"sar", "inhouse"}
    assert "##THIS IS NOT SQL##" in row.sql_match_error or "parse" in row.sql_match_error.lower()


def test_compare_exec_error_surfaces_as_column(tmp_path: Path, fake_db, write_jsonl):
    """If executing one side raises, result_match reports exec_error."""
    inhouse = tmp_path / "audited_gold" / "credit" / "credit_audited.jsonl"
    sar = tmp_path / "sar_audited_gold" / "credit" / "credit_sar_audited.jsonl"
    write_jsonl(
        inhouse, [_inhouse_row("credit_1", "SELECT x FROM t ORDER BY x LIMIT 1", sample_row=[1])]
    )
    write_jsonl(
        sar,
        [
            _sar_row(
                "credit_1",
                "SELECT x FROM t ORDER BY x LIMIT 1",
                "SELECT * FROM missing_table",
                "edited",
                sample_row=None,
            )
        ],
    )
    out = compare.compare_db(db="credit", inhouse_path=inhouse, sar_path=sar, db_path=fake_db)
    row = out.per_instance[0]
    assert row.result_match == "exec_error"
    # The failing side and error text are captured in the payload.
    assert row.result_match_error is not None
    assert row.result_match_failing_side in {"sar", "inhouse"}
    assert (
        "no such table" in row.result_match_error.lower()
        or "missing_table" in row.result_match_error.lower()
    )


def test_markdown_output_contains_header_and_counts(tmp_path: Path, fake_db, write_jsonl):
    inhouse = tmp_path / "audited_gold" / "credit" / "credit_audited.jsonl"
    sar = tmp_path / "sar_audited_gold" / "credit" / "credit_sar_audited.jsonl"
    write_jsonl(
        inhouse, [_inhouse_row("credit_1", "SELECT x FROM t ORDER BY x LIMIT 1", sample_row=[1])]
    )
    write_jsonl(
        sar,
        [
            _sar_row(
                "credit_1",
                "SELECT x FROM t ORDER BY x LIMIT 1",
                "SELECT x FROM t ORDER BY x LIMIT 1",
                "clean",
                sample_row=[1],
            )
        ],
    )
    out = compare.compare_db(db="credit", inhouse_path=inhouse, sar_path=sar, db_path=fake_db)
    md = compare.render_markdown(out)
    assert "instance_id" in md
    assert "in-house status" in md
    assert "SAR status" in md
    assert "sql_match" in md
    assert "result_match" in md
    assert "credit_1" in md


def test_json_output_shape(tmp_path: Path, fake_db, write_jsonl):
    inhouse = tmp_path / "audited_gold" / "credit" / "credit_audited.jsonl"
    sar = tmp_path / "sar_audited_gold" / "credit" / "credit_sar_audited.jsonl"
    write_jsonl(
        inhouse, [_inhouse_row("credit_1", "SELECT x FROM t ORDER BY x LIMIT 1", sample_row=[1])]
    )
    write_jsonl(
        sar,
        [
            _sar_row(
                "credit_1",
                "SELECT x FROM t ORDER BY x LIMIT 1",
                "SELECT x FROM t ORDER BY x LIMIT 1",
                "clean",
                sample_row=[1],
            )
        ],
    )
    out_path = tmp_path / "out.json"
    compare.write_json(
        compare.compare_db(db="credit", inhouse_path=inhouse, sar_path=sar, db_path=fake_db),
        out_path,
    )
    payload = json.loads(out_path.read_text())
    assert "db" in payload
    assert payload["db"] == "credit"
    assert "per_instance" in payload
    assert isinstance(payload["per_instance"], list)
    assert payload["per_instance"][0]["instance_id"] == "credit_1"
