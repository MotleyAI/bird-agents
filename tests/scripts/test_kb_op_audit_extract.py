"""Unit tests for ``scripts/kb_op_audit_extract.py``.

Verifies that the sqlglot-based AST walker produces the right
(operator, bucket) tuples for each gold-SQL operator we care about,
including nested cases and the IS NOT NULL / -> / ->> disambiguation.
The classifier and summarizer steps are not tested here (separate smoke
test).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


extract = _load_module(
    "kb_op_audit_extract",
    SCRIPTS_DIR / "kb_op_audit_extract.py",
)


def _extract(sql: str) -> list[dict]:
    return extract.extract_tuples_from_sql(
        sql,
        instance_id="t1",
        selected_database="db",
        is_hard8=False,
        deleted_kb_ids=[],
        sql_index=0,
    )


def _ops(rows: list[dict]) -> list[str]:
    return [r["operator"] for r in rows]


def _buckets(rows: list[dict]) -> set[str]:
    return {r["bucket"] for r in rows}


# ---------- Bucket A: string-hygiene ----------------------------------------


def test_lower_simple():
    rows = _extract("SELECT LOWER(name) FROM t")
    assert _ops(rows) == ["LOWER"]
    assert rows[0]["bucket"] == "A"
    assert rows[0]["target_columns"] == ["name"]
    assert rows[0]["literal_args"] == []


def test_upper_simple():
    rows = _extract("SELECT UPPER(name) FROM t")
    assert _ops(rows) == ["UPPER"]
    assert rows[0]["bucket"] == "A"


def test_trim_simple():
    rows = _extract("SELECT TRIM(name) FROM t")
    assert _ops(rows) == ["TRIM"]


def test_substr():
    rows = _extract("SELECT SUBSTR(name, 1, 3) FROM t")
    assert _ops(rows) == ["SUBSTR"]
    assert rows[0]["target_columns"] == ["name"]
    assert rows[0]["literal_args"] == [1, 3]


def test_length():
    rows = _extract("SELECT LENGTH(name) FROM t WHERE LENGTH(other) > 5")
    ops = _ops(rows)
    assert ops.count("LENGTH") == 2


def test_replace():
    rows = _extract("SELECT REPLACE(name, 'a', 'b') FROM t")
    assert _ops(rows) == ["REPLACE"]
    assert "a" in rows[0]["literal_args"] and "b" in rows[0]["literal_args"]


def test_instr():
    rows = _extract("SELECT INSTR(name, 'x') FROM t")
    assert _ops(rows) == ["INSTR"]


def test_dpipe_concat():
    rows = _extract("SELECT a || '-' || b FROM t")
    # Two || operators: a||'-' and (a||'-')||b — sqlglot makes them nested.
    ops = _ops(rows)
    assert ops.count("||") == 2
    for r in rows:
        assert r["bucket"] == "A"


def test_strftime():
    rows = _extract("SELECT STRFTIME('%Y', d) FROM t")
    assert _ops(rows) == ["STRFTIME"]


# ---------- Nested string ops ----------------------------------------------


def test_nested_lower_trim():
    """LOWER(TRIM(x)) emits two tuples: outer LOWER and inner TRIM."""
    rows = _extract("SELECT LOWER(TRIM(name)) FROM t")
    ops = _ops(rows)
    assert "LOWER" in ops and "TRIM" in ops
    # Both target the same column.
    assert all(r["target_columns"] == ["name"] for r in rows)


def test_lower_replace_combo():
    rows = _extract("SELECT LOWER(REPLACE(name, ' ', '_')) FROM t")
    assert set(_ops(rows)) == {"LOWER", "REPLACE"}


# ---------- Bucket B: null / conditional -----------------------------------


def test_case_when_simple():
    rows = _extract(
        "SELECT CASE WHEN x > 0 THEN 'pos' WHEN x < 0 THEN 'neg' "
        "ELSE 'zero' END FROM t"
    )
    case_rows = [r for r in rows if r["operator"] == "CASE WHEN"]
    assert len(case_rows) == 1
    assert case_rows[0]["bucket"] == "B"
    assert case_rows[0]["target_columns"] == ["x"]
    assert {"pos", "neg", "zero"} <= set(case_rows[0]["literal_args"])


def test_coalesce():
    rows = _extract("SELECT COALESCE(a, b, c) FROM t")
    assert _ops(rows) == ["COALESCE"]
    assert set(rows[0]["target_columns"]) == {"a", "b", "c"}


def test_nullif():
    rows = _extract("SELECT NULLIF(a, '') FROM t")
    assert _ops(rows) == ["NULLIF"]


def test_is_null():
    rows = _extract("SELECT * FROM t WHERE x IS NULL")
    assert _ops(rows) == ["IS NULL"]
    assert rows[0]["cte_or_clause"] == "WHERE"


def test_is_not_null():
    """IS NOT NULL should emit ONE tuple labeled 'IS NOT NULL', not two,
    and the rendered ``expression`` text should include the NOT.
    """
    rows = _extract("SELECT * FROM t WHERE x IS NOT NULL")
    assert _ops(rows) == ["IS NOT NULL"]
    assert "NOT" in rows[0]["expression"].upper()


def test_is_null_mixed():
    rows = _extract("SELECT * FROM t WHERE a IS NULL OR b IS NOT NULL")
    ops = sorted(_ops(rows))
    assert ops == ["IS NOT NULL", "IS NULL"]


# ---------- Bucket C: typing / JSON ----------------------------------------


def test_cast():
    rows = _extract("SELECT CAST(x AS REAL) FROM t")
    assert _ops(rows) == ["CAST"]
    assert rows[0]["bucket"] == "C"


def test_json_extract_function_form():
    rows = _extract("SELECT JSON_EXTRACT(data, '$.name') FROM t")
    assert _ops(rows) == ["JSON_EXTRACT"]
    assert rows[0]["bucket"] == "C"


def test_json_arrow_extract():
    rows = _extract("SELECT data -> '$.name' FROM t")
    # sqlglot may parse `->` to JSONExtract; we should label it as `->`.
    assert rows and rows[0]["operator"] == "->"
    assert rows[0]["bucket"] == "C"


def test_json_arrow2_extract_scalar():
    rows = _extract("SELECT data ->> '$.name' FROM t")
    assert _ops(rows) == ["->>"]
    assert rows[0]["bucket"] == "C"


# ---------- Surrounding-clause + CTE attribution ---------------------------


def test_clause_where():
    rows = _extract("SELECT * FROM t WHERE LOWER(name) = 'x'")
    lower_rows = [r for r in rows if r["operator"] == "LOWER"]
    assert lower_rows[0]["cte_or_clause"] == "WHERE"


def test_clause_group_by():
    rows = _extract("SELECT LOWER(name), COUNT(*) FROM t GROUP BY LOWER(name)")
    lower_rows = [r for r in rows if r["operator"] == "LOWER"]
    clauses = sorted({r["cte_or_clause"] for r in lower_rows})
    assert "GROUP BY" in clauses
    assert "SELECT projection" in clauses


def test_cte_attribution():
    rows = _extract(
        "WITH foo AS (SELECT LOWER(name) AS n FROM t) SELECT * FROM foo"
    )
    lower_rows = [r for r in rows if r["operator"] == "LOWER"]
    assert lower_rows[0]["cte_or_clause"].startswith("CTE:foo/")


# ---------- Real BIRD-Interact-style task SQL ------------------------------


REALISTIC_BIRD_SQL = """
SELECT CASE
    WHEN p.TechSigProb * (1 - p.NatSrcProb) * p.SigUnique * (0.5 + p.AnomScore/10) < 0.25 THEN 'Low'
    WHEN p.TechSigProb * (1 - p.NatSrcProb) * p.SigUnique * (0.5 + p.AnomScore/10) < 0.75 THEN 'Medium'
    ELSE 'High'
  END AS tol_category,
  COUNT(*) AS signal_count,
  AVG(s.BwHz/(s.CenterFreqMhz * 1000000)) AS avg_bfr,
  AVG(p.AnomScore*p.AnomScore) - AVG(p.AnomScore)*AVG(p.AnomScore) AS anomaly_stddev
FROM Signals s JOIN SignalProbabilities p ON s.SignalRegistry = p.SignalRef
GROUP BY tol_category
"""


def test_realistic_bird_task_alien_2():
    rows = _extract(REALISTIC_BIRD_SQL)
    ops = _ops(rows)
    # CASE WHEN with two thresholds + ELSE → exactly one CASE WHEN tuple.
    assert ops.count("CASE WHEN") == 1
    case_row = next(r for r in rows if r["operator"] == "CASE WHEN")
    assert {"Low", "Medium", "High"} <= set(case_row["literal_args"])
    # No string-hygiene ops in this SQL.
    assert _buckets(rows) == {"B"}


# ---------- Output payload sanity ------------------------------------------


def test_metadata_propagated():
    rows = extract.extract_tuples_from_sql(
        "SELECT LOWER(name) FROM t",
        instance_id="hard_X",
        selected_database="my_db",
        is_hard8=True,
        deleted_kb_ids=[7, 12],
        sql_index=2,
    )
    assert rows[0]["instance_id"] == "hard_X"
    assert rows[0]["selected_database"] == "my_db"
    assert rows[0]["is_hard8"] is True
    assert rows[0]["deleted_kb_ids"] == [7, 12]
    assert rows[0]["sql_index"] == 2


def test_no_operators_returns_empty():
    assert _extract("SELECT a, b FROM t WHERE a = b") == []


# ---------- Adversarial: malformed SQL -------------------------------------


def test_malformed_sql_raises():
    """The function itself raises ``sqlglot.errors.ParseError``; the
    caller (run) is responsible for logging to parse_failures.txt and
    continuing.
    """
    from sqlglot.errors import ParseError

    with pytest.raises(ParseError):
        _extract("SELECT FROM WHERE GARBAGE )(")


def test_run_logs_parse_failures(tmp_path):
    """End-to-end: a task with malformed sol_sql writes a row to
    parse_failures.txt and does not crash the run.
    """
    jsonl = tmp_path / "mini.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "instance_id": "ok_1",
                "selected_database": "db",
                "sol_sql": ["SELECT LOWER(name) FROM t"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "instance_id": "bad_1",
                "selected_database": "db",
                "sol_sql": ["SELECT FROM WHERE GARBAGE )("],
            }
        )
        + "\n"
    )
    out_dir = tmp_path / "out"
    rc = extract.run(mini_interact_jsonl=jsonl, output_dir=out_dir)
    assert rc == 0
    tuples = (out_dir / "operator_tuples.jsonl").read_text().splitlines()
    assert len(tuples) == 1
    assert json.loads(tuples[0])["instance_id"] == "ok_1"
    failures = (out_dir / "parse_failures.txt").read_text()
    assert "bad_1" in failures
