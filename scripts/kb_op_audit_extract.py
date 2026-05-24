#!/usr/bin/env python3
"""Extract gold-SQL operator usages from BIRD-Interact mini tasks.

Reads ``mini-interact/mini_interact.jsonl`` and, for each task, parses every
``sol_sql`` string with sqlglot (SQLite dialect), walks the AST, and emits
one JSONL row per (operator, surrounding-context) tuple. Output drives the
KB-vs-data audit (see ``~/.claude/plans/kb-string-operator-audit.md``).

Operators are split into three buckets:

* A. string-hygiene: LOWER, UPPER, TRIM, REPLACE, SUBSTR, INSTR, LENGTH,
  ``||``, STRFTIME
* B. null / conditional: CASE WHEN, COALESCE, NULLIF, IS NULL, IS NOT NULL
* C. typing / JSON: CAST, JSON_EXTRACT, ``->>``, ``->``

Usage::

    uv run python scripts/kb_op_audit_extract.py \\
        --mini-interact-jsonl ../mini-interact/mini_interact.jsonl \\
        --output-dir results/kb_op_audit_$(date +%Y%m%d)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import sqlglot
from sqlglot import expressions as exp

from bird_interact_agents import paths
from bird_interact_agents.hard8_preprocessor import extract_deleted_kb_ids


# Mapping from sqlglot expression class -> (operator_label, bucket).
# IS NULL / IS NOT NULL / -> / ->> need extra logic and live below.
DIRECT_OPERATORS: dict[type, tuple[str, str]] = {
    exp.Lower: ("LOWER", "A"),
    exp.Upper: ("UPPER", "A"),
    exp.Trim: ("TRIM", "A"),
    exp.Substring: ("SUBSTR", "A"),
    exp.Length: ("LENGTH", "A"),
    exp.Replace: ("REPLACE", "A"),
    exp.StrPosition: ("INSTR", "A"),
    exp.DPipe: ("||", "A"),
    exp.TimeToStr: ("STRFTIME", "A"),
    exp.Case: ("CASE WHEN", "B"),
    exp.Coalesce: ("COALESCE", "B"),
    exp.Nullif: ("NULLIF", "B"),
    exp.Cast: ("CAST", "C"),
}

CLAUSE_ANCESTORS: list[tuple[type, str]] = [
    (exp.Where, "WHERE"),
    (exp.Group, "GROUP BY"),
    (exp.Order, "ORDER BY"),
    (exp.Having, "HAVING"),
    (exp.Join, "JOIN ON"),
]


def _column_text(col: exp.Column) -> str:
    """Pretty-print a Column reference as ``table.col`` or ``col``."""
    return col.sql(dialect="sqlite")


def _collect_columns(node: exp.Expression) -> list[str]:
    """All Column references inside *node* (deduped, order-preserving)."""
    seen: dict[str, None] = {}
    for c in node.find_all(exp.Column):
        seen[_column_text(c)] = None
    return list(seen.keys())


def _literal_value(lit: exp.Literal) -> str | int | float:
    """Return the Python value for a Literal, preserving the SQL form for strings."""
    if lit.is_string:
        return lit.this
    text = lit.this
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def _collect_literals(node: exp.Expression) -> list[str | int | float]:
    """All Literal values inside *node* (order-preserving, with duplicates kept)."""
    return [_literal_value(lit) for lit in node.find_all(exp.Literal)]


def _surrounding_clause(node: exp.Expression) -> str:
    """Best-effort label for the SQL clause containing *node*.

    Walks up ``node.parent`` until we hit a clause keyword; falls back to
    ``"SELECT projection"`` if we hit the top-level Select without a more
    specific match. CTE membership is appended (e.g. ``"CTE:t/WHERE"``).
    """
    cte_name: Optional[str] = None
    clause: Optional[str] = None
    cur = node.parent
    while cur is not None:
        if cte_name is None and isinstance(cur, exp.CTE):
            alias = cur.alias_or_name
            cte_name = alias or "?"
        if clause is None:
            for cls, label in CLAUSE_ANCESTORS:
                if isinstance(cur, cls):
                    clause = label
                    break
        cur = cur.parent
    if clause is None:
        clause = "SELECT projection"
    return f"CTE:{cte_name}/{clause}" if cte_name else clause


def _is_null_label(is_node: exp.Is) -> Optional[tuple[str, str]]:
    """Return (label, bucket) for an Is node iff it's IS NULL / IS NOT NULL.

    sqlglot parses ``x IS NOT NULL`` as ``Not(Is(x, Null))``; ``x IS NULL``
    as ``Is(x, Null)``. We only emit one tuple per usage — for the NOT
    variant we attribute it to the Is node and label as ``IS NOT NULL``,
    so the parent ``Not`` is skipped to avoid double-counting.
    """
    right = is_node.args.get("expression")
    if not isinstance(right, exp.Null):
        return None
    parent = is_node.parent
    if isinstance(parent, exp.Not):
        return ("IS NOT NULL", "B")
    return ("IS NULL", "B")


def _json_extract_label(node: exp.JSONExtract) -> tuple[str, str]:
    """Disambiguate ``JSON_EXTRACT(col, '$.x')`` vs ``col -> '$.x'``.

    sqlglot collapses both into ``JSONExtract`` and renders both as
    ``->`` via ``.sql()``. The remaining signal: the ``->`` operator
    form sets ``only_json_types`` in ``node.args``; the
    ``JSON_EXTRACT()`` function form does not.
    """
    if "only_json_types" in node.args:
        return ("->", "C")
    return ("JSON_EXTRACT", "C")


def _operator_for_node(node: exp.Expression) -> Optional[tuple[str, str]]:
    cls = type(node)
    if cls in DIRECT_OPERATORS:
        return DIRECT_OPERATORS[cls]
    if cls is exp.Is:
        return _is_null_label(node)
    if cls is exp.JSONExtract:
        return _json_extract_label(node)
    if cls is exp.JSONExtractScalar:
        return ("->>", "C")
    return None


def extract_tuples_from_sql(
    sql: str,
    *,
    instance_id: str,
    selected_database: str,
    is_hard8: bool,
    deleted_kb_ids: list[int],
    sql_index: int,
) -> list[dict]:
    """Parse one ``sol_sql`` string and emit one record per operator usage.

    Returns a (possibly empty) list of dicts. Raises ``sqlglot.errors.ParseError``
    on malformed SQL — caller decides how to handle.
    """
    tree = sqlglot.parse_one(sql, dialect="sqlite")
    rows: list[dict] = []
    for node in tree.walk():
        op_label = _operator_for_node(node)
        if op_label is None:
            continue
        operator, bucket = op_label
        # For IS NOT NULL we attribute to the inner Is node so we don't
        # double-emit; render the surrounding Not so the `expression`
        # text reads "x IS NOT NULL", not "x IS NULL".
        text_node: exp.Expression = node
        if operator == "IS NOT NULL" and isinstance(node.parent, exp.Not):
            text_node = node.parent
        rows.append(
            {
                "instance_id": instance_id,
                "selected_database": selected_database,
                "is_hard8": is_hard8,
                "deleted_kb_ids": deleted_kb_ids,
                "bucket": bucket,
                "operator": operator,
                "expression": text_node.sql(dialect="sqlite"),
                "target_columns": _collect_columns(node),
                "literal_args": _collect_literals(node),
                "cte_or_clause": _surrounding_clause(node),
                "sql_index": sql_index,
            }
        )
    return rows


def run(
    *,
    mini_interact_jsonl: Path,
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    tuples_path = output_dir / "operator_tuples.jsonl"
    failures_path = output_dir / "parse_failures.txt"

    n_tasks = 0
    n_tasks_with_ops = 0
    n_tuples = 0
    n_parse_failures = 0
    bucket_counts = {"A": 0, "B": 0, "C": 0}

    with mini_interact_jsonl.open() as f, \
            tuples_path.open("w") as tout, \
            failures_path.open("w") as ferr:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            n_tasks += 1
            instance_id = task["instance_id"]
            db = task["selected_database"]
            deleted = sorted(extract_deleted_kb_ids(task))
            is_hard8 = bool(deleted)

            sols = task.get("sol_sql") or []
            task_n = 0
            for sql_index, sql in enumerate(sols):
                try:
                    rows = extract_tuples_from_sql(
                        sql,
                        instance_id=instance_id,
                        selected_database=db,
                        is_hard8=is_hard8,
                        deleted_kb_ids=deleted,
                        sql_index=sql_index,
                    )
                except Exception as exc:  # sqlglot.ParseError or anything else
                    n_parse_failures += 1
                    ferr.write(
                        f"{instance_id}\tsql_index={sql_index}\t{type(exc).__name__}: {exc}\n"
                    )
                    continue
                for row in rows:
                    tout.write(json.dumps(row) + "\n")
                    task_n += 1
                    bucket_counts[row["bucket"]] += 1
            if task_n:
                n_tasks_with_ops += 1
                n_tuples += task_n

    print(f"Tasks read:              {n_tasks}")
    print(f"Tasks with >=1 operator: {n_tasks_with_ops}")
    print(f"Operator tuples emitted: {n_tuples}")
    print(f"  bucket A (string):     {bucket_counts['A']}")
    print(f"  bucket B (null/cond):  {bucket_counts['B']}")
    print(f"  bucket C (typing/JSON):{bucket_counts['C']}")
    print(f"Parse failures:          {n_parse_failures} (see {failures_path})")
    print(f"Output:                  {tuples_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mini-interact-jsonl",
        default=str(paths.mini_interact_data_file()),
        help=f"Path to mini_interact.jsonl (default: {paths.mini_interact_data_file()}).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory; created if missing.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    return run(
        mini_interact_jsonl=Path(args.mini_interact_jsonl).resolve(),
        output_dir=Path(args.output_dir).resolve(),
    )


if __name__ == "__main__":
    sys.exit(main())
