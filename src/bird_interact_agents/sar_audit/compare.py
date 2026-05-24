"""SAR-audit vs in-house audit comparison."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Literal

import sqlglot
from pydantic import BaseModel

from . import io


_MatchTri = Literal["yes", "no", "n/a", "parse_error", "exec_error"]


class PerInstanceCompare(BaseModel):
    instance_id: str
    inhouse_status: str
    sar_status: str
    sql_match: _MatchTri
    sql_match_error: str | None = None
    sql_match_failing_side: Literal["inhouse", "sar"] | None = None
    result_match: _MatchTri
    result_match_error: str | None = None
    result_match_failing_side: Literal["inhouse", "sar"] | None = None
    sample_row_match: _MatchTri


class CompareDB(BaseModel):
    db: str
    per_instance: list[PerInstanceCompare]


def compare_db(*, db: str, inhouse_path: Path, sar_path: Path, db_path: Path) -> CompareDB:
    inhouse_rows = {r["instance_id"]: r for r in io.read_existing_rows(inhouse_path)}
    sar_rows = {r["instance_id"]: r for r in io.read_existing_rows(sar_path)}
    # `inhouse_path` may have rows that lack our newer required keys; reload it
    # leniently so the comparison sees everything in the file.
    inhouse_rows = {r["instance_id"]: r for r in _read_jsonl_lenient(inhouse_path)}

    all_ids = sorted(set(inhouse_rows) | set(sar_rows))
    out: list[PerInstanceCompare] = []
    for iid in all_ids:
        inhouse = inhouse_rows.get(iid)
        sar = sar_rows.get(iid)
        out.append(_compare_one(iid, inhouse, sar, db_path))
    return CompareDB(db=db, per_instance=out)


def _read_jsonl_lenient(path: Path) -> list[dict]:
    """Read a JSONL file, dropping malformed rows. Only returns dict
    rows with a non-empty string `instance_id` — downstream code indexes
    by that key."""
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            iid = row.get("instance_id")
            if not isinstance(iid, str) or not iid:
                continue
            out.append(row)
    return out


def _compare_one(
    instance_id: str, inhouse: dict | None, sar: dict | None, db_path: Path
) -> PerInstanceCompare:
    if inhouse is None and sar is None:
        # Unreachable given the caller, but keep safe.
        return PerInstanceCompare(
            instance_id=instance_id,
            inhouse_status="(missing)",
            sar_status="(missing)",
            sql_match="n/a",
            result_match="n/a",
            sample_row_match="n/a",
        )

    if inhouse is None:
        return PerInstanceCompare(
            instance_id=instance_id,
            inhouse_status="(missing)",
            sar_status=sar["audit_status"],  # type: ignore[index]
            sql_match="n/a",
            result_match="n/a",
            sample_row_match="n/a",
        )

    if sar is None:
        return PerInstanceCompare(
            instance_id=instance_id,
            inhouse_status=inhouse["audit_status"],
            sar_status="(missing)",
            sql_match="n/a",
            result_match="n/a",
            sample_row_match="n/a",
        )

    sql_in = inhouse["audited_sol_sql"][0]
    sql_sar = sar["audited_sol_sql"][0]
    sql_match, sql_err, sql_failing = _sql_match(sql_in, sql_sar)

    result_match: _MatchTri
    result_err: str | None = None
    result_failing: Literal["inhouse", "sar"] | None = None
    if sql_match == "yes":
        result_match = "yes"
    else:
        result_match, result_err, result_failing = _result_match(sql_in, sql_sar, db_path)

    row_in = inhouse.get("audited_sample_row")
    row_sar = sar.get("audited_sample_row")
    sample_row_match: _MatchTri = "yes" if row_in == row_sar else "no"

    return PerInstanceCompare(
        instance_id=instance_id,
        inhouse_status=inhouse["audit_status"],
        sar_status=sar["audit_status"],
        sql_match=sql_match,
        sql_match_error=sql_err,
        sql_match_failing_side=sql_failing,
        result_match=result_match,
        result_match_error=result_err,
        result_match_failing_side=result_failing,
        sample_row_match=sample_row_match,
    )


def _sql_match(
    sql_in: str, sql_sar: str
) -> tuple[_MatchTri, str | None, Literal["inhouse", "sar"] | None]:
    try:
        ast_in = sqlglot.parse_one(sql_in, dialect="sqlite")
    except Exception as e:
        return "parse_error", f"{type(e).__name__}: {e} -- failing sql: {sql_in}", "inhouse"
    try:
        ast_sar = sqlglot.parse_one(sql_sar, dialect="sqlite")
    except Exception as e:
        return "parse_error", f"{type(e).__name__}: {e} -- failing sql: {sql_sar}", "sar"
    norm_in = ast_in.sql(dialect="sqlite", normalize=True)
    norm_sar = ast_sar.sql(dialect="sqlite", normalize=True)
    return ("yes" if norm_in == norm_sar else "no", None, None)


def _result_match(
    sql_in: str, sql_sar: str, db_path: Path
) -> tuple[_MatchTri, str | None, Literal["inhouse", "sar"] | None]:
    """Compare two SQLs' result sets in execution order.

    We DON'T sort the rows: (a) `sorted()` raises TypeError on mixed-type
    rows like (None, int), and (b) sorting hides ORDER BY regressions —
    if two SQLs return identical rows in different order, that IS a real
    divergence we want to surface.
    """
    target = f"file:{db_path}?mode=ro&immutable=1"
    try:
        con_in = sqlite3.connect(target, uri=True)
        try:
            rows_in = con_in.execute(sql_in).fetchall()
        finally:
            con_in.close()
    except sqlite3.Error as e:
        return "exec_error", str(e), "inhouse"
    try:
        con_sar = sqlite3.connect(target, uri=True)
        try:
            rows_sar = con_sar.execute(sql_sar).fetchall()
        finally:
            con_sar.close()
    except sqlite3.Error as e:
        return "exec_error", str(e), "sar"
    return ("yes" if rows_in == rows_sar else "no", None, None)


def render_markdown(out: CompareDB) -> str:
    headers = [
        "instance_id",
        "in-house status",
        "SAR status",
        "sql_match",
        "result_match",
        "sample_row_match",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in out.per_instance:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.instance_id,
                    row.inhouse_status,
                    row.sar_status,
                    row.sql_match,
                    row.result_match,
                    row.sample_row_match,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_json(out: CompareDB, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out.model_dump(), indent=2))
