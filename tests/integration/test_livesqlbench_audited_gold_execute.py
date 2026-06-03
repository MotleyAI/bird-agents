"""DEV-1510: integration tests — every audited_sol_sql in the museum audit
file actually executes against `museum.sqlite` and produces a sensible
result-set.

Marked `@integration` because they depend on the gitignored livesqlbench
data root (which ships the museum sqlite file). In CI without that data,
these are skipped; locally they're the front-line sanity that the audit
rewrites didn't introduce SQL syntax errors or refer to columns that
don't exist.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

from bird_interact_agents import paths


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def museum_sqlite() -> Path:
    """Path to the museum sqlite the audit rewrites are validated against.

    The benchmark prepares per-task copies for evaluation (`per_task_db_isolation`),
    but `<db>.sqlite` (or `<db>_template.sqlite`) is the canonical schema-
    bearing file. We open read-only so concurrent runs are safe.
    """
    root = paths.benchmark_data_root("livesqlbench-base-lite-sqlite")
    candidates = [
        root / "museum" / "museum.sqlite",
        root / "museum" / "museum_template.sqlite",
    ]
    for c in candidates:
        if c.exists():
            return c
    pytest.skip(
        f"museum sqlite not found at any of {[str(c) for c in candidates]}. "
        f"Run `bird-interact prepare-livesqlbench` or pull the upstream data "
        f"to enable these integration tests."
    )


@pytest.fixture(scope="module")
def audit_rows() -> dict[str, dict]:
    """Load the audited-gold rows keyed by instance_id."""
    path = paths.audited_gold_root() / "livesqlbench_audited.jsonl"
    if not path.exists():
        pytest.skip(
            f"audited-gold deliverable not present: {path}; this PR's tests "
            f"are meaningful only after the file ships."
        )
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["instance_id"]] = row
    return out


def _open_readonly(sqlite_path: Path) -> sqlite3.Connection:
    """Read-only connection so a concurrent eval-reset can't deadlock us
    and so the audit tests never accidentally mutate the canonical db."""
    uri = f"file:{sqlite_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance_id", [f"museum_{i}" for i in range(1, 11)],
)
def test_audited_sol_sql_executes_without_error(
    museum_sqlite: Path, audit_rows: dict[str, dict], instance_id: str,
) -> None:
    """Every string in `audited_sol_sql` must execute against the museum
    sqlite without raising. Catches:
      - syntax errors the auditor introduced during the rewrite,
      - references to columns/tables that don't exist (case-sensitivity
        is a classic gotcha — SQLite is case-insensitive for keywords
        but the audit author may have copy-pasted a misspelled column),
      - dialect drift (museum gold was authored against Postgres; the
        SQLite gold is a translation — we audit the SQLite variant).

    Loops over the FULL `audited_sol_sql` list (not just `[0]`), so a
    multi-statement audit with a malformed second statement is caught.
    """
    row = audit_rows[instance_id]
    audited = row["audited_sol_sql"]
    assert isinstance(audited, list) and audited, (
        f"{instance_id}: audited_sol_sql must be a non-empty list"
    )
    con = _open_readonly(museum_sqlite)
    try:
        for j, sql in enumerate(audited):
            try:
                con.execute(sql).fetchall()
            except sqlite3.Error as e:
                pytest.fail(
                    f"{instance_id}: audited_sol_sql[{j}] failed to execute "
                    f"against museum.sqlite: {type(e).__name__}: {e}\n"
                    f"SQL: {sql!r}"
                )
    finally:
        con.close()


@pytest.mark.parametrize(
    "instance_id", [f"museum_{i}" for i in range(1, 11)],
)
def test_original_sol_sql_still_executes(
    museum_sqlite: Path, audit_rows: dict[str, dict], instance_id: str,
) -> None:
    """Sanity: the recorded `original_sol_sql` (the canonical gold) MUST
    also run on the same db. If it doesn't, the dual-eval would compare
    a passing audited against an evaluator-crashing original — that's a
    false `phase1_passed_audited > phase1_passed_original` signal."""
    row = audit_rows[instance_id]
    original = row["original_sol_sql"]
    assert isinstance(original, list) and original, (
        f"{instance_id}: original_sol_sql must be a non-empty list"
    )
    con = _open_readonly(museum_sqlite)
    try:
        for j, sql in enumerate(original):
            try:
                con.execute(sql).fetchall()
            except sqlite3.Error as e:
                pytest.fail(
                    f"{instance_id}: original_sol_sql[{j}] failed to execute "
                    f"against museum.sqlite: {type(e).__name__}: {e}\n"
                    f"SQL: {sql!r}"
                )
    finally:
        con.close()


@pytest.mark.parametrize(
    "instance_id", [f"museum_{i}" for i in range(1, 11)],
)
def test_audited_sample_row_matches_execution(
    museum_sqlite: Path, audit_rows: dict[str, dict], instance_id: str,
) -> None:
    """`audited_sample_row` is the first row of running
    `audited_sol_sql[0]` against the db.

    Contract:
    - If the SQL is deterministic-ordered (`ORDER BY` clause present),
      the RECORDED first row must equal `rows[0]` of a fresh execution.
    - If the SQL has no ORDER BY, SQLite's row order is implementation-
      defined; we only check that the recorded sample is SOMEWHERE in
      the current result-set. Stricter than that risks false positives
      when SQLite's internal storage order shifts (e.g. after an
      `ANALYZE` rebuild).
    - If the audited SQL returns no rows today, the recorded sample
      must be `[]`.

    Either way, stale sample rows hint that the audit was authored
    against a different museum.sqlite than the one shipped today —
    worth investigating.
    """
    row = audit_rows[instance_id]
    audited = row["audited_sol_sql"]
    sample = row["audited_sample_row"]
    sql = audited[0]
    con = _open_readonly(museum_sqlite)
    try:
        rows = con.execute(sql).fetchall()
    finally:
        con.close()

    if not rows:
        assert sample == [], (
            f"{instance_id}: audited_sol_sql[0] returns no rows but "
            f"audited_sample_row is {sample!r} (expected [])"
        )
        return

    norm_sample = _normalise(sample)
    norm_rows = [_normalise(list(r)) for r in rows]

    if re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE):
        # Deterministic order — strict first-row equality.
        assert norm_rows[0] == norm_sample, (
            f"{instance_id}: audited_sample_row drift (ORDER BY present, "
            f"expected first-row equality).\n"
            f"  Recorded: {sample!r}\n"
            f"  Re-execute first row: {rows[0]!r}\n"
            f"  audited_sol_sql[0]: {sql!r}"
        )
    else:
        # No ORDER BY — sample must still be IN the result set, just
        # not necessarily at position 0.
        assert norm_sample in norm_rows, (
            f"{instance_id}: audited_sample_row drift (no ORDER BY — "
            f"checked membership).\n"
            f"  Recorded: {sample!r}\n"
            f"  Result set head: {[list(r) for r in rows[:5]]!r}\n"
            f"  audited_sol_sql[0]: {sql!r}"
        )


def test_at_least_one_task_returns_more_audited_rows_than_original(
    museum_sqlite: Path, audit_rows: dict[str, dict],
) -> None:
    """Acceptance-criterion sanity (best-effort local proxy for the cloud
    rerun): at least one museum task must have an `edited` audit whose
    result-set differs from the original — otherwise the audit cannot
    move the dial on `phase1_passed_audited > phase1_passed_original` in
    the cloud rerun. museum_7 is the locked example, so we expect at
    LEAST it to differ."""
    differing = []
    con = _open_readonly(museum_sqlite)
    try:
        for iid, row in audit_rows.items():
            if row["audit_status"] != "edited":
                continue
            audited_rows = con.execute(row["audited_sol_sql"][0]).fetchall()
            original_rows = con.execute(row["original_sol_sql"][0]).fetchall()
            if sorted(audited_rows) != sorted(original_rows):
                differing.append(iid)
    finally:
        con.close()
    assert "museum_7" in differing, (
        "museum_7 audited and original gold MUST produce different result-sets "
        f"(the locked decision is a KB-canonical rewrite); got differing="
        f"{differing}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(value):
    """Map SQLite-returned types to JSON-comparable canonical forms.

    Tuples → lists. Floats with integer values → ints (to absorb the
    `1` vs `1.0` ambiguity that comes out of SQLite for AVG/SUM). Strs
    are stripped of trailing whitespace (CHAR(10) padding from the
    schema's fixed-width keys).
    """
    if isinstance(value, tuple):
        return [_normalise(v) for v in value]
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return value.rstrip()
    return value
