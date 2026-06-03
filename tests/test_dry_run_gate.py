"""Dry-run gate: catch invented SQL functions / columns BEFORE the agent
pays the 3-coin submit cost. The gate runs the candidate SQL read-only
against the per-task SQLite DB; on `OperationalError` (missing function /
missing column / etc.) it returns the error free of charge, classifying
the submission as `dry_run_error`. On success it falls through to the
existing paid `execute_submit_action` path.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from bird_interact_agents.harness import ACTION_COSTS


class _FakeState(SimpleNamespace):
    """Mirror of the fixture in test_submit_helpers.py — only the fields
    the helpers touch."""

    def __init__(self, *, db_name: str = "fake_db", data_path_base: str = "/tmp/ignored", **kw):
        defaults = dict(
            status=SimpleNamespace(
                original_data={"selected_database": db_name},
                remaining_budget=100.0,
                total_budget=100.0,
                force_submit=False,
            ),
            data_path_base=data_path_base,
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            user_sim_prompt_version="v2",
            slayer_storage_dir="",
            result=None,
        )
        defaults.update(kw)
        super().__init__(**defaults)


@pytest.fixture
def tiny_db(tmp_path):
    """Lay out a `<base>/<db>/<db>.sqlite` tree that matches the harness'
    expected layout (`data_path_base/<db>/<db>.sqlite`), so the dry-run
    helper opens the right file."""
    db_name = "tinydb"
    db_dir = tmp_path / db_name
    db_dir.mkdir()
    db_path = db_dir / f"{db_name}.sqlite"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE things (id INTEGER, label TEXT)")
        conn.execute("INSERT INTO things VALUES (1, 'one'), (2, 'two')")
        conn.commit()
    finally:
        conn.close()
    return {"data_path_base": str(tmp_path), "db_name": db_name, "db_path": str(db_path)}


# ---------------------------------------------------------------------------
# `_dry_run_sql` — the standalone helper
# ---------------------------------------------------------------------------


def test_dry_run_returns_none_on_valid_sql(tiny_db):
    from bird_interact_agents.agents import _submit

    err = _submit._dry_run_sql(
        "SELECT id, label FROM things WHERE id = 1",
        data_path_base=tiny_db["data_path_base"],
        db_name=tiny_db["db_name"],
    )
    assert err is None


def test_dry_run_catches_missing_function(tiny_db):
    """PERCENTILE_CONT et al. are the dominant SQL_RUNTIME failure mode —
    the agent invents them, SQLite raises `no such function`. The dry-run
    gate must surface this without burning the submit budget."""
    from bird_interact_agents.agents import _submit

    err = _submit._dry_run_sql(
        "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY id) FROM things",
        data_path_base=tiny_db["data_path_base"],
        db_name=tiny_db["db_name"],
    )
    assert err is not None
    assert "no such function" in err.lower() or "syntax error" in err.lower()


def test_dry_run_catches_missing_column(tiny_db):
    """The other invented-name mode — `no such column` — should look
    identical to the agent: an error string, no charge."""
    from bird_interact_agents.agents import _submit

    err = _submit._dry_run_sql(
        "SELECT does_not_exist FROM things",
        data_path_base=tiny_db["data_path_base"],
        db_name=tiny_db["db_name"],
    )
    assert err is not None
    assert "does_not_exist" in err or "no such column" in err.lower()


def test_dry_run_catches_syntax_error(tiny_db):
    from bird_interact_agents.agents import _submit

    err = _submit._dry_run_sql(
        "SELEKT * FROM things",
        data_path_base=tiny_db["data_path_base"],
        db_name=tiny_db["db_name"],
    )
    assert err is not None


def test_dry_run_honours_explicit_db_file_path(tmp_path):
    """Upstream `_resolve_sqlite_db_path` checks `record['db_file_path']`
    FIRST before falling back to `data_path_base/<db>/<db>.sqlite`. The
    dry-run helper must match — otherwise tasks with an explicit DB path
    get rejected for free against the wrong file."""
    from bird_interact_agents.agents import _submit

    # Lay out a DB at a NON-standard location.
    db_path = tmp_path / "elsewhere" / "custom.sqlite"
    db_path.parent.mkdir()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE custom (n INTEGER)")
        conn.execute("INSERT INTO custom VALUES (42)")
        conn.commit()
    finally:
        conn.close()

    # `data_path_base` is unrelated — the standard layout doesn't exist.
    err = _submit._dry_run_sql(
        "SELECT n FROM custom",
        data_path_base=str(tmp_path / "nonexistent"),
        db_name="ignored",
        db_file_path=str(db_path),
    )
    assert err is None

    # And a real error against the same explicit path surfaces.
    err = _submit._dry_run_sql(
        "SELECT no_such_col FROM custom",
        data_path_base=str(tmp_path / "nonexistent"),
        db_name="ignored",
        db_file_path=str(db_path),
    )
    assert err is not None


def test_dry_run_skips_when_db_missing(tmp_path):
    """If the per-task DB doesn't exist at the expected path, the dry-run
    can't run — the helper should return None so the submission falls
    through to the paid path (rather than blocking valid submissions on
    an env mis-config)."""
    from bird_interact_agents.agents import _submit

    err = _submit._dry_run_sql(
        "SELECT 1",
        data_path_base=str(tmp_path),
        db_name="no_such_db",
    )
    assert err is None


def test_dry_run_read_only_connection_rejects_writes(tiny_db):
    """A write statement must surface as a dry-run error AND must not
    actually mutate the DB. Read-only sqlite raises on writes; the
    helper should surface that to the agent rather than silently
    swallowing it and falling through to a paid (and write-attempting)
    submission."""
    from bird_interact_agents.agents import _submit

    err = _submit._dry_run_sql(
        "UPDATE things SET label = 'mutated' WHERE id = 1",
        data_path_base=tiny_db["data_path_base"],
        db_name=tiny_db["db_name"],
    )
    # Read-only mode rejects writes; the error must propagate.
    assert err is not None

    conn = sqlite3.connect(tiny_db["db_path"])
    try:
        row = conn.execute("SELECT label FROM things WHERE id = 1").fetchone()
    finally:
        conn.close()
    assert row == ("one",)


# ---------------------------------------------------------------------------
# `submit_slayer_query` — integration with the new gate
# ---------------------------------------------------------------------------


def _stub_no_op(*a, **kw):
    return None


def test_slayer_dry_run_failure_does_not_charge_budget(monkeypatch):
    from bird_interact_agents.agents import _submit

    monkeypatch.setattr(_submit, "_dry_run_sql", lambda *a, **kw: "no such function: PERCENTILE_CONT")
    # If the gate works, execute_submit_action is never reached.
    monkeypatch.setattr(_submit, "execute_submit_action",
                        lambda *a, **kw: pytest.fail("execute_submit_action must NOT be called on dry-run failure"))
    monkeypatch.setattr(_submit, "capture_result_snapshot", _stub_no_op)
    fake_client = SimpleNamespace(sql_sync=lambda d: "SELECT PERCENTILE_CONT(0.5) FROM t")

    state = _FakeState()
    start = state.status.remaining_budget
    out = _submit.submit_slayer_query(
        state,
        query_json='{"models": ["m"]}',
        slayer_client_factory=lambda s: fake_client,
    )

    assert "no such function" in out.lower()
    assert state.result["submission_status"] == "dry_run_error"
    assert state.result["phase1_passed"] is False
    # Free of charge — same rule as json_failed and translation_failed.
    assert state.status.remaining_budget == start


def test_slayer_dry_run_success_falls_through_to_paid(monkeypatch):
    from bird_interact_agents.agents import _submit

    monkeypatch.setattr(_submit, "_dry_run_sql", lambda *a, **kw: None)
    monkeypatch.setattr(_submit, "execute_submit_action",
                        lambda sql, status, dpb: ("ok", 1.0, True, False, True))
    monkeypatch.setattr(_submit, "capture_result_snapshot", _stub_no_op)
    fake_client = SimpleNamespace(sql_sync=lambda d: "SELECT 1")

    state = _FakeState()
    start = state.status.remaining_budget
    out = _submit.submit_slayer_query(
        state,
        query_json='{"models": ["m"]}',
        slayer_client_factory=lambda s: fake_client,
    )

    assert state.result["phase1_passed"] is True
    assert state.result["submission_status"] == "passed_phase1"
    assert state.status.remaining_budget == start - ACTION_COSTS["submit_query"]


def test_slayer_dry_run_runs_after_translation_gate(monkeypatch):
    """Compilation failures (translation_failed) must still short-circuit
    BEFORE the dry-run helper is reached — otherwise we'd dry-run None."""
    from bird_interact_agents.agents import _submit

    called = {"dry": 0}

    def _spy_dry(*a, **kw):
        called["dry"] += 1
        return None

    class _BoomClient:
        def sql_sync(self, _):
            raise RuntimeError("boom")

    monkeypatch.setattr(_submit, "_dry_run_sql", _spy_dry)
    monkeypatch.setattr(_submit, "capture_result_snapshot", _stub_no_op)
    state = _FakeState()
    out = _submit.submit_slayer_query(
        state,
        query_json='{"models": []}',
        slayer_client_factory=lambda s: _BoomClient(),
    )
    assert "Could not generate SQL" in out
    assert state.result["submission_status"] == "translation_error"
    assert called["dry"] == 0


# ---------------------------------------------------------------------------
# `submit_raw_sql` — integration with the new gate
# ---------------------------------------------------------------------------


def test_raw_dry_run_failure_does_not_charge_budget(monkeypatch):
    from bird_interact_agents.agents import _submit

    monkeypatch.setattr(_submit, "_dry_run_sql", lambda *a, **kw: "no such column: bogus")
    monkeypatch.setattr(_submit, "execute_submit_action",
                        lambda *a, **kw: pytest.fail("execute_submit_action must NOT be called"))
    monkeypatch.setattr(_submit, "capture_result_snapshot", _stub_no_op)

    state = _FakeState()
    start = state.status.remaining_budget
    out = _submit.submit_raw_sql(state, "SELECT bogus FROM t")

    assert "no such column" in out.lower()
    assert state.result["submission_status"] == "dry_run_error"
    assert state.result["phase1_passed"] is False
    assert state.status.remaining_budget == start


def test_raw_dry_run_success_falls_through_to_paid(monkeypatch):
    from bird_interact_agents.agents import _submit

    monkeypatch.setattr(_submit, "_dry_run_sql", lambda *a, **kw: None)
    monkeypatch.setattr(_submit, "execute_submit_action",
                        lambda sql, status, dpb: ("ok", 1.0, True, False, True))
    monkeypatch.setattr(_submit, "capture_result_snapshot", _stub_no_op)

    state = _FakeState()
    start = state.status.remaining_budget
    _submit.submit_raw_sql(state, "SELECT 1")

    assert state.result["phase1_passed"] is True
    assert state.status.remaining_budget == start - ACTION_COSTS["submit_sql"]


# ---------------------------------------------------------------------------
# `classify_submission` accepts the new flag
# ---------------------------------------------------------------------------


def test_capture_result_snapshot_accepts_explicit_db_file_path(tmp_path):
    """capture_result_snapshot must respect db_file_path so diagnostic
    snapshots match the DB the eval actually ran against. Symmetric to
    the dry-run path fix."""
    from bird_interact_agents.agents import _submit

    explicit_db = tmp_path / "custom" / "explicit.sqlite"
    explicit_db.parent.mkdir()
    conn = sqlite3.connect(str(explicit_db))
    try:
        conn.execute("CREATE TABLE marker (n INTEGER)")
        conn.execute("INSERT INTO marker VALUES (42)")
        conn.commit()
    finally:
        conn.close()

    snap = _submit.capture_result_snapshot(
        "SELECT n FROM marker",
        db_name="ignored",
        data_path_base=str(tmp_path / "nonexistent"),
        db_file_path=str(explicit_db),
    )
    assert isinstance(snap, dict)
    assert snap.get("row_count") == 1
    assert snap.get("sample_rows") == [[42]]


def test_diagnostic_payload_threads_db_file_path_into_snapshots(monkeypatch):
    """_diagnostic_payload must extract `db_file_path` from
    `sample_status.original_data` and pass it to BOTH the predicted
    and gold snapshot calls."""
    from types import SimpleNamespace
    from bird_interact_agents.agents import _submit

    seen = []
    def spy(sql, db_name, data_path_base, db_file_path=None):
        seen.append({"sql": sql, "db_name": db_name,
                     "data_path_base": data_path_base,
                     "db_file_path": db_file_path})
        return {"columns": [], "row_count": 0, "row_count_truncated": False, "sample_rows": []}
    monkeypatch.setattr(_submit, "capture_result_snapshot", spy)

    status = SimpleNamespace(
        original_data={
            "selected_database": "fake_db",
            "sol_sql": ["SELECT 1"],
            "db_file_path": "/explicit/path.sqlite",
        },
        current_phase=1,
    )
    _submit._diagnostic_payload(
        submitted_sql="SELECT pred",
        sample_status=status,
        data_path_base="/real/base",
        observation="ok",
        p1=True, p2=False,
    )
    assert len(seen) == 2  # predicted + gold
    assert all(call["db_file_path"] == "/explicit/path.sqlite" for call in seen)


def test_dry_run_prefers_template_db_when_present(tmp_path):
    """When `<db>_template.sqlite` exists alongside `<db>.sqlite`, the
    dry-run should run against the template — that's the canonical
    reset state the paid eval will see, not whatever the per-task
    connection has mutated to."""
    from bird_interact_agents.agents import _submit

    db_name = "tmpldb"
    db_dir = tmp_path / db_name
    db_dir.mkdir()

    # Live DB has a row that will FAIL the dry-run query.
    live = db_dir / f"{db_name}.sqlite"
    conn = sqlite3.connect(str(live))
    try:
        conn.execute("CREATE TABLE t (n INTEGER)")
        conn.commit()
    finally:
        conn.close()

    # Template has a different schema — same table, additional col `flag`
    # that the dry-run query selects.
    tmpl = db_dir / f"{db_name}_template.sqlite"
    conn = sqlite3.connect(str(tmpl))
    try:
        conn.execute("CREATE TABLE t (n INTEGER, flag INTEGER)")
        conn.execute("INSERT INTO t VALUES (1, 0)")
        conn.commit()
    finally:
        conn.close()

    # `SELECT flag FROM t` would fail on the live DB (no `flag` column),
    # succeed on the template. If dry-run prefers template, returns None.
    err = _submit._dry_run_sql(
        "SELECT flag FROM t",
        data_path_base=str(tmp_path),
        db_name=db_name,
    )
    assert err is None


def test_classify_submission_dry_run_error():
    from bird_interact_agents.agents._submit import classify_submission

    assert classify_submission(p1=False, p2=False, observation=None,
                               dry_run_failed=True) == "dry_run_error"


def test_classify_submission_dry_run_flag_dominates_other_flags():
    """If dry_run_failed is set, that's the diagnostic the user wants —
    even if other flags are set somehow (defensive)."""
    from bird_interact_agents.agents._submit import classify_submission

    assert classify_submission(
        p1=False, p2=False, observation=None,
        dry_run_failed=True, infrastructure_failed=True,
    ) == "dry_run_error"


# ---------------------------------------------------------------------------
# Submit fns pass the right `data_path_base` and `db_name` to the helper
# ---------------------------------------------------------------------------


def test_slayer_submit_passes_correct_db_args_to_dry_run(monkeypatch):
    """submit_slayer_query must call _dry_run_sql with the task's actual
    data_path_base and selected_database. Otherwise the dry-run executes
    against the wrong DB and the no-charge short-circuit triggers on
    irrelevant errors."""
    from bird_interact_agents.agents import _submit

    captured = {}

    def _spy(sql, *, data_path_base, db_name, db_file_path=None, benchmark=None):
        captured["sql"] = sql
        captured["data_path_base"] = data_path_base
        captured["db_name"] = db_name
        captured["db_file_path"] = db_file_path
        return None

    monkeypatch.setattr(_submit, "_dry_run_sql", _spy)
    monkeypatch.setattr(_submit, "execute_submit_action",
                        lambda sql, status, dpb: ("ok", 1.0, True, False, True))
    monkeypatch.setattr(_submit, "capture_result_snapshot", _stub_no_op)
    fake_client = SimpleNamespace(sql_sync=lambda d: "SELECT 1")

    state = _FakeState(db_name="real_db", data_path_base="/real/base")
    _submit.submit_slayer_query(
        state,
        query_json='{"models": ["m"]}',
        slayer_client_factory=lambda s: fake_client,
    )
    assert captured["data_path_base"] == "/real/base"
    assert captured["db_name"] == "real_db"
    assert captured["sql"] == "SELECT 1"


def test_raw_submit_passes_correct_db_args_to_dry_run(monkeypatch):
    from bird_interact_agents.agents import _submit

    captured = {}

    def _spy(sql, *, data_path_base, db_name, db_file_path=None, benchmark=None):
        captured["sql"] = sql
        captured["data_path_base"] = data_path_base
        captured["db_name"] = db_name
        captured["db_file_path"] = db_file_path
        return None

    monkeypatch.setattr(_submit, "_dry_run_sql", _spy)
    monkeypatch.setattr(_submit, "execute_submit_action",
                        lambda sql, status, dpb: ("ok", 1.0, True, False, True))
    monkeypatch.setattr(_submit, "capture_result_snapshot", _stub_no_op)

    state = _FakeState(db_name="real_db", data_path_base="/real/base")
    _submit.submit_raw_sql(state, "SELECT 1")
    assert captured["data_path_base"] == "/real/base"
    assert captured["db_name"] == "real_db"
    assert captured["sql"] == "SELECT 1"
