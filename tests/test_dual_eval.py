"""Dual evaluation: when --use-audited-gold-sql is on, score the agent's
submission against BOTH the audited gold (the one the agent interacted
with) AND the original gold (the unedited benchmark reference).

Contract:
* `apply_audited_gold_overlay` preserves the pre-overlay value as
  `task["original_sol_sql"]` whenever it mutates `task["sol_sql"]`.
* `evaluate_dual_gold` runs `execute_submit_action` twice — once with
  audited gold, once with original — returning both p1 verdicts.
* The submit_* helpers dispatch to dual eval iff `original_sol_sql` is
  set on the task; otherwise they keep the current single-eval path.
* `TaskResultRow` carries `phase1_passed_audited` /
  `phase1_passed_original` (+ observation siblings).
* `run.py` aggregates two pass rates into eval.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# `apply_audited_gold_overlay` preserves original_sol_sql
# ---------------------------------------------------------------------------


def _write_audit_sidecar(audited_root: Path, db: str, rows: list[dict]) -> None:
    """Write `<audited_root>/<db>/<db>_audited.jsonl` from a list of dicts."""
    d = audited_root / db
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{db}_audited.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_overlay_preserves_original_sol_sql_for_edited(tmp_path):
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_audit_sidecar(tmp_path, "alien", [
        {"instance_id": "alien_1", "audit_status": "edited",
         "audited_sol_sql": ["SELECT audited FROM t"]},
    ])
    task = {
        "instance_id": "alien_1",
        "selected_database": "alien",
        "sol_sql": ["SELECT original FROM t"],
    }
    log = apply_audited_gold_overlay([task], tmp_path)

    assert log["alien_1"] == "edited"
    assert task["sol_sql"] == ["SELECT audited FROM t"]
    assert task["original_sol_sql"] == ["SELECT original FROM t"]


def test_overlay_preserves_original_for_unrecoverable(tmp_path):
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_audit_sidecar(tmp_path, "alien", [
        {"instance_id": "alien_2", "audit_status": "unrecoverable",
         "audited_sol_sql": ["SELECT rewritten FROM t"]},
    ])
    task = {
        "instance_id": "alien_2",
        "selected_database": "alien",
        "sol_sql": ["SELECT original FROM t"],
    }
    apply_audited_gold_overlay([task], tmp_path)

    assert task["sol_sql"] == ["SELECT rewritten FROM t"]
    assert task["original_sol_sql"] == ["SELECT original FROM t"]


def test_overlay_clean_leaves_no_original_marker(tmp_path):
    """clean rows: no overlay applied → no `original_sol_sql` key should
    appear (otherwise dual-eval would run a wasteful second copy of the
    same SQL on every clean task in a 300-task run)."""
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_audit_sidecar(tmp_path, "alien", [
        {"instance_id": "alien_3", "audit_status": "clean",
         "audited_sol_sql": ["SELECT original FROM t"]},
    ])
    task = {
        "instance_id": "alien_3",
        "selected_database": "alien",
        "sol_sql": ["SELECT original FROM t"],
    }
    apply_audited_gold_overlay([task], tmp_path)

    assert task["sol_sql"] == ["SELECT original FROM t"]
    assert "original_sol_sql" not in task


def test_overlay_missing_audit_leaves_no_original_marker(tmp_path):
    from bird_interact_agents.harness import apply_audited_gold_overlay

    # No sidecar at all for "alien"
    task = {
        "instance_id": "alien_4",
        "selected_database": "alien",
        "sol_sql": ["SELECT original FROM t"],
    }
    log = apply_audited_gold_overlay([task], tmp_path)

    assert log["alien_4"] == "missing-file"
    assert task["sol_sql"] == ["SELECT original FROM t"]
    assert "original_sol_sql" not in task


# ---------------------------------------------------------------------------
# `evaluate_dual_gold` — the new helper
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_status():
    """A fake SampleStatus snapshot. The dual-eval helper may or may not
    mutate `status.original_data["sol_sql"]` — that's an implementation
    detail. Tests assert OBSERVABLE behavior (verdicts + no leftover
    state corruption + no redundant call when golds match)."""
    return SimpleNamespace(
        original_data={
            "selected_database": "alien",
            "sol_sql": ["SELECT audited FROM t"],
            "instance_id": "alien_1",
        },
        current_phase=1,
        remaining_budget=100.0,
        total_budget=100.0,
        force_submit=False,
    )


def _make_dispatcher(verdicts: dict[str, bool]):
    """Build a fake execute_submit_action that returns the verdict from
    `verdicts` keyed by the FIRST string in the active gold list. Tracks
    every call so tests can assert exact dispatch sequence."""
    calls: list[str] = []

    def fake_eval(sol_sql, status, dpb):
        gold = status.original_data["sol_sql"]
        key = gold[0] if isinstance(gold, list) and gold else str(gold)
        calls.append(key)
        passed = verdicts.get(key, False)
        return (f"obs:{key}", 1.0 if passed else 0.0, passed, False, True)

    return fake_eval, calls


def test_evaluate_dual_gold_both_pass(monkeypatch, fake_status):
    """Predicted SQL matches BOTH audited and original golds."""
    from bird_interact_agents import harness

    fake_eval, calls = _make_dispatcher({"SELECT audited FROM t": True,
                                         "SELECT original FROM t": True})
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    result = harness.evaluate_dual_gold(
        pred_sql="SELECT 1",
        audited_sol_sqls=["SELECT audited FROM t"],
        original_sol_sqls=["SELECT original FROM t"],
        status=fake_status,
        data_path_base="/tmp/ignored",
    )

    assert result["audited"]["p1"] is True
    assert result["original"]["p1"] is True
    assert set(calls) == {"SELECT audited FROM t", "SELECT original FROM t"}


def test_evaluate_dual_gold_audited_only(monkeypatch, fake_status):
    """Common case: agent passes audited gold but fails original — the
    edit moved the goalposts."""
    from bird_interact_agents import harness

    fake_eval, _ = _make_dispatcher({"SELECT audited FROM t": True,
                                     "SELECT original FROM t": False})
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    result = harness.evaluate_dual_gold(
        pred_sql="SELECT 1",
        audited_sol_sqls=["SELECT audited FROM t"],
        original_sol_sqls=["SELECT original FROM t"],
        status=fake_status,
        data_path_base="/tmp/ignored",
    )

    assert result["audited"]["p1"] is True
    assert result["original"]["p1"] is False


def test_evaluate_dual_gold_no_redundant_call_when_golds_identical(monkeypatch, fake_status):
    """When audited == original (e.g. on a `clean` task), only ONE eval
    should run — second result re-uses the first. This matters at scale:
    33 extra evals (edited+unrecoverable) is fine; 300 extra evals would
    double the benchmark wall."""
    from bird_interact_agents import harness

    fake_eval, calls = _make_dispatcher({"SELECT same FROM t": True})
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    result = harness.evaluate_dual_gold(
        pred_sql="SELECT 1",
        audited_sol_sqls=["SELECT same FROM t"],
        original_sol_sqls=["SELECT same FROM t"],
        status=fake_status,
        data_path_base="/tmp/ignored",
    )

    assert result["audited"]["p1"] is True
    assert result["original"]["p1"] is True
    assert len(calls) == 1


def test_evaluate_dual_gold_handles_evaluator_exception(monkeypatch, fake_status):
    """If one gold's eval raises (malformed audited row, infra blip,
    whatever), the helper must record the failure for that gold without
    losing the other gold's verdict and without corrupting status."""
    from bird_interact_agents import harness

    def fake_eval(sol_sql, status, dpb):
        gold = status.original_data["sol_sql"][0]
        if "audited" in gold:
            raise RuntimeError("audited eval went boom")
        return ("ok", 1.0, True, False, True)

    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    pre = dict(fake_status.original_data)
    result = harness.evaluate_dual_gold(
        pred_sql="SELECT 1",
        audited_sol_sqls=["SELECT audited FROM t"],
        original_sol_sqls=["SELECT original FROM t"],
        status=fake_status,
        data_path_base="/tmp/ignored",
    )

    assert result["audited"]["p1"] is False
    assert "boom" in result["audited"]["observation"].lower() \
        or "error" in result["audited"]["observation"].lower()
    assert result["original"]["p1"] is True
    # Status untouched after helper returns, regardless of mid-call mutation.
    assert fake_status.original_data == pre


def test_evaluate_dual_gold_propagates_finished(monkeypatch, fake_status, tmp_path):
    """Upstream returns `task_finished=True` whenever phase 1 completes,
    even when no phase 2 exists (mini-interact has no phase 2). The
    helper must preserve that flag — losing it sets `finished: False`
    on every successful audited-task submission."""
    from bird_interact_agents import harness

    def fake_eval(sol_sql, status, dpb):
        # p1=True, p2=False, finished=True — the mini-interact shape.
        return ("ok", 1.0, True, False, True)
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    result = harness.evaluate_dual_gold(
        pred_sql="SELECT 1",
        audited_sol_sqls=["SELECT audited FROM t"],
        original_sol_sqls=["SELECT original FROM t"],
        status=fake_status,
        data_path_base=str(tmp_path),
    )
    assert result["audited"]["finished"] is True
    assert result["original"]["finished"] is True


def test_dispatch_eval_uses_finished_not_p2(monkeypatch, tmp_path):
    """`_dispatch_eval` must return upstream's `finished` flag, not
    `p2`. A phase-1-only task that PASSED should be `finished=True`
    despite `p2=False`."""
    from types import SimpleNamespace
    from bird_interact_agents.agents import _submit
    from bird_interact_agents import harness

    monkeypatch.setattr(_submit, "_dry_run_sql", lambda *a, **kw: None)
    # Phase-1 pass, no phase 2, task IS finished.
    monkeypatch.setattr(harness, "execute_submit_action",
                        lambda sql, status, dpb: ("ok", 1.0, True, False, True))

    state = SimpleNamespace(
        status=SimpleNamespace(
            original_data={
                "selected_database": "fake_db",
                "sol_sql": ["SELECT audited FROM t"],
                "original_sol_sql": ["SELECT original FROM t"],
            },
            remaining_budget=100.0,
            total_budget=100.0,
            force_submit=False,
            current_phase=1,
        ),
        data_path_base=str(tmp_path),
        user_sim_model="x",
        user_sim_prompt_version="v2",
        slayer_storage_dir="",
        result=None,
    )

    _observation, _reward, p1, p2, finished, *_ = _submit._dispatch_eval(state, "SELECT 1")
    assert p1 is True
    assert p2 is False
    assert finished is True  # — NOT False, even though p2 is False.


def test_evaluate_dual_gold_does_not_leave_status_corrupted(monkeypatch, fake_status):
    """After the helper returns, `status.original_data` must look exactly
    as it did before — whether or not the helper temporarily mutated it
    mid-call (implementation choice)."""
    from bird_interact_agents import harness

    monkeypatch.setattr(
        harness, "execute_submit_action",
        lambda sol_sql, status, dpb: ("ok", 1.0, True, False, True),
    )
    pre = dict(fake_status.original_data)
    harness.evaluate_dual_gold(
        pred_sql="SELECT 1",
        audited_sol_sqls=["SELECT audited FROM t"],
        original_sol_sqls=["SELECT original FROM t"],
        status=fake_status,
        data_path_base="/tmp/ignored",
    )
    assert fake_status.original_data == pre


# ---------------------------------------------------------------------------
# `TaskResultRow` gets the four new columns
# ---------------------------------------------------------------------------


def _minimal_row_kwargs() -> dict:
    """Minimal required fields for TaskResultRow construction — mirrors
    the schema in results_db.py."""
    return dict(
        run_id="r1",
        framework="pydantic_ai_recursive",
        mode="a-interact",
        query_mode="slayer",
        instance_id="alien_1",
        database="alien",
        started_at=0.0,
        duration_s=1.0,
        phase1_passed=True,
        phase2_passed=False,
        total_reward=1.0,
        submission_status="passed_phase1",
    )


def test_task_result_row_has_dual_eval_columns():
    """The schema must accept the new dual-eval fields. We instantiate
    with explicit values to verify the fields are wired through."""
    from bird_interact_agents.results_db import TaskResultRow

    row = TaskResultRow(
        **_minimal_row_kwargs(),
        phase1_passed_audited=True,
        phase1_passed_original=False,
        phase1_observation_audited="ok-audit",
        phase1_observation_original="fail-orig",
    )
    assert row.phase1_passed_audited is True
    assert row.phase1_passed_original is False
    assert row.phase1_observation_audited == "ok-audit"
    assert row.phase1_observation_original == "fail-orig"


def test_task_result_row_dual_eval_columns_default_none():
    """Single-eval call sites that don't pass the dual fields still work."""
    from bird_interact_agents.results_db import TaskResultRow

    row = TaskResultRow(**_minimal_row_kwargs())
    assert row.phase1_passed_audited is None
    assert row.phase1_passed_original is None
    assert row.phase1_observation_audited is None
    assert row.phase1_observation_original is None


def test_results_db_open_creates_dual_eval_columns(tmp_path):
    """Schema migration: a fresh results.db must have all four new
    columns so insert_task_result can write them."""
    from bird_interact_agents.results_db import open_db

    db_path = tmp_path / "results.db"
    conn = open_db(db_path)
    try:
        cur = conn.execute("PRAGMA table_info(task_results)")
        cols = {row[1] for row in cur.fetchall()}
    finally:
        conn.close()
    assert "phase1_passed_audited" in cols
    assert "phase1_passed_original" in cols
    assert "phase1_observation_audited" in cols
    assert "phase1_observation_original" in cols


def test_insert_task_result_round_trip_dual_eval_fields(tmp_path):
    """The TaskResultRow → SQL insert must actually persist the new
    fields. The current insert in results_db.py uses an explicit column
    list, so adding model fields without touching the INSERT silently
    drops them — round-trip catches that."""
    from bird_interact_agents.results_db import TaskResultRow, insert_task_result, open_db

    db_path = tmp_path / "results.db"
    conn = open_db(db_path)
    try:
        row = TaskResultRow(
            **_minimal_row_kwargs(),
            phase1_passed_audited=True,
            phase1_passed_original=False,
            phase1_observation_audited="aud-ok",
            phase1_observation_original="orig-fail",
        )
        insert_task_result(conn, row)
        result = conn.execute(
            """SELECT phase1_passed_audited, phase1_passed_original,
                      phase1_observation_audited, phase1_observation_original
               FROM task_results WHERE instance_id = 'alien_1'"""
        ).fetchone()
    finally:
        conn.close()
    assert result == (1, 0, "aud-ok", "orig-fail")


# ---------------------------------------------------------------------------
# `run.py` aggregates dual-eval rates into eval.json
# ---------------------------------------------------------------------------


def test_submit_writes_dual_fields_to_state_result(monkeypatch):
    """Dual-eval fields must land on `state.result` so each framework's
    finalizer can copy them out. Tests `submit_slayer_query`'s plumbing
    end-to-end without going through a real evaluator."""
    from types import SimpleNamespace
    from bird_interact_agents.agents import _submit

    from bird_interact_agents import harness

    monkeypatch.setattr(_submit, "_dry_run_sql", lambda *a, **kw: None)
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)

    def fake_eval(sol_sql, status, dpb):
        gold = status.original_data["sol_sql"][0]
        passed = "audited" in gold
        return (f"obs:{gold}", 1.0 if passed else 0.0, passed, False, True)
    # The dispatcher calls `evaluate_dual_gold` (in harness.py) when
    # `original_sol_sql` is set; that helper in turn calls
    # `harness.execute_submit_action`. Patch the harness binding.
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)
    monkeypatch.setattr(_submit, "execute_submit_action", fake_eval)

    state = SimpleNamespace(
        status=SimpleNamespace(
            original_data={
                "selected_database": "fake_db",
                "sol_sql": ["SELECT audited FROM t"],
                "original_sol_sql": ["SELECT original FROM t"],
            },
            remaining_budget=100.0,
            total_budget=100.0,
            force_submit=False,
            current_phase=1,
        ),
        data_path_base="/tmp/ignored",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
        slayer_storage_dir="",
        result=None,
    )

    fake_client = SimpleNamespace(sql_sync=lambda d: "SELECT 1")
    _submit.submit_slayer_query(
        state,
        query_json='{"models": ["m"]}',
        slayer_client_factory=lambda s: fake_client,
    )
    # The four dual-eval keys MUST be present on state.result — every
    # framework finalizer reads them via submitter.get(...) / result.get(...).
    assert state.result["phase1_passed_audited"] is True
    assert state.result["phase1_passed_original"] is False
    assert "audited" in state.result["phase1_observation_audited"]
    assert "original" in state.result["phase1_observation_original"]


def test_run_aggregation_emits_dual_eval_rates(tmp_path):
    """End of run: eval.json should carry phase1_count_audited /
    phase1_count_original / phase1_rate_audited / phase1_rate_original
    when at least one task has dual-eval columns populated.

    Strategy: write a known set of task results to a fresh results.db,
    then call run.py's aggregation entry point and parse eval.json."""
    pytest.importorskip("bird_interact_agents.run")
    from bird_interact_agents.results_db import TaskResultRow, insert_task_result, open_db
    from bird_interact_agents.run import build_aggregate_eval

    db_path = tmp_path / "results.db"
    conn = open_db(db_path)
    try:
        for i, (aud, orig) in enumerate([(True, True), (True, False), (False, False)], 1):
            insert_task_result(conn, TaskResultRow(
                **{**_minimal_row_kwargs(),
                   "instance_id": f"alien_{i}",
                   "phase1_passed": aud,  # primary = audited
                   "submission_status": "passed_phase1" if aud else "wrong_result"},
                phase1_passed_audited=aud,
                phase1_passed_original=orig,
            ))
        conn.commit()
    finally:
        conn.close()

    agg = build_aggregate_eval(db_path=db_path)

    # 3 tasks: 2 audited-pass, 1 original-pass.
    assert agg["phase1_count_audited"] == 2
    assert agg["phase1_count_original"] == 1
    # Rates are derived from the same n.
    assert agg["phase1_rate_audited"] == pytest.approx(2 / 3)
    assert agg["phase1_rate_original"] == pytest.approx(1 / 3)
