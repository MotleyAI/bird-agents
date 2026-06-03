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


def test_overlay_clean_sets_original_equal_to_sol(tmp_path):
    """clean rows: `sol_sql` is unchanged but `original_sol_sql` is still
    set (= the same SQL) so EVERY task scores against the original gold
    (user directive: ALWAYS score against original). `evaluate_dual_gold`
    short-circuits when the two golds are identical, so this costs no extra
    evaluator call on clean tasks."""
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
    assert task["original_sol_sql"] == ["SELECT original FROM t"]


def test_overlay_missing_audit_sets_original_equal_to_sol(tmp_path):
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
    # Even with no audited sidecar, original_sol_sql is set to the task's
    # own sol_sql so the row dual-evaluates (against an identical gold).
    assert task["original_sol_sql"] == ["SELECT original FROM t"]


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


# DEV-1515: the per-row `phase1_passed_audited` / `phase1_passed_original`
# raw bool columns have been removed; the per-task cascade verdict now
# lives in the SubmissionAnnotation. Tests covering that schema are in
# `tests/test_schema_extension.py` and the cascading aggregator tests in
# `tests/test_cascading_report.py`. The three tests previously here
# (`test_task_result_row_has_dual_eval_columns`,
# `test_task_result_row_dual_eval_columns_default_none`,
# `test_results_db_open_creates_dual_eval_columns`) are intentionally
# removed.


# ---------------------------------------------------------------------------
# DEV-1510: `apply_audited_gold_overlay` learns a `benchmark` kwarg and a
# `single_file` dispatch so livesqlbench (whose audited gold is one consolidated
# `audited_gold/livesqlbench_audited.jsonl`, not per-db dirs) can flow through
# the same code path as mini-interact. The per_db branch must stay
# bit-identical for back-compat (existing tests above pin that).
# ---------------------------------------------------------------------------


def _write_single_file_audit(audited_root: Path, rows: list[dict]) -> Path:
    """Lay down `<audited_root>/livesqlbench_audited.jsonl` from `rows`."""
    audited_root.mkdir(parents=True, exist_ok=True)
    path = audited_root / "livesqlbench_audited.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_overlay_single_file_basic_swap_for_edited(tmp_path):
    """The single_file dispatch reads `<root>/livesqlbench_audited.jsonl` once
    and swaps `sol_sql` for `edited` rows just like the per_db branch."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            "instance_id": "museum_7",
            "selected_database": "museum",
            "benchmark": "livesqlbench",
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT audited FROM t"],
        },
    ])
    task = {
        "instance_id": "museum_7",
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    log = apply_audited_gold_overlay(
        [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
    )

    assert log["museum_7"] == "edited"
    assert task["sol_sql"] == ["SELECT audited FROM t"]
    assert task["original_sol_sql"] == ["SELECT original FROM t"]


def test_overlay_single_file_clean_keeps_original_and_stamps_snapshot(tmp_path):
    """`clean` rows leave `sol_sql` untouched and still stamp the
    `original_sol_sql` snapshot — exactly mirrors per_db semantics."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            "instance_id": "museum_9",
            "selected_database": "museum",
            "benchmark": "livesqlbench",
            "audit_status": "clean",
        },
    ])
    task = {
        "instance_id": "museum_9",
        "selected_database": "museum",
        "sol_sql": ["SELECT identical FROM t"],
    }
    log = apply_audited_gold_overlay(
        [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
    )

    assert log["museum_9"] == "clean"
    assert task["sol_sql"] == ["SELECT identical FROM t"]
    assert task["original_sol_sql"] == ["SELECT identical FROM t"]


def test_overlay_single_file_missing_file_logs_missing_for_all(tmp_path):
    """No `livesqlbench_audited.jsonl` at all → every task in the run logs
    `missing-file`; `sol_sql` untouched; `original_sol_sql` still stamped
    so dual-eval can still run against the same gold on both sides."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    task = {
        "instance_id": "museum_7",
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    log = apply_audited_gold_overlay(
        [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
    )

    assert log["museum_7"] == "missing-file"
    assert task["sol_sql"] == ["SELECT original FROM t"]
    assert task["original_sol_sql"] == ["SELECT original FROM t"]


def test_overlay_single_file_missing_row(tmp_path):
    """The audit file exists but has no row for this task — log
    `missing-row`, leave `sol_sql` untouched."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            "instance_id": "museum_1",
            "selected_database": "museum",
            "audit_status": "clean",
        },
    ])
    task = {
        "instance_id": "museum_7",  # not in the audit file
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    log = apply_audited_gold_overlay(
        [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
    )

    assert log["museum_7"] == "missing-row"
    assert task["sol_sql"] == ["SELECT original FROM t"]


def test_overlay_single_file_row_db_mismatch_is_missing_row(tmp_path, caplog):
    """The single_file layout relies on `instance_id` being globally unique
    within the benchmark — and on the row's `selected_database` matching
    the task's. A mismatch indicates a corrupt audit row (cross-benchmark
    instance_id clash); treat as missing-row + log a warning, never
    silently apply the wrong audit."""
    import logging

    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            "instance_id": "museum_7",
            "selected_database": "WRONG_db",  # mismatch
            "benchmark": "livesqlbench",
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT audited FROM t"],
        },
    ])
    task = {
        "instance_id": "museum_7",
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    with caplog.at_level(logging.WARNING, logger="bird_interact_agents.harness"):
        log = apply_audited_gold_overlay(
            [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
        )

    assert log["museum_7"] == "missing-row"
    assert task["sol_sql"] == ["SELECT original FROM t"]
    # Warning surfaced — operator must see this in the logs.
    assert any(
        "museum_7" in rec.getMessage() and "WRONG_db" in rec.getMessage()
        for rec in caplog.records
    ), f"expected a db-mismatch warning, got: {[r.getMessage() for r in caplog.records]}"


def test_overlay_single_file_row_with_no_selected_database_is_missing_row(
    tmp_path, caplog,
):
    """Codex DEV-1510 review follow-up: a row that's missing
    `selected_database` entirely would slip past the cross-benchmark guard
    if the check only rejected mismatching values (not absent ones). The
    single_file layout REQUIRES the per-DB discriminator — applying an
    audit based on `instance_id` alone defeats the protection that
    motivated the single-file layout in the first place."""
    import logging

    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            # NB: no `selected_database` field at all.
            "instance_id": "museum_7",
            "benchmark": "livesqlbench",
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT audited FROM t"],
        },
    ])
    task = {
        "instance_id": "museum_7",
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    with caplog.at_level(logging.WARNING, logger="bird_interact_agents.harness"):
        log = apply_audited_gold_overlay(
            [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
        )

    assert log["museum_7"] == "missing-row", (
        "missing selected_database must be rejected, NOT silently applied"
    )
    assert task["sol_sql"] == ["SELECT original FROM t"], (
        "sol_sql must NOT have been swapped to the corrupt row's audited_sol_sql"
    )
    # A warning about the missing discriminator must surface.
    assert any(
        "museum_7" in rec.getMessage() and "selected_database" in rec.getMessage()
        for rec in caplog.records
    ), (
        "expected a missing-selected_database warning; got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_overlay_single_file_row_with_empty_selected_database_is_missing_row(
    tmp_path,
):
    """Same guard, with `selected_database` present but empty-string —
    `not row_db` rejects both forms (None and empty)."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            "instance_id": "museum_7",
            "selected_database": "",
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT audited FROM t"],
        },
    ])
    task = {
        "instance_id": "museum_7",
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    log = apply_audited_gold_overlay(
        [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
    )

    assert log["museum_7"] == "missing-row"
    assert task["sol_sql"] == ["SELECT original FROM t"]


def test_overlay_single_file_row_with_wrong_benchmark_is_missing_row(
    tmp_path, caplog,
):
    """Codex post-review-fix follow-up: DB names overlap across benchmarks
    by design (alien, museum, … exist in both mini-interact and livesqlbench
    — that's WHY single_file exists). A row with the right (instance_id,
    selected_database) but the wrong `benchmark` tag would slip past the
    selected_database guard, so the overlay must also verify the
    `benchmark` field matches the active benchmark."""
    import logging

    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            "instance_id": "museum_7",
            "selected_database": "museum",
            "benchmark": "mini_interact",  # wrong benchmark
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT audited FROM t"],
        },
    ])
    task = {
        "instance_id": "museum_7",
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    with caplog.at_level(logging.WARNING, logger="bird_interact_agents.harness"):
        log = apply_audited_gold_overlay(
            [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
        )

    assert log["museum_7"] == "missing-row"
    assert task["sol_sql"] == ["SELECT original FROM t"]
    assert any(
        "museum_7" in rec.getMessage() and "mini_interact" in rec.getMessage()
        for rec in caplog.records
    ), (
        "expected a wrong-benchmark warning; got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_overlay_single_file_row_with_no_benchmark_is_missing_row(tmp_path):
    """Same guard, with the `benchmark` field absent — the schema requires
    it, so missing is treated the same as mismatch."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            "instance_id": "museum_7",
            "selected_database": "museum",
            # NB: no `benchmark` field.
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT audited FROM t"],
        },
    ])
    task = {
        "instance_id": "museum_7",
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    log = apply_audited_gold_overlay(
        [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
    )

    assert log["museum_7"] == "missing-row"
    assert task["sol_sql"] == ["SELECT original FROM t"]


def test_overlay_single_file_unrecoverable_swaps_and_records(tmp_path):
    """`unrecoverable` swaps `sol_sql` to the audited version too (same
    semantics as `edited`) and records the status."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            "instance_id": "museum_5",
            "selected_database": "museum",
            "benchmark": "livesqlbench",
            "audit_status": "unrecoverable",
            "audited_sol_sql": ["SELECT fallback FROM t"],
        },
    ])
    task = {
        "instance_id": "museum_5",
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    log = apply_audited_gold_overlay(
        [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
    )

    assert log["museum_5"] == "unrecoverable"
    assert task["sol_sql"] == ["SELECT fallback FROM t"]
    assert task["original_sol_sql"] == ["SELECT original FROM t"]


def test_overlay_single_file_reads_audit_jsonl_only_once(tmp_path, monkeypatch):
    """Running over a list of N tasks against the single_file layout MUST
    open the audit file once (not N times) — at-scale (180 livesqlbench
    tasks) the per-task open would dominate the overlay wall time."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents import harness

    _write_single_file_audit(tmp_path, [
        {"instance_id": f"museum_{i}", "selected_database": "museum",
         "benchmark": "livesqlbench", "audit_status": "clean"}
        for i in range(1, 11)
    ])
    tasks = [
        {"instance_id": f"museum_{i}", "selected_database": "museum",
         "sol_sql": [f"SELECT {i} FROM t"]} for i in range(1, 11)
    ]

    open_calls = {"n": 0}
    real_open = Path.open

    def counting_open(self, *args, **kwargs):
        if self.name == "livesqlbench_audited.jsonl":
            open_calls["n"] += 1
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    harness.apply_audited_gold_overlay(
        tasks, tmp_path, benchmark=get_benchmark("livesqlbench"),
    )
    assert open_calls["n"] == 1, (
        f"expected one read of the single-file audit, got {open_calls['n']}"
    )


def test_overlay_benchmark_kwarg_default_preserves_per_db_behavior(tmp_path):
    """Existing call sites (mini-interact tests above) call the overlay
    without a `benchmark` kwarg. The default MUST keep dispatching via
    the per_db layout so this PR doesn't break those callers (and so the
    cloud upload-back / dual-eval flow stays bit-identical for mini-interact)."""
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_audit_sidecar(tmp_path, "alien", [
        {"instance_id": "alien_default", "audit_status": "edited",
         "audited_sol_sql": ["SELECT audited FROM t"]},
    ])
    task = {
        "instance_id": "alien_default",
        "selected_database": "alien",
        "sol_sql": ["SELECT original FROM t"],
    }
    # Note: NO `benchmark=` kwarg — verifies the default is per_db.
    log = apply_audited_gold_overlay([task], tmp_path)
    assert log["alien_default"] == "edited"
    assert task["sol_sql"] == ["SELECT audited FROM t"]


def test_overlay_benchmark_kwarg_mini_interact_uses_single_file(tmp_path):
    """DEV-1515: passing `benchmark=mini_interact` dispatches to the
    consolidated ``mini_interact_audited.jsonl`` (not the legacy per_db
    sidecar). Proves the layout dispatch follows the descriptor and that
    mini-interact's new single-file shape is wired through end-to-end."""
    import json
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    single_file = tmp_path / "mini_interact_audited.jsonl"
    single_file.write_text(json.dumps({
        "instance_id": "alien_explicit",
        "selected_database": "alien",
        "benchmark": "mini_interact",
        "variant_id": "primary",
        "primary": True,
        "audit_status": "edited",
        "audited_sol_sql": ["SELECT audited FROM t"],
    }) + "\n")
    task = {
        "instance_id": "alien_explicit",
        "selected_database": "alien",
        "sol_sql": ["SELECT original FROM t"],
    }
    log = apply_audited_gold_overlay(
        [task], tmp_path, benchmark=get_benchmark("mini_interact"),
    )
    assert log["alien_explicit"] == "edited"
    assert task["sol_sql"] == ["SELECT audited FROM t"]


# DEV-1515: the three end-of-pipeline dual-eval tests previously here
# (`test_insert_task_result_round_trip_dual_eval_fields`,
# `test_submit_writes_dual_fields_to_state_result`,
# `test_run_aggregation_emits_dual_eval_rates`) covered the removed
# pre-DEV-1515 per-task raw bool persistence + state.result emission +
# eval.json rates. The equivalents under the cascade are in
# `tests/test_local_run_cascading.py`,
# `tests/test_cascading_report.py`, and the legacy-removal grep-sweep in
# `tests/test_legacy_field_removal.py`.


# ---------------------------------------------------------------------------
# Codex r9: multi-variant audited gold rows for the same instance_id
# (one ``primary=True`` + N alternates) must NOT let an alternate
# overwrite the primary row at index-build time. Pre-fix both index
# helpers (``harness._load_single_file_audited_rows`` via the overlay
# AND ``cloud._audited_gold_check._load_single_file_audit_index``)
# wrote with latest-wins semantics. These two tests pin the new
# primary-first contract — alternates listed AFTER the primary in the
# file must lose the contest.
# ---------------------------------------------------------------------------


def test_overlay_single_file_prefers_primary_when_alternate_listed_after(
    tmp_path,
):
    """Multi-variant file: primary row first, alternate row second.
    The overlay MUST keep the primary's ``audited_sol_sql``."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            "instance_id": "museum_9",
            "selected_database": "museum",
            "benchmark": "livesqlbench",
            "variant_id": "primary",
            "primary": True,
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT primary_reading FROM t"],
        },
        {
            "instance_id": "museum_9",
            "selected_database": "museum",
            "benchmark": "livesqlbench",
            "variant_id": "alt_a",
            "primary": False,
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT alt_reading FROM t"],
        },
    ])
    task = {
        "instance_id": "museum_9",
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    apply_audited_gold_overlay(
        [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
    )
    assert task["sol_sql"] == ["SELECT primary_reading FROM t"], (
        "primary row MUST win over alternates regardless of file order"
    )


def test_overlay_single_file_prefers_primary_when_alternate_listed_first(
    tmp_path,
):
    """Symmetric case: alternate row FIRST, primary row second. The
    primary must still take precedence at the end."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.harness import apply_audited_gold_overlay

    _write_single_file_audit(tmp_path, [
        {
            "instance_id": "museum_9",
            "selected_database": "museum",
            "benchmark": "livesqlbench",
            "variant_id": "alt_a",
            "primary": False,
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT alt_reading FROM t"],
        },
        {
            "instance_id": "museum_9",
            "selected_database": "museum",
            "benchmark": "livesqlbench",
            "variant_id": "primary",
            "primary": True,
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT primary_reading FROM t"],
        },
    ])
    task = {
        "instance_id": "museum_9",
        "selected_database": "museum",
        "sol_sql": ["SELECT original FROM t"],
    }
    apply_audited_gold_overlay(
        [task], tmp_path, benchmark=get_benchmark("livesqlbench"),
    )
    assert task["sol_sql"] == ["SELECT primary_reading FROM t"], (
        "primary row MUST win regardless of where it lands in the file"
    )


def test_cloud_audit_index_prefers_primary_over_alternate(tmp_path):
    """``cloud._audited_gold_check._load_single_file_audit_index``
    is the cloud-side guard against an audited gold layout drift —
    same primary-first rule must hold there."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._audited_gold_check import (
        _load_single_file_audit_index,
    )

    benchmark = get_benchmark("livesqlbench")
    audit_path = tmp_path / f"{benchmark.name}_audited.jsonl"
    audit_path.write_text(
        json.dumps({
            "instance_id": "museum_9",
            "selected_database": "museum",
            "benchmark": "livesqlbench",
            "variant_id": "alt_a",
            "primary": False,
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT alt_reading FROM t"],
        }) + "\n"
        + json.dumps({
            "instance_id": "museum_9",
            "selected_database": "museum",
            "benchmark": "livesqlbench",
            "variant_id": "primary",
            "primary": True,
            # Deliberately different from the alt so we can tell which
            # row landed in the index.
            "audit_status": "clean",
            "audited_sol_sql": [],
        }) + "\n"
    )

    index = _load_single_file_audit_index(tmp_path, benchmark)
    assert index is not None
    status, has_audited_sql, _row_db, _row_bench = index["museum_9"]
    # ``clean`` is the primary's status; ``edited`` is the alt's.
    assert status == "clean", (
        f"primary row's audit_status must survive against the alt's; "
        f"got status={status!r} (alt's status was 'edited')"
    )
    assert has_audited_sql is False, (
        "primary's empty audited_sol_sql must be the one indexed, "
        "not the alt's non-empty list"
    )
