"""T34: extracted `run_one_task` — refactor *contract*, not full parity.

The Step-2 refactor extracts a per-task function out of `run_evaluation`'s
loop body. This file asserts the public contract the cloud actor (and any
other caller) can rely on:

1. `bird_interact_agents.run.run_one_task` exists, is async, and accepts the
   per-task arguments the cloud actor passes in.
2. Its return shape contains exactly the keys the inline `_persist`
   callback in `run_evaluation` consumes — so a `TaskResultRow` can be
   constructed from it without `KeyError`.
3. Errors raised by the per-task body are caught inside `run_one_task` and
   surface as `error` on the returned dict (matching the original inline
   `try/except` behaviour).

End-to-end *parity* with the previous inline-loop implementation is left to
the existing `tests/test_run_db_integration.py` suite, which exercises
`run_evaluation` end-to-end and will continue to pass iff the refactor
preserves behaviour.
"""

from __future__ import annotations

import inspect

import pytest

from bird_interact_agents import run as run_mod  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Symbol + signature contract.
# ---------------------------------------------------------------------------


def test_run_one_task_is_async_callable() -> None:
    fn = getattr(run_mod, "run_one_task", None)
    assert fn is not None, "run_one_task must be exported from run.py"
    assert inspect.iscoroutinefunction(fn)


def test_run_one_task_accepts_all_per_task_kwargs() -> None:
    sig = inspect.signature(run_mod.run_one_task)
    params = set(sig.parameters)
    required = {
        "task_data",
        "data_dir",
        "framework",
        "query_mode",
        "mode",
        "dataset",
        "agent_model",
        "user_sim_model",
        "patience",
        "strict",
        "use_audited_gold_sql",
        "prompt_cache",
        "max_depth",
        "slayer_storage_root",
    }
    missing = required - params
    assert not missing, f"run_one_task missing parameters: {missing}"


# ---------------------------------------------------------------------------
# 2. Return-shape contract — every key `_persist` consumes must be present.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_task_oracle_returns_persist_consumable_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oracle mode is the cheapest probe: no LLM, no real DB walk."""

    async def fake_oracle(td, _data_dir):
        return {
            "task_id": td["instance_id"],
            "instance_id": td["instance_id"],
            "database": td["selected_database"],
            "phase1_passed": True,
            "phase2_passed": True,
            "total_reward": 1.0,
            "submitted_sql": "SELECT 1;",
            "submitted_query": None,
            "trajectory": [],
            "error": None,
            "submission_status": "submitted_correct",
            "phase1_observation": "OK",
            "phase2_observation": None,
            "predicted_result_json": "[[1]]",
            "gold_result_json": "[[1]]",
            "n_agent_turns": 0,
        }

    monkeypatch.setattr(run_mod, "run_oracle_task", fake_oracle)

    row = await run_mod.run_one_task(
        task_data={
            "instance_id": "oracle_smoke_1",
            "selected_database": "california_schools",
            "sol_sql": ["SELECT 1;"],
            "amb_user_query": "smoke",
        },
        data_dir="/tmp/nonexistent",
        framework="pydantic_ai",
        query_mode="raw",
        mode="oracle",
        dataset="mini-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        patience=3,
        strict=False,
        use_audited_gold_sql=False,
        prompt_cache=True,
        max_depth=3,
        slayer_storage_root=None,
    )

    # The exact key set `_persist` consumes inside the original loop body.
    consumed_by_persist = {
        "instance_id",
        "database",
        "phase1_passed",
        "phase2_passed",
        "total_reward",
        "submitted_sql",
        "submitted_query",
        "ground_truth_sql",  # may be None — but the key must be present
        "error",
        "submission_status",
        "phase1_observation",
        "phase2_observation",
        "predicted_result_json",
        "gold_result_json",
        "n_agent_turns",
        "duration_s",
    }
    missing = consumed_by_persist - set(row)
    assert not missing, f"run_one_task return is missing keys: {missing}"


# ---------------------------------------------------------------------------
# 3. Error-capture contract — exceptions are caught and surfaced as `error`.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CR-r3 — run_one_task_with_runner guards post-processing against non-dict
# runner returns (None, etc.) by raising a TypeError that the same
# try/except converts into an error row.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_one_task_with_runner_handles_non_dict_return() -> None:
    """A misbehaving custom runner that returns None must NOT crash the
    actor with a TypeError on `r["duration_s"]` — it must produce an
    `error` row like any other per-task failure."""

    async def bad_runner(td, data_dir, patience, user_sim_model):
        return None  # type: ignore[return-value]

    row = await run_mod.run_one_task_with_runner(
        bad_runner,
        task_data={
            "instance_id": "bad_runner_1",
            "selected_database": "db_a",
            "sol_sql": ["SELECT 1"],
        },
        data_dir="/tmp",
        patience=3,
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
    )
    assert isinstance(row, dict)
    assert row["instance_id"] == "bad_runner_1"
    assert row.get("error")
    assert "dict" in row["error"]
    assert row["phase1_passed"] is False


@pytest.mark.asyncio
async def test_run_one_task_with_runner_passes_dict_returns_through() -> None:
    """The happy path: a runner that returns a proper dict must NOT be
    treated as an error."""

    async def good_runner(td, data_dir, patience, user_sim_model):
        return {
            "instance_id": td["instance_id"],
            "database": td["selected_database"],
            "phase1_passed": True,
            "phase2_passed": True,
            "total_reward": 1.0,
            "error": None,
        }

    row = await run_mod.run_one_task_with_runner(
        good_runner,
        task_data={
            "instance_id": "good_1",
            "selected_database": "db_a",
            "sol_sql": ["SELECT 1"],
        },
        data_dir="/tmp",
        patience=3,
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
    )
    assert row["error"] is None
    assert row["phase1_passed"] is True


@pytest.mark.asyncio
async def test_run_one_task_catches_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The previous inline loop wrapped each task in try/except and surfaced
    failures as a `finalize_result_row` with `error` set. The extracted
    function MUST preserve that contract — otherwise the cloud actor would
    have to re-implement it."""

    async def boom(_td, _data_dir):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(run_mod, "run_oracle_task", boom)

    row = await run_mod.run_one_task(
        task_data={
            "instance_id": "oracle_boom",
            "selected_database": "x",
            "sol_sql": [""],
        },
        data_dir="/tmp/x",
        framework="pydantic_ai",
        query_mode="raw",
        mode="oracle",
        dataset="mini-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        patience=3,
        strict=False,
        use_audited_gold_sql=False,
        prompt_cache=True,
        max_depth=3,
        slayer_storage_root=None,
    )

    assert row["instance_id"] == "oracle_boom"
    assert row.get("error")
    assert "kaboom" in row["error"]
    assert row["phase1_passed"] is False
    assert row["phase2_passed"] is False
