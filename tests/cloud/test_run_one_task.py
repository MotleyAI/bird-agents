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
async def test_run_one_task_caps_runaway_at_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEV-1535 r3 (Codex): the wall-clock cap originally landed only
    in `run_one_task_with_runner` (the `cached_runner` path used for
    raw query_mode). SLayer / non-raw runs fall through to
    `run_one_task` — pre-fix that path ran uncapped, defeating the
    cap on exactly the runs most likely to thrash. The cap now wraps
    both entry points."""
    import asyncio as _asyncio

    monkeypatch.setenv("BIRD_INTERACT_PER_TASK_TIMEOUT_S", "0.2")
    # DEV-1555 follow-up: outer wait_for cap = agent budget + runaway
    # grace. Zero the grace in tests so the outer cap fires inside the
    # test sleep window.
    monkeypatch.setenv("BIRD_INTERACT_RUNAWAY_GRACE_S", "0")

    async def thrasher(td, data_dir, patience, user_sim_model):
        await _asyncio.sleep(5.0)
        return {"instance_id": td["instance_id"]}

    # Replace _make_runner so we exercise the real run_one_task path.
    monkeypatch.setattr(run_mod, "_make_runner", lambda **_kw: thrasher)
    monkeypatch.setattr(run_mod, "_validate_slayer_setup", lambda **_kw: None)

    row = await run_mod.run_one_task(
        task_data={
            "instance_id": "thrasher_slayer_1",
            "selected_database": "db_a",
            "sol_sql": ["SELECT 1"],
        },
        data_dir="/tmp",
        framework="claude_sdk",
        dataset="mini-interact",
        query_mode="slayer",
        mode="a-interact",
        agent_model="anthropic/claude-opus-4-7",
        user_sim_model="anthropic/claude-sonnet-4-6",
        patience=3,
        strict=False,
        use_audited_gold_sql=False,
        prompt_cache=True,
        max_depth=3,
        slayer_storage_root=None,
        slayer_setup="on-the-fly",
        reasoning_effort=None,
    )
    assert isinstance(row, dict)
    assert row["instance_id"] == "thrasher_slayer_1"
    assert row.get("error")
    assert "timeout" in row["error"].lower()
    assert row["phase1_passed"] is False
    assert row["duration_s"] < 2.0


@pytest.mark.asyncio
async def test_run_one_task_with_runner_caps_runaway_at_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEV-1535 wall-clock cap. A runner that loops past the per-task
    timeout (configured here to 0.2 s) must be killed by `asyncio.wait_for`
    and produce an error row — NOT block the actor indefinitely."""
    import asyncio as _asyncio

    monkeypatch.setenv("BIRD_INTERACT_PER_TASK_TIMEOUT_S", "0.2")
    # DEV-1555 follow-up: outer wait_for cap = agent budget + runaway
    # grace. Zero the grace in tests so the outer cap fires inside the
    # test sleep window.
    monkeypatch.setenv("BIRD_INTERACT_RUNAWAY_GRACE_S", "0")

    async def thrasher(td, data_dir, patience, user_sim_model):
        await _asyncio.sleep(5.0)  # well past the 0.2 s cap
        return {"instance_id": td["instance_id"]}

    row = await run_mod.run_one_task_with_runner(
        thrasher,
        task_data={
            "instance_id": "thrasher_1",
            "selected_database": "db_a",
            "sol_sql": ["SELECT 1"],
        },
        data_dir="/tmp",
        patience=3,
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
    )
    assert isinstance(row, dict)
    assert row["instance_id"] == "thrasher_1"
    assert row.get("error")
    # The exception is asyncio.TimeoutError (or its renamed alias).
    assert "timeout" in row["error"].lower()
    assert row["phase1_passed"] is False
    # The duration is bounded by the cap, not the full sleep.
    assert row["duration_s"] < 2.0


@pytest.mark.asyncio
async def test_run_one_task_with_runner_zero_timeout_disables_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting BIRD_INTERACT_PER_TASK_TIMEOUT_S=0 disables the cap — useful
    for benchmarks where individual tasks legitimately exceed 15 minutes."""
    import asyncio as _asyncio

    monkeypatch.setenv("BIRD_INTERACT_PER_TASK_TIMEOUT_S", "0")

    async def slow_but_legitimate(td, data_dir, patience, user_sim_model):
        await _asyncio.sleep(0.05)
        return {
            "instance_id": td["instance_id"], "database": "db_a",
            "phase1_passed": True, "phase2_passed": True,
            "total_reward": 1.0, "error": None,
        }

    row = await run_mod.run_one_task_with_runner(
        slow_but_legitimate,
        task_data={"instance_id": "slow_1", "selected_database": "db_a",
                   "sol_sql": ["SELECT 1"]},
        data_dir="/tmp",
        patience=3,
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
    )
    assert row["error"] is None
    assert row["phase1_passed"] is True


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
