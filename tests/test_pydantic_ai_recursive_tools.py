"""Unit tests for the `pydantic_ai_recursive` adapter's tool wiring.

These tests are framework-only — no LLM call, no SLayer MCP server, no
network. They lock down:

* Which tools each agent role actually exposes (root / sub-clarifier /
  query-constructor each have a different allowed set).
* `spawn_subagent` respects `max_depth` without spawning a real child run.
* Per-agent dialogue isolation (sibling sub-clarifiers do not share their
  `user_sim_transcript`).
* The shared `SampleStatus` budget is mutated by every sub-tree.
* `_LegacyAdapter` correctly routes flat-attribute access to nested
  `shared` / per-agent deps.
* `run_task` rejects raw mode and non-a-interact mode with `ValueError`.
"""

from __future__ import annotations

import pytest
from pydantic_ai import RunContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_status(remaining_budget: float = 100.0):
    from bird_interact_agents.harness import SampleStatus

    return SampleStatus(
        idx=0,
        original_data={
            "selected_database": "fake_db",
            "instance_id": "fake_1",
            "amb_user_query": "show me X and Y",
            "knowledge_ambiguity": [],
        },
        remaining_budget=remaining_budget,
        total_budget=remaining_budget,
    )


def _make_shared(remaining_budget: float = 100.0):
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
        SharedTaskState,
    )

    return SharedTaskState(
        status=_make_status(remaining_budget),
        data_path_base="/tmp/ignored",
        db_name="fake_db",
        amb_user_query="show me X and Y",
        slayer_storage_dir="",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )


def _make_deps(shared=None, depth: int = 0, max_depth: int = 3,
               self_record_idx: int | None = None):
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import TaskDeps

    return TaskDeps(
        shared=shared or _make_shared(),
        depth=depth,
        max_depth=max_depth,
        self_record_idx=self_record_idx,
    )


def _ctx(deps):
    """Build a minimal RunContext for direct tool invocation."""
    return RunContext(
        deps=deps, model=None, usage=None, prompt="", run_step=0,
    )


def _tools(agent) -> dict:
    """Pydantic-AI 1.82 keeps native @agent.tool definitions on the agent's
    internal _function_toolset. Tests pull them by name to assert presence
    or absence without touching private toolset code paths."""
    return dict(agent._function_toolset.tools)


# ---------------------------------------------------------------------------
# 1-2: depth bound + dialogue isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_depth_bound_returns_message_without_building_child(monkeypatch):
    """Calling spawn_subagent at depth >= max_depth must return the
    'max_depth reached' message without building or running a child
    agent at all. Both invariants matter: building a child still incurs
    pydantic-ai's Agent construction cost (toolset wiring, prepare_tools
    binding), so the depth check must fire BEFORE _build_sub_clarifier."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    build_calls = []
    run_calls = []

    class _ShouldNotRun:
        async def run(self, *a, **kw):
            run_calls.append((a, kw))
            raise AssertionError("Agent.run must NOT be called at max_depth")

    def _builder(*a, **kw):
        build_calls.append((a, kw))
        return _ShouldNotRun()

    monkeypatch.setattr(factories, "_build_sub_clarifier", _builder)

    deps = _make_deps(depth=2, max_depth=2)
    agent = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=2,
    )
    spawn = _tools(agent)["spawn_subagent"]
    out = await spawn.function(_ctx(deps), focus="x", instruction="...")
    assert "max_depth" in out
    assert "2" in out
    assert run_calls == []
    assert build_calls == [], (
        "child agent must not be constructed when depth bound is hit"
    )


@pytest.mark.asyncio
async def test_sibling_subagents_have_isolated_user_sim_transcripts(monkeypatch):
    """Two sibling spawns must each get their own TaskDeps.user_sim_transcript
    list. Sharing would let one chunk's clarifications pollute the other's
    context — the whole point of the decomposition."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import factories
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import TaskDeps

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    captured: list[TaskDeps] = []

    class _CapturingAgent:
        async def run(self, *a, deps, **kw):
            captured.append(deps)
            deps.user_sim_transcript.append(
                {"phase": "encoder", "agent_question": id(deps)},
            )
            from types import SimpleNamespace
            return SimpleNamespace(
                output="ok",
                usage=lambda: SimpleNamespace(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    monkeypatch.setattr(
        factories, "_build_sub_clarifier",
        lambda *a, **kw: _CapturingAgent(),
    )
    deps = _make_deps()
    agent = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=3,
    )
    spawn = _tools(agent)["spawn_subagent"]
    await spawn.function(_ctx(deps), focus="x", instruction="first")
    await spawn.function(_ctx(deps), focus="y", instruction="second")

    assert len(captured) == 2
    d0, d1 = captured
    # Different list objects:
    assert d0.user_sim_transcript is not d1.user_sim_transcript
    # Each holds only its own entry:
    assert len(d0.user_sim_transcript) == 1
    assert len(d1.user_sim_transcript) == 1
    assert d0.user_sim_transcript[0]["agent_question"] != \
        d1.user_sim_transcript[0]["agent_question"]


# ---------------------------------------------------------------------------
# 3: shared budget — one SampleStatus across the whole tree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_budget_decrements_across_subtree(monkeypatch):
    """All deps in the spawn tree must point at one SharedTaskState — and
    therefore one SampleStatus — so two env actions in different subtrees
    drain the same budget pool."""
    from bird_interact_agents.agents._submit import run_env_action
    from bird_interact_agents.agents._tool_specs import BIRD_INTERACT_TOOLS
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
        _LegacyAdapter,
    )

    monkeypatch.setattr(
        "bird_interact_agents.agents._submit.execute_env_action",
        lambda *a, **kw: ("obs", None),
    )

    shared = _make_shared(remaining_budget=10.0)
    deps_a = _make_deps(shared=shared, depth=1, max_depth=3, self_record_idx=1)
    deps_b = _make_deps(shared=shared, depth=1, max_depth=3, self_record_idx=2)

    # Pick a spec costing 1 (execute_sql) — two of them = 2.
    spec = next(s for s in BIRD_INTERACT_TOOLS if s.name == "execute_sql")
    adapter_a = _LegacyAdapter(deps_a)
    adapter_b = _LegacyAdapter(deps_b)
    run_env_action(adapter_a, spec, "raw", sql="SELECT 1")
    run_env_action(adapter_b, spec, "raw", sql="SELECT 1")

    assert shared.status.remaining_budget == 8.0
    assert deps_a.shared.status is shared.status
    assert deps_b.shared.status is shared.status


# ---------------------------------------------------------------------------
# 4: _LegacyAdapter routing — both read AND write paths
# ---------------------------------------------------------------------------


def test_legacy_adapter_routes_shared_and_per_agent_attrs():
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
        _LegacyAdapter,
    )

    shared = _make_shared()
    deps = _make_deps(shared=shared)
    adapter = _LegacyAdapter(deps)

    # Shared fields:
    assert adapter.status is shared.status
    assert adapter.data_path_base == shared.data_path_base
    assert adapter.user_sim_model == shared.user_sim_model
    assert adapter.user_sim_prompt_version == shared.user_sim_prompt_version
    assert adapter.slayer_storage_dir == shared.slayer_storage_dir

    # Per-agent fields:
    assert adapter.user_sim_transcript is deps.user_sim_transcript
    assert adapter.usage is deps.usage

    # Writes to `result` route to shared.submitter_result:
    payload = {"phase1_passed": True, "submitted_query": "{}"}
    adapter.result = payload
    assert shared.submitter_result == payload
    assert adapter.result == payload   # read-back round-trips

    # Writes to slayer cache attributes route to shared:
    sentinel_client = object()
    sentinel_storage = object()
    adapter._slayer_client = sentinel_client
    adapter._slayer_storage = sentinel_storage
    assert shared._slayer_client is sentinel_client
    assert shared._slayer_storage is sentinel_storage


# ---------------------------------------------------------------------------
# 5-8: which tools each agent role registers
# ---------------------------------------------------------------------------


def test_root_clarifier_tools():
    """Root clarifier has spawn_subagent, but NO ask_user, NO submit_query.
    The 'no ask_user' rule is load-bearing: it forces the root to defer
    every ambiguity to its sub-tree rather than asking from the top-level
    dialogue context."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agent = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=3,
    )
    names = set(_tools(agent))
    assert "spawn_subagent" in names
    assert "ask_user" not in names
    assert "submit_query" not in names
    assert "submit_sql" not in names


def test_sub_clarifier_tools():
    """Sub-clarifier has ask_user + spawn_subagent (for compound replies),
    but NO submit_query."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agent = factories._build_sub_clarifier(
        model="test", model_settings=None, shared_slayer_server=None,
    )
    names = set(_tools(agent))
    assert "ask_user" in names
    assert "spawn_subagent" in names
    assert "submit_query" not in names
    assert "submit_sql" not in names


def test_query_constructor_tools():
    """Query-constructor has ask_user + submit_query, NO spawn_subagent.
    Without spawn, the constructor can't shard out its decision-making —
    it must own the assembly + count-check + submit path itself.

    Per DEV-1432, the factory now requires `confirmed_projection` —
    Stage 2's output. Passing a sentinel tuple here keeps the test
    focused on tool wiring (the closure-bound count check has its own
    tests in `test_constructor_closure_gate.py`)."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agent = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("col_a", "col_b"),
    )
    names = set(_tools(agent))
    assert "ask_user" in names
    assert "submit_query" in names
    assert "spawn_subagent" not in names
    assert "submit_sql" not in names


def test_spawn_subagent_is_sequential():
    """The spawn_subagent tool must carry sequential=True so a model batch
    that emits multiple spawn calls runs them serially. Parallel execution
    would race on shared.agent_records.append, causing parent_idx pointers
    to land on the wrong sibling slot."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agent = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=3,
    )
    spawn = _tools(agent)["spawn_subagent"]
    assert spawn.sequential is True


# ---------------------------------------------------------------------------
# 9-10: run_task rejects unsupported modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_task_rejects_raw_query_mode():
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    inst = PydanticAIRecursiveAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError, match="slayer"):
        await inst.run_task(
            {"selected_database": "fake_db",
             "instance_id": "fake_1",
             "amb_user_query": "?"},
            data_path_base="/tmp",
            budget=10.0,
            query_mode="raw",
            eval_mode="a-interact",
        )


@pytest.mark.asyncio
async def test_run_task_rejects_c_interact_eval_mode():
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    inst = PydanticAIRecursiveAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError, match="a-interact"):
        await inst.run_task(
            {"selected_database": "fake_db",
             "instance_id": "fake_1",
             "amb_user_query": "?"},
            data_path_base="/tmp",
            budget=10.0,
            query_mode="slayer",
            eval_mode="c-interact",
        )


@pytest.mark.asyncio
async def test_run_task_rejects_oracle_eval_mode():
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    inst = PydanticAIRecursiveAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError, match="a-interact"):
        await inst.run_task(
            {"selected_database": "fake_db",
             "instance_id": "fake_1",
             "amb_user_query": "?"},
            data_path_base="/tmp",
            budget=10.0,
            query_mode="slayer",
            eval_mode="oracle",
        )
