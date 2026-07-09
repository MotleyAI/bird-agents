"""Per-agent trajectory capture for the recursive adapter.

The new adapter writes ``trajectory = {final_output_excerpt, agents: [...]}``
where each agent entry carries its own role / depth / parent_idx / focus /
instruction / output / usage / messages / user_sim_transcript / tool stats.
This shape lets offline analysis rebuild the spawn tree from a flat list,
and lets a partial trajectory survive an error path with whatever sub-agents
already finished.

These tests pin the contract for AgentRecord population, parent-idx
correctness under nested spawning, sibling-spawn serialisation,
constructor-reserve budget protection, the compound-naming behavioural
expectation, and the error-path partial-snapshot guarantee.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from pydantic_ai import RunContext


def _make_shared(remaining_budget: float = 100.0,
                 amb_user_query: str = "show me X and Y"):
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
        SharedTaskState,
    )
    from bird_interact_agents.harness import SampleStatus

    return SharedTaskState(
        status=SampleStatus(
            idx=0,
            original_data={
                "selected_database": "fake_db",
                "instance_id": "fake_1",
                "amb_user_query": amb_user_query,
                "knowledge_ambiguity": [],
            },
            remaining_budget=remaining_budget,
            total_budget=remaining_budget,
        ),
        data_path_base="/tmp",
        db_name="fake_db",
        amb_user_query=amb_user_query,
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
    return RunContext(deps=deps, model=None, usage=None, prompt="", run_step=0)


def _make_fake_run(output="ok", input_tokens=1, output_tokens=1,
                   all_messages=None):
    return SimpleNamespace(
        output=output,
        usage=lambda: SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ),
        all_messages=lambda: list(all_messages or []),
    )


# ---------------------------------------------------------------------------
# 13: AgentRecord fields populated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_subagent_populates_full_agent_record(monkeypatch):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    class _StubChild:
        async def run(self, *a, deps, **kw):
            deps.user_sim_transcript.append({"phase": "encoder", "x": 1})
            return _make_fake_run(
                output="child-output",
                all_messages=[SimpleNamespace(parts=[])],
            )

    monkeypatch.setattr(
        factories, "_build_sub_clarifier",
        lambda *a, **kw: _StubChild(),
    )

    shared = _make_shared()
    deps = _make_deps(shared=shared)
    agent = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=3,
    )
    spawn = agent._function_toolset.tools["spawn_subagent"]
    out = await spawn.function(_ctx(deps), focus="my_focus",
                               instruction="please clarify")
    assert out == "child-output"

    assert len(shared.agent_records) == 1
    rec = shared.agent_records[0]
    assert rec.role == "sub_clarifier"
    assert rec.depth == 1
    assert rec.focus == "my_focus"
    assert rec.instruction == "please clarify"
    assert rec.output == "child-output"
    assert rec.started_at > 0
    assert rec.ended_at >= rec.started_at
    assert rec.user_sim_transcript == [{"phase": "encoder", "x": 1}]
    assert rec.error is None
    # Some messages list must be populated (even if a stub one-element list).
    assert isinstance(rec.messages, list)


# ---------------------------------------------------------------------------
# 14-15: parent_idx — flat list rebuilds the spawn tree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parent_idx_lets_flat_list_rebuild_tree(monkeypatch):
    """Root spawns two children — both children's records must carry
    parent_idx pointing at the root's record."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import factories
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
        AgentRecord,
    )

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    shared = _make_shared()
    # The root's record is pre-reserved at index 0 by run_task. Here we
    # mimic that by appending one root placeholder before any spawn, and
    # build a deps whose self_record_idx points at it.
    shared.agent_records.append(AgentRecord(
        role="root_clarifier", depth=0, parent_idx=None,
        instruction="root", started_at=time.monotonic(),
    ))
    deps = _make_deps(shared=shared, self_record_idx=0)

    class _StubChild:
        async def run(self, *a, deps, **kw):
            return _make_fake_run(output="ok")
    monkeypatch.setattr(factories, "_build_sub_clarifier",
                        lambda *a, **kw: _StubChild())

    agent = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=3,
    )
    spawn = agent._function_toolset.tools["spawn_subagent"]
    await spawn.function(_ctx(deps), focus="a", instruction="...")
    await spawn.function(_ctx(deps), focus="b", instruction="...")

    assert [r.role for r in shared.agent_records] == [
        "root_clarifier", "sub_clarifier", "sub_clarifier",
    ]
    # Both children's parent_idx points to root's slot (0).
    assert shared.agent_records[1].parent_idx == 0
    assert shared.agent_records[2].parent_idx == 0


@pytest.mark.asyncio
async def test_nested_spawn_grandchild_points_to_child_not_sibling(monkeypatch):
    """The pre-reserve-slot pattern: when sub-clarifier A spawns a
    grandchild G, G.parent_idx must point at A's slot — NOT at A's parent
    or at a sibling that happened to append first.

    Drives the nested spawn directly: first spawn from the root's deps
    creates A; we capture the deps the root's spawn-wrapper handed to A;
    we then invoke spawn_subagent from THAT captured-A-deps to create G,
    and assert G's record points at A.
    """
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import factories
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
        AgentRecord,
    )

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    shared = _make_shared()
    shared.agent_records.append(AgentRecord(
        role="root_clarifier", depth=0, parent_idx=None,
        instruction="root", started_at=time.monotonic(),
    ))

    # Capture the deps the spawn wrapper hands to the child so we can
    # invoke spawn_subagent from A's perspective.
    captured_child_deps: list = []

    class _ChildLikeAgent:
        async def run(self, *a, deps, **kw):
            captured_child_deps.append(deps)
            return _make_fake_run(output="child-or-grandchild-out")

    monkeypatch.setattr(factories, "_build_sub_clarifier",
                        lambda *a, **kw: _ChildLikeAgent())

    deps = _make_deps(shared=shared, self_record_idx=0)
    root = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=3,
    )
    spawn_from_root = root._function_toolset.tools["spawn_subagent"]
    # 1) Root spawns child A.
    await spawn_from_root.function(_ctx(deps), focus="a", instruction="A")
    assert len(captured_child_deps) == 1
    a_deps = captured_child_deps[0]
    a_idx = a_deps.self_record_idx
    assert a_idx == 1, f"child A expected at slot 1, got {a_idx}"

    # 2) Simulate A spawning grandchild G. A's own agent is a
    # sub_clarifier, so use the SAME spawn tool registered on the
    # sub-clarifier agent. _build_sub_clarifier is monkey-patched to
    # _ChildLikeAgent above — but we want to drive the REAL spawn
    # registration. Build the sub-clarifier via the unpatched factory.
    monkeypatch.undo()
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    sub_agent = factories._build_sub_clarifier(
        model="test", model_settings=None, shared_slayer_server=None,
    )
    # Re-patch _build_sub_clarifier so the grandchild instantiation
    # inside the next spawn returns our stub.
    monkeypatch.setattr(factories, "_build_sub_clarifier",
                        lambda *a, **kw: _ChildLikeAgent())
    spawn_from_a = sub_agent._function_toolset.tools["spawn_subagent"]
    await spawn_from_a.function(_ctx(a_deps), focus="g",
                                instruction="grandchild")

    roles = [r.role for r in shared.agent_records]
    assert roles == ["root_clarifier", "sub_clarifier", "sub_clarifier"], roles
    g_record = shared.agent_records[2]
    assert g_record.parent_idx == a_idx, (
        f"grandchild points at {g_record.parent_idx}, expected {a_idx} "
        "(child A) — pre-reserve-slot pattern broken"
    )
    # And the deps passed to the grandchild's run carried
    # self_record_idx==2 (G's own slot), which encodes that G's children
    # would point at G.
    assert len(captured_child_deps) == 2
    g_deps = captured_child_deps[1]
    assert g_deps.self_record_idx == 2


# ---------------------------------------------------------------------------
# 18: failed subagent — record still captured, parent continues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_subagent_records_error_and_returns_observation(monkeypatch):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    class _BoomChild:
        async def run(self, *a, **kw):
            raise RuntimeError("kaboom inside child")

    monkeypatch.setattr(factories, "_build_sub_clarifier",
                        lambda *a, **kw: _BoomChild())

    shared = _make_shared()
    deps = _make_deps(shared=shared)
    root = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=3,
    )
    spawn = root._function_toolset.tools["spawn_subagent"]
    out = await spawn.function(_ctx(deps), focus="x", instruction="...")
    # Tool returns an error-string observation, NOT raises — parent must
    # keep going.
    assert "Subagent error" in out
    assert "kaboom inside child" in out

    assert len(shared.agent_records) == 1
    rec = shared.agent_records[0]
    assert rec.error is not None
    assert "kaboom inside child" in rec.error
    assert rec.ended_at >= rec.started_at


# ---------------------------------------------------------------------------
# 19: top-level error in constructor — partial trajectory survives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_error_snapshots_partial_agents(monkeypatch):
    """When the constructor raises, the result row's trajectory.agents
    must contain whatever sub-agent records already completed (NOT empty),
    and `error` must carry the constructor's exception text."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as pa_rec
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
        AgentRecord,
    )

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(pa_rec, "load_db_data_if_needed", lambda *a, **kw: None)

    async def _no_storage(**kw): return "", []
    monkeypatch.setattr(pa_rec, "resolve_task_storage_dir", _no_storage)

    class _Noop:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(pa_rec, "_build_shared_slayer_server",
                        lambda *a, **kw: _Noop())

    class _StubRoot:
        async def run(self, *a, deps, **kw):
            deps.shared.agent_records.append(AgentRecord(
                role="sub_clarifier", depth=1, parent_idx=None,
                instruction="completed-sub", output="some output",
                started_at=time.monotonic(),
                ended_at=time.monotonic(),
            ))
            return _make_fake_run(output="spec")

    class _BoomConstructor:
        async def run(self, *a, **kw):
            raise RuntimeError("constructor exploded")

    class _ResolverThatReturnsOneCol:
        async def run(self, *a, **kw):
            return _make_fake_run(output=["col_a"])

    monkeypatch.setattr(pa_rec, "_build_root_clarifier",
                        lambda *a, **kw: _StubRoot())
    monkeypatch.setattr(pa_rec, "_build_projection_resolver",
                        lambda *a, **kw: _ResolverThatReturnsOneCol())
    monkeypatch.setattr(pa_rec, "_build_query_constructor",
                        lambda *a, **kw: _BoomConstructor())

    inst = pa_rec.PydanticAIRecursiveAgent(model="anthropic/claude-sonnet-4-5")
    result = await inst.run_task(
        {"selected_database": "fake_db",
         "instance_id": "fake_1",
         "amb_user_query": "?",
         "knowledge_ambiguity": [],
         "dataset": "mini-interact"},
        data_path_base="/tmp",
        budget=100.0,
        query_mode="slayer",
        eval_mode="a-interact",
    )

    assert result["error"] is not None
    assert "constructor exploded" in result["error"]
    agents = result["trajectory"]["agents"]
    assert any(a["instruction"] == "completed-sub" for a in agents), (
        "partial completed sub-agent must survive in the error-path snapshot"
    )


# ---------------------------------------------------------------------------
# 20: CONSTRUCTOR_RESERVE — count-check ask_user can still fire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_reserve_protects_count_check_ask_user(monkeypatch):
    """Run with a tiny total budget where the clarifier phase would
    normally drain everything. After reservation + restoration, the
    constructor must see at least CONSTRUCTOR_RESERVE bird-coins so its
    mandatory ask_user is not rejected by gate_or_none."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as pa_rec
    from bird_interact_agents.harness import ACTION_COSTS

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(pa_rec, "load_db_data_if_needed", lambda *a, **kw: None)

    async def _no_storage(**kw): return "", []
    monkeypatch.setattr(pa_rec, "resolve_task_storage_dir", _no_storage)

    class _Noop:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(pa_rec, "_build_shared_slayer_server",
                        lambda *a, **kw: _Noop())

    constructor_entry_budget = []

    class _DrainingRoot:
        async def run(self, *a, deps, **kw):
            # Simulate the root + sub-clarifier tree spending the
            # remaining budget down to zero.
            deps.shared.status.remaining_budget = 0.0
            return _make_fake_run(output="spec")

    class _ConstructorThatChecksBudget:
        async def run(self, *a, deps, **kw):
            constructor_entry_budget.append(deps.shared.status.remaining_budget)
            deps.shared.submitter_result = {
                "phase1_passed": False, "phase2_passed": False,
                "total_reward": 0.0,
                "submitted_query": "{}", "submitted_sql": "SELECT 1",
                "submission_status": "wrong_result",
            }
            return _make_fake_run(output="done")

    class _ResolverThatReturnsOneCol:
        async def run(self, *a, **kw):
            return _make_fake_run(output=["col_a"])

    monkeypatch.setattr(pa_rec, "_build_root_clarifier",
                        lambda *a, **kw: _DrainingRoot())
    monkeypatch.setattr(pa_rec, "_build_projection_resolver",
                        lambda *a, **kw: _ResolverThatReturnsOneCol())
    monkeypatch.setattr(pa_rec, "_build_query_constructor",
                        lambda *a, **kw: _ConstructorThatChecksBudget())

    inst = pa_rec.PydanticAIRecursiveAgent(model="anthropic/claude-sonnet-4-5")
    await inst.run_task(
        {"selected_database": "fake_db",
         "instance_id": "fake_1",
         "amb_user_query": "?",
         "knowledge_ambiguity": [],
         "dataset": "mini-interact"},
        data_path_base="/tmp",
        budget=10.0,
        query_mode="slayer",
        eval_mode="a-interact",
    )

    assert len(constructor_entry_budget) == 1
    # Reserve covers: 2*ask_user + submit_query (MCP tools like `query`
    # don't decrement bird-coin budget in this adapter — see
    # _constructor_reserve docstring).
    expected_reserve = (
        2 * ACTION_COSTS["ask_user"]
        + ACTION_COSTS["submit_query"]
    )
    assert constructor_entry_budget[0] >= expected_reserve - 1e-9, (
        f"constructor entered with {constructor_entry_budget[0]}, "
        f"expected >= {expected_reserve}"
    )


@pytest.mark.asyncio
async def test_constructor_reserve_clears_force_submit_flag(monkeypatch):
    """When the clarifier phase trips ``status.force_submit = True``
    (because remaining_budget dropped to submit cost during a clarifier
    ask_user), restoring the budget alone is not enough — gate_or_none
    checks force_submit FIRST and would reject the constructor's
    mandatory ask_user even with the budget restored. The fix: clear
    force_submit when the restored budget can afford submit_query."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as pa_rec

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(pa_rec, "load_db_data_if_needed", lambda *a, **kw: None)

    async def _no_storage(**kw): return "", []
    monkeypatch.setattr(pa_rec, "resolve_task_storage_dir", _no_storage)

    class _Noop:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(pa_rec, "_build_shared_slayer_server",
                        lambda *a, **kw: _Noop())

    constructor_entry_force_submit = []

    class _ClarifierTripsForceSubmit:
        async def run(self, *a, deps, **kw):
            # Simulate a clarifier ask_user dropping the budget to the
            # submit-cost threshold, which the harness flips
            # force_submit=True for.
            deps.shared.status.remaining_budget = 0.0
            deps.shared.status.force_submit = True
            return _make_fake_run(output="spec")

    class _ConstructorRecordsFlag:
        async def run(self, *a, deps, **kw):
            constructor_entry_force_submit.append(
                deps.shared.status.force_submit,
            )
            deps.shared.submitter_result = {
                "phase1_passed": False, "phase2_passed": False,
                "total_reward": 0.0, "submitted_query": "{}",
                "submitted_sql": "SELECT 1",
                "submission_status": "wrong_result",
            }
            return _make_fake_run(output="done")

    class _ResolverThatReturnsOneCol:
        async def run(self, *a, **kw):
            return _make_fake_run(output=["col_a"])

    monkeypatch.setattr(pa_rec, "_build_root_clarifier",
                        lambda *a, **kw: _ClarifierTripsForceSubmit())
    monkeypatch.setattr(pa_rec, "_build_projection_resolver",
                        lambda *a, **kw: _ResolverThatReturnsOneCol())
    monkeypatch.setattr(pa_rec, "_build_query_constructor",
                        lambda *a, **kw: _ConstructorRecordsFlag())

    inst = pa_rec.PydanticAIRecursiveAgent(model="anthropic/claude-sonnet-4-5")
    await inst.run_task(
        {"selected_database": "fake_db",
         "instance_id": "fake_1",
         "amb_user_query": "?",
         "knowledge_ambiguity": [],
         "dataset": "mini-interact"},
        data_path_base="/tmp",
        budget=20.0,
        query_mode="slayer",
        eval_mode="a-interact",
    )

    assert constructor_entry_force_submit == [False], (
        "constructor entered with force_submit=True — gate_or_none would "
        "reject its mandatory ask_user"
    )


# ---------------------------------------------------------------------------
# 21: trajectory shape — final_output_excerpt + agents key present
# ---------------------------------------------------------------------------


def test_merge_tool_stats_handles_duplicates_sort_and_truncation():
    """_merge_tool_stats aggregates per-tool counts across agents,
    preserves totals, sorts by -n_calls then name, and truncates
    error_samples at the per-task cap. A bug here silently corrupts
    diagnostics — pin the shape so regressions surface."""
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        _TOOL_ERROR_SAMPLES_PER_TASK,
        _merge_tool_stats,
    )

    # Three agent-records' worth of stats, with overlapping tool names
    # so we can verify the per-tool sum.
    parts = [
        {
            "per_tool": [
                {"tool": "search", "n_calls": 3, "n_errors": 1},
                {"tool": "ask_user", "n_calls": 2, "n_errors": 0},
            ],
            "total_calls": 5,
            "total_errors": 1,
            "error_samples": [
                {"tool": "search", "error": f"err-{i}"} for i in range(7)
            ],
        },
        {
            "per_tool": [
                {"tool": "search", "n_calls": 4, "n_errors": 2},
                {"tool": "inspect_model", "n_calls": 1, "n_errors": 0},
            ],
            "total_calls": 5,
            "total_errors": 2,
            "error_samples": [
                {"tool": "search", "error": f"err-other-{i}"} for i in range(6)
            ],
        },
        None,   # an agent with no tool stats — must be tolerated
    ]
    out = _merge_tool_stats(parts)
    assert out is not None
    per_tool = out["per_tool"]
    by_name = {t["tool"]: t for t in per_tool}
    assert by_name["search"]["n_calls"] == 7
    assert by_name["search"]["n_errors"] == 3
    assert by_name["ask_user"]["n_calls"] == 2
    assert by_name["inspect_model"]["n_calls"] == 1
    # Sort: descending n_calls, ties broken by tool name asc.
    assert [t["tool"] for t in per_tool] == [
        "search", "ask_user", "inspect_model",
    ]
    assert out["total_calls"] == 10
    assert out["total_errors"] == 3
    # Error samples truncated at the per-task cap; first-come-first-kept.
    assert len(out["error_samples"]) == _TOOL_ERROR_SAMPLES_PER_TASK
    # The first 7 come from part[0] (which had 7), then 3 from part[1].
    first_seven = [s["error"] for s in out["error_samples"][:7]]
    assert first_seven == [f"err-{i}" for i in range(7)]


def test_merge_tool_stats_returns_none_when_all_parts_empty():
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        _merge_tool_stats,
    )

    assert _merge_tool_stats([]) is None
    assert _merge_tool_stats([None, None]) is None


@pytest.mark.asyncio
async def test_trajectory_has_agents_key_and_final_output_excerpt(monkeypatch):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as pa_rec
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))
    monkeypatch.setattr(pa_rec, "load_db_data_if_needed", lambda *a, **kw: None)

    # a-interact mode: the projection-resolver TestModel auto-invokes the
    # ``ask_user`` tool, whose impl (``_submit.ask_user_impl``) fires a REAL
    # user-simulator LLM call via the default ``user_sim_model`` (haiku). Stub
    # the tool boundary so the test is genuinely offline — TestModel handled the
    # AGENT model, but not the user-sim.
    async def _fake_ask_user(*a, **kw):
        return "canned user-sim answer"
    monkeypatch.setattr(factories, "ask_user_impl", _fake_ask_user)

    async def _no_storage(**kw): return "", []
    monkeypatch.setattr(pa_rec, "resolve_task_storage_dir", _no_storage)

    class _Noop:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(pa_rec, "_build_shared_slayer_server",
                        lambda *a, **kw: _Noop())

    class _StubRoot:
        async def run(self, *a, **kw):
            return _make_fake_run(output="spec")
    class _StubConstructor:
        async def run(self, *a, deps, **kw):
            deps.shared.submitter_result = {
                "phase1_passed": True, "phase2_passed": False,
                "total_reward": 1.0, "submitted_query": "{}",
                "submitted_sql": "SELECT 1",
                "submission_status": "passed_phase1",
            }
            return _make_fake_run(output="A" * 600)

    monkeypatch.setattr(pa_rec, "_build_root_clarifier",
                        lambda *a, **kw: _StubRoot())
    monkeypatch.setattr(pa_rec, "_build_query_constructor",
                        lambda *a, **kw: _StubConstructor())

    # model="test" (pydantic-ai TestModel): root + constructor are stubbed
    # above; the projection-resolver phase runs on TestModel (canned list[str],
    # instant). Combined with the ask_user stub above, this trajectory-shape
    # test is fully offline + deterministic — no agent OR user-sim network call.
    inst = pa_rec.PydanticAIRecursiveAgent(model="test")
    result = await inst.run_task(
        {"selected_database": "fake_db",
         "instance_id": "fake_1",
         "amb_user_query": "?",
         "knowledge_ambiguity": [],
         "dataset": "mini-interact"},
        data_path_base="/tmp",
        budget=100.0,
        query_mode="slayer",
        eval_mode="a-interact",
    )

    traj = result["trajectory"]
    assert "agents" in traj
    assert "final_output_excerpt" in traj
    # final_output_excerpt cropped at 500.
    assert 0 < len(traj["final_output_excerpt"]) <= 500
    assert isinstance(traj["agents"], list)
    # The constructor's record must be present.
    assert any(a["role"] == "query_constructor" for a in traj["agents"])
