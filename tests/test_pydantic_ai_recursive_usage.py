"""Token-usage accounting for the recursive adapter.

The dominant regression risk is that a child agent's `Agent.run()` tokens
get dropped on the floor. pydantic-ai's UserSim wrapper writes user-sim
tokens onto `deps.usage` automatically via `acompletion_tracked`, but the
AGENT's own input/output/cache tokens live on the run object's `.usage()`
return value and must be folded in explicitly with an
`add_call(scope='agent', ...)` from the spawn wrapper. If we forget that
step, every child's agent-side tokens silently vanish from the per-task
total — these tests pin the behaviour.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai import RunContext


def _make_shared():
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
                "amb_user_query": "?",
                "knowledge_ambiguity": [],
            },
            remaining_budget=100.0,
            total_budget=100.0,
        ),
        data_path_base="/tmp",
        db_name="fake_db",
        amb_user_query="?",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )


def _ctx(deps):
    return RunContext(deps=deps, model=None, usage=None, prompt="", run_step=0)


@pytest.mark.asyncio
async def test_spawn_subagent_records_child_agent_tokens(monkeypatch):
    """The child's `run.usage()` must be explicitly folded onto the
    child's AgentRecord usage with a `scope='agent'` row.

    This is the bug that bit the existing pydantic_ai adapter: the
    user-sim wrapper writes user_sim tokens onto deps.usage, but the
    agent's own input/output tokens come from the run-object's
    `.usage()` and have to be added with `add_call(scope='agent', ...)`
    or they are lost.
    """
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import factories
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import TaskDeps

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    class _StubAgent:
        async def run(self, *a, deps, **kw):
            return SimpleNamespace(
                output="child_output_text",
                usage=lambda: SimpleNamespace(
                    input_tokens=123,
                    output_tokens=45,
                    cache_read_tokens=10,
                    cache_write_tokens=5,
                ),
                all_messages=lambda: [],
            )

    monkeypatch.setattr(
        factories, "_build_sub_clarifier",
        lambda *a, **kw: _StubAgent(),
    )

    shared = _make_shared()
    deps = TaskDeps(shared=shared, depth=0, max_depth=3, parent_idx=None)
    root = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=3,
    )
    spawn = root._function_toolset.tools["spawn_subagent"]
    out = await spawn.function(_ctx(deps), focus="x", instruction="...")
    assert out == "child_output_text"

    assert len(shared.agent_records) == 1
    rec = shared.agent_records[0]
    agent_rows = [r for r in rec.usage.breakdown if r.scope == "agent"]
    assert agent_rows, "child agent's own tokens were dropped"
    row = agent_rows[0]
    assert row.prompt_tokens == 123
    assert row.completion_tokens == 45
    assert row.cache_read_tokens == 10
    assert row.cache_write_tokens == 5
    assert row.n_calls == 1


@pytest.mark.asyncio
async def test_run_task_aggregates_usage_across_all_agents(monkeypatch):
    """The top-level `run_task` must merge every AgentRecord's usage into
    a single returned `usage` dict. If the merge step is missing or wrong,
    the per-task total only reflects one agent's tokens."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as pa_rec
    from bird_interact_agents.usage import TokenUsage

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    # Skip DB-load / storage-resolve I/O.
    monkeypatch.setattr(pa_rec, "load_db_data_if_needed", lambda *a, **kw: None)

    async def _no_storage(**kw):
        return "", []
    monkeypatch.setattr(pa_rec, "resolve_task_storage_dir", _no_storage)

    # Skip the MCP server lifecycle.
    class _Noop:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(pa_rec, "_build_shared_slayer_server", lambda *a, **kw: _Noop())

    # Stub the root agent so it appends three records with known usages
    # and returns a spec, then the constructor adds its own.
    def _make_record(role, agent_prompt, agent_completion):
        from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
            AgentRecord,
        )
        rec_usage = TokenUsage()
        rec_usage.add_call(
            scope="agent", model="anthropic/claude-sonnet-4-5",
            prompt=agent_prompt, completion=agent_completion,
            cache_read=0, cache_write=0,
        )
        return AgentRecord(
            role=role, depth=0, parent_idx=None,
            instruction="...", output="...", usage=rec_usage,
        )

    root_record = _make_record("root_clarifier", 100, 10)
    sub_record = _make_record("sub_clarifier", 200, 20)
    constructor_record = _make_record("query_constructor", 300, 30)

    class _StubRoot:
        async def run(self, *a, deps, **kw):
            deps.shared.agent_records.append(root_record)
            deps.shared.agent_records.append(sub_record)
            return SimpleNamespace(
                output="spec",
                usage=lambda: SimpleNamespace(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    class _StubConstructor:
        async def run(self, *a, deps, **kw):
            deps.shared.agent_records.append(constructor_record)
            deps.shared.submitter_result = {
                "phase1_passed": True, "phase2_passed": False,
                "total_reward": 1.0,
                "submitted_query": "{}", "submitted_sql": "SELECT 1",
                "submission_status": "passed_phase1",
            }
            return SimpleNamespace(
                output="done",
                usage=lambda: SimpleNamespace(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    class _StubResolver:
        """Stage 2 stub — emits no tokens of its own so the existing
        per-agent totals stay valid. The resolver IS recorded as an
        AgentRecord by run_task itself; its usage is empty TokenUsage,
        contributing 0 prompt + 0 completion to the merged total."""

        async def run(self, *a, **kw):
            return SimpleNamespace(
                output=["col_a"],
                usage=lambda: SimpleNamespace(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    # Patch the *internal* builder seams pa_rec.run_task uses.
    monkeypatch.setattr(pa_rec, "_build_root_clarifier",
                        lambda *a, **kw: _StubRoot())
    monkeypatch.setattr(pa_rec, "_build_projection_resolver",
                        lambda *a, **kw: _StubResolver())
    monkeypatch.setattr(pa_rec, "_build_query_constructor",
                        lambda *a, **kw: _StubConstructor())

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

    rebuilt = TokenUsage.model_validate(result["usage"])
    # 100 + 200 + 300 = 600 prompt; 10 + 20 + 30 = 60 completion.
    # Stage 2 resolver records 0 tokens (stub above), so merged totals
    # stay identical to the pre-DEV-1432 expectations.
    assert rebuilt.prompt_tokens == 600
    assert rebuilt.completion_tokens == 60
    # All three rows preserved in the breakdown (or merged into one
    # scope::model row — either is acceptable, but tokens must sum).
    total_agent_prompt = sum(
        r.prompt_tokens for r in rebuilt.breakdown if r.scope == "agent"
    )
    assert total_agent_prompt == 600


@pytest.mark.asyncio
async def test_run_task_captures_root_and_constructor_agent_tokens(monkeypatch):
    """The most regression-prone path: a production implementation could
    forget to fold `root_run.usage()` and/or `constructor_run.usage()`
    onto their AgentRecords (the same bug the existing pydantic_ai
    adapter handles at agent.py:559-567 with an explicit add_call). This
    test drives run_task with stubs that return nonzero `.usage()` and
    do NOT pre-populate any AgentRecord — every agent token must come
    from the run-object's `.usage()` capture, not from the stub itself.
    """
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as pa_rec
    from bird_interact_agents.usage import TokenUsage

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
            # No record-append, no usage manipulation — just return a run
            # whose .usage() reports tokens. The harness must fold them in.
            return SimpleNamespace(
                output="spec",
                usage=lambda: SimpleNamespace(
                    input_tokens=777, output_tokens=11,
                    cache_read_tokens=4, cache_write_tokens=3,
                ),
                all_messages=lambda: [],
            )

    class _StubConstructor:
        async def run(self, *a, deps, **kw):
            deps.shared.submitter_result = {
                "phase1_passed": True, "phase2_passed": False,
                "total_reward": 1.0,
                "submitted_query": "{}", "submitted_sql": "SELECT 1",
                "submission_status": "passed_phase1",
            }
            return SimpleNamespace(
                output="done",
                usage=lambda: SimpleNamespace(
                    input_tokens=555, output_tokens=9,
                    cache_read_tokens=2, cache_write_tokens=1,
                ),
                all_messages=lambda: [],
            )

    class _StubResolver:
        """Stage 2 stub — emits NONZERO tokens so the test also pins
        that the resolver's run.usage() gets folded into its
        AgentRecord. Adds 333 prompt + 17 completion + 0/0 cache to
        the expected totals."""

        async def run(self, *a, **kw):
            return SimpleNamespace(
                output=["col_a"],
                usage=lambda: SimpleNamespace(
                    input_tokens=333, output_tokens=17,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    monkeypatch.setattr(pa_rec, "_build_root_clarifier",
                        lambda *a, **kw: _StubRoot())
    monkeypatch.setattr(pa_rec, "_build_projection_resolver",
                        lambda *a, **kw: _StubResolver())
    monkeypatch.setattr(pa_rec, "_build_query_constructor",
                        lambda *a, **kw: _StubConstructor())

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

    rebuilt = TokenUsage.model_validate(result["usage"])
    # 777 (root) + 333 (resolver) + 555 (constructor) = 1665 prompt.
    assert rebuilt.prompt_tokens == 777 + 333 + 555, (
        "root, resolver, or constructor agent-side tokens were dropped"
    )
    assert rebuilt.completion_tokens == 11 + 17 + 9
    assert rebuilt.cache_read_tokens == 4 + 0 + 2
    assert rebuilt.cache_write_tokens == 3 + 0 + 1
    agent_rows = [r for r in rebuilt.breakdown if r.scope == "agent"]
    assert agent_rows, "no agent-scope rows in the merged usage breakdown"
    # And the per-agent records each carry their own usage row.
    agents = result["trajectory"]["agents"]
    roots = [a for a in agents if a["role"] == "root_clarifier"]
    constructors = [a for a in agents if a["role"] == "query_constructor"]
    assert roots and constructors
    for rec in (roots[0], constructors[0]):
        per_agent = TokenUsage.model_validate(rec["usage"])
        assert per_agent.prompt_tokens > 0, (
            f"{rec['role']} record has zero prompt_tokens — "
            "agent-side capture missing in run_task"
        )
