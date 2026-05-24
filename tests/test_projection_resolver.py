"""Stage 2 projection-resolver agent (DEV-1432).

Stage 2 sits between the clarifier tree and the query-constructor. It
reads the original user question + the clarifier-tree spec and asks
the user-sim to confirm an ordered list of user-facing column names.
Its output drives BOTH the constructor prompt (the "CONFIRMED
PROJECTION" section) AND the closure-bound count check on submit_query.

These tests pin the framework wiring — no LLM call, no real ask_user.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _tools(agent) -> dict:
    """Mirror the helper in test_pydantic_ai_recursive_tools.py — pull
    native @agent.tool registrations off the internal function toolset."""
    return dict(agent._function_toolset.tools)


# ---------------------------------------------------------------------------
# Factory: tool list + structured output type
# ---------------------------------------------------------------------------


def test_projection_resolver_factory_exists():
    """The resolver factory must be exported from the factories module
    so agent.py can import + build it."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    assert hasattr(factories, "_build_projection_resolver"), (
        "factories._build_projection_resolver missing — Stage 2 cannot "
        "be wired into the run_task flow without it."
    )


def test_projection_resolver_tools_are_ask_user_only():
    """Stage 2's tool list must be `ask_user` only — no submit_query,
    no query, no spawn_subagent, no search/inspect_model. The resolver
    has ONE job (pin a column list with the user-sim) and any extra
    tool surface invites scope creep."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agent = factories._build_projection_resolver(
        model="test", model_settings=None,
    )
    names = set(_tools(agent))
    assert "ask_user" in names
    assert "submit_query" not in names
    assert "submit_sql" not in names
    assert "spawn_subagent" not in names


def test_projection_resolver_output_type_is_list_of_str():
    """Pydantic-AI's structured-output contract: the resolver must
    declare `output_type=list[str]` so its return value is already a
    parsed list when agent.py reads `run.output`. Without this, agent.py
    would need to parse a string and risk format drift between runs."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agent = factories._build_projection_resolver(
        model="test", model_settings=None,
    )
    # pydantic-ai 1.x stores the output spec on the agent. The exact
    # attribute name is `_output_schema` or similar — we don't pin
    # internals, just verify the agent is configured to return a list
    # via either the `output_type` constructor arg or its compiled
    # equivalent. Easiest robust check: the agent's output_type
    # attribute (if exposed) or roundtrip via pydantic-ai's
    # `Agent.output_type` if available.
    # Use a duck-typed check that survives pydantic-ai version bumps:
    # try several known attribute names and assert at least one
    # advertises a `list` type.
    candidates = [
        getattr(agent, attr, None) for attr in (
            "output_type", "_output_type", "_output_schema",
        )
    ]
    candidates = [c for c in candidates if c is not None]
    assert candidates, (
        "Cannot find any output-type attribute on the resolver agent — "
        "pydantic-ai may have refactored the attribute name. Update "
        "this test to match the current attribute and ensure the "
        "factory still passes output_type=list[str]."
    )
    # The spec we passed must be `list[str]` or pydantic-ai's coerced
    # equivalent (which renders containing 'list' / 'str' in its repr).
    found_list = any(
        "list" in repr(c).lower() and "str" in repr(c).lower()
        for c in candidates
    )
    assert found_list, (
        f"Resolver agent does not appear to be configured for "
        f"list[str] output. Candidates inspected: "
        f"{[repr(c) for c in candidates]!r}"
    )


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------


def test_projection_resolver_prompt_constant_exists():
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    assert hasattr(prompts, "PROJECTION_RESOLVER_PROMPT"), (
        "prompts.PROJECTION_RESOLVER_PROMPT missing — agent.py cannot "
        "format the resolver's instructions without it."
    )
    body = prompts.PROJECTION_RESOLVER_PROMPT
    assert isinstance(body, str) and body.strip(), (
        "PROJECTION_RESOLVER_PROMPT must be a non-empty string."
    )


def test_projection_resolver_prompt_has_required_template_vars():
    """The resolver prompt is formatted with {amb_user_query}, {spec},
    {budget}, and {db_name}. Check the placeholders are present so
    agent.py's .format(...) doesn't blow up at runtime."""
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    body = prompts.PROJECTION_RESOLVER_PROMPT
    for placeholder in ("{amb_user_query}", "{spec}", "{budget}", "{db_name}"):
        assert placeholder in body, (
            f"PROJECTION_RESOLVER_PROMPT missing {placeholder!r} "
            f"placeholder."
        )


def test_projection_resolver_prompt_teaches_iteration_cap():
    """The resolver must know it has at most 3 ask_user rounds before
    it should finalize. This is the prompt-side enforcement of the cap
    — the wrapper in agent.py is the deterministic backstop."""
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    body = prompts.PROJECTION_RESOLVER_PROMPT
    lower = body.lower()
    # Cap signal: must mention "3" (or "three") near "round" or "ask".
    assert "3" in body or "three" in lower
    # Plus an instruction to FINALIZE / return after the cap.
    assert any(s in lower for s in (
        "finalize", "return", "stop asking", "submit the list",
        "no more", "after that",
    )), (
        "PROJECTION_RESOLVER_PROMPT must instruct the agent to finalize "
        "the list after the iteration cap."
    )


def test_projection_resolver_prompt_says_ordered_list():
    """Order matters: the constructor must place columns in the exact
    sequence Stage 2 confirmed. The prompt must say so."""
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    body = prompts.PROJECTION_RESOLVER_PROMPT
    lower = body.lower()
    assert "order" in lower or "ordered" in lower or "sequence" in lower
    # Plus a directive to NOT permute / reorder the list.
    assert any(s in lower for s in (
        "in this order", "preserve the order", "keep the order",
        "same order", "matching order",
    )), (
        "PROJECTION_RESOLVER_PROMPT must tell the agent the list is "
        "ordered and the constructor must preserve that order."
    )


def test_projection_resolver_prompt_says_user_facing_names():
    """Names in the list are USER-FACING (the strings the user-sim
    would recognise), NOT internal SLayer measure names. The constructor
    maps them to dims+measures itself. Pin this so the resolver doesn't
    start emitting `clinid` instead of `clinician ID`."""
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    body = prompts.PROJECTION_RESOLVER_PROMPT
    lower = body.lower()
    assert "user-facing" in lower or "user facing" in lower or (
        "human-readable" in lower or "human readable" in lower
    ) or "the user would" in lower, (
        "PROJECTION_RESOLVER_PROMPT must instruct the agent to use "
        "user-facing column names, not internal SLayer names."
    )


# ---------------------------------------------------------------------------
# Wrapper behavior — empty-list guard, projection_resolver_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolver_wrapper_records_confirmed_on_non_empty_output(monkeypatch):
    """When Agent.run returns a non-empty list on the first attempt,
    the wrapper must capture it WITHOUT triggering the empty-list
    guard, and return status `confirmed`."""
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as agent_mod

    run_calls = []

    class _NormalResolver:
        async def run(self, *a, **kw):
            run_calls.append(kw.get("user_prompt", ""))
            return SimpleNamespace(
                output=["clinician ID", "facility ID", "stability score"],
                usage=lambda: SimpleNamespace(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    from bird_interact_agents.usage import TokenUsage
    from bird_interact_agents import usage as usage_mod

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    out = await agent_mod._run_projection_resolver(
        resolver_agent=_NormalResolver(),
        instructions="ignored",
        user_prompt="ignored",
        deps=SimpleNamespace(usage=TokenUsage()),
        model_id="test",
    )
    assert len(run_calls) == 1, (
        "Non-empty first attempt must not trigger the empty-list guard "
        "retry; expected exactly one resolver.run call."
    )
    assert out.projection == ["clinician ID", "facility ID", "stability score"]
    assert out.status == "confirmed"


@pytest.mark.asyncio
async def test_resolver_wrapper_empty_list_triggers_one_more_attempt(monkeypatch):
    """If Stage 2's first run returns `[]`, the wrapper must call
    Agent.run a second time with a 'your list was empty' nudge. If the
    second run also returns `[]`, the wrapper falls into the
    `empty_after_guard` terminal state."""
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as agent_mod

    run_calls = []

    class _Resolver:
        async def run(self, *a, **kw):
            run_calls.append(kw.get("user_prompt", ""))
            # First call returns empty; second call returns ["col1"].
            value = ["col1"] if len(run_calls) >= 2 else []
            return SimpleNamespace(
                output=value,
                usage=lambda: SimpleNamespace(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    # The wrapper must:
    #   1. Call resolver.run once.
    #   2. See empty list.
    #   3. Call resolver.run again with a recovery prompt.
    #   4. Return the recovered list + status='confirmed' (NOT
    #      'empty_after_guard' since the second attempt produced
    #      a non-empty list).
    from bird_interact_agents.usage import TokenUsage
    from bird_interact_agents import usage as usage_mod

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    out = await agent_mod._run_projection_resolver(
        resolver_agent=_Resolver(),
        instructions="ignored",
        user_prompt="ignored",
        deps=SimpleNamespace(usage=TokenUsage()),
        model_id="test",
    )
    assert len(run_calls) == 2, (
        f"Empty-list guard must retry exactly once on first empty; "
        f"got {len(run_calls)} calls."
    )
    assert out.projection == ["col1"]
    assert out.status == "confirmed"


@pytest.mark.asyncio
async def test_resolver_wrapper_empty_after_guard_status(monkeypatch):
    """If BOTH the first and the empty-list-guard runs return `[]`, the
    wrapper must return status `empty_after_guard` so agent.py can skip
    the constructor and finalize as never_submitted."""
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as agent_mod

    class _AlwaysEmpty:
        async def run(self, *a, **kw):
            return SimpleNamespace(
                output=[],
                usage=lambda: SimpleNamespace(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    from bird_interact_agents.usage import TokenUsage
    from bird_interact_agents import usage as usage_mod

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    out = await agent_mod._run_projection_resolver(
        resolver_agent=_AlwaysEmpty(),
        instructions="ignored",
        user_prompt="ignored",
        deps=SimpleNamespace(usage=TokenUsage()),
        model_id="test",
    )
    assert out.projection == []
    assert out.status == "empty_after_guard"


# ---------------------------------------------------------------------------
# Integration: run_task wires Stage 2 between root and constructor.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_task_wires_root_then_resolver_then_constructor(monkeypatch, tmp_path):
    """The recursive adapter's `run_task` must call Stage 1 (root),
    THEN Stage 2 (resolver), THEN Stage 3 (constructor), in that order.
    The constructor's factory must receive `confirmed_projection`
    matching the resolver's output. We monkey-patch all three factories
    with mock agents so the test stays hermetic."""
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as agent_mod
    from bird_interact_agents.agents.pydantic_ai_recursive import factories
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    call_order: list[str] = []
    captured_confirmed_projection: list = []

    def _fake_build_root(**kw):
        class _RootAgent:
            async def run(self, *a, **kw):
                call_order.append("root")
                return SimpleNamespace(
                    output="spec from root",
                    usage=lambda: SimpleNamespace(
                        input_tokens=0, output_tokens=0,
                        cache_read_tokens=0, cache_write_tokens=0,
                    ),
                    all_messages=lambda: [],
                )
        return _RootAgent()

    def _fake_build_resolver(**kw):
        class _ResolverAgent:
            async def run(self, *a, **kw):
                call_order.append("resolver")
                return SimpleNamespace(
                    output=["col_a", "col_b", "col_c"],
                    usage=lambda: SimpleNamespace(
                        input_tokens=0, output_tokens=0,
                        cache_read_tokens=0, cache_write_tokens=0,
                    ),
                    all_messages=lambda: [],
                )
        return _ResolverAgent()

    def _fake_build_constructor(confirmed_projection, **kw):
        captured_confirmed_projection.append(confirmed_projection)

        class _ConstructorAgent:
            async def run(self, *a, **kw):
                call_order.append("constructor")
                return SimpleNamespace(
                    output="constructor done",
                    usage=lambda: SimpleNamespace(
                        input_tokens=0, output_tokens=0,
                        cache_read_tokens=0, cache_write_tokens=0,
                    ),
                    all_messages=lambda: [],
                )
        return _ConstructorAgent()

    monkeypatch.setattr(agent_mod, "_build_root_clarifier", _fake_build_root)
    monkeypatch.setattr(
        agent_mod, "_build_projection_resolver", _fake_build_resolver,
    )
    monkeypatch.setattr(
        agent_mod, "_build_query_constructor", _fake_build_constructor,
    )
    # Avoid touching the slayer MCP / DB loaders.
    monkeypatch.setattr(
        agent_mod, "load_db_data_if_needed", lambda *a, **kw: None,
    )

    async def _no_storage(**kw):
        return ("", [])

    monkeypatch.setattr(agent_mod, "resolve_task_storage_dir", _no_storage)
    monkeypatch.setattr(
        agent_mod, "_build_shared_slayer_server", lambda *a, **kw: None,
    )

    inst = PydanticAIRecursiveAgent(model="anthropic/claude-sonnet-4-5")
    await inst.run_task(
        task_data={
            "selected_database": "fake_db",
            "instance_id": "fake_1",
            "amb_user_query": "show me X",
            "knowledge_ambiguity": [],
        },
        data_path_base=str(tmp_path),
        budget=20.0,
        query_mode="slayer",
        eval_mode="a-interact",
    )

    assert call_order == ["root", "resolver", "constructor"], (
        f"Expected exact stage order root→resolver→constructor; "
        f"got {call_order}."
    )
    assert len(captured_confirmed_projection) == 1, (
        "Constructor factory must be called exactly once."
    )
    # The list passed to the constructor must be EXACTLY what the
    # resolver returned (same names, same order). Tuple-coerce-safe:
    captured = captured_confirmed_projection[0]
    assert tuple(captured) == ("col_a", "col_b", "col_c"), (
        f"Constructor received {captured!r} instead of resolver output "
        f"('col_a', 'col_b', 'col_c')."
    )


@pytest.mark.asyncio
async def test_run_task_skips_constructor_on_empty_after_guard(monkeypatch, tmp_path):
    """When the resolver returns empty TWICE (initial + guard retry),
    `run_task` MUST skip the constructor entirely and finalize with
    `submission_status=never_submitted` and a `projection_resolver_status`
    diagnostic of `empty_after_guard`. The constructor builder must not
    be called at all."""
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as agent_mod
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    constructor_calls = []

    def _fake_build_root(**kw):
        class _RootAgent:
            async def run(self, *a, **kw):
                return SimpleNamespace(
                    output="spec",
                    usage=lambda: SimpleNamespace(
                        input_tokens=0, output_tokens=0,
                        cache_read_tokens=0, cache_write_tokens=0,
                    ),
                    all_messages=lambda: [],
                )
        return _RootAgent()

    def _fake_build_resolver(**kw):
        class _AlwaysEmpty:
            async def run(self, *a, **kw):
                return SimpleNamespace(
                    output=[],
                    usage=lambda: SimpleNamespace(
                        input_tokens=0, output_tokens=0,
                        cache_read_tokens=0, cache_write_tokens=0,
                    ),
                    all_messages=lambda: [],
                )
        return _AlwaysEmpty()

    def _fake_build_constructor(**kw):
        constructor_calls.append(kw)
        pytest.fail(
            "Constructor must NOT be built when resolver returns "
            "empty_after_guard — run_task should skip Stage 3."
        )

    monkeypatch.setattr(agent_mod, "_build_root_clarifier", _fake_build_root)
    monkeypatch.setattr(
        agent_mod, "_build_projection_resolver", _fake_build_resolver,
    )
    monkeypatch.setattr(
        agent_mod, "_build_query_constructor", _fake_build_constructor,
    )
    monkeypatch.setattr(
        agent_mod, "load_db_data_if_needed", lambda *a, **kw: None,
    )

    async def _no_storage(**kw):
        return ("", [])

    monkeypatch.setattr(agent_mod, "resolve_task_storage_dir", _no_storage)
    monkeypatch.setattr(
        agent_mod, "_build_shared_slayer_server", lambda *a, **kw: None,
    )

    inst = PydanticAIRecursiveAgent(model="anthropic/claude-sonnet-4-5")
    row = await inst.run_task(
        task_data={
            "selected_database": "fake_db",
            "instance_id": "fake_1",
            "amb_user_query": "show me X",
            "knowledge_ambiguity": [],
        },
        data_path_base=str(tmp_path),
        budget=20.0,
        query_mode="slayer",
        eval_mode="a-interact",
    )

    assert constructor_calls == []
    assert row["submission_status"] == "never_submitted"
    # The diagnostic must surface the specific failure mode so FMA can
    # distinguish it from generic constructor failures.
    assert row.get("projection_resolver_status") == "empty_after_guard", (
        f"Expected projection_resolver_status='empty_after_guard' on "
        f"the result row; got: {row.get('projection_resolver_status')!r}. "
        f"Row keys: {sorted(row.keys())}"
    )
