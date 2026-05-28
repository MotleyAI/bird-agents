"""DEV-1462 — one-shot factories + prompts for `pydantic_ai_recursive`.

The one-shot flavor strips `ask_user` from every role and swaps the
spawn target from `_build_sub_clarifier` to `_build_sub_explorer`. These
tests pin the framework wiring — no LLM call, no SLayer MCP server, no
network. They mirror the existing `test_pydantic_ai_recursive_tools.py`
style.

Prompt checks are absence-only (no positive prompt-content assertions,
per the project's TDD-style + no-prompt-content rule): every one-shot
prompt + tool docstring + ModelRetry message must be free of `ask_user`
and user-sim language (Codex #7).
"""

from __future__ import annotations

import re
from typing import Any

import pytest


def _tools(agent: Any) -> dict:
    """Mirror the helper from `test_pydantic_ai_recursive_tools.py`."""
    return dict(agent._function_toolset.tools)


# ---------------------------------------------------------------------------
# Factory presence — the new one-shot builders must exist alongside the
# existing a-interact builders.
# ---------------------------------------------------------------------------


def test_one_shot_factories_exist():
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    for name in (
        "_build_sub_explorer",
        "_build_projection_resolver_oneshot",
        "_build_query_constructor_oneshot",
    ):
        assert hasattr(factories, name), (
            f"factories.{name} missing — agent.py cannot wire the one-shot "
            f"branch without it."
        )


# ---------------------------------------------------------------------------
# Tool surfaces — every one-shot agent role MUST have no ask_user.
# ---------------------------------------------------------------------------


def test_sub_explorer_tools_have_no_ask_user():
    """The MCP-side `search`/`inspect_model` tools come from the shared
    SLayer server (passed in as `shared_slayer_server`); without one, the
    explorer's NATIVE function-toolset surface is just `spawn_subagent`.
    We don't pin MCP tool names here (they're owned by the SLayer MCP
    server, not by us), only the NATIVE toolset — which MUST NOT carry
    ask_user, submit_query, or any of the constructor's write tools."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agent = factories._build_sub_explorer(
        model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
    )
    names = set(_tools(agent))
    assert "ask_user" not in names
    assert "submit_query" not in names
    assert "spawn_subagent" in names, (
        "one-shot sub-explorer MUST still expose spawn_subagent so the "
        "recursive decomposition reaches grandchild explorers."
    )


def test_projection_resolver_oneshot_tools_have_no_ask_user():
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agent = factories._build_projection_resolver_oneshot(
        model="test", model_settings=None,
    )
    names = set(_tools(agent))
    assert "ask_user" not in names
    assert "submit_query" not in names
    assert "spawn_subagent" not in names


def test_projection_resolver_oneshot_output_type_is_list_of_str():
    """One-shot resolver keeps structured `output_type=list[str]` (same as
    a-interact recursive) — the constructor's closure count-check depends
    on a parseable list."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agent = factories._build_projection_resolver_oneshot(
        model="test", model_settings=None,
    )
    candidates = [
        getattr(agent, attr, None) for attr in (
            "output_type", "_output_type", "_output_schema",
        )
    ]
    candidates = [c for c in candidates if c is not None]
    assert candidates, "no output-type attribute on the one-shot resolver"
    assert any(
        "list" in repr(c).lower() and "str" in repr(c).lower()
        for c in candidates
    ), (
        f"one-shot resolver must declare list[str] output; got: "
        f"{[repr(c) for c in candidates]!r}"
    )


def test_query_constructor_oneshot_tools_have_no_ask_user():
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agent = factories._build_query_constructor_oneshot(
        model="test", model_settings=None,
        shared_slayer_server=None,
        confirmed_projection=("col_a", "col_b"),
        self_model_id="test",
    )
    names = set(_tools(agent))
    assert "ask_user" not in names
    assert "submit_query" in names, (
        "one-shot constructor MUST keep submit_query — it's the final delivery."
    )
    assert "spawn_subagent" not in names


def test_query_constructor_oneshot_keeps_closure_count_check():
    """The closure-bound projection count check is load-bearing against
    over/under-projection — must survive the one-shot transformation."""
    from pydantic_ai import ModelRetry
    from pydantic_ai import RunContext

    from bird_interact_agents.agents.pydantic_ai_recursive import factories
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
        SharedTaskState, TaskDeps,
    )
    from bird_interact_agents.harness import SampleStatus

    agent = factories._build_query_constructor_oneshot(
        model="test", model_settings=None,
        shared_slayer_server=None,
        confirmed_projection=("col_a", "col_b", "col_c"),  # n=3
        self_model_id="test",
    )
    submit = _tools(agent)["submit_query"]
    status = SampleStatus(
        idx=0, original_data={
            "selected_database": "fake_db",
            "instance_id": "fake_1",
            "amb_user_query": "x",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench",
        },
        remaining_budget=100.0, total_budget=100.0,
    )
    shared = SharedTaskState(
        status=status, data_path_base="/tmp/ignored", db_name="fake_db",
        amb_user_query="x", slayer_storage_dir="",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )
    deps = TaskDeps(shared=shared, depth=0, max_depth=0, self_record_idx=None)
    ctx = RunContext(
        deps=deps, model=None, usage=None, prompt="", run_step=0,
    )
    # n=2 (mismatched) → must ModelRetry with a one-shot-appropriate message.
    bad_payload = '{"source_model": "x", "dimensions": ["a"], "measures": ["b:sum"]}'
    import asyncio
    with pytest.raises(ModelRetry) as exc_info:
        asyncio.run(submit.function(ctx, query_json=bad_payload))
    msg = str(exc_info.value)
    assert "ask_user" not in msg, (
        "one-shot constructor's ModelRetry text MUST NOT mention "
        f"ask_user; got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Spawn dispatch — root + sub-explorer must spawn explorers, not clarifiers,
# when configured for one-shot.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_in_oneshot_root_builds_explorer_not_clarifier(monkeypatch):
    """The factory's `_register_spawn_subagent` is parametrized with the
    child builder; in one-shot, the root's spawn MUST construct an explorer
    (no ask_user), NOT a clarifier."""
    from pydantic_ai import RunContext

    from bird_interact_agents.agents.pydantic_ai_recursive import factories
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
        SharedTaskState, TaskDeps,
    )
    from bird_interact_agents.harness import SampleStatus
    from bird_interact_agents import usage as usage_mod

    # The spy stub uses model="test" which litellm can't price; without
    # this patch the spawn wrapper's usage record-keeping raises.
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    explorer_calls: list = []
    clarifier_calls: list = []

    real_explorer = factories._build_sub_explorer
    real_clarifier = factories._build_sub_clarifier

    def spy_explorer(**kw):
        explorer_calls.append(kw)
        return real_explorer(**kw)

    def spy_clarifier(**kw):
        clarifier_calls.append(kw)
        return real_clarifier(**kw)

    monkeypatch.setattr(factories, "_build_sub_explorer", spy_explorer)
    monkeypatch.setattr(factories, "_build_sub_clarifier", spy_clarifier)

    root = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=3,
        self_model_id="test",
        eval_mode="one-shot",
    )
    spawn = _tools(root)["spawn_subagent"]
    status = SampleStatus(
        idx=0, original_data={
            "selected_database": "fake_db",
            "instance_id": "fake_1",
            "amb_user_query": "x",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench",
        },
        remaining_budget=100.0, total_budget=100.0,
    )
    shared = SharedTaskState(
        status=status, data_path_base="/tmp/ignored", db_name="fake_db",
        amb_user_query="x", slayer_storage_dir="",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )
    deps = TaskDeps(shared=shared, depth=0, max_depth=3, self_record_idx=None)
    ctx = RunContext(deps=deps, model=None, usage=None, prompt="", run_step=0)

    # We don't want to actually invoke a real LLM in the spawned child;
    # the explorer/clarifier factories build pydantic-ai Agents whose
    # `.run` we'd need to stub. Replace both spy builders with a fake
    # that returns a stub agent with an async run(...) returning a tiny
    # SimpleNamespace.
    from types import SimpleNamespace

    class _StubAgent:
        async def run(self, *a, **kw):
            return SimpleNamespace(
                output="explored",
                usage=lambda: SimpleNamespace(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    monkeypatch.setattr(factories, "_build_sub_explorer", lambda **kw: (
        explorer_calls.append(kw) or _StubAgent()
    ))
    monkeypatch.setattr(factories, "_build_sub_clarifier", lambda **kw: (
        clarifier_calls.append(kw) or _StubAgent()
    ))

    await spawn.function(ctx, focus="qty", instruction="explore widgets.qty")
    assert explorer_calls, (
        "one-shot root spawn MUST build a sub-explorer; "
        f"clarifier_calls={clarifier_calls}, explorer_calls={explorer_calls}"
    )
    assert not clarifier_calls, (
        "one-shot root spawn MUST NOT build a sub-clarifier (which carries ask_user)"
    )


# ---------------------------------------------------------------------------
# Prompts exist + have required placeholders; absence-only content checks.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,placeholders",
    [
        ("ROOT_EXPLORER_PROMPT", ("{budget}", "{db_name}", "{user_query}")),
        ("SUB_EXPLORER_PROMPT", ("{budget}", "{db_name}",
                                 "{focus}", "{instruction}")),
        ("PROJECTION_RESOLVER_ONESHOT_PROMPT",
         ("{amb_user_query}", "{spec}", "{budget}", "{db_name}")),
        ("QUERY_CONSTRUCTOR_ONESHOT_PROMPT",
         ("{amb_user_query}", "{spec}",
          "{confirmed_projection}", "{budget}", "{db_name}")),
    ],
)
def test_one_shot_prompts_exist_with_required_placeholders(name, placeholders):
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    body = getattr(prompts, name, None)
    assert isinstance(body, str) and body.strip(), (
        f"prompts.{name} must be a non-empty string."
    )
    for p in placeholders:
        assert p in body, (
            f"{name} missing required placeholder {p!r}; "
            f"agent.py's .format(...) will blow up at runtime."
        )


_USER_SIM_FORBIDDEN_RE = re.compile(
    r"\b("
    r"ask[_ ]user"
    r"|user[- ]sim"
    r"|user-sim"
    r"|ask the user"
    r"|the user-sim"
    r")\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    "name",
    [
        "ROOT_EXPLORER_PROMPT", "SUB_EXPLORER_PROMPT",
        "PROJECTION_RESOLVER_ONESHOT_PROMPT",
        "QUERY_CONSTRUCTOR_ONESHOT_PROMPT",
    ],
)
def test_one_shot_prompts_have_no_ask_user_or_user_sim_language(name):
    """One-shot is non-interactive — the prompt must NEVER tell the
    model to call ask_user (a tool the one-shot agent doesn't have)
    or to converse with a user simulator (Codex #7)."""
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts

    body = getattr(prompts, name)
    m = _USER_SIM_FORBIDDEN_RE.search(body)
    assert m is None, (
        f"{name} contains forbidden ask_user/user-sim language: "
        f"{m.group(0)!r} at offset {m.start()}; full text:\n{body!r}"
    )


def test_one_shot_tool_docstrings_have_no_ask_user_language():
    """A one-shot tool's docstring is part of the model's tool catalogue —
    if it mentions ask_user, the model may try to call a tool that
    doesn't exist (and that ScopeAssertion-paths can't unwind cleanly).
    Codex #7 — scan EVERY native tool's docstring across all one-shot
    factories for forbidden `ask_user`/`user-sim` text."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    agents = {
        "sub_explorer": factories._build_sub_explorer(
            model="test", model_settings=None,
            shared_slayer_server=None, self_model_id="test",
        ),
        "projection_resolver_oneshot": factories._build_projection_resolver_oneshot(
            model="test", model_settings=None,
        ),
        "query_constructor_oneshot": factories._build_query_constructor_oneshot(
            model="test", model_settings=None,
            shared_slayer_server=None,
            confirmed_projection=("c",),
            self_model_id="test",
        ),
    }
    offenders: list[tuple[str, str, str]] = []
    for role, agent in agents.items():
        for tool_name, tool in _tools(agent).items():
            doc = (
                getattr(tool, "description", None)
                or getattr(getattr(tool, "function", None), "__doc__", None)
                or ""
            )
            m = _USER_SIM_FORBIDDEN_RE.search(doc)
            if m is not None:
                offenders.append((role, tool_name, m.group(0)))
    assert not offenders, (
        "One-shot tool docstrings must not reference ask_user / user-sim. "
        f"Offenders: {offenders!r}"
    )
