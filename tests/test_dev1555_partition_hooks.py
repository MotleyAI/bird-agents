"""DEV-1555 Stage 1: shared subagent-partition + context-budget helpers.

Pins the pure-function contracts:

1. ``partition.make_partition_deny_hook`` — denies discovery-only tools in
   the MAIN loop (no ``agent_id`` in the hook input) and allows them inside
   the discovery subagent (``agent_id`` present). Global ``disallowed_tools``
   cannot be used for this: it applies inside subagents too.
2. ``context_budget.context_window_for`` — model-string → window lookup.
3. ``context_budget.update_context_tokens`` — stream-side context tracking
   from per-turn ``AssistantMessage.usage``.
4. ``context_budget.make_context_budget_hook`` — one-shot warnings at 80%
   and 90% of the window.
5. Regression pins for existing guards under subagent-originated calls:
   the a-interact ask gate and the turn-budget hook count subagent calls
   (hook inputs carrying ``agent_id``) exactly like main-loop calls.
6. ``accumulate_assistant_usage`` counts subagent-tagged AssistantMessages
   (``parent_tool_use_id`` set).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# 1. Discovery client cap
#
# DEV-1581 R2 removed the SDK-subagent split (and the ``partition_deny`` hook
# / ``DISCOVERY_AGENT_NAME``): discovery is now a SEPARATE persistent client,
# so there is no per-call deny hook to test. ``DISCOVERY_MAX_TURNS`` still caps
# one discovery answer.
# ---------------------------------------------------------------------------


def test_discovery_max_turns_matches_model_turn_budget():
    """SDK ``max_turns`` caps each discovery answer (one ``ask_discovery``
    round); pin it to the base model-turn budget so a runaway discovery sweep
    cannot spin unbounded."""
    from bird_interact_agents.agents.claude_sdk import partition
    from bird_interact_agents.harness import MAX_MODEL_TURNS

    assert partition.DISCOVERY_MAX_TURNS == MAX_MODEL_TURNS


# ---------------------------------------------------------------------------
# 2. context_window_for
# ---------------------------------------------------------------------------

def test_context_window_for_anthropic_models_is_behavior_preserving():
    """Stage 1 must not change Claude-run behavior: real opus sessions
    were measured at 262K context, so the Stage-1 window for anthropic
    models is set high enough that the warning hook never fires."""
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        context_window_for,
    )

    assert context_window_for("anthropic/claude-opus-4-7") == 1_000_000
    assert context_window_for("anthropic/claude-sonnet-4-6") == 1_000_000


def test_context_window_for_unknown_provider_defaults_conservative():
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        context_window_for,
    )

    # An UNKNOWN provider / bare id falls back to the conservative 200K default.
    assert context_window_for("unknownprovider/some-model") == 200_000
    assert context_window_for("bare-model-id") == 200_000
    # DEV-1639: doubleword IS a known provider now, with its published window.
    assert context_window_for("doubleword/zai-org/GLM-5.2-FP8") == 1_048_576


# ---------------------------------------------------------------------------
# 3. update_context_tokens
# ---------------------------------------------------------------------------

class _FakeAssistant:
    def __init__(self, usage):
        self.usage = usage


_FakeAssistant.__name__ = "AssistantMessage"


def test_update_context_tokens_sums_input_and_cache_dict_usage():
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        update_context_tokens,
    )

    state: dict = {}
    msg = _FakeAssistant(
        {
            "input_tokens": 10,
            "output_tokens": 999,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 200,
        }
    )
    update_context_tokens(state, msg)
    assert state["context_tokens"] == 1210


def test_update_context_tokens_attr_style_usage():
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        update_context_tokens,
    )

    state: dict = {}
    msg = _FakeAssistant(
        SimpleNamespace(
            input_tokens=5,
            cache_read_input_tokens=70,
            cache_creation_input_tokens=25,
        )
    )
    update_context_tokens(state, msg)
    assert state["context_tokens"] == 100


def test_update_context_tokens_ignores_unknown_message_types():
    """Codex r4: ``update_context_tokens`` now reads ``UserMessage``
    (tool_result content) and ``ResultMessage`` (cumulative usage) on
    top of ``AssistantMessage``. Other message types (e.g. SDK
    ``SystemMessage``) are still ignored.
    """
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        update_context_tokens,
    )

    class _Other:
        usage = {"input_tokens": 1}
        content = None

    _Other.__name__ = "SystemMessage"

    state: dict = {}
    update_context_tokens(state, _Other())
    assert "context_tokens" not in state


def test_update_context_tokens_tracks_latest_not_max():
    """Context is the LAST call's size (compaction can shrink it)."""
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        update_context_tokens,
    )

    state: dict = {}
    update_context_tokens(
        state, _FakeAssistant({"input_tokens": 0, "cache_read_input_tokens": 500})
    )
    update_context_tokens(
        state, _FakeAssistant({"input_tokens": 0, "cache_read_input_tokens": 100})
    )
    assert state["context_tokens"] == 100


# ---------------------------------------------------------------------------
# 4. make_context_budget_hook
# ---------------------------------------------------------------------------

def _ctx_hook(state, window=100_000):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        make_context_budget_hook,
    )
    return make_context_budget_hook(state, window)


@pytest.mark.asyncio
async def test_context_hook_silent_below_first_threshold():
    state = {"context_tokens": 79_999}
    hook = _ctx_hook(state)
    assert await hook({"tool_name": "x"}, "tu", None) == {}


@pytest.mark.asyncio
async def test_context_hook_warns_once_at_80_percent():
    state = {"context_tokens": 80_001}
    hook = _ctx_hook(state)
    out = await hook({"tool_name": "x"}, "tu", None)
    hso = out["hookSpecificOutput"]
    assert hso["hookEventName"] == "PostToolUse"
    assert isinstance(hso["additionalContext"], str) and hso["additionalContext"]
    # Same threshold again → silent (one-shot).
    assert await hook({"tool_name": "x"}, "tu", None) == {}


@pytest.mark.asyncio
async def test_context_hook_second_warning_at_90_percent_once():
    state = {"context_tokens": 80_001}
    hook = _ctx_hook(state)
    assert await hook({"tool_name": "x"}, "tu", None) != {}
    state["context_tokens"] = 90_001
    out = await hook({"tool_name": "x"}, "tu", None)
    assert out["hookSpecificOutput"]["additionalContext"]
    assert await hook({"tool_name": "x"}, "tu", None) == {}


@pytest.mark.asyncio
async def test_context_hook_skipping_straight_to_90_warns_once():
    state = {"context_tokens": 95_000}
    hook = _ctx_hook(state)
    assert await hook({"tool_name": "x"}, "tu", None) != {}
    assert await hook({"tool_name": "x"}, "tu", None) == {}


def test_context_hook_callback_name_is_pinned():
    assert _ctx_hook({"context_tokens": 0}).__name__ == "context_budget_warning"


# ---------------------------------------------------------------------------
# 5. Existing guards under subagent-originated calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module_name,submit_tool",
    [
        ("claude_sdk_otf_ainteract_v1", "mcp__bird-interact-tools__submit_query"),
        ("claude_sdk_otf_ainteract_raw_v1", "mcp__bird-interact-tools__submit_sql"),
    ],
)
async def test_ask_gate_satisfied_by_subagent_originated_ask(
    module_name, submit_tool,
):
    """Codex r1 #3 (light remedy): a discovery-subagent ask_user call
    (hook input carries agent_id) increments the counter and unlocks
    submit — the gate is origin-agnostic by design. Both a-interact
    flavors share the contract."""
    import importlib

    m = importlib.import_module(
        f"bird_interact_agents.agents.{module_name}.agent"
    )
    pre_submit_gate, post_ask_counter, _post_nag = m._make_ask_user_guards()

    denied = await pre_submit_gate({"tool_name": submit_tool}, "tu", None)
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"

    await post_ask_counter(
        {
            "tool_name": "mcp__bird-interact-tools__ask_user",
            "agent_id": "agent-discovery-1",
        },
        "tu",
        None,
    )
    assert await pre_submit_gate({"tool_name": submit_tool}, "tu", None) == {}


@pytest.mark.asyncio
async def test_turn_budget_hook_counts_subagent_calls():
    """Hooks fire inside subagent sessions; the combined turn-budget
    counter must treat those calls identically."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as m

    hook = m._make_turn_budget_hook(5, warn_within=3)
    out1 = await hook({"tool_name": "t", "agent_id": "agent-d1"}, "tu", None)
    assert out1 == {}  # remaining 4 > warn_within
    out2 = await hook({"tool_name": "t", "agent_id": "agent-d1"}, "tu", None)
    assert out2["hookSpecificOutput"]["hookEventName"] == "PostToolUse"


# ---------------------------------------------------------------------------
# 6. Usage accumulation for subagent-tagged messages
# ---------------------------------------------------------------------------

def test_accumulate_assistant_usage_counts_subagent_messages(monkeypatch):
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.claude_sdk.agent import (
        accumulate_assistant_usage,
    )
    from bird_interact_agents.usage import TokenUsage

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    class AssistantMessage:
        usage = {
            "input_tokens": 3,
            "output_tokens": 7,
            "cache_read_input_tokens": 11,
            "cache_creation_input_tokens": 13,
        }
        parent_tool_use_id = "tu_task_1"  # subagent-originated

    accum = TokenUsage()
    accumulate_assistant_usage(accum, AssistantMessage(), "anthropic/x")
    assert accum.n_calls == 1
    assert accum.prompt_tokens == 3
    assert accum.completion_tokens == 7
    assert accum.cache_read_tokens == 11
    assert accum.cache_write_tokens == 13
