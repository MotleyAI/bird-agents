"""DEV-1555 Stage 2: context_window_for consults the provider registry.

Stage-1 pins (anthropic -> 1M behavior-preserving, unknown -> 200K
conservative) must survive; registry models get their real windows.
"""

from __future__ import annotations

from bird_interact_agents.agents.claude_sdk.context_budget import (
    context_window_for,
)


def test_stage1_pins_unchanged():
    assert context_window_for("anthropic/claude-opus-4-7") == 1_000_000
    assert context_window_for("anthropic/claude-sonnet-4-6") == 1_000_000
    assert context_window_for("unknownprov/some-model") == 200_000
    assert context_window_for("bare-model-id") == 200_000


def test_moonshot_window_from_registry():
    assert context_window_for("moonshot/kimi-k2.7-code") == 262_144


def test_per_model_override_beats_provider_default(monkeypatch):
    from bird_interact_agents import provider_registry as pr  # noqa: PLC0415

    spec = pr.get_provider("moonshot/kimi-k2.7-code")
    patched = spec.model_copy(
        update={"model_context_windows": {"kimi-special": 123_456}}
    )
    monkeypatch.setitem(pr.REGISTRY, "moonshot", patched)
    assert context_window_for("moonshot/kimi-special") == 123_456
    # Other models of the same provider keep the provider default.
    assert context_window_for("moonshot/kimi-k2.7-code") == 262_144


# ---------------------------------------------------------------------------
# Codex r1 / CR r1: hardening for `update_context_tokens`, `per_task_timeout_s`,
# and the wall-clock deny hook.
# ---------------------------------------------------------------------------


class _FakeAssistantMessage:
    """Bare class whose ``type(msg).__name__`` is ``AssistantMessage``."""


_FakeAssistantMessage.__name__ = "AssistantMessage"


class _FakeResultMessage:
    pass


_FakeResultMessage.__name__ = "ResultMessage"


def _make_msg(cls, usage):
    msg = cls()
    msg.usage = usage
    return msg


def test_update_context_tokens_reads_assistant_message_usage():
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        update_context_tokens,
    )

    state: dict = {}
    msg = _make_msg(_FakeAssistantMessage, {
        "input_tokens": 1000, "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 50,
    })
    update_context_tokens(state, msg)
    assert state["context_tokens"] == 1250


def test_update_context_tokens_reads_result_message_when_assistant_was_zero():
    """Moonshot/Kimi reports zero per-turn AssistantMessage.usage; only the
    terminal ResultMessage carries the real cumulative numbers. Without
    reading ResultMessage, ``context_tokens`` stays at 0 and the 80%/90%
    warnings never fire for the open-weight target."""
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        update_context_tokens,
    )

    state: dict = {}
    # Stream: zero-valued AssistantMessage then ResultMessage with cumulative.
    update_context_tokens(state, _make_msg(_FakeAssistantMessage, {
        "input_tokens": 0, "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }))
    assert state.get("context_tokens", 0) == 0
    update_context_tokens(state, _make_msg(_FakeResultMessage, {
        "input_tokens": 50_000, "cache_read_input_tokens": 100_000,
        "cache_creation_input_tokens": 12_345,
    }))
    assert state["context_tokens"] == 162_345


def test_update_context_tokens_ignores_other_message_types():
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        update_context_tokens,
    )

    class _Other:
        pass

    _Other.__name__ = "UserMessage"
    msg = _Other()
    msg.usage = {"input_tokens": 9999}
    state: dict = {"context_tokens": 7}
    update_context_tokens(state, msg)
    assert state["context_tokens"] == 7


def test_per_task_timeout_falls_back_on_nan(monkeypatch):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        per_task_timeout_s,
        _DEFAULT_AGENT_BUDGET_S,
    )

    monkeypatch.setenv("BIRD_INTERACT_PER_TASK_TIMEOUT_S", "nan")
    assert per_task_timeout_s() == _DEFAULT_AGENT_BUDGET_S


def test_per_task_timeout_falls_back_on_inf(monkeypatch):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        per_task_timeout_s,
        _DEFAULT_AGENT_BUDGET_S,
    )

    monkeypatch.setenv("BIRD_INTERACT_PER_TASK_TIMEOUT_S", "inf")
    assert per_task_timeout_s() == _DEFAULT_AGENT_BUDGET_S


def test_per_task_timeout_accepts_finite_float(monkeypatch):
    from bird_interact_agents.agents.claude_sdk.context_budget import (
        per_task_timeout_s,
    )

    monkeypatch.setenv("BIRD_INTERACT_PER_TASK_TIMEOUT_S", "1234")
    assert per_task_timeout_s() == 1234.0


def test_wall_clock_deny_leaf_exact_match_for_submit_tool(monkeypatch):
    """The deny hook must allow ONLY the configured submit_tool past the
    deadline — not any tool whose name CONTAINS `submit_query` /
    `submit_sql` as a substring."""
    import asyncio
    from bird_interact_agents.agents.claude_sdk import context_budget as cb

    state = {"wall_clock_start": 0.0}
    monkeypatch.setattr(cb.time, "monotonic", lambda: 9999.0)
    _warn, hook = cb.make_wall_clock_budget_hook(
        state, budget_s=1.0, submit_tool="submit_query",
    )
    # Allow: the exact submit tool, registered under our MCP namespace.
    assert (
        asyncio.run(hook(
            {"tool_name": "mcp__bird-interact-tools__submit_query"}, None, None,
        ))
        == {}
    )
    # Deny: an unrelated third-party tool whose name CONTAINS the substring.
    out = asyncio.run(hook(
        {"tool_name": "mcp__other-server__do_submit_query_thing"}, None, None,
    ))
    assert out.get("hookSpecificOutput", {}).get(
        "permissionDecision"
    ) == "deny", out


def test_wall_clock_deny_no_op_inside_budget(monkeypatch):
    """Belt: the deny hook is a no-op when budget hasn't elapsed."""
    import asyncio
    from bird_interact_agents.agents.claude_sdk import context_budget as cb

    state = {"wall_clock_start": 0.0}
    monkeypatch.setattr(cb.time, "monotonic", lambda: 0.5)
    _warn, hook = cb.make_wall_clock_budget_hook(
        state, budget_s=1.0, submit_tool="submit_query",
    )
    assert (
        asyncio.run(hook({"tool_name": "anything"}, None, None)) == {}
    )
