"""DEV-1555 Stage 1, Codex r1 #2: live verification that ClaudeAgentOptions
hooks fire for tool calls made INSIDE a Task subagent, with ``agent_id``
populated in the hook input.

The whole partition design rests on this SDK behavior (doc-cited but never
exercised in this repo): the ``partition_deny`` PreToolUse hook must see
subagent-originated calls WITH ``agent_id`` (so it allows them) and
main-loop calls WITHOUT it (so it denies discovery-only tools).

Integration-marked: spawns the real claude CLI and makes a paid haiku call.
Run with ``pytest -m integration -k dev1555``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_hooks_fire_for_subagent_tool_calls_with_agent_id():
    from claude_agent_sdk import (
        AgentDefinition,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        create_sdk_mcp_server,
        tool,
    )
    from claude_agent_sdk.types import HookMatcher

    seen: list[dict] = []

    @tool("probe", "Return a fixed probe string.", {})
    async def probe(args):
        return {"content": [{"type": "text", "text": "probe-ok"}]}

    async def record_tool_use(input_data, tool_use_id, context):
        seen.append(
            {
                "tool_name": input_data.get("tool_name"),
                "agent_id": input_data.get("agent_id"),
            }
        )
        return {}

    record_tool_use.__name__ = "record_tool_use"

    server = create_sdk_mcp_server(name="probe-tools", tools=[probe])
    options = ClaudeAgentOptions(
        system_prompt=(
            "You MUST delegate to the 'discovery' subagent (Task tool) and "
            "have it call the probe tool once. Then reply DONE."
        ),
        mcp_servers={"probe-tools": server},
        tools=["Task"],
        allowed_tools=["Task", "mcp__probe-tools__probe"],
        agents={
            "discovery": AgentDefinition(
                description="Calls the probe tool once and reports back.",
                prompt="Call mcp__probe-tools__probe once, then summarize.",
                tools=["mcp__probe-tools__probe"],
                maxTurns=4,
            )
        },
        setting_sources=[],
        model="claude-haiku-4-5-20251001",
        max_turns=6,
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[record_tool_use])]},
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Run the probe via the discovery subagent.")
        async for _msg in client.receive_response():
            pass

    probe_calls = [s for s in seen if s["tool_name"] == "mcp__probe-tools__probe"]
    assert probe_calls, f"probe never called; hook saw: {seen}"
    # The subagent-originated call must carry agent_id — the partition
    # deny hook keys on exactly this discriminator.
    assert any(c["agent_id"] for c in probe_calls), (
        f"no agent_id on subagent probe calls: {probe_calls}"
    )
    # The main loop's Task invocation must be hooked and carry NO agent_id
    # (Codex test-review #2: assert non-empty so this can't pass vacuously).
    task_calls = [s for s in seen if s["tool_name"] == "Task"]
    assert task_calls, f"Task call never seen by the hook: {seen}"
    for c in task_calls:
        assert not c["agent_id"]
