"""DEV-1561: claude_sdk_otf_ainteract_raw must capture full dataclass
trajectory entries, not 500-char string truncations.

Without this, the discovery subagent's intermediate AssistantMessage /
TaskProgressMessage / tool_use blocks land in the trajectory as truncated
``str(msg)[:500]`` instead of structured dicts — defeating
``_run_capture.extract_tool_stats_from_claude_sdk_trajectory`` and
hiding the subagent's full activity. The non-raw ainteract agent already
captures the dataclass dict; pin the raw variant to the same shape.
"""

from __future__ import annotations

import dataclasses

import pytest

from tests import test_claude_sdk_otf_ainteract_raw_v1_agent as raw_t


@dataclasses.dataclass
class _FakeContentBlock:
    type: str
    name: str | None = None
    id: str | None = None


@dataclasses.dataclass
class _FakeAssistantMessage:
    """Stand-in for the SDK's AssistantMessage. Real shape: a dataclass
    carrying a ``content`` list of tool_use / text blocks. ``asdict`` must
    survive the cycle so trajectory entries are walkable downstream."""

    content: list


_FakeAssistantMessage.__name__ = "AssistantMessage"


@dataclasses.dataclass
class _FakeTaskProgressMessage:
    """Discovery-subagent progress event. Same dataclass shape contract."""

    agent_id: str
    body: str


_FakeTaskProgressMessage.__name__ = "TaskProgressMessage"


@pytest.mark.asyncio
async def test_raw_ainteract_trajectory_entries_carry_dict_data(
    monkeypatch, tmp_path,
):
    """Trajectory entries for both top-level AssistantMessage AND
    subagent TaskProgressMessage land as ``data`` dicts, not strings.
    A non-dict ``data`` defeats the tool-stats walker downstream."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1 import (
        agent as m,
    )

    msg_assistant = _FakeAssistantMessage(
        content=[
            _FakeContentBlock(type="tool_use", name="execute_sql", id="t-1"),
        ],
    )
    msg_progress = _FakeTaskProgressMessage(
        agent_id="discovery", body="exploring schema",
    )

    raw_t._stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[msg_assistant, msg_progress],
    )

    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(raw_t._TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    traj = row["trajectory"]
    assert len(traj) >= 2, traj
    entry_assistant = next(t for t in traj if t["type"] == "AssistantMessage")
    entry_progress = next(t for t in traj if t["type"] == "TaskProgressMessage")
    # The dataclass payload must round-trip as a dict, not a truncated
    # string. ``_run_capture`` walks ``data["content"][...]["type"]``
    # for tool_use counting — that path requires structured data.
    assert isinstance(entry_assistant["data"], dict), entry_assistant
    assert isinstance(entry_assistant["data"].get("content"), list)
    assert (
        entry_assistant["data"]["content"][0]["type"] == "tool_use"
    ), entry_assistant
    assert (
        entry_assistant["data"]["content"][0]["name"] == "execute_sql"
    ), entry_assistant
    assert isinstance(entry_progress["data"], dict), entry_progress
    assert entry_progress["data"]["agent_id"] == "discovery"
    assert entry_progress["data"]["body"] == "exploring schema"
    # Sanity: the old behavior was ``str(msg)[:500]`` — a truncated repr.
    # If a regression reverts to that, this distinguishes string-from-dict.
    assert not isinstance(entry_assistant["data"], str)
    assert not isinstance(entry_progress["data"], str)


@pytest.mark.asyncio
async def test_raw_ainteract_trajectory_handles_non_dataclass_message(
    monkeypatch, tmp_path,
):
    """A non-dataclass message (no asdict path) must still produce a
    trajectory entry — fall back to ``str(msg)`` rather than crashing
    the receive loop. Belt-and-suspenders for SDK message classes that
    aren't dataclasses (e.g., SystemMessage variants)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1 import (
        agent as m,
    )

    class _NotADataclass:
        def __repr__(self):
            return "not-a-dataclass"

    _NotADataclass.__name__ = "WeirdMessage"

    raw_t._stub_env(
        monkeypatch, m, tmp_path / "store",
        messages=[_NotADataclass()],
    )
    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(raw_t._TASK), str(tmp_path), 20.0, "raw", eval_mode="a-interact",
    )
    traj = row["trajectory"]
    entry = next(t for t in traj if t["type"] == "WeirdMessage")
    # Fallback: string. The receive loop didn't crash.
    assert isinstance(entry["data"], str)
    assert "not-a-dataclass" in entry["data"]
