"""DEV-1545: `make_runner` propagates `user_sim_prompt_version` into
the closure-bound `agent.run_task(...)` call.

Mechanical contract per Codex review:
  * #9: kwarg arrives at agent unchanged when explicit;
  * #10: omitted CLI flag → run.py normalises `None` → `"v2"` at the
    closure level before calling `run_task` (Python defaults DON'T apply
    to explicit None).

Pattern follows tests/test_run_framework_dispatch.py — monkeypatch the
agent class so its `run_task` is a stub that records the kwargs it was
called with.
"""

from __future__ import annotations

import pytest

import bird_interact_agents.run as run_mod


def _runner_kwargs(**overrides) -> dict:
    base = dict(
        framework="claude_sdk",
        dataset="mini-interact",
        query_mode="slayer",
        mode="a-interact",
        agent_model="anthropic/claude-haiku-4-5-20251001",
        strict=False,
        prompt_cache=False,
        max_depth=3,
        slayer_storage_root=None,
        slayer_setup="on-the-fly",
    )
    base.update(overrides)
    return base


class _RecordingAgent:
    """Captures the kwargs of each run_task call into a class-level list
    so the test can inspect them after the closure fires."""

    last_kwargs: dict | None = None

    def __init__(self, **_kwargs):
        pass

    async def run_task(self, td, data_dir, budget, query_mode, **kwargs):
        type(self).last_kwargs = kwargs
        return {"reward": 0.0}


def _patch_ainteract_agent(monkeypatch):
    """Both the `claude_sdk` path's `_make_runner` and the dedicated
    `claude_sdk_otf_ainteract` path import `ClaudeSDKOtfAInteractAgent`
    from `bird_interact_agents.agents.claude_sdk_otf_ainteract`. Patch
    both possible attribute paths so whichever the dispatch chooses
    lands on our stub."""
    _RecordingAgent.last_kwargs = None
    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_ainteract.ClaudeSDKOtfAInteractAgent",
        _RecordingAgent,
        raising=False,
    )


@pytest.mark.asyncio
async def test_explicit_v3_propagates_to_run_task(monkeypatch) -> None:
    """Explicit `user_sim_prompt_version="v3"` reaches the agent's
    `run_task` unchanged."""
    _patch_ainteract_agent(monkeypatch)

    runner = run_mod.make_runner(
        **_runner_kwargs(),
        user_sim_prompt_version="v3",
    )
    await runner(
        {"selected_database": "fakedb", "instance_id": "f_1"},
        "/tmp/data", 100, "anthropic/claude-haiku-4-5-20251001",
    )
    assert _RecordingAgent.last_kwargs is not None
    assert _RecordingAgent.last_kwargs.get("user_sim_prompt_version") == "v3"


@pytest.mark.asyncio
async def test_explicit_v2_propagates_to_run_task(monkeypatch) -> None:
    _patch_ainteract_agent(monkeypatch)

    runner = run_mod.make_runner(
        **_runner_kwargs(),
        user_sim_prompt_version="v2",
    )
    await runner(
        {"selected_database": "fakedb", "instance_id": "f_1"},
        "/tmp/data", 100, "anthropic/claude-haiku-4-5-20251001",
    )
    assert _RecordingAgent.last_kwargs.get("user_sim_prompt_version") == "v2"


@pytest.mark.asyncio
async def test_none_normalised_to_v2_at_closure(monkeypatch) -> None:
    """Codex #10: when the CLI flag is absent, `user_sim_prompt_version`
    flows as `None` through the cloud + run layers. The closure in
    `run.py` must normalise that to `"v2"` before calling
    `agent.run_task` — otherwise the explicit `None` shadows the agent's
    Python `"v2"` default and falls through to a missing-key crash on
    upstream `USER_SIMULATOR_ENCODER`."""
    _patch_ainteract_agent(monkeypatch)

    runner = run_mod.make_runner(
        **_runner_kwargs(),
        user_sim_prompt_version=None,
    )
    await runner(
        {"selected_database": "fakedb", "instance_id": "f_1"},
        "/tmp/data", 100, "anthropic/claude-haiku-4-5-20251001",
    )
    assert _RecordingAgent.last_kwargs.get("user_sim_prompt_version") == "v2", (
        "None must be normalised to 'v2' at the closure — explicit None "
        "would shadow the agent's Python default."
    )


@pytest.mark.asyncio
async def test_omitted_kwarg_defaults_to_v2_at_closure(monkeypatch) -> None:
    """`make_runner` is also called without the kwarg by older callers
    (e.g., resubmit paths reconstructed from an old manifest). The
    closure must still pass `"v2"` to `run_task`."""
    _patch_ainteract_agent(monkeypatch)

    runner = run_mod.make_runner(**_runner_kwargs())
    await runner(
        {"selected_database": "fakedb", "instance_id": "f_1"},
        "/tmp/data", 100, "anthropic/claude-haiku-4-5-20251001",
    )
    assert _RecordingAgent.last_kwargs.get("user_sim_prompt_version") == "v2"
