"""DEV-1561 (codex review of plan): a failed ``ClaudeSDKClient.__aenter__()``
must still emit ``[otf_timing] run_task.sdk_client_enter.error elapsed_s=…
exc=<Type>`` — otherwise the attribution channel loses the exact failure
mode it was built to surface (the initialize-handshake timeout).

Before this contract was pinned the agent emitted ``.start`` then nothing on
enter-time failure; the outer ``except Exception`` swallowed the timing.

Covers both a-interact flavors (slayer + raw): the otf_timer + AsyncExitStack
wrapping pattern lives in both ``run_task`` paths and would regress
independently.
"""

from __future__ import annotations

import logging

import pytest

from tests import test_claude_sdk_otf_ainteract_v1_agent as ainteract_t
from tests import test_claude_sdk_otf_ainteract_raw_v1_agent as ainteract_raw_t


_TIMING_LOGGER_NAME = "bird_interact_agents.otf_timing"


class _EnterBoom(RuntimeError):
    """Stand-in for the SDK's initialize-handshake failure (control request
    timeout). The exact type isn't load-bearing — what we pin is the
    timing channel's behaviour when ``__aenter__`` raises."""


def _make_enter_failing_client(captured):
    """A ``ClaudeSDKClient`` look-alike whose ``__aenter__`` raises before
    yielding the client. The agent's ``run_task`` reaches the wrapped
    ``otf_timer`` span; the timer must convert the propagating exception
    into a ``.error`` event before re-raising.
    """
    class _FakeClient:
        def __init__(self, options):
            captured["options"] = options

        async def __aenter__(self):
            raise _EnterBoom("simulated initialize timeout")

        async def __aexit__(self, *a):
            # __aexit__ on a context manager whose __aenter__ raised is
            # never called by the runtime, but provide it defensively so
            # a misuse doesn't surface as a different (less informative)
            # error in the test.
            return None  # pragma: no cover

        async def query(self, *a, **kw):  # pragma: no cover - never reached
            return None

        async def receive_response(self):  # pragma: no cover - never reached
            if False:
                yield None

    return _FakeClient


@pytest.fixture
def caplog_otf(caplog):
    caplog.set_level(logging.INFO, logger=_TIMING_LOGGER_NAME)
    yield caplog


def _timing_msgs(caplog):
    return [r.getMessage() for r in caplog.records if r.name == _TIMING_LOGGER_NAME]


@pytest.mark.asyncio
async def test_sdk_client_enter_failure_emits_error_event_slayer(
    monkeypatch, tmp_path, caplog_otf,
):
    """Slayer ainteract: enter-time failure must surface as
    ``sdk_client_enter.error elapsed_s=… exc=_EnterBoom``."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    captured = ainteract_t._stub_env(monkeypatch, m, tmp_path / "store")
    monkeypatch.setattr(m, "ClaudeSDKClient", _make_enter_failing_client(captured))

    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    # run_task catches the exception and finalises a row with error=...;
    # we just need the timing-channel side effect (the .error log line),
    # so the row's contents are intentionally not asserted.
    row = await agent.run_task(
        dict(ainteract_t._TASK), str(tmp_path), 20.0, "slayer",
        eval_mode="a-interact",
    )
    assert row.get("error"), "agent must finalise an error row when __aenter__ raises"

    msgs = _timing_msgs(caplog_otf)
    starts = [m_ for m_ in msgs if "sdk_client_enter.start" in m_]
    errors = [m_ for m_ in msgs if "sdk_client_enter.error" in m_]
    dones = [m_ for m_ in msgs if "sdk_client_enter.done" in m_]
    assert len(starts) == 1, msgs
    assert len(errors) == 1, msgs
    assert dones == [], msgs  # never reached enter
    err = errors[0]
    assert "elapsed_s=" in err
    assert "exc=_EnterBoom" in err


@pytest.mark.asyncio
async def test_sdk_client_enter_failure_emits_error_event_raw(
    monkeypatch, tmp_path, caplog_otf,
):
    """Raw ainteract: same contract on the no-SLayer variant."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1 import (
        agent as m,
    )

    captured = ainteract_raw_t._stub_env(monkeypatch, m, tmp_path / "store")
    monkeypatch.setattr(m, "ClaudeSDKClient", _make_enter_failing_client(captured))

    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    row = await agent.run_task(
        dict(ainteract_raw_t._TASK), str(tmp_path), 20.0, "raw",
        eval_mode="a-interact",
    )
    assert row.get("error"), "agent must finalise an error row when __aenter__ raises"

    msgs = _timing_msgs(caplog_otf)
    starts = [m_ for m_ in msgs if "sdk_client_enter.start" in m_]
    errors = [m_ for m_ in msgs if "sdk_client_enter.error" in m_]
    dones = [m_ for m_ in msgs if "sdk_client_enter.done" in m_]
    assert len(starts) == 1, msgs
    assert len(errors) == 1, msgs
    assert dones == [], msgs
    err = errors[0]
    assert "elapsed_s=" in err
    assert "exc=_EnterBoom" in err


@pytest.mark.asyncio
async def test_sdk_client_enter_success_emits_done_not_error_slayer(
    monkeypatch, tmp_path, caplog_otf,
):
    """Positive control: under the normal happy path the channel emits
    ``.done`` (with ``elapsed_s``) and NO ``.error``. Pins the AsyncExitStack
    + otf_timer wrapping doesn't accidentally swallow the success case.
    """
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as m

    ainteract_t._stub_env(monkeypatch, m, tmp_path / "store")
    # The default ainteract_t._stub_env installs a fake ClaudeSDKClient whose
    # __aenter__ returns self — that's the happy path. No further patching.

    agent = m.ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(ainteract_t._TASK), str(tmp_path), 20.0, "slayer",
        eval_mode="a-interact",
    )
    msgs = _timing_msgs(caplog_otf)
    done_msgs = [m_ for m_ in msgs if "sdk_client_enter.done" in m_]
    assert done_msgs, msgs
    # Docstring contract: success emits `.done` WITH `elapsed_s` — the
    # whole point of the timer is the attribution payload, not just the
    # marker. Pin it.
    assert any("elapsed_s=" in m_ for m_ in done_msgs), msgs
    assert not any("sdk_client_enter.error" in m_ for m_ in msgs), msgs


@pytest.mark.asyncio
async def test_sdk_client_enter_success_emits_done_not_error_raw(
    monkeypatch, tmp_path, caplog_otf,
):
    """Raw-variant positive control. The raw a-interact agent has its own
    copy of the AsyncExitStack + otf_timer wrap; it can regress independently
    from the slayer variant (different test file, different `_stub_env`).
    Pin the same `.done` + `elapsed_s` contract on the raw path.
    """
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1 import (
        agent as m,
    )

    ainteract_raw_t._stub_env(monkeypatch, m, tmp_path / "store")

    agent = m.ClaudeSDKOtfAInteractRawAgent(model="anthropic/claude-sonnet-4-5")
    await agent.run_task(
        dict(ainteract_raw_t._TASK), str(tmp_path), 20.0, "raw",
        eval_mode="a-interact",
    )
    msgs = _timing_msgs(caplog_otf)
    done_msgs = [m_ for m_ in msgs if "sdk_client_enter.done" in m_]
    assert done_msgs, msgs
    assert any("elapsed_s=" in m_ for m_ in done_msgs), msgs
    assert not any("sdk_client_enter.error" in m_ for m_ in msgs), msgs
