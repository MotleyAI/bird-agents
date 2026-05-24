"""SARAuditLoop.run — happy path + failure modes that must surface to the driver."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bird_interact_agents.sar_audit.audit_loop import SARAuditLoop


class _FakeBlock:
    def __init__(self, *, type, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, *, content, stop_reason="end_turn", model="claude-opus-4-7-20260121"):
        self.content = content
        self.stop_reason = stop_reason
        self.model = model
        self.usage = _FakeUsage()


class _FakeMessages:
    """Returns queued responses in order."""

    def __init__(self, queued):
        self._queued = list(queued)

    def create(self, **kwargs):
        if not self._queued:
            raise RuntimeError("FakeMessages: out of queued responses")
        return self._queued.pop(0)


class _FakeAnthropic:
    def __init__(self, queued):
        self.messages = _FakeMessages(queued)


@pytest.fixture
def fake_db_for_loop(tmp_path: Path) -> Path:
    p = tmp_path / "loop.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()
    return p


def test_run_happy_path_single_terminate(fake_db_for_loop, monkeypatch):
    """One LLM turn → terminate → parsed verdict."""
    import anthropic

    queued = [
        _FakeResponse(
            content=[
                _FakeBlock(type="text", text="Reasoning…"),
                _FakeBlock(
                    type="tool_use",
                    id="tu1",
                    name="terminate",
                    input={
                        "analyze_result": (
                            "Correctness: Yes\n"
                            "Is_ambiguous: No\n"
                            "Explanation: all good\n"
                            "Revised:"
                        )
                    },
                ),
            ],
            stop_reason="tool_use",
        )
    ]
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: _FakeAnthropic(queued))

    loop = SARAuditLoop(
        model="claude-opus-4-7",
        prompt="audit me",
        api_key="sk-test",
        db_path=fake_db_for_loop,
    )
    result = loop.run(max_steps=30)

    assert result.verdict.correctness_flag is True
    assert result.verdict.ambiguity_flag is False
    assert result.step_count == 1


def test_run_exhausts_steps_without_terminate_raises(fake_db_for_loop, monkeypatch):
    """Model never calls `terminate` → loop raises so the driver routes
    the failure to the sidecar instead of writing a faux-clean row."""
    import anthropic

    # Two turns of read_sqlite_query (never terminating), then we cap
    # the loop with max_steps=2.
    sql_call = _FakeBlock(
        type="tool_use",
        id="tu1",
        name="read_sqlite_query",
        input={"query": "SELECT 1", "explaination": "probe"},
    )
    queued = [
        _FakeResponse(content=[sql_call], stop_reason="tool_use"),
        _FakeResponse(content=[sql_call], stop_reason="tool_use"),
    ]
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: _FakeAnthropic(queued))

    loop = SARAuditLoop(
        model="claude-opus-4-7",
        prompt="audit me",
        api_key="sk-test",
        db_path=fake_db_for_loop,
    )

    with pytest.raises(RuntimeError, match="exhausted max_steps"):
        loop.run(max_steps=2)


def test_run_read_sqlite_then_terminate(fake_db_for_loop, monkeypatch):
    """Two turns: one read probe, then terminate."""
    import anthropic

    queued = [
        _FakeResponse(
            content=[
                _FakeBlock(
                    type="tool_use",
                    id="tu1",
                    name="read_sqlite_query",
                    input={"query": "SELECT x FROM t LIMIT 5", "explaination": "probe"},
                ),
            ],
            stop_reason="tool_use",
        ),
        _FakeResponse(
            content=[
                _FakeBlock(
                    type="tool_use",
                    id="tu2",
                    name="terminate",
                    input={
                        "analyze_result": (
                            "Correctness: No\n"
                            "Is_ambiguous: No\n"
                            "Explanation: SQL is wrong\n"
                            "Revised: SELECT DISTINCT x FROM t"
                        )
                    },
                ),
            ],
            stop_reason="tool_use",
        ),
    ]
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: _FakeAnthropic(queued))

    loop = SARAuditLoop(
        model="claude-opus-4-7",
        prompt="audit me",
        api_key="sk-test",
        db_path=fake_db_for_loop,
    )
    result = loop.run(max_steps=30)

    assert result.verdict.correctness_flag is False
    assert result.verdict.revised_sql is not None
    assert "DISTINCT" in result.verdict.revised_sql
    assert result.step_count == 2
