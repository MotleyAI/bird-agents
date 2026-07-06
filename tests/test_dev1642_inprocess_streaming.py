"""DEV-1642: the legacy in-process local path (BIRD_INTERACT_LOCAL_INPROCESS=1)
must ALSO stream the claude_sdk trajectory to rows/<iid>/partial_transcript.jsonl
as it arrives — parity with the default process-pool path. The wrapping is a
small helper, ``run._inprocess_transcript_recorder``, so it is unit-testable
without driving a full ``run_evaluation``.
"""
from __future__ import annotations

import contextlib
import json

import pytest

from bird_interact_agents import run
from bird_interact_agents.agents.claude_sdk import sdk_env


def test_inprocess_recorder_streams_claude_sdk_to_rows(tmp_path):
    rows_dir = tmp_path / "rows"
    dest = rows_dir / "db_a_1" / "partial_transcript.jsonl"

    cm = run._inprocess_transcript_recorder(rows_dir, "db_a_1", "claude_sdk_otf")
    with cm:
        sink = sdk_env._partial_transcript_sink.get()
        assert sink is not None, "recorder must install the partial-transcript sink"
        sink({"type": "AssistantMessage", "data": "one"})
        # Incremental: on disk before the next message / before completion.
        assert json.loads(dest.read_text(encoding="utf-8").splitlines()[0])["data"] == "one"
        sink({"type": "ResultMessage", "data": "two"})

    lines = dest.read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["data"] for x in lines] == ["one", "two"]


def test_inprocess_recorder_creates_row_dir(tmp_path):
    # rows/<iid>/ need not pre-exist — the helper makes the parent so the
    # append can open the file.
    rows_dir = tmp_path / "rows"
    cm = run._inprocess_transcript_recorder(rows_dir, "fresh_iid", "claude_sdk")
    with cm:
        sink = sdk_env._partial_transcript_sink.get()
        sink({"type": "T", "data": "x"})
    assert (rows_dir / "fresh_iid" / "partial_transcript.jsonl").exists()


def test_inprocess_recorder_noop_for_non_claude_framework(tmp_path):
    rows_dir = tmp_path / "rows"
    cm = run._inprocess_transcript_recorder(rows_dir, "db_a_1", "pydantic_ai")
    with cm:
        # No sink installed for non-claude frameworks.
        assert sdk_env._partial_transcript_sink.get() is None
    # And no partial file was created.
    assert not (rows_dir / "db_a_1" / "partial_transcript.jsonl").exists()


@pytest.mark.asyncio
async def test_run_with_sem_wraps_runner_in_the_recorder(tmp_path, monkeypatch):
    """Wiring guard (Codex): the in-process ``_run_with_sem`` loop MUST call
    ``_inprocess_transcript_recorder(rows_dir, iid, framework)`` around the
    runner, so the streaming can't silently regress. Reuses the proven
    escape-hatch harness (BIRD_INTERACT_LOCAL_INPROCESS=1, oracle mode); the
    spy returns a no-op CM so behaviour is unchanged."""
    monkeypatch.setenv("BIRD_INTERACT_LOCAL_INPROCESS", "1")

    from bird_interact_agents import usage as usage_mod
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    tasks = [
        {"instance_id": "t1", "selected_database": "fake", "amb_user_query": "q1"},
    ]
    monkeypatch.setattr(run, "load_benchmark_tasks", lambda *a, **k: tasks)
    monkeypatch.setattr(run, "calculate_budget", lambda *a, **kw: 18)

    async def fake_oracle(td, dpb):
        return {
            "task_id": td["instance_id"], "instance_id": td["instance_id"],
            "database": "fake", "phase1_passed": False, "phase2_passed": False,
            "total_reward": 0.0, "trajectory": [], "error": None,
            "usage": usage_mod.TokenUsage().model_dump(),
        }

    monkeypatch.setattr(run, "run_oracle_task", fake_oracle)

    def _boom(**kwargs):
        raise AssertionError("process pool must NOT run under the escape hatch")

    monkeypatch.setattr(run, "dispatch_local_process_pool", _boom)

    calls: list = []

    def _spy(rows_dir, instance_id, framework):
        calls.append((str(rows_dir), instance_id, framework))
        return contextlib.nullcontext()

    monkeypatch.setattr(run, "_inprocess_transcript_recorder", _spy)

    out = tmp_path / "eval.json"
    await run.run_evaluation(
        data_path="ignored", data_dir="ignored", output_path=str(out),
        mode="oracle", query_mode="raw", framework="pydantic_ai",
        concurrency=1,
    )

    # _run_with_sem invoked the recorder exactly once, for our task, with the
    # run's framework and a rows/ directory.
    assert len(calls) == 1
    rows_dir_str, iid, framework = calls[0]
    assert iid == "t1"
    assert framework == "pydantic_ai"
    assert rows_dir_str.endswith("rows")
