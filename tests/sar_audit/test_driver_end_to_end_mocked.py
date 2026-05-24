"""End-to-end mocked driver run + chdir containment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bird_interact_agents.sar_audit import driver
from tests.sar_audit._stubs import StubSARRunResult, StubSARVerdict


def _tasks():
    return [
        {
            "instance_id": "fake_1",
            "selected_database": "fake",
            "sol_sql": ["SELECT x FROM t ORDER BY x LIMIT 1"],
            "amb_user_query": "first",
            "external_knowledge": [1],
            "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
            "knowledge_ambiguity": [],
        },
        {
            "instance_id": "fake_2",
            "selected_database": "fake",
            "sol_sql": ["SELECT x FROM t ORDER BY x DESC LIMIT 1"],
            "amb_user_query": "second",
            "external_knowledge": [1],
            "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
            "knowledge_ambiguity": [],
        },
    ]


def test_two_task_clean_then_edited_run(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, read_jsonl
):
    factory, handle = stub_sar_agent

    # First task: clean.
    handle.queue(
        StubSARRunResult(
            verdict=StubSARVerdict(
                correctness_flag=True, ambiguity_flag=False, reasoning="all good"
            ),
            step_count=3,
            cost_usd=0.002,
            audit_model_actual="claude-opus-4-7-20260121",
        )
    )
    # Second task: edited.
    handle.queue(
        StubSARRunResult(
            verdict=StubSARVerdict(
                correctness_flag=False,
                ambiguity_flag=False,
                revised_sql="SELECT x FROM t ORDER BY x DESC LIMIT 1 -- revised",
                reasoning="missing comment",
            ),
            step_count=7,
            cost_usd=0.01,
            audit_model_actual="claude-opus-4-7-20260121",
        )
    )

    output_dir = tmp_path / "out"
    result = driver.run_db(
        db="fake",
        tasks=_tasks(),
        db_path=fake_db,
        full_kb=[{"id": 1, "knowledge": "anything"}],
        full_column_meanings={"t|x": "the column"},
        audit_model="claude-opus-4-7",
        max_steps=5,
        output_dir=output_dir,
        sar_agent_factory=factory,
    )

    rows = read_jsonl(output_dir / "fake_sar_audited.jsonl")
    assert len(rows) == 2

    r1 = next(r for r in rows if r["instance_id"] == "fake_1")
    assert r1["audit_status"] == "clean"
    assert r1["changes"] == []
    assert r1["audit_model_requested"] == "claude-opus-4-7"
    assert r1["audit_model_actual"] == "claude-opus-4-7-20260121"
    assert r1["skill_version"] == "sar-agent/1.0"
    assert r1["audited_at"] == fixed_now
    # Read-only sqlite executed and stored the actual smallest x.
    assert r1["audited_sample_row"] == [1]
    assert r1["audited_sample_row_status"] == "ok"

    r2 = next(r for r in rows if r["instance_id"] == "fake_2")
    assert r2["audit_status"] == "edited"
    assert r2["audited_sol_sql"] == [
        "SELECT x FROM t ORDER BY x DESC LIMIT 1 -- revised"
    ]
    assert len(r2["changes"]) == 1
    assert r2["changes"][0]["clause_kind"] == "sar_revision"
    assert r2["changes"][0]["source"] == "sar_agent"

    assert result.audited == 2
    assert result.skipped == 0
    assert result.failed == 0


def test_second_invocation_skips_both(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, read_jsonl
):
    factory, handle = stub_sar_agent

    # First run.
    for _ in range(2):
        handle.queue(
            StubSARRunResult(
                verdict=StubSARVerdict(correctness_flag=True, ambiguity_flag=False),
                audit_model_actual="claude-opus-4-7-20260121",
            )
        )

    output_dir = tmp_path / "out"
    driver.run_db(
        db="fake",
        tasks=_tasks(),
        db_path=fake_db,
        full_kb=[],
        full_column_meanings={},
        audit_model="claude-opus-4-7",
        max_steps=5,
        output_dir=output_dir,
        sar_agent_factory=factory,
    )

    rows_after_first = read_jsonl(output_dir / "fake_sar_audited.jsonl")
    constructed_after_first = list(handle.constructed_with)

    # Second run — no new verdicts queued, expect no SAR constructions.
    result = driver.run_db(
        db="fake",
        tasks=_tasks(),
        db_path=fake_db,
        full_kb=[],
        full_column_meanings={},
        audit_model="claude-opus-4-7",
        max_steps=5,
        output_dir=output_dir,
        sar_agent_factory=factory,
    )

    rows_after_second = read_jsonl(output_dir / "fake_sar_audited.jsonl")
    assert rows_after_first == rows_after_second
    assert handle.constructed_with == constructed_after_first
    assert result.audited == 0
    assert result.skipped == 2


def test_end_to_end_run_never_instantiates_openai(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, monkeypatch
):
    """No code path through `driver.run_db` may instantiate an OpenAI client.

    Guards against accidental drift in the SARAgent wiring: even if the
    upstream library tries to construct its own OpenAI client, our adapter
    must intercept before that happens.
    """
    factory, handle = stub_sar_agent
    handle.queue(
        StubSARRunResult(
            verdict=StubSARVerdict(correctness_flag=True, ambiguity_flag=False),
            audit_model_actual="claude-opus-4-7-20260121",
        )
    )

    import openai

    instantiations = []

    class TrapOpenAI:
        def __init__(self, *args, **kwargs):
            instantiations.append((args, kwargs))
            raise AssertionError("OpenAI must not be instantiated during a SAR-audit run")

    monkeypatch.setattr(openai, "OpenAI", TrapOpenAI)
    if hasattr(openai, "AsyncOpenAI"):
        monkeypatch.setattr(openai, "AsyncOpenAI", TrapOpenAI)

    output_dir = tmp_path / "out"
    driver.run_db(
        db="fake",
        tasks=_tasks()[:1],
        db_path=fake_db,
        full_kb=[],
        full_column_meanings={},
        audit_model="claude-opus-4-7",
        max_steps=5,
        output_dir=output_dir,
        sar_agent_factory=factory,
    )

    assert instantiations == []


def test_chdir_containment_no_analyze_result_leak(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, monkeypatch
):
    """SAR-Agent writes to ./analyze_result/results.jsonl unconditionally.
    The driver chdirs into a temp dir around the agent invocation so that
    file ends up in /tmp, not in our working tree."""
    factory, handle = stub_sar_agent

    # Make the stub also write to ./analyze_result/results.jsonl to simulate
    # the side-effect upstream does.
    class SideEffectFactory:
        def __init__(self, base):
            self._base = base

        def __call__(self, **kwargs):
            agent = self._base(**kwargs)
            original_run = agent.run

            def patched_run(*, max_steps):
                # Simulate SAR-Agent's hard-coded side effect.
                Path("./analyze_result").mkdir(exist_ok=True)
                Path("./analyze_result/results.jsonl").write_text("simulated leak\n")
                return original_run(max_steps=max_steps)

            agent.run = patched_run
            return agent

    handle.queue(
        StubSARRunResult(
            verdict=StubSARVerdict(correctness_flag=True, ambiguity_flag=False),
            audit_model_actual="claude-opus-4-7-20260121",
        )
    )

    # Anchor cwd to a clean dir.
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    output_dir = tmp_path / "out"
    driver.run_db(
        db="fake",
        tasks=_tasks()[:1],
        db_path=fake_db,
        full_kb=[],
        full_column_meanings={},
        audit_model="claude-opus-4-7",
        max_steps=5,
        output_dir=output_dir,
        sar_agent_factory=SideEffectFactory(factory),
    )

    # No `./analyze_result/` directory leaked into the workdir.
    assert not (workdir / "analyze_result").exists()
