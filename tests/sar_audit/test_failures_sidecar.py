"""Driver writes failures to a sibling JSONL and exits non-zero."""

from __future__ import annotations

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
            "external_knowledge": [],
            "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
            "knowledge_ambiguity": [],
        },
        {
            "instance_id": "fake_2",
            "selected_database": "fake",
            "sol_sql": ["SELECT x FROM t ORDER BY x DESC LIMIT 1"],
            "amb_user_query": "second",
            "external_knowledge": [],
            "user_query_ambiguity": {"critical_ambiguity": [], "non_critical_ambiguity": []},
            "knowledge_ambiguity": [],
        },
    ]


class _CountingFactory:
    """Wraps the stub factory to make fake_2 raise."""

    def __init__(self, base_factory):
        self._base = base_factory
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        # On the second call (fake_2), the constructed agent's run() raises.
        if self.calls == 2:
            class Erroring:
                def run(self, *, max_steps: int):
                    raise RuntimeError("fake LLM rate-limit (RateLimitError)")
            return Erroring()
        return self._base(**kwargs)


def test_failure_writes_sidecar_and_returns_nonzero(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, read_jsonl
):
    factory, handle = stub_sar_agent
    handle.queue(
        StubSARRunResult(
            verdict=StubSARVerdict(correctness_flag=True, ambiguity_flag=False),
            audit_model_actual="claude-opus-4-7-20260121",
        )
    )

    counting_factory = _CountingFactory(factory)
    output_dir = tmp_path / "out"

    result = driver.run_db(
        db="fake",
        tasks=_tasks(),
        db_path=fake_db,
        full_kb=[],
        full_column_meanings={},
        audit_model="claude-opus-4-7",
        max_steps=5,
        output_dir=output_dir,
        sar_agent_factory=counting_factory,
    )

    audited = read_jsonl(output_dir / "fake_sar_audited.jsonl")
    failures = read_jsonl(output_dir / "fake_sar_failures.jsonl")

    # Main JSONL has only fake_1.
    assert {r["instance_id"] for r in audited} == {"fake_1"}
    # Failures JSONL has fake_2 with the right shape.
    assert len(failures) == 1
    fail = failures[0]
    assert fail["instance_id"] == "fake_2"
    assert fail["selected_database"] == "fake"
    assert fail["audit_model_requested"] == "claude-opus-4-7"
    assert "RateLimitError" in fail["error_message"] or fail["error_class"] == "RuntimeError"
    assert fail["skill_version"] == "sar-agent/1.0"
    assert "failed_at" in fail

    # Driver result reports failure count.
    assert result.failed == 1
    assert result.audited == 1


def test_main_jsonl_unchanged_when_only_failures(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, read_jsonl
):
    """If every task fails, the main JSONL is not created (or empty)."""
    factory, handle = stub_sar_agent

    class _AlwaysErrors:
        def __init__(self, base):
            self._base = base
            self.calls = 0

        def __call__(self, **kwargs):
            class Erroring:
                def run(self, *, max_steps: int):
                    raise RuntimeError("always fails")
            return Erroring()

    output_dir = tmp_path / "out"

    result = driver.run_db(
        db="fake",
        tasks=_tasks(),
        db_path=fake_db,
        full_kb=[],
        full_column_meanings={},
        audit_model="claude-opus-4-7",
        max_steps=5,
        output_dir=output_dir,
        sar_agent_factory=_AlwaysErrors(factory),
    )

    audited = read_jsonl(output_dir / "fake_sar_audited.jsonl")
    failures = read_jsonl(output_dir / "fake_sar_failures.jsonl")

    assert audited == []
    assert len(failures) == 2
    assert result.failed == 2
    assert result.audited == 0
