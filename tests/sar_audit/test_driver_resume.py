"""Driver resume semantics: skip / force / redo-instance / model-drift."""

from __future__ import annotations

from pathlib import Path

import pytest

from bird_interact_agents.sar_audit import driver


def _build_tasks(db_path: Path):
    """Two trivial tasks pointing at a sqlite DB with table t."""
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


def _make_clean_result(model_actual="claude-opus-4-7-20260121"):
    from tests.sar_audit._stubs import StubSARRunResult, StubSARVerdict

    return StubSARRunResult(
        verdict=StubSARVerdict(correctness_flag=True, ambiguity_flag=False),
        step_count=2,
        cost_usd=0.001,
        audit_model_actual=model_actual,
    )


def _run(
    *,
    db,
    db_path,
    tasks,
    output_dir,
    factory,
    force=False,
    redo_instance=None,
):
    return driver.run_db(
        db=db,
        tasks=tasks,
        db_path=db_path,
        full_kb=[],
        full_column_meanings={},
        audit_model="claude-opus-4-7",
        max_steps=5,
        output_dir=output_dir,
        sar_agent_factory=factory,
        force=force,
        redo_instance=redo_instance,
    )


def test_skip_existing_with_matching_model(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, write_jsonl, read_jsonl
):
    factory, handle = stub_sar_agent
    tasks = _build_tasks(fake_db)
    output_dir = tmp_path / "out"
    audited = output_dir / "fake_sar_audited.jsonl"

    # Pre-populate fake_1 with a matching model.
    write_jsonl(
        audited,
        [
            {
                "instance_id": "fake_1",
                "selected_database": "fake",
                "audit_status": "clean",
                "original_sol_sql": tasks[0]["sol_sql"],
                "audited_sol_sql": tasks[0]["sol_sql"],
                "audited_sample_row": [1],
                "audited_sample_row_status": "ok",
                "audited_sample_row_error": None,
                "changes": [],
                "reasoning_summary": "pre-existing",
                "skill_version": "sar-agent/1.0",
                "audited_at": "2025-01-01T00:00:00+00:00",
                "sar_correctness_flag": True,
                "sar_ambiguity_flag": False,
                "revised_question": None,
                "step_count": 1,
                "cost_usd": 0.0,
                "audit_model_requested": "claude-opus-4-7",
                "audit_model_actual": "claude-opus-4-7-20260121",
                "raw_trajectory": None,
            }
        ],
    )

    # Queue only one verdict for fake_2.
    handle.queue(_make_clean_result())

    result = _run(db="fake", db_path=fake_db, tasks=tasks, output_dir=output_dir, factory=factory)

    # Only fake_2 went through SAR.
    constructed = handle.constructed_with
    assert len(constructed) == 1, f"expected 1 SAR construction, got {len(constructed)}"

    rows = read_jsonl(audited)
    assert len(rows) == 2
    assert {r["instance_id"] for r in rows} == {"fake_1", "fake_2"}
    # Pre-existing row's audited_at is preserved (not overwritten).
    fake_1 = next(r for r in rows if r["instance_id"] == "fake_1")
    assert fake_1["audited_at"] == "2025-01-01T00:00:00+00:00"
    assert fake_1["reasoning_summary"] == "pre-existing"

    assert result.audited == 1
    assert result.skipped == 1
    assert result.failed == 0


def test_redo_when_model_drifts(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, write_jsonl
):
    factory, handle = stub_sar_agent
    tasks = _build_tasks(fake_db)
    output_dir = tmp_path / "out"
    audited = output_dir / "fake_sar_audited.jsonl"

    # Pre-existing row uses a different model than the current run.
    write_jsonl(
        audited,
        [
            {
                "instance_id": "fake_1",
                "selected_database": "fake",
                "audit_status": "clean",
                "original_sol_sql": tasks[0]["sol_sql"],
                "audited_sol_sql": tasks[0]["sol_sql"],
                "audited_sample_row": [1],
                "audited_sample_row_status": "ok",
                "audited_sample_row_error": None,
                "changes": [],
                "reasoning_summary": "from another model",
                "skill_version": "sar-agent/1.0",
                "audited_at": "2025-01-01T00:00:00+00:00",
                "sar_correctness_flag": True,
                "sar_ambiguity_flag": False,
                "revised_question": None,
                "step_count": 1,
                "cost_usd": 0.0,
                "audit_model_requested": "claude-sonnet-4-6",  # DIFFERENT
                "audit_model_actual": "claude-sonnet-4-6-20251001",
                "raw_trajectory": None,
            }
        ],
    )

    # Two verdicts: one each for fake_1 (redo) and fake_2 (new).
    handle.queue(_make_clean_result())
    handle.queue(_make_clean_result())

    _run(db="fake", db_path=fake_db, tasks=tasks, output_dir=output_dir, factory=factory)

    assert len(handle.constructed_with) == 2


def test_redo_when_skill_version_drifts(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, write_jsonl
):
    """Pre-existing row stamped with an older skill_version must be redone
    even when the audit_model matches."""
    factory, handle = stub_sar_agent
    tasks = _build_tasks(fake_db)
    output_dir = tmp_path / "out"
    audited = output_dir / "fake_sar_audited.jsonl"

    write_jsonl(
        audited,
        [
            {
                "instance_id": "fake_1",
                "selected_database": "fake",
                "audit_status": "clean",
                "original_sol_sql": tasks[0]["sol_sql"],
                "audited_sol_sql": tasks[0]["sol_sql"],
                "audited_sample_row": [1],
                "audited_sample_row_status": "ok",
                "audited_sample_row_error": None,
                "changes": [],
                "reasoning_summary": "older skill",
                "skill_version": "sar-agent/0.9",  # OLDER
                "audited_at": "2025-01-01T00:00:00+00:00",
                "sar_correctness_flag": True,
                "sar_ambiguity_flag": False,
                "revised_question": None,
                "step_count": 1,
                "cost_usd": 0.0,
                "audit_model_requested": "claude-opus-4-7",
                "audit_model_actual": "claude-opus-4-7-20260121",
                "raw_trajectory": None,
            }
        ],
    )

    handle.queue(_make_clean_result())
    handle.queue(_make_clean_result())

    _run(db="fake", db_path=fake_db, tasks=tasks, output_dir=output_dir, factory=factory)
    # Both tasks went through SAR: fake_1 because of skill_version drift, fake_2 because absent.
    assert len(handle.constructed_with) == 2


def test_force_redoes_everything(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, write_jsonl
):
    factory, handle = stub_sar_agent
    tasks = _build_tasks(fake_db)
    output_dir = tmp_path / "out"
    audited = output_dir / "fake_sar_audited.jsonl"

    write_jsonl(
        audited,
        [
            {
                "instance_id": "fake_1",
                "selected_database": "fake",
                "audit_status": "clean",
                "original_sol_sql": tasks[0]["sol_sql"],
                "audited_sol_sql": tasks[0]["sol_sql"],
                "audited_sample_row": [1],
                "audited_sample_row_status": "ok",
                "audited_sample_row_error": None,
                "changes": [],
                "reasoning_summary": "pre-existing",
                "skill_version": "sar-agent/1.0",
                "audited_at": "2025-01-01T00:00:00+00:00",
                "sar_correctness_flag": True,
                "sar_ambiguity_flag": False,
                "revised_question": None,
                "step_count": 1,
                "cost_usd": 0.0,
                "audit_model_requested": "claude-opus-4-7",
                "audit_model_actual": "claude-opus-4-7-20260121",
                "raw_trajectory": None,
            }
        ],
    )

    handle.queue(_make_clean_result())
    handle.queue(_make_clean_result())

    _run(
        db="fake",
        db_path=fake_db,
        tasks=tasks,
        output_dir=output_dir,
        factory=factory,
        force=True,
    )

    assert len(handle.constructed_with) == 2


def test_redo_instance(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, write_jsonl
):
    factory, handle = stub_sar_agent
    tasks = _build_tasks(fake_db)
    output_dir = tmp_path / "out"
    audited = output_dir / "fake_sar_audited.jsonl"

    write_jsonl(
        audited,
        [
            {
                "instance_id": "fake_1",
                "selected_database": "fake",
                "audit_status": "clean",
                "original_sol_sql": tasks[0]["sol_sql"],
                "audited_sol_sql": tasks[0]["sol_sql"],
                "audited_sample_row": [1],
                "audited_sample_row_status": "ok",
                "audited_sample_row_error": None,
                "changes": [],
                "reasoning_summary": "pre-existing",
                "skill_version": "sar-agent/1.0",
                "audited_at": "2025-01-01T00:00:00+00:00",
                "sar_correctness_flag": True,
                "sar_ambiguity_flag": False,
                "revised_question": None,
                "step_count": 1,
                "cost_usd": 0.0,
                "audit_model_requested": "claude-opus-4-7",
                "audit_model_actual": "claude-opus-4-7-20260121",
                "raw_trajectory": None,
            },
            {
                "instance_id": "fake_2",
                "selected_database": "fake",
                "audit_status": "clean",
                "original_sol_sql": tasks[1]["sol_sql"],
                "audited_sol_sql": tasks[1]["sol_sql"],
                "audited_sample_row": [3],
                "audited_sample_row_status": "ok",
                "audited_sample_row_error": None,
                "changes": [],
                "reasoning_summary": "pre-existing",
                "skill_version": "sar-agent/1.0",
                "audited_at": "2025-01-01T00:00:00+00:00",
                "sar_correctness_flag": True,
                "sar_ambiguity_flag": False,
                "revised_question": None,
                "step_count": 1,
                "cost_usd": 0.0,
                "audit_model_requested": "claude-opus-4-7",
                "audit_model_actual": "claude-opus-4-7-20260121",
                "raw_trajectory": None,
            },
        ],
    )

    handle.queue(_make_clean_result())

    _run(
        db="fake",
        db_path=fake_db,
        tasks=tasks,
        output_dir=output_dir,
        factory=factory,
        redo_instance="fake_1",
    )

    # Only fake_1 went through SAR.
    assert len(handle.constructed_with) == 1


def test_invalid_existing_row_treated_as_absent(
    tmp_path, fake_db, stub_sar_agent, stub_upstream, fixed_now, write_jsonl
):
    factory, handle = stub_sar_agent
    tasks = _build_tasks(fake_db)
    output_dir = tmp_path / "out"
    audited = output_dir / "fake_sar_audited.jsonl"

    # Row missing audit_status — must be redone.
    write_jsonl(
        audited,
        [{"instance_id": "fake_1", "selected_database": "fake"}],  # invalid
    )

    handle.queue(_make_clean_result())
    handle.queue(_make_clean_result())

    _run(db="fake", db_path=fake_db, tasks=tasks, output_dir=output_dir, factory=factory)

    # Both tasks went through SAR (fake_1 because its row was invalid, fake_2 because absent).
    assert len(handle.constructed_with) == 2
