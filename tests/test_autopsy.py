"""Tests for DEV-1521: autopsy agent.

Covers:
1. Schema: AutopsyAnalysis, AutopsyResult, SubmissionAnnotation.autopsy field
2. Hard precondition: run_task fails fast when TaskAnnotation absent on disk
3. run_autopsy: overflow → None, error → None, valid → AutopsyResult, no-diagnostics OK
4. grade_and_write: embeds autopsy when provided, leaves null otherwise
5. _read_kb_text: returns "" for missing dir, parses real memories.yaml
6. Trigger helper: genuine miss only
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_task_annotation(instance_id: str = "test_1", db: str = "testdb"):
    from bird_interact_agents.eval.annotation_schema import (
        MetadataSufficiency,
        Provenance,
        TaskAnnotation,
    )
    return TaskAnnotation(
        instance_id=instance_id,
        selected_database=db,
        annotated_by="test",
        annotated_at="2026-01-01",
        amb_user_query="How many rows?",
        metadata_sufficiency=MetadataSufficiency(
            verdict="sufficient", rationale="test"
        ),
        provenance=Provenance(
            task_jsonl_path="mini_interact.jsonl",
            task_jsonl_instance_id=instance_id,
        ),
    )


def _build_sqlite(tmp_path: Path, db_name: str = "x") -> Path:
    db_dir = tmp_path / db_name
    db_dir.mkdir(parents=True, exist_ok=True)
    db = db_dir / f"{db_name}.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE t (id INTEGER)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit()
    con.close()
    return db


def _fake_cascade(*, n9: bool):
    """Build a monotone CascadeVerdict with all tiers set to n9."""
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
    return CascadeVerdict(
        n1_original_gold=n9,
        n2_audited_primary=n9,
        n3_any_audited_variant=n9,
        n4_tie_order=n9,
        n5_llm_judge=n9,
        n6_numeric_epsilon=n9,
        n7_trailing_whitespace=n9,
        n8_column_order=n9,
        n9_case_fold=n9,
        variant_matches=[],
        rowset_relations=[],
        matched_variant_id=None,
        novel_reading_judgment=None,
        miss_diagnostics=None,
    )


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------

_ALL_AUTOPSY_PATTERNS = [
    "never_asked_key_question",
    "asked_but_ignored_answer",
    "user_sim_misleading",
    "late_mutation_corrupted_result",
    "wrong_join_path",
    "output_schema_misread",
    "slayer_generation_artifact",
    "exhausted_budget_guessing",
    "other",
]


@pytest.mark.parametrize("pattern", _ALL_AUTOPSY_PATTERNS)
def test_autopsy_analysis_accepts_all_patterns(pattern):
    from bird_interact_agents.eval.annotation_schema import AutopsyAnalysis

    a = AutopsyAnalysis(pattern=pattern, narrative="n", remediation="r")
    assert a.pattern == pattern


def test_autopsy_analysis_rejects_invalid_pattern():
    from pydantic import ValidationError

    from bird_interact_agents.eval.annotation_schema import AutopsyAnalysis

    with pytest.raises(ValidationError):
        AutopsyAnalysis(pattern="made_up_pattern", narrative="n", remediation="r")


def test_autopsy_analysis_schema_round_trip():
    from bird_interact_agents.eval.annotation_schema import AutopsyAnalysis

    a = AutopsyAnalysis(
        pattern="never_asked_key_question",
        narrative="The agent missed the key question.",
        remediation="Improve prompt to enforce ask_user.",
    )
    data = json.loads(a.model_dump_json())
    assert data["pattern"] == "never_asked_key_question"
    assert data["other_details"] is None
    assert AutopsyAnalysis.model_validate(data) == a


def test_autopsy_analysis_other_details_optional():
    from bird_interact_agents.eval.annotation_schema import AutopsyAnalysis

    a = AutopsyAnalysis(
        pattern="other",
        other_details="Novel failure mode X.",
        narrative="Explanation.",
        remediation="Fix it.",
    )
    assert a.other_details == "Novel failure mode X."
    # also works without other_details
    AutopsyAnalysis(
        pattern="exhausted_budget_guessing",
        narrative="Budget exhausted.",
        remediation="Increase budget.",
    )


def test_autopsy_result_lives_in_schema():
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyAnalysis,
        AutopsyResult,
        TrajectoryDecisionPoint,
        UserSimInteraction,
    )
    r = AutopsyResult(
        analysis=AutopsyAnalysis(
            pattern="wrong_join_path",
            narrative="Used wrong join.",
            remediation="Fix host discovery.",
        ),
        decision_point=TrajectoryDecisionPoint(
            trajectory_item_index=3,
            description="Wrong join at step 3.",
        ),
        user_sim_interaction=UserSimInteraction(n_asks=1),
    )
    assert r.analysis.pattern == "wrong_join_path"
    assert r.decision_point is not None
    assert r.decision_point.trajectory_item_index == 3


def test_submission_annotation_autopsy_field_none():
    from bird_interact_agents.eval.annotation_schema import (
        FailureClassification,
        SubmissionAnnotation,
        SubmissionEvaluation,
        SubmissionMetadata,
    )
    ann = SubmissionAnnotation(
        instance_id="x_1",
        selected_database="x",
        task_annotation_ref="ref",
        annotated_by="test",
        annotated_at="2026-01-01",
        submission=SubmissionMetadata(cloud_run_id="r", trajectory_path="t"),
        evaluation=SubmissionEvaluation(
            phase1_against_original_gold="fail",
            phase1_against_audited_primary="fail",
            phase1_against_any_audited_variant="fail",
            verdict="invalid",
        ),
        failure_classification=FailureClassification(
            primary="agent_miss",
            agent_at_fault=True,
            remediation_target="agent",
        ),
        autopsy=None,
    )
    assert ann.autopsy is None
    data = json.loads(ann.model_dump_json())
    assert data["autopsy"] is None


def test_submission_annotation_autopsy_field_filled():
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyAnalysis,
        FailureClassification,
        SubmissionAnnotation,
        SubmissionEvaluation,
        SubmissionMetadata,
    )
    a = AutopsyAnalysis(
        pattern="slayer_generation_artifact",
        narrative="Integer division in SLayer SQL.",
        remediation="Cast to REAL.",
    )
    ann = SubmissionAnnotation(
        instance_id="x_1",
        selected_database="x",
        task_annotation_ref="ref",
        annotated_by="test",
        annotated_at="2026-01-01",
        submission=SubmissionMetadata(cloud_run_id="r", trajectory_path="t"),
        evaluation=SubmissionEvaluation(
            phase1_against_original_gold="fail",
            phase1_against_audited_primary="fail",
            phase1_against_any_audited_variant="fail",
            verdict="invalid",
        ),
        failure_classification=FailureClassification(
            primary="agent_miss",
            agent_at_fault=True,
            remediation_target="agent",
        ),
        autopsy=a,
    )
    assert ann.autopsy == a
    data = json.loads(ann.model_dump_json())
    assert data["autopsy"]["pattern"] == "slayer_generation_artifact"


# ---------------------------------------------------------------------------
# 2. Hard precondition: run_task fails fast
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_ainteract_fails_fast_no_annotation(tmp_path, monkeypatch):
    """claude_sdk_otf_ainteract.run_task returns skip row when no annotation on disk."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    monkeypatch.setenv("BIRD_ANNOTATIONS_ROOT", str(tmp_path / "annotations"))

    agent = ClaudeSDKOtfAInteractAgent(model="anthropic/claude-sonnet-4-5")
    result = await agent.run_task(
        task_data={
            "instance_id": "hh_1",
            "selected_database": "households",
            "amb_user_query": "q",
            "sol_sql": ["SELECT 1"],
            "dataset": "mini_interact",
        },
        data_path_base=str(tmp_path),
        budget=10.0,
        query_mode="slayer",
        eval_mode="a-interact",
    )
    assert result.get("phase1_passed") is False
    assert result.get("total_reward", 1.0) == 0.0
    assert result.get("trajectory", None) is not None  # present in row
    err = result.get("error") or ""
    assert "no TaskAnnotation" in err
    assert "hh_1" in err


@pytest.mark.asyncio
async def test_run_task_otf_fails_fast_no_annotation(tmp_path, monkeypatch):
    """claude_sdk_otf.run_task returns skip row when no annotation on disk."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    monkeypatch.setenv("BIRD_ANNOTATIONS_ROOT", str(tmp_path / "annotations"))

    agent = ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    result = await agent.run_task(
        task_data={
            "instance_id": "lsb_1",
            "selected_database": "mydb",
            "amb_user_query": "q",
            "sol_sql": ["SELECT 1"],
            "dataset": "livesqlbench",
        },
        data_path_base=str(tmp_path),
        budget=10.0,
        query_mode="slayer",
        eval_mode="one-shot",
    )
    assert result.get("phase1_passed") is False
    assert result.get("total_reward", 1.0) == 0.0
    err = result.get("error") or ""
    assert "no TaskAnnotation" in err
    assert "lsb_1" in err


# ---------------------------------------------------------------------------
# 3. run_autopsy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_autopsy_returns_none_on_overflow(tmp_path):
    """anthropic.BadRequestError with context-window language → None."""
    import anthropic

    task_ann = _minimal_task_annotation()

    mock_client = AsyncMock()
    mock_client.messages.create.side_effect = anthropic.BadRequestError(
        message="prompt is too long: 220000 tokens exceeds context window",
        response=MagicMock(status_code=400),
        body={},
    )

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        from bird_interact_agents.eval.autopsy import run_autopsy
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[{"type": "T", "data": "x"}],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
        )
    assert result is None


@pytest.mark.asyncio
async def test_run_autopsy_returns_none_on_other_error(tmp_path):
    """Arbitrary exception from the LLM call → None."""
    task_ann = _minimal_task_annotation()

    mock_client = AsyncMock()
    mock_client.messages.create.side_effect = RuntimeError("network timeout")

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        from bird_interact_agents.eval.autopsy import run_autopsy
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
        )
    assert result is None


@pytest.mark.asyncio
async def test_run_autopsy_maps_output_to_result(tmp_path):
    """Valid mock tool-use response → correctly mapped AutopsyResult."""
    from bird_interact_agents.eval.annotation_schema import AutopsyResult
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()

    tool_input = {
        "pattern": "never_asked_key_question",
        "other_details": None,
        "narrative": "The agent failed to ask the key question.",
        "remediation": "Improve the prompt to enforce ask_user.",
        "decision_point_trajectory_index": 5,
        "decision_point_description": "Agent skipped clarification at step 5.",
        "n_asks": 1,
        "key_asks": [{"trajectory_idx": 3, "summary": "Asked about thresholds"}],
        "disclosed_resolutions": ["threshold is > 3"],
        "undisclosed_resolutions": ["bracket definition"],
    }
    mock_tool = MagicMock()
    mock_tool.type = "tool_use"
    mock_tool.name = "autopsy_output"
    mock_tool.input = tool_input

    mock_response = MagicMock()
    mock_response.content = [mock_tool]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[{"type": "T", "data": str(i)} for i in range(6)],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
        )

    assert isinstance(result, AutopsyResult)
    assert result.analysis.pattern == "never_asked_key_question"
    assert result.analysis.narrative == "The agent failed to ask the key question."
    assert result.analysis.other_details is None
    assert result.decision_point is not None
    assert result.decision_point.trajectory_item_index == 5
    assert result.decision_point.description == "Agent skipped clarification at step 5."
    assert result.user_sim_interaction.n_asks == 1
    assert len(result.user_sim_interaction.key_responses) == 1
    assert result.user_sim_interaction.key_responses[0].trajectory_idx == 3
    assert result.user_sim_interaction.key_responses[0].summary == "Asked about thresholds"
    assert result.user_sim_interaction.disclosed_resolutions == ["threshold is > 3"]
    assert result.user_sim_interaction.undisclosed_resolutions == ["bracket definition"]


@pytest.mark.asyncio
async def test_autopsy_no_miss_diagnostics(tmp_path):
    """run_autopsy handles miss_diagnostics=None without crashing."""
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()

    tool_input = {
        "pattern": "other",
        "other_details": "Novel failure mode.",
        "narrative": "Something went wrong.",
        "remediation": "Investigate.",
        "decision_point_trajectory_index": None,
        "decision_point_description": None,
        "n_asks": 0,
        "key_asks": [],
        "disclosed_resolutions": [],
        "undisclosed_resolutions": [],
    }
    mock_tool = MagicMock()
    mock_tool.type = "tool_use"
    mock_tool.name = "autopsy_output"
    mock_tool.input = tool_input

    mock_response = MagicMock()
    mock_response.content = [mock_tool]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,  # key: no diagnostics
            model="anthropic/claude-sonnet-4-5",
        )
    assert result is not None
    assert result.analysis.pattern == "other"
    assert result.analysis.other_details == "Novel failure mode."
    assert result.decision_point is None  # no trajectory index given
    assert result.user_sim_interaction.n_asks == 0


# ---------------------------------------------------------------------------
# 4. grade_and_write embeds / omits autopsy
# ---------------------------------------------------------------------------

def test_grade_and_write_embeds_autopsy(tmp_path):
    """When autopsy_result is provided, annotation JSON carries autopsy,
    decision_point, and user_sim_interaction."""
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyAnalysis,
        AutopsyResult,
        TrajectoryDecisionPoint,
        UserSimInteraction,
    )
    from bird_interact_agents.eval.grade_in_place import grade_and_write

    db = _build_sqlite(tmp_path)
    task_ann = _minimal_task_annotation(instance_id="x_1", db="x")

    autopsy_result = AutopsyResult(
        analysis=AutopsyAnalysis(
            pattern="output_schema_misread",
            narrative="The agent projected the wrong column.",
            remediation="Fix the output schema instructions.",
        ),
        decision_point=TrajectoryDecisionPoint(
            trajectory_item_index=7,
            description="Agent encoded wrong column at step 7.",
        ),
        user_sim_interaction=UserSimInteraction(
            n_asks=2,
            disclosed_resolutions=["threshold = 5"],
            undisclosed_resolutions=[],
        ),
    )

    ann_path = grade_and_write(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        benchmark="mini_interact",
        run_id="test-run",
        task_annotation=task_ann,
        audited_gold_rows=[],
        original_sol_sql=["SELECT id FROM t"],
        submitted_sql="SELECT * FROM t",  # mismatch → miss
        db_path=db,
        trajectory_path="rows/x_1/attempt-1.json",
        autopsy_result=autopsy_result,
    )

    data = json.loads(ann_path.read_text())
    assert data["autopsy"]["pattern"] == "output_schema_misread"
    assert data["decision_point"]["trajectory_item_index"] == 7
    assert data["decision_point"]["description"] == "Agent encoded wrong column at step 7."
    assert data["user_sim_interaction"]["n_asks"] == 2
    assert data["user_sim_interaction"]["disclosed_resolutions"] == ["threshold = 5"]


def test_grade_and_write_no_autopsy(tmp_path):
    """autopsy_result=None → autopsy=null in annotation; existing behaviour unchanged."""
    from bird_interact_agents.eval.grade_in_place import grade_and_write

    db = _build_sqlite(tmp_path)
    task_ann = _minimal_task_annotation(instance_id="x_1", db="x")

    ann_path = grade_and_write(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        benchmark="mini_interact",
        run_id="test-run",
        task_annotation=task_ann,
        audited_gold_rows=[],
        original_sol_sql=["SELECT COUNT(*) FROM t"],
        submitted_sql="SELECT COUNT(*) FROM t",
        db_path=db,
        trajectory_path="rows/x_1/attempt-1.json",
        autopsy_result=None,
    )

    data = json.loads(ann_path.read_text())
    assert data["autopsy"] is None
    # decision_point and user_sim_interaction stay at defaults
    assert data["decision_point"] is None
    assert data["user_sim_interaction"]["n_asks"] == 0


# ---------------------------------------------------------------------------
# 5. _read_kb_text
# ---------------------------------------------------------------------------

def test_read_kb_text_returns_empty_for_missing_dir():
    from bird_interact_agents.eval.autopsy import _read_kb_text
    assert _read_kb_text("/nonexistent/path/that/does/not/exist", "testdb") == ""


def test_read_kb_text_parses_memories_yaml(tmp_path):
    """_read_kb_text extracts learning text from matching KB entries only."""
    from bird_interact_agents.eval.autopsy import _read_kb_text

    memories = [
        {
            "version": 1,
            "id": "mydb_kb_1",
            "learning": "KB 1 — Household Tenure\nOwned or rented.",
            "entities": [],
            "query": None,
            "created_at": "1970-01-01T00:00:00+00:00",
        },
        {
            "version": 1,
            "id": "mydb_kb_2",
            "learning": "KB 2 — Income Classification\nLow to high.",
            "entities": [],
            "query": None,
            "created_at": "1970-01-01T00:00:00+00:00",
        },
        {
            "version": 1,
            "id": "otherdb_kb_1",
            "learning": "Should be excluded from mydb output.",
            "entities": [],
            "query": None,
            "created_at": "1970-01-01T00:00:00+00:00",
        },
    ]
    (tmp_path / "memories.yaml").write_text(yaml.dump(memories))

    text = _read_kb_text(str(tmp_path), "mydb")
    assert "KB 1" in text
    assert "KB 2" in text
    assert "Should be excluded from mydb output" not in text
    # Both KB items appear as separate paragraphs
    assert text.count("KB 1") == 1
    assert text.count("KB 2") == 1


def test_read_kb_text_returns_empty_for_missing_yaml(tmp_path):
    """No memories.yaml at all → empty string."""
    from bird_interact_agents.eval.autopsy import _read_kb_text
    assert _read_kb_text(str(tmp_path), "mydb") == ""


# ---------------------------------------------------------------------------
# 6. Trigger: _is_genuine_miss
# ---------------------------------------------------------------------------

def test_is_genuine_miss_true_when_all_tiers_fail():
    from bird_interact_agents.eval.autopsy import _is_genuine_miss
    assert _is_genuine_miss(_fake_cascade(n9=False)) is True


def test_is_genuine_miss_false_when_n9_passes():
    from bird_interact_agents.eval.autopsy import _is_genuine_miss
    assert _is_genuine_miss(_fake_cascade(n9=True)) is False


def test_is_genuine_miss_false_when_n3_passes_but_n9_true():
    """Monotone cascade: N3 pass means N4..N9 also pass."""
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
    from bird_interact_agents.eval.autopsy import _is_genuine_miss

    # N1/N2 fail, N3..N9 pass (monotone)
    cascade = CascadeVerdict(
        n1_original_gold=False,
        n2_audited_primary=False,
        n3_any_audited_variant=True,
        n4_tie_order=True,
        n5_llm_judge=True,
        n6_numeric_epsilon=True,
        n7_trailing_whitespace=True,
        n8_column_order=True,
        n9_case_fold=True,
        variant_matches=[],
        rowset_relations=[],
        matched_variant_id=None,
        novel_reading_judgment=None,
        miss_diagnostics=None,
    )
    assert _is_genuine_miss(cascade) is False


def test_is_genuine_miss_false_when_only_n9_passes():
    """Defensive: N1..N8 fail but N9 passes → not a genuine miss.
    Non-monotone input (shouldn't occur after grade_submission enforces
    monotonicity) but _is_genuine_miss must still return False since n9=True."""
    from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
    from bird_interact_agents.eval.autopsy import _is_genuine_miss

    cascade = CascadeVerdict(
        n1_original_gold=False,
        n2_audited_primary=False,
        n3_any_audited_variant=False,
        n4_tie_order=False,
        n5_llm_judge=False,
        n6_numeric_epsilon=False,
        n7_trailing_whitespace=False,
        n8_column_order=False,
        n9_case_fold=True,  # only N9 passes
        variant_matches=[],
        rowset_relations=[],
        matched_variant_id=None,
        novel_reading_judgment=None,
        miss_diagnostics=None,
    )
    assert _is_genuine_miss(cascade) is False
