"""Tests for DEV-1521 + DEV-1541: autopsy agent.

DEV-1521 (original):
1. Schema: AutopsyAnalysis, AutopsyResult, SubmissionAnnotation.autopsy field
2. Hard precondition: run_task fails fast when TaskAnnotation absent on disk
3. run_autopsy: overflow / error / valid / no-diagnostics paths
4. grade_and_write: embeds autopsy when provided, leaves null otherwise
5. _read_kb_text: returns "" for missing dir, parses real memories.yaml
6. Trigger helper: genuine miss only

DEV-1541 extends:
- One-shot vs a-interact split: AutopsyAnalysisOneShot, AutopsyLLMOutputOneShot,
  _AUTOPSY_TOOL_SCHEMA_ONE_SHOT, branched _build_prompt / _map_output.
- AutopsyError type capturing validation_error / context_overflow /
  api_error / network_error / missing_tool_use / unknown.
- AutopsyResult.analysis is Optional + new AutopsyResult.error field;
  exactly-one invariant.
- run_autopsy never returns None on failure; returns
  AutopsyResult(error=AutopsyError(...)).
- one-shot benchmarks default user_sim_interaction to None (not zero-asks).
- grade_and_write only overwrites top-level decision_point / user_sim
  when autopsy_result.analysis is not None (don't clobber on error path).
- Legacy on-disk autopsy rows without analysis.kind still parse.
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
            verdict="agent_miss",
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
        AutopsyResult,
        FailureClassification,
        SubmissionAnnotation,
        SubmissionEvaluation,
        SubmissionMetadata,
        UserSimInteraction,
    )
    analysis = AutopsyAnalysis(
        pattern="slayer_generation_artifact",
        narrative="Integer division in SLayer SQL.",
        remediation="Cast to REAL.",
    )
    result = AutopsyResult(
        analysis=analysis,
        user_sim_interaction=UserSimInteraction(n_asks=0),
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
            verdict="agent_miss",
        ),
        failure_classification=FailureClassification(
            primary="agent_miss",
            agent_at_fault=True,
            remediation_target="agent",
        ),
        autopsy=result,
    )
    assert ann.autopsy == result
    data = json.loads(ann.model_dump_json())
    assert data["autopsy"]["analysis"]["pattern"] == "slayer_generation_artifact"


# ---------------------------------------------------------------------------
# 2. Annotation precondition: missing annotation uses implicit, skips autopsy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_ainteract_no_annotation_uses_implicit(tmp_path, monkeypatch):
    """claude_sdk_otf_ainteract.run_task does NOT fail fast when no annotation on disk.
    Instead it loads an implicit annotation and only skips autopsy."""
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
            "dataset": "mini-interact",
        },
        data_path_base=str(tmp_path),
        budget=10.0,
        query_mode="slayer",
        eval_mode="a-interact",
    )
    # Agent proceeds (not stopped by missing annotation) — it hits some
    # other failure (e.g. missing DB file) or completes; either way the
    # error is NOT the old "no TaskAnnotation" abort.
    err = result.get("error") or ""
    assert "no TaskAnnotation" not in err


@pytest.mark.asyncio
async def test_run_task_otf_no_annotation_uses_implicit(tmp_path, monkeypatch):
    """claude_sdk_otf.run_task does NOT fail fast when no annotation on disk.
    Instead it loads an implicit annotation and only skips autopsy."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    monkeypatch.setenv("BIRD_ANNOTATIONS_ROOT", str(tmp_path / "annotations"))

    agent = ClaudeSDKOtfAgent(model="anthropic/claude-sonnet-4-5")
    result = await agent.run_task(
        task_data={
            "instance_id": "lsb_1",
            "selected_database": "mydb",
            "amb_user_query": "q",
            "sol_sql": ["SELECT 1"],
            "dataset": "livesqlbench-base-lite-sqlite",
        },
        data_path_base=str(tmp_path),
        budget=10.0,
        query_mode="slayer",
        eval_mode="one-shot",
    )
    # Agent proceeds (not stopped by missing annotation) — it hits some
    # other failure (e.g. missing DB file) or completes; either way the
    # error is NOT the old "no TaskAnnotation" abort.
    err = result.get("error") or ""
    assert "no TaskAnnotation" not in err


# ---------------------------------------------------------------------------
# 3. run_autopsy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_autopsy_context_overflow_returns_autopsy_error(tmp_path):
    """DEV-1541: anthropic.BadRequestError with context-window language →
    AutopsyResult(error=AutopsyError(kind="context_overflow", ...)),
    NOT None — silent failures are the bug we are fixing."""
    import anthropic
    from bird_interact_agents.eval.annotation_schema import AutopsyResult

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
            is_one_shot=True,
        )
    assert isinstance(result, AutopsyResult)
    assert result.analysis is None
    assert result.error is not None
    assert result.error.kind == "context_overflow"
    # Codex r2 #6: assert full error metadata is captured at the
    # error-path level (not just the helper-level _truncate test).
    # FQN follows f'{module}.{qualname}' — must include the module
    # prefix, not bare class name.
    assert "." in result.error.exception_class
    assert "BadRequestError" in result.error.exception_class
    # Quick-look stats — kb_chars present even when 0; prompt_chars > 0.
    assert result.error.trajectory_items == 1
    assert result.error.prompt_chars > 0
    assert result.error.kb_chars == 0
    # Model is recorded from the call site.
    assert result.error.model == "anthropic/claude-sonnet-4-5"
    # Timestamp is tz-aware UTC.
    assert result.error.timestamp.tzinfo is not None
    assert result.error.timestamp.utcoffset() == _import_timedelta_zero()


@pytest.mark.asyncio
async def test_run_autopsy_generic_runtime_error_returns_unknown(tmp_path):
    """DEV-1541: arbitrary Exception from the LLM call →
    AutopsyResult(error=AutopsyError(kind="unknown", ...))."""
    from bird_interact_agents.eval.annotation_schema import AutopsyResult

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
            is_one_shot=False,
        )
    assert isinstance(result, AutopsyResult)
    assert result.analysis is None
    assert result.error is not None
    assert result.error.kind == "unknown"
    assert result.error.exception_class == "builtins.RuntimeError"
    assert "network timeout" in result.error.message_excerpt
    # traceback is non-empty (we capture format_exc).
    assert len(result.error.traceback_excerpt) > 0


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
            is_one_shot=False,
        )

    assert isinstance(result, AutopsyResult)
    assert result.analysis is not None
    assert result.error is None
    assert result.analysis.kind == "a_interact"
    assert result.analysis.pattern == "never_asked_key_question"
    assert result.analysis.narrative == "The agent failed to ask the key question."
    assert result.analysis.other_details is None
    assert result.decision_point is not None
    assert result.decision_point.trajectory_item_index == 5
    assert result.decision_point.description == "Agent skipped clarification at step 5."
    assert result.user_sim_interaction is not None
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
            is_one_shot=False,
        )
    assert result is not None
    assert result.analysis is not None
    assert result.error is None
    assert result.analysis.pattern == "other"
    assert result.analysis.other_details == "Novel failure mode."
    assert result.decision_point is None  # no trajectory index given
    assert result.user_sim_interaction is not None
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
        benchmark="mini-interact",
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
    assert data["autopsy"]["analysis"]["pattern"] == "output_schema_misread"
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
        benchmark="mini-interact",
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
    assert _read_kb_text("/nonexistent/path/that/does/not/exist", "testdb", [1]) == ""


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

    text = _read_kb_text(str(tmp_path), "mydb", [1, 2])
    assert "KB 1" in text
    assert "KB 2" in text
    assert "Should be excluded from mydb output" not in text
    # Both KB items appear as separate paragraphs
    assert text.count("KB 1") == 1
    assert text.count("KB 2") == 1


def test_read_kb_text_returns_empty_for_missing_yaml(tmp_path):
    """No memories.yaml at all → empty string."""
    from bird_interact_agents.eval.autopsy import _read_kb_text
    assert _read_kb_text(str(tmp_path), "mydb", [1]) == ""


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


def test_compress_trajectory_strips_thinking():
    """_compress_trajectory_for_autopsy replaces thinking content with a size indicator."""
    from bird_interact_agents.eval.autopsy import _compress_trajectory_for_autopsy

    thinking_text = "x" * 5000
    trajectory = [
        {
            "type": "AssistantMessage",
            "data": {
                "content": [
                    {"thinking": thinking_text, "signature": "abc123"},
                    {"text": "I will now proceed."},
                ],
                "model": "claude-test",
            },
        }
    ]
    result = _compress_trajectory_for_autopsy(trajectory)
    assert len(result) == 1
    content = result[0]["data"]["content"]
    thinking_block = content[0]
    assert thinking_block["thinking"] == f"[thinking: {len(thinking_text)} chars]"
    assert thinking_block["signature"] == "abc123"
    assert content[1]["text"] == "I will now proceed."


def test_compress_trajectory_preserves_non_thinking_content():
    """_compress_trajectory_for_autopsy leaves text, tool use/result blocks unchanged."""
    from bird_interact_agents.eval.autopsy import _compress_trajectory_for_autopsy

    trajectory = [
        {
            "type": "AssistantMessage",
            "data": {
                "content": [
                    {"text": "Let me check the schema."},
                    {"id": "tu_1", "name": "slayer_query", "input": {"sql": "SELECT 1"}},
                ],
                "model": "claude-test",
            },
        },
        {
            "type": "UserMessage",
            "data": {
                "content": [
                    {"tool_use_id": "tu_1", "content": "[(1,)]", "is_error": False}
                ]
            },
        },
    ]
    result = _compress_trajectory_for_autopsy(trajectory)
    assert result[0]["data"]["content"][0]["text"] == "Let me check the schema."
    assert result[0]["data"]["content"][1]["input"]["sql"] == "SELECT 1"
    assert result[1]["data"]["content"][0]["tool_use_id"] == "tu_1"


def test_compress_trajectory_old_string_format_passthrough():
    """Items whose data is a str (legacy repr format) are returned unchanged."""
    from bird_interact_agents.eval.autopsy import _compress_trajectory_for_autopsy

    legacy_item = {
        "type": "AssistantMessage",
        "data": "AssistantMessage(content=[ThinkingBlock(thinking='big text', signature='sig')], model='...')",
    }
    result = _compress_trajectory_for_autopsy([legacy_item])
    assert len(result) == 1
    assert result[0]["type"] == legacy_item["type"]
    assert result[0]["data"] == legacy_item["data"]  # string unchanged


def test_build_prompt_includes_all_items_with_compression():
    """_build_prompt includes all trajectory items; compression strips thinking only."""
    from bird_interact_agents.eval.autopsy import _build_prompt

    ta = _minimal_task_annotation()
    n = 200
    trajectory = [
        {
            "type": "AssistantMessage",
            "data": {
                "content": [
                    {"thinking": "x" * 10000, "signature": f"sig{i}"},
                    {"text": f"step {i}"},
                ],
                "model": "claude-test",
            },
        }
        for i in range(n)
    ]
    prompt = _build_prompt(
        task_annotation=ta,
        trajectory=trajectory,
        kb_text="",
        miss_diagnostics=None,
        is_one_shot=False,
    )
    assert "omitted for length" not in prompt
    assert f'"step 0"' in prompt
    assert f'"step {n - 1}"' in prompt
    assert "[thinking: 10000 chars]" in prompt
    assert "x" * 10000 not in prompt


# ===========================================================================
# DEV-1541 — one-shot/a-interact split + AutopsyError + silent-fail backfill
# ===========================================================================

_ONE_SHOT_PATTERNS = [
    "late_mutation_corrupted_result",
    "wrong_join_path",
    "output_schema_misread",
    "slayer_generation_artifact",
    "slayer_overaggregation",
    "exhausted_budget_guessing",
    "other",
]
_ASK_USER_PATTERNS = [
    "never_asked_key_question",
    "asked_but_ignored_answer",
    "user_sim_misleading",
]


def _import_timedelta_zero():
    """Helper for the tz-aware UTC assertion."""
    import datetime as _dt
    return _dt.timedelta(0)


# ---------------------------------------------------------------------------
# DEV-1541 §A. Schema layer: AutopsyAnalysis split + AutopsyError + AutopsyResult
# ---------------------------------------------------------------------------

def test_autopsy_analysis_default_kind_a_interact():
    """AutopsyAnalysis (existing class) defaults kind='a_interact' for
    discriminated-union routing. Old code that builds AutopsyAnalysis without
    naming `kind` keeps working."""
    from bird_interact_agents.eval.annotation_schema import AutopsyAnalysis

    a = AutopsyAnalysis(
        pattern="wrong_join_path",
        narrative="n",
        remediation="r",
    )
    assert a.kind == "a_interact"


@pytest.mark.parametrize("pattern", _ONE_SHOT_PATTERNS)
def test_autopsy_analysis_one_shot_accepts_one_shot_patterns(pattern):
    """AutopsyAnalysisOneShot validates every one-shot pattern."""
    from bird_interact_agents.eval.annotation_schema import AutopsyAnalysisOneShot

    a = AutopsyAnalysisOneShot(pattern=pattern, narrative="n", remediation="r")
    assert a.pattern == pattern
    assert a.kind == "one_shot"


@pytest.mark.parametrize("pattern", _ASK_USER_PATTERNS)
def test_autopsy_analysis_one_shot_rejects_ask_user_patterns(pattern):
    """AutopsyAnalysisOneShot rejects the 3 ask_user-related patterns.
    Repro target: livesqlbench-base-lite-sqlite one-shot runs that
    were mis-tagged with `never_asked_key_question`."""
    from pydantic import ValidationError
    from bird_interact_agents.eval.annotation_schema import AutopsyAnalysisOneShot

    with pytest.raises(ValidationError):
        AutopsyAnalysisOneShot(pattern=pattern, narrative="n", remediation="r")


def test_autopsy_result_exactly_one_neither_raises():
    """AutopsyResult must have exactly one of analysis or error set."""
    from pydantic import ValidationError
    from bird_interact_agents.eval.annotation_schema import AutopsyResult

    with pytest.raises(ValidationError):
        AutopsyResult()


def test_autopsy_result_exactly_one_both_raises():
    """AutopsyResult: setting both analysis AND error simultaneously is a
    contract violation."""
    import datetime as _dt
    from pydantic import ValidationError
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyAnalysis,
        AutopsyError,
        AutopsyResult,
    )

    analysis = AutopsyAnalysis(pattern="other", narrative="n", remediation="r")
    err = AutopsyError(
        kind="unknown",
        exception_class="builtins.RuntimeError",
        message_excerpt="e",
        traceback_excerpt="t",
        prompt_chars=0,
        kb_chars=0,
        trajectory_items=0,
        model="anthropic/claude-test",
        timestamp=_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0),
    )
    with pytest.raises(ValidationError):
        AutopsyResult(analysis=analysis, error=err)


def test_autopsy_result_analysis_only_ok():
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyAnalysis,
        AutopsyResult,
    )
    r = AutopsyResult(
        analysis=AutopsyAnalysis(
            pattern="other", narrative="n", remediation="r"
        ),
    )
    assert r.analysis is not None
    assert r.error is None


def test_autopsy_result_error_only_ok():
    import datetime as _dt
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyError,
        AutopsyResult,
    )
    r = AutopsyResult(
        error=AutopsyError(
            kind="unknown",
            exception_class="builtins.RuntimeError",
            message_excerpt="e",
            traceback_excerpt="t",
            prompt_chars=0,
            kb_chars=0,
            trajectory_items=0,
            model="anthropic/claude-test",
            timestamp=_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0),
        ),
    )
    assert r.analysis is None
    assert r.error is not None


def test_autopsy_result_user_sim_now_optional():
    """user_sim_interaction is Optional and defaults to None when omitted
    on a fresh construction (vs. the pre-DEV-1541 default_factory behaviour)."""
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyAnalysis,
        AutopsyResult,
    )
    r = AutopsyResult(
        analysis=AutopsyAnalysis(pattern="other", narrative="n", remediation="r"),
    )
    assert r.user_sim_interaction is None


def test_autopsy_result_legacy_read_no_kind_through_submission_annotation():
    """Codex r1 #3 + #6: legacy on-disk autopsy rows omit `analysis.kind`.
    Validate the FULL SubmissionAnnotation round-trip (not just AutopsyResult
    in isolation) so the discriminated-union pre-validator fires at the
    correct level."""
    from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation

    legacy_json = {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": "x_1",
        "selected_database": "x",
        "task_annotation_ref": "ref",
        "annotated_by": "test",
        "annotated_at": "2026-01-01",
        "submission": {"cloud_run_id": "r", "trajectory_path": "t"},
        "evaluation": {
            "phase1_against_original_gold": "fail",
            "phase1_against_audited_primary": "fail",
            "phase1_against_any_audited_variant": "fail",
            "verdict": "agent_miss",
        },
        "failure_classification": {
            "primary": "agent_miss",
            "agent_at_fault": True,
            "remediation_target": "agent",
        },
        "autopsy": {
            "analysis": {
                "pattern": "wrong_join_path",
                "narrative": "n",
                "remediation": "r",
                # NOTE: no `kind` field — pre-DEV-1541 shape
            },
            "user_sim_interaction": {"n_asks": 0},
        },
    }
    ann = SubmissionAnnotation.model_validate(legacy_json)
    assert ann.autopsy is not None
    assert ann.autopsy.analysis is not None
    assert ann.autopsy.analysis.kind == "a_interact"
    assert ann.autopsy.analysis.pattern == "wrong_join_path"


# ---------------------------------------------------------------------------
# DEV-1541 §B. AutopsyError serialization details
# ---------------------------------------------------------------------------

def _make_autopsy_error(**overrides):
    import datetime as _dt
    from bird_interact_agents.eval.annotation_schema import AutopsyError
    defaults = dict(
        kind="validation_error",
        exception_class="pydantic_core._pydantic_core.ValidationError",
        message_excerpt="2 validation errors for AutopsyLLMOutput",
        traceback_excerpt="Traceback (most recent call last): ...",
        prompt_chars=1234,
        kb_chars=567,
        trajectory_items=8,
        model="anthropic/claude-sonnet-4-5",
        timestamp=_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0),
    )
    defaults.update(overrides)
    return AutopsyError(**defaults)


def test_autopsy_error_round_trip():
    from bird_interact_agents.eval.annotation_schema import AutopsyError

    err = _make_autopsy_error()
    data = json.loads(err.model_dump_json())
    assert data["kind"] == "validation_error"
    assert data["exception_class"] == "pydantic_core._pydantic_core.ValidationError"
    assert data["prompt_chars"] == 1234
    assert data["trajectory_items"] == 8
    # Round-trip through validate_json reconstructs an equal object.
    err2 = AutopsyError.model_validate(data)
    assert err2 == err


def test_autopsy_error_message_excerpt_truncation():
    """Long messages are hard-capped at 500 chars with ...[truncated] suffix."""
    from bird_interact_agents.eval.autopsy import _truncate

    long = "x" * 1000
    out = _truncate(long, 500)
    assert len(out) <= 500
    assert out.endswith("...[truncated]")
    # short strings are unchanged
    assert _truncate("hello", 500) == "hello"


def test_autopsy_error_traceback_excerpt_truncation():
    """Long traceback excerpts are hard-capped at 2000 chars."""
    from bird_interact_agents.eval.autopsy import _truncate

    long = "y" * 5000
    out = _truncate(long, 2000)
    assert len(out) <= 2000
    assert out.endswith("...[truncated]")


def test_autopsy_error_timestamp_is_utc_aware():
    """timestamp must round-trip with UTC tz info."""
    err = _make_autopsy_error()
    assert err.timestamp.tzinfo is not None
    # serialized form contains a UTC marker (Z or +00:00).
    data = json.loads(err.model_dump_json())
    assert data["timestamp"].endswith("Z") or data["timestamp"].endswith("+00:00")


def test_autopsy_error_fqn_format_from_helper():
    """FQN format is `f'{module}.{qualname}'`. We don't expose the helper
    directly but the run_autopsy paths use this format; verify via a
    hand-built error that ends up in a real SubmissionAnnotation."""
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyResult,
        FailureClassification,
        SubmissionAnnotation,
        SubmissionEvaluation,
        SubmissionMetadata,
    )

    err = _make_autopsy_error(exception_class="builtins.RuntimeError")
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
            verdict="agent_miss",
        ),
        failure_classification=FailureClassification(
            primary="agent_miss", agent_at_fault=True, remediation_target="agent",
        ),
        autopsy=AutopsyResult(error=err),
    )
    data = json.loads(ann.model_dump_json())
    assert data["autopsy"]["error"]["exception_class"] == "builtins.RuntimeError"
    # Full round-trip through model_validate_json works.
    ann2 = SubmissionAnnotation.model_validate_json(ann.model_dump_json())
    assert ann2.autopsy is not None
    assert ann2.autopsy.error is not None
    assert ann2.autopsy.error.exception_class == "builtins.RuntimeError"


# ---------------------------------------------------------------------------
# DEV-1541 §C. SubmissionAnnotation.user_sim_interaction default semantics
# ---------------------------------------------------------------------------

def test_submission_annotation_user_sim_missing_field_defaults_to_zero_asks():
    """Codex r1 #1: legacy a-interact annotations omitted user_sim_interaction
    when zero asks. The field default must remain a zero-asks UserSimInteraction
    (not None) when omitted from JSON, so legacy reads don't lose semantics."""
    from bird_interact_agents.eval.annotation_schema import (
        SubmissionAnnotation,
        UserSimInteraction,
    )

    legacy_json = {
        "schema_version": 1,
        "kind": "submission_annotation",
        "instance_id": "x_1",
        "selected_database": "x",
        "task_annotation_ref": "ref",
        "annotated_by": "test",
        "annotated_at": "2026-01-01",
        "submission": {"cloud_run_id": "r", "trajectory_path": "t"},
        "evaluation": {
            "phase1_against_original_gold": "pass",
            "phase1_against_audited_primary": "pass",
            "phase1_against_any_audited_variant": "pass",
            "verdict": "correct",
        },
        "failure_classification": {
            "primary": "no_fail",
            "agent_at_fault": False,
            "remediation_target": "other",
        },
        # NB: no `user_sim_interaction` field
    }
    ann = SubmissionAnnotation.model_validate(legacy_json)
    assert isinstance(ann.user_sim_interaction, UserSimInteraction)
    assert ann.user_sim_interaction.n_asks == 0


def test_submission_annotation_user_sim_explicit_none_persists():
    """Explicit None (new one-shot writes) round-trips as null."""
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
            verdict="agent_miss",
        ),
        failure_classification=FailureClassification(
            primary="agent_miss", agent_at_fault=True, remediation_target="agent",
        ),
        user_sim_interaction=None,
    )
    data = json.loads(ann.model_dump_json())
    assert data["user_sim_interaction"] is None


# ---------------------------------------------------------------------------
# DEV-1541 §D. _build_prompt / tool-schema branching
# ---------------------------------------------------------------------------

_ASK_USER_TOKENS = (
    # Codex r2 #9 + CodeRabbit r2: include spaced/hyphenated phrasings —
    # the prompt is Python-generated so this is defensive against future
    # edits introducing "ask user" or "ask-user" wording.
    "ask_user",
    "ask user",
    "ask-user",
    "disclosed_resolutions",
    "undisclosed_resolutions",
    "user_sim",
    "user-sim",
    "key clarification",
    "never_asked_key_question",
    "asked_but_ignored_answer",
    "user_sim_misleading",
)


def test_build_prompt_one_shot_drops_all_ask_user_language():
    """Codex r1 #7 + r2 #9: the one-shot prompt must not name ask_user-related
    tokens AT ALL — including negative phrasings like 'do NOT use ask_user'
    or casing variants. Reframe positively."""
    from bird_interact_agents.eval.autopsy import _build_prompt

    ta = _minimal_task_annotation()
    prompt = _build_prompt(
        task_annotation=ta,
        trajectory=[],
        kb_text="",
        miss_diagnostics=None,
        is_one_shot=True,
    )
    # Case-insensitive check — catches 'Ask_User' / 'USER_SIM' / similar
    prompt_lower = prompt.lower()
    leaked = [tok for tok in _ASK_USER_TOKENS if tok in prompt_lower]
    assert leaked == [], f"one-shot prompt leaks ask-user tokens: {leaked}"


def test_build_prompt_one_shot_lists_six_patterns():
    """The one-shot prompt enumerates the 6 valid patterns."""
    from bird_interact_agents.eval.autopsy import _build_prompt

    ta = _minimal_task_annotation()
    prompt = _build_prompt(
        task_annotation=ta,
        trajectory=[],
        kb_text="",
        miss_diagnostics=None,
        is_one_shot=True,
    )
    for p in _ONE_SHOT_PATTERNS:
        assert p in prompt, f"one-shot prompt missing pattern {p!r}"


def test_build_prompt_a_interact_regression_still_lists_ask_user_patterns():
    """Regression guard: the a-interact prompt stays as today — all 9 patterns
    described, including the 3 ask_user-related ones."""
    from bird_interact_agents.eval.autopsy import _build_prompt

    ta = _minimal_task_annotation()
    prompt = _build_prompt(
        task_annotation=ta,
        trajectory=[],
        kb_text="",
        miss_diagnostics=None,
        is_one_shot=False,
    )
    for p in _ASK_USER_PATTERNS:
        assert p in prompt, f"a-interact prompt missing pattern {p!r}"
    for p in _ONE_SHOT_PATTERNS:
        # 'other' is shared too, but the rest must appear too
        assert p in prompt


def test_tool_schema_one_shot_drops_ask_user_properties_and_required():
    """Codex r1 #8: 4 properties removed (n_asks, key_asks,
    disclosed_resolutions, undisclosed_resolutions); 3 `required` entries
    removed (n_asks was never required)."""
    from bird_interact_agents.eval.autopsy import (
        _AUTOPSY_TOOL_SCHEMA,
        _AUTOPSY_TOOL_SCHEMA_ONE_SHOT,
    )

    a_props = set(_AUTOPSY_TOOL_SCHEMA["input_schema"]["properties"])
    o_props = set(_AUTOPSY_TOOL_SCHEMA_ONE_SHOT["input_schema"]["properties"])
    removed_props = a_props - o_props
    assert removed_props == {
        "n_asks", "key_asks", "disclosed_resolutions", "undisclosed_resolutions",
    }

    a_req = set(_AUTOPSY_TOOL_SCHEMA["input_schema"]["required"])
    o_req = set(_AUTOPSY_TOOL_SCHEMA_ONE_SHOT["input_schema"]["required"])
    removed_req = a_req - o_req
    assert removed_req == {
        "key_asks", "disclosed_resolutions", "undisclosed_resolutions",
    }
    # pattern remains required.
    assert "pattern" in o_req


def test_tool_schema_one_shot_pattern_enum_matches(_ONE_SHOT_PATTERNS=_ONE_SHOT_PATTERNS):
    from bird_interact_agents.eval.autopsy import _AUTOPSY_TOOL_SCHEMA_ONE_SHOT

    pattern_enum = set(
        _AUTOPSY_TOOL_SCHEMA_ONE_SHOT["input_schema"]["properties"]["pattern"]["enum"]
    )
    assert pattern_enum == set(_ONE_SHOT_PATTERNS)


# ---------------------------------------------------------------------------
# DEV-1541 §E. run_autopsy — one-shot path + typed exception clauses
# ---------------------------------------------------------------------------

def _stub_tool_use(input_dict: dict):
    """Build a mock anthropic Message with a single tool_use block."""
    mock_tool = MagicMock()
    mock_tool.type = "tool_use"
    mock_tool.name = "autopsy_output"
    mock_tool.input = input_dict
    mock_response = MagicMock()
    mock_response.content = [mock_tool]
    return mock_response


@pytest.mark.asyncio
async def test_run_autopsy_one_shot_validates_and_returns_one_shot_analysis(tmp_path):
    """One-shot tool_input (no key_asks/resolutions) with is_one_shot=True
    yields an AutopsyAnalysisOneShot with user_sim_interaction=None."""
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyAnalysisOneShot,
        AutopsyResult,
    )
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()
    tool_input = {
        "pattern": "wrong_join_path",
        "other_details": None,
        "narrative": "Agent used the wrong join path.",
        "remediation": "Fix host discovery.",
        "decision_point_trajectory_index": 4,
        "decision_point_description": "Wrong join chosen at step 4.",
    }
    mock_response = _stub_tool_use(tool_input)
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )

    assert isinstance(result, AutopsyResult)
    assert result.error is None
    assert isinstance(result.analysis, AutopsyAnalysisOneShot)
    assert result.analysis.kind == "one_shot"
    assert result.analysis.pattern == "wrong_join_path"
    assert result.decision_point is not None
    assert result.decision_point.trajectory_item_index == 4
    # One-shot has no user-sim by construction.
    assert result.user_sim_interaction is None


@pytest.mark.asyncio
async def test_run_autopsy_validation_error_to_autopsy_error_repro_robot_10(tmp_path):
    """Robot_10 repro: stub the actual LLM payload shape that broke
    autopsy.py:384 — `key_asks: []`, no resolutions arrays — and run
    against the A-INTERACT schema. The pydantic ValidationError must
    surface as AutopsyResult.error.kind == 'validation_error', NOT
    swallowed."""
    from bird_interact_agents.eval.annotation_schema import AutopsyResult
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()
    # The actual robot_10 payload shape (one-shot, no resolutions).
    tool_input = {
        "pattern": "exhausted_budget_guessing",
        "other_details": None,
        "narrative": "Budget exhausted on KB exploration.",
        "remediation": "Increase agent budget for KB-heavy tasks.",
        "decision_point_trajectory_index": None,
        "decision_point_description": None,
        "n_asks": 0,
        "key_asks": [],
        # `disclosed_resolutions` and `undisclosed_resolutions` missing —
        # the production failure mode.
    }
    mock_response = _stub_tool_use(tool_input)
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            # Force the a-interact schema — that's what robot_10 hit.
            is_one_shot=False,
        )

    assert isinstance(result, AutopsyResult)
    assert result.analysis is None
    assert result.error is not None
    assert result.error.kind == "validation_error"
    assert "ValidationError" in result.error.exception_class
    # The failing fields are named in the excerpt.
    assert "disclosed_resolutions" in result.error.message_excerpt
    assert "undisclosed_resolutions" in result.error.message_excerpt


@pytest.mark.asyncio
async def test_run_autopsy_api_connection_error_maps_to_network_error(tmp_path):
    """Codex r1 #5: APIConnectionError must be caught BEFORE APIError so it
    maps to kind='network_error', not 'api_error'. Use the actual installed
    anthropic exception class."""
    import anthropic
    from bird_interact_agents.eval.annotation_schema import AutopsyResult
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()
    mock_client = AsyncMock()
    mock_client.messages.create.side_effect = anthropic.APIConnectionError(
        request=MagicMock(),
    )

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )
    assert isinstance(result, AutopsyResult)
    assert result.error is not None
    assert result.error.kind == "network_error"
    assert "ConnectionError" in result.error.exception_class


@pytest.mark.asyncio
async def test_run_autopsy_api_timeout_is_network_error(tmp_path):
    """Codex r1 #5: APITimeoutError is a subclass of APIConnectionError in
    the installed Anthropic SDK; it must also resolve to network_error."""
    import anthropic
    from bird_interact_agents.eval.annotation_schema import AutopsyResult
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()
    mock_client = AsyncMock()
    mock_client.messages.create.side_effect = anthropic.APITimeoutError(
        request=MagicMock(),
    )

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )
    assert isinstance(result, AutopsyResult)
    assert result.error is not None
    assert result.error.kind == "network_error"


@pytest.mark.asyncio
async def test_run_autopsy_non_overflow_bad_request_is_api_error(tmp_path):
    """BadRequestError without context-window language → api_error
    (distinct from context_overflow)."""
    import anthropic
    from bird_interact_agents.eval.annotation_schema import AutopsyResult
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()
    mock_client = AsyncMock()
    mock_client.messages.create.side_effect = anthropic.BadRequestError(
        message="invalid_request_error: missing required parameter `tools`",
        response=MagicMock(status_code=400),
        body={},
    )

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )
    assert isinstance(result, AutopsyResult)
    assert result.error is not None
    assert result.error.kind == "api_error"


@pytest.mark.asyncio
async def test_run_autopsy_missing_tool_use_block(tmp_path):
    """Anthropic response without a tool_use block → kind='missing_tool_use'."""
    from bird_interact_agents.eval.annotation_schema import AutopsyResult
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()
    mock_response = MagicMock()
    # No tool_use block — just a text block.
    text_block = MagicMock()
    text_block.type = "text"
    mock_response.content = [text_block]
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )
    assert isinstance(result, AutopsyResult)
    assert result.error is not None
    assert result.error.kind == "missing_tool_use"


# ---------------------------------------------------------------------------
# DEV-1541 §F. user_sim default plumbing on one-shot benchmarks
# ---------------------------------------------------------------------------

def test_resolve_default_user_sim_one_shot_returns_none():
    from bird_interact_agents.eval.grade_in_place import _resolve_default_user_sim

    result = _resolve_default_user_sim(
        benchmark="livesqlbench-base-lite-sqlite",
        provided=None,
        n_ask_user_calls=None,
    )
    assert result is None


def test_resolve_default_user_sim_a_interact_returns_zero_asks():
    from bird_interact_agents.eval.annotation_schema import UserSimInteraction
    from bird_interact_agents.eval.grade_in_place import _resolve_default_user_sim

    result = _resolve_default_user_sim(
        benchmark="mini-interact",
        provided=None,
        n_ask_user_calls=None,
    )
    assert isinstance(result, UserSimInteraction)
    assert result.n_asks == 0


def test_resolve_default_user_sim_a_interact_threads_n_ask_user_calls():
    from bird_interact_agents.eval.grade_in_place import _resolve_default_user_sim

    result = _resolve_default_user_sim(
        benchmark="mini-interact",
        provided=None,
        n_ask_user_calls=7,
    )
    assert result is not None
    assert result.n_asks == 7


def test_resolve_default_user_sim_passes_through_explicit_provided():
    from bird_interact_agents.eval.annotation_schema import UserSimInteraction
    from bird_interact_agents.eval.grade_in_place import _resolve_default_user_sim

    provided = UserSimInteraction(n_asks=3, disclosed_resolutions=["x"])
    result = _resolve_default_user_sim(
        benchmark="livesqlbench-base-lite-sqlite",
        provided=provided,
        n_ask_user_calls=None,
    )
    assert result is provided


def test_user_sim_interaction_from_trajectory_one_shot_returns_none():
    """One-shot benchmarks have no user-sim — the helper returns None."""
    from bird_interact_agents.eval.annotate import _user_sim_interaction_from_trajectory

    traj = [
        {"role": "tool_call", "name": "ask_user", "args": "q"},
        {"role": "tool_response", "name": "ask_user", "content": "a"},
    ]
    result = _user_sim_interaction_from_trajectory(traj, one_shot=True)
    assert result is None


def test_user_sim_interaction_from_trajectory_a_interact_unchanged():
    """A-interact path: behaviour is unchanged from pre-DEV-1541."""
    from bird_interact_agents.eval.annotate import _user_sim_interaction_from_trajectory

    traj = [
        {"role": "tool_call", "name": "ask_user", "args": "q"},
        {"role": "tool_response", "name": "ask_user", "content": "a"},
    ]
    result = _user_sim_interaction_from_trajectory(traj, one_shot=False)
    assert result is not None
    assert result.n_asks == 1


def test_user_sim_interaction_from_trajectory_default_kwarg_is_a_interact():
    """Default `one_shot=False` keeps the legacy a-interact behaviour for
    callers that haven't been updated yet."""
    from bird_interact_agents.eval.annotate import _user_sim_interaction_from_trajectory

    result = _user_sim_interaction_from_trajectory([])
    assert result is not None
    assert result.n_asks == 0


# ---------------------------------------------------------------------------
# DEV-1541 §G. grade_and_write — guard against autopsy_error clobbering top-level
# ---------------------------------------------------------------------------

def test_grade_and_write_autopsy_error_does_not_overwrite_top_level_user_sim(tmp_path):
    """Codex r1 #2: an AutopsyResult carrying ONLY an `error` (no
    analysis) must NOT overwrite the top-level user_sim_interaction
    that grade_and_write already computed."""
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyResult,
        UserSimInteraction,
    )
    from bird_interact_agents.eval.grade_in_place import grade_and_write

    db = _build_sqlite(tmp_path)
    task_ann = _minimal_task_annotation(instance_id="x_1", db="x")
    err = _make_autopsy_error(kind="validation_error")
    autopsy_with_error = AutopsyResult(error=err)
    # Provide a known top-level user_sim — this is what must survive.
    provided_us = UserSimInteraction(
        n_asks=4,
        disclosed_resolutions=["threshold=5"],
    )

    ann_path = grade_and_write(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        benchmark="mini-interact",
        run_id="test-run",
        task_annotation=task_ann,
        audited_gold_rows=[],
        original_sol_sql=["SELECT id FROM t"],
        submitted_sql="SELECT * FROM t",
        db_path=db,
        trajectory_path="rows/x_1/attempt-1.json",
        user_sim_interaction=provided_us,
        autopsy_result=autopsy_with_error,
    )

    data = json.loads(ann_path.read_text())
    # autopsy.error is persisted
    assert data["autopsy"]["error"]["kind"] == "validation_error"
    assert data["autopsy"]["analysis"] is None
    # but top-level user_sim_interaction was NOT clobbered by None
    assert data["user_sim_interaction"] is not None
    assert data["user_sim_interaction"]["n_asks"] == 4
    assert data["user_sim_interaction"]["disclosed_resolutions"] == ["threshold=5"]


def test_grade_and_write_autopsy_error_does_not_overwrite_top_level_decision_point(tmp_path):
    """Codex r2 #11: strengthen the decision_point guard by giving the
    error-bearing AutopsyResult its OWN decision_point. The guard must
    still skip the top-level overwrite because `analysis is None`."""
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyResult,
        TrajectoryDecisionPoint,
    )
    from bird_interact_agents.eval.grade_in_place import grade_and_write

    db = _build_sqlite(tmp_path)
    task_ann = _minimal_task_annotation(instance_id="x_1", db="x")
    # Give the error-bearing AutopsyResult a non-null decision_point. A
    # buggy unconditional copy would overwrite the top-level field with
    # this stray decision_point even though the autopsy failed.
    autopsy_with_error_and_dp = AutopsyResult(
        error=_make_autopsy_error(),
        decision_point=TrajectoryDecisionPoint(
            trajectory_item_index=99,
            description="stray dp from a failed autopsy",
        ),
    )

    ann_path = grade_and_write(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        benchmark="mini-interact",
        run_id="test-run",
        task_annotation=task_ann,
        audited_gold_rows=[],
        original_sol_sql=["SELECT id FROM t"],
        submitted_sql="SELECT * FROM t",
        db_path=db,
        trajectory_path="rows/x_1/attempt-1.json",
        autopsy_result=autopsy_with_error_and_dp,
    )
    data = json.loads(ann_path.read_text())
    # autopsy.error persisted (incl. the stray dp on the autopsy itself)
    assert data["autopsy"]["error"] is not None
    assert data["autopsy"]["decision_point"]["trajectory_item_index"] == 99
    # But top-level decision_point stays at the default None — the
    # stray dp on the failed autopsy did not propagate.
    assert data["decision_point"] is None


def test_grade_and_write_autopsy_analysis_still_overwrites_top_level(tmp_path):
    """Regression guard: when autopsy.analysis IS populated, the top-level
    overwrite still works (existing DEV-1521 contract)."""
    from bird_interact_agents.eval.annotation_schema import (
        AutopsyAnalysis,
        AutopsyResult,
        TrajectoryDecisionPoint,
        UserSimInteraction,
    )
    from bird_interact_agents.eval.grade_in_place import grade_and_write

    db = _build_sqlite(tmp_path)
    task_ann = _minimal_task_annotation(instance_id="x_1", db="x")

    autopsy_with_analysis = AutopsyResult(
        analysis=AutopsyAnalysis(
            pattern="wrong_join_path",
            narrative="n",
            remediation="r",
        ),
        decision_point=TrajectoryDecisionPoint(
            trajectory_item_index=2,
            description="step 2",
        ),
        user_sim_interaction=UserSimInteraction(n_asks=9),
    )

    ann_path = grade_and_write(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        benchmark="mini-interact",
        run_id="test-run",
        task_annotation=task_ann,
        audited_gold_rows=[],
        original_sol_sql=["SELECT id FROM t"],
        submitted_sql="SELECT * FROM t",
        db_path=db,
        trajectory_path="rows/x_1/attempt-1.json",
        autopsy_result=autopsy_with_analysis,
    )
    data = json.loads(ann_path.read_text())
    assert data["decision_point"]["trajectory_item_index"] == 2
    assert data["user_sim_interaction"]["n_asks"] == 9


def test_grade_and_write_one_shot_benchmark_user_sim_is_null(tmp_path):
    """End-to-end: grade_and_write on a one-shot benchmark with no
    user_sim provided → annotation has user_sim_interaction=null."""
    from bird_interact_agents.eval.grade_in_place import grade_and_write

    db = _build_sqlite(tmp_path)
    task_ann = _minimal_task_annotation(instance_id="x_1", db="x")

    ann_path = grade_and_write(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        benchmark="livesqlbench-base-lite-sqlite",
        run_id="test-run",
        task_annotation=task_ann,
        audited_gold_rows=[],
        original_sol_sql=["SELECT id FROM t"],
        submitted_sql="SELECT id FROM t",
        db_path=db,
        trajectory_path="rows/x_1/attempt-1.json",
    )
    data = json.loads(ann_path.read_text())
    assert data["user_sim_interaction"] is None


def test_write_failed_submission_annotation_one_shot_user_sim_none(tmp_path):
    """CodeRabbit r2: one-shot benchmarks failing BEFORE the grader runs
    must also persist user_sim_interaction=null, not a fake zero-asks
    UserSimInteraction()."""
    from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation
    from bird_interact_agents.eval.grade_in_place import (
        write_failed_submission_annotation,
    )

    out_path = write_failed_submission_annotation(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        selected_database="x",
        benchmark="livesqlbench-base-lite-sqlite",
        run_id="r",
        trajectory_path="rows/x_1/attempt-1.json",
        failure_details="boom",
    )
    ann = SubmissionAnnotation.model_validate_json(out_path.read_text())
    assert ann.user_sim_interaction is None


def test_write_failed_submission_annotation_a_interact_user_sim_default(tmp_path):
    """Regression guard: a-interact failed-submission writers keep the
    legacy zero-asks default."""
    from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation
    from bird_interact_agents.eval.grade_in_place import (
        write_failed_submission_annotation,
    )

    out_path = write_failed_submission_annotation(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        selected_database="x",
        benchmark="mini-interact",
        run_id="r",
        trajectory_path="rows/x_1/attempt-1.json",
        failure_details="boom",
        n_ask_user_calls=3,
    )
    ann = SubmissionAnnotation.model_validate_json(out_path.read_text())
    assert ann.user_sim_interaction is not None
    assert ann.user_sim_interaction.n_asks == 3


def test_grade_and_write_a_interact_benchmark_user_sim_default(tmp_path):
    """End-to-end regression guard: grade_and_write on an a-interact
    benchmark with no user_sim provided → annotation has the legacy
    default-constructed UserSimInteraction(n_asks=0)."""
    from bird_interact_agents.eval.grade_in_place import grade_and_write

    db = _build_sqlite(tmp_path)
    task_ann = _minimal_task_annotation(instance_id="x_1", db="x")

    ann_path = grade_and_write(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        benchmark="mini-interact",
        run_id="test-run",
        task_annotation=task_ann,
        audited_gold_rows=[],
        original_sol_sql=["SELECT id FROM t"],
        submitted_sql="SELECT id FROM t",
        db_path=db,
        trajectory_path="rows/x_1/attempt-1.json",
    )
    data = json.loads(ann_path.read_text())
    assert data["user_sim_interaction"] is not None
    assert data["user_sim_interaction"]["n_asks"] == 0


# ---------------------------------------------------------------------------
# DEV-1541 §H. Codex r2 gap-fill: tool-schema actually sent, agent call sites,
# LLM-output schema contracts, additional exception paths, harness-confirmed
# path, AutopsyError validators.
# ---------------------------------------------------------------------------

# H.1 — AutopsyLLMOutput / AutopsyLLMOutputOneShot direct contracts (Codex r2 #4, #5)

# CodeRabbit r2 nitpick: alias the canonical list at module top to
# avoid drift between the schema parametrize and the LLM-output
# parametrize.
_ALL_AUTOPSY_PATTERNS_LIST = _ALL_AUTOPSY_PATTERNS


@pytest.mark.parametrize("pattern", _ALL_AUTOPSY_PATTERNS_LIST)
def test_autopsy_llm_output_accepts_each_pattern(pattern):
    """AutopsyLLMOutput (a-interact LLM-output schema, internal to autopsy.py)
    accepts every one of the 9 patterns."""
    from bird_interact_agents.eval.autopsy import AutopsyLLMOutput

    out = AutopsyLLMOutput(
        pattern=pattern,
        narrative="n",
        remediation="r",
        n_asks=0,
        key_asks=[],
        disclosed_resolutions=[],
        undisclosed_resolutions=[],
    )
    assert out.pattern == pattern


def test_autopsy_llm_output_rejects_invalid_pattern():
    """A-interact LLM-output schema rejects bogus patterns."""
    from pydantic import ValidationError
    from bird_interact_agents.eval.autopsy import AutopsyLLMOutput

    with pytest.raises(ValidationError):
        AutopsyLLMOutput(
            pattern="made_up_pattern",
            narrative="n",
            remediation="r",
            n_asks=0,
            key_asks=[],
            disclosed_resolutions=[],
            undisclosed_resolutions=[],
        )


@pytest.mark.parametrize("pattern", _ONE_SHOT_PATTERNS)
def test_autopsy_llm_output_one_shot_accepts_subset(pattern):
    """AutopsyLLMOutputOneShot (one-shot LLM-output schema) accepts each
    of the 6 valid one-shot patterns."""
    from bird_interact_agents.eval.autopsy import AutopsyLLMOutputOneShot

    out = AutopsyLLMOutputOneShot(
        pattern=pattern,
        narrative="n",
        remediation="r",
    )
    assert out.pattern == pattern


@pytest.mark.parametrize("pattern", _ASK_USER_PATTERNS)
def test_autopsy_llm_output_one_shot_rejects_ask_user_patterns(pattern):
    """One-shot LLM-output schema rejects the 3 ask_user-related patterns."""
    from pydantic import ValidationError
    from bird_interact_agents.eval.autopsy import AutopsyLLMOutputOneShot

    with pytest.raises(ValidationError):
        AutopsyLLMOutputOneShot(
            pattern=pattern,
            narrative="n",
            remediation="r",
        )


def test_autopsy_llm_output_one_shot_lacks_ask_user_fields():
    """AutopsyLLMOutputOneShot drops the 4 ask_user-related fields entirely
    (Codex r1 #8: 4 properties, not 3 — n_asks plus the three list fields)."""
    from bird_interact_agents.eval.autopsy import AutopsyLLMOutputOneShot

    fields = set(AutopsyLLMOutputOneShot.model_fields)
    removed = {"n_asks", "key_asks", "disclosed_resolutions", "undisclosed_resolutions"}
    assert fields.isdisjoint(removed), (
        f"AutopsyLLMOutputOneShot still carries removed fields: "
        f"{fields & removed}"
    )


# H.2 — run_autopsy actually sends the right tool schema (Codex r2 #3)

@pytest.mark.asyncio
async def test_run_autopsy_one_shot_sends_one_shot_tool_schema(tmp_path):
    """Verify run_autopsy(is_one_shot=True) actually forwards the
    one-shot tool schema to messages.create — not the a-interact one."""
    from bird_interact_agents.eval.autopsy import (
        _AUTOPSY_TOOL_SCHEMA_ONE_SHOT,
        run_autopsy,
    )

    task_ann = _minimal_task_annotation()
    tool_input = {
        "pattern": "wrong_join_path",
        "other_details": None,
        "narrative": "n",
        "remediation": "r",
        "decision_point_trajectory_index": None,
        "decision_point_description": None,
    }
    mock_response = _stub_tool_use(tool_input)
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )
    call_kwargs = mock_client.messages.create.call_args.kwargs
    sent_tools = call_kwargs["tools"]
    assert sent_tools == [_AUTOPSY_TOOL_SCHEMA_ONE_SHOT]


@pytest.mark.asyncio
async def test_run_autopsy_a_interact_sends_a_interact_tool_schema(tmp_path):
    """Regression guard: a-interact still gets the original tool schema."""
    from bird_interact_agents.eval.autopsy import (
        _AUTOPSY_TOOL_SCHEMA,
        run_autopsy,
    )

    task_ann = _minimal_task_annotation()
    tool_input = {
        "pattern": "wrong_join_path",
        "other_details": None,
        "narrative": "n",
        "remediation": "r",
        "decision_point_trajectory_index": None,
        "decision_point_description": None,
        "n_asks": 0,
        "key_asks": [],
        "disclosed_resolutions": [],
        "undisclosed_resolutions": [],
    }
    mock_response = _stub_tool_use(tool_input)
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=False,
        )
    call_kwargs = mock_client.messages.create.call_args.kwargs
    sent_tools = call_kwargs["tools"]
    assert sent_tools == [_AUTOPSY_TOOL_SCHEMA]


# H.3 — Agent call sites pass is_one_shot correctly (Codex r2 #2)

def test_claude_sdk_otf_call_site_passes_is_one_shot_true():
    """claude_sdk_otf is the one-shot OTF agent — must pass is_one_shot=True."""
    import inspect

    from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

    src = inspect.getsource(ClaudeSDKOtfAgent.run_task)
    # The run_autopsy(...) call site must explicitly name is_one_shot=True.
    assert "is_one_shot=True" in src, (
        "claude_sdk_otf.run_task must call run_autopsy(..., is_one_shot=True)"
    )


def test_claude_sdk_otf_ainteract_call_site_passes_is_one_shot_false():
    """claude_sdk_otf_ainteract is the a-interact OTF agent — must pass
    is_one_shot=False."""
    import inspect

    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        ClaudeSDKOtfAInteractAgent,
    )

    src = inspect.getsource(ClaudeSDKOtfAInteractAgent.run_task)
    assert "is_one_shot=False" in src, (
        "claude_sdk_otf_ainteract.run_task must call run_autopsy(..., is_one_shot=False)"
    )


# H.4 — Additional run_autopsy exception paths (Codex r2 #8)

@pytest.mark.asyncio
async def test_run_autopsy_generic_anthropic_api_error_maps_to_api_error(tmp_path):
    """A bare anthropic.APIError (not BadRequestError, not APIConnectionError)
    must map to kind='api_error', distinct from network_error/context_overflow."""
    import anthropic
    from bird_interact_agents.eval.annotation_schema import AutopsyResult
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()
    mock_client = AsyncMock()
    # anthropic.APIError direct construction needs message + request + body.
    mock_client.messages.create.side_effect = anthropic.APIError(
        message="generic API error",
        request=MagicMock(),
        body={},
    )

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=True,
        )
    assert isinstance(result, AutopsyResult)
    assert result.error is not None
    assert result.error.kind == "api_error"


@pytest.mark.asyncio
async def test_run_autopsy_post_call_unknown_exception(tmp_path):
    """A generic Exception raised AFTER messages.create succeeds — e.g.
    during tool_use iteration — but NOT a pydantic.ValidationError and NOT
    a StopIteration — maps to kind='unknown'. Distinct from validation_error
    and missing_tool_use."""
    from bird_interact_agents.eval.annotation_schema import AutopsyResult
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()
    # Build a response whose `content` raises a non-StopIteration when
    # iterated. RuntimeError is generic Exception — must end up in
    # the post-call `except Exception` branch with kind='unknown'.
    bad_response = MagicMock()
    type(bad_response).content = property(
        lambda self: (_ for _ in ()).throw(RuntimeError("iteration boom"))
    )
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=bad_response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=False,
        )
    assert isinstance(result, AutopsyResult)
    assert result.error is not None
    assert result.error.kind == "unknown"
    assert "RuntimeError" in result.error.exception_class


# H.5 — _write_harness_confirmed_annotation user_sim plumbing (Codex r2 #10)

def test_write_harness_confirmed_annotation_user_sim_one_shot(tmp_path):
    """The harness-confirmed writer must also default user_sim to None on
    one-shot benchmarks (via _resolve_default_user_sim). Pre-DEV-1541 it
    hard-coded UserSimInteraction(n_asks=n_ask_user_calls or 0)."""
    from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation
    from bird_interact_agents.eval.grade_in_place import (
        _write_harness_confirmed_annotation,
    )

    task_ann = _minimal_task_annotation(instance_id="x_1", db="x")
    out_path = _write_harness_confirmed_annotation(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        selected_database="x",
        benchmark="livesqlbench-base-lite-sqlite",
        run_id="r",
        attempt=1,
        task_annotation=task_ann,
        cost_usd_agent=None,
        cost_usd_user_sim=None,
        duration_s=None,
        n_agent_turns=None,
        n_ask_user_calls=None,
        predicted_row_count=None,
        user_sim_interaction=None,
    )
    ann = SubmissionAnnotation.model_validate_json(out_path.read_text())
    assert ann.user_sim_interaction is None


def test_write_harness_confirmed_annotation_user_sim_a_interact(tmp_path):
    """A-interact regression: still defaults to UserSimInteraction(n_asks=...)."""
    from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation
    from bird_interact_agents.eval.grade_in_place import (
        _write_harness_confirmed_annotation,
    )

    task_ann = _minimal_task_annotation(instance_id="x_1", db="x")
    out_path = _write_harness_confirmed_annotation(
        rows_dir=tmp_path / "rows",
        instance_id="x_1",
        selected_database="x",
        benchmark="mini-interact",
        run_id="r",
        attempt=1,
        task_annotation=task_ann,
        cost_usd_agent=None,
        cost_usd_user_sim=None,
        duration_s=None,
        n_agent_turns=None,
        n_ask_user_calls=2,
        predicted_row_count=None,
        user_sim_interaction=None,
    )
    ann = SubmissionAnnotation.model_validate_json(out_path.read_text())
    assert ann.user_sim_interaction is not None
    assert ann.user_sim_interaction.n_asks == 2


# H.6 — AutopsyError field-level cap normalization (Codex r2 #7)

@pytest.mark.asyncio
async def test_run_autopsy_kb_read_raise_yields_unknown_error(tmp_path):
    """CodeRabbit r2: prep-time exceptions (KB read, prompt build) must
    NOT bypass the error boundary. If ``_read_kb_text`` raises before
    the LLM call ever happens, run_autopsy must still return
    AutopsyResult(error=AutopsyError(kind="unknown", ...)) — anything
    else lets the agent caller's outer except swallow it and persist
    autopsy=None, which is the silent-fail this PR exists to kill."""
    from bird_interact_agents.eval.annotation_schema import AutopsyResult
    from bird_interact_agents.eval.autopsy import run_autopsy

    task_ann = _minimal_task_annotation()

    # Anthropic client should never be instantiated — exception fires
    # before we get there.
    with patch(
        "bird_interact_agents.eval.autopsy._read_kb_text",
        side_effect=RuntimeError("KB read boom"),
    ), patch("anthropic.AsyncAnthropic") as mock_anthropic:
        result = await run_autopsy(
            task_annotation=task_ann,
            trajectory=[],
            slayer_storage_dir=str(tmp_path),
            miss_diagnostics=None,
            model="anthropic/claude-sonnet-4-5",
            is_one_shot=False,
        )

    mock_anthropic.assert_not_called()
    assert isinstance(result, AutopsyResult)
    assert result.analysis is None
    assert result.error is not None
    assert result.error.kind == "unknown"
    assert "RuntimeError" in result.error.exception_class
    assert "KB read boom" in result.error.message_excerpt
    # Stats reflect that prep never finished — prompt_chars is 0 because
    # _build_prompt never ran; kb_chars is 0 because _read_kb_text raised.
    assert result.error.prompt_chars == 0
    assert result.error.kb_chars == 0


def test_autopsy_error_message_excerpt_capped_at_construction():
    """An overlong message_excerpt passed at construction time gets
    auto-truncated by the schema's field validator — not the caller's
    responsibility to remember to call _truncate."""
    from bird_interact_agents.eval.annotation_schema import AutopsyError

    long_msg = "z" * 5000
    err = _make_autopsy_error(message_excerpt=long_msg)
    # Field validator on AutopsyError caps at 500 with "...[truncated]" suffix.
    assert len(err.message_excerpt) <= 500
    assert err.message_excerpt.endswith("...[truncated]")


def test_autopsy_error_traceback_excerpt_capped_at_construction():
    """Same auto-cap for traceback_excerpt at 2000 chars."""
    from bird_interact_agents.eval.annotation_schema import AutopsyError

    long_tb = "w" * 5000
    err = _make_autopsy_error(traceback_excerpt=long_tb)
    assert len(err.traceback_excerpt) <= 2000
    assert err.traceback_excerpt.endswith("...[truncated]")
