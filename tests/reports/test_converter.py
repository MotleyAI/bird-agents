"""End-to-end converter tests: trajectory + results.db → SubmissionRow.

Spec (DEV-1553) tests #5 (adapter integration), #6 (phase SQL extraction),
#7 (end-to-end), #14 (results.db cross-check warning), #18 (--no-thinking).
"""

from __future__ import annotations

from tests.reports._fixtures import (
    assistant_msg,
    build_trajectory,
    system_msg,
    tool_result_msg,
    tool_use_block,
    trajectory_no_submits,
    trajectory_one_phase_pass,
    trajectory_phase1_retry_then_phase2,
    trajectory_two_phase_pass,
    user_text_msg,
)


def _convert(traj_obj, *, task_data=None, patience=3, include_thinking=True):
    """Test helper — discards the warnings list (a few tests below assert
    on it explicitly via ``_convert_with_warnings``)."""
    row, _ = _convert_with_warnings(
        traj_obj,
        task_data=task_data,
        patience=patience,
        include_thinking=include_thinking,
    )
    return row


def _convert_with_warnings(
    traj_obj, *, task_data=None, patience=3, include_thinking=True
):
    from bird_interact_agents.reports.converter import build_submission_row

    return build_submission_row(
        trajectory_obj=traj_obj,
        framework="claude_sdk",
        agent_model="anthropic/claude-opus-4-7",
        user_sim_model="anthropic/claude-sonnet-4-6",
        task_data=task_data or {"user_query_ambiguity": {}, "knowledge_ambiguity": []},
        patience=patience,
        include_thinking=include_thinking,
    )


# ---------------------------------------------------------------------------
# Phase SQL extraction
# ---------------------------------------------------------------------------


def test_phase_sql_one_phase_pass(fake_count_tokens):
    traj = trajectory_one_phase_pass(sql="SELECT 1", instance_id="alien_1")
    row = _convert(traj)
    assert row.instance_id == "alien_1"
    assert row.subtask_1_predicted_sql == ["SELECT 1"]
    assert row.subtask_2_predicted_sql == []


def test_phase_sql_two_phase_pass(fake_count_tokens):
    traj = trajectory_two_phase_pass(
        phase1_sql="SELECT 1", phase2_sql="SELECT 2", instance_id="alien_2"
    )
    row = _convert(traj)
    assert row.subtask_1_predicted_sql == ["SELECT 1"]
    assert row.subtask_2_predicted_sql == ["SELECT 2"]


def test_phase_sql_retry_takes_final_per_phase(fake_count_tokens):
    """Phase-1 has 2 submits (wrong then right) — the RIGHT one wins."""
    traj = trajectory_phase1_retry_then_phase2(
        phase1_wrong_sql="SELECT 999",
        phase1_right_sql="SELECT 1",
        phase2_sql="SELECT 2",
    )
    row = _convert(traj)
    assert row.subtask_1_predicted_sql == ["SELECT 1"]
    assert row.subtask_2_predicted_sql == ["SELECT 2"]


def test_phase_sql_no_submits_both_empty(fake_count_tokens):
    traj = trajectory_no_submits(instance_id="alien_4")
    row = _convert(traj)
    assert row.subtask_1_predicted_sql == []
    assert row.subtask_2_predicted_sql == []


# ---------------------------------------------------------------------------
# prompt_flow shape
# ---------------------------------------------------------------------------


def test_prompt_flow_one_entry_per_tool_use(fake_count_tokens):
    traj = trajectory_phase1_retry_then_phase2()
    row = _convert(traj)
    # 3 submit_query calls.
    assert len(row.prompt_flow) == 3
    actions = [e.action for e in row.prompt_flow]
    assert all(a.startswith("submit(") for a in actions)


def test_prompt_flow_carries_model_and_user_sim_per_step(fake_count_tokens):
    traj = trajectory_one_phase_pass()
    row = _convert(traj)
    e = row.prompt_flow[0]
    assert e.model == "anthropic/claude-opus-4-7"
    assert e.user_simulator == "anthropic/claude-sonnet-4-6"


def test_prompt_flow_carries_action_costs_for_submit(fake_count_tokens):
    """``submit`` is a fixed-cost action; Section VI says cost = 3."""
    traj = trajectory_one_phase_pass()
    row = _convert(traj)
    assert row.prompt_flow[0].action == "submit(SELECT 1)"
    assert row.prompt_flow[0].action_cost == 3


def test_prompt_flow_remaining_budget_is_replayed(fake_count_tokens):
    """remaining_budget = max(0, total_budget - cum_section_vi)."""
    traj = trajectory_two_phase_pass()
    # total_budget with 0 amb, patience=3 = 12. Two submits cost 3+3=6.
    row = _convert(traj)
    rb = [e.remaining_budget for e in row.prompt_flow]
    assert rb == [9.0, 6.0]


def test_prompt_flow_token_count_uses_fake(fake_count_tokens):
    """``action_input_tokens`` / ``action_output_tokens`` come from the
    fake count_tokens (len // 4) so tests are deterministic."""
    traj = trajectory_one_phase_pass(sql="SELECT 1")
    row = _convert(traj)
    e = row.prompt_flow[0]
    # action_args_string for submit = the SQL string "SELECT 1" (len 8) → 2
    assert e.action_input_tokens == 2
    # action_output_tokens = len(observation) // 4 → some positive count
    assert e.action_output_tokens >= 1


# ---------------------------------------------------------------------------
# include_thinking flag (Codex finding #3 / spec --no-thinking)
# ---------------------------------------------------------------------------


def _traj_with_thinking():
    """A trajectory whose one assistant turn has thinking + text + tool_use.
    The literal token ``"thinking"`` appears in the rendered response iff
    the thinking block survives rendering — that's the structural envelope
    we assert on (avoids prompt-content assertion)."""
    steps = [
        system_msg(),
        user_text_msg(text="Task."),
        assistant_msg(
            thinking="reasoning aloud",
            text="visible text",
            tool_use=tool_use_block(
                tool_use_id="tu_1",
                name="mcp__bird-interact-tools__submit_query",
                inp={"query_json": "SELECT 1"},
            ),
        ),
        tool_result_msg(tool_use_id="tu_1", content="Phase 1 SQL Correct!"),
    ]
    return build_trajectory(trajectory_steps=steps, submitted_sql="SELECT 1")


def test_include_thinking_true_keeps_thinking_envelope(fake_count_tokens):
    row = _convert(_traj_with_thinking(), include_thinking=True)
    # Structural envelope check: the rendered response carries a "thinking"
    # marker (the SDK content-block type label) when thinking is preserved.
    assert "thinking" in row.prompt_flow[0].response


def test_include_thinking_false_strips_thinking_envelope(fake_count_tokens):
    row = _convert(_traj_with_thinking(), include_thinking=False)
    # No thinking envelope when stripped.
    assert "thinking" not in row.prompt_flow[0].response
    # Tool-use payload still present in the rendered response.
    assert "submit_query" in row.prompt_flow[0].response


def test_include_thinking_default_is_true(fake_count_tokens):
    """Confirm the converter's default keeps thinking blocks."""
    # No include_thinking kwarg — must default to True per spec.
    from bird_interact_agents.reports.converter import build_submission_row

    row, _warnings = build_submission_row(
        trajectory_obj=_traj_with_thinking(),
        framework="claude_sdk",
        agent_model="anthropic/claude-opus-4-7",
        user_sim_model="anthropic/claude-sonnet-4-6",
        task_data={"user_query_ambiguity": {}, "knowledge_ambiguity": []},
        patience=3,
    )
    assert "thinking" in row.prompt_flow[0].response


# ---------------------------------------------------------------------------
# results.db cross-check warning (Codex finding #6 — last-submit only)
# ---------------------------------------------------------------------------


def test_results_db_mismatch_yields_warning(fake_count_tokens):
    """When the trajectory's last submit differs from results.db's stored
    ``submitted_sql``, the converter records a warning. Not a hard error."""
    traj = trajectory_two_phase_pass(phase1_sql="A", phase2_sql="B")
    # Inject a deliberate mismatch.
    results_db_submitted_sql = "C"  # neither phase-1 nor phase-2 SQL
    from bird_interact_agents.reports.converter import (
        cross_check_results_db_sql,
    )

    row = _convert(traj)
    warnings = cross_check_results_db_sql(
        row=row, results_db_submitted_sql=results_db_submitted_sql
    )
    assert len(warnings) >= 1
    assert any("mismatch" in w.lower() for w in warnings)


def test_results_db_matching_yields_no_warning(fake_count_tokens):
    traj = trajectory_two_phase_pass(phase1_sql="A", phase2_sql="B")
    from bird_interact_agents.reports.converter import (
        cross_check_results_db_sql,
    )

    row = _convert(traj)
    warnings = cross_check_results_db_sql(
        row=row, results_db_submitted_sql="B"
    )
    assert warnings == []


def test_cross_check_uses_last_submit_only(fake_count_tokens):
    """results.db stores only ONE submitted_sql. Earlier phase-1 retries
    that differ from the DB string must NOT warn — only the final overall
    submit is checked against the DB column."""
    traj = trajectory_phase1_retry_then_phase2(
        phase1_wrong_sql="WRONG_PHASE1",
        phase1_right_sql="RIGHT_PHASE1",
        phase2_sql="FINAL_PHASE2",
    )
    from bird_interact_agents.reports.converter import (
        cross_check_results_db_sql,
    )

    row = _convert(traj)
    # Earlier wrong-phase-1 submit differs from DB-stored SQL; no warning.
    warnings = cross_check_results_db_sql(
        row=row, results_db_submitted_sql="FINAL_PHASE2"
    )
    assert warnings == []
