"""Trajectory + task-data → ``SubmissionRow``.

Top-level entry point: ``build_submission_row``.

* Walks the trajectory via the framework's registered adapter.
* Canonicalises each tool_use to upstream action names.
* Counts tokens for ``action_input_tokens`` / ``action_output_tokens``.
* Replays Section VI costs to compute per-step ``action_cost`` and
  cumulative ``remaining_budget``.
* Splits submits into phase-1 vs phase-2 using ``phase_split``.
* Stitches the final SQL per phase into ``subtask_K_predicted_sql``.
* Strips thinking blocks from each step's ``response`` when
  ``include_thinking=False``.
"""

from __future__ import annotations

import json

from bird_interact_agents.reports.action_canonicalize import (
    action_args_string,
    canonicalize_action,
)
from bird_interact_agents.reports.adapters import get_adapter
from bird_interact_agents.reports.budget import (
    calculate_total_budget,
    replay_remaining_budget,
)
from bird_interact_agents.reports.cost import compute_action_cost
from bird_interact_agents.reports.phase_split import split_phases, SplitResult
from bird_interact_agents.reports.schema import PromptFlowEntry, SubmissionRow
from bird_interact_agents.reports import tokens as _tokens


def _strip_thinking(response_raw: str) -> str:
    """Remove every ``{"type":"thinking", …}`` content item from the
    JSON-encoded ``response_raw``. Falls back to a no-op if the response
    isn't JSON-encoded for some reason."""
    try:
        items = json.loads(response_raw)
    except (json.JSONDecodeError, TypeError):
        return response_raw
    if not isinstance(items, list):
        return response_raw
    kept = [i for i in items if not (isinstance(i, dict) and i.get("type") == "thinking")]
    return json.dumps(kept, separators=(",", ":"))


def build_submission_row(
    *,
    trajectory_obj: dict,
    framework: str,
    agent_model: str,
    user_sim_model: str,
    task_data: dict,
    patience: int,
    include_thinking: bool = True,
    query_mode: str = "slayer",
) -> tuple[SubmissionRow, list[str]]:
    """Build the SubmissionRow PLUS the list of converter warnings
    (phase-split warnings, etc.) that the CLI surfaces in
    ``manifest.warnings_by_instance``."""
    walk = get_adapter(framework, query_mode=query_mode)
    steps = trajectory_obj.get("trajectory") or []
    turns = list(walk(steps))

    # ---- Canonicalize + token counts + Section VI cost --------------
    canonical_actions: list[str] = []
    action_in_toks: list[int] = []
    action_out_toks: list[int] = []
    action_costs: list[float] = []
    for t in turns:
        canonical = canonicalize_action(t.tool_name, t.tool_input)
        # canonical_op = "ask" / "submit" / "execute" / <full-fallback>.
        # The cost classifier reads the leading token.
        op = canonical.split("(", 1)[0]
        args_str = action_args_string(t.tool_name, t.tool_input)
        in_tokens = _tokens.count_tokens(args_str) if args_str else 0
        out_tokens = (
            _tokens.count_tokens(t.observation) if t.observation else 0
        )
        cost = compute_action_cost(
            op, input_tokens=in_tokens, output_tokens=out_tokens
        )
        canonical_actions.append(canonical)
        action_in_toks.append(in_tokens)
        action_out_toks.append(out_tokens)
        action_costs.append(cost)

    # ---- Budget replay ----------------------------------------------
    total_budget = calculate_total_budget(task_data, patience=patience)
    remaining = replay_remaining_budget(
        total_budget=total_budget, action_costs=action_costs
    )

    # ---- Phase split on the submit subset ---------------------------
    submit_idxs = [
        i for i, a in enumerate(canonical_actions) if a.startswith("submit(")
    ]
    submit_observations = [turns[i].observation for i in submit_idxs]
    phase_result: SplitResult = split_phases(submit_observations)
    # Reconstruct full per-turn phase labels: None outside the submit
    # subset. ``last_phase1_sql`` / ``last_phase2_sql`` track the final
    # SQL per phase.
    last_phase1_sql = ""
    last_phase2_sql = ""
    have_phase1 = False
    have_phase2 = False
    for k, idx in enumerate(submit_idxs):
        label = phase_result.labels[k] if k < len(phase_result.labels) else None
        # Extract SQL from the canonical action string ``submit(<sql>)``.
        canonical = canonical_actions[idx]
        sql = canonical[len("submit(") : -1]
        if label == "phase2":
            last_phase2_sql = sql
            have_phase2 = True
        else:
            last_phase1_sql = sql
            have_phase1 = True

    subtask_1_predicted_sql = [last_phase1_sql] if have_phase1 else []
    subtask_2_predicted_sql = [last_phase2_sql] if have_phase2 else []

    # ---- Build prompt_flow ------------------------------------------
    entries: list[PromptFlowEntry] = []
    for k, t in enumerate(turns):
        response = (
            t.response_raw if include_thinking else _strip_thinking(t.response_raw)
        )
        entries.append(
            PromptFlowEntry(
                model=agent_model,
                user_simulator=user_sim_model,
                prompt=t.prompt,
                response=response,
                action=canonical_actions[k],
                remaining_budget=remaining[k] if k < len(remaining) else total_budget,
                action_input_tokens=action_in_toks[k],
                action_output_tokens=action_out_toks[k],
                action_cost=action_costs[k],
            )
        )

    row = SubmissionRow(
        instance_id=str(trajectory_obj.get("instance_id") or ""),
        subtask_1_predicted_sql=subtask_1_predicted_sql,
        subtask_2_predicted_sql=subtask_2_predicted_sql,
        prompt_flow=entries,
    )
    return row, list(phase_result.warnings)


def cross_check_results_db_sql(
    *, row: SubmissionRow, results_db_submitted_sql: str
) -> list[str]:
    """Compare ``results.db.task_results.submitted_sql`` to the LAST phase's
    final SQL. Mismatch → return a warning string list (never raises).

    Codex finding #6: the DB only stores ONE final SQL, so the cross-check
    is necessarily last-submit only. Earlier phase-1 retries that differ
    from the DB are NOT flagged.
    """
    if row.subtask_2_predicted_sql:
        last = row.subtask_2_predicted_sql[0]
        phase = "2"
    elif row.subtask_1_predicted_sql:
        last = row.subtask_1_predicted_sql[0]
        phase = "1"
    else:
        # Nothing to cross-check.
        return []

    if last == results_db_submitted_sql:
        return []
    return [
        f"results.db.submitted_sql mismatches phase-{phase} reconstructed SQL "
        f"(instance_id={row.instance_id}): DB stored {results_db_submitted_sql!r}, "
        f"trajectory ended on {last!r}"
    ]
