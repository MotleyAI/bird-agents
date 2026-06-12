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
    instance_id: str | None = None,
) -> tuple[SubmissionRow, list[str]]:
    """Build the SubmissionRow PLUS the list of converter warnings
    (phase-split warnings, etc.) that the CLI surfaces in
    ``manifest.warnings_by_instance``.

    Codex round 9: when ``instance_id`` is supplied it is the authoritative
    id (from the selection / results.db row); the trajectory's stamped id is
    validated against it. A mismatch indicates a stale / mis-copied
    trajectory file and adds a manifest warning. When ``instance_id`` is
    None the trajectory's stamped id is used as a fallback (kept for
    legacy callers / unit tests that don't pre-thread the trusted id).
    """
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
    extra_warnings: list[str] = []

    # The compiled SQL the leaderboard grades comes from one of two
    # places depending on query_mode:
    # * ``raw``: the agent literally passed SQL in ``query`` /
    #   ``query_json``; ``submit(<sql>)`` already has it. We pull from
    #   ``turn.tool_input`` directly so we don't depend on the
    #   canonical-string slice (which keeps any wrapping the
    #   canonicalizer added).
    # * ``slayer``: the agent passed a SlayerQuery JSON DSL; the SERVER
    #   compiled it to SQL but only the LAST submit's compiled SQL is
    #   persisted (as ``trajectory.submitted_sql``). Earlier-phase
    #   compiled SQL is LOST — we emit an empty list AND a manifest
    #   warning so the operator knows about the gap.
    trajectory_final_sql = str(trajectory_obj.get("submitted_sql") or "")

    def _looks_like_json_dsl(s: str) -> bool:
        """SLayer's ``submit_query.query_json`` accepts either a single
        SlayerQuery object (``{...}``) OR a nested-DAG array of stage
        objects (``[...]`` — see claude_sdk/agent.py:374). Both must
        route through the compiled-SQL extraction path; treating ``[``
        as raw SQL would silently emit JSON DSL where SQL is expected
        (Codex round 5 finding)."""
        head = s.strip()
        return head.startswith("{") or head.startswith("[")

    def _raw_input_sql(turn) -> str:
        for key in ("query", "query_json", "sql"):
            if key in turn.tool_input:
                return str(turn.tool_input[key])
        return ""

    last_phase1_sql = ""
    last_phase2_sql = ""
    have_phase1 = False
    have_phase2 = False
    last_submit_idx = submit_idxs[-1] if submit_idxs else None

    for k, idx in enumerate(submit_idxs):
        label = phase_result.labels[k] if k < len(phase_result.labels) else None
        turn = turns[idx]
        raw_input = _raw_input_sql(turn)
        if _looks_like_json_dsl(raw_input):
            # SLayer DSL: compiled SQL only available for the LAST
            # overall submit (the trajectory's `submitted_sql`).
            if idx == last_submit_idx and trajectory_final_sql:
                sql = trajectory_final_sql
            else:
                sql = ""  # earlier-phase compiled SQL not recoverable
        else:
            sql = raw_input

        if label == "phase2":
            last_phase2_sql = sql
            have_phase2 = True
        else:
            last_phase1_sql = sql
            have_phase1 = True

    # Manifest warning when an earlier phase's compiled SQL was lost
    # because the agent ran in SLayer mode and only the FINAL submit's
    # SQL is persisted (Codex round 4 finding).
    if have_phase1 and have_phase2 and not last_phase1_sql:
        extra_warnings.append(
            "SLayer-mode phase-1 SQL is not recoverable from the trajectory: "
            "only the final submit's compiled SQL is stored as "
            "`submitted_sql`; phase-1 was overwritten when phase-2 ran. "
            "Emitting empty subtask_1_predicted_sql; review trajectory manually "
            "or re-run with query_mode=raw to capture per-submit SQL."
        )

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

    payload_instance_id = str(trajectory_obj.get("instance_id") or "")
    if instance_id is not None:
        effective_instance_id = instance_id
        if payload_instance_id and payload_instance_id != instance_id:
            extra_warnings.append(
                f"trajectory.instance_id={payload_instance_id!r} mismatches "
                f"the source-resolved instance_id={instance_id!r}; using the "
                "source-resolved id. Check for a stale or mis-copied "
                "trajectory file."
            )
    else:
        effective_instance_id = payload_instance_id

    row = SubmissionRow(
        instance_id=effective_instance_id,
        subtask_1_predicted_sql=subtask_1_predicted_sql,
        subtask_2_predicted_sql=subtask_2_predicted_sql,
        prompt_flow=entries,
    )
    return row, list(phase_result.warnings) + extra_warnings


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
