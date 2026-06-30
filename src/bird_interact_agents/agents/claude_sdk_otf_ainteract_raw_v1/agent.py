"""Claude Agent SDK raw-SQL OTF agent (mini-interact / a-interact flavor).

A counterpart to ``claude_sdk_otf_ainteract`` that uses no SLayer at all —
the agent issues raw SQL via the bird-interact tool suite and submits via
``submit_sql``. Bound to ``--dataset mini_interact --mode a-interact
--query-mode raw``.

Adds the same hard ``ask_user``-before-``submit_sql`` discipline as the
slayer ainteract variant: Rule 0 plus per-task PreToolUse/PostToolUse guards
built by ``_make_ask_user_guards``.
"""

from __future__ import annotations

import logging
import time

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from bird_interact_agents.agents.claude_sdk.agent import (
    _ctx_var,
    SdkUsageTracker,
    build_bird_interact_server,
)
from bird_interact_agents.agents.claude_sdk.context_budget import (
    context_window_for,
    make_context_budget_hook,
    make_wall_clock_budget_hook,
    per_task_timeout_s,
    update_wall_clock_start,
)
from bird_interact_agents.agents.claude_sdk.discovery_runtime import (
    run_main_with_discovery,
)
from bird_interact_agents.agents.claude_sdk.partition import (
    DISCOVERY_MAX_TURNS,
    build_main_workflow_note,
    build_discovery_prompt,
)
from bird_interact_agents.agents.claude_sdk_otf_v1.agent import (
    _MAX_TURNS,
    _make_turn_budget_hook,
)
from bird_interact_agents.agents.claude_sdk_otf_raw_v1.agent import (
    DISCOVERY_NATIVE_TOOL_NAMES as _ONE_SHOT_RAW_DISCOVERY_TOOL_NAMES,
)
from bird_interact_agents.agents.claude_sdk_otf_raw_v1.agent import (
    MAIN_NATIVE_TOOL_NAMES as _ONE_SHOT_RAW_MAIN_TOOL_NAMES,
)
from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
    RAW_OTF_AINTERACT,
)
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.provider_registry import (
    is_supported_agent_model,
)
from bird_interact_agents.harness import (
    SampleStatus,
    _ambiguity_count,
    finalize_result_row,
    load_db_data_if_needed,
    materialize_task_db,
)
from bird_interact_agents.slayer_otf.timing import log_otf_event, otf_timer
from bird_interact_agents.usage import TokenUsage

logger = logging.getLogger(__name__)

# Full MCP tool name of the native ask_user tool — used as the PostToolUse
# counter's matcher and the nag hook's race-skip predicate.
_ASK_USER_TOOL = "mcp__bird-interact-tools__ask_user"

# Full MCP tool name of the raw submission tool — gated by the PreToolUse hook.
_SUBMIT_SQL_TOOL = "mcp__bird-interact-tools__submit_sql"

# How often (in total tool calls without ask_user) the nag fires.
_NAG_EVERY = 10

# DEV-1581 R2: a-interact raw partition = the one-shot raw partition +
# ask_user in BOTH contexts (two separate persistent clients).
MAIN_NATIVE_TOOL_NAMES = [*_ONE_SHOT_RAW_MAIN_TOOL_NAMES, _ASK_USER_TOOL]
DISCOVERY_NATIVE_TOOL_NAMES = [*_ONE_SHOT_RAW_DISCOVERY_TOOL_NAMES, _ASK_USER_TOOL]


def _make_ask_user_guards():
    """Build per-task hook callables that enforce ask-user-before-submit_sql.

    Returns ``(pre_submit_gate, post_ask_counter, post_nag)`` sharing a
    single per-task ``state`` closure. The factory MUST be invoked inside
    ``run_task`` (per task), NOT stored on the agent — a single agent
    instance is reused across concurrent tasks via ``make_runner``, and
    cross-task counter bleed would let one task's submit pass through
    another task's denial gate.

    The gate is scoped to ``submit_sql`` only; ``submit_query`` (if ever
    called) is not denied.
    """
    state = {"ask_count": 0, "tool_calls": 0}

    async def pre_submit_gate(input_data, tool_use_id, context):
        if input_data.get("tool_name") != _SUBMIT_SQL_TOOL:
            return {}
        if state["ask_count"] == 0:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "You have not called ask_user. The user-sim has the "
                        "masked-KB ground truth that is unrecoverable from "
                        "knowledge definitions alone. Identify your single "
                        "most-uncertain operationalisation choice (threshold / "
                        "value list / aggregation / sort / unit / rounding) "
                        "and call ask_user on it before submitting."
                    ),
                }
            }
        return {}

    async def post_ask_counter(input_data, tool_use_id, context):
        state["ask_count"] += 1
        return {}

    async def post_nag(input_data, tool_use_id, context):
        if input_data.get("tool_name") == _ASK_USER_TOOL:
            return {}
        state["tool_calls"] += 1
        if state["ask_count"] == 0 and state["tool_calls"] % _NAG_EVERY == 0:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": (
                        f"[BENCHMARK NOTE] You have made {state['tool_calls']} "
                        "tool calls without consulting the user-sim. The "
                        "user-sim has the masked-KB ground truth — clarify "
                        "your single most uncertain operationalisation choice "
                        "before continuing."
                    ),
                }
            }
        return {}

    return pre_submit_gate, post_ask_counter, post_nag


def _build_prompt(eval_mode: str, task_data: dict, budget: float) -> str:
    if eval_mode != "a-interact":
        raise ValueError(
            "claude_sdk_otf_ainteract_raw supports only eval_mode='a-interact'; "
            f"got {eval_mode!r}"
        )
    return RAW_OTF_AINTERACT.format(
        budget=budget,
        db_name=task_data["selected_database"],
        user_query=task_data["amb_user_query"],
    )


class ClaudeSDKOtfAInteractRawAgent:
    """SystemAgent: Claude SDK raw-SQL OTF agent with enforced ask-user discipline.

    Anthropic-only (the SDK is locked to Anthropic). Bound to
    ``--dataset mini_interact --mode a-interact --query-mode raw``. No SLayer
    MCP server. Mismatched dataset, eval_mode, or query_mode is rejected at
    the agent boundary.
    """

    _EFFORT_CHOICES = ("low", "medium", "high", "max")

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-5",
        reasoning_effort: str | None = None,
    ) -> None:
        if reasoning_effort is not None and reasoning_effort not in self._EFFORT_CHOICES:
            raise ValueError(
                f"reasoning_effort must be one of {self._EFFORT_CHOICES} or None; "
                f"got {reasoning_effort!r}"
            )
        self.model = model
        self.reasoning_effort = reasoning_effort

    async def run_task(
        self,
        task_data: dict,
        data_path_base: str,
        budget: float,
        query_mode: str,
        eval_mode: str = "a-interact",
        user_sim_model: str = "anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version: str = "v2",
    ) -> dict:
        if query_mode != "raw":
            raise ValueError(
                "claude_sdk_otf_ainteract_raw supports only --query-mode raw; "
                f"got {query_mode!r}"
            )
        if eval_mode != "a-interact":
            raise ValueError(
                "claude_sdk_otf_ainteract_raw supports only --mode a-interact; "
                f"got {eval_mode!r}"
            )

        dataset = task_data.get("dataset")
        if not dataset:
            raise ValueError("task_data missing required 'dataset' field")
        if get_benchmark(dataset).one_shot:
            raise ValueError(
                "claude_sdk_otf_ainteract_raw requires an a-interact benchmark; "
                f"got dataset={dataset!r}"
            )

        instance_id = task_data["instance_id"]
        db_name = task_data["selected_database"]

        if not is_supported_agent_model(self.model):
            msg = (
                f"claude_sdk_otf_ainteract_raw requires an Anthropic or registry "
                f"open-weight model; "
                f"got {self.model!r}. "
                "Skipped — use --framework pydantic_ai for non-Anthropic models."
            )
            logger.warning("[%s] %s", instance_id, msg)
            return finalize_result_row(
                {
                    "task_id": instance_id,
                    "instance_id": instance_id,
                    "database": db_name,
                    "phase1_passed": False,
                    "phase2_passed": False,
                    "total_reward": 0.0,
                    "trajectory": [],
                    "error": msg,
                },
                deleted_kb_ids=[],
                slayer_storage_dir="",
            )

        status = SampleStatus(
            idx=0,
            original_data=task_data,
            remaining_budget=budget,
            total_budget=budget,
        )

        max_asks = _ambiguity_count(task_data) + 3

        accum = TokenUsage()
        usage_tracker = SdkUsageTracker(accum, self.model)
        trajectory: list[dict] = []
        ctx_dict: dict | None = None
        try:
            log_otf_event("run_task.start", instance_id=instance_id, db=db_name)
            with otf_timer(
                "run_task.load_db_data", instance_id=instance_id, db=db_name,
            ):
                load_db_data_if_needed(db_name, data_path_base)
            with otf_timer(
                "run_task.materialize_task_db",
                instance_id=instance_id, db=db_name,
            ):
                materialize_task_db(task_data, data_path_base)

            ctx_dict = {
                "status": status,
                "data_path_base": data_path_base,
                "user_sim_model": user_sim_model,
                # DEV-1613: agent model for the in-task N5 judge.
                "agent_model": self.model,
                "user_sim_prompt_version": user_sim_prompt_version,
                "_discovery": None,
                "result": None,
                "eval_mode": eval_mode,
                "query_mode": query_mode,
                "max_asks": max_asks,
                "asks_used": 0,
                "usage": accum,
            }
            _ctx_var.set(ctx_dict)

            prompt = _build_prompt(eval_mode, task_data, budget)

            # DEV-1581 R2: two persistent in-process clients (raw a-interact).
            main_tools = list(MAIN_NATIVE_TOOL_NAMES)
            discovery_tools = list(DISCOVERY_NATIVE_TOOL_NAMES)
            with otf_timer(
                "run_task.create_sdk_mcp_server", instance_id=instance_id,
            ):
                main_mcp_servers = {
                    "bird-interact-tools": build_bird_interact_server(main_tools),
                }
                discovery_mcp_servers = {
                    "bird-interact-tools": build_bird_interact_server(
                        discovery_tools,
                    ),
                }

            pre_submit_gate, post_ask_counter, post_nag = _make_ask_user_guards()

            context_state: dict = {}
            update_wall_clock_start(context_state)
            (
                wall_clock_warning,
                wall_clock_deny,
            ) = make_wall_clock_budget_hook(
                context_state,
                budget_s=per_task_timeout_s(),
                submit_tool="submit_sql",
            )

            # DEV-1579: build each client's options from the policy-owned env
            # kwargs. The agent only supplies its own tool surface / hooks.
            # ask_user lives on BOTH clients.
            def _build_main_options(_opt_kwargs: dict) -> ClaudeAgentOptions:
                return ClaudeAgentOptions(
                    **_opt_kwargs,
                    system_prompt=prompt + build_main_workflow_note(query_mode='raw'),
                    mcp_servers=main_mcp_servers,
                    allowed_tools=list(main_tools),
                    tools=[],
                    setting_sources=[],
                    model=native_model_id(self.model),
                    effort=self.reasoning_effort,
                    max_turns=_MAX_TURNS,
                    hooks={
                        "PreToolUse": [
                            HookMatcher(
                                matcher=_SUBMIT_SQL_TOOL,
                                hooks=[pre_submit_gate],
                            ),
                            HookMatcher(hooks=[wall_clock_deny]),
                        ],
                        "PostToolUse": [
                            HookMatcher(
                                matcher=_ASK_USER_TOOL,
                                hooks=[post_ask_counter],
                            ),
                            HookMatcher(hooks=[post_nag]),
                            HookMatcher(
                                hooks=[_make_turn_budget_hook(
                                    _MAX_TURNS, submit_tool="submit_sql",
                                )],
                            ),
                            HookMatcher(
                                hooks=[
                                    make_context_budget_hook(
                                        context_state,
                                        context_window_for(self.model),
                                    )
                                ]
                            ),
                            HookMatcher(hooks=[wall_clock_warning]),
                        ],
                    },
                )

            def _build_discovery_options(_opt_kwargs: dict) -> ClaudeAgentOptions:
                return ClaudeAgentOptions(
                    **_opt_kwargs,
                    system_prompt=build_discovery_prompt(
                        with_ask_user=True, query_mode="raw"
                    ),
                    mcp_servers=discovery_mcp_servers,
                    allowed_tools=list(discovery_tools),
                    tools=[],
                    setting_sources=[],
                    model=native_model_id(self.model),
                    effort=self.reasoning_effort,
                    max_turns=DISCOVERY_MAX_TURNS,
                    # Discovery's ask_user must increment the SAME shared counter
                    # the main submit_sql gate reads (post_ask_counter closure
                    # shared with _build_main_options), else a discovery-side ask
                    # leaves the main gate closed. Discovery also shares main's
                    # per-task wall-clock guardrails (Codex PR #56).
                    hooks={
                        "PreToolUse": [HookMatcher(hooks=[wall_clock_deny])],
                        "PostToolUse": [
                            HookMatcher(
                                matcher=_ASK_USER_TOOL,
                                hooks=[post_ask_counter],
                            ),
                            HookMatcher(hooks=[wall_clock_warning]),
                        ],
                    },
                )

            # DEV-1561: per-message timing log + otf_timer around the main
            # client's __aenter__ and first query (preserved through
            # run_main_with_discovery). The first-message elapsed is measured
            # from immediately after the query returned (passed as the 3rd arg).
            _msg_timing = {"prev_t": None}

            def _on_main_message(msg, seq, t_after_query):
                now = time.monotonic()
                msg_type = type(msg).__name__
                if seq == 1:
                    log_otf_event(
                        "run_task.sdk_first_message",
                        instance_id=instance_id,
                        msg_type=msg_type,
                        elapsed_s=f"{now - t_after_query:.3f}",
                    )
                else:
                    log_otf_event(
                        "run_task.sdk_message",
                        instance_id=instance_id,
                        seq=seq,
                        msg_type=msg_type,
                        gap_s=f"{now - (_msg_timing['prev_t'] or now):.3f}",
                    )
                _msg_timing["prev_t"] = now

            await run_main_with_discovery(
                model=self.model,
                accum=accum,
                usage_tracker=usage_tracker,
                context_state=context_state,
                main_mcp_servers=main_mcp_servers,
                discovery_mcp_servers=discovery_mcp_servers,
                build_main_options=_build_main_options,
                build_discovery_options=_build_discovery_options,
                initial_query=task_data["amb_user_query"],
                trajectory=trajectory,
                enter_cm_factory=lambda: otf_timer(
                    "run_task.sdk_client_enter", instance_id=instance_id,
                ),
                query_cm_factory=lambda: otf_timer(
                    "run_task.sdk_first_query", instance_id=instance_id,
                ),
                on_main_message=_on_main_message,
            )
            usage_tracker.finalize()
        except Exception as e:
            usage_tracker.finalize()
            logger.error(
                "claude_sdk_otf_ainteract_raw error on %s: %s",
                instance_id, e, exc_info=True,
            )
            result = (ctx_dict or {}).get("result") or {}
            return finalize_result_row(
                {
                    "task_id": instance_id,
                    "instance_id": instance_id,
                    "database": db_name,
                    "phase1_passed": result.get("phase1_passed", False),
                    "phase2_passed": result.get("phase2_passed", False),
                    "total_reward": result.get("total_reward", 0.0),
                    "submitted_sql": result.get("submitted_sql"),
                    "submitted_query": result.get("submitted_query"),
                    "submission_status": result.get("submission_status"),
                    "predicted_result_json": result.get("predicted_result_json"),
                    "gold_result_json": result.get("gold_result_json"),
                    "phase1_observation": result.get("phase1_observation"),
                    "phase2_observation": result.get("phase2_observation"),
                    "trajectory": trajectory,
                    "error": str(e),
                    "usage": {
                        **accum.model_dump(),
                        "n_ask_user_calls": (ctx_dict or {}).get("asks_used", 0),
                    },
                    "phase1_observation_audited": result.get("phase1_observation_audited"),
                    "phase1_observation_original": result.get("phase1_observation_original"),
                },
                deleted_kb_ids=[],
                slayer_storage_dir="",
            )

        result = (ctx_dict or {}).get("result") or {}
        return finalize_result_row(
            {
                "task_id": instance_id,
                "instance_id": instance_id,
                "database": db_name,
                "phase1_passed": result.get("phase1_passed", False),
                "phase2_passed": result.get("phase2_passed", False),
                "total_reward": result.get("total_reward", 0.0),
                "submitted_sql": result.get("submitted_sql"),
                "submitted_query": result.get("submitted_query"),
                "submission_status": result.get("submission_status"),
                "predicted_result_json": result.get("predicted_result_json"),
                "gold_result_json": result.get("gold_result_json"),
                "phase1_observation": result.get("phase1_observation"),
                "phase2_observation": result.get("phase2_observation"),
                "trajectory": trajectory,
                "error": None,
                "usage": {
                    **accum.model_dump(),
                    "n_ask_user_calls": ctx_dict.get("asks_used", 0),
                },
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
            },
            deleted_kb_ids=[],
            slayer_storage_dir="",
        )
