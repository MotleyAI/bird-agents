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

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    create_sdk_mcp_server,
)

from bird_interact_agents.agents.claude_sdk.agent import (
    _ctx_var,
    SdkUsageTracker,
    ask_user,
    execute_sql,
    get_all_column_meanings,
    get_all_external_knowledge_names,
    get_all_knowledge_definitions,
    get_column_meaning,
    get_knowledge_definition,
    get_schema,
    submit_sql,
)
from bird_interact_agents.agents.claude_sdk.sdk_env import (
    hermetic_claude_sdk_session,
)
from bird_interact_agents.slayer_otf.timing import otf_timer
from bird_interact_agents.agents.claude_sdk_otf.agent import (
    _MAX_TURNS,
    _make_turn_budget_hook,
)
from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
    RAW_OTF_AINTERACT,
)
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.provider_registry import is_supported_agent_model
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.harness import (
    SampleStatus,
    _ambiguity_count,
    finalize_result_row,
    load_db_data_if_needed,
    materialize_task_db,
)
from bird_interact_agents.usage import TokenUsage

logger = logging.getLogger(__name__)

# Full MCP tool name of the native ask_user tool — used as the PostToolUse
# counter's matcher and the nag hook's race-skip predicate.
_ASK_USER_TOOL = "mcp__bird-interact-tools__ask_user"

# Full MCP tool name of the raw submission tool — gated by the PreToolUse hook.
_SUBMIT_SQL_TOOL = "mcp__bird-interact-tools__submit_sql"

# How often (in total tool calls without ask_user) the nag fires.
_NAG_EVERY = 10

# All 7 BIRD raw-exploration tools + ask_user + raw submission tool.
_AINTERACT_RAW_TOOLS = [
    execute_sql,
    get_schema,
    get_all_column_meanings,
    get_column_meaning,
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    ask_user,
    submit_sql,
]


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


def _select_tools(eval_mode: str) -> list:
    if eval_mode != "a-interact":
        raise ValueError(
            "claude_sdk_otf_ainteract_raw supports only eval_mode='a-interact'; "
            f"got {eval_mode!r}"
        )
    return list(_AINTERACT_RAW_TOOLS)


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

    Supports Anthropic and registry open-weight models (DEV-1579); an
    unsupported model short-circuits with a skip-shaped row. Bound to
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

        # DEV-1579: this agent now runs Anthropic AND registry open-weight
        # models (the hermetic session layers the provider base-url/auth). A
        # genuinely-unsupported provider (not Anthropic, not in the registry)
        # still gets a graceful skip row.
        if not is_supported_agent_model(self.model):
            msg = (
                f"claude_sdk_otf_ainteract_raw requires an Anthropic or registry "
                f"open-weight model; got {self.model!r}. Skipped — use "
                "--framework pydantic_ai for non-supported models."
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
            load_db_data_if_needed(db_name, data_path_base)
            materialize_task_db(task_data, data_path_base)

            ctx_dict = {
                "status": status,
                "data_path_base": data_path_base,
                "user_sim_model": user_sim_model,
                "user_sim_prompt_version": user_sim_prompt_version,
                "result": None,
                "eval_mode": eval_mode,
                "query_mode": query_mode,
                "max_asks": max_asks,
                "asks_used": 0,
                "usage": accum,
            }
            _ctx_var.set(ctx_dict)

            tools = _select_tools(eval_mode)
            prompt = _build_prompt(eval_mode, task_data, budget)

            server = create_sdk_mcp_server(
                name="bird-interact-tools", version="1.0.0", tools=tools,
            )
            tool_names = [f"mcp__bird-interact-tools__{t.name}" for t in tools]

            mcp_servers: dict = {"bird-interact-tools": server}

            pre_submit_gate, post_ask_counter, post_nag = _make_ask_user_guards()

            # DEV-1579: build the agent's options from the policy-owned env
            # kwargs (telemetry-disable + hermetic CLAUDE_CONFIG_DIR + any
            # registry session env / thinking). The agent only supplies its
            # own tool surface / hooks.
            def _build_options(_opt_kwargs: dict) -> ClaudeAgentOptions:
                return ClaudeAgentOptions(
                    **_opt_kwargs,
                    system_prompt=prompt,
                    mcp_servers=mcp_servers,
                    allowed_tools=tool_names,
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
                        ],
                    },
                )

            # DEV-1579: the hermetic session owns CLAUDE_CONFIG_DIR isolation,
            # API-key auth enforcement, the MCP parity assertion, and cleanup.
            # DEV-1561: wrap `__aenter__` in `otf_timer` (via enter_cm_factory)
            # so a failed initialize handshake still emits `.error elapsed_s=…`.
            async with hermetic_claude_sdk_session(
                self.model,
                mcp_servers=mcp_servers,
                build_options=_build_options,
                enter_cm_factory=lambda: otf_timer(
                    "run_task.sdk_client_enter", instance_id=instance_id,
                ),
            ) as client:
                await client.query(task_data["amb_user_query"])
                async for msg in client.receive_response():
                    trajectory.append(
                        {"type": str(type(msg).__name__), "data": str(msg)[:500]}
                    )
                    usage_tracker.observe(msg)
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
