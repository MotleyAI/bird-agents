"""Claude Agent SDK raw-SQL OTF agent (livesqlbench / one-shot flavor).

A counterpart to ``claude_sdk_otf`` that uses no SLayer at all — the agent
issues raw SQL via the bird-interact tool suite and submits via ``submit_sql``.
Bound to ``--dataset livesqlbench --mode one-shot --query-mode raw``.

Prompt structure mirrors ``claude_sdk_otf`` through shared ``_shared_otf_prompts``
constants; the exploration + test + mutation-check discipline is the same, but
operates on raw SQL instead of a SLayer model store.
"""

from __future__ import annotations

import logging

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    create_sdk_mcp_server,
)

from bird_interact_agents.agents.claude_sdk.agent import (
    _ctx_var,
    SdkUsageTracker,
    accumulate_assistant_usage,
    execute_sql,
    get_all_column_meanings,
    get_all_external_knowledge_names,
    get_all_knowledge_definitions,
    get_column_meaning,
    get_knowledge_definition,
    get_schema,
    submit_sql,
)
from bird_interact_agents.agents.claude_sdk.context_budget import (
    context_window_for,
    make_context_budget_hook,
    make_wall_clock_budget_hook,
    per_task_timeout_s,
    update_context_tokens,
    update_wall_clock_start,
)
from bird_interact_agents.agents.claude_sdk.partition import (
    DISCOVERY_AGENT_NAME,
    DISCOVERY_MAX_TURNS,
    MAIN_WORKFLOW_NOTE,
    build_discovery_prompt,
    make_partition_deny_hook,
)
from bird_interact_agents.agents.claude_sdk.sdk_env import (
    disable_cli_telemetry_env,
)
from bird_interact_agents.agents.claude_sdk_otf_v1.agent import (
    _MAX_TURNS,
    _TURN_BUDGET_WARN_WITHIN,
    _make_turn_budget_hook as _make_turn_budget_hook_base,
)
from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import RAW_OTF_ONE_SHOT
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.provider_registry import (
    get_provider,
    is_supported_agent_model,
    requires_thinking,
    sdk_session_env,
)
from bird_interact_agents.harness import (
    SampleStatus,
    finalize_result_row,
    load_db_data_if_needed,
    materialize_task_db,
)
from bird_interact_agents.usage import TokenUsage

logger = logging.getLogger(__name__)


def _make_turn_budget_hook(
    max_turns: int,
    warn_within: int = _TURN_BUDGET_WARN_WITHIN,
    submit_tool: str = "submit_sql",
):
    """Thin wrapper around the shared hook; defaults ``submit_tool`` to
    ``submit_sql`` (raw agents never have ``submit_query``).
    """
    return _make_turn_budget_hook_base(max_turns, warn_within, submit_tool)


# All 7 BIRD raw-exploration tools + the raw submission tool.
_RAW_TOOLS = [
    execute_sql,
    get_schema,
    get_all_column_meanings,
    get_column_meaning,
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    submit_sql,
]


def _select_tools(eval_mode: str) -> list:
    if eval_mode != "one-shot":
        raise ValueError(
            "claude_sdk_otf_raw supports only eval_mode='one-shot'; "
            f"got {eval_mode!r}"
        )
    return list(_RAW_TOOLS)


# DEV-1555: discovery/main tool partition (raw flavor). Discovery owns
# schema/column/KB introspection plus execute_sql for data profiling; the
# main loop keeps execute_sql (candidate verification) and submit_sql.
_RAW_PREFIX = "mcp__bird-interact-tools__"

DISCOVERY_TOOLS = [
    f"{_RAW_PREFIX}get_schema",
    f"{_RAW_PREFIX}get_all_column_meanings",
    f"{_RAW_PREFIX}get_column_meaning",
    f"{_RAW_PREFIX}get_all_external_knowledge_names",
    f"{_RAW_PREFIX}get_knowledge_definition",
    f"{_RAW_PREFIX}get_all_knowledge_definitions",
    f"{_RAW_PREFIX}execute_sql",
]

MAIN_TOOLS = [
    "Task",
    f"{_RAW_PREFIX}execute_sql",
    f"{_RAW_PREFIX}submit_sql",
]


def _build_prompt(eval_mode: str, task_data: dict, budget: float) -> str:
    if eval_mode != "one-shot":
        raise ValueError(
            "claude_sdk_otf_raw supports only eval_mode='one-shot'; "
            f"got {eval_mode!r}"
        )
    return RAW_OTF_ONE_SHOT.format(
        budget=budget,
        db_name=task_data["selected_database"],
        user_query=task_data["amb_user_query"],
    )


class ClaudeSDKOtfRawAgent:
    """SystemAgent: Claude SDK raw-SQL OTF agent.

    Anthropic-only (the SDK is locked to Anthropic). Bound to
    ``--dataset livesqlbench --mode one-shot --query-mode raw``. No SLayer
    MCP server, no slayer_setup requirement. Mismatched dataset, eval_mode,
    or query_mode is rejected at the agent boundary.
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
        eval_mode: str = "one-shot",
        user_sim_model: str = "anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version: str = "v2",
    ) -> dict:
        if query_mode != "raw":
            raise ValueError(
                "claude_sdk_otf_raw supports only --query-mode raw; "
                f"got {query_mode!r}"
            )
        if eval_mode != "one-shot":
            raise ValueError(
                "claude_sdk_otf_raw supports only --mode one-shot; "
                f"got {eval_mode!r}"
            )

        dataset = task_data.get("dataset")
        if not dataset:
            raise ValueError("task_data missing required 'dataset' field")
        if not get_benchmark(dataset).one_shot:
            raise ValueError(
                "claude_sdk_otf_raw requires a one-shot benchmark; "
                f"got dataset={dataset!r}"
            )

        instance_id = task_data["instance_id"]
        db_name = task_data["selected_database"]

        if not is_supported_agent_model(self.model):
            msg = (
                f"claude_sdk_otf_raw requires an Anthropic or registry open-weight "
                f"model; got {self.model!r}. "
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
                "usage": accum,
            }
            _ctx_var.set(ctx_dict)

            tools = _select_tools(eval_mode)
            prompt = _build_prompt(eval_mode, task_data, budget)

            server = create_sdk_mcp_server(
                name="bird-interact-tools", version="1.0.0", tools=tools,
            )

            # DEV-1555: discovery/main split (see claude_sdk_otf.agent).
            discovery_only = sorted(set(DISCOVERY_TOOLS) - set(MAIN_TOOLS))
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

            # DEV-1555 Stage 2: registry open-weight backends get a
            # per-run session env (ANTHROPIC_BASE_URL + auth token);
            # anthropic models keep the SDK defaults untouched.
            # DEV-1561: always layer in the disable-CLI-telemetry vars so
            # the bundled `claude` Node binary doesn't burn 5-10 min on
            # outbound telemetry / error-reporting / auto-updater calls
            # during the initialize handshake.
            _session_env_kwargs: dict = {"env": disable_cli_telemetry_env()}
            if get_provider(self.model) is not None:
                _session_env_kwargs["env"].update(sdk_session_env(self.model))
                if requires_thinking(self.model):
                    # Probed live: e.g. kimi-k2.7-code rejects requests
                    # without thinking enabled.
                    _session_env_kwargs["thinking"] = {
                        "type": "enabled", "budget_tokens": 8192,
                    }

            options = ClaudeAgentOptions(
                **_session_env_kwargs,
                system_prompt=prompt + MAIN_WORKFLOW_NOTE,
                mcp_servers={"bird-interact-tools": server},
                allowed_tools=sorted(set(MAIN_TOOLS) | set(DISCOVERY_TOOLS)),
                tools=["Task"],
                setting_sources=[],
                agents={
                    DISCOVERY_AGENT_NAME: AgentDefinition(
                        description=(
                            "Schema/data introspection for the current task; "
                            "returns a structured handoff report."
                        ),
                        prompt=build_discovery_prompt(with_ask_user=False),
                        tools=list(DISCOVERY_TOOLS),
                        maxTurns=DISCOVERY_MAX_TURNS,
                    ),
                },
                model=native_model_id(self.model),
                effort=self.reasoning_effort,
                max_turns=_MAX_TURNS,
                hooks={
                    "PreToolUse": [
                        HookMatcher(
                            matcher="|".join(discovery_only),
                            hooks=[make_partition_deny_hook(discovery_only)],
                        ),
                        HookMatcher(hooks=[wall_clock_deny]),
                    ],
                    "PostToolUse": [
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

            async with ClaudeSDKClient(options=options) as client:
                await client.query(task_data["amb_user_query"])
                async for msg in client.receive_response():
                    trajectory.append(
                        {"type": str(type(msg).__name__), "data": str(msg)[:500]}
                    )
                    usage_tracker.observe(msg)
                    update_context_tokens(context_state, msg)
            usage_tracker.finalize()
        except Exception as e:
            usage_tracker.finalize()
            logger.error(
                "claude_sdk_otf_raw error on %s: %s",
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
                    # DEV-1535 follow-up: see claude_sdk_otf/agent.py
                    # — symmetry with the a-interact variant.
                    "usage": {
                        **accum.model_dump(),
                        "n_ask_user_calls": (ctx_dict or {}).get(
                            "asks_used", 0,
                        ),
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
                    "n_ask_user_calls": (ctx_dict or {}).get(
                        "asks_used", 0,
                    ),
                },
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
            },
            deleted_kb_ids=[],
            slayer_storage_dir="",
        )
