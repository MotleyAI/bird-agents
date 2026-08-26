"""Claude Agent SDK cube-mode agent (livesqlbench / one-shot, postgres, local).

A counterpart to ``claude_sdk_otf_raw`` that answers via the Cube.js REST API
instead of raw SQL: the agent explores cubes with ``cube_meta`` / ``cube_load``
/ ``cube_sql`` (plus read-only schema + KB tools) and submits a final Cube query
through ``submit_cube_query``, which compiles it to SQL for grading. Bound to a
postgres one-shot benchmark, ``--query-mode cube``, ``--mode one-shot``.
"""

from __future__ import annotations

import logging
import os

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    create_sdk_mcp_server,
)

from bird_interact_agents.agents.claude_sdk.agent import (
    _ctx_var,
    SdkUsageTracker,
    cube_load,
    cube_meta,
    cube_sql,
    get_all_column_meanings,
    get_all_external_knowledge_names,
    get_all_knowledge_definitions,
    get_column_meaning,
    get_knowledge_definition,
    get_schema,
    submit_cube_query,
)
from bird_interact_agents.agents.claude_sdk.sdk_env import (
    hermetic_claude_sdk_session,
    serialize_sdk_message,
)
from bird_interact_agents.slayer_otf.timing import otf_timer
from bird_interact_agents.agents.claude_sdk_otf.agent import (
    _MAX_TURNS,
    _make_turn_budget_hook as _make_turn_budget_hook_base,
)
from bird_interact_agents.agents.claude_sdk_otf_cube.prompts import CUBE_ONE_SHOT
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.cube_local.client import CubeClient
from bird_interact_agents.provider_registry import is_supported_agent_model
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.harness import (
    SampleStatus,
    finalize_result_row,
    load_db_data_if_needed,
    materialize_task_db,
)
from bird_interact_agents.usage import TokenUsage

logger = logging.getLogger(__name__)


# Cube exploration tools + read-only docs tools + the cube submission tool.
# No execute_sql, no ask_user (one-shot).
_CUBE_TOOLS = [
    cube_meta,
    cube_load,
    cube_sql,
    get_schema,
    get_all_column_meanings,
    get_column_meaning,
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    submit_cube_query,
]


class ClaudeSDKOtfCubeAgent:
    """SystemAgent: Claude SDK cube-mode agent.

    Anthropic + registry open-weight models (DEV-1579). Bound to a postgres
    one-shot benchmark, ``--query-mode cube``, ``--mode one-shot``. Reaches Cube
    via ``BIRD_CUBE_URL`` / ``BIRD_CUBE_API_SECRET`` (set by the local cube
    bootstrap). Mismatched query_mode / eval_mode is rejected at the boundary.
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
        if query_mode != "cube":
            raise ValueError(
                f"claude_sdk_otf_cube supports only --query-mode cube; got {query_mode!r}"
            )
        if eval_mode != "one-shot":
            raise ValueError(
                f"claude_sdk_otf_cube supports only --mode one-shot; got {eval_mode!r}"
            )

        dataset = task_data.get("dataset")
        if not dataset:
            raise ValueError("task_data missing required 'dataset' field")
        if not get_benchmark(dataset).one_shot:
            raise ValueError(
                f"claude_sdk_otf_cube requires a one-shot benchmark; got dataset={dataset!r}"
            )

        instance_id = task_data["instance_id"]
        db_name = task_data["selected_database"]

        if not is_supported_agent_model(self.model):
            msg = (
                f"claude_sdk_otf_cube requires an Anthropic or registry "
                f"open-weight model; got {self.model!r}. Skipped."
            )
            logger.warning("[%s] %s", instance_id, msg)
            return finalize_result_row(
                {
                    "task_id": instance_id, "instance_id": instance_id,
                    "database": db_name, "phase1_passed": False,
                    "phase2_passed": False, "total_reward": 0.0,
                    "trajectory": [], "error": msg,
                },
                deleted_kb_ids=[], slayer_storage_dir="",
            )

        cube_url = os.environ.get("BIRD_CUBE_URL")
        cube_secret = os.environ.get("BIRD_CUBE_API_SECRET")
        if not cube_url or not cube_secret:
            raise RuntimeError(
                "cube mode requires BIRD_CUBE_URL / BIRD_CUBE_API_SECRET "
                "(exported by the local cube bootstrap)."
            )

        status = SampleStatus(
            idx=0, original_data=task_data,
            remaining_budget=budget, total_budget=budget,
        )
        accum = TokenUsage()
        usage_tracker = SdkUsageTracker(accum, self.model)
        trajectory: list[dict] = []
        ctx_dict: dict | None = None
        try:
            load_db_data_if_needed(db_name, data_path_base)
            materialize_task_db(task_data, data_path_base)

            # The JWT db claim is set by us to the task's own DB (never by the
            # agent), so the tenant is allow-listed by construction (Codex C10).
            cube_client = CubeClient(cube_url, cube_secret, db_name)

            ctx_dict = {
                "status": status,
                "data_path_base": data_path_base,
                "user_sim_model": user_sim_model,
                "agent_model": self.model,
                "user_sim_prompt_version": user_sim_prompt_version,
                "cube": cube_client,
                "result": None,
                "eval_mode": eval_mode,
                "query_mode": query_mode,
                "usage": accum,
            }
            _ctx_var.set(ctx_dict)

            prompt = CUBE_ONE_SHOT.format(
                budget=budget, db_name=db_name,
                user_query=task_data["amb_user_query"],
            )

            tools = list(_CUBE_TOOLS)
            server = create_sdk_mcp_server(
                name="bird-interact-tools", version="1.0.0", tools=tools,
            )
            tool_names = [f"mcp__bird-interact-tools__{t.name}" for t in tools]
            mcp_servers: dict = {"bird-interact-tools": server}

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
                        "PostToolUse": [
                            HookMatcher(hooks=[_make_turn_budget_hook_base(
                                _MAX_TURNS, submit_tool="submit_cube_query",
                            )]),
                        ],
                    },
                )

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
                    trajectory.append(serialize_sdk_message(msg, truncate=500))
                    usage_tracker.observe(msg)
            usage_tracker.finalize()
        except Exception as e:
            usage_tracker.finalize()
            logger.error("claude_sdk_otf_cube error on %s: %s", instance_id, e, exc_info=True)
            result = (ctx_dict or {}).get("result") or {}
            return self._row(instance_id, db_name, result, trajectory, accum, ctx_dict, error=str(e))

        result = (ctx_dict or {}).get("result") or {}
        return self._row(instance_id, db_name, result, trajectory, accum, ctx_dict, error=None)

    @staticmethod
    def _row(instance_id, db_name, result, trajectory, accum, ctx_dict, *, error):
        return finalize_result_row(
            {
                "task_id": instance_id, "instance_id": instance_id,
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
                "error": error,
                "usage": {
                    **accum.model_dump(),
                    "n_ask_user_calls": (ctx_dict or {}).get("asks_used", 0),
                },
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
            },
            deleted_kb_ids=[], slayer_storage_dir="",
        )
