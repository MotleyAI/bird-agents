"""mcp-agent (LastMile AI) implementation for BIRD-Interact.

mcp-agent is MCP-native: each agent is configured with a list of MCP
server names, and the SDK wires them up automatically. We use this to
attach the slayer MCP server in slayer mode without wrapping any of its
tools ourselves.

Native tools (`ask_user`, `submit_sql`, `submit_query`) are plain async
Python functions passed via `Agent.functions`. Bodies live in
`bird_interact_agents.agents._submit`; this module supplies the
mcp-agent-specific async wrappers.
"""

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bird_interact_agents.agents._prompt_builders import (
    build_raw_c_interact_prompt,
    build_slayer_c_interact_prompt,
)
from bird_interact_agents.agents._submit import (
    ask_user_impl,
    run_env_action,
    submit_raw_sql,
    submit_slayer_query,
)
from bird_interact_agents.agents._tool_specs import BIRD_INTERACT_TOOLS
from bird_interact_agents.agents.claude_sdk.prompts import (
    RAW_A_INTERACT,
    SLAYER_A_INTERACT,
)
from bird_interact_agents.harness import (
    SampleStatus,
    _schema_cache,
    build_user_decoder_prompt,
    build_user_encoder_prompt,
    execute_env_action,
    execute_submit_action,
    finalize_result_row,
    load_db_data_if_needed,
    parse_encoder_response,
    resolve_task_storage_dir,
    slayer_mcp_stdio_config,
)
from bird_interact_agents.usage import TokenUsage


_BY_NAME = {t.name: t for t in BIRD_INTERACT_TOOLS}

logger = logging.getLogger(__name__)


class TaskState(BaseModel):
    """Per-task state, closed over by the native tool functions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: SampleStatus
    data_path_base: str
    user_sim_model: str
    # DEV-1613: agent model for the in-task N5 insufficient-task judge.
    agent_model: str | None = None
    user_sim_prompt_version: str
    slayer_storage_dir: str = ""
    result: dict | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    _slayer_client: Any = None
    _slayer_storage: Any = None


def _slayer_client(state: TaskState):
    if state._slayer_client is None:
        from slayer.client.slayer_client import SlayerClient
        from slayer.storage.yaml_storage import YAMLStorage

        storage = YAMLStorage(base_dir=state.slayer_storage_dir)
        state._slayer_client = SlayerClient(storage=storage)
        state._slayer_storage = storage
    return state._slayer_client


def _build_native_functions(
    state: TaskState, query_mode: str, eval_mode: str = "a-interact",
) -> list:
    """Construct the native tool functions, closing over per-task state.

    Each wrapper is a one-liner over the shared helpers in `_submit`,
    which apply budget gating + bookkeeping.
    """

    async def ask_user(question: str) -> str:
        """Ask the user a clarification question about their query."""
        return await ask_user_impl(state, question, query_mode)

    async def execute_sql(sql: str) -> str:
        """Execute a SQL query against the database and return results."""
        return run_env_action(state, _BY_NAME["execute_sql"], query_mode, sql=sql)

    async def get_schema() -> str:
        """Get the database schema (CREATE TABLE statements with sample data)."""
        return run_env_action(state, _BY_NAME["get_schema"], query_mode)

    async def get_all_column_meanings() -> str:
        """Get the meanings/descriptions of all columns in the database."""
        return run_env_action(state, _BY_NAME["get_all_column_meanings"], query_mode)

    async def get_column_meaning(table_name: str, column_name: str) -> str:
        """Get the meaning of a specific column in a table."""
        return run_env_action(
            state, _BY_NAME["get_column_meaning"], query_mode,
            table_name=table_name, column_name=column_name,
        )

    async def get_all_external_knowledge_names() -> str:
        """List all available external knowledge entry names for this database."""
        return run_env_action(
            state, _BY_NAME["get_all_external_knowledge_names"], query_mode,
        )

    async def get_knowledge_definition(knowledge_name: str) -> str:
        """Get the definition of a specific external knowledge entry."""
        return run_env_action(
            state, _BY_NAME["get_knowledge_definition"], query_mode,
            knowledge_name=knowledge_name,
        )

    async def get_all_knowledge_definitions() -> str:
        """Get all external knowledge definitions for this database."""
        return run_env_action(
            state, _BY_NAME["get_all_knowledge_definitions"], query_mode,
        )

    async def submit_sql(sql: str) -> str:
        """Submit your final SQL query for evaluation. Only submit when confident."""
        return submit_raw_sql(state, sql)

    async def submit_query(query_json: str) -> str:
        """Submit your final SLayer query for evaluation.

        `query_json` is a JSON string whose top-level value is either a
        single SlayerQuery object (e.g. `{"source_model": "orders",
        "measures": ["amount:sum"]}`) or a nested-DAG array of stage
        objects (same shape `query_nested` accepts; last element is the
        DAG root). The chosen query is translated to SQL deterministically
        and tested against ground truth.
        """
        return submit_slayer_query(state, query_json, _slayer_client)

    if query_mode == "raw" and eval_mode == "c-interact":
        return [ask_user, submit_sql]
    if query_mode == "raw":
        return [
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
    return [ask_user, submit_query]  # slayer mode — slayer MCP provides discovery


async def _build_prompt(
    query_mode: str, eval_mode: str, task_data: dict, budget: float, state: TaskState
) -> str:
    user_query = task_data["amb_user_query"]
    db_name = task_data["selected_database"]

    if query_mode == "raw" and eval_mode == "a-interact":
        return RAW_A_INTERACT.format(
            budget=budget, db_name=db_name, user_query=user_query
        )
    if query_mode == "raw" and eval_mode == "c-interact":
        return await build_raw_c_interact_prompt(
            budget=budget,
            db_name=db_name,
            user_query=user_query,
            task_data=task_data,
        )
    if query_mode == "slayer" and eval_mode == "a-interact":
        return SLAYER_A_INTERACT.format(budget=budget, user_query=user_query)
    if query_mode == "slayer" and eval_mode == "c-interact":
        return await build_slayer_c_interact_prompt(
            budget=budget,
            user_query=user_query,
            slayer_storage_dir=state.slayer_storage_dir,
            db_name=db_name,
            task_data=task_data,
        )
    raise ValueError(f"Unknown mode combo: {query_mode}/{eval_mode}")


def _build_settings(query_mode: str, slayer_storage_dir: str, model: str):
    """Build mcp-agent Settings, registering the slayer MCP server in slayer mode.

    For non-Anthropic models we point mcp-agent's OpenAI-compatible client
    at the appropriate provider's base URL + key. mcp-agent ships only
    AnthropicAugmentedLLM and OpenAIAugmentedLLM (no LiteLLM shim), so we
    route every non-Anthropic provider through OpenAIAugmentedLLM with a
    custom base URL.
    """
    import os
    from mcp_agent.config import (
        MCPServerSettings, MCPSettings, OpenAISettings, Settings,
    )

    from bird_interact_agents.model_string import is_anthropic, native_model_id

    servers: dict = {}
    if query_mode == "slayer":
        cfg = slayer_mcp_stdio_config(slayer_storage_dir)
        servers["slayer"] = MCPServerSettings(
            name="slayer",
            transport="stdio",
            command=cfg["command"],
            args=cfg["args"],
            env=cfg["env"],
        )

    settings_kwargs: dict = {"mcp": MCPSettings(servers=servers)}
    if not is_anthropic(model):
        provider = model.split("/", 1)[0]
        # Map LiteLLM provider name -> OpenAI-compat base URL + env var.
        # Add entries here as needed; the table is intentionally short
        # because most experiments will go through cerebras/openrouter.
        endpoints = {
            "cerebras":    ("https://api.cerebras.ai/v1",            "CEREBRAS_API_KEY"),
            "openrouter":  ("https://openrouter.ai/api/v1",          "OPENROUTER_API_KEY"),
            "zhipu":       ("https://api.z.ai/api/paas/v4",          "ZHIPU_API_KEY"),
            "fireworks_ai":("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
        }
        base_url, env_key = endpoints.get(
            provider, ("https://api.openai.com/v1", "OPENAI_API_KEY")
        )
        settings_kwargs["openai"] = OpenAISettings(
            base_url=base_url,
            api_key=os.environ.get(env_key),
            default_model=native_model_id(model),
        )
    return Settings(**settings_kwargs)


class McpAgentAgent:
    """SystemAgent implementation using lastmile-ai/mcp-agent."""

    def __init__(
        self,
        slayer_storage_root: str | None = None,
        model: str = "anthropic/claude-sonnet-4-5",
        strict: bool = False,
    ) -> None:
        self.slayer_storage_root = slayer_storage_root
        # Canonical LiteLLM-style "provider/model_id". For Anthropic, mcp-agent
        # uses AnthropicAugmentedLLM and the bare model id; for everything
        # else we use OpenAIAugmentedLLM pointed at the provider's base URL
        # (configured in _build_settings).
        self.model = model
        # mcp-agent's OpenAIAugmentedLLM builds tool definitions inline
        # inside its `generate` method (no override hook for individual
        # tools). There's a TODO in the upstream source about exposing a
        # `strict` knob; until that lands, --strict True isn't honourable
        # for this framework, so fail fast at construction.
        if strict:
            from bird_interact_agents.strict import warn_unsupported

            warn_unsupported("mcp_agent")
        self.strict = strict

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
        from mcp_agent.agents.agent import Agent
        from mcp_agent.app import MCPApp
        from mcp_agent.workflows.llm.augmented_llm import RequestParams

        from bird_interact_agents.model_string import (
            is_anthropic, native_model_id,
        )

        if is_anthropic(self.model):
            from mcp_agent.workflows.llm.augmented_llm_anthropic import (
                AnthropicAugmentedLLM as _LLM,
            )
            request_model = native_model_id(self.model)
        else:
            from mcp_agent.workflows.llm.augmented_llm_openai import (
                OpenAIAugmentedLLM as _LLM,
            )
            request_model = native_model_id(self.model)

        db_name = task_data["selected_database"]
        instance_id = task_data["instance_id"]
        load_db_data_if_needed(db_name, data_path_base)

        slayer_storage_dir, deleted_kb_ids = await resolve_task_storage_dir(
            slayer_storage_root=self.slayer_storage_root,
            db_name=db_name,
            task_data=task_data,
            query_mode=query_mode,
        )
        state = TaskState(
            status=SampleStatus(
                idx=0, original_data=task_data,
                remaining_budget=budget, total_budget=budget,
            ),
            data_path_base=data_path_base,
            user_sim_model=user_sim_model,
            agent_model=self.model,
            user_sim_prompt_version=user_sim_prompt_version,
            slayer_storage_dir=slayer_storage_dir,
        )

        functions = _build_native_functions(state, query_mode, eval_mode)
        prompt = await _build_prompt(query_mode, eval_mode, task_data, budget, state)
        server_names = ["slayer"] if query_mode == "slayer" else []

        settings = _build_settings(query_mode, slayer_storage_dir, self.model)
        app = MCPApp(name="bird-interact-mcp-agent", settings=settings)

        # mcp-agent's SDK does not expose per-call token counts — we still
        # capture user-sim usage, but flag the whole run partial so the
        # comparison report can footnote it instead of fabricating
        # agent-leg numbers.
        state.usage.partial = True

        try:
            async with app.run():
                agent = Agent(
                    name="bird_interact_agent",
                    instruction=prompt,
                    server_names=server_names,
                    functions=functions,
                )
                async with agent:
                    llm = await agent.attach_llm(_LLM)
                    output = await llm.generate_str(
                        message=task_data["amb_user_query"],
                        request_params=RequestParams(model=request_model),
                    )
        except Exception as e:
            logger.error("Agent error on %s: %s", instance_id, e)
            return finalize_result_row(
                {
                    "task_id": instance_id, "instance_id": instance_id,
                    "database": db_name,
                    "phase1_passed": False, "phase2_passed": False,
                    "total_reward": 0.0, "trajectory": [],
                    "error": str(e),
                    "usage": state.usage.model_dump(),
                },
                deleted_kb_ids=deleted_kb_ids,
                slayer_storage_dir=slayer_storage_dir,
            )

        result = state.result or {}
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
                "trajectory": [{"final_output": str(output)[:500]}],
                "error": None,
                "usage": state.usage.model_dump(),
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
            },
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
        )
