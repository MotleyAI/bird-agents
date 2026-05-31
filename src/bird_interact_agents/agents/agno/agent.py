"""Agno (formerly Phidata) implementation for BIRD-Interact.

In SLayer mode the slayer MCP server is attached via `MCPTools` (Agno's
stdio MCP toolkit). Native tools are plain async closures over per-task
state — same pattern as the mcp-agent adapter.

Tool *bodies* live in `bird_interact_agents.agents._submit`; this
module just supplies framework-specific async wrappers.
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


def _build_native_tools(
    state: TaskState, query_mode: str, eval_mode: str = "a-interact",
) -> list:
    """Construct the native tool functions, closing over per-task state.

    Each tool is a plain async function — Agno auto-converts these into
    tool definitions from the docstring + signature. Bodies are
    one-liners over the shared helpers in `_submit`, which apply
    budget gating + bookkeeping.
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
    return [ask_user, submit_query]


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


def _build_strict_litellm_class():
    """Subclass `agno.models.litellm.LiteLLM` so we can mutate the outgoing
    `tools` payload in `get_request_params`. agno's stock LiteLLM never
    writes `strict`; this is the surgical place to inject a uniform value.
    """
    from agno.models.litellm import LiteLLM

    class _StrictLiteLLM(LiteLLM):
        def __init__(self, *a, _strict_value: bool, **kw):
            super().__init__(*a, **kw)
            self._strict_value = _strict_value

        def get_request_params(self, tools=None):
            params = super().get_request_params(tools=tools)
            for t in params.get("tools") or []:
                t.setdefault("function", {})["strict"] = self._strict_value
            return params

    return _StrictLiteLLM


class AgnoAgent:
    """SystemAgent implementation using Agno."""

    def __init__(
        self,
        slayer_storage_root: str | None = None,
        model_id: str = "anthropic/claude-sonnet-4-5",
        strict: bool = False,
    ) -> None:
        self.slayer_storage_root = slayer_storage_root
        # Canonical LiteLLM-style "provider/model_id". For Anthropic we
        # construct an `agno.models.anthropic.Claude(id=...)` with the
        # bare model id; for everything else we use Agno's LiteLLM shim
        # which accepts the prefixed string verbatim.
        self.model_id = model_id
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
        from agno.agent import Agent
        from agno.tools.mcp import MCPTools

        from bird_interact_agents.model_string import is_anthropic, native_model_id

        if is_anthropic(self.model_id):
            from agno.models.anthropic import Claude
            # Anthropic doesn't have a tool-level strict concept; --strict
            # is silently a no-op for the anthropic/* path.
            agno_model = Claude(id=native_model_id(self.model_id))
        else:
            StrictLiteLLM = _build_strict_litellm_class()
            agno_model = StrictLiteLLM(id=self.model_id, _strict_value=self.strict)

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
            user_sim_prompt_version=user_sim_prompt_version,
            slayer_storage_dir=slayer_storage_dir,
        )

        native_tools = _build_native_tools(state, query_mode, eval_mode)
        prompt = await _build_prompt(query_mode, eval_mode, task_data, budget, state)

        async def _run_with_tools(tools_list: list) -> str:
            agent = Agent(
                model=agno_model,
                tools=tools_list,
                instructions=prompt,
            )
            response = await agent.arun(task_data["amb_user_query"])
            metrics = getattr(response, "metrics", None) if response else None
            if metrics is not None:
                state.usage.add_call(
                    scope="agent",
                    model=self.model_id,
                    prompt=getattr(metrics, "input_tokens", 0) or 0,
                    completion=getattr(metrics, "output_tokens", 0) or 0,
                )
            return str(response.content) if response and response.content else ""

        try:
            if query_mode == "slayer":
                cfg = slayer_mcp_stdio_config(slayer_storage_dir)
                async with MCPTools(
                    command=cfg["command"] + " " + " ".join(cfg["args"]),
                    env=cfg["env"],
                ) as slayer_mcp:
                    output = await _run_with_tools([slayer_mcp, *native_tools])
            else:
                output = await _run_with_tools(native_tools)
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
                "trajectory": [{"final_output": output[:500]}],
                "error": None,
                "usage": state.usage.model_dump(),
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
            },
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
        )
