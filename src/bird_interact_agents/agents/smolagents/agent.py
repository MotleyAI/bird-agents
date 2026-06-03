"""smolagents (HuggingFace) implementation for BIRD-Interact.

smolagents' agent loop is synchronous — we run it inside `asyncio.to_thread`
so the rest of our async runner is unaffected. Native tools are defined
with the smolagents `@tool` decorator and close over a per-task state
holder, making concurrent task runs safe (each task gets its own state).

In SLayer mode the slayer MCP server is attached via smolagents' MCPClient,
which exposes the server's tools as native smolagents Tool instances.

Tool *bodies* live in `bird_interact_agents.agents._submit`; this
module supplies the smolagents-specific @tool wrappers + a per-call
`_ensure_thread_safe` shim that scrubs the harness's thread-bound
sqlite3 connection cache before any tool that touches the DB.
"""

import asyncio
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
    close_db_connection,
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


def _ensure_thread_safe(db_name: str, data_path_base: str) -> None:
    """Make sure the harness's SQLite connection cache has no entry for
    this DB before we run a tool, so the current thread will open a fresh
    connection.

    The harness caches sqlite3 connections per db_path module-globally.
    smolagents tool calls hop between threads, and sqlite3 raises
    "SQLite objects created in a thread can only be used in that same
    thread" when one thread tries to use (or close!) a connection that
    another thread opened.

    `close_db_connection` calls `.close()` on the cached object first,
    which itself raises cross-thread and leaves the dict entries intact.
    We bypass that and just `del` the cache entries, leaking the old
    connection (it gets GC'd when its thread ends).
    """
    from bird_interact_agents import harness as _h

    db_path = f"{data_path_base}/{db_name}/{db_name}.sqlite"
    # Reach into the BIRD-Interact action_handler module's caches.
    import batch_run_bird_interact.action_handler_sqlite as _ah  # type: ignore[import-not-found]
    _ah._db_connections.pop(db_path, None)
    _ah._db_cursors.pop(db_path, None)
    _ah._db_configs.pop(db_path, None)
    # Silence unused-import warning for the harness alias.
    _ = _h, close_db_connection

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
    """Construct the native smolagents tools, closing over per-task state.

    smolagents' @tool decorator wraps a regular sync function. We use
    asyncio.run() inside ask_user because the smolagents agent loop runs
    in a worker thread that has no active event loop. Tools that touch
    the SQLite DB also call `_ensure_thread_safe` first so the harness's
    thread-bound connection cache is scrubbed. Bodies route through
    `_submit` which applies budget gating + bookkeeping.
    """
    from smolagents import tool

    db_name = state.status.original_data["selected_database"]

    @tool
    def ask_user(question: str) -> str:
        """Ask the user a clarification question about their query.

        Args:
            question: The clarification question to ask.
        """
        return asyncio.run(ask_user_impl(state, question, query_mode))

    @tool
    def execute_sql(sql: str) -> str:
        """Execute a SQL query against the database and return results.

        Args:
            sql: The SQL query to execute.
        """
        _ensure_thread_safe(db_name, state.data_path_base)
        return run_env_action(state, _BY_NAME["execute_sql"], query_mode, sql=sql)

    @tool
    def get_schema() -> str:
        """Get the database schema (CREATE TABLE statements with sample data)."""
        return run_env_action(state, _BY_NAME["get_schema"], query_mode)

    @tool
    def get_all_column_meanings() -> str:
        """Get the meanings/descriptions of all columns in the database."""
        return run_env_action(state, _BY_NAME["get_all_column_meanings"], query_mode)

    @tool
    def get_column_meaning(table_name: str, column_name: str) -> str:
        """Get the meaning of a specific column in a table.

        Args:
            table_name: The table containing the column.
            column_name: The column name.
        """
        return run_env_action(
            state, _BY_NAME["get_column_meaning"], query_mode,
            table_name=table_name, column_name=column_name,
        )

    @tool
    def get_all_external_knowledge_names() -> str:
        """List all available external knowledge entry names for this database."""
        return run_env_action(
            state, _BY_NAME["get_all_external_knowledge_names"], query_mode,
        )

    @tool
    def get_knowledge_definition(knowledge_name: str) -> str:
        """Get the definition of a specific external knowledge entry.

        Args:
            knowledge_name: The knowledge entry name.
        """
        return run_env_action(
            state, _BY_NAME["get_knowledge_definition"], query_mode,
            knowledge_name=knowledge_name,
        )

    @tool
    def get_all_knowledge_definitions() -> str:
        """Get all external knowledge definitions for this database."""
        return run_env_action(
            state, _BY_NAME["get_all_knowledge_definitions"], query_mode,
        )

    @tool
    def submit_sql(sql: str) -> str:
        """Submit your final SQL query for evaluation. Only submit when confident.

        Args:
            sql: The final SQL query to submit.
        """
        _ensure_thread_safe(db_name, state.data_path_base)
        return submit_raw_sql(state, sql)

    @tool
    def submit_query(query_json: str) -> str:
        """Submit your final SLayer query for evaluation. Translates to SQL deterministically and tests against the ground truth.

        Args:
            query_json: A JSON string whose top-level value is either a
                single SlayerQuery object, e.g.
                '{"source_model": "orders", "dimensions": ["status"],
                "measures": ["amount:sum"], "limit": 10}', OR a nested-DAG
                array of stage objects — same shape `query_nested`
                accepts — e.g. '[{"name": "monthly",
                "source_model": "orders", "measures": ["amount:sum"]},
                {"source_model": "monthly", "measures": ["amount:avg"]}]'.
                In the array form the last element is the DAG root and
                every non-final element must have `name`.
        """
        _ensure_thread_safe(db_name, state.data_path_base)
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


def _build_strict_litellm_model_class():
    """Subclass smolagents.LiteLLMModel so we can mutate the `tools` payload
    after `_prepare_completion_kwargs` builds it. Returns the subclass.

    smolagents itself never writes `strict` on tool definitions. To force
    a uniform value we hook the prepare-then-complete path: each entry in
    `kwargs["tools"]` is `{"type": "function", "function": {...}}`; we
    inject `"strict": <value>` into every entry's `function` dict before
    handing to `litellm.completion`.
    """
    from smolagents import LiteLLMModel

    class _StrictLiteLLMModel(LiteLLMModel):
        def __init__(self, *args, _strict_value: bool, **kwargs):
            super().__init__(*args, **kwargs)
            self._strict_value = _strict_value

        def _prepare_completion_kwargs(self, *args, **kwargs):
            ck = super()._prepare_completion_kwargs(*args, **kwargs)
            for tool in ck.get("tools") or []:
                tool.setdefault("function", {})["strict"] = self._strict_value
            return ck

    return _StrictLiteLLMModel


def _run_smolagents_sync(
    prompt: str,
    user_query: str,
    native_tools: list,
    query_mode: str,
    slayer_storage_dir: str,
    model_id: str,
    strict: bool,
    state: TaskState,
) -> str:
    """Synchronous helper that runs the smolagents loop. Called via to_thread."""
    from smolagents import ToolCallingAgent

    StrictModel = _build_strict_litellm_model_class()
    model = StrictModel(model_id=model_id, _strict_value=strict)

    # max_tool_threads=1 keeps every tool call on the same OS thread —
    # required because the BIRD-Interact harness caches sqlite3 connections
    # per thread and SQLite connections are not thread-safe by default.
    if query_mode == "slayer":
        from mcp import StdioServerParameters
        from smolagents import MCPClient

        cfg = slayer_mcp_stdio_config(slayer_storage_dir)
        params = StdioServerParameters(
            command=cfg["command"], args=cfg["args"], env=cfg["env"]
        )
        with MCPClient(server_parameters=params) as mcp_client:
            slayer_tools = list(mcp_client.tools)
            agent = ToolCallingAgent(
                tools=[*slayer_tools, *native_tools],
                model=model,
                instructions=prompt,
                max_tool_threads=1,
            )
            result = agent.run(user_query)
    else:
        agent = ToolCallingAgent(
            tools=native_tools, model=model, instructions=prompt,
            max_tool_threads=1,
        )
        result = agent.run(user_query)

    # smolagents accumulates token counts on agent.monitor across the run.
    monitor = getattr(agent, "monitor", None)
    if monitor is not None:
        state.usage.add_call(
            scope="agent",
            model=model_id,
            prompt=getattr(monitor, "total_input_token_count", 0) or 0,
            completion=getattr(monitor, "total_output_token_count", 0) or 0,
        )
    return str(result)


class SmolagentsAgent:
    """SystemAgent implementation using smolagents."""

    def __init__(
        self,
        slayer_storage_root: str | None = None,
        model_id: str = "anthropic/claude-sonnet-4-5",
        strict: bool = False,
    ) -> None:
        self.slayer_storage_root = slayer_storage_root
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

        try:
            output = await asyncio.to_thread(
                _run_smolagents_sync,
                prompt,
                task_data["amb_user_query"],
                native_tools,
                query_mode,
                slayer_storage_dir,
                self.model_id,
                self.strict,
                state,
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
                "trajectory": [{"final_output": output[:500]}],
                "error": None,
                "usage": state.usage.model_dump(),
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
            },
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
        )
