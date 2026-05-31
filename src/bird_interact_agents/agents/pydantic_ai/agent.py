"""PydanticAI implementation for BIRD-Interact.

Per-task context is passed via PydanticAI's `deps` mechanism — no global
state, no subprocess. Tools access task data through `ctx.deps`.

The bird-interact discovery + submission tool *bodies* live in
`bird_interact_agents.agents._submit`; this file only contains the
PydanticAI-specific wiring (decorators, RunContext type binding,
prepare_tools strict-mode shim, MCP toolset, agent factory).
"""

import logging
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits

from bird_interact_agents.agents._prompt_builders import (
    build_raw_c_interact_prompt,
    build_slayer_c_interact_prompt,
)
from bird_interact_agents.agents._run_capture import (
    _extract_tool_stats,
    _serialize_messages,
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
    ACTION_COSTS,
    MAX_MODEL_TURNS,
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
    SLAYER_MCP_STARTUP_TIMEOUT_S,
    update_budget,
)
from bird_interact_agents.usage import TokenUsage


_BY_NAME = {t.name: t for t in BIRD_INTERACT_TOOLS}


# Higher retry budget than pydantic-ai's default of 2. The Anthropic SDK
# already implements jittered exp backoff + retry-after on 429 and 5xx;
# we just want a higher ceiling for concurrent benchmark bursts.
_ANTHROPIC_CLIENT_RETRIES = 8


def _build_anthropic_model_with_retries(model_id: str):
    """Construct a PydanticAI `AnthropicModel` whose underlying SDK client
    carries a higher `max_retries` budget than the library default.

    Returns `None` (caller falls back to the default colon-form model
    string) when the local environment can't construct the SDK client —
    chiefly when an `ALL_PROXY=socks5h://…` env var is set without
    `socksio` installed, which httpx flags at AsyncClient creation. We'd
    rather lose the retry hardening than break the whole agent factory in
    test environments.
    """
    try:
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
    except Exception:  # noqa: BLE001 — defensive, never block on import
        return None
    try:
        client = AsyncAnthropic(max_retries=_ANTHROPIC_CLIENT_RETRIES)
    except Exception as e:  # noqa: BLE001 — env-dependent (e.g. SOCKS)
        logger.warning(
            "Falling back to default Anthropic client (max_retries=2): %s", e,
        )
        return None
    return AnthropicModel(
        model_id, provider=AnthropicProvider(anthropic_client=client),
    )


def _make_prepare_tools(strict_value: bool):
    """Return a `prepare_tools` callback that forces a uniform `strict` on
    every tool definition right before each model request.

    Cerebras's OpenAI-compatible API rejects requests with inconsistent
    `strict` values across entries — PydanticAI's native @agent.tool
    defaults to None and MCP-server tools also come in as None, so the
    merged list violates the constraint. This callback is PydanticAI's
    documented mechanism for cross-source uniformity (see the example
    `turn_on_strict_if_openai` in pydantic_ai.tools).
    """

    async def _force_strict(
        ctx: RunContext, tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [replace(td, strict=strict_value) for td in tool_defs]

    return _force_strict

logger = logging.getLogger(__name__)


class TaskDeps(BaseModel):
    """Per-task dependencies injected into every tool call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: SampleStatus
    data_path_base: str
    user_sim_model: str = "anthropic/claude-haiku-4-5-20251001"
    user_sim_prompt_version: str = "v2"
    slayer_storage_dir: str = ""
    # Mutable scratch space — recorded by submit tools, read by the runner.
    result: dict | None = None
    # Token usage accumulator — written by the user-sim wrapper and the
    # post-run capture in run_task.
    usage: TokenUsage = Field(default_factory=TokenUsage)
    # Full user-sim LLM exchange transcript (encoder + decoder calls per
    # ask_user invocation). Each entry: {"phase": "encoder"|"decoder",
    # "agent_question": str, "prompt": str, "response": str}.
    user_sim_transcript: list[dict] = Field(default_factory=list)
    # SLayer client/storage cache (initialised lazily)
    _slayer_client: Any = None
    _slayer_storage: Any = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slayer_client(deps: TaskDeps):
    if deps._slayer_client is None:
        from slayer.client.slayer_client import SlayerClient
        from slayer.storage.yaml_storage import YAMLStorage

        storage = YAMLStorage(base_dir=deps.slayer_storage_dir)
        client = SlayerClient(storage=storage)
        deps._slayer_client = client
        deps._slayer_storage = storage
    return deps._slayer_client


def _register_bird_interact_tools(agent: Agent, query_mode: str) -> None:
    """Wire the seven raw-mode discovery tools onto a PydanticAI agent.

    Each wrapper is a one-liner over the shared `run_env_action` helper
    (which applies budget gating + bookkeeping). Only the framework-
    specific decoration (signature + @agent.tool) lives here.
    """

    @agent.tool
    async def execute_sql(ctx: RunContext[TaskDeps], sql: str) -> str:
        """Execute a SQL query against the database and return results."""
        return run_env_action(ctx.deps, _BY_NAME["execute_sql"], query_mode, sql=sql)

    @agent.tool
    async def get_schema(ctx: RunContext[TaskDeps]) -> str:
        """Get the database schema (CREATE TABLE statements with sample data)."""
        return run_env_action(ctx.deps, _BY_NAME["get_schema"], query_mode)

    @agent.tool
    async def get_all_column_meanings(ctx: RunContext[TaskDeps]) -> str:
        """Get the meanings/descriptions of all columns in the database."""
        return run_env_action(ctx.deps, _BY_NAME["get_all_column_meanings"], query_mode)

    @agent.tool
    async def get_column_meaning(
        ctx: RunContext[TaskDeps], table_name: str, column_name: str,
    ) -> str:
        """Get the meaning of a specific column in a table."""
        return run_env_action(
            ctx.deps, _BY_NAME["get_column_meaning"], query_mode,
            table_name=table_name, column_name=column_name,
        )

    @agent.tool
    async def get_all_external_knowledge_names(ctx: RunContext[TaskDeps]) -> str:
        """List all available external knowledge entry names for this database."""
        return run_env_action(
            ctx.deps, _BY_NAME["get_all_external_knowledge_names"], query_mode,
        )

    @agent.tool
    async def get_knowledge_definition(
        ctx: RunContext[TaskDeps], knowledge_name: str,
    ) -> str:
        """Get the definition of a specific external knowledge entry."""
        return run_env_action(
            ctx.deps, _BY_NAME["get_knowledge_definition"], query_mode,
            knowledge_name=knowledge_name,
        )

    @agent.tool
    async def get_all_knowledge_definitions(ctx: RunContext[TaskDeps]) -> str:
        """Get all external knowledge definitions for this database."""
        return run_env_action(
            ctx.deps, _BY_NAME["get_all_knowledge_definitions"], query_mode,
        )


# ---------------------------------------------------------------------------
# Agent factory — one agent per (query_mode, eval_mode) combo, since
# PydanticAI agents register tools at construction time.
# ---------------------------------------------------------------------------

def _register_ask_user(agent: Agent, query_mode: str) -> None:
    @agent.tool
    async def ask_user(ctx: RunContext[TaskDeps], question: str) -> str:
        """Ask the user a clarification question about their query."""
        return await ask_user_impl(ctx.deps, question, query_mode)


def _register_submit_sql(agent: Agent) -> None:
    @agent.tool
    async def submit_sql(ctx: RunContext[TaskDeps], sql: str) -> str:
        """Submit your final SQL query for evaluation. Only submit when confident."""
        return submit_raw_sql(ctx.deps, sql)


def _register_submit_query(agent: Agent) -> None:
    @agent.tool
    async def submit_query(ctx: RunContext[TaskDeps], query_json: str) -> str:
        """Submit your final SLayer query for evaluation.

        `query_json` is a JSON string whose top-level value is either a
        single SlayerQuery object (e.g. `{"source_model": "orders",
        "measures": ["amount:sum"]}`) or a nested-DAG array of stage
        objects (same shape `query_nested` accepts; last element is the
        DAG root and every non-final element must have `name`). The
        chosen query is translated to SQL deterministically and tested
        against the ground truth.
        """
        return submit_slayer_query(ctx.deps, query_json, _slayer_client)


def _anthropic_cache_settings():
    """Construct AnthropicModelSettings that cache both the system prompt
    and the tool definitions.

    Returns None if pydantic-ai's Anthropic adapter isn't importable in this
    environment (e.g. anthropic SDK missing) — callers fall through to a
    no-caching Agent in that case rather than crashing the factory.
    """
    try:
        from pydantic_ai.models.anthropic import AnthropicModelSettings
    except Exception:  # noqa: BLE001 — defensive
        return None
    return AnthropicModelSettings(
        anthropic_cache_instructions=True,
        anthropic_cache_tool_definitions=True,
    )


def _build_raw_a_agent(
    model: Any, strict_value: bool = False, model_settings: Any = None,
) -> Agent:
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps, retries=2,
        prepare_tools=_make_prepare_tools(strict_value),
    )
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_bird_interact_tools(agent, "raw")
    _register_ask_user(agent, "raw")
    _register_submit_sql(agent)
    return agent


def _build_raw_c_agent(
    model: Any, strict_value: bool = False, model_settings: Any = None,
) -> Agent:
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps, retries=2,
        prepare_tools=_make_prepare_tools(strict_value),
    )
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_ask_user(agent, "raw")
    _register_submit_sql(agent)
    return agent


def _build_slayer_agent(
    model: Any, slayer_storage_dir: str, strict_value: bool = False,
    model_settings: Any = None,
) -> Agent:
    """Build a SLayer-mode agent (shared between a- and c-interact variants).

    Discovery tools come from the actual `slayer mcp` server attached as a
    toolset. We only register `ask_user` and `submit_query` natively.
    """
    cfg = slayer_mcp_stdio_config(slayer_storage_dir)
    slayer_server = MCPServerStdio(
        command=cfg["command"], args=cfg["args"], env=cfg["env"],
        # Retries happen at the tool-execution layer, not the agent's
        # reasoning layer — they don't consume LLM tokens, only wall
        # time. The benchmark needs liberal retries because SLayer
        # query MCP calls occasionally flake under load, and a
        # never_submitted task is a worse outcome than a slow one.
        # households_11 ran into the previous max_retries=3 on a
        # complex SEI-formula query and bailed without submitting.
        max_retries=100,
        # Startup-handshake budget: the slayer MCP runs `--ingest-on-startup`,
        # whose cost is the datasource schema RE-REFLECTION + semantic-layer
        # rebuild (NOT embeddings — those are prebuilt and hash-skipped). It
        # scales with schema size and balloons under multi-actor CPU
        # contention; see SLAYER_MCP_STARTUP_TIMEOUT_S for the sizing rationale.
        timeout=SLAYER_MCP_STARTUP_TIMEOUT_S,
    )
    kwargs: dict[str, Any] = dict(
        model=model, deps_type=TaskDeps, retries=2, toolsets=[slayer_server],
        prepare_tools=_make_prepare_tools(strict_value),
    )
    if model_settings is not None:
        kwargs["model_settings"] = model_settings
    agent = Agent(**kwargs)
    _register_ask_user(agent, "slayer")
    _register_submit_query(agent)
    return agent


# ---------------------------------------------------------------------------
# Top-level agent class
# ---------------------------------------------------------------------------


class PydanticAIAgent:
    """SystemAgent implementation using PydanticAI."""

    def __init__(
        self,
        slayer_storage_root: str | None = None,
        model: str = "anthropic/claude-sonnet-4-5",
        strict: bool = False,
        prompt_cache: bool = True,
    ) -> None:
        from bird_interact_agents.model_string import (
            build_pydantic_ai_model,
            is_anthropic,
            native_model_id,
        )

        self.slayer_storage_root = slayer_storage_root
        # Accept the canonical LiteLLM-style `provider/model_id` and convert
        # to whatever PydanticAI needs — for native providers that's a
        # `provider:model_id` string; for OpenAI-compatible third parties
        # like DeepInfra / Doubleword it's a fully-built OpenAIChatModel.
        self.model_id = model  # litellm form, kept for cost lookup
        anthropic_model = (
            _build_anthropic_model_with_retries(native_model_id(model))
            if is_anthropic(model) else None
        )
        self.model = anthropic_model or build_pydantic_ai_model(model)
        self.strict = strict
        # Prompt caching only meaningful (and only correctly typed) for
        # Anthropic models — AnthropicModelSettings keys are no-ops on
        # other providers but pydantic-ai may warn. Gate explicitly.
        self._model_settings = (
            _anthropic_cache_settings() if (prompt_cache and is_anthropic(model)) else None
        )

    def _select_agent(
        self,
        query_mode: str,
        eval_mode: str,
        slayer_storage_dir: str = "",
    ) -> Agent:
        if query_mode == "raw" and eval_mode == "a-interact":
            return _build_raw_a_agent(
                self.model, self.strict, model_settings=self._model_settings,
            )
        if query_mode == "raw" and eval_mode == "c-interact":
            return _build_raw_c_agent(
                self.model, self.strict, model_settings=self._model_settings,
            )
        if query_mode == "slayer":
            # Same agent shape for a- and c-interact; the system prompt
            # differs in the runner.
            return _build_slayer_agent(
                self.model, slayer_storage_dir, self.strict,
                model_settings=self._model_settings,
            )
        raise ValueError(f"Unknown mode combo: {query_mode}/{eval_mode}")

    async def _build_prompt(
        self, query_mode: str, eval_mode: str, task_data: dict, budget: float, deps: TaskDeps
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
                slayer_storage_dir=deps.slayer_storage_dir,
                db_name=db_name,
                task_data=task_data,
            )
        raise ValueError(f"Unknown mode combo: {query_mode}/{eval_mode}")

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
        status = SampleStatus(
            idx=0, original_data=task_data,
            remaining_budget=budget, total_budget=budget,
        )
        slayer_storage_dir, deleted_kb_ids = await resolve_task_storage_dir(
            slayer_storage_root=self.slayer_storage_root,
            db_name=db_name,
            task_data=task_data,
            query_mode=query_mode,
        )

        deps = TaskDeps(
            status=status,
            data_path_base=data_path_base,
            user_sim_model=user_sim_model,
            user_sim_prompt_version=user_sim_prompt_version,
            slayer_storage_dir=slayer_storage_dir,
        )

        agent = self._select_agent(query_mode, eval_mode, slayer_storage_dir)
        prompt = await self._build_prompt(query_mode, eval_mode, task_data, budget, deps)

        n_agent_turns: int | None = None
        tool_stats: dict | None = None
        try:
            # Lift PydanticAI's default request_limit (50) above our
            # MAX_MODEL_TURNS cap so long-tail tasks don't get killed
            # mid-flight before they can submit. Multiply by 2 because
            # one of our "turns" can map to multiple HTTP requests when
            # the model issues a tool call followed by its consumption
            # of the tool result in the same step.
            agent_run = await agent.run(
                user_prompt=task_data["amb_user_query"],
                instructions=prompt,
                deps=deps,
                usage_limits=UsageLimits(request_limit=MAX_MODEL_TURNS * 2),
            )
            output_text = str(agent_run.output)
            run_usage = agent_run.usage()
            deps.usage.add_call(
                scope="agent",
                model=self.model_id,
                prompt=getattr(run_usage, "input_tokens", 0) or 0,
                completion=getattr(run_usage, "output_tokens", 0) or 0,
                cache_read=getattr(run_usage, "cache_read_tokens", 0) or 0,
                cache_write=getattr(run_usage, "cache_write_tokens", 0) or 0,
            )
            try:
                n_agent_turns = sum(
                    1 for m in agent_run.all_messages()
                    if type(m).__name__ == "ModelResponse"
                )
            except Exception:  # noqa: BLE001 — best-effort metric
                n_agent_turns = None
            tool_stats = _extract_tool_stats(agent_run)
            agent_messages = _serialize_messages(agent_run)
        except Exception as e:
            # `logger.exception` gets us the traceback. For
            # ExceptionGroup-shaped errors (e.g. pydantic-ai's MCP
            # TaskGroup), also walk `.exceptions` so the wrapped
            # sub-exception lines up next to the wrapper instead of
            # being hidden behind a one-liner like "TaskGroup (1 sub-exception)".
            logger.exception("Agent error on %s: %s", instance_id, e)
            subs = getattr(e, "exceptions", None)
            if subs:
                for i, sub in enumerate(subs):
                    logger.error("  sub-exception %d: %r", i, sub)
            partial = deps.result or {}
            return finalize_result_row(
                {
                    "task_id": instance_id, "instance_id": instance_id,
                    "database": db_name,
                    "phase1_passed": partial.get("phase1_passed", False),
                    "phase2_passed": partial.get("phase2_passed", False),
                    "total_reward": partial.get("total_reward", 0.0),
                    "submitted_sql": partial.get("submitted_sql"),
                    "submitted_query": partial.get("submitted_query"),
                    "trajectory": {
                        "final_output_excerpt": "",
                        "messages": [],
                        "user_sim_transcript": list(deps.user_sim_transcript),
                    },
                    "error": str(e),
                    "usage": deps.usage.model_dump(),
                    "submission_status": partial.get(
                        "submission_status", "never_submitted",
                    ),
                    "phase1_observation": partial.get("phase1_observation"),
                    "phase2_observation": partial.get("phase2_observation"),
                    "predicted_result_json": partial.get("predicted_result_json"),
                    "gold_result_json": partial.get("gold_result_json"),
                    "n_agent_turns": n_agent_turns,
                    "tool_call_stats": tool_stats,
                    "phase1_observation_audited": partial.get("phase1_observation_audited"),
                    "phase1_observation_original": partial.get("phase1_observation_original"),
                },
                deleted_kb_ids=deleted_kb_ids,
                slayer_storage_dir=slayer_storage_dir,
            )

        result = deps.result or {}
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
                "trajectory": {
                    "final_output_excerpt": output_text[:500],
                    "messages": agent_messages,
                    "user_sim_transcript": list(deps.user_sim_transcript),
                },
                "error": None,
                "usage": deps.usage.model_dump(),
                "submission_status": result.get(
                    "submission_status", "never_submitted",
                ),
                "phase1_observation": result.get("phase1_observation"),
                "phase2_observation": result.get("phase2_observation"),
                "predicted_result_json": result.get("predicted_result_json"),
                "gold_result_json": result.get("gold_result_json"),
                "n_agent_turns": n_agent_turns,
                "tool_call_stats": tool_stats,
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
            },
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
        )
