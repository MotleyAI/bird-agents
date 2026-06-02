"""Claude Agent SDK implementation for BIRD-Interact."""

import contextvars
import logging
from types import SimpleNamespace

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)

from bird_interact_agents.agents._prompt_builders import (
    build_raw_c_interact_prompt,
    build_slayer_c_interact_prompt,
)
from bird_interact_agents.agents._submit import (
    ask_user_impl,
    submit_raw_sql,
    submit_slayer_query,
)
from bird_interact_agents.agents._tool_specs import (
    BIRD_INTERACT_TOOLS,
    render_action,
)
from bird_interact_agents.agents.claude_sdk.prompts import (
    RAW_A_INTERACT,
    SLAYER_A_INTERACT,
)
from bird_interact_agents.harness import (
    ACTION_COSTS,
    MAX_MODEL_TURNS,
    SampleStatus,
    execute_env_action,
    execute_submit_action,
    finalize_result_row,
    load_db_data_if_needed,
    parse_encoder_response,
    resolve_task_storage_dir,
    slayer_mcp_stdio_config,
    update_budget,
)
from bird_interact_agents.usage import TokenUsage


_BY_NAME = {t.name: t for t in BIRD_INTERACT_TOOLS}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-task context — uses contextvars so concurrent task runs don't collide.
# Each invocation of run_task() sets the var, and tools read from it.
# `_ctx` is exposed as a dict-like proxy for backward compat with tests.
# ---------------------------------------------------------------------------
_ctx_var: contextvars.ContextVar[dict] = contextvars.ContextVar("_ctx_var")


class _CtxProxy:
    """Dict-like proxy that reads/writes the current contextvar value.

    Tests do `agent_mod._ctx = {...}` and `agent_mod._ctx["key"]` — both
    work via this proxy by setting/reading the contextvar.
    """

    def __getitem__(self, key):
        return _ctx_var.get()[key]

    def __setitem__(self, key, value):
        _ctx_var.get()[key] = value

    def __contains__(self, key):
        try:
            return key in _ctx_var.get()
        except LookupError:
            return False

    def get(self, key, default=None):
        try:
            return _ctx_var.get().get(key, default)
        except LookupError:
            return default

    def update(self, *args, **kwargs):
        try:
            current = _ctx_var.get()
        except LookupError:
            current = {}
            _ctx_var.set(current)
        current.update(*args, **kwargs)


_ctx = _CtxProxy()


def _text(msg: str) -> dict:
    """Helper to build a tool return value."""
    return {"content": [{"type": "text", "text": str(msg)}]}


def _budget_note(status: SampleStatus) -> str:
    return (
        f"\n\n[Remaining budget: {status.remaining_budget:.1f}"
        f" / {status.total_budget:.1f}]"
    )


def _usage_value(usage: object, key: str) -> int:
    """Read a token field from a Claude Agent SDK usage object.

    The live SDK delivers `message.usage` as a **dict**
    (`{'input_tokens': …, 'output_tokens': …,
    'cache_read_input_tokens': …, 'cache_creation_input_tokens': …}`);
    mocked/older paths use an attribute object. Handle both — reading a
    dict with `getattr` silently returns 0, which is why agent token usage
    (and therefore agent cost) was previously recorded as zero.
    """
    if usage is None:
        return 0
    val = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    return val or 0


def accumulate_assistant_usage(accum: TokenUsage, msg: object, model: str) -> None:
    """Fold one streamed message's token usage into `accum` (scope=agent).

    Only `AssistantMessage`s carry per-turn usage; the cumulative
    `ResultMessage.usage` is intentionally skipped so totals aren't
    double-counted. Shared by the `claude_sdk` and `claude_sdk_otf`
    adapters so SDK-backed agents report cost consistently with the
    litellm-tracked frameworks.
    """
    if type(msg).__name__ != "AssistantMessage":
        return
    usage = getattr(msg, "usage", None)
    if usage is None:
        return
    accum.add_call(
        scope="agent",
        model=model,
        prompt=_usage_value(usage, "input_tokens"),
        completion=_usage_value(usage, "output_tokens"),
        cache_read=_usage_value(usage, "cache_read_input_tokens"),
        cache_write=_usage_value(usage, "cache_creation_input_tokens"),
    )


def _gate(action_name: str, status: SampleStatus) -> str | None:
    """Reject a non-submit tool call when budget would go below submit cost.

    Returns an error message to surface back to the agent, or None if OK.
    Mirrors the `force_submit` gating in the original mini_interact_agent
    and ADK before_tool_callback.
    """
    if action_name.startswith("submit_"):
        return None
    submit_tool = "submit_query" if _ctx.get("query_mode") == "slayer" else "submit_sql"
    submit_cost = ACTION_COSTS[submit_tool]
    cost = ACTION_COSTS.get(action_name, 0)
    if status.force_submit or status.remaining_budget < cost + submit_cost:
        return (
            f"Budget exhausted ({status.remaining_budget:.1f} remaining, "
            f"{action_name} costs {cost}). You MUST call {submit_tool} now "
            "with your best answer."
        )
    return None


async def _run_env(action_name: str, action_str: str) -> dict:
    """Shared body for raw exploration tools: gate → execute → bookkeep → annotate."""
    status: SampleStatus = _ctx["status"]
    err = _gate(action_name, status)
    if err is not None:
        return _text(err)
    observation, _ = execute_env_action(action_str, status, _ctx["data_path_base"])
    update_budget(status, action_name)
    return _text(str(observation) + _budget_note(status))


def _state_view() -> SimpleNamespace:
    """Return a SimpleNamespace adapter onto the current contextvar dict.

    The shared `_submit` helpers expect a state object exposing
    `status`/`data_path_base`/`user_sim_model`/`user_sim_prompt_version`/
    `usage`/`result`/`slayer_storage_dir` as attributes. claude_sdk
    keeps everything in a contextvar dict for legacy reasons; this thin
    view bridges the two.

    Note: only `result` is mutated by the helpers (via submit_*), and a
    SimpleNamespace assignment to `result` won't write back to the dict.
    Helpers that mutate state must use the explicit ctx-writing helpers
    below. Reads of `state.result` DO see the live ctx dict — the submit
    helpers' cross-phase observation-preservation logic relies on it
    (`prior = state.result or {}` → `state.result["phaseN_observation"] =
    prior.get(...)`); hard-coding to None silently blanked the other-phase
    observation on every multi-submit (DEV-1511 follow-up).
    """
    d = _ctx_var.get()
    return SimpleNamespace(
        status=d.get("status"),
        data_path_base=d.get("data_path_base"),
        user_sim_model=d.get("user_sim_model", "anthropic/claude-haiku-4-5-20251001"),
        user_sim_prompt_version=d.get("user_sim_prompt_version", "v2"),
        slayer_storage_dir=d.get("slayer_storage_dir", ""),
        usage=d.get("usage") or TokenUsage(),
        result=d.get("result"),
    )


# ---------------------------------------------------------------------------
# Raw-mode tools — direct DB exploration + SQL execution
#
# Each wrapper looks up its action template from BIRD_INTERACT_TOOLS, renders
# it with the model-supplied args, then routes through `_run_env` which
# applies budget gating and bookkeeping.
# ---------------------------------------------------------------------------

@tool("execute_sql", _BY_NAME["execute_sql"].description, {"sql": str})
async def execute_sql(args: dict) -> dict:
    return await _run_env(
        "execute_sql", render_action(_BY_NAME["execute_sql"], sql=args["sql"]),
    )


@tool("get_schema", _BY_NAME["get_schema"].description, {})
async def get_schema(args: dict) -> dict:
    return await _run_env("get_schema", render_action(_BY_NAME["get_schema"]))


@tool(
    "get_all_column_meanings",
    _BY_NAME["get_all_column_meanings"].description,
    {},
)
async def get_all_column_meanings(args: dict) -> dict:
    return await _run_env(
        "get_all_column_meanings",
        render_action(_BY_NAME["get_all_column_meanings"]),
    )


@tool(
    "get_column_meaning",
    _BY_NAME["get_column_meaning"].description,
    {"table_name": str, "column_name": str},
)
async def get_column_meaning(args: dict) -> dict:
    return await _run_env(
        "get_column_meaning",
        render_action(
            _BY_NAME["get_column_meaning"],
            table_name=args["table_name"], column_name=args["column_name"],
        ),
    )


@tool(
    "get_all_external_knowledge_names",
    _BY_NAME["get_all_external_knowledge_names"].description,
    {},
)
async def get_all_external_knowledge_names(args: dict) -> dict:
    return await _run_env(
        "get_all_external_knowledge_names",
        render_action(_BY_NAME["get_all_external_knowledge_names"]),
    )


@tool(
    "get_knowledge_definition",
    _BY_NAME["get_knowledge_definition"].description,
    {"knowledge_name": str},
)
async def get_knowledge_definition(args: dict) -> dict:
    return await _run_env(
        "get_knowledge_definition",
        render_action(
            _BY_NAME["get_knowledge_definition"],
            knowledge_name=args["knowledge_name"],
        ),
    )


@tool(
    "get_all_knowledge_definitions",
    _BY_NAME["get_all_knowledge_definitions"].description,
    {},
)
async def get_all_knowledge_definitions(args: dict) -> dict:
    return await _run_env(
        "get_all_knowledge_definitions",
        render_action(_BY_NAME["get_all_knowledge_definitions"]),
    )


# ---------------------------------------------------------------------------
# Shared tools — user simulator + submission
# ---------------------------------------------------------------------------

async def _ask_user_impl(question: str) -> str:
    """Thin shim that delegates to the shared user-sim helper."""
    return await ask_user_impl(_state_view(), question)


@tool(
    "ask_user",
    "Ask the user a clarification question about their query",
    {"question": str},
)
async def ask_user(args: dict) -> dict:
    status: SampleStatus = _ctx["status"]
    err = _gate("ask_user", status)
    if err is not None:
        return _text(err)
    answer = await _ask_user_impl(args["question"])
    update_budget(status, "ask_user")
    _ctx["asks_used"] = _ctx.get("asks_used", 0) + 1

    suffix = _budget_note(status)
    # In c-interact, the budget IS the turn budget; surface remaining ask
    # rounds explicitly (matches ADK callbacks_cinteract.after_tool_callback).
    if _ctx.get("eval_mode") == "c-interact":
        max_asks = _ctx.get("max_asks", 0)
        remaining = max(0, max_asks - _ctx["asks_used"])
        suffix += (
            f"\n[Clarification turns remaining: {remaining}/{max_asks}]"
        )
    return _text(answer + suffix)


@tool(
    "submit_sql",
    "Submit your final SQL query for evaluation. Only submit when confident.",
    {"sql": str},
)
async def submit_sql(args: dict) -> dict:
    state = _state_view()
    observation = submit_raw_sql(state, args["sql"])
    # `state` is a SimpleNamespace view — `state.result` doesn't write back
    # to the contextvar dict, so persist it explicitly. Budget + budget-note
    # are owned by `submit_raw_sql`.
    _ctx["result"] = {**state.result, "observation": observation}
    return _text(observation)


# ---------------------------------------------------------------------------
# SLayer-mode tools — agent reaches SLayer through its native MCP server
# (configured in run_task via mcp_servers={"slayer": ...}). The only native
# wrapper we keep is `submit_query`, which uses an in-process SlayerClient
# to translate the agent's SLayer query into deterministic SQL and submit
# it through bird-interact's eval pipeline.
# ---------------------------------------------------------------------------

def _slayer_client():
    """Get or build a SlayerClient for the current task's DB (used by submit_query)."""
    client = _ctx.get("_slayer_client")
    if client is None:
        from slayer.client.slayer_client import SlayerClient
        from slayer.storage.yaml_storage import YAMLStorage

        storage_dir = _ctx["slayer_storage_dir"]
        storage = YAMLStorage(base_dir=storage_dir)
        client = SlayerClient(storage=storage)
        _ctx["_slayer_client"] = client
        _ctx["_slayer_storage"] = storage
    return client


@tool(
    "submit_query",
    (
        "Submit your final SLayer query for evaluation. `query_json` is a "
        "JSON string whose top-level value is either a single SlayerQuery "
        'object (e.g. {"source_model": "orders", "measures": '
        '["amount:sum"]}) or a nested-DAG array of stage objects (same '
        "shape `query_nested` accepts; last element is the DAG root). "
        "The chosen query is translated to SQL deterministically and "
        "tested against the ground truth."
    ),
    {"query_json": str},
)
async def submit_query(args: dict) -> dict:
    state = _state_view()
    observation = submit_slayer_query(
        state, args["query_json"], lambda _s: _slayer_client(),
    )
    if state.result is None:
        # Helper already charged + appended the budget note for the
        # bad-JSON / failed-sql_sync paths; just propagate.
        return _text(observation)
    _ctx["result"] = {**state.result, "observation": observation}
    return _text(observation)


# ---------------------------------------------------------------------------
# Tool lists
# ---------------------------------------------------------------------------

RAW_A_TOOLS = [
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

# c-interact: schema and knowledge are injected in the prompt; agent only
# clarifies and submits.
RAW_C_TOOLS = [ask_user, submit_sql]

# In SLayer mode, exploration tools (help, models_summary, inspect_model,
# query, list_datasources) come from the slayer MCP server itself —
# wired into ClaudeAgentOptions.mcp_servers. We also expose the native
# bird-interact knowledge tools so SLayer agents have the same access
# to external domain knowledge that raw agents do.
SLAYER_A_TOOLS = [
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    ask_user,
    submit_query,
]
# c-interact slayer: knowledge is injected upfront in the prompt (matches
# raw c-interact's contract), so no separate knowledge tools are needed.
SLAYER_C_TOOLS = [ask_user, submit_query]


def _select_tools(query_mode: str, eval_mode: str) -> list:
    if query_mode == "raw" and eval_mode == "a-interact":
        return RAW_A_TOOLS
    if query_mode == "raw" and eval_mode == "c-interact":
        return RAW_C_TOOLS
    if query_mode == "slayer" and eval_mode == "a-interact":
        return SLAYER_A_TOOLS
    if query_mode == "slayer" and eval_mode == "c-interact":
        return SLAYER_C_TOOLS
    raise ValueError(f"Unknown mode combo: query_mode={query_mode} eval_mode={eval_mode}")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

async def _build_prompt(
    query_mode: str, eval_mode: str, task_data: dict, budget: float
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
            slayer_storage_dir=_ctx["slayer_storage_dir"],
            db_name=db_name,
            task_data=task_data,
        )

    raise ValueError(f"Unknown mode combo: query_mode={query_mode} eval_mode={eval_mode}")


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class ClaudeSDKAgent:
    """SystemAgent implementation using the Claude Agent SDK.

    The SDK is locked to Anthropic models — passing a non-Anthropic
    `model` causes `run_task` to short-circuit with a skip-shaped row so
    the 3-way comparison still renders cleanly. Use a different
    framework (`pydantic_ai`, `smolagents`, ...) for non-Anthropic models.
    """

    def __init__(
        self,
        slayer_storage_root: str | None = None,
        model: str = "anthropic/claude-sonnet-4-5",
    ) -> None:
        self.slayer_storage_root = slayer_storage_root
        self.model = model

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
        from bird_interact_agents.model_string import is_anthropic

        instance_id = task_data["instance_id"]
        db_name = task_data["selected_database"]
        if not is_anthropic(self.model):
            msg = (
                f"claude_sdk requires an Anthropic model; got {self.model!r}. "
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
        load_db_data_if_needed(db_name, data_path_base)

        # Per-task SLayer storage path (only relevant for slayer query mode).
        # Builds a HARD-8 variant on-the-fly when the task has
        # knowledge_ambiguity[*].deleted_knowledge entries.
        slayer_storage_dir, deleted_kb_ids = await resolve_task_storage_dir(
            slayer_storage_root=self.slayer_storage_root,
            db_name=db_name,
            task_data=task_data,
            query_mode=query_mode,
        )

        # Number of clarification turns the c-interact contract grants to the
        # agent — used for the post-ask_user "turns remaining" notice.
        from bird_interact_agents.harness import _ambiguity_count

        max_asks = _ambiguity_count(task_data) + 3  # +patience(3); matches ADK

        # Set the per-task context dict via contextvars — concurrent task
        # runs each get their own dict instance. The LOCAL `ctx_dict`
        # binding is what the exception path reads from, so a stale
        # ContextVar from a prior task in the same async context cannot
        # leak its `result` into this row.
        accum = TokenUsage()
        ctx_dict: dict = {
            "status": status,
            "data_path_base": data_path_base,
            "user_sim_model": user_sim_model,
            "user_sim_prompt_version": user_sim_prompt_version,
            "slayer_storage_dir": slayer_storage_dir,
            "_slayer_client": None,
            "_slayer_storage": None,
            "result": None,
            "eval_mode": eval_mode,
            "query_mode": query_mode,
            "max_asks": max_asks,
            "asks_used": 0,
            "usage": accum,
        }
        _ctx_var.set(ctx_dict)

        tools = _select_tools(query_mode, eval_mode)
        prompt = await _build_prompt(query_mode, eval_mode, task_data, budget)

        server = create_sdk_mcp_server(
            name="bird-interact-tools", version="1.0.0", tools=tools
        )
        tool_names = [f"mcp__bird-interact-tools__{t.name}" for t in tools]

        mcp_servers: dict = {"bird-interact-tools": server}
        if query_mode == "slayer":
            mcp_servers["slayer"] = slayer_mcp_stdio_config(slayer_storage_dir)
            # Allow the slayer MCP tools the agent will need
            slayer_tools = [
                "help",
                "list_datasources",
                "models_summary",
                "inspect_model",
                "search",
                "query",
            ]
            tool_names.extend(f"mcp__slayer__{t}" for t in slayer_tools)

        options = ClaudeAgentOptions(
            system_prompt=prompt,
            mcp_servers=mcp_servers,
            allowed_tools=tool_names,
        )

        trajectory: list[dict] = []
        try:
            # Auth env-var invariant validated at actor bootstrap; see ray_app._assert_actor_oauth_invariant.
            async with ClaudeSDKClient(options=options) as client:
                await client.query(task_data["amb_user_query"])
                turns = 0
                async for msg in client.receive_response():
                    trajectory.append(
                        {"type": str(type(msg).__name__), "data": str(msg)[:500]}
                    )
                    # Each assistant turn carries a `usage` block (a dict);
                    # fold it in so we record partial usage on exception or
                    # early break.
                    accumulate_assistant_usage(accum, msg, self.model)
                    # Count assistant model turns; cap at MAX_MODEL_TURNS to
                    # match the original mini_interact_agent (--max_turns=60)
                    # and ADK before_model_callback.
                    if type(msg).__name__ == "AssistantMessage":
                        turns += 1
                        if turns >= MAX_MODEL_TURNS:
                            logger.warning(
                                "Max model turns (%d) reached for %s; stopping.",
                                MAX_MODEL_TURNS, instance_id,
                            )
                            break
        except Exception as e:
            logger.error("Agent error on %s: %s", instance_id, e)
            # Read from the LOCAL ctx_dict (not _ctx_var.get()) so a stale
            # context from a prior task in the same async context cannot
            # leak its diagnostics into this row.
            result = ctx_dict.get("result") or {}
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
                    "usage": accum.model_dump(),
                    "phase1_passed_audited": result.get("phase1_passed_audited"),
                    "phase1_passed_original": result.get("phase1_passed_original"),
                    "phase1_observation_audited": result.get("phase1_observation_audited"),
                    "phase1_observation_original": result.get("phase1_observation_original"),
                },
                deleted_kb_ids=deleted_kb_ids,
                slayer_storage_dir=slayer_storage_dir,
            )

        result = ctx_dict.get("result") or {}
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
                "usage": accum.model_dump(),
                "phase1_passed_audited": result.get("phase1_passed_audited"),
                "phase1_passed_original": result.get("phase1_passed_original"),
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
            },
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
        )
