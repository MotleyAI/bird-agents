"""Claude Agent SDK adapter that encodes KB items on the fly (DEV-1505).

A single agent (no forced stages, no recursion) that runs off the
deterministic OTF cache — base models + KB items pre-loaded as SLayer
memories — and is given the SLayer MCP with WRITE tools so it can encode
the relevant KB items into named columns/measures (in dependency order,
referencing earlier entities through declared joins) and then query off
them, instead of inlining everything.

slayer-query-mode only; eval modes ``a-interact`` and ``one-shot``.

The contextvar plumbing, the native user-sim / submission / knowledge
tools, ``_gate``/``_state_view``, the usage loop and ``finalize_result_row``
are reused verbatim from the sibling ``claude_sdk`` adapter — only the
per-task storage source (the deterministic cache, NOT committed models),
the SLayer write-tool whitelist, the prompts, and the mode gating differ.
"""

from __future__ import annotations

import logging

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    create_sdk_mcp_server,
)

# Reuse the sibling claude_sdk adapter's contextvar + native tools. These
# tools read/write the SAME `_ctx_var` we set below, so binding them here
# is sound.
from bird_interact_agents.agents.claude_sdk.agent import (
    _ctx_var,
    accumulate_assistant_usage,
    ask_user,
    get_all_external_knowledge_names,
    get_all_knowledge_definitions,
    get_knowledge_definition,
    submit_query,
)
from bird_interact_agents.agents.claude_sdk_otf.prompts import (
    SLAYER_OTF_A_INTERACT,
    SLAYER_OTF_ONE_SHOT,
)
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.model_string import is_anthropic, native_model_id
from bird_interact_agents.harness import (
    MAX_MODEL_TURNS,
    SampleStatus,
    _ambiguity_count,
    finalize_result_row,
    load_db_data_if_needed,
    materialize_task_db,
    slayer_mcp_stdio_config,
)
from bird_interact_agents.slayer_otf import resolve_otf_task_storage_dir
from bird_interact_agents.usage import TokenUsage

logger = logging.getLogger(__name__)


# SLayer MCP tools the OTF agent may call. The existing claude_sdk slayer
# mode is read-only; this adapter ADDS the write tools (create_model /
# edit_model / save_memory / validate_models) plus query_nested so the
# agent can encode KB items and submit nested-DAG queries.
SLAYER_MCP_TOOLS = [
    "help",
    "list_datasources",
    "models_summary",
    "inspect_model",
    "search",
    "query",
    "query_nested",
    "create_model",
    "edit_model",
    "save_memory",
    "validate_models",
]


def _slayer_tool_names() -> list[str]:
    return [f"mcp__slayer__{t}" for t in SLAYER_MCP_TOOLS]


# Encode-then-query is turn-expensive (one turn per KB column created/tested),
# so this agent needs more headroom than the base agentic cap. 2x the base.
_MAX_TURNS = 2 * MAX_MODEL_TURNS

# Warn the agent to submit once it's within this many turns of the cap.
_TURN_BUDGET_WARN_WITHIN = 3


def _make_turn_budget_hook(
    max_turns: int, warn_within: int = _TURN_BUDGET_WARN_WITHIN
):
    """Build a PostToolUse hook that nudges the agent to submit when it's
    within ``warn_within`` tool-calls of the hard ``max_turns`` cap.

    The hook counts its own invocations (≈ one per agent turn, since the
    agent emits a tool call per turn) rather than reading the receive loop's
    state, so it stays self-contained and per-task. The returned
    ``additionalContext`` is injected into the model's context after the tool
    result — an un-submitted task scores zero, so the nudge is load-bearing
    for encode-heavy tasks that run long.
    """
    state = {"calls": 0}

    async def _hook(input_data, tool_use_id, context):
        state["calls"] += 1
        remaining = max_turns - state["calls"]
        if 0 < remaining <= warn_within:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": (
                        f"[TURN BUDGET] ~{remaining} model turn(s) remain before "
                        f"the hard limit of {max_turns}. If you have a candidate "
                        "answer, call submit_query NOW — an un-submitted task "
                        "scores zero."
                    ),
                }
            }
        return {}

    return _hook


# Native (in-process) tools registered under the "bird-interact-tools"
# MCP server. a-interact gets `ask_user`; one-shot decides autonomously
# (no user simulator) so it is dropped. Both keep the knowledge-lookup
# tools and `submit_query`.
_KNOWLEDGE_TOOLS = [
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
]


def _select_tools(eval_mode: str) -> list:
    if eval_mode == "a-interact":
        return [*_KNOWLEDGE_TOOLS, ask_user, submit_query]
    if eval_mode == "one-shot":
        return [*_KNOWLEDGE_TOOLS, submit_query]
    raise ValueError(
        f"claude_sdk_otf supports eval_mode a-interact or one-shot; "
        f"got {eval_mode!r}"
    )


def _build_prompt(eval_mode: str, task_data: dict, budget: float) -> str:
    user_query = task_data["amb_user_query"]
    db_name = task_data["selected_database"]
    template = SLAYER_OTF_A_INTERACT if eval_mode == "a-interact" else SLAYER_OTF_ONE_SHOT
    return template.format(budget=budget, db_name=db_name, user_query=user_query)


class ClaudeSDKOtfAgent:
    """SystemAgent: Claude SDK agent with on-the-fly KB encoding.

    Anthropic-only (the SDK is locked to Anthropic); a non-Anthropic model
    short-circuits with a skip-shaped row. slayer-query-mode only;
    ``slayer_setup`` must be ``on-the-fly``.
    """

    #: Reasoning-effort levels accepted by the Claude SDK (`ClaudeAgentOptions.effort`).
    _EFFORT_CHOICES = ("low", "medium", "high", "max")

    def __init__(
        self,
        slayer_storage_root: str | None = None,
        model: str = "anthropic/claude-sonnet-4-5",
        slayer_setup: str = "on-the-fly",
        reasoning_effort: str | None = None,
    ) -> None:
        if slayer_setup != "on-the-fly":
            raise ValueError(
                "claude_sdk_otf requires slayer_setup='on-the-fly'; "
                f"got {slayer_setup!r}"
            )
        if reasoning_effort is not None and reasoning_effort not in self._EFFORT_CHOICES:
            raise ValueError(
                f"reasoning_effort must be one of {self._EFFORT_CHOICES} or None; "
                f"got {reasoning_effort!r}"
            )
        self.slayer_storage_root = slayer_storage_root
        self.model = model
        self.slayer_setup = slayer_setup
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
        if query_mode != "slayer":
            raise ValueError(
                "claude_sdk_otf supports only --query-mode slayer; "
                f"got {query_mode!r}"
            )
        if eval_mode not in ("a-interact", "one-shot"):
            raise ValueError(
                "claude_sdk_otf supports only --mode a-interact or "
                f"--mode one-shot; got {eval_mode!r}"
            )

        instance_id = task_data["instance_id"]
        db_name = task_data["selected_database"]

        if not is_anthropic(self.model):
            msg = (
                f"claude_sdk_otf requires an Anthropic model; got {self.model!r}. "
                "Skipped — use --framework pydantic_ai_otf_encode for "
                "non-Anthropic models."
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

        benchmark = get_benchmark(task_data.get("dataset") or "mini_interact")
        if eval_mode == "one-shot" and not benchmark.one_shot:
            raise ValueError(
                "--mode one-shot requires a task whose benchmark declares "
                "one_shot=True (its loader stamps task_data['dataset']); got "
                f"dataset={task_data.get('dataset')!r}",
            )

        status = SampleStatus(
            idx=0,
            original_data=task_data,
            remaining_budget=budget,
            total_budget=budget,
        )

        deleted_kb_ids: list[int] = []
        slayer_storage_dir = ""
        accum = TokenUsage()
        trajectory: list[dict] = []
        try:
            load_db_data_if_needed(db_name, data_path_base)
            # LiveSQLBench one-shot: per-task isolated working sqlite (no-op
            # for mini-interact).
            materialize_task_db(task_data, data_path_base)
            # Cache-only per-task storage: deterministic OTF cache copied to
            # a scratch dir with this task's deleted KBs masked. The agent
            # encodes into THIS dir at task time.
            slayer_storage_dir, deleted_kb_ids = await resolve_otf_task_storage_dir(
                db_name=db_name,
                task_data=task_data,
                data_path_base=data_path_base,
                benchmark=benchmark.name,
            )

            max_asks = _ambiguity_count(task_data) + 3  # +patience(3); matches ADK

            _ctx_var.set({
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
            })

            tools = _select_tools(eval_mode)
            prompt = _build_prompt(eval_mode, task_data, budget)

            server = create_sdk_mcp_server(
                name="bird-interact-tools", version="1.0.0", tools=tools,
            )
            tool_names = [f"mcp__bird-interact-tools__{t.name}" for t in tools]

            mcp_servers: dict = {
                "bird-interact-tools": server,
                "slayer": slayer_mcp_stdio_config(
                    slayer_storage_dir, ingest_on_startup=False,
                ),
            }
            tool_names.extend(_slayer_tool_names())

            options = ClaudeAgentOptions(
                system_prompt=prompt,
                mcp_servers=mcp_servers,
                allowed_tools=tool_names,
                # Restrict to ONLY our MCP tools: drop every Claude Code
                # built-in (Bash/Edit/Task/WebFetch/ToolSearch/...). Removing
                # ToolSearch is load-bearing — with the built-ins gone the
                # ~16 MCP tools are exposed directly instead of being deferred
                # behind ToolSearch (which previously wasted ~5 turns/run while
                # the agent re-discovered its own tools). setting_sources=[]
                # also keeps the run reproducible (no user/project settings,
                # no CLAUDE.md bleed-through).
                tools=[],
                setting_sources=[],
                # Pin the requested Anthropic model (bare id, no provider
                # prefix) so --agent-model actually takes effect instead of
                # the claude CLI's configured default.
                model=native_model_id(self.model),
                # Reasoning-effort level (None => SDK default).
                effort=self.reasoning_effort,
                # Native turn cap (2x the base). Unlike a manual break on the
                # receive stream, max_turns lets the FINAL turn's tool (e.g.
                # submit_query) execute before the run stops — the off-by-one
                # that previously dropped a last-turn submission.
                max_turns=_MAX_TURNS,
                # Nudge the agent to submit when it nears the cap.
                hooks={
                    "PostToolUse": [
                        HookMatcher(hooks=[_make_turn_budget_hook(_MAX_TURNS)]),
                    ],
                },
            )

            async with ClaudeSDKClient(options=options) as client:
                await client.query(task_data["amb_user_query"])
                # No manual turn-break: `max_turns` caps the run and lets the
                # final turn's tool call execute, so the stream ends cleanly.
                async for msg in client.receive_response():
                    trajectory.append(
                        {"type": str(type(msg).__name__), "data": str(msg)[:500]}
                    )
                    accumulate_assistant_usage(accum, msg, self.model)
        except Exception as e:
            logger.error("claude_sdk_otf error on %s: %s", instance_id, e)
            return finalize_result_row(
                {
                    "task_id": instance_id,
                    "instance_id": instance_id,
                    "database": db_name,
                    "phase1_passed": False,
                    "phase2_passed": False,
                    "total_reward": 0.0,
                    "trajectory": trajectory,
                    "error": str(e),
                    "usage": accum.model_dump(),
                },
                deleted_kb_ids=deleted_kb_ids,
                slayer_storage_dir=slayer_storage_dir,
            )

        result = _ctx_var.get().get("result") or {}
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
