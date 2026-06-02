"""Claude Agent SDK adapter that encodes KB items on the fly (DEV-1505).

A single agent (no forced stages, no recursion) that runs off the
deterministic OTF cache — base models + KB items pre-loaded as SLayer
memories — and is given the SLayer MCP with WRITE tools so it can encode
the relevant KB items into named columns/measures (in dependency order,
referencing earlier entities through declared joins) and then query off
them, instead of inlining everything.

After DEV-1507 this adapter is **livesqlbench / one-shot only**; the
sibling ``claude_sdk_otf_ainteract`` flavor handles
``mini-interact / a-interact`` with a hard ``ask_user``-before-submit
discipline.

The contextvar plumbing, the native submission / knowledge tools,
``_gate``/``_state_view``, the usage loop and ``finalize_result_row`` are
reused verbatim from the sibling ``claude_sdk`` adapter — only the
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
    get_all_external_knowledge_names,
    get_all_knowledge_definitions,
    get_knowledge_definition,
    submit_query,
)
from bird_interact_agents.agents.claude_sdk_otf.prompts import (
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


# SLayer query tools that satisfy the pre-submit verification gate.
# Any `query` or `query_nested` call immediately before `submit_query`
# is treated as the required output-inspection step.
SLAYER_QUERY_TOOLS: frozenset[str] = frozenset(
    {"mcp__slayer__query", "mcp__slayer__query_nested"}
)


def _make_query_before_submit_guard():
    """Enforce that the last completed tool call before ``submit_query``
    is a SLayer ``query`` or ``query_nested`` call.

    Returns ``(pre_submit_gate, post_tool_tracker)`` sharing a single
    per-task ``state`` closure. The factory MUST be invoked inside
    ``run_task`` (per task) — never on the agent constructor — to avoid
    cross-task state bleed when a single agent instance handles concurrent
    tasks via ``make_runner``.

    * ``pre_submit_gate`` (PreToolUse, matcher ``submit_query``): denies
      ``submit_query`` when the last completed tool call was not in
      ``SLAYER_QUERY_TOOLS``, with an explicit message telling the agent
      what to do next.
    * ``post_tool_tracker`` (PostToolUse, all tools): records the name of
      every completed tool call in ``state["last_tool"]``. PostToolUse only
      fires when a tool actually ran (i.e. a denied PreToolUse does NOT
      trigger PostToolUse), so a denied ``submit_query`` never overwrites
      the last-call record with its own name.
    """
    state: dict = {"last_tool": None}

    async def pre_submit_gate(input_data, tool_use_id, context):
        if state["last_tool"] not in SLAYER_QUERY_TOOLS:
            last = state["last_tool"] or "none"
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Your last tool call was `{last}`, not `query`. "
                        "Before submitting you MUST run the exact query JSON "
                        "you intend to submit through `query` and inspect the "
                        "output: verify that row count is non-zero and "
                        "plausible, that numeric aggregates are in the "
                        "expected range (non-zero proportions, correct units, "
                        "no all-zero values that would indicate integer "
                        "division), and that string values have the expected "
                        "casing and whitespace. Then call `submit_query` "
                        "immediately — with no other tool calls in between."
                    ),
                }
            }
        return {}

    async def post_tool_tracker(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name") or ""
        if tool_name:
            state["last_tool"] = tool_name
        return {}

    return pre_submit_gate, post_tool_tracker


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
# MCP server. one-shot decides autonomously (no user simulator), so the
# knowledge-lookup tools + `submit_query` are the only natives the agent
# needs. (The a-interact flavor adds `ask_user` in
# ``claude_sdk_otf_ainteract``.)
_KNOWLEDGE_TOOLS = [
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
]


def _select_tools(eval_mode: str) -> list:
    if eval_mode != "one-shot":
        raise ValueError(
            "claude_sdk_otf supports only eval_mode='one-shot' "
            "(use claude_sdk_otf_ainteract for a-interact); "
            f"got {eval_mode!r}"
        )
    return [*_KNOWLEDGE_TOOLS, submit_query]


def _build_prompt(eval_mode: str, task_data: dict, budget: float) -> str:
    if eval_mode != "one-shot":
        raise ValueError(
            "claude_sdk_otf supports only eval_mode='one-shot'; "
            f"got {eval_mode!r}"
        )
    user_query = task_data["amb_user_query"]
    db_name = task_data["selected_database"]
    return SLAYER_OTF_ONE_SHOT.format(
        budget=budget, db_name=db_name, user_query=user_query,
    )


class ClaudeSDKOtfAgent:
    """SystemAgent: Claude SDK agent with on-the-fly KB encoding.

    Anthropic-only (the SDK is locked to Anthropic); a non-Anthropic model
    short-circuits with a skip-shaped row. slayer-query-mode only;
    ``slayer_setup`` must be ``on-the-fly``. After DEV-1507 this flavor is
    bound to **livesqlbench / one-shot**; mismatched dataset or eval_mode
    is rejected at the agent boundary.
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
        eval_mode: str = "one-shot",
        user_sim_model: str = "anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version: str = "v2",
    ) -> dict:
        if query_mode != "slayer":
            raise ValueError(
                "claude_sdk_otf supports only --query-mode slayer; "
                f"got {query_mode!r}"
            )
        if eval_mode != "one-shot":
            raise ValueError(
                "claude_sdk_otf supports only --mode one-shot "
                "(use --framework claude_sdk_otf_ainteract for "
                f"--mode a-interact); got {eval_mode!r}"
            )

        # Defense in depth on top of the CLI gate: a programmatic caller
        # (`make_runner` has no dataset arg) cannot bypass dataset binding
        # by passing task_data with the wrong marker. Canonicalize via
        # ``get_benchmark`` so documented aliases are accepted (matches
        # `_validate_framework_dataset_mode`'s behavior).
        dataset = task_data.get("dataset") or "mini_interact"
        if get_benchmark(dataset).name != "livesqlbench":
            raise ValueError(
                "claude_sdk_otf is bound to --dataset livesqlbench "
                "(use --framework claude_sdk_otf_ainteract for "
                f"mini_interact); got dataset={dataset!r}"
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

        benchmark = get_benchmark(dataset)
        # one-shot REQUIRES a benchmark that declares one_shot=True. The
        # narrowed-flavor's dataset gate above already enforces livesqlbench,
        # but the benchmark check is kept as a load-bearing assertion in case
        # a future benchmark joins the one-shot family.
        if not benchmark.one_shot:
            raise ValueError(
                "--mode one-shot requires a task whose benchmark declares "
                "one_shot=True (its loader stamps task_data['dataset']); got "
                f"dataset={dataset!r}",
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
        # Local handle to the per-task context dict. We read from THIS
        # local on the exception path instead of `_ctx_var.get()` — a
        # stale ContextVar from a prior task in the same async context
        # would otherwise leak its `result` into this row when an early
        # setup failure (before _ctx_var.set, below) hits the except.
        ctx_dict: dict | None = None
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

            ctx_dict = {
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
                # ~15 MCP tools are exposed directly instead of being deferred
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
                async for msg in client.receive_response():
                    trajectory.append(
                        {"type": str(type(msg).__name__), "data": str(msg)[:500]}
                    )
                    accumulate_assistant_usage(accum, msg, self.model)
        except Exception as e:
            logger.error(
                "claude_sdk_otf error on %s: %s",
                instance_id, e, exc_info=True,
            )
            # Read from the LOCAL ctx_dict (not _ctx_var.get()) so a stale
            # context from a prior task in the same async context cannot
            # leak its diagnostics into this row. ctx_dict is None when an
            # early-setup call raised before `_ctx_var.set(...)`.
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
                    "usage": accum.model_dump(),
                    "phase1_observation_audited": result.get("phase1_observation_audited"),
                    "phase1_observation_original": result.get("phase1_observation_original"),
                },
                deleted_kb_ids=deleted_kb_ids,
                slayer_storage_dir=slayer_storage_dir,
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
                "usage": accum.model_dump(),
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
            },
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
        )
