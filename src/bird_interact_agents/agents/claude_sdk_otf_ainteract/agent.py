"""Claude Agent SDK adapter that encodes KB items on the fly, a-interact
flavor (DEV-1507).

Bound to ``--dataset mini_interact --mode a-interact``. Shares the OTF
cache pipeline and the slayer write-tool whitelist with the sibling
``claude_sdk_otf`` package (one-shot/livesqlbench), but adds:

* A native ``ask_user`` tool.
* Three PreToolUse/PostToolUse guards that enforce a hard
  ``ask_user``-before-``submit_query`` discipline (built by
  ``_make_ask_user_guards`` — invoked once **per task**, never on the
  agent constructor, because a single agent instance is reused across
  concurrent tasks via ``make_runner``).

The encoding workflow is the same — see ``prompts.py::SLAYER_OTF_AINTERACT``
for Rule 0 (ask before encode) plus the shared encode-then-query rules.
"""

from __future__ import annotations

import logging

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    create_sdk_mcp_server,
)

from bird_interact_agents.agents.claude_sdk.agent import (
    _ctx_var,
    accumulate_assistant_usage,
    ask_user,
    get_all_external_knowledge_names,
    get_all_knowledge_definitions,
    get_knowledge_definition,
    submit_query,
)
from bird_interact_agents.agents.claude_sdk_otf.agent import (
    _MAX_TURNS,
    _make_turn_budget_hook,
    _slayer_tool_names,
)
from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
    SLAYER_OTF_AINTERACT,
)
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.model_string import is_anthropic, native_model_id
from bird_interact_agents.harness import (
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


# The full MCP tool name of the native ask_user tool — used as the
# PostToolUse counter's matcher AND as the race-skip predicate inside
# the nag hook.
_ASK_USER_TOOL = "mcp__bird-interact-tools__ask_user"

# How often (in TOTAL tool calls without ask_user) the nag fires.
_NAG_EVERY = 10


def _make_ask_user_guards():
    """Build the three per-task hook callables that enforce the
    ask-user-before-submit discipline.

    Returns ``(pre_submit_gate, post_ask_counter, post_nag)`` sharing a
    single per-task ``state`` closure. The factory MUST be invoked inside
    ``run_task`` (per task), NOT stored on the agent — a single agent
    instance is reused across concurrent tasks via ``make_runner``, and
    cross-task counter bleed would let one task's submit pass through
    another task's denial gate.

    * ``pre_submit_gate`` (PreToolUse, matcher ``submit_query``): denies
      ``submit_query`` while ``ask_count == 0``, with an explicit reason
      the SDK passes through to the model.
    * ``post_ask_counter`` (PostToolUse, matcher ``ask_user``): increments
      ``ask_count``.
    * ``post_nag`` (PostToolUse, all tools): every 10 tool calls without
      any ``ask_user`` yet, nudges the agent that the user-sim holds
      ground-truth it can't reach via KB alone. SKIPS when the current
      tool IS ``ask_user`` — otherwise a 10th call that happens to BE
      the first ask would emit a false-positive nag if the SDK fires
      ``post_nag`` before ``post_ask_counter``.
    """
    state = {"ask_count": 0, "tool_calls": 0}

    async def pre_submit_gate(input_data, tool_use_id, context):
        if state["ask_count"] == 0:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "You have not called ask_user. The user-sim has the "
                        "masked-KB ground truth that is unrecoverable from KB "
                        "alone. Identify your single most-uncertain "
                        "operationalisation choice (threshold / value list / "
                        "aggregation / sort / unit / rounding) and call "
                        "ask_user on it before submitting."
                    ),
                }
            }
        return {}

    async def post_ask_counter(input_data, tool_use_id, context):
        state["ask_count"] += 1
        return {}

    async def post_nag(input_data, tool_use_id, context):
        # Race-skip: when the current tool IS ask_user, never nag — the
        # ask-counter may fire AFTER us, and we'd emit a false positive
        # nag for the very call that satisfies the gate.
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


_KNOWLEDGE_TOOLS = [
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
]


def _select_tools(eval_mode: str) -> list:
    if eval_mode != "a-interact":
        raise ValueError(
            "claude_sdk_otf_ainteract supports only eval_mode='a-interact' "
            "(use claude_sdk_otf for one-shot); "
            f"got {eval_mode!r}"
        )
    return [*_KNOWLEDGE_TOOLS, ask_user, submit_query]


def _build_prompt(eval_mode: str, task_data: dict, budget: float) -> str:
    if eval_mode != "a-interact":
        raise ValueError(
            "claude_sdk_otf_ainteract supports only eval_mode='a-interact'; "
            f"got {eval_mode!r}"
        )
    user_query = task_data["amb_user_query"]
    db_name = task_data["selected_database"]
    return SLAYER_OTF_AINTERACT.format(
        budget=budget, db_name=db_name, user_query=user_query,
    )


class ClaudeSDKOtfAInteractAgent:
    """SystemAgent: Claude SDK agent with on-the-fly KB encoding + enforced
    ask-user discipline.

    Anthropic-only (the SDK is locked to Anthropic). Bound to
    ``--dataset mini_interact --mode a-interact``; mismatched dataset or
    eval_mode is rejected at the agent boundary.
    """

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
                "claude_sdk_otf_ainteract requires slayer_setup='on-the-fly'; "
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
                "claude_sdk_otf_ainteract supports only --query-mode slayer; "
                f"got {query_mode!r}"
            )
        if eval_mode != "a-interact":
            raise ValueError(
                "claude_sdk_otf_ainteract supports only --mode a-interact "
                "(use --framework claude_sdk_otf for --mode one-shot); "
                f"got {eval_mode!r}"
            )

        # Defense in depth on top of the CLI gate. Canonicalize via
        # ``get_benchmark`` so the documented ``mini-interact`` alias is
        # accepted (matches `_validate_framework_dataset_mode`'s behavior;
        # a programmatic caller using the alias must reach the agent).
        dataset = task_data.get("dataset") or "mini_interact"
        if get_benchmark(dataset).name != "mini_interact":
            raise ValueError(
                "claude_sdk_otf_ainteract is bound to --dataset mini_interact "
                "(use --framework claude_sdk_otf for livesqlbench); "
                f"got dataset={dataset!r}"
            )

        instance_id = task_data["instance_id"]
        db_name = task_data["selected_database"]

        if not is_anthropic(self.model):
            msg = (
                f"claude_sdk_otf_ainteract requires an Anthropic model; "
                f"got {self.model!r}. Skipped — use --framework "
                "pydantic_ai_otf_encode for non-Anthropic models."
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
        # Local handle to the per-task context dict. The exception path
        # reads from THIS local instead of `_ctx_var.get()` — a stale
        # ContextVar from a prior task in the same async context would
        # otherwise leak its `result` into this row when an early setup
        # failure (before _ctx_var.set, below) hits the except.
        ctx_dict: dict | None = None
        try:
            load_db_data_if_needed(db_name, data_path_base)
            materialize_task_db(task_data, data_path_base)
            slayer_storage_dir, deleted_kb_ids = await resolve_otf_task_storage_dir(
                db_name=db_name,
                task_data=task_data,
                data_path_base=data_path_base,
                benchmark=benchmark.name,
            )

            max_asks = _ambiguity_count(task_data) + 3

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
                "slayer": slayer_mcp_stdio_config(slayer_storage_dir),
            }
            tool_names.extend(_slayer_tool_names())

            # Per-task hook factory — never share state across tasks.
            pre_submit_gate, post_ask_counter, post_nag = _make_ask_user_guards()

            options = ClaudeAgentOptions(
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
                            matcher="mcp__bird-interact-tools__submit_query",
                            hooks=[pre_submit_gate],
                        ),
                    ],
                    "PostToolUse": [
                        HookMatcher(
                            matcher=_ASK_USER_TOOL,
                            hooks=[post_ask_counter],
                        ),
                        HookMatcher(hooks=[post_nag]),
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
                "claude_sdk_otf_ainteract error on %s: %s",
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
                    "phase1_passed_audited": result.get("phase1_passed_audited"),
                    "phase1_passed_original": result.get("phase1_passed_original"),
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
                "phase1_passed_audited": result.get("phase1_passed_audited"),
                "phase1_passed_original": result.get("phase1_passed_original"),
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
            },
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
        )
