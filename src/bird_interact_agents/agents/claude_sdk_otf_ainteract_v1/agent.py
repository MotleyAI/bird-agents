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
import time
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from bird_interact_agents.agents.claude_sdk.agent import (
    _ctx_var,
    SdkUsageTracker,
    build_bird_interact_server,
    native_tool_full_name,
)
from bird_interact_agents.agents.claude_sdk.context_budget import (
    context_window_for,
    make_context_budget_hook,
    make_wall_clock_budget_hook,
    per_task_timeout_s,
    update_wall_clock_start,
)
from bird_interact_agents.agents.claude_sdk.discovery_runtime import (
    run_main_with_discovery,
)
from bird_interact_agents.agents.claude_sdk.partition import (
    DISCOVERY_MAX_TURNS,
    build_main_workflow_note,
    build_discovery_prompt,
)
from bird_interact_agents.agents.claude_sdk_otf_v1.agent import (
    _MAX_TURNS,
    _make_query_before_submit_guard,
    _make_turn_budget_hook,
)
from bird_interact_agents.agents.claude_sdk_otf_v1.agent import (
    DISCOVERY_NATIVE_TOOL_NAMES as _ONE_SHOT_DISCOVERY_TOOL_NAMES,
)
from bird_interact_agents.agents.claude_sdk_otf_v1.agent import (
    MAIN_NATIVE_TOOL_NAMES as _ONE_SHOT_MAIN_TOOL_NAMES,
)
from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
    SLAYER_OTF_AINTERACT,
)
from bird_interact_agents.agents._pre_encoded import (
    resolve_pre_encoded_storage_dir,
    strip_write_tool_names,
    validate_pre_encoded_source,
)
from bird_interact_agents.agents._pre_encoded_prompts import (
    SLAYER_PRE_ENCODED_AINTERACT,
)
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.provider_registry import (
    is_supported_agent_model,
)
from bird_interact_agents.harness import (
    SampleStatus,
    _ambiguity_count,
    finalize_result_row,
    load_db_data_if_needed,
    materialize_task_db,
)
from bird_interact_agents.eval.annotation_io import (
    task_annotation_path,
)
from bird_interact_agents.eval.autopsy import _is_genuine_miss, run_autopsy
from bird_interact_agents.eval.grade_in_place import (
    load_audited_gold_rows_for,
    load_task_annotation_or_implicit,
    normalize_sol_sql,
)
from bird_interact_agents.eval.tolerant_grader import grade_submission, make_executor
from bird_interact_agents.slayer_otf import resolve_otf_task_storage_dir
from bird_interact_agents.slayer_otf.timing import log_otf_event, otf_timer
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


# DEV-1581 R2: a-interact partition = the one-shot partition + ask_user in
# BOTH contexts (discovery does the bulk of clarification; the main loop can
# still ask directly when submit feedback reveals an ambiguity). The two
# halves are separate persistent clients (see claude_sdk_otf_v1.agent).
MAIN_NATIVE_TOOL_NAMES = [*_ONE_SHOT_MAIN_TOOL_NAMES, _ASK_USER_TOOL]
DISCOVERY_NATIVE_TOOL_NAMES = [*_ONE_SHOT_DISCOVERY_TOOL_NAMES, _ASK_USER_TOOL]


def _build_prompt(
    eval_mode: str, task_data: dict, budget: float,
    pre_encoded_source: str | None = None,
) -> str:
    if eval_mode != "a-interact":
        raise ValueError(
            "claude_sdk_otf_ainteract supports only eval_mode='a-interact'; "
            f"got {eval_mode!r}"
        )
    user_query = task_data["amb_user_query"]
    db_name = task_data["selected_database"]
    template = (
        SLAYER_PRE_ENCODED_AINTERACT if pre_encoded_source
        else SLAYER_OTF_AINTERACT
    )
    return template.format(
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
        pre_encoded_source: str | None = None,
    ) -> None:
        # DEV-1586: `pre_encoded_source` (None | "otf" | "custom") selects the
        # read-only pre-encoded mode; `slayer_setup` is derived upstream.
        validate_pre_encoded_source(pre_encoded_source)
        if pre_encoded_source is None and slayer_setup != "on-the-fly":
            raise ValueError(
                "claude_sdk_otf_ainteract requires slayer_setup='on-the-fly' "
                f"when no pre_encoded_source is set; got {slayer_setup!r}"
            )
        if pre_encoded_source is not None and slayer_setup != "pre-encoded":
            raise ValueError(
                "claude_sdk_otf_ainteract with a pre_encoded_source requires "
                f"slayer_setup='pre-encoded'; got {slayer_setup!r}"
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
        self.pre_encoded_source = pre_encoded_source

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
        dataset = task_data.get("dataset")
        if not dataset:
            raise ValueError("task_data missing required 'dataset' field")
        if get_benchmark(dataset).one_shot:
            raise ValueError(
                "claude_sdk_otf_ainteract requires an a-interact benchmark "
                "(use claude_sdk for one-shot benchmarks); "
                f"got dataset={dataset!r}"
            )

        instance_id = task_data["instance_id"]
        db_name = task_data["selected_database"]

        if not is_supported_agent_model(self.model):
            msg = (
                f"claude_sdk_otf_ainteract requires an Anthropic or registry "
                f"open-weight model; "
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

        _ann_path = task_annotation_path(
            benchmark=benchmark.name,
            selected_database=db_name,
            instance_id=instance_id,
        )
        _ann_from_disk = _ann_path.exists()
        task_annotation = load_task_annotation_or_implicit(
            benchmark=benchmark.name,
            selected_database=db_name,
            instance_id=instance_id,
            amb_user_query=task_data.get("amb_user_query", ""),
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
        usage_tracker = SdkUsageTracker(accum, self.model)
        trajectory: list[dict] = []
        # Local handle to the per-task context dict. The exception path
        # reads from THIS local instead of `_ctx_var.get()` — a stale
        # ContextVar from a prior task in the same async context would
        # otherwise leak its `result` into this row when an early setup
        # failure (before _ctx_var.set, below) hits the except.
        ctx_dict: dict | None = None
        try:
            log_otf_event("run_task.start", instance_id=instance_id, db=db_name)
            with otf_timer(
                "run_task.load_db_data", instance_id=instance_id, db=db_name,
            ):
                load_db_data_if_needed(db_name, data_path_base)
            with otf_timer(
                "run_task.materialize_task_db",
                instance_id=instance_id, db=db_name,
            ):
                materialize_task_db(task_data, data_path_base)
            # DEV-1586: pre-encoded mode reads an ALREADY-encoded reference
            # read-only; on-the-fly (default) copies the deterministic cache.
            if self.pre_encoded_source:
                slayer_storage_dir, deleted_kb_ids = await resolve_pre_encoded_storage_dir(
                    db_name=db_name,
                    task_data=task_data,
                    data_path_base=data_path_base,
                    benchmark=benchmark.name,
                    source=self.pre_encoded_source,
                )
            else:
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
                "_discovery": None,
                "result": None,
                "eval_mode": eval_mode,
                "query_mode": query_mode,
                "max_asks": max_asks,
                "asks_used": 0,
                "usage": accum,
            }
            _ctx_var.set(ctx_dict)

            prompt = _build_prompt(
                eval_mode, task_data, budget, self.pre_encoded_source,
            )

            # DEV-1581 R2: two persistent in-process clients. DEV-1586
            # pre-encoded mode strips the SLayer WRITE tools from MAIN (the
            # agent only introspects); ask_user + discovery tools are
            # unaffected.
            main_tools = (
                strip_write_tool_names(MAIN_NATIVE_TOOL_NAMES)
                if self.pre_encoded_source else list(MAIN_NATIVE_TOOL_NAMES)
            )
            discovery_tools = list(DISCOVERY_NATIVE_TOOL_NAMES)

            with otf_timer(
                "run_task.create_sdk_mcp_server", instance_id=instance_id,
            ):
                main_mcp_servers = {
                    "bird-interact-tools": build_bird_interact_server(main_tools),
                }
                discovery_mcp_servers = {
                    "bird-interact-tools": build_bird_interact_server(
                        discovery_tools,
                    ),
                }

            # Per-task hook factories — never share state across tasks.
            pre_submit_gate, post_ask_counter, post_nag = _make_ask_user_guards()
            pre_query_gate, post_tool_tracker = _make_query_before_submit_guard()

            context_state: dict = {}
            update_wall_clock_start(context_state)
            (
                wall_clock_warning,
                wall_clock_deny,
            ) = make_wall_clock_budget_hook(
                context_state,
                budget_s=per_task_timeout_s(),
                submit_tool="submit_query",
            )

            # DEV-1579: build each client's options from the policy-owned env
            # kwargs. The agent only supplies its own tool surface / hooks per
            # client. ask_user lives on BOTH clients.
            def _build_main_options(_opt_kwargs: dict) -> ClaudeAgentOptions:
                return ClaudeAgentOptions(
                    **_opt_kwargs,
                    system_prompt=prompt + build_main_workflow_note(query_mode='slayer'),
                    mcp_servers=main_mcp_servers,
                    allowed_tools=list(main_tools),
                    tools=[],
                    setting_sources=[],
                    model=native_model_id(self.model),
                    effort=self.reasoning_effort,
                    max_turns=_MAX_TURNS,
                    hooks={
                        "PreToolUse": [
                            HookMatcher(
                                matcher="mcp__bird-interact-tools__submit_query",
                                # ask_user gate runs first; query gate second.
                                hooks=[pre_submit_gate, pre_query_gate],
                            ),
                            HookMatcher(hooks=[wall_clock_deny]),
                        ],
                        "PostToolUse": [
                            HookMatcher(
                                matcher=_ASK_USER_TOOL,
                                hooks=[post_ask_counter],
                            ),
                            HookMatcher(hooks=[post_nag]),
                            HookMatcher(hooks=[_make_turn_budget_hook(_MAX_TURNS)]),
                            HookMatcher(
                                hooks=[
                                    make_context_budget_hook(
                                        context_state,
                                        context_window_for(self.model),
                                    )
                                ]
                            ),
                            HookMatcher(hooks=[wall_clock_warning]),
                            # Must be last so it captures the true last-completed
                            # tool name after all other PostToolUse hooks have run.
                            HookMatcher(hooks=[post_tool_tracker]),
                        ],
                    },
                )

            def _build_discovery_options(_opt_kwargs: dict) -> ClaudeAgentOptions:
                return ClaudeAgentOptions(
                    **_opt_kwargs,
                    system_prompt=build_discovery_prompt(with_ask_user=True),
                    mcp_servers=discovery_mcp_servers,
                    allowed_tools=list(discovery_tools),
                    tools=[],
                    setting_sources=[],
                    model=native_model_id(self.model),
                    effort=self.reasoning_effort,
                    max_turns=DISCOVERY_MAX_TURNS,
                )

            # DEV-1561: per-message timing log + otf_timer around the main
            # client's __aenter__ (preserved through run_main_with_discovery).
            _msg_timing = {"first_t": None, "prev_t": None}

            def _on_main_message(msg, seq):
                now = time.monotonic()
                msg_type = type(msg).__name__
                if seq == 1:
                    _msg_timing["first_t"] = now
                    log_otf_event(
                        "run_task.sdk_first_message",
                        instance_id=instance_id,
                        msg_type=msg_type,
                        elapsed_s="0.000",
                    )
                else:
                    log_otf_event(
                        "run_task.sdk_message",
                        instance_id=instance_id,
                        seq=seq,
                        msg_type=msg_type,
                        gap_s=f"{now - (_msg_timing['prev_t'] or now):.3f}",
                    )
                _msg_timing["prev_t"] = now

            await run_main_with_discovery(
                model=self.model,
                accum=accum,
                usage_tracker=usage_tracker,
                context_state=context_state,
                main_mcp_servers=main_mcp_servers,
                discovery_mcp_servers=discovery_mcp_servers,
                build_main_options=_build_main_options,
                build_discovery_options=_build_discovery_options,
                initial_query=task_data["amb_user_query"],
                trajectory=trajectory,
                enter_cm_factory=lambda: otf_timer(
                    "run_task.sdk_client_enter", instance_id=instance_id,
                ),
                on_main_message=_on_main_message,
            )
            usage_tracker.finalize()
        except Exception as e:
            usage_tracker.finalize()
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
                    # ``asks_used`` is incremented by the ``ask_user`` tool
                    # inside the per-task context; without it on the usage
                    # dict the grader (grade_in_place._user_sim_n_asks
                    # plumbing) defaults to 0 and falsely flags
                    # ``never_asked_user`` on every interactive miss. Use
                    # ``(ctx_dict or {})`` because early-setup failures can
                    # raise BEFORE ctx_dict is constructed.
                    "usage": {
                        **accum.model_dump(),
                        "n_ask_user_calls": (ctx_dict or {}).get(
                            "asks_used", 0,
                        ),
                    },
                    "phase1_observation_audited": result.get("phase1_observation_audited"),
                    "phase1_observation_original": result.get("phase1_observation_original"),
                },
                deleted_kb_ids=deleted_kb_ids,
                slayer_storage_dir=slayer_storage_dir,
            )

        result = (ctx_dict or {}).get("result") or {}
        _autopsy_result = None
        _submitted_sql = result.get("submitted_sql") or ""
        if _submitted_sql:
            try:
                _audited_rows = load_audited_gold_rows_for(
                    benchmark=benchmark.name, instance_id=instance_id,
                )
                _orig_sql = normalize_sol_sql(
                    task_data.get("original_sol_sql") or task_data.get("sol_sql"),
                )
                _db_path = Path(
                    task_data.get("db_file_path")
                    or (Path(data_path_base) / db_name / f"{db_name}.sqlite")
                )
                _cascade = grade_submission(
                    task_annotation=task_annotation,
                    audited_gold_rows=_audited_rows,
                    original_sol_sql=_orig_sql,
                    submitted_sql=_submitted_sql,
                    db_path=_db_path,
                    benchmark=benchmark,
                    executor=make_executor(benchmark),
                    user_sim_n_asks=ctx_dict.get("asks_used", 0),
                )
                if _ann_from_disk and _is_genuine_miss(_cascade):
                    _autopsy_result = await run_autopsy(
                        task_annotation=task_annotation,
                        trajectory=trajectory,
                        slayer_storage_dir=slayer_storage_dir,
                        miss_diagnostics=_cascade.miss_diagnostics,
                        model=self.model,
                        is_one_shot=False,
                    )
            except Exception:
                logger.exception(
                    "[otf_ainteract] autopsy grading failed for %s; continuing without autopsy",
                    instance_id,
                )
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
                # ctx_dict is guaranteed-set on the success path, but the
                # ``{**accum.model_dump(), n_ask_user_calls: …}`` shape
                # mirrors the error path above for readability.
                "usage": {
                    **accum.model_dump(),
                    "n_ask_user_calls": ctx_dict.get("asks_used", 0),
                },
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
                "_autopsy": _autopsy_result,
                "_task_annotation": task_annotation,
            },
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
        )
