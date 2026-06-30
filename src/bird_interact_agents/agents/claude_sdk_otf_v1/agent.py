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
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

# Reuse the sibling claude_sdk adapter's contextvar + native tools / native
# in-process server builder. These tools read/write the SAME `_ctx_var` we set
# below, so binding them here is sound.
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
from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import (
    SLAYER_OTF_ONE_SHOT,
)
from bird_interact_agents.agents._pre_encoded import (
    resolve_pre_encoded_storage_dir,
    strip_write_tool_names,
    validate_pre_encoded_source,
)
from bird_interact_agents.agents._pre_encoded_prompts import (
    SLAYER_PRE_ENCODED_ONE_SHOT,
)
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.provider_registry import (
    is_supported_agent_model,
)
from bird_interact_agents.harness import (
    MAX_MODEL_TURNS,
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
from bird_interact_agents.usage import TokenUsage

logger = logging.getLogger(__name__)


# SLayer query tools that satisfy the pre-submit verification gate.
# Any `query` or `query_nested` call immediately before `submit_query`
# is treated as the required output-inspection step.
#
# DEV-1534 Fix C: these now point at our bird-interact-tools wrappers
# (which forward to SLayer's MCP query/query_nested but with the
# `normalize_filters` opt-out), NOT at the raw subprocess MCP tools.
SLAYER_QUERY_TOOLS: frozenset[str] = frozenset(
    {
        "mcp__bird-interact-tools__query",
    }
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


# Encode-then-query is turn-expensive (one turn per KB column created/tested),
# so this agent needs more headroom than the base agentic cap. 2x the base.
_MAX_TURNS = 2 * MAX_MODEL_TURNS

# Warn the agent to submit once it's within this many turns of the cap.
_TURN_BUDGET_WARN_WITHIN = 3


def _make_turn_budget_hook(
    max_turns: int, warn_within: int = _TURN_BUDGET_WARN_WITHIN,
    submit_tool: str = "submit_query",
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
                        f"answer, call {submit_tool} NOW — an un-submitted task "
                        "scores zero."
                    ),
                }
            }
        return {}

    return _hook


# DEV-1581 R2: the discovery/main partition is now two SEPARATE persistent
# clients, each with its OWN in-process ``bird-interact-tools`` server holding
# only its half. Because the clients are separate sessions, main's per-turn
# tool schema NEVER contains discovery's introspection tools (the core
# guarantee). KB lookups live in BOTH so exact formulas never pass through a
# lossy summary before encoding; the main loop reaches discovery's schema/data
# tools only through ``ask_discovery``.
_KB_NATIVE_BARE = [
    "get_all_external_knowledge_names",
    "get_knowledge_definition",
    "get_all_knowledge_definitions",
]

# Bare native names per client (full ``mcp__bird-interact-tools__*`` derived
# below). MAIN owns encode/query/submit + the ask_discovery bridge; DISCOVERY
# owns the introspection tools (+ ``query`` so it can profile sample values).
_MAIN_NATIVE_BARE = [
    "query",
    "query_nested",
    "submit_query",
    "create_model",
    "edit_model",
    "validate_models",
    "help",
    "ask_discovery",
    *_KB_NATIVE_BARE,
]
_DISCOVERY_NATIVE_BARE = [
    "search",
    # DEV-1591: discovery owns the targeted point-lookup `inspect` (the
    # prompts route all column / memory detail reads to it). `search` is
    # discovery-only and hardwired-compact; `inspect` is how discovery
    # reads full bodies.
    "inspect",
    "models_summary",
    "inspect_model",
    "query",
    *_KB_NATIVE_BARE,
]

MAIN_NATIVE_TOOL_NAMES = [native_tool_full_name(b) for b in _MAIN_NATIVE_BARE]
DISCOVERY_NATIVE_TOOL_NAMES = [
    native_tool_full_name(b) for b in _DISCOVERY_NATIVE_BARE
]


def _build_prompt(
    eval_mode: str, task_data: dict, budget: float,
    pre_encoded_source: str | None = None,
) -> str:
    if eval_mode != "one-shot":
        raise ValueError(
            "claude_sdk_otf supports only eval_mode='one-shot'; "
            f"got {eval_mode!r}"
        )
    user_query = task_data["amb_user_query"]
    db_name = task_data["selected_database"]
    template = (
        SLAYER_PRE_ENCODED_ONE_SHOT if pre_encoded_source
        else SLAYER_OTF_ONE_SHOT
    )
    return template.format(
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
        pre_encoded_source: str | None = None,
    ) -> None:
        # DEV-1586: `pre_encoded_source` (None | "otf" | "custom") selects the
        # read-only pre-encoded mode; `slayer_setup` is derived upstream.
        validate_pre_encoded_source(pre_encoded_source)
        if pre_encoded_source is None and slayer_setup != "on-the-fly":
            raise ValueError(
                "claude_sdk_otf requires slayer_setup='on-the-fly' when no "
                f"pre_encoded_source is set; got {slayer_setup!r}"
            )
        if pre_encoded_source is not None and slayer_setup != "pre-encoded":
            raise ValueError(
                "claude_sdk_otf with a pre_encoded_source requires "
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
        dataset = task_data.get("dataset")
        if not dataset:
            raise ValueError("task_data missing required 'dataset' field")
        if not get_benchmark(dataset).one_shot:
            raise ValueError(
                "claude_sdk_otf requires a one-shot benchmark "
                "(use claude_sdk for a-interact benchmarks); "
                f"got dataset={dataset!r}"
            )

        instance_id = task_data["instance_id"]
        db_name = task_data["selected_database"]

        if not is_supported_agent_model(self.model):
            msg = (
                f"claude_sdk_otf requires an Anthropic or registry open-weight "
                f"model; got {self.model!r}. "
                "Skipped — use --framework claude_sdk_otf_encode for "
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
            # DEV-1586: pre-encoded mode reads an ALREADY-encoded reference
            # read-only; on-the-fly (default) copies the deterministic cache
            # and the agent encodes into it at task time.
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

            max_asks = _ambiguity_count(task_data) + 3  # +patience(3); matches ADK

            ctx_dict = {
                "status": status,
                "data_path_base": data_path_base,
                "user_sim_model": user_sim_model,
                # DEV-1613: agent model for the in-task N5 judge.
                "agent_model": self.model,
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
            # agent only introspects); discovery is read-only already.
            main_tools = (
                strip_write_tool_names(MAIN_NATIVE_TOOL_NAMES)
                if self.pre_encoded_source else list(MAIN_NATIVE_TOOL_NAMES)
            )
            discovery_tools = list(DISCOVERY_NATIVE_TOOL_NAMES)

            main_mcp_servers = {
                "bird-interact-tools": build_bird_interact_server(main_tools),
            }
            discovery_mcp_servers = {
                "bird-interact-tools": build_bird_interact_server(discovery_tools),
            }

            # Per-task hook factories — must be created here (not on the
            # agent constructor) to avoid cross-task state bleed.
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
            # kwargs (telemetry-disable + hermetic CLAUDE_CONFIG_DIR + any
            # registry session env / thinking). The agent only supplies its own
            # tool surface / hooks per client.
            def _build_main_options(_opt_kwargs: dict) -> ClaudeAgentOptions:
                return ClaudeAgentOptions(
                    **_opt_kwargs,
                    system_prompt=prompt + build_main_workflow_note(query_mode='slayer'),
                    mcp_servers=main_mcp_servers,
                    allowed_tools=list(main_tools),
                    # No Claude Code built-ins: ask_discovery is an in-process
                    # native, so there is no Task/subagent surface. Dropping the
                    # built-ins also exposes the MCP tools directly (no
                    # ToolSearch deferral) and keeps the run reproducible.
                    tools=[],
                    setting_sources=[],
                    model=native_model_id(self.model),
                    effort=self.reasoning_effort,
                    # Native turn cap (2x the base) on the MAIN loop. max_turns
                    # lets the FINAL turn's tool (e.g. submit_query) run before
                    # the loop stops.
                    max_turns=_MAX_TURNS,
                    hooks={
                        "PreToolUse": [
                            HookMatcher(
                                matcher="mcp__bird-interact-tools__submit_query",
                                hooks=[pre_query_gate],
                            ),
                            HookMatcher(hooks=[wall_clock_deny]),
                        ],
                        "PostToolUse": [
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
                    system_prompt=build_discovery_prompt(
                        with_ask_user=False, query_mode="slayer"
                    ),
                    mcp_servers=discovery_mcp_servers,
                    allowed_tools=list(discovery_tools),
                    tools=[],
                    setting_sources=[],
                    model=native_model_id(self.model),
                    effort=self.reasoning_effort,
                    # Per-answer turn cap; the per-task number of ask_discovery
                    # rounds is bounded by the DiscoveryChannel call cap.
                    max_turns=DISCOVERY_MAX_TURNS,
                    # Discovery work counts against the SAME per-task wall-clock
                    # budget as main (shared context_state); without these the
                    # introspection client could run outside the guardrails
                    # while main is blocked inside ask_discovery (Codex PR #56).
                    hooks={
                        "PreToolUse": [HookMatcher(hooks=[wall_clock_deny])],
                        "PostToolUse": [HookMatcher(hooks=[wall_clock_warning])],
                    },
                )

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
            )
            usage_tracker.finalize()
        except Exception as e:
            usage_tracker.finalize()
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
                    # DEV-1535 follow-up: surface asks_used as
                    # n_ask_user_calls so the one-shot variant is
                    # field-symmetric with the a-interact variant. For
                    # one-shot SLAYER the value is always 0 (tool list
                    # excludes ask_user), but writing 0 rather than
                    # absent keeps the usage shape consistent across
                    # all four claude_sdk_otf flavors.
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
                    user_sim_n_asks=None,
                )
                if _ann_from_disk and _is_genuine_miss(_cascade):
                    _autopsy_result = await run_autopsy(
                        task_annotation=task_annotation,
                        trajectory=trajectory,
                        slayer_storage_dir=slayer_storage_dir,
                        miss_diagnostics=_cascade.miss_diagnostics,
                        model=self.model,
                        is_one_shot=True,
                    )
            except Exception:
                logger.exception(
                    "[otf] autopsy grading failed for %s; continuing without autopsy",
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
                "usage": {
                    **accum.model_dump(),
                    "n_ask_user_calls": (ctx_dict or {}).get(
                        "asks_used", 0,
                    ),
                },
                "phase1_observation_audited": result.get("phase1_observation_audited"),
                "phase1_observation_original": result.get("phase1_observation_original"),
                "_autopsy": _autopsy_result,
                "_task_annotation": task_annotation,
            },
            deleted_kb_ids=deleted_kb_ids,
            slayer_storage_dir=slayer_storage_dir,
        )
