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
from pathlib import Path

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    create_sdk_mcp_server,
)

from bird_interact_agents.agents.claude_sdk.agent import (
    _ctx_var,
    SdkUsageTracker,
    ask_user,
    get_all_external_knowledge_names,
    get_all_knowledge_definitions,
    get_knowledge_definition,
    query,
    submit_query,
)
from bird_interact_agents.agents.claude_sdk.sdk_env import (
    hermetic_claude_sdk_session,
    serialize_sdk_message,
)
from bird_interact_agents.slayer_otf.timing import otf_timer
from bird_interact_agents.agents.claude_sdk_otf.agent import (
    _MAX_TURNS,
    _NORMALIZE_WRITE_FILTERS_MATCHER,
    _SLAYER_SEARCH_TOOL,
    SLAYER_MCP_TOOLS,
    _force_compact_search_hook,
    _make_query_before_submit_guard,
    _make_turn_budget_hook,
    _normalize_write_tool_filters_hook,
    _slayer_tool_names,
)
from bird_interact_agents.agents._slayer_tool_surface import (
    derive_disallowed_slayer_tools,
)
from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
    SLAYER_OTF_AINTERACT,
)
from bird_interact_agents.agents._pre_encoded import (
    resolve_pre_encoded_storage_dir,
    strip_write_slayer_tools,
    validate_pre_encoded_source,
)
from bird_interact_agents.agents._pre_encoded_prompts import (
    SLAYER_PRE_ENCODED_AINTERACT,
)
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.provider_registry import is_supported_agent_model
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.harness import (
    SampleStatus,
    _ambiguity_count,
    finalize_result_row,
    load_db_data_if_needed,
    materialize_task_db,
    slayer_mcp_stdio_config,
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


# DEV-1534 Fix C: `query` / `query_nested` are bird-interact-tools
# wrappers (not SLayer subprocess tools) so the agent can opt out of
# filter normalization mid-flight via the `normalize_filters` parameter.
_KNOWLEDGE_TOOLS = [
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    query,
]


def _select_tools(eval_mode: str) -> list:
    if eval_mode != "a-interact":
        raise ValueError(
            "claude_sdk_otf_ainteract supports only eval_mode='a-interact' "
            "(use claude_sdk_otf for one-shot); "
            f"got {eval_mode!r}"
        )
    return [*_KNOWLEDGE_TOOLS, ask_user, submit_query]


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

    Supports Anthropic and registry open-weight models (DEV-1579); an
    unsupported model short-circuits with a skip-shaped row. Bound to
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

        # DEV-1579: this agent now runs Anthropic AND registry open-weight
        # models (the hermetic session layers the provider base-url/auth). A
        # genuinely-unsupported provider (not Anthropic, not in the registry)
        # still gets a graceful skip row.
        if not is_supported_agent_model(self.model):
            msg = (
                f"claude_sdk_otf_ainteract requires an Anthropic or registry "
                f"open-weight model; got {self.model!r}. Skipped — use "
                "--framework claude_sdk_otf_encode for non-supported models."
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
            load_db_data_if_needed(db_name, data_path_base)
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
                # DEV-1613: agent model for the in-task N5 judge.
                "agent_model": self.model,
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
            prompt = _build_prompt(
                eval_mode, task_data, budget, self.pre_encoded_source,
            )

            server = create_sdk_mcp_server(
                name="bird-interact-tools", version="1.0.0", tools=tools,
            )
            tool_names = [f"mcp__bird-interact-tools__{t.name}" for t in tools]

            mcp_servers: dict = {
                "bird-interact-tools": server,
                # DEV-1508: like the sibling claude_sdk_otf adapter, the
                # per-task slayer storage here comes from the
                # `ensure_db_cache`-backed OTF cache (full post-ingestion
                # state). Skip `--ingest-on-startup` so the Claude Agent
                # SDK (no MCP startup-timeout knob) doesn't proceed with
                # slayer status='pending' on big-schema DBs.
                "slayer": slayer_mcp_stdio_config(
                    slayer_storage_dir, ingest_on_startup=False,
                ),
            }
            # DEV-1586: pre-encoded mode drops the SLayer WRITE tools (the
            # agent introspects only); on-the-fly keeps the full whitelist.
            # DEV-1644: the disallowed set is DERIVED as the complement of the
            # (mode-specific) allow-list against the live SLayer surface, so a
            # write-stripped allow-list hides the write schemas automatically.
            if self.pre_encoded_source:
                slayer_tools = strip_write_slayer_tools(SLAYER_MCP_TOOLS)
            else:
                slayer_tools = SLAYER_MCP_TOOLS
            tool_names.extend(f"mcp__slayer__{t}" for t in slayer_tools)
            disallowed_tool_names = derive_disallowed_slayer_tools(slayer_tools)

            # Per-task hook factories — never share state across tasks.
            pre_submit_gate, post_ask_counter, post_nag = _make_ask_user_guards()
            pre_query_gate, post_tool_tracker = _make_query_before_submit_guard()

            # DEV-1586: the create_model/edit_model filter-normalization hook
            # is moot in pre-encoded mode (no write tools).
            pre_tool_use_matchers = [
                HookMatcher(
                    matcher="mcp__bird-interact-tools__submit_query",
                    # ask_user gate runs first; query gate runs second.
                    hooks=[pre_submit_gate, pre_query_gate],
                ),
                # DEV-1591: hardwire SLayer `search` to compact=True so broad
                # discovery never drags full per-entity renders into context.
                HookMatcher(
                    matcher=_SLAYER_SEARCH_TOOL,
                    hooks=[_force_compact_search_hook],
                ),
            ]
            if not self.pre_encoded_source:
                pre_tool_use_matchers.append(
                    HookMatcher(
                        matcher=_NORMALIZE_WRITE_FILTERS_MATCHER,
                        hooks=[_normalize_write_tool_filters_hook],
                    )
                )

            # DEV-1579: build the agent's options from the policy-owned env
            # kwargs (telemetry-disable + hermetic CLAUDE_CONFIG_DIR + any
            # registry session env / thinking). The agent only supplies its
            # own tool surface / hooks.
            def _build_options(_opt_kwargs: dict) -> ClaudeAgentOptions:
                return ClaudeAgentOptions(
                    **_opt_kwargs,
                    system_prompt=prompt,
                    mcp_servers=mcp_servers,
                    allowed_tools=tool_names,
                    # DEV-1548: same cacheable-prefix shrink as the sibling
                    # one-shot adapter. The constant lives in
                    # claude_sdk_otf.agent (imported above) so a single edit
                    # propagates to both adapters; the negative-assertion test
                    # in test_claude_sdk_otf_disallowed_slayer_tools.py pins
                    # the symmetry.
                    disallowed_tools=disallowed_tool_names,
                    tools=[],
                    setting_sources=[],
                    model=native_model_id(self.model),
                    effort=self.reasoning_effort,
                    max_turns=_MAX_TURNS,
                    hooks={
                        "PreToolUse": pre_tool_use_matchers,
                        "PostToolUse": [
                            HookMatcher(
                                matcher=_ASK_USER_TOOL,
                                hooks=[post_ask_counter],
                            ),
                            HookMatcher(hooks=[post_nag]),
                            HookMatcher(hooks=[_make_turn_budget_hook(_MAX_TURNS)]),
                            # Must be last so it captures the true last-completed
                            # tool name after all other PostToolUse hooks have run.
                            HookMatcher(hooks=[post_tool_tracker]),
                        ],
                    },
                )

            # DEV-1579: the hermetic session owns CLAUDE_CONFIG_DIR isolation,
            # API-key auth enforcement, the MCP parity assertion, and cleanup.
            # DEV-1561: wrap `__aenter__` in `otf_timer` (via enter_cm_factory)
            # so a failed initialize handshake still emits `.error elapsed_s=…`.
            async with hermetic_claude_sdk_session(
                self.model,
                mcp_servers=mcp_servers,
                build_options=_build_options,
                enter_cm_factory=lambda: otf_timer(
                    "run_task.sdk_client_enter", instance_id=instance_id,
                ),
            ) as client:
                await client.query(task_data["amb_user_query"])
                async for msg in client.receive_response():
                    # DEV-1639: serialize_sdk_message stamps a per-turn `ts` so
                    # the persisted trajectory supports 5m-vs-1h cache analysis.
                    trajectory.append(serialize_sdk_message(msg))
                    usage_tracker.observe(msg)
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
                    conditions=task_data.get("conditions"),
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
