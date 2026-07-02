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

import dataclasses
import logging
from pathlib import Path

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    create_sdk_mcp_server,
)

# Reuse the sibling claude_sdk adapter's contextvar + native tools. These
# tools read/write the SAME `_ctx_var` we set below, so binding them here
# is sound.
from bird_interact_agents.agents.claude_sdk.agent import (
    _ctx_var,
    SdkUsageTracker,
    get_all_external_knowledge_names,
    get_all_knowledge_definitions,
    get_knowledge_definition,
    query,
    submit_query,
)
from bird_interact_agents.agents.claude_sdk.sdk_env import (
    hermetic_claude_sdk_session,
)
from bird_interact_agents.slayer_otf.timing import otf_timer
from bird_interact_agents.agents.claude_sdk_otf.prompts import (
    SLAYER_OTF_ONE_SHOT,
)
from bird_interact_agents.agents._pre_encoded import (
    resolve_pre_encoded_storage_dir,
    strip_write_slayer_tools,
    validate_pre_encoded_source,
    WRITE_SLAYER_TOOL_NAMES,
)
from bird_interact_agents.agents._pre_encoded_prompts import (
    SLAYER_PRE_ENCODED_ONE_SHOT,
)
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.provider_registry import is_supported_agent_model
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.harness import (
    MAX_MODEL_TURNS,
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
from bird_interact_agents.slayer_pipeline.filter_normalization import (
    normalize_tool_filters,
)
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


# SLayer MCP tools the OTF agent may call. The existing claude_sdk slayer
# mode is read-only; this adapter ADDS the write tools (create_model /
# edit_model / save_memory / validate_models) so the agent can encode KB
# items.
#
# DEV-1534 Fix C: `query` and `query_nested` are served by our own
# wrappers on the bird-interact-tools SDK MCP server (registered via
# `_KNOWLEDGE_TOOLS` below) so the agent can opt out of filter
# normalization mid-flight via the `normalize_filters` parameter. They
# are NOT in this allowlist.
SLAYER_MCP_TOOLS = [
    "help",
    "list_datasources",
    "models_summary",
    "inspect_model",
    "search",
    "recommend_root_model",
    "create_model",
    "edit_model",
    "save_memory",
    "validate_models",
]


def _slayer_tool_names() -> list[str]:
    return [f"mcp__slayer__{t}" for t in SLAYER_MCP_TOOLS]


# DEV-1548: SLayer MCP tools the OTF agent never (or essentially never)
# calls in steady-state slayer-mode runs but whose JSON Schemas would
# otherwise sit in the per-turn cacheable prefix. Listed in
# `ClaudeAgentOptions.disallowed_tools=` to remove them from the model's
# context entirely (`allowed_tools=` only gates auto-execute permission,
# not visibility). A 399-trajectory audit showed zero calls for the
# first five names; `ingest_datasource_models` had one exploratory call
# the OTF bootstrap path handles separately.
#
# `save_memory` is INTENTIONALLY NOT listed here. The audit shows zero
# calls today, but the encoder retains the affordance on the allow-list
# (see `SLAYER_MCP_TOOLS` above) — preserving headroom in case future
# prompts re-engage it. Filed as a follow-up: if the next post-merge
# trajectory sweep also shows 0 `save_memory` calls across a comparable
# sample, open a sibling Linear issue to shave the residual.
SLAYER_MCP_DISALLOWED_TOOL_NAMES: list[str] = [
    "mcp__slayer__forget_memory",
    "mcp__slayer__get_datasource_priority",
    "mcp__slayer__set_datasource_priority",
    "mcp__slayer__create_datasource",
    "mcp__slayer__delete_datasource",
    "mcp__slayer__ingest_datasource_models",
]


# Full MCP tool names whose payloads carry backing-query filter strings
# that we need to normalize (lower(trim(col)) wrap) before SLayer
# persists them on a model. Without this hook, the agent's `create_model`
# / `edit_model` calls baked raw text-equality filters into the stored
# entity definition — and the DEV-1534 Fix C `query` / `query_nested`
# wrappers can never repair filters hidden inside the model's own
# backing SQL (Codex post-merge catch; the pydantic_ai_otf_encode
# adapter already runs the same `normalize_tool_filters` call on every
# write).
_WRITE_TOOLS_NEEDING_NORMALIZATION = (
    "mcp__slayer__create_model",
    "mcp__slayer__edit_model",
)
_NORMALIZE_WRITE_FILTERS_MATCHER = "|".join(_WRITE_TOOLS_NEEDING_NORMALIZATION)


async def _normalize_write_tool_filters_hook(input_data, tool_use_id, context):
    """PreToolUse hook: rewrite ``create_model`` / ``edit_model`` payloads
    so their backing-query ``filters`` strings are wrapped in
    ``lower(trim(col)) = '<lower>'`` before SLayer persists them.

    Returns the SDK's ``updatedInput`` directive on the
    ``hookSpecificOutput`` envelope — the Claude Agent SDK applies it to
    the tool invocation (see ``PreToolUseHookSpecificOutput`` in
    ``claude_agent_sdk.types``). When normalization is a no-op (no
    in-scope filters), the hook still returns the deep-copy
    ``normalize_tool_filters`` produced — harmless and keeps the hook
    side-effect-free with respect to the original input dict.
    """
    tool_name = input_data.get("tool_name") or ""
    if tool_name not in _WRITE_TOOLS_NEEDING_NORMALIZATION:
        return {}
    # Strip the `mcp__slayer__` prefix so `normalize_tool_filters` sees
    # the bare SLayer tool name it expects (`create_model` / `edit_model`).
    bare = tool_name.split("__", 2)[-1]
    tool_input = input_data.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return {}
    updated = normalize_tool_filters(bare, tool_input)
    if not isinstance(updated, dict):
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "updatedInput": updated,
        }
    }


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


# Native (in-process) tools registered under the "bird-interact-tools"
# MCP server. one-shot decides autonomously (no user simulator), so the
# knowledge-lookup tools + `submit_query` are the only natives the agent
# needs. (The a-interact flavor adds `ask_user` in
# ``claude_sdk_otf_ainteract``.)
#
# DEV-1534 Fix C: `query` / `query_nested` are bird-interact-tools
# wrappers (not SLayer subprocess tools) so the agent can opt out of
# filter normalization mid-flight.
_KNOWLEDGE_TOOLS = [
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    query,
]


def _select_tools(eval_mode: str) -> list:
    if eval_mode != "one-shot":
        raise ValueError(
            "claude_sdk_otf supports only eval_mode='one-shot' "
            "(use claude_sdk_otf_ainteract for a-interact); "
            f"got {eval_mode!r}"
        )
    return [*_KNOWLEDGE_TOOLS, submit_query]


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

    Supports Anthropic and registry open-weight models (DEV-1579); an
    unsupported model short-circuits with a skip-shaped row. slayer-query-mode only;
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
        # read-only pre-encoded mode. `slayer_setup` is derived upstream
        # ("pre-encoded" when a source is set, else "on-the-fly"); enforce the
        # two consistent shapes and reject contradictions.
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

        # DEV-1579: claude_sdk_otf now runs Anthropic AND registry open-weight
        # models (the hermetic session layers the provider base-url/auth). A
        # genuinely-unsupported provider (not Anthropic, not in the registry)
        # still gets a graceful skip row.
        if not is_supported_agent_model(self.model):
            msg = (
                f"claude_sdk_otf requires an Anthropic or registry open-weight "
                f"model; got {self.model!r}. Skipped — use --framework "
                "claude_sdk_otf_encode for non-supported models."
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
            # DEV-1586: pre-encoded mode copies an ALREADY-encoded reference
            # (otf=encoding-agent output, custom=hand-curated slayer_models)
            # into a per-task scratch dir with this task's deleted KBs masked,
            # and the agent only introspects it. On-the-fly mode (default)
            # copies the deterministic OTF cache and the agent encodes into it.
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
                "slayer": slayer_mcp_stdio_config(
                    slayer_storage_dir, ingest_on_startup=False,
                ),
            }
            # DEV-1586: pre-encoded mode drops the SLayer WRITE tools (the
            # agent introspects only). On-the-fly keeps the full whitelist.
            slayer_tools = (
                strip_write_slayer_tools(SLAYER_MCP_TOOLS)
                if self.pre_encoded_source else SLAYER_MCP_TOOLS
            )
            tool_names.extend(f"mcp__slayer__{t}" for t in slayer_tools)
            # Also HIDE the write tools' schemas from the model's cacheable
            # prefix (allowed_tools only gates auto-execute; disallowed_tools
            # removes visibility — see DEV-1548).
            disallowed_tool_names = list(SLAYER_MCP_DISALLOWED_TOOL_NAMES)
            if self.pre_encoded_source:
                disallowed_tool_names.extend(sorted(WRITE_SLAYER_TOOL_NAMES))

            # Per-task hook factories — must be created here (not on the
            # agent constructor) to avoid cross-task state bleed.
            pre_query_gate, post_tool_tracker = _make_query_before_submit_guard()

            # DEV-1586: the create_model/edit_model filter-normalization hook
            # is moot in pre-encoded mode (no write tools). Build the
            # PreToolUse matcher list conditionally.
            pre_tool_use_matchers = [
                HookMatcher(
                    matcher="mcp__bird-interact-tools__submit_query",
                    hooks=[pre_query_gate],
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
                    # DEV-1548: hide the SLayer MCP tools the agent never calls
                    # from the model's per-turn cacheable prefix. allowed_tools
                    # only gates auto-execute permission; disallowed_tools is
                    # what removes the JSON Schema from the model's view.
                    disallowed_tools=disallowed_tool_names,
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
                    # Pin the requested model (bare id, no provider prefix) so
                    # --agent-model takes effect instead of the claude CLI's
                    # configured default.
                    model=native_model_id(self.model),
                    # Reasoning-effort level (None => SDK default).
                    effort=self.reasoning_effort,
                    # Native turn cap (2x the base). Unlike a manual break on the
                    # receive stream, max_turns lets the FINAL turn's tool (e.g.
                    # submit_query) execute before the run stops — the off-by-one
                    # that previously dropped a last-turn submission.
                    max_turns=_MAX_TURNS,
                    hooks={
                        "PreToolUse": pre_tool_use_matchers,
                        "PostToolUse": [
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
                    try:
                        _data: object = dataclasses.asdict(msg)
                    except Exception:  # noqa: BLE001
                        _data = str(msg)
                    trajectory.append({"type": str(type(msg).__name__), "data": _data})
                    usage_tracker.observe(msg)
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
                    conditions=task_data.get("conditions"),
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
