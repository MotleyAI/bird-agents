"""DEV-1555 Stage 1: context-budget tracking for claude_sdk_otf* agents.

Open-weight backends cap usable context well below Claude's effective
window (measured claude_sdk_otf slayer sessions peak at 262K tokens). The
runner tracks the live context size from each streamed AssistantMessage's
per-turn usage and a PostToolUse hook warns the agent — once at 80% and
once more at 90% of the model's window — to submit its best candidate
before the window overflows.

Stage 1 keeps Claude runs behavior-identical: anthropic models resolve to
a 1M window (observed sessions ran to 262K, so the effective window is
far above the 200K default), which keeps both thresholds out of reach.
Stage 2 wires the open-weight provider registry into
``context_window_for``.
"""

from __future__ import annotations

import math
import os
import time

from bird_interact_agents.provider_registry import get_provider

# Shared with ``run.py``'s outer ``asyncio.wait_for`` safety net; the
# agent's hook reads the same env var so the soft warnings + hard deny
# fire BEFORE the outer cap kicks in.
_PER_TASK_TIMEOUT_ENV = "BIRD_INTERACT_PER_TASK_TIMEOUT_S"
# Codex r2: default to 0 (no cap) so the agent-side wall-clock hook
# stays consistent with ``run.py._DEFAULT_PER_TASK_TIMEOUT_S = 0.0``.
# Rate-limited cloud runs with throttled LLM back-offs were pushing
# legitimate retries past the prior 15-min default and converting
# them into permanent eval_failed; the outer wait_for was flipped to
# uncapped — the agent-side budget must mirror that or the deny hook
# starts blocking non-submit tools after 15 min anyway.
_DEFAULT_AGENT_BUDGET_S = 0.0


def per_task_timeout_s() -> float:
    """Per-task agent wall-clock budget (seconds).

    Reads ``BIRD_INTERACT_PER_TASK_TIMEOUT_S`` (shared with the outer
    ``asyncio.wait_for`` cap in ``run.py``). 0 / negative disables the
    agent-level enforcement (and the outer cap, which mirrors); the
    DEFAULT is now 0 (no cap), matching the outer default."""
    raw = os.environ.get(_PER_TASK_TIMEOUT_ENV)
    if raw is None:
        return _DEFAULT_AGENT_BUDGET_S
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_AGENT_BUDGET_S
    # `float("nan")` / `float("inf")` parse cleanly; nan breaks
    # `elapsed < budget_s` comparisons (always False, deny never fires)
    # and inf disables the cap entirely. Fall back to the default.
    if not math.isfinite(value):
        return _DEFAULT_AGENT_BUDGET_S
    return value

_DEFAULT_WINDOW = 200_000
_ANTHROPIC_WINDOW = 1_000_000

_WARN_FRACTION = 0.8
_FINAL_FRACTION = 0.9


def context_window_for(model: str) -> int:
    """Context window (tokens) for a LiteLLM-style ``provider/model`` string."""
    if model.split("/", 1)[0] == "anthropic":
        return _ANTHROPIC_WINDOW
    spec = get_provider(model)
    if spec is not None:
        _, _, native_id = model.partition("/")
        return spec.model_context_windows.get(
            native_id, spec.default_context_window,
        )
    return _DEFAULT_WINDOW


def _usage_value(usage: object, key: str) -> int:
    if usage is None:
        return 0
    val = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    return val or 0


_CHARS_PER_TOKEN = 4.0
"""Rough Anthropic-tokenizer ratio (≈3.5-4 chars/token in English). Used
as the mid-stream estimator for backends whose per-turn AssistantMessage
usage is unreliable (Moonshot/Kimi)."""


def _estimate_message_chars(msg: object) -> int:
    """Best-effort character count of a streamed message's content blocks.

    Walks the SDK's content list and sums the lengths of every
    ``text`` / ``thinking`` block plus the JSON-stringified
    ``input`` of every ``tool_use`` block and the ``content`` of every
    ``tool_result`` block. Misses some fixed-cost overhead per block
    but mirrors what the LLM actually pays for in the input context.
    """
    import json as _json

    content = getattr(msg, "content", None)
    if content is None:
        return 0
    total = 0
    blocks = content if isinstance(content, list) else [content]
    for b in blocks:
        if isinstance(b, dict):
            t = b.get("type")
            if t in ("text", "thinking"):
                total += len(b.get("text") or b.get("thinking") or "")
            elif t == "tool_use":
                inp = b.get("input")
                if inp is not None:
                    total += len(_json.dumps(inp))
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, str):
                    total += len(c)
                elif isinstance(c, list):
                    for sub in c:
                        if isinstance(sub, dict):
                            total += len(sub.get("text") or "")
        else:
            # Dataclass-style: extract any text-like attribute.
            for attr in ("text", "thinking"):
                val = getattr(b, attr, None)
                if isinstance(val, str):
                    total += len(val)
            inp = getattr(b, "input", None)
            if inp is not None:
                total += len(_json.dumps(inp))
            tr = getattr(b, "content", None)
            if isinstance(tr, str):
                total += len(tr)
    return total


def update_context_tokens(state: dict, msg: object) -> None:
    """Record the LATEST call's context size from a streamed message.

    Latest, not max: compaction can legitimately shrink the context, and
    the warning should reflect where the session is now.

    Codex r1 / DEV-1555: also extract from ``ResultMessage.usage``.
    Anthropic backends populate per-turn ``AssistantMessage.usage``
    correctly, but Moonshot/Kimi reports all-zero on each
    ``AssistantMessage`` and only fills cumulative usage on the
    terminal ``ResultMessage`` (documented at
    ``claude_sdk/agent.py:SdkUsageTracker``).

    Codex r3: ``ResultMessage`` is the SESSION-terminal message — by
    the time it arrives, the agent has finished and no further
    PostToolUse hooks fire. Reading only ResultMessage means the 80%/
    90% warnings would never fire for Moonshot/Kimi mid-stream, which
    defeats the whole point of the budget hook for the open-weight
    target. When AssistantMessage reports zero usage, fall back to a
    char-based estimate (rough Anthropic-tokenizer ratio) so the
    warnings fire at approximately the right turn count.
    """
    msg_type = type(msg).__name__
    # AssistantMessage + ResultMessage carry the LLM-reported usage.
    # UserMessage in the SDK trajectory carries the agent's tool
    # results — Codex r4: those are the dominant context consumers
    # (schema dumps, query rows). Counting only AssistantMessage
    # under-estimates the running context by a wide margin for
    # providers whose per-turn usage is unreliable (Moonshot/Kimi).
    if msg_type not in ("AssistantMessage", "ResultMessage", "UserMessage"):
        return
    usage = getattr(msg, "usage", None)
    tokens = 0
    if usage is not None:
        tokens = (
            _usage_value(usage, "input_tokens")
            + _usage_value(usage, "cache_read_input_tokens")
            + _usage_value(usage, "cache_creation_input_tokens")
        )
    if tokens > 0:
        # Real usage available — authoritative.
        state["context_tokens"] = tokens
        return
    # Zero-usage path (AssistantMessage on Moonshot/Kimi, or
    # UserMessage which has no usage attached at all): grow the
    # running char-based estimate so the PostToolUse hook has
    # something to compare against mid-stream.
    if msg_type in ("AssistantMessage", "UserMessage"):
        chars = _estimate_message_chars(msg)
        if chars > 0:
            running = state.get("context_chars_estimate", 0) + chars
            state["context_chars_estimate"] = running
            state["context_tokens"] = int(running / _CHARS_PER_TOKEN)


def make_context_budget_hook(state: dict, window: int):
    """Build the PostToolUse hook emitting the one-shot 80%/90% warnings.

    ``state`` is the per-task dict the stream consumer feeds via
    ``update_context_tokens``; the hook only reads it.
    """
    fired = {"warn": False, "final": False}
    warn_at = int(window * _WARN_FRACTION)
    final_at = int(window * _FINAL_FRACTION)

    def _envelope(text: str) -> dict:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": text,
            }
        }

    async def context_budget_warning(input_data, tool_use_id, context):
        tokens = state.get("context_tokens") or 0
        if tokens > final_at and not fired["final"]:
            fired["final"] = True
            fired["warn"] = True
            return _envelope(
                f"[CONTEXT BUDGET] FINAL WARNING: ~{tokens:,} tokens of the "
                f"{window:,}-token context window are used (>90%). Submit "
                "your best candidate NOW — overflowing the window aborts "
                "the task and scores zero."
            )
        if tokens > warn_at and not fired["warn"]:
            fired["warn"] = True
            return _envelope(
                f"[CONTEXT BUDGET] ~{tokens:,} tokens of the {window:,}-token "
                "context window are used (>80%). Stop broad exploration, "
                "converge on a candidate, verify it once, and submit before "
                "the window runs out."
            )
        return {}

    return context_budget_warning


# ---------------------------------------------------------------------------
# DEV-1555 follow-up: wall-clock budget enforced AT THE AGENT LEVEL.
#
# Per-task wall-clock cap was previously enforced from OUTSIDE the SDK via
# ``asyncio.wait_for`` in ``run.py``. When it fired the agent's in-memory
# trajectory was lost (last seen on Kimi r7 → traj=0, cost=0, no usage).
# Same shape as ``make_context_budget_hook`` but the clock is wall-time:
# warn at 80% / 90%, then DENY non-submit tools at 100% so the agent is
# forced to call submit_query / submit_sql next. The outer
# ``asyncio.wait_for`` stays as a runaway safety net at budget + grace.
# ---------------------------------------------------------------------------

# Full MCP tool names that the deny hook allows past the wall-clock
# deadline. Any name CONTAINING one of these substrings is allowed (the
# agents register the submit tools under
# ``mcp__bird-interact-tools__submit_query`` /
# ``mcp__bird-interact-tools__submit_sql``).
_SUBMIT_TOOL_NAMES = ("submit_query", "submit_sql")


def update_wall_clock_start(state: dict) -> None:
    """Stamp ``time.monotonic()`` onto ``state["wall_clock_start"]``.

    Call once per task right before entering the SDK receive loop. Last
    write wins (mirrors ``update_context_tokens`` 'latest, not max'
    semantics)."""
    state["wall_clock_start"] = time.monotonic()


def make_wall_clock_budget_hook(
    state: dict,
    *,
    budget_s: float | None,
    submit_tool: str,
) -> tuple:
    """Build (PostToolUse warning, PreToolUse deny) for a wall-clock budget.

    The warning hook fires once at 80% and once at 90% of ``budget_s`` —
    same one-shot pattern as ``make_context_budget_hook``. The deny hook
    refuses every PreToolUse for a non-submit tool past 100% of
    ``budget_s``, forcing the agent to call ``{submit_tool}`` next so the
    trajectory + best-candidate query are preserved.

    ``budget_s`` of ``None``, ``0``, or negative disables BOTH hooks (they
    become no-ops). Preserves the existing
    ``BIRD_INTERACT_PER_TASK_TIMEOUT_S=0`` UX.

    ``submit_tool`` is the bare tool name (``"submit_query"`` for slayer
    flavors, ``"submit_sql"`` for raw flavors). The deny check matches by
    substring against the called tool name so the bird-interact-tools MCP
    prefix is handled transparently.
    """
    disabled = budget_s is None or budget_s <= 0
    fired = {"warn": False, "final": False}
    warn_at = (budget_s or 0) * _WARN_FRACTION
    final_at = (budget_s or 0) * _FINAL_FRACTION

    def _elapsed() -> float | None:
        start = state.get("wall_clock_start")
        if start is None:
            return None
        return time.monotonic() - start

    async def wall_clock_budget_warning(input_data, tool_use_id, context):
        if disabled:
            return {}
        elapsed = _elapsed()
        if elapsed is None:
            return {}
        if elapsed > final_at and not fired["final"]:
            fired["final"] = True
            fired["warn"] = True
            remaining = max(0.0, budget_s - elapsed)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": (
                        f"[WALL-CLOCK BUDGET] FINAL WARNING: {elapsed:.0f}s "
                        f"of the {budget_s:.0f}s budget used (>90%). "
                        f"{remaining:.0f}s left before non-submit tools are "
                        f"denied — call {submit_tool} NOW with your best "
                        "candidate."
                    ),
                }
            }
        if elapsed > warn_at and not fired["warn"]:
            fired["warn"] = True
            remaining = max(0.0, budget_s - elapsed)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": (
                        f"[WALL-CLOCK BUDGET] {elapsed:.0f}s of the "
                        f"{budget_s:.0f}s budget used (>80%). "
                        f"{remaining:.0f}s remain before forced submit; if "
                        f"you have a candidate answer, call {submit_tool} "
                        "now."
                    ),
                }
            }
        return {}

    async def wall_clock_budget_deny(input_data, tool_use_id, context):
        if disabled:
            return {}
        elapsed = _elapsed()
        if elapsed is None or elapsed < budget_s:
            return {}
        tool_name = input_data.get("tool_name") or ""
        # Codex r1: match the LEAF tool name (after the final `__`
        # separator), not any substring. `s in tool_name` would let any
        # tool whose name CONTAINS `submit_query` / `submit_sql` (e.g.
        # an unrelated third-party MCP server's `do_submit_query_thing`)
        # bypass the deny gate.
        tool_leaf = tool_name.rsplit("__", 1)[-1]
        if tool_leaf == submit_tool or tool_leaf in _SUBMIT_TOOL_NAMES:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"Wall-clock budget exhausted ({elapsed:.0f}s / "
                    f"{budget_s:.0f}s). Submit your best candidate "
                    f"immediately via {submit_tool} — further tool calls "
                    "will be denied until you do."
                ),
            }
        }

    return wall_clock_budget_warning, wall_clock_budget_deny
