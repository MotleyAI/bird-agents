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

from bird_interact_agents.provider_registry import get_provider

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


def update_context_tokens(state: dict, msg: object) -> None:
    """Record the LATEST call's context size from a streamed message.

    Latest, not max: compaction can legitimately shrink the context, and
    the warning should reflect where the session is now.
    """
    if type(msg).__name__ != "AssistantMessage":
        return
    usage = getattr(msg, "usage", None)
    if usage is None:
        return
    state["context_tokens"] = (
        _usage_value(usage, "input_tokens")
        + _usage_value(usage, "cache_read_input_tokens")
        + _usage_value(usage, "cache_creation_input_tokens")
    )


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
