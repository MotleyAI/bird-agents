"""Shared serialization helpers for pydantic-ai ``AgentRun`` capture.

Both the original ``pydantic_ai`` adapter and the new
``pydantic_ai_recursive`` adapter need to dump a run's full message
history to JSON-safe dicts and walk it for per-tool call/error
statistics. Centralising them here keeps one source of truth and avoids
drift between adapters.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelMessagesTypeAdapter


# Cap on the number of error-sample blobs persisted per task. The samples
# are for ad-hoc inspection; full error history can always be reconstructed
# from the live run logs if needed.
_TOOL_ERROR_SAMPLES_PER_TASK = 10
_TOOL_ERROR_SAMPLE_CHARS = 400


def _serialize_messages(agent_run: Any) -> list[dict]:
    """Serialize a full PydanticAI message history to JSON-safe dicts.

    Returns ``[]`` if the messages can't be retrieved or serialized —
    this is a best-effort debug channel and must not block the run.
    """
    try:
        messages = list(agent_run.all_messages())
    except Exception:  # noqa: BLE001 — defensive
        return []
    try:
        return ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    except Exception:  # noqa: BLE001 — fall back to part-by-part dump
        out: list[dict] = []
        for m in messages:
            try:
                out.append(m.model_dump(mode="json"))
            except Exception:  # noqa: BLE001
                out.append({"_repr": repr(m)})
        return out


def _extract_tool_stats(agent_run: Any) -> dict | None:
    """Walk PydanticAI's recorded message history and produce per-tool
    call/error statistics for offline failure-mode analysis.

    Counts:
    - ``ToolCallPart`` instances per ``tool_name`` → successful tool
      invocations (from the agent's POV — the tool was found, args
      parsed, body ran).
    - ``RetryPromptPart`` instances per ``tool_name`` → erroring
      invocations the runtime asked the model to retry (Pydantic
      validation errors on tool args, missing tool name, ``ModelRetry``
      raised inside a tool body, plain text where structured output was
      expected). This is the harness's cleanest signal that *something
      went wrong inside the tool layer*, separate from the submit_status
      path that records evaluator outcomes.

    Returns ``None`` if the trajectory can't be walked — best-effort
    metric, must not block the run.
    """
    try:
        messages = list(agent_run.all_messages())
    except Exception:  # noqa: BLE001 — defensive
        return None

    calls: dict[str, int] = {}
    errors: dict[str, int] = {}
    error_samples: list[dict[str, str]] = []

    for msg in messages:
        for part in getattr(msg, "parts", None) or []:
            kind = getattr(part, "part_kind", None)
            if kind == "tool-call":
                name = getattr(part, "tool_name", None) or "<unknown>"
                calls[name] = calls.get(name, 0) + 1
            elif kind == "retry-prompt":
                name = getattr(part, "tool_name", None) or "<unknown>"
                errors[name] = errors.get(name, 0) + 1
                if len(error_samples) < _TOOL_ERROR_SAMPLES_PER_TASK:
                    content = getattr(part, "content", None)
                    text = (
                        content if isinstance(content, str) else str(content)
                    )[:_TOOL_ERROR_SAMPLE_CHARS]
                    error_samples.append({"tool": name, "error": text})

    seen = set(calls) | set(errors)
    per_tool = sorted(
        ({"tool": t, "n_calls": calls.get(t, 0), "n_errors": errors.get(t, 0)}
         for t in seen),
        key=lambda x: (-x["n_calls"], x["tool"]),
    )
    return {
        "per_tool": per_tool,
        "total_calls": sum(calls.values()),
        "total_errors": sum(errors.values()),
        "error_samples": error_samples,
    }


def _count_turns(agent_run: Any) -> int | None:
    """Best-effort: number of ``ModelResponse`` entries in the run's
    full message history. Used as a coarse "how many round-trips did
    this take" metric."""
    try:
        return sum(
            1 for m in agent_run.all_messages()
            if type(m).__name__ == "ModelResponse"
        )
    except Exception:  # noqa: BLE001 — best-effort
        return None


# ---------------------------------------------------------------------------
# DEV-1535 follow-up — claude_sdk trajectory walkers.
#
# The claude_sdk family records its trajectory as a flat list of
# ``{"type": "AssistantMessage" | "UserMessage" | "ResultMessage" | ...,
#   "data": <dataclass-asdict>}`` items (see e.g.
# `agents/claude_sdk_otf_ainteract/agent.py:425`). Pre-fix only the
# pydantic_ai* adapters produced `tool_call_stats`; reconstructing the
# same shape from the claude_sdk trajectory was a manual post-hoc walk
# every time. These two helpers centralise the walk so finalize_result_row
# can backfill identically to the pydantic_ai path.
# ---------------------------------------------------------------------------


def _looks_like_claude_sdk_trajectory(trajectory: list[Any] | None) -> bool:
    """Discriminator used by the finalize_result_row dispatch — return
    True iff the trajectory shape matches the claude_sdk convention
    (every item is a dict with a `type` string and a DICT `data` value).
    A pydantic_ai trajectory (no `type` at the top level) returns False
    so the dispatcher doesn't mis-route.

    DEV-1535 r2 (Codex): also reject the claude_sdk_*_raw shape, which
    serialises `data` as a 500-char string (`str(msg)[:500]`) rather
    than a dataclass dict. The walker can't extract anything from
    strings — accepting these trajectories would have returned an
    empty stats dict, falsely indicating "0 tool calls / 0 errors"
    instead of "not extractable".
    """
    if not isinstance(trajectory, list) or not trajectory:
        return False
    for item in trajectory:
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("type"), str):
            return False
        if not isinstance(item.get("data"), dict):
            return False
    return True


def _is_tool_use_block(block: dict) -> bool:
    """True iff a serialized content block is a tool call.

    The trajectory serializer uses ``dataclasses.asdict(msg)`` and the
    Claude Agent SDK's ``ToolUseBlock`` dataclass has fields
    ``{id, name, input}`` with NO ``type`` field — so keying on
    ``block["type"] == "tool_use"`` never matched and produced all-zero
    stats. Detect structurally (``name`` + ``input`` present, and NOT a
    ``tool_use_id``-bearing result block), while still honouring an
    explicit ``type`` when a future serializer tags one.
    """
    if block.get("type") == "tool_use":
        return True
    return (
        "name" in block and "input" in block and "tool_use_id" not in block
    )


def _is_tool_result_block(block: dict) -> bool:
    """True iff a serialized content block is a tool result. The SDK's
    ``ToolResultBlock`` dataclass has ``{tool_use_id, content, is_error}``
    and no ``type`` field — detect via ``tool_use_id`` presence (honouring
    an explicit ``type`` when tagged)."""
    return block.get("type") == "tool_result" or "tool_use_id" in block


def extract_tool_stats_from_claude_sdk_trajectory(
    trajectory: list[Any] | None,
) -> dict | None:
    """Walk a claude_sdk-shaped trajectory and produce per-tool stats
    in the SAME shape as ``_extract_tool_stats`` (the pydantic_ai
    sibling). Output:

    ``{"per_tool": [{"tool": ..., "n_calls": ..., "n_errors": ...}, ...],
       "total_calls": ..., "total_errors": ...,
       "error_samples": [{"tool": ..., "error": ...}, ...]}``

    Counting:
    1. Build a ``tool_use_id → tool_name`` map from ``AssistantMessage``
       content blocks that ``_is_tool_use_block`` (``ToolUseBlock`` asdict:
       ``{id, name, input}`` — no ``type`` field, so detected structurally).
    2. ``n_calls[name]`` = count of those tool_use blocks.
    3. ``n_errors[name]`` = count of ``tool_result`` blocks (in
       ``UserMessage`` content) where ``is_error == True``; resolve
       name via the map (``"<unknown>"`` for unresolved ids).
    4. Capture up to ``_TOOL_ERROR_SAMPLES_PER_TASK`` error samples
       (text capped at ``_TOOL_ERROR_SAMPLE_CHARS``).

    Returns ``None`` if the trajectory doesn't look claude_sdk-shaped
    so the finalize_result_row dispatcher can safely chain.
    """
    if not _looks_like_claude_sdk_trajectory(trajectory):
        return None

    tool_use_id_to_name: dict[str, str] = {}
    calls: dict[str, int] = {}
    errors: dict[str, int] = {}
    error_samples: list[dict[str, str]] = []

    # First pass: register every tool_use block + count calls.
    for item in trajectory or []:
        if item.get("type") != "AssistantMessage":
            continue
        data = item.get("data") or {}
        if not isinstance(data, dict):
            continue
        content = data.get("content") or []
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if not _is_tool_use_block(block):
                continue
            name = block.get("name") or "<unknown>"
            calls[name] = calls.get(name, 0) + 1
            use_id = block.get("id")
            if isinstance(use_id, str):
                tool_use_id_to_name[use_id] = name

    # Second pass: count errors from tool_result blocks.
    for item in trajectory or []:
        if item.get("type") != "UserMessage":
            continue
        data = item.get("data") or {}
        if not isinstance(data, dict):
            continue
        content = data.get("content") or []
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if not _is_tool_result_block(block):
                continue
            if not block.get("is_error"):
                continue
            use_id = block.get("tool_use_id")
            name = tool_use_id_to_name.get(use_id, "<unknown>") \
                if isinstance(use_id, str) else "<unknown>"
            errors[name] = errors.get(name, 0) + 1
            if len(error_samples) < _TOOL_ERROR_SAMPLES_PER_TASK:
                raw = block.get("content")
                if isinstance(raw, list):
                    # Tool result content is a list of
                    # `{"type":"text","text":"…"}` blocks.
                    text_parts: list[str] = []
                    for c in raw:
                        if isinstance(c, dict):
                            t = c.get("text")
                            if isinstance(t, str):
                                text_parts.append(t)
                    text = "".join(text_parts)[:_TOOL_ERROR_SAMPLE_CHARS]
                else:
                    text = (raw if isinstance(raw, str) else str(raw))[
                        :_TOOL_ERROR_SAMPLE_CHARS
                    ]
                error_samples.append({"tool": name, "error": text})

    seen = set(calls) | set(errors)
    per_tool = sorted(
        ({"tool": t, "n_calls": calls.get(t, 0), "n_errors": errors.get(t, 0)}
         for t in seen),
        key=lambda x: (-x["n_calls"], x["tool"]),
    )
    return {
        "per_tool": per_tool,
        "total_calls": sum(calls.values()),
        "total_errors": sum(errors.values()),
        "error_samples": error_samples,
    }


def extract_claude_sdk_result_metadata(
    trajectory: list[Any] | None,
) -> dict | None:
    """Walk the trajectory in reverse for the last ``ResultMessage`` —
    the SDK's authoritative termination signal. Returns:

    ``{"sdk_num_turns": int | None, "sdk_stop_reason": str | None,
       "sdk_duration_ms": int | None, "sdk_duration_api_ms": int | None}``

    Missing fields on the ResultMessage are mapped to ``None`` rather
    than omitted, so the result-row schema is stable. Returns ``None``
    if no ResultMessage is present (e.g. very-early agent crash).
    """
    if not _looks_like_claude_sdk_trajectory(trajectory):
        return None
    for item in reversed(trajectory or []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "ResultMessage":
            continue
        data = item.get("data") or {}
        if not isinstance(data, dict):
            continue
        return {
            "sdk_num_turns": data.get("num_turns"),
            "sdk_stop_reason": data.get("stop_reason"),
            "sdk_duration_ms": data.get("duration_ms"),
            "sdk_duration_api_ms": data.get("duration_api_ms"),
        }
    return None
