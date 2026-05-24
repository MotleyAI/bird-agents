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
