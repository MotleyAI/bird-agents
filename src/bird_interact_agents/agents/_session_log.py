"""Per-sub-agent session logging.

One self-contained file per spawned agent so any single agent's session is
trivially isolatable — open one file — instead of being reconstructed from the
interleaved, unattributed MCP-server stdout. Used by the setup encoder (the
per-DB reference build, where N encoders run concurrently) and by the task-time
clarifier tree.

Each session produces two artifacts in ``sessions_dir``:

* ``<session_id>.md``  — a human-readable transcript: a header (role, status,
  model round-trips, per-tool call/error counts — the "thrashing fingerprint" —
  final output, error) followed by a step-by-step trace of every tool call,
  tool return, retry/error, and text part in order.
* ``<session_id>.json`` — the raw serialised message history (full fidelity).

A scope-level ``INDEX.md`` (written via :func:`write_index`) lists every session
in one table for triage at a glance.

This module is best-effort: a logging failure must never break a run, so every
public function swallows its own exceptions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bird_interact_agents.agents._run_capture import (
    _count_turns,
    _extract_tool_stats,
    _serialize_messages,
)

logger = logging.getLogger(__name__)

# Per-block truncation in the .md transcript. The .json sidecar keeps the full
# content; the .md is for fast human reading.
_TRUNCATE = 2000


def session_from_run(agent_run: Any) -> dict:
    """Extract the serialisable parts of a pydantic-ai ``AgentRun`` / run result.

    Returns a dict with ``messages`` / ``tool_call_stats`` / ``n_turns`` ready to
    pass straight into :func:`write_session`. Safe on ``None`` (e.g. when
    ``agent.iter()`` never produced a run object). Best-effort: although the
    extraction helpers each swallow their own errors, this is called from the
    setup encoder's failure-handling path, so wrap defensively — a logging
    extraction error must never abort the encode.
    """
    if agent_run is None:
        return {"messages": [], "tool_call_stats": None, "n_turns": None}
    try:
        return {
            "messages": _serialize_messages(agent_run),
            "tool_call_stats": _extract_tool_stats(agent_run),
            "n_turns": _count_turns(agent_run),
        }
    except Exception:  # noqa: BLE001 — logging extraction must not break the run
        logger.exception("session_from_run extraction failed")
        return {"messages": [], "tool_call_stats": None, "n_turns": None}


def write_session(
    sessions_dir: Any,
    session_id: str,
    *,
    messages: list[dict] | None,
    tool_call_stats: dict | None = None,
    n_turns: int | None = None,
    role: str | None = None,
    meta: dict | None = None,
    status: str | None = None,
    output: Any = None,
    error: Any = None,
    usage: Any = None,
    duration_s: float | None = None,
) -> dict:
    """Write ``<session_id>.md`` + ``<session_id>.json`` for one agent session.

    ``messages`` is the serialised history (from :func:`session_from_run` for the
    setup encoder, or an ``AgentRecord.messages`` list for the task path).
    Returns a one-row dict for :func:`write_index`. Never raises.
    """
    row = _index_row(
        session_id, role=role, meta=meta, status=status, error=error,
        tool_call_stats=tool_call_stats, n_turns=n_turns,
    )
    try:
        d = Path(sessions_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{session_id}.json").write_text(
            json.dumps(messages or [], indent=2, default=str)
        )
        (d / f"{session_id}.md").write_text(
            _render_md(
                session_id, messages or [], role=role, meta=meta, status=status,
                output=output, error=error, tool_call_stats=tool_call_stats,
                n_turns=n_turns, usage=usage, duration_s=duration_s,
            )
        )
    except Exception:  # noqa: BLE001 — logging must not break the run
        logger.exception("write_session failed for %s", session_id)
    return row


def write_index(sessions_dir: Any, rows: list[dict]) -> None:
    """Write ``INDEX.md`` — one table row per session for at-a-glance triage."""
    try:
        d = Path(sessions_dir)
        d.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Sessions index",
            "",
            f"{len(rows)} session(s).",
            "",
            "| session | role | kb | status | turns | tool calls | tool errs | error |",
            "|---|---|---|---|---|---|---|---|",
        ]
        cols = (
            "session_id", "role", "kb_id", "status", "n_turns",
            "tool_calls", "tool_errors", "error",
        )
        for r in sorted(rows, key=lambda r: str(r.get("session_id"))):
            cells = ["" if r.get(c) is None else str(r.get(c)) for c in cols]
            lines.append("| " + " | ".join(cells) + " |")
        (d / "INDEX.md").write_text("\n".join(lines) + "\n")
    except Exception:  # noqa: BLE001
        logger.exception("write_index failed for %s", sessions_dir)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _trunc(value: Any) -> str:
    s = "" if value is None else str(value)
    if len(s) <= _TRUNCATE:
        return s
    return s[:_TRUNCATE] + f"\n… [+{len(s) - _TRUNCATE} chars truncated]"


def _render_part(step: int, part: dict) -> str:
    kind = part.get("part_kind")
    if kind == "tool-call":
        return (
            f"\n**[{step}] → TOOL CALL `{part.get('tool_name')}`**\n"
            f"```json\n{_trunc(part.get('args'))}\n```"
        )
    if kind == "tool-return":
        return (
            f"\n**[{step}] ← return `{part.get('tool_name')}`** "
            f"(outcome={part.get('outcome')})\n```\n{_trunc(part.get('content'))}\n```"
        )
    if kind == "retry-prompt":
        return (
            f"\n**[{step}] ⚠️ RETRY/ERROR `{part.get('tool_name')}`**\n"
            f"```\n{_trunc(part.get('content'))}\n```"
        )
    if kind == "text":
        return f"\n**[{step}] TEXT**\n{_trunc(part.get('content'))}"
    if kind == "user-prompt":
        return f"\n**[{step}] USER**\n{_trunc(part.get('content'))}"
    if kind in ("system-prompt", "instructions"):
        return (
            f"\n**[{step}] {kind.upper()}**\n```\n{_trunc(part.get('content'))}\n```"
        )
    leftover = {k: v for k, v in part.items() if k != "part_kind"}
    return f"\n**[{step}] {kind}**\n```\n{_trunc(leftover)}\n```"


def _render_md(
    session_id: str,
    messages: list[dict],
    *,
    role: str | None,
    meta: dict | None,
    status: str | None,
    output: Any,
    error: Any,
    tool_call_stats: dict | None,
    n_turns: int | None,
    usage: Any,
    duration_s: float | None,
) -> str:
    lines = [f"# Session `{session_id}`", "", "## Header", f"- role: {role}"]
    for k, v in (meta or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append(f"- status: {status}")
    lines.append(f"- model round-trips (turns): {n_turns}")
    if duration_s is not None:
        lines.append(f"- duration_s: {duration_s:.1f}")
    if usage is not None:
        lines.append(f"- usage: {usage}")
    if tool_call_stats:
        lines.append(
            f"- tool calls: {tool_call_stats.get('total_calls')} | "
            f"tool errors: {tool_call_stats.get('total_errors')}"
        )
        for t in tool_call_stats.get("per_tool", []) or []:
            lines.append(
                f"    - {t.get('tool')}: {t.get('n_calls')} calls, "
                f"{t.get('n_errors')} errors"
            )
    if error:
        lines.append(f"- error: {_trunc(error)}")
    if output is not None:
        lines += ["", "## Final output", "```", _trunc(output), "```"]
    lines += ["", "## Step trace"]
    step = 0
    for msg in messages:
        for part in (msg.get("parts") or []):
            step += 1
            lines.append(_render_part(step, part))
    if step == 0:
        lines.append("\n_(no messages captured)_")
    return "\n".join(lines) + "\n"


def _index_row(
    session_id: str,
    *,
    role: str | None,
    meta: dict | None,
    status: str | None,
    error: Any,
    tool_call_stats: dict | None,
    n_turns: int | None,
) -> dict:
    return {
        "session_id": session_id,
        "role": role,
        "kb_id": (meta or {}).get("kb_id"),
        "status": status,
        "n_turns": n_turns,
        "tool_calls": (tool_call_stats or {}).get("total_calls"),
        "tool_errors": (tool_call_stats or {}).get("total_errors"),
        "error": (str(error)[:120].replace("\n", " ") if error else ""),
    }
