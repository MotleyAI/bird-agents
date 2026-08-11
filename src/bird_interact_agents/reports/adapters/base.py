"""Shared ``Turn`` dataclass for trajectory adapters."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Turn(BaseModel):
    """One agent assistant-message + its emitted action + the resulting
    observation. Framework-agnostic intermediate representation.
    """

    # Agent model used for this turn (e.g. ``claude-opus-4-7``). Most
    # runs hold this constant across turns, but we carry it per-turn so
    # multi-model trajectories work the day someone files one.
    model: str

    # The exact text the agent saw to produce this turn. For turn 0 this
    # is the initial task statement; for later turns it's the
    # concatenation of every tool_result + free UserMessage text seen
    # between the previous tool_use and this one.
    prompt: str

    # Rendered raw assistant message. Carries thinking + text + tool_use
    # JSON when include_thinking=True; thinking stripped otherwise.
    response_raw: str

    # Tool call.
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str

    # Tool result text (collapsed into a single string).
    observation: str

    model_config = {"arbitrary_types_allowed": True}
