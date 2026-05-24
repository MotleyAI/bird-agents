"""Shape-agnostic accessors for the per-task ``trajectory`` dict.

Two shapes are in flight in this repo:

* **Old** (``pydantic_ai`` adapter and friends):
  ``{final_output_excerpt, messages, user_sim_transcript}``
* **New** (``pydantic_ai_recursive`` adapter):
  ``{final_output_excerpt, agents: [{role, depth, parent_idx, focus,
  instruction, output, messages, user_sim_transcript, usage, ...}, ...]}``

These accessors hide the shape difference behind a stable list-returning
contract so analysis code works on either without branching. For the old
shape, ``iter_agent_records`` synthesises a single record with role
``"legacy_single_agent"`` carrying the flat messages and transcript, so
downstream code can iterate uniformly.
"""

from __future__ import annotations

from typing import Any


def iter_messages(trajectory: Any) -> list[dict]:
    """Flatten every agent's serialized messages in order. Returns a new
    list — mutating it does not leak into the trajectory dict.

    Tolerates non-dict input: ``run.py``'s outer error path writes
    ``"trajectory": []`` for catastrophic failures (no per-task data
    captured), so analysis code may see a list where a dict is expected.
    Return an empty list rather than raising in that case.
    """
    if not isinstance(trajectory, dict):
        return []
    agents = trajectory.get("agents")
    if isinstance(agents, list):
        out: list[dict] = []
        for a in agents:
            if isinstance(a, dict):
                out.extend(a.get("messages") or [])
        return out
    return list(trajectory.get("messages") or [])


def iter_user_sim_transcript(trajectory: Any) -> list[dict]:
    """Flatten every agent's user-sim encoder/decoder transcript in
    order. Returns a new list. Tolerates non-dict input."""
    if not isinstance(trajectory, dict):
        return []
    agents = trajectory.get("agents")
    if isinstance(agents, list):
        out: list[dict] = []
        for a in agents:
            if isinstance(a, dict):
                out.extend(a.get("user_sim_transcript") or [])
        return out
    return list(trajectory.get("user_sim_transcript") or [])


def iter_agent_records(trajectory: Any) -> list[dict]:
    """Return per-agent records. For the new shape, pass through
    ``trajectory["agents"]``. For the old shape, synthesise a single
    record with role ``"legacy_single_agent"`` so downstream code can
    iterate uniformly. Tolerates non-dict input (returns ``[]``)."""
    if not isinstance(trajectory, dict):
        return []
    agents = trajectory.get("agents")
    if isinstance(agents, list):
        return list(agents)
    return [{
        "role": "legacy_single_agent",
        "depth": 0,
        "parent_idx": None,
        "messages": list(trajectory.get("messages") or []),
        "user_sim_transcript": list(trajectory.get("user_sim_transcript") or []),
    }]
