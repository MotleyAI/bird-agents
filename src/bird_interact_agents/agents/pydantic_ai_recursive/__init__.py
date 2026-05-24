"""Recursive pydantic-ai adapter for SLayer a-interact benchmarks.

The adapter splits the single-prompt agent (``SLAYER_A_INTERACT``) into a
tree of focused agents: a root clarifier decomposes the user's question
and spawns one sub-clarifier per logical block; each sub-clarifier
nails down its slice via ``search``/``ask_user``/optional recursive
spawning; a separate query-constructor receives the concatenated chunk
descriptions plus the original user query, drafts the SLayer JSON with
an active count-check against the user-named columns, and submits.

Public class: :class:`PydanticAIRecursiveAgent` — drop-in adapter
selectable via ``--framework pydantic_ai_recursive``.
"""

from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
    PydanticAIRecursiveAgent,
)

__all__ = ["PydanticAIRecursiveAgent"]
