"""PydanticAI on-the-fly KB-encode adapter (DEV-1454).

Sibling of `pydantic_ai_recursive` that elevates knowledge-base items
into first-class SLayer entities at task time via a dedicated encoder
sub-agent. See `agent.py` for the entry class, `factories.py` for the
tool registrars, and `prompts.py` for the role prompts.

Public re-export: ``PydanticAIOtfEncodeAgent``.
"""

from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
    PydanticAIOtfEncodeAgent,
)

__all__ = ["PydanticAIOtfEncodeAgent"]
