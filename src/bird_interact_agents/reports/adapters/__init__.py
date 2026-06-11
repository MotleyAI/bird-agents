"""Per-framework trajectory→Turn adapters.

The ``claude_sdk_otf_*`` family (claude_sdk_otf / _raw / _ainteract /
_ainteract_raw) shares the Claude Agent SDK message schema, so one
adapter covers all four. Other frameworks (pydantic_ai*, smolagents,
agno, mcp_agent) are out of scope for DEV-1553 and will register their
own adapters when needed.
"""

from __future__ import annotations

from typing import Callable, Iterable

from bird_interact_agents.reports.adapters.base import Turn
from bird_interact_agents.reports.adapters.claude_sdk_otf import (
    walk_trajectory as _claude_sdk_otf_walk,
)


_REGISTRY: dict[str, Callable[[list[dict]], Iterable[Turn]]] = {
    "claude_sdk_otf": _claude_sdk_otf_walk,
    "claude_sdk_otf_raw": _claude_sdk_otf_walk,
    "claude_sdk_otf_ainteract": _claude_sdk_otf_walk,
    "claude_sdk_otf_ainteract_raw": _claude_sdk_otf_walk,
}


class UnknownFrameworkError(ValueError):
    pass


def get_adapter(framework: str) -> Callable[[list[dict]], Iterable[Turn]]:
    """Resolve the framework name to its trajectory walker."""
    try:
        return _REGISTRY[framework]
    except KeyError:
        raise UnknownFrameworkError(
            f"no submission-report adapter registered for framework "
            f"{framework!r}; supported: {sorted(_REGISTRY)}"
        )


__all__ = ["Turn", "get_adapter", "UnknownFrameworkError"]
