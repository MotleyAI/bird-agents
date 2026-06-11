"""Per-framework trajectory→Turn adapters.

The SLayer-mode ``claude_sdk_otf`` / ``claude_sdk_otf_ainteract`` agents
persist Anthropic SDK messages as nested dicts (``data`` is a dict with
``content``, ``model``, …) and share one walker.

The ``_raw`` variants (``claude_sdk_otf_raw`` /
``claude_sdk_otf_ainteract_raw``) persist trajectory entries' ``data``
field as Python-repr STRINGS instead — the dict-based walker cannot read
them. Until a string-repr parser lands they raise
``UnknownFrameworkError`` here so the failure surfaces at the CLI's
source-resolution step rather than mid-walk with a confusing
``AttributeError``. Other frameworks (pydantic_ai*, smolagents, agno,
mcp_agent) are out of scope for DEV-1553.
"""

from __future__ import annotations

from typing import Callable, Iterable

from bird_interact_agents.reports.adapters.base import Turn
from bird_interact_agents.reports.adapters.claude_sdk_otf import (
    walk_trajectory as _claude_sdk_otf_walk,
)


_REGISTRY: dict[str, Callable[[list[dict]], Iterable[Turn]]] = {
    "claude_sdk_otf": _claude_sdk_otf_walk,
    "claude_sdk_otf_ainteract": _claude_sdk_otf_walk,
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
