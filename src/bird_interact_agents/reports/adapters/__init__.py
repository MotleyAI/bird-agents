"""Per-framework trajectory→Turn adapters.

Real cloud runs persist ``framework="claude_sdk"`` in ``results.db.
run_metadata`` (that's the CLI flag of ``bird-interact-cloud submit``);
the SLayer-vs-raw / one-shot-vs-a-interact distinction lives in the
``query_mode`` and ``mode`` columns. The submission report is supported
only on the ``(framework="claude_sdk", query_mode="slayer")`` combo:

* ``query_mode="slayer"``: Anthropic SDK messages are persisted as
  nested dicts (``data`` is a dict with ``content``, ``model``, …) and
  the shared dict-walker handles them.
* ``query_mode="raw"``: trajectory entries' ``data`` field is a
  Python-repr STRING; needs a separate string-repr parser that isn't
  written yet.

We also accept ``framework="claude_sdk_otf"`` /
``"claude_sdk_otf_ainteract"`` for forward-compatibility with any future
run-metadata schema that promotes the internal agent name to a
persisted field — both share the same dict walker.

Other frameworks (pydantic_ai*, smolagents, agno, mcp_agent) are out of
scope for DEV-1553.
"""

from __future__ import annotations

from typing import Callable, Iterable

from bird_interact_agents.reports.adapters.base import Turn
from bird_interact_agents.reports.adapters.claude_sdk_otf import (
    walk_trajectory as _claude_sdk_otf_walk,
)


# Keyed by (framework, query_mode). All entries point at the dict-walker;
# the only thing that varies is which combinations we accept.
_REGISTRY: dict[tuple[str, str], Callable[[list[dict]], Iterable[Turn]]] = {
    ("claude_sdk", "slayer"): _claude_sdk_otf_walk,
    ("claude_sdk_otf", "slayer"): _claude_sdk_otf_walk,
    ("claude_sdk_otf_ainteract", "slayer"): _claude_sdk_otf_walk,
}


class UnknownFrameworkError(ValueError):
    pass


def get_adapter(
    framework: str, *, query_mode: str = "slayer"
) -> Callable[[list[dict]], Iterable[Turn]]:
    """Resolve ``(framework, query_mode)`` to its trajectory walker.

    Raises ``UnknownFrameworkError`` for any unsupported combo — the
    error message spells out the supported list so the failure surfaces
    with actionable guidance, not a mid-walk AttributeError.
    """
    try:
        return _REGISTRY[(framework, query_mode)]
    except KeyError:
        raise UnknownFrameworkError(
            f"no submission-report adapter registered for "
            f"(framework={framework!r}, query_mode={query_mode!r}); "
            f"supported: {sorted(_REGISTRY)}"
        )


__all__ = ["Turn", "get_adapter", "UnknownFrameworkError"]
