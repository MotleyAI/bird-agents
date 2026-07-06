"""DEV-1644: single-source-of-truth SLayer tool gating.

The Claude Agent SDK exposes two independent knobs:

* ``allowed_tools=`` gates AUTO-EXECUTION only.
* ``disallowed_tools=`` is the ONLY knob that strips a tool's JSON schema from
  the model's per-turn context (DEV-1579).

The SLayer stdio MCP server advertises ALL of its tools. The OTF agents used to
maintain a hand-written allow-list PLUS a partial hand-written deny-list; every
tool in neither leaked its schema, so the model called it (e.g.
``mcp__slayer__query``) and ate a "permission not granted" error — ~46% of the
recoverable tool-errors in the DEV-1644 failure-mode analysis.

This module collapses that two-list design to a single allow-list and DERIVES
``disallowed = (all advertised slayer tools) − allowed``. The full surface is
introspected from the installed slayer package, so it tracks the lock-pinned
version with zero hand-maintenance and nothing can leak by construction.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

_SLAYER_PREFIX = "mcp__slayer__"


@lru_cache(maxsize=1)
def all_slayer_mcp_tool_names() -> frozenset[str]:
    """Every tool the SLayer stdio MCP server advertises, ``mcp__slayer__``-
    prefixed (the form the SDK matches ``disallowed_tools=`` against).

    Introspects the installed slayer package's server factory rather than
    hard-coding a list, so the set follows the lock-pinned slayer version.
    ``slayer`` is imported lazily so importing an agent module without the
    ``slayer`` extra installed still works.

    Raises:
        RuntimeError: if introspection yields zero tools — failing loud here
            beats silently returning an empty surface, which would make
            :func:`derive_disallowed_slayer_tools` disallow nothing and
            re-open the schema leak.
    """
    import tempfile
    import warnings

    from slayer.mcp.server import create_mcp_server
    from slayer.storage.yaml_storage import YAMLStorage

    with tempfile.TemporaryDirectory() as tmp:
        with warnings.catch_warnings():
            # SLayer's FastMCP tool registration triggers a benign pydantic
            # "default not JSON serializable" schema warning; silence only that.
            try:
                from pydantic.json_schema import PydanticJsonSchemaWarning

                warnings.simplefilter("ignore", PydanticJsonSchemaWarning)
            except Exception:  # pragma: no cover - pydantic internals moved
                pass
            # ingest_on_startup=False: registering tools must not connect to a
            # datasource; we only need the advertised tool names.
            server = create_mcp_server(YAMLStorage(base_dir=tmp), ingest_on_startup=False)
        bare = {tool.name for tool in server._tool_manager.list_tools()}

    if not bare:
        raise RuntimeError(
            "SLayer advertised zero MCP tools — tool-surface introspection is "
            "broken; refusing to derive an empty disallow-list (would re-open "
            "the schema leak)."
        )
    return frozenset(f"{_SLAYER_PREFIX}{name}" for name in bare)


def derive_disallowed_slayer_tools(allowed_bare: Iterable[str]) -> list[str]:
    """Return ``sorted((all advertised slayer tools) − allowed)``.

    ``allowed_bare`` are BARE tool names (the ``SLAYER_MCP_TOOLS`` allow-list
    form). Every name is prefixed with ``mcp__slayer__`` before the set
    difference.

    Raises:
        ValueError: if any allowed name is not an advertised SLayer tool (a
            typo or an upstream rename) — otherwise the allow-list would grant
            a dead tool and the real one would silently fall into the
            disallowed complement, losing the capability. Fail loud instead.
    """
    surface = all_slayer_mcp_tool_names()
    allowed = {f"{_SLAYER_PREFIX}{name}" for name in allowed_bare}
    unknown = allowed - surface
    if unknown:
        raise ValueError(
            "allow-list names are not advertised SLayer tools "
            f"(typo or upstream rename): {sorted(unknown)}"
        )
    return sorted(surface - allowed)
