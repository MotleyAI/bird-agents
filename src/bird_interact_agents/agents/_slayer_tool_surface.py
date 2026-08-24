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

import logging
from functools import lru_cache
from typing import Iterable, Optional

from bird_interact_agents.agents._pre_encoded import WRITE_SLAYER_TOOLS

_SLAYER_PREFIX = "mcp__slayer__"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DEV-1666: slayer-only ``lean_introspection`` + ``readonly_mode`` flag surface.
#
# The flags drop tools redundant with ``search`` / ``inspect`` (lean) or the
# SLayer WRITE tools (readonly) from the v0 + v1 SLayer QUERY agents. The drop
# is expressed as a set of BARE tool names; each agent filters its own
# allow-list / native-tool list through :func:`filter_flag_drops`.
# ---------------------------------------------------------------------------

#: Read tools dropped by ``lean_introspection`` that live on the SLayer MCP
#: surface, all subsumed by the unified ``inspect`` (DEV-1668). ``inspect_model``
#: is redundant with ``inspect(entity_type="model", sections=["columns","joins"],
#: compact=True)`` (per-model shape); ``models_summary`` is redundant with the
#: null-reference COLLECTION view ``inspect(reference=None, entity_type="model")``
#: (slayer 0.9.6 / DEV-1667); ``list_datasources`` is dead weight (exactly one
#: datasource per task) and equivalently ``inspect(reference=None,
#: entity_type="datasource")``. ``help`` is NOT here — it is removed
#: unconditionally (the tool no longer exists on slayer 0.9.6), not gate-able.
LEAN_DROP_SLAYER_MCP: frozenset[str] = frozenset(
    {"inspect_model", "models_summary", "list_datasources"}
)

#: Read tools dropped by ``lean_introspection`` that live on the in-process
#: ``bird-interact-tools`` server (the knowledge tools). Superseded by SLayer
#: memories: ``search`` surfaces KB items (``kind: memory``) and
#: ``inspect(entity_type="memory", reference=["memory:<id>"])`` reads them.
LEAN_DROP_NATIVE_KB: frozenset[str] = frozenset({
    "get_all_external_knowledge_names",
    "get_knowledge_definition",
    "get_all_knowledge_definitions",
})

#: Write tools dropped by ``readonly_mode`` — the SAME set the pre-encoded /
#: apply-edited paths already strip (single source of truth).
READONLY_DROP: frozenset[str] = WRITE_SLAYER_TOOLS

#: The frameworks whose slayer path builds an in-scope QUERY agent that CONSUMES
#: the flags (behavioural effect + recorded identity). Everything else (raw,
#: ``claude_sdk_otf_encode``, non-``claude_sdk``) is exempt: the flags are
#: ignored and recorded as ``None``.
FLAG_CONSUMING_FRAMEWORKS: frozenset[str] = frozenset({
    "claude_sdk",
    "claude_sdk_v1",
    "claude_sdk_otf",
    "claude_sdk_otf_ainteract",
    "claude_sdk_otf_v1",
    "claude_sdk_otf_ainteract_v1",
})


def slayer_flag_drops(
    *, lean_introspection: bool, readonly_mode: bool
) -> frozenset[str]:
    """Bare tool names to drop for the given flags (union of the applicable
    sets). ``False``/``False`` returns the empty set — the identity invariant."""
    drops: set[str] = set()
    if lean_introspection:
        drops |= LEAN_DROP_SLAYER_MCP | LEAN_DROP_NATIVE_KB
    if readonly_mode:
        drops |= READONLY_DROP
    return frozenset(drops)


def _bare_name(name: str) -> str:
    """The bare tool-name suffix of a possibly ``mcp__<server>__`` name. Mirrors
    ``_pre_encoded.strip_write_tool_names`` so the filter matches both the v0
    bare allow-list form and the v1 full ``mcp__bird-interact-tools__`` form."""
    return name.split("__")[-1] if name.startswith("mcp__") else name


def filter_flag_drops(
    names: Iterable[str], *, lean_introspection: bool, readonly_mode: bool
) -> list[str]:
    """Return ``names`` with the flag-dropped tools removed, order-preserving and
    idempotent, matching on the bare suffix (prefix-agnostic). Only drops names
    that are present, so it composes cleanly with ``strip_write_slayer_tools``."""
    drops = slayer_flag_drops(
        lean_introspection=lean_introspection, readonly_mode=readonly_mode
    )
    return [n for n in names if _bare_name(n) not in drops]


def flags_apply(*, framework: str, query_mode: str) -> bool:
    """True iff a run under ``(framework, query_mode)`` actually applies the
    flags (behavioural effect) — an in-scope query framework in slayer mode."""
    return framework in FLAG_CONSUMING_FRAMEWORKS and query_mode == "slayer"


def resolve_recorded_flags(
    *,
    framework: str,
    query_mode: str,
    lean_introspection: bool,
    readonly_mode: bool,
) -> tuple[Optional[bool], Optional[bool]]:
    """The ``(lean_introspection, readonly_mode)`` values to STAMP on a run
    record / manifest: the resolved bools when the run applies the flags,
    ``(None, None)`` otherwise (raw OR an exempt slayer framework)."""
    if flags_apply(framework=framework, query_mode=query_mode):
        return lean_introspection, readonly_mode
    return None, None


def maybe_warn_ignored_flags(
    *,
    framework: str,
    query_mode: str,
    lean_introspection: bool,
    readonly_mode: bool,
    log: logging.Logger | None = None,
) -> None:
    """Warn when an EXPLICIT deviation flag (``--no-lean`` or ``--readonly-mode``)
    was set on a run that will NOT apply it (raw OR an exempt slayer framework).
    The defaults (lean on / readonly off) are not a deviation, so a plain run is
    silent."""
    is_deviation = (not lean_introspection) or readonly_mode
    if is_deviation and not flags_apply(framework=framework, query_mode=query_mode):
        (log or logger).warning(
            "--no-lean / --readonly-mode apply only to the SLayer query agents "
            "(claude_sdk[_v1] / *_otf[_v1] query variants) in slayer mode; "
            "ignored here (framework=%s, query_mode=%s).",
            framework,
            query_mode,
        )


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
            # _seed_help=False (DEV-1668/DEV-1669): a metadata-only build must not
            # trigger slayer 0.9.6's DEV-1658 help-seeding (storage round-trip +
            # event loop) — we only enumerate tool names here.
            server = create_mcp_server(
                YAMLStorage(base_dir=tmp), ingest_on_startup=False, _seed_help=False,
            )
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
