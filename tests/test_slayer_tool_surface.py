"""DEV-1644 Fix 1a: the shared SLayer tool-surface helper.

The Claude Agent SDK only strips a tool's JSON schema from the model's
per-turn context via ``disallowed_tools=`` (``allowed_tools=`` merely gates
auto-execution — DEV-1579). The SLayer stdio MCP server advertises ALL of
its tools, so any tool that is neither allowed nor disallowed leaks its
schema and the model burns turns calling it for a "permission not granted"
error.

The fix collapses the two hand-written lists to a single allow-list and
DERIVES ``disallowed = (all advertised slayer tools) − allowed``. This module
provides the two primitives that make that leak-proof by construction:

* ``all_slayer_mcp_tool_names()`` — the live SLayer surface, introspected
  from the installed slayer package (tracks the lock-pinned version).
* ``derive_disallowed_slayer_tools(allowed_bare)`` — the complement.
"""

from __future__ import annotations

import pytest


# The five tools the DEV-1644 failure-mode analysis found leaking on the
# livesqlbench-large v2 run — none were on the agent allow-list nor on the
# old partial deny-list, so their schemas leaked to the model.
_HISTORICAL_LEAK_TOOLS = {
    "mcp__slayer__query",
    "mcp__slayer__query_nested",
    "mcp__slayer__describe_datasource",
    "mcp__slayer__edit_datasource",
    "mcp__slayer__delete_model",
}

# The full advertised surface for the lock-pinned slayer (0.9.6). Pinning the
# exact set turns any upstream add/remove into a loud CI failure (the lock only
# moves on a deliberate version bump) — the strongest anti-leak contract.
# DEV-1668: slayer 0.9.6 removed the `help` MCP tool (help content is now
# `inspect(reference="memory:help.intro", entity_type="memory")`) → 21 tools.
_EXPECTED_SURFACE = {
    f"mcp__slayer__{t}"
    for t in (
        "create_datasource", "create_model", "delete_datasource",
        "delete_model", "describe_datasource", "edit_datasource",
        "edit_model", "forget_memory", "get_datasource_priority",
        "ingest_datasource_models", "inspect", "inspect_model",
        "list_datasources", "models_summary", "query", "query_nested",
        "recommend_root_model", "save_memory", "search",
        "set_datasource_priority", "validate_models",
    )
}


def _fresh_surface():
    from bird_interact_agents.agents._slayer_tool_surface import (
        all_slayer_mcp_tool_names,
    )

    all_slayer_mcp_tool_names.cache_clear()
    return all_slayer_mcp_tool_names()


def test_surface_is_prefixed_nonempty_frozenset():
    surface = _fresh_surface()
    assert isinstance(surface, frozenset)
    assert surface, "SLayer must advertise at least one MCP tool"
    for name in surface:
        assert name.startswith("mcp__slayer__"), name
        # Non-empty bare suffix.
        assert name[len("mcp__slayer__"):]


def test_surface_contains_historical_leak_tools():
    """Regression pin: every tool that leaked in the DEV-1644 analysis must
    be part of the enumerated surface, so the complement can hide it."""
    surface = _fresh_surface()
    missing = _HISTORICAL_LEAK_TOOLS - surface
    assert not missing, f"leak tools absent from enumerated surface: {sorted(missing)}"


def test_surface_matches_pinned_slayer_set():
    """Exact-set pin on the lock-pinned slayer surface (Codex round-2). A
    tool added/removed upstream fails here on the next version bump, forcing a
    review so a new tool can't silently leak. DEV-1668: 0.9.6 = 21 tools
    (``help`` removed)."""
    assert _fresh_surface() == _EXPECTED_SURFACE


def test_surface_superset_of_every_allow_list():
    """The enumerated surface must be a superset of all three real
    allow-lists — otherwise an allowed name is not a real SLayer tool and the
    complement would be computed against a wrong universe."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import SLAYER_MCP_TOOLS as one_shot
    from bird_interact_agents.agents.claude_sdk_otf_encode.setup_encoder import (
        SLAYER_MCP_TOOLS as encoder,
    )

    surface = _fresh_surface()
    for allow in (one_shot, encoder):
        prefixed = {f"mcp__slayer__{t}" for t in allow}
        assert prefixed <= surface, sorted(prefixed - surface)


# ---------------------------------------------------------------------------
# Helper internals (monkeypatched — independent of the real 22-tool surface).
# ---------------------------------------------------------------------------


class _FakeTool:
    def __init__(self, name):
        self.name = name


class _FakeMgr:
    def __init__(self, names):
        self._names = names

    def list_tools(self):
        return [_FakeTool(n) for n in self._names]


class _FakeServer:
    def __init__(self, names):
        self._tool_manager = _FakeMgr(names)


def test_enumeration_uses_create_mcp_server_without_ingest(monkeypatch):
    """The helper must build the server with ingest_on_startup=False (a boot
    ingest would connect to a datasource) and prefix each bare tool name."""
    import slayer.mcp.server as sms
    from bird_interact_agents.agents import _slayer_tool_surface as sts

    calls = {}

    def _fake_create(storage, *, ingest_on_startup=True, _seed_help=True):
        calls["ingest_on_startup"] = ingest_on_startup
        calls["_seed_help"] = _seed_help
        calls["storage_type"] = type(storage).__name__
        return _FakeServer(["help", "query"])

    monkeypatch.setattr(sms, "create_mcp_server", _fake_create)
    sts.all_slayer_mcp_tool_names.cache_clear()
    try:
        out = sts.all_slayer_mcp_tool_names()
        assert calls["ingest_on_startup"] is False
        # DEV-1668/DEV-1669: metadata-only enumeration must not trigger
        # slayer 0.9.6's help-seeding.
        assert calls["_seed_help"] is False
        assert calls["storage_type"] == "YAMLStorage"
        assert out == frozenset({"mcp__slayer__help", "mcp__slayer__query"})
    finally:
        sts.all_slayer_mcp_tool_names.cache_clear()


def test_enumeration_raises_when_no_tools(monkeypatch):
    """A broken introspection returning zero tools must fail LOUD — silently
    returning an empty surface would make derive() disallow nothing and
    re-open the leak."""
    import slayer.mcp.server as sms
    from bird_interact_agents.agents import _slayer_tool_surface as sts

    monkeypatch.setattr(sms, "create_mcp_server", lambda *a, **k: _FakeServer([]))
    sts.all_slayer_mcp_tool_names.cache_clear()
    try:
        with pytest.raises(RuntimeError):
            sts.all_slayer_mcp_tool_names()
    finally:
        sts.all_slayer_mcp_tool_names.cache_clear()


def test_surface_is_cached_identity_stable():
    from bird_interact_agents.agents._slayer_tool_surface import (
        all_slayer_mcp_tool_names,
    )

    all_slayer_mcp_tool_names.cache_clear()
    first = all_slayer_mcp_tool_names()
    second = all_slayer_mcp_tool_names()
    assert first is second, "lru_cache must return the same frozenset object"


# ---------------------------------------------------------------------------
# derive_disallowed_slayer_tools — the complement + partition contract
# ---------------------------------------------------------------------------


def test_derive_partitions_the_surface():
    """disallowed ∪ allowed == full surface, and disallowed ∩ allowed == ∅.
    Nothing can leak by construction."""
    from bird_interact_agents.agents._slayer_tool_surface import (
        all_slayer_mcp_tool_names,
        derive_disallowed_slayer_tools,
    )

    all_slayer_mcp_tool_names.cache_clear()
    surface = all_slayer_mcp_tool_names()
    # A representative allow-list drawn from the real surface (DEV-1668: `help`
    # is no longer advertised — use `inspect_model` as the representative name).
    allow_bare = ["inspect_model", "search", "models_summary", "inspect"]
    allow = {f"mcp__slayer__{t}" for t in allow_bare}
    disallowed = derive_disallowed_slayer_tools(allow_bare)

    assert set(disallowed).isdisjoint(allow)
    assert set(disallowed) | allow == surface
    # Sorted, deduplicated list output.
    assert disallowed == sorted(set(disallowed))


def test_derive_hides_the_leak_tools_for_the_agent_allow_list():
    from bird_interact_agents.agents._slayer_tool_surface import (
        derive_disallowed_slayer_tools,
    )
    from bird_interact_agents.agents.claude_sdk_otf.agent import SLAYER_MCP_TOOLS

    disallowed = set(derive_disallowed_slayer_tools(SLAYER_MCP_TOOLS))
    assert _HISTORICAL_LEAK_TOOLS <= disallowed


def test_derive_rejects_allow_list_not_in_surface():
    """Codex F2: a typo'd / upstream-renamed allowed name must fail loudly in
    production, not silently compute a broken complement."""
    from bird_interact_agents.agents._slayer_tool_surface import (
        derive_disallowed_slayer_tools,
    )

    with pytest.raises(ValueError):
        derive_disallowed_slayer_tools(["search", "not_a_real_slayer_tool"])


def test_derive_write_stripped_allow_list_disallows_write_tools():
    """Pre-encoded mode strips the WRITE tools from the allow-list; the derived
    complement must therefore HIDE their schemas (the old code did this via an
    explicit .extend(WRITE_SLAYER_TOOL_NAMES) — now automatic)."""
    from bird_interact_agents.agents._pre_encoded import (
        WRITE_SLAYER_TOOL_NAMES,
        strip_write_slayer_tools,
    )
    from bird_interact_agents.agents._slayer_tool_surface import (
        derive_disallowed_slayer_tools,
    )
    from bird_interact_agents.agents.claude_sdk_otf.agent import SLAYER_MCP_TOOLS

    stripped = strip_write_slayer_tools(SLAYER_MCP_TOOLS)
    disallowed = set(derive_disallowed_slayer_tools(stripped))
    assert WRITE_SLAYER_TOOL_NAMES <= disallowed
