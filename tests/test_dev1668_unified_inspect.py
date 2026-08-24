"""DEV-1668: adopt unified ``inspect``; drop ``models_summary`` /
``list_datasources`` / ``help`` from the claude_sdk_otf* SLayer surface.

slayer 0.9.6 removed the ``help`` MCP tool entirely (help content is now
``inspect(reference="memory:help.intro", entity_type="memory")``) and gave
``inspect`` a null-reference COLLECTION view
(``inspect(reference=None, entity_type="model"|"datasource")``) that subsumes
``models_summary`` / ``list_datasources`` (retained as thin aliases in 0.9.6).

This module pins the bird-agents adoption:

* ``help`` is removed UNCONDITIONALLY from every claude_sdk_otf* surface (the
  tool is gone; allow-listing it would crash ``derive_disallowed_slayer_tools``,
  and the v1 native bridge could not resolve its schema).
* ``models_summary`` + ``list_datasources`` are LEAN-GATED on the QUERY agents
  (v0 + v1) — NOT removed outright: ``lean_introspection=True`` (default) drops
  them (broad inventory routes through ``inspect(reference=None, …)``), while
  ``lean_introspection=False`` (legacy) KEEPS them. ``False``/``False`` stays the
  legacy identity.
* The ENCODER (``claude_sdk_otf_encode``) is exempt from the lean surface
  reduction: it drops ONLY ``help`` (forced) and KEEPS ``models_summary`` (and the
  rest of its allow-list) unchanged.

Per the project rule against prompt-content tests, nothing here asserts prompt
anchor phrases — only the structural tool surface, the build-does-not-raise
contract, and a behavioural check that the 0.9.6 collection/help views work.
"""

from __future__ import annotations

import tempfile
import warnings

import pytest

from bird_interact_agents.agents import _slayer_tool_surface as tsurf


# ---------------------------------------------------------------------------
# 1. ``help`` is gone from every claude_sdk_otf* surface, and building each
#    effective surface + deriving the disallowed complement does NOT raise on
#    slayer 0.9.6 (``help`` is no longer an advertised tool).
# ---------------------------------------------------------------------------


def test_help_absent_from_v0_allow_list():
    from bird_interact_agents.agents.claude_sdk_otf.agent import SLAYER_MCP_TOOLS

    assert "help" not in SLAYER_MCP_TOOLS


def test_help_absent_from_v0_ainteract_allow_list():
    # a-interact imports the same constant; assert via its own module so a future
    # divergence is caught.
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
        SLAYER_MCP_TOOLS,
    )

    assert "help" not in SLAYER_MCP_TOOLS


def test_help_absent_from_encoder_allow_list():
    from bird_interact_agents.agents.claude_sdk_otf_encode.setup_encoder import (
        SLAYER_MCP_TOOLS,
    )

    assert "help" not in SLAYER_MCP_TOOLS


def test_help_absent_from_v1_main_natives():
    from bird_interact_agents.agents.claude_sdk_otf_v1.agent import _MAIN_NATIVE_BARE

    assert "help" not in _MAIN_NATIVE_BARE


def test_help_absent_from_ainteract_v1_main_natives():
    """a-interact_v1 extends the one-shot v1 MAIN list; removing help from the
    v1 base must carry through to its exported surface."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent import (
        MAIN_NATIVE_TOOL_NAMES,
    )

    bare = {n.split("__")[-1] for n in MAIN_NATIVE_TOOL_NAMES}
    assert "help" not in bare


@pytest.mark.parametrize("lean", [True, False])
@pytest.mark.parametrize("readonly", [True, False])
def test_v0_derive_disallowed_does_not_raise_on_0_9_6(lean, readonly):
    """The v0 effective allow-list must contain only tools advertised by the
    installed slayer (``derive_disallowed_slayer_tools`` raises otherwise). With
    ``help`` still allow-listed this raises on 0.9.6."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        effective_slayer_allow,
    )
    from bird_interact_agents.agents._slayer_tool_surface import (
        derive_disallowed_slayer_tools,
    )

    allow = effective_slayer_allow(
        lean_introspection=lean, readonly_mode=readonly, pre_encoded_source=None,
    )
    # Must not raise (every name is a real 0.9.6 tool).
    derive_disallowed_slayer_tools(allow)


def test_encoder_disallowed_does_not_raise_on_0_9_6():
    from bird_interact_agents.agents.claude_sdk_otf_encode.setup_encoder import (
        disallowed_tool_names,
    )

    # Must not raise.
    disallowed_tool_names()


# ---------------------------------------------------------------------------
# 2. ``models_summary`` + ``list_datasources`` are lean-gated on the QUERY
#    agents; ``inspect`` survives in every mode.
# ---------------------------------------------------------------------------


def test_lean_drop_set_includes_models_summary_and_list_datasources():
    assert {"models_summary", "list_datasources", "inspect_model"} <= (
        tsurf.LEAN_DROP_SLAYER_MCP
    )
    # ``inspect`` is NEVER dropped — it is the unified replacement.
    assert "inspect" not in tsurf.LEAN_DROP_SLAYER_MCP


@pytest.mark.parametrize("tool", ["models_summary", "list_datasources"])
def test_v0_query_lean_drops_but_legacy_keeps(tool):
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        effective_slayer_allow,
    )

    lean = effective_slayer_allow(
        lean_introspection=True, readonly_mode=False, pre_encoded_source=None,
    )
    legacy = effective_slayer_allow(
        lean_introspection=False, readonly_mode=False, pre_encoded_source=None,
    )
    assert tool not in lean
    assert tool in legacy
    # ``inspect`` present in both.
    assert "inspect" in lean and "inspect" in legacy


@pytest.mark.parametrize("tool", ["models_summary", "list_datasources"])
def test_v0_query_lean_hides_dropped_tool_schema(tool):
    """Dropping from the allow-list is not enough — the derived DISALLOWED
    complement must carry the tool so its JSON schema is stripped from the
    model's per-turn context."""
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        effective_slayer_allow,
    )
    from bird_interact_agents.agents._slayer_tool_surface import (
        derive_disallowed_slayer_tools,
    )

    lean_allow = effective_slayer_allow(
        lean_introspection=True, readonly_mode=False, pre_encoded_source=None,
    )
    disallowed = set(derive_disallowed_slayer_tools(lean_allow))
    assert f"mcp__slayer__{tool}" in disallowed


def test_v1_discovery_lean_drops_models_summary_legacy_keeps():
    from bird_interact_agents.agents.claude_sdk_otf_v1.agent import (
        effective_discovery_tools,
    )

    lean = {n.split("__")[-1] for n in effective_discovery_tools(
        lean_introspection=True)}
    legacy = {n.split("__")[-1] for n in effective_discovery_tools(
        lean_introspection=False)}
    assert "models_summary" not in lean
    assert "models_summary" in legacy
    assert "inspect" in lean and "inspect" in legacy


def test_ainteract_v1_discovery_lean_gating():
    """a-interact_v1 filters its DISCOVERY list inline in run_task via the same
    ``filter_flag_drops`` on its exported ``DISCOVERY_NATIVE_TOOL_NAMES``; pin
    the same lean-drop / legacy-keep contract on that surface."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent import (
        DISCOVERY_NATIVE_TOOL_NAMES,
    )
    from bird_interact_agents.agents._slayer_tool_surface import filter_flag_drops

    lean = {n.split("__")[-1] for n in filter_flag_drops(
        DISCOVERY_NATIVE_TOOL_NAMES, lean_introspection=True, readonly_mode=False)}
    legacy = {n.split("__")[-1] for n in filter_flag_drops(
        DISCOVERY_NATIVE_TOOL_NAMES, lean_introspection=False, readonly_mode=False)}
    assert "models_summary" not in lean
    assert "models_summary" in legacy
    assert "inspect" in lean and "inspect" in legacy


# ---------------------------------------------------------------------------
# 3. The encoder is exempt from the lean surface reduction: it keeps
#    ``models_summary`` and drops ONLY ``help`` — the rest of its allow-list is
#    preserved exactly.
# ---------------------------------------------------------------------------


def test_encoder_drops_only_help_rest_preserved():
    from bird_interact_agents.agents.claude_sdk_otf_encode.setup_encoder import (
        SLAYER_MCP_TOOLS,
    )

    # The exact encoder allow-list after DEV-1668 = the pre-1668 set minus help.
    # Pinning the full set turns any accidental drop (search / query / a write
    # tool) into a failure, not just the help removal.
    assert set(SLAYER_MCP_TOOLS) == {
        "create_model", "edit_model", "save_memory", "validate_models",
        "search", "models_summary", "inspect_model", "inspect",
        "recommend_root_model", "query", "query_nested",
    }
    assert "help" not in SLAYER_MCP_TOOLS


# ---------------------------------------------------------------------------
# 4. Legacy (lean=False) still loses ``help`` — it is NOT gate-able (the tool
#    no longer exists), unlike models_summary / list_datasources.
# ---------------------------------------------------------------------------


def test_legacy_v0_still_lacks_help_but_keeps_the_gated_tools():
    from bird_interact_agents.agents.claude_sdk_otf.agent import (
        effective_slayer_allow,
    )

    legacy = effective_slayer_allow(
        lean_introspection=False, readonly_mode=False, pre_encoded_source=None,
    )
    assert "help" not in legacy
    assert "models_summary" in legacy
    assert "list_datasources" in legacy


# ---------------------------------------------------------------------------
# 5. The 0.9.6 collection/help views actually work — proves ``inspect`` covers
#    the function of the tools it replaces (not just that it is allow-listed).
#    Behavioural check against installed slayer; not a prompt assertion.
# ---------------------------------------------------------------------------


def _inspect_fn():
    """The 0.9.6 ``inspect`` MCP tool callable, from a seeded fresh server
    (``create_mcp_server`` seeds the ``memory:help.*`` conceptual memories)."""
    from slayer.mcp.server import create_mcp_server
    from slayer.storage.yaml_storage import YAMLStorage

    tmp = tempfile.TemporaryDirectory()
    with warnings.catch_warnings():
        try:
            from pydantic.json_schema import PydanticJsonSchemaWarning

            warnings.simplefilter("ignore", PydanticJsonSchemaWarning)
        except ImportError:  # pragma: no cover
            pass
        server = create_mcp_server(
            YAMLStorage(base_dir=tmp.name), ingest_on_startup=False,
        )
    tool = {t.name: t for t in server._tool_manager.list_tools()}["inspect"]
    fn = getattr(tool, "fn", None) or getattr(tool, "_fn", None)
    assert fn is not None, "could not resolve the inspect tool callable"
    return fn, tmp  # keep tmp alive


@pytest.mark.parametrize("entity_type", ["model", "datasource"])
async def test_inspect_collection_view_reachable(entity_type):
    """``inspect(reference=None, entity_type=…)`` (the models_summary /
    list_datasources replacement) must not error on 0.9.6."""
    fn, tmp = _inspect_fn()
    try:
        out = await fn(entity_type=entity_type, reference=None)
        assert isinstance(out, str)
    finally:
        tmp.cleanup()


async def test_inspect_help_intro_reachable():
    """The help replacement — ``inspect(memory:help.intro, compact=False)`` —
    must return a non-empty body on 0.9.6."""
    fn, tmp = _inspect_fn()
    try:
        out = await fn(
            entity_type="memory", reference="memory:help.intro", compact=False,
        )
        assert isinstance(out, str) and out.strip()
    finally:
        tmp.cleanup()
