"""DEV-1581 R2: per-agent tool-partition contract (all four v1 agents).

R2 gives each task two persistent clients. The MAIN client holds the
orchestration + encode/query/submit tools and the ``ask_discovery`` bridge;
the DISCOVERY client holds the schema/KB introspection tools.

DEV-1629: slayer v1 main ALSO holds ``search`` / ``inspect_model`` directly
(the main loop nails down query details itself); only ``models_summary`` stays
discovery-exclusive for slayer. Raw v1 keeps the full introspection split
(``get_schema`` / ``get_all_column_meanings`` discovery-only). The
``introspection_exclusive`` spec per agent captures exactly the tools that must
NOT leak into that agent's main.

Each v1 agent module exposes the partition as two stable constants of full
``mcp__…`` tool names:

* ``MAIN_NATIVE_TOOL_NAMES`` — what the main client registers.
* ``DISCOVERY_NATIVE_TOOL_NAMES`` — what the discovery client registers.

This test pins, per agent:

* the introspection-exclusive tools are in DISCOVERY and NOT in MAIN
  (schema precision — the core DEV-1581 guarantee);
* ``ask_discovery`` is a MAIN tool and NOT a discovery tool (it is the
  bridge, not something discovery calls);
* submit / write tools never leak into discovery;
* in pre-encoded mode (DEV-1586) the write tools strip OUT of main via the
  existing ``strip_write_tool_names`` helper.

No prompt-content assertions (repo rule) — only the mechanical tool-set
contract.
"""

from __future__ import annotations

import importlib

import pytest

from bird_interact_agents.agents._pre_encoded import strip_write_tool_names


def _bare(names):
    """Strip the ``mcp__<server>__`` prefix → bare tool names for set math."""
    out = set()
    for n in names:
        out.add(n.split("__")[-1] if n.startswith("mcp__") else n)
    return out


# agent module → partition expectations (bare tool names).
# DEV-1629: `search` / `inspect_model` moved ONTO slayer v1 main (the main loop
# nails down query details itself). `models_summary` stays discovery-only, so it
# is the lone slayer tool still exclusive to the discovery client.
_SLAYER_MAIN_INTROSPECTION = {"search", "inspect_model"}
_SLAYER_DISCOVERY_ONLY = {"models_summary"}
_SLAYER_INTROSPECTION = _SLAYER_MAIN_INTROSPECTION | _SLAYER_DISCOVERY_ONLY
_RAW_INTROSPECTION = {"get_schema", "get_all_column_meanings"}
_KB = {
    "get_all_external_knowledge_names",
    "get_knowledge_definition",
    "get_all_knowledge_definitions",
}

# Tools that must NEVER appear in the discovery client (the bridge + every
# finalization/write tool) — discovery only reads.
_SLAYER_WRITE_SUBMIT = {"submit_query", "create_model", "edit_model",
                        "validate_models", "save_memory"}
_RAW_WRITE_SUBMIT = {"submit_sql"}

_SPECS = {
    "claude_sdk_otf_v1": {
        "introspection_exclusive": _SLAYER_DISCOVERY_ONLY,
        "main_required": ({"query", "submit_query", "create_model", "edit_model",
                           "validate_models", "help", "ask_discovery"}
                          | _SLAYER_MAIN_INTROSPECTION | _KB),
        "discovery_required": _SLAYER_INTROSPECTION | _KB,
        "discovery_forbidden": {"ask_discovery"} | _SLAYER_WRITE_SUBMIT,
        "ask_user": False,
    },
    "claude_sdk_otf_ainteract_v1": {
        "introspection_exclusive": _SLAYER_DISCOVERY_ONLY,
        "main_required": ({"query", "submit_query", "create_model", "edit_model",
                           "validate_models", "help", "ask_discovery",
                           "ask_user"} | _SLAYER_MAIN_INTROSPECTION | _KB),
        "discovery_required": _SLAYER_INTROSPECTION | _KB | {"ask_user"},
        "discovery_forbidden": {"ask_discovery"} | _SLAYER_WRITE_SUBMIT,
        "ask_user": True,
    },
    "claude_sdk_otf_raw_v1": {
        "introspection_exclusive": _RAW_INTROSPECTION,
        "main_required": {"execute_sql", "submit_sql", "get_column_meaning",
                          "ask_discovery"} | _KB,
        "discovery_required": _RAW_INTROSPECTION | {"get_column_meaning",
                                                    "execute_sql"} | _KB,
        "discovery_forbidden": {"ask_discovery"} | _RAW_WRITE_SUBMIT,
        "ask_user": False,
    },
    "claude_sdk_otf_ainteract_raw_v1": {
        "introspection_exclusive": _RAW_INTROSPECTION,
        "main_required": {"execute_sql", "submit_sql", "get_column_meaning",
                          "ask_discovery", "ask_user"} | _KB,
        "discovery_required": _RAW_INTROSPECTION | {"get_column_meaning",
                                                    "execute_sql", "ask_user"} | _KB,
        "discovery_forbidden": {"ask_discovery"} | _RAW_WRITE_SUBMIT,
        "ask_user": True,
    },
}


def _agent_mod(name):
    return importlib.import_module(f"bird_interact_agents.agents.{name}.agent")


@pytest.mark.parametrize("agent_name", sorted(_SPECS))
def test_main_excludes_introspection(agent_name):
    """The core schema-precision guarantee: introspection-exclusive tools are
    NOT in the main client's tool surface (for slayer AND raw)."""
    spec = _SPECS[agent_name]
    main = _bare(_agent_mod(agent_name).MAIN_NATIVE_TOOL_NAMES)
    leaked = main & spec["introspection_exclusive"]
    assert not leaked, f"{agent_name}: introspection leaked into main: {leaked}"


@pytest.mark.parametrize("agent_name", sorted(_SPECS))
def test_main_has_required_tools_including_ask_discovery(agent_name):
    spec = _SPECS[agent_name]
    main = _bare(_agent_mod(agent_name).MAIN_NATIVE_TOOL_NAMES)
    missing = spec["main_required"] - main
    assert not missing, f"{agent_name}: main missing {missing}"
    assert "ask_discovery" in main


@pytest.mark.parametrize("agent_name", sorted(_SPECS))
def test_discovery_has_introspection_and_not_the_bridge(agent_name):
    spec = _SPECS[agent_name]
    disc = _bare(_agent_mod(agent_name).DISCOVERY_NATIVE_TOOL_NAMES)
    missing = spec["discovery_required"] - disc
    assert not missing, f"{agent_name}: discovery missing {missing}"
    bad = disc & spec["discovery_forbidden"]
    assert not bad, f"{agent_name}: discovery must not hold {bad}"


@pytest.mark.parametrize("agent_name", sorted(_SPECS))
def test_no_sdk_subagent_definition(agent_name):
    """R2 replaces the SDK-subagent split: the agent module must not reference
    ``AgentDefinition`` nor pass an ``agents=`` subagent map.

    Checks both the module attribute (catches ``from … import AgentDefinition``)
    AND the source text (catches aliased / in-function imports and the
    ``agents=`` keyword)."""
    import inspect

    mod = _agent_mod(agent_name)
    assert not hasattr(mod, "AgentDefinition"), (
        f"{agent_name}: still imports AgentDefinition — R2 uses two clients, "
        "not an SDK subagent"
    )
    src = inspect.getsource(mod)
    assert "AgentDefinition(" not in src, f"{agent_name}: builds an AgentDefinition"
    assert "agents=" not in src, f"{agent_name}: still passes an agents= subagent map"


@pytest.mark.parametrize("agent_name", ["claude_sdk_otf_v1",
                                        "claude_sdk_otf_ainteract_v1"])
def test_pre_encoded_strips_write_tools_from_main(agent_name):
    """DEV-1586 pre-encoded mode: write tools strip OUT of main's set (models
    are pre-built). Reuses the existing strip helper on R2's main list."""
    main = _agent_mod(agent_name).MAIN_NATIVE_TOOL_NAMES
    stripped = _bare(strip_write_tool_names(main))
    assert "create_model" not in stripped
    assert "edit_model" not in stripped
    # Non-write main tools survive.
    assert "query" in stripped and "submit_query" in stripped
    assert "ask_discovery" in stripped
