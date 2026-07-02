"""DEV-1629 — mechanical surface tests for the SLayer ``recommend_root_model``
tool in the claude_sdk agents.

Per ``feedback_no_prompt_content_tests``: NO assertions on prompt
natural-language content / anchor phrases. These tests only pin mechanical
contracts:

* ``recommend_root_model`` is on the tool surface of every in-scope claude_sdk
  agent (main query agents + OTF encoders), placed on the MAIN client (never the
  DISCOVERY client) for the partitioned v1 agents.
* It is treated as READ-ONLY: never in a write-normalize set, never in
  ``WRITE_SLAYER_TOOLS``, survives ``strip_write_slayer_tools`` (so the
  pre-encoded query-only path keeps it), and never hidden by a DISALLOWED list.
* The in-process native bridge resolves it — the installed slayer (>=0.9.3)
  advertises the real signature, including the ``root_hint`` parameter.
* The 158-line HOST DISCOVERY playbook is no longer imported on the claude_sdk
  side, while the pydantic_ai agents still import it (the claude_sdk-only scope).

Behavioural validation (root actually chosen well, ``root_hint`` honored) lives
in the cloud OTF smoke, not in stub-LLM tests.
"""

from __future__ import annotations

import importlib
import string

import pytest

TOOL = "recommend_root_model"
SLAYER_PREFIXED = f"mcp__slayer__{TOOL}"
NATIVE_FULL = f"mcp__bird-interact-tools__{TOOL}"


def _format_fields(template: str) -> set[str]:
    return {
        fname
        for _, fname, _, _ in string.Formatter().parse(template)
        if fname
    }


def _const(module_path: str, name: str) -> str:
    return getattr(importlib.import_module(module_path), name)


# ---------------------------------------------------------------------------
# 1. Main query / bridging surface — claude_sdk.agent
# ---------------------------------------------------------------------------


def test_tool_in_slayer_native_names():
    from bird_interact_agents.agents.claude_sdk import agent as A

    assert TOOL in A._SLAYER_NATIVE_NAMES


def test_tool_is_read_only_not_normalize_write():
    from bird_interact_agents.agents.claude_sdk import agent as A

    assert TOOL not in A._SLAYER_NATIVE_NORMALIZE_WRITE


def test_resolve_native_tool_builds_the_bridge():
    """The in-process native must resolve (currently KeyErrors until the name
    is added to ``_SLAYER_NATIVE_NAMES``)."""
    from bird_interact_agents.agents.claude_sdk import agent as A

    native = A.resolve_native_tool(TOOL)
    assert native is not None
    # ``tool(name, ...)`` uses the BARE name; the server prefix supplies the rest.
    assert getattr(native, "name", TOOL) == TOOL
    assert A.native_tool_full_name(TOOL) == NATIVE_FULL


def test_installed_slayer_advertises_root_hint():
    """Bridge derives the schema from the installed slayer's MCP server; the
    feature requires slayer >=0.9.3 whose tool carries ``root_hint``."""
    from bird_interact_agents.agents.claude_sdk import agent as A

    description, schema = A._slayer_tool_metadata(TOOL)
    assert isinstance(description, str) and description
    params = set((schema.get("properties") or {}).keys())
    assert {"items", "data_source", "root_hint", "format"} <= params
    # Query-formulation usage relies on root_hint being OPTIONAL; items required.
    required = set(schema.get("required") or [])
    assert "items" in required
    assert "root_hint" not in required
    assert "data_source" not in required


# ---------------------------------------------------------------------------
# 2. OTF one-shot / a-interact single-client surface — claude_sdk_otf.agent
#    (claude_sdk_otf_ainteract imports SLAYER_MCP_TOOLS from here)
# ---------------------------------------------------------------------------


def test_tool_in_otf_slayer_mcp_tools():
    from bird_interact_agents.agents.claude_sdk_otf import agent as A

    assert TOOL in A.SLAYER_MCP_TOOLS
    assert SLAYER_PREFIXED in A._slayer_tool_names()


def test_tool_not_in_otf_disallowed():
    from bird_interact_agents.agents.claude_sdk_otf import agent as A

    assert SLAYER_PREFIXED not in A.SLAYER_MCP_DISALLOWED_TOOL_NAMES
    assert TOOL not in A.SLAYER_MCP_DISALLOWED_TOOL_NAMES


def test_tool_not_in_otf_write_normalize_registry():
    """Read-only: never wired into the create/edit filter-normalization hook."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as A

    assert SLAYER_PREFIXED not in A._WRITE_TOOLS_NEEDING_NORMALIZATION
    assert TOOL not in A._WRITE_TOOLS_NEEDING_NORMALIZATION


def test_ainteract_v0_shares_the_same_slayer_mcp_tools():
    """v0 a-interact reuses the ONE-SHOT surface (imports the same list), so the
    tool reaches it by construction — not a drifted copy."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as one_shot
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import agent as ainteract

    assert ainteract.SLAYER_MCP_TOOLS is one_shot.SLAYER_MCP_TOOLS


def test_tool_survives_pre_encoded_strip():
    """Codex F4: the pre-encoded query-only allowed set (write tools stripped)
    must still expose recommend_root_model — it is read-only."""
    from bird_interact_agents.agents._pre_encoded import strip_write_slayer_tools
    from bird_interact_agents.agents.claude_sdk_otf import agent as A

    stripped = strip_write_slayer_tools(A.SLAYER_MCP_TOOLS)
    assert TOOL in stripped


# ---------------------------------------------------------------------------
# 3. Build-time encoder surface — claude_sdk_otf_encode.setup_encoder
# ---------------------------------------------------------------------------


def test_tool_in_encoder_slayer_mcp_tools():
    from bird_interact_agents.agents.claude_sdk_otf_encode import setup_encoder as S

    assert TOOL in S.SLAYER_MCP_TOOLS


def test_tool_not_in_encoder_disallowed():
    from bird_interact_agents.agents.claude_sdk_otf_encode import setup_encoder as S

    assert SLAYER_PREFIXED not in S.DISALLOWED_TOOL_NAMES
    assert TOOL not in S.DISALLOWED_TOOL_NAMES


def test_tool_not_in_encoder_write_normalize_registry():
    from bird_interact_agents.agents.claude_sdk_otf_encode import setup_encoder as S

    assert SLAYER_PREFIXED not in S._WRITE_TOOLS_NEEDING_NORMALIZATION
    assert TOOL not in S._WRITE_TOOLS_NEEDING_NORMALIZATION


# ---------------------------------------------------------------------------
# 4. v1 partition — MAIN, never DISCOVERY (claude_sdk_otf_v1 +
#    claude_sdk_otf_ainteract_v1 which inherits the one-shot lists)
# ---------------------------------------------------------------------------


def test_tool_on_v1_main_not_discovery():
    from bird_interact_agents.agents.claude_sdk_otf_v1 import agent as A

    assert TOOL in A._MAIN_NATIVE_BARE
    assert TOOL not in A._DISCOVERY_NATIVE_BARE
    assert NATIVE_FULL in A.MAIN_NATIVE_TOOL_NAMES
    assert NATIVE_FULL not in A.DISCOVERY_NATIVE_TOOL_NAMES


def test_tool_on_ainteract_v1_main_not_discovery():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as A

    assert NATIVE_FULL in A.MAIN_NATIVE_TOOL_NAMES
    assert NATIVE_FULL not in A.DISCOVERY_NATIVE_TOOL_NAMES


def test_ainteract_v1_main_inherits_one_shot_v1():
    """v1 a-interact composes its MAIN surface from the one-shot v1 list, so the
    tool reaches it by construction (not a hand-maintained copy)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as ainteract
    from bird_interact_agents.agents.claude_sdk_otf_v1 import agent as one_shot

    assert set(one_shot.MAIN_NATIVE_TOOL_NAMES) <= set(ainteract.MAIN_NATIVE_TOOL_NAMES)


def test_tool_is_read_only_global_invariant():
    """Read-only across the shared write-tool registry."""
    from bird_interact_agents.agents._pre_encoded import WRITE_SLAYER_TOOLS

    assert TOOL not in WRITE_SLAYER_TOOLS


def test_inspect_tool_on_surface_alongside_inspect_model():
    """DEV-1629: the SLayer single-entity `inspect` tool (for reading a known
    column's sample values) is on the surface ALONGSIDE `inspect_model`, and
    survives the pre-encoded write-strip (it is read-only)."""
    from bird_interact_agents.agents._pre_encoded import (
        strip_write_slayer_tools,
        strip_write_tool_names,
    )
    from bird_interact_agents.agents.claude_sdk import agent as base
    from bird_interact_agents.agents.claude_sdk_otf import agent as v0
    from bird_interact_agents.agents.claude_sdk_otf_v1 import agent as v1

    assert {"inspect", "inspect_model"} <= base._SLAYER_NATIVE_NAMES
    assert {"inspect", "inspect_model"} <= set(v0.SLAYER_MCP_TOOLS)
    inspect_full = "mcp__bird-interact-tools__inspect"
    assert inspect_full in v1.MAIN_NATIVE_TOOL_NAMES
    # read-only → survives the pre-encoded strip on both v0 and v1.
    assert "inspect" in strip_write_slayer_tools(v0.SLAYER_MCP_TOOLS)
    assert inspect_full in strip_write_tool_names(v1.MAIN_NATIVE_TOOL_NAMES)
    # bridge resolves its real schema from the installed slayer.
    desc, _schema = base._slayer_tool_metadata("inspect")
    assert isinstance(desc, str) and desc


def test_pre_encoded_v1_drops_discovery_only_inventory_tools():
    """DEV-1629 unshare: the v1 pre-encoded prompts must NOT advertise
    `list_datasources` / `models_summary` (not on the slayer v1 MAIN surface),
    while the v0 prompts (single client, both tools present) keep them."""
    from bird_interact_agents.agents import _pre_encoded_prompts as p

    for tool in ("list_datasources", "models_summary"):
        assert tool in p.SLAYER_PRE_ENCODED_ONE_SHOT
        assert tool in p.SLAYER_PRE_ENCODED_AINTERACT
        assert tool not in p.SLAYER_PRE_ENCODED_ONE_SHOT_V1
        assert tool not in p.SLAYER_PRE_ENCODED_AINTERACT_V1
    # The v1 agents consume the v1 variants (not the v0 shared constants).
    from bird_interact_agents.agents.claude_sdk_otf_v1 import agent as one_shot
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1 import agent as ainteract

    assert one_shot.SLAYER_PRE_ENCODED_ONE_SHOT is p.SLAYER_PRE_ENCODED_ONE_SHOT_V1
    assert ainteract.SLAYER_PRE_ENCODED_AINTERACT is p.SLAYER_PRE_ENCODED_AINTERACT_V1


# ---------------------------------------------------------------------------
# 5. Playbook divergence (Codex F7): claude_sdk side drops the playbook import;
#    pydantic_ai side keeps it. (claude_sdk-only scope.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "bird_interact_agents.agents.claude_sdk_otf_v1.prompts",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts",
        "bird_interact_agents.agents._pre_encoded_prompts",
    ],
)
def test_claude_sdk_prompts_no_longer_import_playbook(module_path):
    mod = importlib.import_module(module_path)
    assert not hasattr(mod, "_HOST_DISCOVERY_PLAYBOOK"), (
        f"{module_path} still imports the HOST DISCOVERY playbook; DEV-1629 "
        f"replaces it with recommend_root_model guidance on the claude_sdk side."
    )


@pytest.mark.parametrize(
    "module_path",
    [
        "bird_interact_agents.agents.pydantic_ai_recursive.prompts",
        "bird_interact_agents.agents.pydantic_ai_otf_encode.prompts",
    ],
)
def test_pydantic_ai_prompts_still_import_playbook(module_path):
    from bird_interact_agents.agents._host_discovery_playbook import (
        HOST_DISCOVERY_PLAYBOOK,
    )

    mod = importlib.import_module(module_path)
    assert getattr(mod, "_HOST_DISCOVERY_PLAYBOOK", None) is HOST_DISCOVERY_PLAYBOOK


# ---------------------------------------------------------------------------
# 6. Two shared guidance constants, placed by role (query vs encode) across
#    v0 + v1 + single-purpose prompts. Mechanical containment / no-format
#    drift-guards — NOT prompt-prose assertions.
# ---------------------------------------------------------------------------

_SHARED = "bird_interact_agents.agents._shared_otf_prompts"

# (module_path, public prompt constant) — the four OTF slayer prompts carry BOTH
# guidance blocks; the single-purpose prompts carry exactly one.
_OTF_SLAYER_PROMPTS = [
    ("bird_interact_agents.agents.claude_sdk_otf.prompts", "SLAYER_OTF_ONE_SHOT"),
    ("bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts", "SLAYER_OTF_AINTERACT"),
    ("bird_interact_agents.agents.claude_sdk_otf_v1.prompts", "SLAYER_OTF_ONE_SHOT"),
    ("bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts", "SLAYER_OTF_AINTERACT"),
]
_PRE_ENCODED_PROMPTS = [
    ("bird_interact_agents.agents._pre_encoded_prompts", "SLAYER_PRE_ENCODED_ONE_SHOT"),
    ("bird_interact_agents.agents._pre_encoded_prompts", "SLAYER_PRE_ENCODED_AINTERACT"),
]
_ENCODER_PROMPT = ("bird_interact_agents.agents.claude_sdk_otf_encode.prompts", "ENCODER_PROMPT")


@pytest.mark.parametrize("const_name", ["QUERY_ROOT_GUIDANCE", "ENCODE_HOST_GUIDANCE"])
def test_guidance_constant_exists_and_has_no_format_fields(const_name):
    """Both blocks are concatenated into templates that are later
    ``.format(budget=…, db_name=…, user_query=…)``-ed; a stray ``{field}``
    would raise at those call sites."""
    const = _const(_SHARED, const_name)
    assert isinstance(const, str) and const.strip()
    assert _format_fields(const) == set(), (
        f"{const_name} must have no .format() fields; got {_format_fields(const)}"
    )


@pytest.mark.parametrize("module_path,const_name", _OTF_SLAYER_PROMPTS + _PRE_ENCODED_PROMPTS)
def test_query_root_guidance_present(module_path, const_name):
    """Query-root guidance (no hint) is in every query-formulating prompt:
    both OTF slayer prompts (v0+v1) and the pre-encoded query-only prompts."""
    guidance = _const(_SHARED, "QUERY_ROOT_GUIDANCE")
    assert guidance in _const(module_path, const_name)


def test_query_root_guidance_absent_from_encoder():
    """The build-time encoder never formulates a benchmark query."""
    guidance = _const(_SHARED, "QUERY_ROOT_GUIDANCE")
    assert guidance not in _const(*_ENCODER_PROMPT)


@pytest.mark.parametrize("module_path,const_name", _OTF_SLAYER_PROMPTS + [_ENCODER_PROMPT])
def test_encode_host_guidance_present(module_path, const_name):
    """Encode-host guidance (with root_hint) is in every entity-creating
    prompt: both OTF slayer prompts (v0+v1) and the build-time encoder."""
    guidance = _const(_SHARED, "ENCODE_HOST_GUIDANCE")
    assert guidance in _const(module_path, const_name)


@pytest.mark.parametrize("module_path,const_name", _PRE_ENCODED_PROMPTS)
def test_encode_host_guidance_absent_from_pre_encoded(module_path, const_name):
    """Pre-encoded agents only query already-built models — no host choice."""
    guidance = _const(_SHARED, "ENCODE_HOST_GUIDANCE")
    assert guidance not in _const(module_path, const_name)


def test_role_split_root_hint_token():
    """The whole point of the split: the ENCODE block uses ``root_hint`` and the
    QUERY block must NOT (query formulation trusts the auto pick). This checks
    the API-parameter token, not prose."""
    query = _const(_SHARED, "QUERY_ROOT_GUIDANCE")
    encode = _const(_SHARED, "ENCODE_HOST_GUIDANCE")
    assert "root_hint" in encode
    assert "root_hint" not in query


@pytest.mark.parametrize(
    "module_path,const_name",
    _OTF_SLAYER_PROMPTS + _PRE_ENCODED_PROMPTS,
)
def test_playbook_text_removed_from_claude_sdk_prompts(module_path, const_name):
    """The 158-line playbook is gone from every claude_sdk prompt — including the
    v0 monoliths that INLINED it (no import to check there). Drift-guard on the
    canonical constant, not a prose assertion."""
    from bird_interact_agents.agents._host_discovery_playbook import (
        HOST_DISCOVERY_PLAYBOOK,
    )

    assert HOST_DISCOVERY_PLAYBOOK not in _const(module_path, const_name)
    # (The build-time encoder never used the shared playbook constant — it had
    # its own inline host prose — so a HOST_DISCOVERY_PLAYBOOK-absence check
    # there is a false-green. Its old prose removal is covered by the positive
    # ENCODE_HOST_GUIDANCE-present test instead.)


# A dangling pointer to the deleted playbook ("follow the HOST DISCOVERY
# playbook below") is worse than harmless — it sends the agent looking for
# guidance that is no longer there. This is an API/anchor token, not prose:
# the phrase names a retired mechanism, so its absence is a mechanical contract.
_LEGACY_PLAYBOOK_ANCHOR = "HOST DISCOVERY playbook"


@pytest.mark.parametrize(
    "module_path,const_name",
    _OTF_SLAYER_PROMPTS + _PRE_ENCODED_PROMPTS + [_ENCODER_PROMPT],
)
def test_no_dangling_playbook_pointer(module_path, const_name):
    assert _LEGACY_PLAYBOOK_ANCHOR not in _const(module_path, const_name)
