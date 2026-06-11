"""Tests for shared OTF prompt constants (_shared_otf_prompts.py).

Covers three concerns:
1. SHA-256 snapshot tests that SLAYER_OTF_ONE_SHOT and SLAYER_OTF_AINTERACT
   are byte-for-byte unchanged after the refactoring that imports constants
   from _shared_otf_prompts.
2. That the shared constants render correctly with their format params.
3. That the shared constants appear verbatim in both slayer and raw OTF
   prompts (after implementation), and that raw prompts are free of
   SLayer-specific vocabulary.
"""

from __future__ import annotations

import hashlib

import pytest

# ---------------------------------------------------------------------------
# SHA-256 snapshot tests — must remain passing before AND after the
# refactoring that extracts shared constants from the slayer prompts.
# Hashes were captured from the pre-refactoring source.
# ---------------------------------------------------------------------------

# Hashes re-baselined for the DEV-1545 + DEV-1546 + DEV-1550 merge:
#   * DEV-1545 added `_TABLE_SET_PROBE` (one-shot + a-interact) and
#     `_GRADER_ZERO_VS_ONE_DIAGNOSTIC` (a-interact only).
#   * DEV-1546 added `_DEDUP_VS_RAW_ROWS` (both flavours), rewrote
#     `_SLAYER_SQL_ARTIFACT_CHECK` item-1 (primary fix is now
#     `distinct_dimension_values: false`), and migrated the per-kind
#     search kwargs to `max_results` + the unified `results` list.
#   * DEV-1550 added the new "READ A KNOWN MEMORY'S FULL BODY" drill-in
#     paragraph documenting the compact-mode opt-out
#     (`search(entities=["memory:<id>"], max_results=1, compact=False,
#     cypher_filter='MATCH (n:Memory) RETURN n.id AS id')`), inserted as
#     a sibling between the existing column-drill-in paragraph and the
#     `ENCODE-THEN-QUERY DISCIPLINE:` header in the new shared
#     `_SLAYER_TOOLS_BLOCK` (extracted from the previously byte-identical
#     `_AINTERACT_SLAYER_TOOLS` / `_ENCODE_CORE_HEAD`). DEV-1550 also
#     adds the `:Column` / `:Memory` cypher kind filter to all
#     known-entity drill-in patterns in the prompts + the host-discovery
#     playbook (`_HOST_DISCOVERY_PLAYBOOK`).
#
# The snapshot's purpose stays the same: catch ACCIDENTAL prompt drift
# on later refactors; deliberate prompt changes re-baseline here.
_ONE_SHOT_SHA256 = "2ef7a1fc6abacc9a8e8efc52701a74ddc6559793be2fad52421c1c50bbe7d6ef"
_AINTERACT_SHA256 = "b35e3e5454bb37b2028515c917100d12d5d35ce3e0fed82abbabf6c5970a8708"


def test_slayer_otf_one_shot_unchanged():
    """Byte-for-byte contract against the re-baselined snapshot."""
    from bird_interact_agents.agents.claude_sdk_otf.prompts import SLAYER_OTF_ONE_SHOT

    digest = hashlib.sha256(SLAYER_OTF_ONE_SHOT.encode()).hexdigest()
    assert digest == _ONE_SHOT_SHA256, (
        f"SLAYER_OTF_ONE_SHOT changed (len={len(SLAYER_OTF_ONE_SHOT)}).\n"
        f"  expected: {_ONE_SHOT_SHA256}\n"
        f"  actual:   {digest}"
    )


def test_slayer_otf_ainteract_unchanged():
    """Byte-for-byte contract against the re-baselined snapshot."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    digest = hashlib.sha256(SLAYER_OTF_AINTERACT.encode()).hexdigest()
    assert digest == _AINTERACT_SHA256, (
        f"SLAYER_OTF_AINTERACT changed (len={len(SLAYER_OTF_AINTERACT)}).\n"
        f"  expected: {_AINTERACT_SHA256}\n"
        f"  actual:   {digest}"
    )


# ---------------------------------------------------------------------------
# Shared-constant accessibility
# ---------------------------------------------------------------------------

def test_shared_constants_all_nonempty():
    """All exported template constants must be non-empty strings.

    Covers the six original constants AND the four DEV-1534 additions
    (column-names rule, SLayer artifact sanity-check, pivot-after-3,
    user-sim trust calibration) so a future re-baseline that
    accidentally drops one is caught here.
    """
    from bird_interact_agents.agents._shared_otf_prompts import (
        _NO_USER_TO_CONSULT,
        _DECOMPOSE_DISCIPLINE,
        _RULE_0_ASK_BEFORE,
        _ASK_AGAIN_RULE,
        _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT,
        _PRE_SUBMIT_MUTATION_CHECK_AINTERACT,
        _COLUMN_NAMES_DONT_AFFECT_GRADING,
        _SLAYER_SQL_ARTIFACT_CHECK,
        _PIVOT_AFTER_REPEATED_FAILURES,
        _USER_SIM_TRUST_CALIBRATION,
    )

    for name, val in [
        ("_NO_USER_TO_CONSULT", _NO_USER_TO_CONSULT),
        ("_DECOMPOSE_DISCIPLINE", _DECOMPOSE_DISCIPLINE),
        ("_RULE_0_ASK_BEFORE", _RULE_0_ASK_BEFORE),
        ("_ASK_AGAIN_RULE", _ASK_AGAIN_RULE),
        ("_PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT", _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT),
        ("_PRE_SUBMIT_MUTATION_CHECK_AINTERACT", _PRE_SUBMIT_MUTATION_CHECK_AINTERACT),
        ("_COLUMN_NAMES_DONT_AFFECT_GRADING", _COLUMN_NAMES_DONT_AFFECT_GRADING),
        ("_SLAYER_SQL_ARTIFACT_CHECK", _SLAYER_SQL_ARTIFACT_CHECK),
        ("_PIVOT_AFTER_REPEATED_FAILURES", _PIVOT_AFTER_REPEATED_FAILURES),
        ("_USER_SIM_TRUST_CALIBRATION", _USER_SIM_TRUST_CALIBRATION),
    ]:
        assert isinstance(val, str) and val.strip(), f"{name} must be a non-empty string"


# ---------------------------------------------------------------------------
# Rendering tests — shared constants with their specific format params
# ---------------------------------------------------------------------------

def test_no_user_to_consult_renders_with_sources_desc():
    from bird_interact_agents.agents._shared_otf_prompts import _NO_USER_TO_CONSULT

    rendered = _NO_USER_TO_CONSULT.format(sources_desc="the schema and knowledge definitions")
    assert "the schema and knowledge definitions" in rendered
    assert "NO user to consult" in rendered
    assert "autonomously" in rendered


def test_rule0_renders_for_slayer_params():
    """Slayer-specific rendering: ENCODE + submit_query."""
    from bird_interact_agents.agents._shared_otf_prompts import _RULE_0_ASK_BEFORE

    rendered = _RULE_0_ASK_BEFORE.format(
        action_label="ENCODE",
        action_context="BEFORE the encoding loop below,",
        submit_tool="submit_query",
    )
    assert "RULE 0 — ASK BEFORE YOU ENCODE." in rendered
    assert "BEFORE the encoding loop below," in rendered
    assert "submit_query" in rendered
    assert "ask_user" in rendered
    assert "operationalisation" in rendered


def test_rule0_renders_for_raw_params():
    """Raw-mode rendering: SUBMIT + submit_sql."""
    from bird_interact_agents.agents._shared_otf_prompts import _RULE_0_ASK_BEFORE

    rendered = _RULE_0_ASK_BEFORE.format(
        action_label="SUBMIT",
        action_context="BEFORE writing your SQL query,",
        submit_tool="submit_sql",
    )
    assert "RULE 0 — ASK BEFORE YOU SUBMIT." in rendered
    assert "BEFORE writing your SQL query," in rendered
    assert "submit_sql" in rendered
    assert "submit_query" not in rendered


def test_ask_again_rule_renders_with_knowledge_source():
    from bird_interact_agents.agents._shared_otf_prompts import _ASK_AGAIN_RULE

    for source, expected in [
        ("a memory", "not pinned by a memory or column"),
        ("a knowledge definition", "not pinned by a knowledge definition or column"),
    ]:
        rendered = _ASK_AGAIN_RULE.format(knowledge_source=source)
        assert expected in rendered
        assert "ASK AGAIN IF NEEDED" in rendered
        assert "ask_user" in rendered


def test_pre_submit_check_one_shot_renders_for_slayer():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT,
    )

    rendered = _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT.format(
        submit_tool="submit_query", clause_b="encoded KB",
    )
    assert "submit_query" in rendered
    assert "encoded KB" in rendered
    assert "PRE-SUBMIT MUTATION CHECK" in rendered
    assert "submit_sql" not in rendered


def test_pre_submit_check_one_shot_renders_for_raw():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT,
    )

    rendered = _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT.format(
        submit_tool="submit_sql", clause_b="knowledge definition",
    )
    assert "submit_sql" in rendered
    assert "knowledge definition" in rendered
    assert "submit_query" not in rendered


def test_pre_submit_check_ainteract_renders_for_slayer():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_AINTERACT,
    )

    rendered = _PRE_SUBMIT_MUTATION_CHECK_AINTERACT.format(
        submit_tool="submit_query", clause_c="encoded KB",
    )
    assert "submit_query" in rendered
    assert "encoded KB" in rendered
    assert "ask_user" in rendered
    assert "PRE-SUBMIT MUTATION CHECK" in rendered


def test_pre_submit_check_ainteract_renders_for_raw():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_AINTERACT,
    )

    rendered = _PRE_SUBMIT_MUTATION_CHECK_AINTERACT.format(
        submit_tool="submit_sql", clause_c="knowledge definition",
    )
    assert "submit_sql" in rendered
    assert "knowledge definition" in rendered
    assert "submit_query" not in rendered


# ---------------------------------------------------------------------------
# Shared constants appear verbatim in both slayer prompts.
# These tests should pass both BEFORE and AFTER the refactoring —
# before because the slayer prompts already contain these strings;
# after because refactoring preserves byte-for-byte identity (see snapshots).
# ---------------------------------------------------------------------------

def test_decompose_discipline_in_slayer_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import _DECOMPOSE_DISCIPLINE
    from bird_interact_agents.agents.claude_sdk_otf.prompts import SLAYER_OTF_ONE_SHOT

    assert _DECOMPOSE_DISCIPLINE in SLAYER_OTF_ONE_SHOT


def test_decompose_discipline_in_slayer_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _DECOMPOSE_DISCIPLINE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    assert _DECOMPOSE_DISCIPLINE in SLAYER_OTF_AINTERACT


def test_no_user_to_consult_in_slayer_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import _NO_USER_TO_CONSULT
    from bird_interact_agents.agents.claude_sdk_otf.prompts import SLAYER_OTF_ONE_SHOT

    rendered = _NO_USER_TO_CONSULT.format(
        sources_desc="the memories and column\ndescriptions"
    )
    assert rendered in SLAYER_OTF_ONE_SHOT


def test_rule0_rendered_in_slayer_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _RULE_0_ASK_BEFORE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    rendered = _RULE_0_ASK_BEFORE.format(
        action_label="ENCODE",
        action_context="BEFORE the encoding loop below,",
        submit_tool="submit_query",
    )
    assert rendered in SLAYER_OTF_AINTERACT


def test_ask_again_in_slayer_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _ASK_AGAIN_RULE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    rendered = _ASK_AGAIN_RULE.format(knowledge_source="a memory")
    assert rendered in SLAYER_OTF_AINTERACT


def test_pre_submit_one_shot_in_slayer_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT,
    )
    from bird_interact_agents.agents.claude_sdk_otf.prompts import SLAYER_OTF_ONE_SHOT

    rendered = _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT.format(
        submit_tool="submit_query", clause_b="encoded KB",
    )
    assert rendered in SLAYER_OTF_ONE_SHOT


def test_pre_submit_ainteract_in_slayer_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_AINTERACT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    rendered = _PRE_SUBMIT_MUTATION_CHECK_AINTERACT.format(
        submit_tool="submit_query", clause_c="encoded KB",
    )
    assert rendered in SLAYER_OTF_AINTERACT


# ---------------------------------------------------------------------------
# DEV-1550 A3: shared `_SLAYER_TOOLS_BLOCK` lives in `_shared_otf_prompts`
# and appears verbatim in both slayer prompts.
# ---------------------------------------------------------------------------


def test_slayer_tools_block_in_slayer_one_shot():
    """The extracted `_SLAYER_TOOLS_BLOCK` is the source of truth shared
    between the ainteract and one-shot SLayer prompts. Mechanical
    containment: not a prompt-content behaviour test (no anchor-phrase
    grep) — consistent with `feedback_no_prompt_content_tests`."""
    from bird_interact_agents.agents._shared_otf_prompts import _SLAYER_TOOLS_BLOCK
    from bird_interact_agents.agents.claude_sdk_otf.prompts import SLAYER_OTF_ONE_SHOT

    assert _SLAYER_TOOLS_BLOCK in SLAYER_OTF_ONE_SHOT


def test_slayer_tools_block_in_slayer_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _SLAYER_TOOLS_BLOCK
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    assert _SLAYER_TOOLS_BLOCK in SLAYER_OTF_AINTERACT


# ---------------------------------------------------------------------------
# Shared constants appear verbatim in raw prompts (requires implementation).
# These tests FAIL until the raw agent modules are created.
# ---------------------------------------------------------------------------

def test_decompose_discipline_in_raw_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import _DECOMPOSE_DISCIPLINE
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import (
        RAW_OTF_ONE_SHOT,
    )

    assert _DECOMPOSE_DISCIPLINE in RAW_OTF_ONE_SHOT


def test_decompose_discipline_in_raw_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _DECOMPOSE_DISCIPLINE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    assert _DECOMPOSE_DISCIPLINE in RAW_OTF_AINTERACT


def test_pre_submit_one_shot_in_raw_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import (
        RAW_OTF_ONE_SHOT,
    )

    rendered = _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT.format(
        submit_tool="submit_sql", clause_b="knowledge definition",
    )
    assert rendered in RAW_OTF_ONE_SHOT


def test_pre_submit_ainteract_in_raw_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_AINTERACT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    rendered = _PRE_SUBMIT_MUTATION_CHECK_AINTERACT.format(
        submit_tool="submit_sql", clause_c="knowledge definition",
    )
    assert rendered in RAW_OTF_AINTERACT


def test_no_user_to_consult_in_raw_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import _NO_USER_TO_CONSULT
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import (
        RAW_OTF_ONE_SHOT,
    )

    # Raw one-shot has no user to consult; the template must appear rendered.
    assert "NO user to consult" in RAW_OTF_ONE_SHOT


def test_rule0_rendered_in_raw_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _RULE_0_ASK_BEFORE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    rendered = _RULE_0_ASK_BEFORE.format(
        action_label="SUBMIT",
        action_context="BEFORE writing your SQL query,",
        submit_tool="submit_sql",
    )
    assert rendered in RAW_OTF_AINTERACT


def test_ask_again_in_raw_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _ASK_AGAIN_RULE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    rendered = _ASK_AGAIN_RULE.format(knowledge_source="a knowledge definition")
    assert rendered in RAW_OTF_AINTERACT


# ---------------------------------------------------------------------------
# Raw prompts must NOT contain SLayer-specific vocabulary.
# Fails until the raw agent modules are created.
# ---------------------------------------------------------------------------

_SLAYER_VOCAB = [
    "submit_query",
    "create_model",
    "edit_model",
    "[kb=",
    "memory:",
    "slayer",
    "SLayer",
    "mcp__slayer__",
]


def test_raw_one_shot_prompt_absent_slayer_vocab():
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import RAW_OTF_ONE_SHOT

    # Render with placeholder values (budget/db_name/user_query not SLayer-related).
    rendered = RAW_OTF_ONE_SHOT.format(
        budget=20.0, db_name="shop", user_query="how many items?",
    )
    for term in _SLAYER_VOCAB:
        assert term not in rendered, (
            f"raw one-shot prompt must not contain SLayer term {term!r}"
        )


def test_raw_ainteract_prompt_absent_slayer_vocab():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    rendered = RAW_OTF_AINTERACT.format(
        budget=20.0, db_name="shop", user_query="how many items?",
    )
    for term in _SLAYER_VOCAB:
        assert term not in rendered, (
            f"raw ainteract prompt must not contain SLayer term {term!r}"
        )


def test_raw_one_shot_prompt_mentions_submit_sql():
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import RAW_OTF_ONE_SHOT

    assert "submit_sql" in RAW_OTF_ONE_SHOT


def test_raw_ainteract_prompt_mentions_submit_sql_and_ask_user():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    assert "submit_sql" in RAW_OTF_AINTERACT
    assert "ask_user" in RAW_OTF_AINTERACT


def test_raw_prompts_use_synthetic_examples_only():
    """Guards feedback_prompts_synthetic_examples_only: no real eval-set
    DB / table / column / value names may appear in either raw prompt."""
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import RAW_OTF_ONE_SHOT
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    banned = [
        "households", "tenure_type", "income_bracket", "dwelling_class",
        "socsupport", "service_types", "stellardist", "photo_band",
        "taguatinga",
    ]
    for name, text in [("RAW_OTF_ONE_SHOT", RAW_OTF_ONE_SHOT), ("RAW_OTF_AINTERACT", RAW_OTF_AINTERACT)]:
        low = text.lower()
        for banned_name in banned:
            assert banned_name not in low, (
                f"real eval-set name {banned_name!r} leaked into {name}"
            )
