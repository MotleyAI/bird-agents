"""Tests for shared OTF prompt constants (_shared_otf_prompts.py).

Covers three concerns:
1. SHA-256 snapshot tests that SLAYER_OTF_ONE_SHOT and SLAYER_OTF_AINTERACT
   are byte-for-byte unchanged after the refactoring that imports constants
   from _shared_otf_prompts.
2. That the shared constants render correctly with their format params.
3. That the mode-agnostic shared constants appear verbatim in both slayer
   and raw OTF prompts (after implementation), and that raw prompts are
   free of SLayer-specific vocabulary. (The DEV-1591
   ``_COMPACT_SEARCH_DISCIPLINE`` constant is SLayer-only — it talks about
   the SLayer ``search`` tool's ``compact`` / ``cypher_filter`` args — so
   it lands only in the slayer prompts, never the raw ones.)
"""

from __future__ import annotations

import hashlib

import pytest

# ---------------------------------------------------------------------------
# SHA-256 snapshot tests — must remain passing before AND after the
# refactoring that extracts shared constants from the slayer prompts.
# Hashes were captured from the pre-refactoring source.
# ---------------------------------------------------------------------------

# Hashes re-baselined for the DEV-1555 Stage-1 prompt fixes (drop
# `query_nested` from agent vocabulary, document single-tool
# list-of-stages shape, note the partition-deny redirect for
# `mcp__slayer__query`). The snapshot's purpose is the same — catch
# ACCIDENTAL prompt drift on later refactors; deliberate prompt changes
# re-baseline here.
# Re-baselined again for the DEV-1555 CR r1 / O1 unification: single
# `query` tool accepts object OR list of stages; legacy `query_nested`
# and `query_json` single-string parameter are gone from both v0 and
# v1 prompts.
# Re-baselined once more after the origin/main merge layered in
# DEV-1545 (_TABLE_SET_PROBE), DEV-1546 (_DEDUP_VS_RAW_ROWS), and
# DEV-1550 (ModelColumn label + memory drill-in paragraph in
# _SLAYER_TOOLS_BLOCK).
# Re-baselined for DEV-1581 R2: the two-stage v1 main loop no longer holds the
# schema-introspection tools (search / inspect_model / models_summary) — they
# moved to the discovery client reached via ask_discovery — so the v1 slayer
# prompts now route schema/sample-value/entity discovery through ask_discovery
# while keeping KB lookups (get_knowledge_definition) on the main surface.
# Re-baselined for DEV-1623: the shared `_SAMPLE_VALUE_FILTER_MANDATE` fragment
# (mandate sample-value-aware filter literals to cut submit-verify thrash) is
# now spliced into both v1 slayer prompts after the rules-2/3 block, rendered
# with `sample_source="`ask_discovery`"`. Reworded per PR #72 review to assign
# each variation kind to exactly one strategy (case/whitespace -> LOWER(TRIM);
# abbreviation/spelling -> IN-set enumeration). Round 2 (Codex): also narrowed
# rule 3's "Symmetric companion" bullet itself so it no longer lists
# case/whitespace among the extend-the-IN-set triggers — the two rules are now
# fully complementary with zero overlap.
# Re-baselined for DEV-1629: (1) the HOST DISCOVERY playbook is retired from the
# claude_sdk prompts in favour of the shared `QUERY_ROOT_GUIDANCE` /
# `ENCODE_HOST_GUIDANCE` blocks that drive the SLayer `recommend_root_model`
# tool (query = no hint; encode = descriptions-first `root_hint`); (2) slayer v1
# main now HOLDS `search` / `inspect_model` directly, so both v1 prompts say to
# call them directly (and read sample values via them) instead of routing every
# introspection through `ask_discovery`.
# Re-baselined for the DEV-1629 follow-up (merged via origin/main): the v1
# slayer inventory line was reworded to list `search` (find) / `inspect_model`
# (whole model) / `inspect` (a single known column's Description + Sample
# values), moving these two digests.
# DEV-1591 ∩ DEV-1629 merge note: the DEV-1591 search-vs-inspect compact
# discipline reaches the v1 slayer MAIN loop through
# `partition.build_main_workflow_note` (appended to these prompts at runtime,
# see agent.py), NOT through the static `SLAYER_OTF_ONE_SHOT` /
# `SLAYER_OTF_AINTERACT` constants hashed here — so the discipline splice does
# not itself move these digests (the rewording above did).
# Re-baselined for DEV-1646 (mode B, reference-before-define): a `DEFINE BEFORE
# YOU REFERENCE` fragment (`_DEFINE_BEFORE_REFERENCE`) was spliced into the four
# SLAYER OTF prompts after `ENCODE_HOST_GUIDANCE` — any name in a
# filter/dimension/measure/order must already exist on the model that stage
# queries; create it via create_model/edit_model (or project it from the prior
# stage) before referencing it. It is OTF-only (NOT in the shared
# `QUERY_ROOT_GUIDANCE`), so the pre-encoded prompts — whose write tools are
# stripped — are unchanged. Cuts the turn-waste where the agent filters on an
# undefined nested-stage name and self-corrects.
# DEV-1668 re-baseline: the shared SLAYER-TOOLS onboarding line was re-pointed
# from the removed `help` tool to
# `inspect(reference="memory:help.intro", entity_type="memory", compact=False)`
# (the only byte change vs the prior snapshot).
_ONE_SHOT_SHA256 = "157efb9d19e961cbb1973e001e69516b799c495b01833ca4a90bc9f1d31e988d"
_AINTERACT_SHA256 = "9a43d3ebdc125ea45a3a6027a2cd2a533e40d11ae75b7f108619a676a28062ba"


def test_slayer_otf_one_shot_unchanged():
    """Byte-for-byte contract against the re-baselined snapshot."""
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import SLAYER_OTF_ONE_SHOT

    digest = hashlib.sha256(SLAYER_OTF_ONE_SHOT.encode()).hexdigest()
    assert digest == _ONE_SHOT_SHA256, (
        f"SLAYER_OTF_ONE_SHOT changed (len={len(SLAYER_OTF_ONE_SHOT)}).\n"
        f"  expected: {_ONE_SHOT_SHA256}\n"
        f"  actual:   {digest}"
    )


def test_slayer_otf_ainteract_unchanged():
    """Byte-for-byte contract against the re-baselined snapshot."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
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
        _COMPACT_SEARCH_DISCIPLINE,
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
        ("_COMPACT_SEARCH_DISCIPLINE", _COMPACT_SEARCH_DISCIPLINE),
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
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import SLAYER_OTF_ONE_SHOT

    assert _DECOMPOSE_DISCIPLINE in SLAYER_OTF_ONE_SHOT


def test_decompose_discipline_in_slayer_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _DECOMPOSE_DISCIPLINE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    assert _DECOMPOSE_DISCIPLINE in SLAYER_OTF_AINTERACT


def test_no_user_to_consult_in_slayer_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import _NO_USER_TO_CONSULT
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import SLAYER_OTF_ONE_SHOT

    rendered = _NO_USER_TO_CONSULT.format(
        sources_desc="the memories and column\ndescriptions"
    )
    assert rendered in SLAYER_OTF_ONE_SHOT


def test_rule0_rendered_in_slayer_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _RULE_0_ASK_BEFORE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    rendered = _ASK_AGAIN_RULE.format(knowledge_source="a memory")
    assert rendered in SLAYER_OTF_AINTERACT


def test_pre_submit_one_shot_in_slayer_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import SLAYER_OTF_ONE_SHOT

    rendered = _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT.format(
        submit_tool="submit_query", clause_b="encoded KB",
    )
    assert rendered in SLAYER_OTF_ONE_SHOT


def test_pre_submit_ainteract_in_slayer_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_AINTERACT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    rendered = _PRE_SUBMIT_MUTATION_CHECK_AINTERACT.format(
        submit_tool="submit_query", clause_c="encoded KB",
    )
    assert rendered in SLAYER_OTF_AINTERACT


# ---------------------------------------------------------------------------
# Shared constants appear verbatim in raw prompts (requires implementation).
# These tests FAIL until the raw agent modules are created.
# ---------------------------------------------------------------------------

def test_decompose_discipline_in_raw_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import _DECOMPOSE_DISCIPLINE
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import (
        RAW_OTF_ONE_SHOT,
    )

    assert _DECOMPOSE_DISCIPLINE in RAW_OTF_ONE_SHOT


def test_decompose_discipline_in_raw_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _DECOMPOSE_DISCIPLINE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
        RAW_OTF_AINTERACT,
    )

    assert _DECOMPOSE_DISCIPLINE in RAW_OTF_AINTERACT


def test_pre_submit_one_shot_in_raw_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _PRE_SUBMIT_MUTATION_CHECK_ONE_SHOT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import (
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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
        RAW_OTF_AINTERACT,
    )

    rendered = _PRE_SUBMIT_MUTATION_CHECK_AINTERACT.format(
        submit_tool="submit_sql", clause_c="knowledge definition",
    )
    assert rendered in RAW_OTF_AINTERACT


def test_no_user_to_consult_in_raw_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import _NO_USER_TO_CONSULT
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import (
        RAW_OTF_ONE_SHOT,
    )

    # Raw one-shot has no user to consult; the template must appear rendered.
    assert "NO user to consult" in RAW_OTF_ONE_SHOT


def test_rule0_rendered_in_raw_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import _RULE_0_ASK_BEFORE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
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
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
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
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import RAW_OTF_ONE_SHOT

    # Render with placeholder values (budget/db_name/user_query not SLayer-related).
    rendered = RAW_OTF_ONE_SHOT.format(
        budget=20.0, db_name="shop", user_query="how many items?",
    )
    for term in _SLAYER_VOCAB:
        assert term not in rendered, (
            f"raw one-shot prompt must not contain SLayer term {term!r}"
        )


def test_raw_ainteract_prompt_absent_slayer_vocab():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
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
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import RAW_OTF_ONE_SHOT

    assert "submit_sql" in RAW_OTF_ONE_SHOT


def test_raw_ainteract_prompt_mentions_submit_sql_and_ask_user():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
        RAW_OTF_AINTERACT,
    )

    assert "submit_sql" in RAW_OTF_AINTERACT
    assert "ask_user" in RAW_OTF_AINTERACT


def test_raw_prompts_use_synthetic_examples_only():
    """Guards feedback_prompts_synthetic_examples_only: no real eval-set
    DB / table / column / value names may appear in either raw prompt."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import RAW_OTF_ONE_SHOT
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
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
