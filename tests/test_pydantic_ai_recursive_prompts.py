"""Prompt-coverage guard for the recursive adapter.

The recursive adapter splits SLAYER_A_INTERACT (≈230 lines) across three
prompts. Every directive in the original must land in at least one of the
three new prompts (matrix per the spec); each agent only sees the bits
relevant to its task.

These tests pin coverage so a future trim doesn't quietly drop a load-
bearing nudge. They also pin the new hardened directives the recursive
adapter introduces — the active count-check, the override rule, the
compound-naming default, the help mandate, the Step A table shape — so a
prompt rewrite can't accidentally regress the B-extra / B-missing fixes
this whole adapter exists to ship.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def prompts():
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts as p

    return p


@pytest.fixture(scope="module")
def all_three_concatenated(prompts):
    return (
        prompts.ROOT_CLARIFIER_PROMPT
        + "\n\n=====\n\n"
        + prompts.SUB_CLARIFIER_PROMPT
        + "\n\n=====\n\n"
        + prompts.QUERY_CONSTRUCTOR_PROMPT
    )


# ---------------------------------------------------------------------------
# 22: SLAYER_A_INTERACT coverage — every directive from the matrix lands
#     in at least one of the three prompts.
# ---------------------------------------------------------------------------


# Curated from the SLAYER_A_INTERACT coverage matrix in the spec. Generic
# substrings only — no task IDs (per the "no task IDs in prompts" rule).
COVERAGE_SUBSTRINGS = [
    # L23-27 (help + colon-aggregation)
    "help",
    "colon",
    # L29-35 (decompose into logical blocks)
    "logical block",
    # L36-42 (surface-form details)
    "case",
    # L44-56 (search + ask_user loop, hints not authorisation)
    "search",
    "HINTS",
    # L57-65 (specific complete details)
    "SPECIFIC",
    # L67-68 (composite reply iteration)
    "composite",
    # L70-79 (AND / both / multiple criteria)
    "AND",
    "both",
    # L81-149 (five operationalisations)
    "Aggregation",
    "Grouping",
    "Sort direction",
    "numeric constants",
    "Units of measure",
    # L155-162 (deferred memories)
    "deferred",
    # L164-165 (≥1 ask_user)
    "ask_user",
    # L167-170 (qualifier→encoding map)
    "qualifier",
    # L172-181 (echo back projection list)
    "echo",
    "projection",
    # L185-186 (search with full query)
    "complete original question",
    # L188 (`query` tool)
    "`query`",
    # L190 (`submit_query`)
    "`submit_query`",
    # L192-196 (JSONB trap)
    "JSONB",
    # L197-206 (output shape exactly)
    "output shape",
    # L207-211 (LIMIT rules)
    "LIMIT",
    # L212-216 (how many vs list)
    "how many",
    # L217-223 (MUST call submit_query)
    "MUST",
    # L224-237 (filter syntax)
    "filter",
    "ModelExtension",
    # L239-244 (budget breakdown)
    "bird-coin",
]


def test_slayer_a_interact_coverage(all_three_concatenated):
    """Every directive substring from the SLAYER_A_INTERACT coverage
    matrix must appear in at least one of the three prompts. Case-
    insensitive match — the prompts may rephrase capitalisation, but the
    concept must be reachable."""
    blob = all_three_concatenated.lower()
    missing = [s for s in COVERAGE_SUBSTRINGS if s.lower() not in blob]
    assert not missing, (
        f"SLAYER_A_INTERACT directives missing from new prompts: {missing}"
    )


# Distinctive verbatim phrases from SLAYER_A_INTERACT. These are
# load-bearing — generic substring matches like "AND" or "filter" pass
# even when the actual directive is gone, so we anchor on phrases that
# are specific to the original prompt's intent.
ANCHORED_PHRASES = [
    "hints, not authorisation",                # search-not-authorisation
    "labelled formula",                        # only-skip-when-quoted rule
    "single biggest miss",                     # AND-conjunct count rule
    "each side as its own filter",             # AND-conjunct application
    "concrete predicate",                      # ask for specific predicate
    "propose your best guess",                 # phrasing of ask_user
    "raw column value",                        # raw vs normalised grouping
    "ROUND",                                   # output precision rule
    "complete original question",              # search-with-full-question
    "what was submitted",                      # MUST submit
    "computed value",                          # filter syntax / ModelExtension
]


def test_slayer_a_interact_anchored_phrases(all_three_concatenated):
    """Beyond the bag-of-tokens coverage matrix, pin specific verbatim
    SLAYER_A_INTERACT phrases that carry directive meaning. A regression
    here means the prompt was rewritten enough to drop the original rule's
    teeth — not just its vocabulary."""
    lower = all_three_concatenated.lower()
    missing = [p for p in ANCHORED_PHRASES if p.lower() not in lower]
    assert not missing, (
        f"Load-bearing SLAYER_A_INTERACT phrases missing from new prompts: "
        f"{missing}"
    )


# ---------------------------------------------------------------------------
# 23: constructor prompt has the B-extra mitigation language
# ---------------------------------------------------------------------------


def test_constructor_prompt_has_b_extra_mitigation(prompts):
    """The hardened count-check + per-column ask_user + banned anti-
    patterns + AUTHORITATIVE PROJECTION RULE must all be present in the
    constructor prompt. These are the load-bearing fixes for the
    over-projection failure mode.

    Note: per DEV-1432, the projection-scope cue interpretation
    ("just / only / no / without") moves to the PROJECTION_RESOLVER_PROMPT
    (Stage 2). The constructor's job becomes simpler — reproduce the
    Stage 2 confirmed list exactly. We no longer require those words
    in this prompt; they live in Stage 2 now (asserted separately in
    `test_override_rule_fixtures.py`)."""
    body = prompts.QUERY_CONSTRUCTOR_PROMPT
    lower = body.lower()
    must_contain = [
        "count",                  # count check
        "ask_user",               # explicit per-column ask
        "extra",                  # extras language
        "concatenat",             # concatenation split
        "banned anti-pattern",    # the banned section
        "do not submit on your own judgment",
    ]
    missing = [s for s in must_contain if s.lower() not in lower]
    assert not missing, (
        f"constructor prompt missing B-extra mitigation phrases: {missing}"
    )


# ---------------------------------------------------------------------------
# 24: Step A table shape + ranking/filtering rule
# ---------------------------------------------------------------------------


def test_constructor_step_a_defines_5_column_table(prompts):
    body = prompts.QUERY_CONSTRUCTOR_PROMPT
    lower = body.lower()
    # The five column labels of the projection-decision table.
    for label in (
        "verbatim phrase",
        "source",
        "output?",
        "projection slot",
        "forbidden extras",
    ):
        assert label.lower() in lower, (
            f"Step A table missing column label: {label!r}"
        )
    # The ranking/filtering/grouping → output? no rule.
    assert "ranking" in lower
    assert "filtering" in lower
    # Per DEV-1432, the old `OVERRIDE RULE` is replaced by the
    # AUTHORITATIVE PROJECTION RULE — Stage 2's list is the source of
    # truth, the constructor reproduces it exactly. The new directive
    # uses "authoritative" not "override".
    assert "authoritative" in lower or "stage 2" in lower


# ---------------------------------------------------------------------------
# 25: constructor must have explicit `help` first-step mandate
# ---------------------------------------------------------------------------


def test_constructor_prompt_mandates_help_first(prompts):
    body = prompts.QUERY_CONSTRUCTOR_PROMPT
    lower = body.lower()
    # `help` shows up somewhere AND is described as a mandatory first step.
    assert "help" in lower
    # Anchor: must say "first" near help OR have a "Step 0" or "first step".
    assert ("step 0" in lower) or ("call `help` first" in lower) or (
        "first step" in lower and "help" in lower
    )


# ---------------------------------------------------------------------------
# 26: root prompt has compound-naming default
# ---------------------------------------------------------------------------


def test_root_prompt_has_compound_naming_default(prompts):
    body = prompts.ROOT_CLARIFIER_PROMPT
    lower = body.lower()
    # The "compound-naming" rule itself.
    assert "compound" in lower or "and / both" in lower or "and/both" in lower
    # The default: two columns per "X and Y", not one concatenation.
    assert "two" in lower and ("column" in lower or "slot" in lower)
    # Anti-pattern callout for concatenation.
    assert "concatenat" in lower


# ---------------------------------------------------------------------------
# Bonus shape checks — non-spec but cheap and guard against trivial regressions
# ---------------------------------------------------------------------------


def test_root_prompt_has_no_ask_user_or_submit_directives(prompts):
    """The root agent has no ask_user / submit_query tools — its prompt
    must not instruct the model to call them, or the model will burn a
    turn on a tool-not-found retry."""
    body = prompts.ROOT_CLARIFIER_PROMPT
    # The prompt should TELL the agent it has no ask_user.
    assert "no `ask_user`" in body.lower() or "no ask_user" in body.lower()
    # And nothing in the root prompt should instruct submission.
    assert "submit_query" not in body or "cannot submit" in body.lower()


def test_sub_clarifier_prompt_forbids_full_query(prompts):
    """Sub-clarifier owns ONE logical unit — it must not draft the
    complete query or fill the source_model. That's the constructor's
    job."""
    body = prompts.SUB_CLARIFIER_PROMPT
    lower = body.lower()
    # The "you cannot submit and cannot call query" rule.
    assert "cannot submit" in lower
    assert "cannot call `query`" in lower or "cannot call query" in lower
    # The "do not produce a complete query / source_model" rule.
    assert "complete query" in lower or "source_model" in lower


def test_all_three_prompts_include_budget_reminder(prompts):
    """Every agent in the tree shares one budget pool — every prompt
    needs at least a one-line bird-coin reminder so the model doesn't
    burn the constructor's reservation."""
    for name, body in (
        ("ROOT_CLARIFIER_PROMPT", prompts.ROOT_CLARIFIER_PROMPT),
        ("SUB_CLARIFIER_PROMPT", prompts.SUB_CLARIFIER_PROMPT),
        ("QUERY_CONSTRUCTOR_PROMPT", prompts.QUERY_CONSTRUCTOR_PROMPT),
    ):
        assert "bird-coin" in body.lower() or "budget" in body.lower(), (
            f"{name} has no budget reminder"
        )


def test_prompts_use_no_specific_task_ids(prompts):
    """The recursive-adapter prompts must stay GENERIC — no task IDs
    bleed through (per the explicit 'No prompt-side mentions of specific
    task IDs' rule in the spec)."""
    import re

    forbidden_pattern = re.compile(
        r"\b("
        r"planets_data_\d+|cross_db_\d+|alien_\d+|households_\d+|"
        r"hulushows_\d+|crypto_\d+|news_\d+|robot_\d+|polar_\d+|"
        r"vaccine_\d+|gaming_\d+|mental_\d+|fake_\d+|disaster_\d+|"
        r"solar_\d+|insider_\d+|museum_\d+|virtual_\d+|exchange_traded_funds_\d+|"
        r"reverse_logistics_\d+|organ_transplant_\d+|sports_events_\d+|"
        r"cybermarket_\d+|cold_chain_pharma_compliance_\d+|"
        r"archeology_\d+|credit_\d+|labor_certification_applications_\d+"
        r")\b"
    )
    for name, body in (
        ("ROOT_CLARIFIER_PROMPT", prompts.ROOT_CLARIFIER_PROMPT),
        ("SUB_CLARIFIER_PROMPT", prompts.SUB_CLARIFIER_PROMPT),
        ("QUERY_CONSTRUCTOR_PROMPT", prompts.QUERY_CONSTRUCTOR_PROMPT),
        ("PROJECTION_RESOLVER_PROMPT", prompts.PROJECTION_RESOLVER_PROMPT),
    ):
        match = forbidden_pattern.search(body)
        assert match is None, (
            f"{name} leaks a specific task ID: {match.group(0)!r}"
        )


# ---------------------------------------------------------------------------
# DEV-1432: sub-clarifier table-family disambiguation hook
# ---------------------------------------------------------------------------


def test_sub_clarifier_has_table_family_disambiguation(prompts):
    """The sub-clarifier must check for table-family ambiguity AFTER
    its initial `search` — search surfaces candidate tables + columns
    so the disambiguation question can be concrete (table_a.col_x vs
    table_b.col_y), not abstract."""
    body = prompts.SUB_CLARIFIER_PROMPT
    lower = body.lower()
    # Must mention table-family disambiguation.
    assert "table" in lower
    # Must call out the multi-table-match scenario.
    assert "multiple tables" in lower or "more than one table" in lower or (
        "two tables" in lower
    ), (
        "SUB_CLARIFIER_PROMPT must teach the agent to disambiguate "
        "between candidate tables when a noun in the focus could "
        "match more than one."
    )
    # Must say "table" not "model" in user-sim-facing context. The
    # word "model" still legitimately appears as a tool name
    # (`inspect_model`, `models_summary`), but the user-facing
    # phrasing should use "table". Anchor: at least one prose use of
    # "tables" near `ask_user`.
    assert "ask the user" in lower or "ask_user" in lower
    # The ordering: search FIRST, then table-family check.
    search_idx = lower.find("search")
    table_check_idx = max(
        lower.find("multiple tables"),
        lower.find("two tables"),
        lower.find("more than one table"),
    )
    assert search_idx >= 0 and table_check_idx > search_idx, (
        "Table-family check must come AFTER `search` in the sub-"
        "clarifier flow — search results give the LLM the candidate "
        "tables to choose between."
    )


def test_sub_clarifier_table_disambiguation_uses_table_not_model(prompts):
    """User-facing phrasing for the table-family question must use
    'table' (the SQL term the user-sim recognises), not 'model' (the
    SLayer-layer abstraction). The disambiguation prompt's example
    must show table-style references like `tableA.col_x` rather than
    `modelA.col_x`."""
    body = prompts.SUB_CLARIFIER_PROMPT
    lower = body.lower()
    # Find the section after the "multiple tables" / "two tables"
    # mention and confirm it speaks in `table.column` terms, not
    # `model.column` terms.
    anchor_phrases = ["multiple tables", "two tables", "more than one table"]
    anchor_idx = -1
    for phrase in anchor_phrases:
        idx = lower.find(phrase)
        if idx >= 0:
            anchor_idx = idx
            break
    assert anchor_idx >= 0, (
        "Sub-clarifier prompt missing table-family anchor phrase."
    )
    # 800 chars is generous — covers the disambiguation example.
    window = lower[anchor_idx : anchor_idx + 800]
    # Look for ANY dotted column reference like `name.column` — this
    # is the SQL `table.col` style the prompt should teach. A name
    # before the dot + a name after (one or more word chars).
    import re

    dotted_refs = re.findall(r"[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*", window)
    assert dotted_refs, (
        f"The disambiguation example must show at least one dotted "
        f"reference (`<name>.<col>`) so the agent learns to phrase "
        f"the disambiguation as `tableA.col_x vs tableB.col_y`. "
        f"Window (first 200 chars): {window[:200]!r}"
    )
    # And the prompt must use "table" rather than "model" in the
    # user-facing wording (the dotted refs above + the noun "table"
    # in the immediate context).
    assert "table" in window
