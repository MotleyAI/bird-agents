"""Fixture-style tests for the simplified projection override (DEV-1432).

With Stage 2 (projection-resolver) in the architecture, the constructor's
prompt no longer needs to scan `amb_user_query` for "just / only / no /
without" overrides — Stage 2 already does that interactively with the
user-sim. The constructor's job is simpler: REPRODUCE the confirmed
list, no extras, no aliases, no equivalents.

These tests pin the simplification:

* The constructor's old multi-line OVERRIDE RULE block is gone.
* A short AUTHORITATIVE PROJECTION RULE says Stage 2's list is the
  source of truth.
* Step D forbids projecting anything outside the confirmed list,
  including aliases / duplicates that restate a listed column under
  a second name.
* The Stage 2 prompt teaches the "just/only/no/without" interpretation
  with explicit filter/ranking/quality non-cases.
* Specific failure-mode cases (`labor_certification_applications_2`
  duplicate-projection, `hulushows_15` amb_user_query "just count")
  are addressed by directives that exist in the right place.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def prompts():
    from bird_interact_agents.agents.pydantic_ai_recursive import prompts as p

    return p


# ---------------------------------------------------------------------------
# Constructor: old multi-line OVERRIDE RULE must be REMOVED.
# Replaced by a short AUTHORITATIVE PROJECTION RULE.
# ---------------------------------------------------------------------------


def test_constructor_no_longer_has_multiline_override_rule(prompts):
    """The old `OVERRIDE RULE:` block that scanned user-sim replies for
    just/only/no/without IS the chicken-and-egg liability — the new
    architecture moves that work to Stage 2. The constructor prompt
    must NOT carry the old block."""
    body = prompts.QUERY_CONSTRUCTOR_PROMPT
    # The literal old-rule header.
    assert "OVERRIDE RULE:" not in body, (
        "QUERY_CONSTRUCTOR_PROMPT still carries the old multi-line "
        "OVERRIDE RULE block; that block was replaced by the new "
        "AUTHORITATIVE PROJECTION RULE per DEV-1432."
    )


def test_constructor_has_authoritative_projection_rule(prompts):
    """The replacement rule must be present, short, and tell the
    constructor the confirmed list is the source of truth. Anchor on
    the literal string used in the spec."""
    body = prompts.QUERY_CONSTRUCTOR_PROMPT
    assert "AUTHORITATIVE PROJECTION RULE" in body, (
        "QUERY_CONSTRUCTOR_PROMPT is missing the AUTHORITATIVE "
        "PROJECTION RULE that replaces the old override rule."
    )
    lower = body.lower()
    # The rule must reference Stage 2 / confirmed-projection / source-of-truth.
    assert "confirmed projection" in lower or "stage 2" in lower
    assert "source of truth" in lower or "authoritative" in lower
    # And it must say the constructor's job is to REPRODUCE the list,
    # not invent new columns.
    assert "reproduce" in lower or "exactly" in lower


def test_constructor_step_d_forbids_aliases_and_duplicates(prompts):
    """Step D (banned anti-patterns) must include an explicit anti-
    duplicate / anti-alias rule. This is what catches `labor_certi-
    fication_applications_2`: the agent projected both `is_certified_avg`
    AND `success_rate` — two names for the same metric. The closure-
    bound count check rejects on count; this prompt rule explains
    WHY the second projection is wrong (it's a duplicate)."""
    body = prompts.QUERY_CONSTRUCTOR_PROMPT
    lower = body.lower()
    # Step D bullet anchors — must talk about duplicates / aliases /
    # restating the same column under a second name.
    duplicates_signals = [
        "duplicate",
        "alias",
        "equivalent",
        "second name",
        "another name",
        "restate",
    ]
    assert any(s in lower for s in duplicates_signals), (
        f"Step D must forbid projecting equivalent columns under "
        f"different names. None of {duplicates_signals} found in the "
        f"constructor prompt."
    )


def test_constructor_prompt_references_confirmed_projection_template_var(prompts):
    """The constructor receives the confirmed projection list as a
    template variable. The prompt template must reference it so the
    list lands somewhere the LLM sees it. We don't pin the variable
    NAME (could be `confirmed_projection`, `projection`, etc.) but we
    DO require the prompt template carry a `{confirmed_projection}`
    placeholder so agent.py can format it in."""
    body = prompts.QUERY_CONSTRUCTOR_PROMPT
    # The template should have a placeholder that agent.py will fill.
    # We check for the canonical name used in the spec.
    assert "{confirmed_projection}" in body, (
        "QUERY_CONSTRUCTOR_PROMPT must contain {confirmed_projection} "
        "placeholder so agent.py can inject Stage 2's output. The "
        "spec uses that exact name."
    )


# ---------------------------------------------------------------------------
# Stage 2 (projection-resolver): owns "just/only/no/without" scoping.
# ---------------------------------------------------------------------------


def test_stage_2_prompt_teaches_just_only_no_without(prompts):
    """The projection-resolver prompt must teach the LLM that "just",
    "only", "no", and "without" can restrict the OUTPUT projection.
    Without this, the resolver may propose a list that includes
    grouping nouns / ranking columns the user didn't actually ask
    for (the `hulushows_15` failure mode)."""
    body = prompts.PROJECTION_RESOLVER_PROMPT
    lower = body.lower()
    for word in ("just", "only", "no", "without"):
        assert word in lower, (
            f"PROJECTION_RESOLVER_PROMPT missing trigger word {word!r}; "
            f"the resolver must know these are projection-scope cues."
        )


def test_stage_2_prompt_distinguishes_projection_from_filter_ranking_quality(prompts):
    """The same trigger words appear in non-projection contexts
    ("only applications from 2020" = filter; "without missing values"
    = data quality; "just top 5" = ranking). The resolver prompt must
    explicitly call out these NON-projection cases so it doesn't
    over-fire."""
    body = prompts.PROJECTION_RESOLVER_PROMPT
    lower = body.lower()
    # At least two of the three non-projection categories must be named.
    non_projection_signals = ["filter", "ranking", "quality", "threshold"]
    hits = [s for s in non_projection_signals if s in lower]
    assert len(hits) >= 2, (
        f"PROJECTION_RESOLVER_PROMPT must distinguish projection-scope "
        f"cues from filter/ranking/quality/threshold non-cases. "
        f"Found only: {hits}."
    )


def test_stage_2_prompt_mentions_ask_user_for_ambiguity(prompts):
    """When the resolver can't decide whether a trigger phrase is
    projection-scoped or not, the prompt must direct it to ask_user
    explicitly. This is the safety valve that prevents bad guesses."""
    body = prompts.PROJECTION_RESOLVER_PROMPT
    lower = body.lower()
    assert "ask_user" in lower or "ask the user" in lower
    # Plus an "if unclear / ambiguous" hedge.
    assert any(s in lower for s in (
        "unclear", "ambiguous", "if you can't", "if you cannot", "if in doubt",
    )), (
        "PROJECTION_RESOLVER_PROMPT must instruct the agent to ask_user "
        "when projection scope is unclear."
    )


# ---------------------------------------------------------------------------
# Failure-mode coverage — anchor specific FMA cases to the right directives.
# ---------------------------------------------------------------------------


def test_labor_certification_applications_2_failure_mode_addressed(prompts):
    """B-EXTRA: agent projected `is_certified_avg` AND `success_rate` —
    same metric under two names. The constructor's Step D anti-pattern
    must explicitly forbid this. Combined with the closure-bound count
    check, this case becomes structurally impossible to submit."""
    body = prompts.QUERY_CONSTRUCTOR_PROMPT
    lower = body.lower()
    # Step D must call out the "restate / equivalent / alias" anti-pattern
    # for the SAME measure under a second name — that's the specific
    # B-EXTRA the count check + prompt jointly prevent.
    same_metric_signals = [
        ("equivalent", "name"),
        ("equivalent measure",),
        ("under another name",),
        ("under a second name",),
        ("restate", "name"),
        ("alias", "listed"),
    ]
    matched = any(
        all(s in lower for s in group) for group in same_metric_signals
    )
    assert matched, (
        "Step D must explicitly forbid restating a listed column under "
        "a different name (equivalent measure / alias). This is the "
        "labor_certification_applications_2 failure mode."
    )


def test_hulushows_15_failure_mode_addressed_in_stage_2(prompts):
    """`hulushows_15`'s `amb_user_query` says "just count how many shows".
    The resolver must read the original question and treat "just count"
    as a projection-scope restriction. Anchor on the resolver prompt
    naming both `amb_user_query` (or equivalent) AND the projection-
    scope words."""
    body = prompts.PROJECTION_RESOLVER_PROMPT
    lower = body.lower()
    # The resolver MUST read the original user query.
    assert any(s in lower for s in (
        "{amb_user_query}",
        "amb_user_query",
        "original question",
        "user question",
        "user's question",
        "original user query",
    )), (
        "PROJECTION_RESOLVER_PROMPT must reference the original user "
        "query so it can detect projection-scope language IN the "
        "question itself (hulushows_15)."
    )
    # And it must teach that "just <agg>" / "just X" is projection-scoping.
    assert "just" in lower
