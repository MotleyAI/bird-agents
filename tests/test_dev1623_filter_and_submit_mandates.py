"""DEV-1623 — mechanical contracts for the new filter-literal + query-before-
submit prompt fragments.

The fix (cut submit-verify thrash on noisy categorical columns) is prompt-only.
Per ``feedback_no_prompt_content_tests`` these tests assert ONLY structural /
format contracts — fragment existence, format-placeholder coverage, and clean
``.format()`` substitution. They deliberately do NOT assert anchor phrases,
substrings, or a placement coverage-matrix; the behavioural validation is the
``households_4`` / ``households_17`` cloud re-run documented on the issue.

What each fix adds (for reference — NOT asserted here):
* ``_SAMPLE_VALUE_FILTER_MANDATE`` — Fix 1, a ``{sample_source}``-parameterised
  block spliced into all eight OTF prompts (slayer + raw, v0 literals + v1
  compositions) after the existing "Sample values" paragraph.
* ``_QUERY_BEFORE_SUBMIT`` — Fix 2, spliced into the two v0 slayer literals
  (v1 already carries an equivalent verify-before-submit checklist).
* the ``ask_discovery`` reporting nudge in ``claude_sdk.partition`` — Fix 1b,
  v1-only.
"""

from __future__ import annotations

import importlib
from string import Formatter

import pytest


def _format_fields(template: str) -> set[str]:
    """The set of replacement-field names in a ``str.format`` template."""
    return {
        field for _, field, _, _ in Formatter().parse(template) if field is not None
    }


# ---------------------------------------------------------------------------
# Fix 1 — _SAMPLE_VALUE_FILTER_MANDATE (shared, parameterised by sample_source)
# ---------------------------------------------------------------------------


def test_sample_value_filter_mandate_exists_and_nonempty():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _SAMPLE_VALUE_FILTER_MANDATE,
    )

    assert isinstance(_SAMPLE_VALUE_FILTER_MANDATE, str)
    assert _SAMPLE_VALUE_FILTER_MANDATE.strip()


def test_sample_value_filter_mandate_only_sample_source_field():
    """The fragment must expose exactly the ``{sample_source}`` field — no
    stray ``{budget}`` / ``{db_name}`` / unescaped literal brace that would
    later break the host prompt's ``.format(...)``."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        _SAMPLE_VALUE_FILTER_MANDATE,
    )

    assert _format_fields(_SAMPLE_VALUE_FILTER_MANDATE) == {"sample_source"}


def test_sample_value_filter_mandate_formats_clean():
    """Once ``sample_source`` is supplied, no residual braces remain, so the
    fragment can be concatenated into a host prompt that is itself
    ``.format(...)``-ed downstream."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        _SAMPLE_VALUE_FILTER_MANDATE,
    )

    rendered = _SAMPLE_VALUE_FILTER_MANDATE.format(sample_source="`get_column_meaning`")
    assert "{" not in rendered
    assert "}" not in rendered


def test_sample_value_filter_mandate_raw_render_is_slayer_free():
    """When rendered for the raw agents (``get_column_meaning``), the shared
    fragment must carry NO SLayer-specific vocabulary — mirrors the raw-vocab
    contract in ``test_shared_otf_prompts.py`` so a shared fragment cannot leak
    slayer terms into the raw prompts."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        _SAMPLE_VALUE_FILTER_MANDATE,
    )

    rendered = _SAMPLE_VALUE_FILTER_MANDATE.format(sample_source="`get_column_meaning`")
    for term in (
        "submit_query",
        "create_model",
        "edit_model",
        "[kb=",
        "memory:",
        "slayer",
        "SLayer",
        "mcp__slayer__",
        "inspect_model",
        "ask_discovery",
    ):
        assert term not in rendered, f"raw-rendered mandate leaked {term!r}"


# ---------------------------------------------------------------------------
# Fix 2 — _QUERY_BEFORE_SUBMIT (v0 slayer literals only)
# ---------------------------------------------------------------------------


def test_query_before_submit_exists_and_nonempty():
    from bird_interact_agents.agents._shared_otf_prompts import _QUERY_BEFORE_SUBMIT

    assert isinstance(_QUERY_BEFORE_SUBMIT, str)
    assert _QUERY_BEFORE_SUBMIT.strip()


def test_query_before_submit_has_no_format_fields():
    """The fragment is spliced verbatim into a ``.format(...)``-ed host prompt,
    so it must contain no replacement fields and no unescaped literal brace."""
    from bird_interact_agents.agents._shared_otf_prompts import _QUERY_BEFORE_SUBMIT

    assert _format_fields(_QUERY_BEFORE_SUBMIT) == set()
    assert "{" not in _QUERY_BEFORE_SUBMIT
    assert "}" not in _QUERY_BEFORE_SUBMIT


# ---------------------------------------------------------------------------
# Regression guard — every affected OTF prompt still ``.format(...)`` cleanly
# after the splices (an unescaped brace would raise here). Covers the four v0
# literals and the four v1 re-exports.
# ---------------------------------------------------------------------------

_OTF_CONSTANTS = [
    ("_shared_otf_prompts", "SLAYER_OTF_ONE_SHOT_V0"),
    ("_shared_otf_prompts", "SLAYER_OTF_AINTERACT_V0"),
    ("_shared_otf_prompts", "RAW_OTF_ONE_SHOT_V0"),
    ("_shared_otf_prompts", "RAW_OTF_AINTERACT_V0"),
    ("claude_sdk_otf_v1.prompts", "SLAYER_OTF_ONE_SHOT"),
    ("claude_sdk_otf_ainteract_v1.prompts", "SLAYER_OTF_AINTERACT"),
    ("claude_sdk_otf_raw_v1.prompts", "RAW_OTF_ONE_SHOT"),
    ("claude_sdk_otf_ainteract_raw_v1.prompts", "RAW_OTF_AINTERACT"),
]


@pytest.mark.parametrize("module_suffix,const_name", _OTF_CONSTANTS)
def test_otf_constant_formats_without_error(module_suffix: str, const_name: str):
    module = importlib.import_module(
        f"bird_interact_agents.agents.{module_suffix}"
    )
    const = getattr(module, const_name)
    assert isinstance(const, str) and const.strip()
    # Exact template API: the three documented params and nothing else. A
    # splice that introduced a stray single brace (an un-doubled ``{``) would
    # add a spurious field here; one that dropped ``{user_query}`` / ``{db_name}``
    # would remove a required field. ``.format`` ignores extra kwargs, so the
    # call below alone would NOT catch a dropped placeholder — this set check does.
    assert _format_fields(const) == {"budget", "db_name", "user_query"}, (
        f"{const_name} placeholder set drifted: {_format_fields(const)}"
    )
    const.format(budget=20.0, db_name="shop", user_query="how many items?")


# ---------------------------------------------------------------------------
# Fix 1b — the discovery sub-agent prompt still composes after the nudge.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_ask_user", [True, False])
def test_discovery_prompt_still_builds(with_ask_user: bool):
    from bird_interact_agents.agents.claude_sdk.partition import build_discovery_prompt

    # DEV-1591 ∩ DEV-1623 merge: build_discovery_prompt now requires the
    # `query_mode` kwarg (the compact-search / sample-value discipline is
    # slayer-only), so pin the slayer mode where the DEV-1623 nudge applies.
    prompt = build_discovery_prompt(with_ask_user=with_ask_user, query_mode="slayer")
    assert isinstance(prompt, str) and prompt.strip()
