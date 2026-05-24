"""Behaviour tests for the encoder prompt template.

Per `feedback_no_prompt_content_tests.md` — NO substring / anchor-phrase /
coverage assertions on the prompt's natural-language content. Tests here
only verify mechanical behaviour:

* All `{...}` Python format placeholders fill cleanly (no leftover
  unsubstituted braces).
* The KB row's YAML body appears in the formatted prompt (so the
  encoder sees what it's encoding).
* The deps map appears when dependencies are supplied (so R-RESOLVE
  references resolve correctly).
* No `{...}` placeholder is missing from the template signature
  expected by the call site.

These are mechanical contracts, not phrasing contracts.
"""

from __future__ import annotations

import re

import yaml


def _format_args():
    """The minimum arg set the KB_ENCODER_PROMPT must accept. Matches
    what `_run_kb_encoder` passes."""
    return dict(
        db_name="tinydb",
        kb_id=5,
        kb_row_yaml=yaml.safe_dump(
            {"id": 5, "knowledge": "test", "definition": "x"},
            sort_keys=False,
        ),
        deps_block="(none)",
        budget=100.0,
    )


def test_template_fills_with_no_leftover_braces():
    """For each KEY supplied to `.format(**args)`, the rendered prompt
    must NOT contain `{<key>}` (a sign that key was never inserted).
    Tests format-arg coverage, not arbitrary `{x}` literals in the
    prompt content (which are legal — e.g., `{value}` in R-AGG
    formula templates)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
        KB_ENCODER_PROMPT,
    )

    args = _format_args()
    out = KB_ENCODER_PROMPT.format(**args)
    for key in args:
        unsubstituted = "{" + key + "}"
        assert unsubstituted not in out, (
            f"Unsubstituted placeholder `{unsubstituted}` in rendered "
            f"encoder prompt — key was never substituted."
        )


def test_template_includes_kb_row_yaml_when_supplied():
    """The encoder needs to see the row it's encoding."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
        KB_ENCODER_PROMPT,
    )

    args = _format_args()
    out = KB_ENCODER_PROMPT.format(**args)
    # The YAML body must literally appear in the output — the agent
    # reads `definition` and `knowledge` out of it.
    yaml_str: str = args["kb_row_yaml"]  # type: ignore[assignment]
    assert yaml_str.strip() in out


def test_template_includes_deps_block_when_supplied():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
        KB_ENCODER_PROMPT,
    )

    args = _format_args()
    args["deps_block"] = (
        "  - KB 3 → tinydb.households.tier_score (kind=column)\n"
        "  - KB 4 → tinydb.households.assistance_score (kind=column)"
    )
    out = KB_ENCODER_PROMPT.format(**args)
    assert "tier_score" in out
    assert "assistance_score" in out


def test_template_includes_db_name_for_data_source_scoping():
    """Every MCP write call needs `data_source=<db>`; the prompt's
    intro must surface the db_name so the encoder can pin it."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
        KB_ENCODER_PROMPT,
    )

    args = _format_args()
    out = KB_ENCODER_PROMPT.format(**args)
    assert "tinydb" in out


def test_template_includes_kb_id_for_self_annotation():
    """The KB id is the `meta.kb_id` the encoder stamps on every entity;
    the prompt must show the id so the encoder can copy it."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
        KB_ENCODER_PROMPT,
    )

    args = _format_args()
    out = KB_ENCODER_PROMPT.format(**args)
    assert "5" in out


def test_sub_clarifier_prompt_template_fills_with_no_leftover_braces():
    """Same mechanical check on SUB_CLARIFIER_PROMPT — every supplied
    arg must be substituted at its placeholder. Doesn't fail on
    arbitrary `{x}` literals in prose."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
        SUB_CLARIFIER_PROMPT,
    )

    args = dict(
        budget=100.0, db_name="tinydb",
        focus="x", instruction="figure it out",
    )
    out = SUB_CLARIFIER_PROMPT.format(**args)
    for key in args:
        unsubstituted = "{" + key + "}"
        assert unsubstituted not in out


# ---------------------------------------------------------------------------
# SETUP_ENCODER_PROMPT mechanical contract (DEV-1466). Same policy: mechanical
# only — no assertions on natural-language phrasing. The setup encoder gains a
# `{reverse_deps_block}` placeholder so it can see the KBs that reference the
# current KB (its parents).
# ---------------------------------------------------------------------------


def _setup_format_args():
    """The arg set SETUP_ENCODER_PROMPT must accept — matches the call site in
    setup_encoder._run_setup_encoder."""
    return dict(
        db_name="tinydb",
        kb_id=6,
        kb_body="KB 6 — Dwelling Type\n\nKB item (verbatim):\nid: 6",
        deps_block="(none)",
        reverse_deps_block="(none)",
    )


def test_setup_prompt_fills_with_no_leftover_braces():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
        SETUP_ENCODER_PROMPT,
    )

    args = _setup_format_args()
    out = SETUP_ENCODER_PROMPT.format(**args)
    for key in args:
        unsubstituted = "{" + key + "}"
        assert unsubstituted not in out, (
            f"Unsubstituted placeholder `{unsubstituted}` in rendered "
            f"setup-encoder prompt — key was never substituted."
        )


def test_setup_prompt_declares_reverse_deps_block_field():
    """The placeholder must actually exist in the template (not merely be an
    ignored extra kwarg). `str.format` silently drops unknown kwargs, so assert
    on the parsed field set instead."""
    import string

    from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
        SETUP_ENCODER_PROMPT,
    )

    fields = {
        fname
        for _, fname, _, _ in string.Formatter().parse(SETUP_ENCODER_PROMPT)
        if fname
    }
    assert "reverse_deps_block" in fields


def test_setup_prompt_includes_reverse_deps_block_when_supplied():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
        SETUP_ENCODER_PROMPT,
    )

    args = _setup_format_args()
    args["reverse_deps_block"] = (
        '  - KB 44 (calculation_knowledge) "Dwelling Type Score": 4/3/1 scoring'
    )
    out = SETUP_ENCODER_PROMPT.format(**args)
    assert "Dwelling Type Score" in out
    assert "KB 44" in out


def test_setup_prompt_includes_deps_block_when_supplied():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.prompts import (
        SETUP_ENCODER_PROMPT,
    )

    args = _setup_format_args()
    args["deps_block"] = (
        "  - KB 3 -> tinydb.infrastructure.water_access_score (kind=column)"
    )
    out = SETUP_ENCODER_PROMPT.format(**args)
    assert "water_access_score" in out
