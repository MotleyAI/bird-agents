"""DEV-1545: mechanical contract tests for the two new agent-side
residual prompt blocks targeting `wrong_join_path` and
`never_asked_key_question`.

Per `feedback_no_prompt_content_tests.md` — no phrase / anchor /
substring assertions on the prompt's natural-language content. Only:

  * Block constants exist and are non-empty strings.
  * They render with the `{knowledge_label}` format param without
    leftover braces.
  * They are STITCHED into the SLAYER_OTF_AINTERACT (interactive) and
    SLAYER_OTF_ONE_SHOT (one-shot) prompts so the agent actually sees
    them at runtime.

The `test_shared_otf_prompts.py` SHA-256 snapshot tests will need to be
re-baselined as part of the implementation step (the prompts deliberately
change). That re-baselining is a separate change in that file — not
this file.
"""

from __future__ import annotations

import re


def test_table_set_probe_block_exists_and_nonempty() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import _TABLE_SET_PROBE

    assert isinstance(_TABLE_SET_PROBE, str)
    assert _TABLE_SET_PROBE.strip(), "_TABLE_SET_PROBE must be non-empty"


def test_grader_zero_vs_one_diagnostic_block_exists_and_nonempty() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import (
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC,
    )

    assert isinstance(_GRADER_ZERO_VS_ONE_DIAGNOSTIC, str)
    assert _GRADER_ZERO_VS_ONE_DIAGNOSTIC.strip()


def test_table_set_probe_renders_with_knowledge_label_kb() -> None:
    """Mechanical: the block uses `{knowledge_label}` (matches the
    existing `_USER_SIM_TRUST_CALIBRATION` shape) and substitutes
    cleanly for slayer-mode 'KB'."""
    from bird_interact_agents.agents._shared_otf_prompts import _TABLE_SET_PROBE

    # DEV-1603: the probe is now also parameterised by `{schema_source}` so
    # raw mode can fill it without leaking "SLayer"; the slayer fill below
    # reproduces today's rendered text.
    rendered = _TABLE_SET_PROBE.format(
        knowledge_label="KB", schema_source="SLayer's schema lookup"
    )
    assert "{" not in re.sub(r"\{[^a-zA-Z_]", "", rendered), (
        f"unsubstituted `{{...}}` after .format(): {rendered!r}"
    )


def test_grader_zero_vs_one_diagnostic_renders_with_knowledge_label_kb() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import (
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC,
    )

    # DEV-1603: parameterised by `{attempt_noun}`/`{apply_verb}` for raw mode;
    # slayer fill reproduces today's text.
    rendered = _GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(
        attempt_noun="encoding", apply_verb="encode"
    )
    assert "{" not in re.sub(r"\{[^a-zA-Z_]", "", rendered)


def test_blocks_stitched_into_slayer_ainteract() -> None:
    """Both blocks must appear in the rendered a-interact agent system
    prompt. Mechanical-only — we assert the constant string is a
    substring of the final prompt; we don't lock natural-language
    phrases."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC,
        _TABLE_SET_PROBE,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    rendered_table_set = _TABLE_SET_PROBE.format(
        knowledge_label="KB", schema_source="SLayer's schema lookup"
    )
    rendered_diag = _GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(
        attempt_noun="encoding", apply_verb="encode"
    )
    assert rendered_table_set in SLAYER_OTF_AINTERACT, (
        "_TABLE_SET_PROBE not stitched into SLAYER_OTF_AINTERACT"
    )
    assert rendered_diag in SLAYER_OTF_AINTERACT, (
        "_GRADER_ZERO_VS_ONE_DIAGNOSTIC not stitched into "
        "SLAYER_OTF_AINTERACT"
    )


def test_blocks_stitched_into_slayer_one_shot() -> None:
    """One-shot has no user-sim, so the `_GRADER_ZERO_VS_ONE_DIAGNOSTIC`
    block — which prescribes a `ask_user` step — must NOT appear in the
    one-shot prompt (its trigger is unreachable). `_TABLE_SET_PROBE`
    applies to one-shot too (the structural-pivot half does not need a
    user-sim) and SHOULD appear."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC,
        _TABLE_SET_PROBE,
    )
    from bird_interact_agents.agents.claude_sdk_otf.prompts import (
        SLAYER_OTF_ONE_SHOT,
    )

    rendered_table_set = _TABLE_SET_PROBE.format(
        knowledge_label="KB", schema_source="SLayer's schema lookup"
    )
    rendered_diag = _GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(
        attempt_noun="encoding", apply_verb="encode"
    )
    assert rendered_table_set in SLAYER_OTF_ONE_SHOT, (
        "_TABLE_SET_PROBE not stitched into SLAYER_OTF_ONE_SHOT"
    )
    assert rendered_diag not in SLAYER_OTF_ONE_SHOT, (
        "_GRADER_ZERO_VS_ONE_DIAGNOSTIC must NOT appear in one-shot "
        "(no user-sim to ask)."
    )
