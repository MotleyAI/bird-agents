"""DEV-1672 Cause 1: the stern "don't re-derive an already-encoded quantity"
rule must appear ONLY in the readonly branch of the v0 OTF slayer prompts
(``--apply-edited-models --readonly-mode``), for BOTH one-shot and a-interact.

Per the project's no-prompt-content-tests rule these are WIRING/composition
assertions only — they reference the shared ``_NO_REDERIVE_READONLY`` constant
OBJECT and the readonly gate, never a hand-typed phrase or coverage matrix
(mirrors the existing DEV-1666 readonly-gating tests).

Scope guards: the block must be ABSENT from the non-readonly templates, from
the pre-encoded prompts, and from the v1 OTF prompts (v1 readonly wiring is a
deferred follow-up).
"""

from __future__ import annotations

import pytest

from bird_interact_agents.agents._shared_otf_prompts import (
    SLAYER_OTF_AINTERACT_V0,
    SLAYER_OTF_ONE_SHOT_V0,
    _NO_REDERIVE_READONLY,
    build_slayer_otf_ainteract_v0,
    build_slayer_otf_one_shot_v0,
)
from bird_interact_agents.agents._pre_encoded_prompts import (
    SLAYER_PRE_ENCODED_AINTERACT,
    SLAYER_PRE_ENCODED_ONE_SHOT,
    build_slayer_pre_encoded_one_shot,
)

_FMT = dict(budget=100, db_name="mydb", user_query="how many?")


# ---------------------------------------------------------------------------
# Present in the readonly branch of BOTH v0 builders.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [build_slayer_otf_one_shot_v0, build_slayer_otf_ainteract_v0])
def test_rule_present_only_when_readonly(builder) -> None:
    assert _NO_REDERIVE_READONLY in builder(readonly_mode=True)
    assert _NO_REDERIVE_READONLY not in builder(readonly_mode=False)


@pytest.mark.parametrize("builder", [build_slayer_otf_one_shot_v0, build_slayer_otf_ainteract_v0])
def test_rule_survives_lean_gating(builder) -> None:
    """lean_introspection gates a DIFFERENT block (the tools tail); it must not
    drop the re-derivation rule from the readonly template."""
    assert _NO_REDERIVE_READONLY in builder(readonly_mode=True, lean_introspection=True)


# ---------------------------------------------------------------------------
# Absent from the non-readonly / frozen / pre-encoded templates.
# ---------------------------------------------------------------------------


def test_rule_absent_from_frozen_v0_constants() -> None:
    assert _NO_REDERIVE_READONLY not in SLAYER_OTF_ONE_SHOT_V0
    assert _NO_REDERIVE_READONLY not in SLAYER_OTF_AINTERACT_V0


def test_rule_absent_from_pre_encoded_prompts() -> None:
    assert _NO_REDERIVE_READONLY not in SLAYER_PRE_ENCODED_ONE_SHOT
    assert _NO_REDERIVE_READONLY not in SLAYER_PRE_ENCODED_AINTERACT
    # The v0 one-shot agent uses build_slayer_pre_encoded_one_shot(readonly_mode=)
    # in the pre-encoded (--pre-encoded-models) path — also out of scope.
    assert _NO_REDERIVE_READONLY not in build_slayer_pre_encoded_one_shot(
        readonly_mode=True
    )


def test_rule_absent_from_v1_otf_prompts() -> None:
    """v1 readonly-prompt wiring is out of scope (deferred follow-up); the v1
    OTF templates must be untouched by this change."""
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import (
        SLAYER_OTF_ONE_SHOT as V1_ONE_SHOT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT as V1_AINTERACT,
    )

    assert _NO_REDERIVE_READONLY not in V1_ONE_SHOT
    assert _NO_REDERIVE_READONLY not in V1_AINTERACT


# ---------------------------------------------------------------------------
# The readonly template still substitutes cleanly (no stray/undoubled braces).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [build_slayer_otf_one_shot_v0, build_slayer_otf_ainteract_v0])
def test_readonly_template_formats_cleanly(builder) -> None:
    """The template still substitutes the three format params without raising
    (a KeyError/IndexError would mean the new block introduced an undoubled
    brace). NB: these prompts legitimately contain doubled ``{{...}}`` JSON
    examples that become real single braces after ``.format`` — so we do NOT
    assert 'no braces remain', only that substitution succeeds and the named
    placeholders are consumed."""
    rendered = builder(readonly_mode=True).format(**_FMT)
    assert str(_FMT["budget"]) in rendered
    assert _FMT["db_name"] in rendered
    assert _FMT["user_query"] in rendered
    assert "{budget}" not in rendered and "{db_name}" not in rendered


def test_constant_is_nonempty_str() -> None:
    assert isinstance(_NO_REDERIVE_READONLY, str) and _NO_REDERIVE_READONLY.strip()
