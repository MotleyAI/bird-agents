"""DEV-1670: trim the v0 readonly OTF QUERY prompt (apply-saved-models regime).

The lean+readonly+``--apply-edited-models`` QUERY agent renders
``build_slayer_otf_{one_shot,ainteract}_v0(readonly_mode=True)`` (confirmed:
``claude_sdk_otf/agent.py::_build_prompt`` selects it when
``pre_encoded_source is None``). This suite pins the readonly-branch changes:

* R3 — the ENCODE authoring guidance is dropped from the readonly branch
  (``ENCODE_HOST_GUIDANCE``), while the query-quality blocks and DEV-1672's
  ``_NO_REDERIVE_READONLY`` survive the DISCOVER reframe.
* R1/R2 — the new readonly-only nudges are wired into the readonly branch ONLY,
  and do NOT leak into the shared ``_COMPACT_SEARCH_DISCIPLINE`` block, the
  pre-encoded templates, or the v1 OTF prompts.
* Format/brace safety + non-readonly byte identity are preserved.

Per the project's no-prompt-content-tests rule these are WIRING/composition
assertions ONLY — every check references a shared CONSTANT OBJECT (or a tool
name) and the readonly gate, never a hand-typed prose phrase or coverage
matrix (mirrors the DEV-1666 / DEV-1672 readonly-gating tests). Token +
behaviour deltas (R1/R2/R4 wording, help.intro repositioning) are validated by
a real benchmark run, not here.
"""

from __future__ import annotations

import string

import pytest

from bird_interact_agents.agents import _shared_otf_prompts as sp

_FMT = dict(budget=100, db_name="mydb", user_query="how many?")
_BUILDERS = [sp.build_slayer_otf_one_shot_v0, sp.build_slayer_otf_ainteract_v0]
_LEGACY = {
    sp.build_slayer_otf_one_shot_v0: sp.SLAYER_OTF_ONE_SHOT_V0,
    sp.build_slayer_otf_ainteract_v0: sp.SLAYER_OTF_AINTERACT_V0,
}

# The readonly-only nudge constants introduced by DEV-1670 (referenced via
# getattr so a missing feature fails on assertion, not module collection).
_NUDGE_CONSTS = ("_ONE_QUERY_DISCIPLINE_READONLY", "_BATCH_INSPECT_READONLY")


def _fields(s: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(s) if f}


# ---------------------------------------------------------------------------
# R3 — ENCODE authoring guidance dropped in readonly, kept in non-readonly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _BUILDERS)
def test_encode_host_guidance_dropped_in_readonly(builder) -> None:
    assert sp.ENCODE_HOST_GUIDANCE in builder(readonly_mode=False)
    assert sp.ENCODE_HOST_GUIDANCE not in builder(readonly_mode=True)


@pytest.mark.parametrize("builder", _BUILDERS)
def test_encode_host_guidance_dropped_under_readonly_lean(builder) -> None:
    # lean gates a DIFFERENT block (the tools tail); it must not resurrect the
    # authoring host-selection guidance in the readonly template.
    assert sp.ENCODE_HOST_GUIDANCE not in builder(
        readonly_mode=True, lean_introspection=True
    )


# ---------------------------------------------------------------------------
# R3 — the DEV-1672 re-derive rule + the query-quality blocks survive the
# reframe (regression guards for the readonly rewrite).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _BUILDERS)
def test_no_rederive_rule_survives_reframe(builder) -> None:
    assert sp._NO_REDERIVE_READONLY in builder(readonly_mode=True)


@pytest.mark.parametrize("builder", _BUILDERS)
@pytest.mark.parametrize("block_name", ["_QUERY_BEFORE_SUBMIT", "QUERY_ROOT_GUIDANCE"])
def test_query_quality_blocks_survive_reframe(builder, block_name) -> None:
    assert getattr(sp, block_name) in builder(readonly_mode=True)


@pytest.mark.parametrize("builder", _BUILDERS)
def test_sample_value_filter_mandate_survives_reframe(builder) -> None:
    # The correctness-critical noisy-categorical filter mandate (rendered with
    # the slayer `inspect` sample_source) must survive the DISCOVER reframe.
    rendered_block = sp._SAMPLE_VALUE_FILTER_MANDATE.format(sample_source="`inspect`")
    assert rendered_block in builder(readonly_mode=True)


# ---------------------------------------------------------------------------
# R1/R2 — new readonly-only nudges wired into readonly ONLY.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _BUILDERS)
@pytest.mark.parametrize("const_name", _NUDGE_CONSTS)
def test_nudges_present_only_when_readonly(builder, const_name) -> None:
    nudge = getattr(sp, const_name, None)
    assert nudge, f"expected DEV-1670 feature constant {const_name} to exist + be non-empty"
    assert nudge in builder(readonly_mode=True)
    assert nudge not in builder(readonly_mode=False)


@pytest.mark.parametrize("builder", _BUILDERS)
@pytest.mark.parametrize("const_name", _NUDGE_CONSTS)
def test_nudges_survive_lean_gating(builder, const_name) -> None:
    nudge = getattr(sp, const_name, None)
    assert nudge, f"expected DEV-1670 feature constant {const_name} to exist"
    assert nudge in builder(readonly_mode=True, lean_introspection=True)


# ---------------------------------------------------------------------------
# Scope guards (Codex finding #2): R2 is an ADDED readonly-only sentence, NOT an
# edit to the shared _COMPACT_SEARCH_DISCIPLINE; nudges stay out of the
# pre-encoded + v1 surfaces.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _BUILDERS)
def test_shared_compact_search_block_present_in_both_branches(builder) -> None:
    # The shared block must remain present in BOTH branches — the reframe must
    # not have dropped it from the readonly discover flow.
    assert sp._COMPACT_SEARCH_DISCIPLINE in builder(readonly_mode=True)
    assert sp._COMPACT_SEARCH_DISCIPLINE in builder(readonly_mode=False)


@pytest.mark.parametrize("const_name", _NUDGE_CONSTS)
def test_r2_nudge_is_not_an_edit_to_shared_compact_search(const_name) -> None:
    # Codex finding #2: R2 must be an ADDED readonly-only sentence, NOT an
    # in-place edit of the shared _COMPACT_SEARCH_DISCIPLINE (which feeds
    # non-readonly + v1 surfaces). Mutating the shared block would leave the
    # nudge text INSIDE it — assert it is not there.
    nudge = getattr(sp, const_name, None)
    assert nudge, f"expected DEV-1670 feature constant {const_name} to exist"
    assert nudge not in sp._COMPACT_SEARCH_DISCIPLINE


def test_nudges_absent_from_pre_encoded_and_v1_surfaces() -> None:
    from bird_interact_agents.agents._pre_encoded_prompts import (
        SLAYER_PRE_ENCODED_AINTERACT,
        SLAYER_PRE_ENCODED_ONE_SHOT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT as V1_AINTERACT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import (
        SLAYER_OTF_ONE_SHOT as V1_ONE_SHOT,
    )

    surfaces = (
        SLAYER_PRE_ENCODED_ONE_SHOT,
        SLAYER_PRE_ENCODED_AINTERACT,
        V1_ONE_SHOT,
        V1_AINTERACT,
    )
    for const_name in _NUDGE_CONSTS:
        nudge = getattr(sp, const_name, None)
        assert nudge, f"expected DEV-1670 feature constant {const_name} to exist"
        for surface in surfaces:
            assert nudge not in surface


# ---------------------------------------------------------------------------
# Codex finding #3 — no write-tool names anywhere in the readonly template.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _BUILDERS)
@pytest.mark.parametrize("lean", [True, False])
def test_readonly_template_has_no_write_tool_names(builder, lean) -> None:
    ro = builder(readonly_mode=True, lean_introspection=lean)
    assert "create_model" not in ro
    assert "edit_model" not in ro


# ---------------------------------------------------------------------------
# Codex finding #4 — a-interact readonly still mandates ask_user before submit
# (the hard submit gate denies submit_query while ask_count == 0). Constant-
# based: the reframed RULE-0 must be the QUERY variant of _RULE_0_ASK_BEFORE
# (ENCODE -> QUERY), which still carries the "call ask_user ONCE ... submit gate
# will REFUSE submit_query until you have called ask_user" mandate.
# ---------------------------------------------------------------------------


def test_ainteract_readonly_uses_query_rule0_ask_before() -> None:
    rendered_rule0 = sp._RULE_0_ASK_BEFORE.format(
        action_label="QUERY",
        action_context="BEFORE building the final query,",
        submit_tool="submit_query",
    )
    assert rendered_rule0 in sp.build_slayer_otf_ainteract_v0(readonly_mode=True)


def test_ainteract_readonly_drops_encode_rule0() -> None:
    # The old ENCODE-framed RULE-0 must be gone from the readonly a-interact
    # template (it belongs to the write-tool authoring flow).
    encode_rule0 = sp._RULE_0_ASK_BEFORE.format(
        action_label="ENCODE",
        action_context="BEFORE the encoding loop below,",
        submit_tool="submit_query",
    )
    assert encode_rule0 not in sp.build_slayer_otf_ainteract_v0(readonly_mode=True)


# ---------------------------------------------------------------------------
# Format / brace safety across the reframe.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _BUILDERS)
def test_readonly_template_formats_cleanly(builder) -> None:
    rendered = builder(readonly_mode=True).format(**_FMT)
    assert str(_FMT["budget"]) in rendered
    assert _FMT["db_name"] in rendered
    assert _FMT["user_query"] in rendered
    assert "{budget}" not in rendered
    assert "{db_name}" not in rendered
    assert "{user_query}" not in rendered


@pytest.mark.parametrize("builder", _BUILDERS)
@pytest.mark.parametrize("lean", [True, False])
def test_readonly_format_fields_are_exactly_the_three(builder, lean) -> None:
    assert _fields(builder(readonly_mode=True, lean_introspection=lean)) == {
        "budget",
        "db_name",
        "user_query",
    }


# ---------------------------------------------------------------------------
# Codex finding #6 — non-readonly byte identity preserved for BOTH builders.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _BUILDERS)
def test_non_readonly_false_false_is_legacy_identity(builder) -> None:
    assert builder(lean_introspection=False, readonly_mode=False) == _LEGACY[builder]
