"""DEV-1603 — align the raw a-interact prompt with the slayer a-interact prompt.

Scope (decided): a-interact only; the two portable failure-mode probes
(`_TABLE_SET_PROBE`, `_GRADER_ZERO_VS_ONE_DIAGNOSTIC`) + a condensed raw
host/join-path principle are ported into the RAW prompt via shared constants
that both modes reference; the SLAYER prompt's rendered text is UNCHANGED.

v0-only for the probes (v1 stays the curated lean set); the raw host principle
goes to BOTH raw v0 and raw v1 (raw v1's one genuine asymmetry vs slayer v1).

Mechanical contracts only (per ``feedback_no_prompt_content_tests``): golden
value-equality on our own output, format-field coverage, composition wiring of
the shared constants, and a no-leak guard. No hand-authored anchor phrases.
"""

from __future__ import annotations

import importlib
import pathlib
import string

import pytest

_DATA = pathlib.Path(__file__).parent / "data" / "dev1603"
_GOLDEN = _DATA / "slayer_ainteract_v0.golden.txt"
_RAW_GOLDEN = _DATA / "raw_ainteract_v0.golden.txt"

# Slayer fills MUST reproduce today's rendered text byte-for-byte.
_SLAYER_TABLE_SET = dict(knowledge_label="KB", schema_source="SLayer's schema lookup")
_SLAYER_GRADER = dict(attempt_noun="encoding", apply_verb="encode")
# Raw fills.
_RAW_TABLE_SET = dict(knowledge_label="knowledge definition", schema_source="the schema")
_RAW_GRADER = dict(attempt_noun="query", apply_verb="add")

_RUNTIME_FIELDS = {"budget", "db_name", "user_query"}


def _fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


# ---------------------------------------------------------------------------
# Hard requirement: the SLAYER a-interact v0 prompt must NOT change.
# ---------------------------------------------------------------------------
def test_slayer_ainteract_v0_unchanged() -> None:
    # Public re-export (what the v0 agent consumes) so re-export wiring drift
    # is caught alongside the constant.
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    expected = _GOLDEN.read_text()
    assert SLAYER_OTF_AINTERACT == expected, (
        "SLAYER_OTF_AINTERACT changed — DEV-1603 must not alter the slayer "
        f"prompt (len now={len(SLAYER_OTF_AINTERACT)}, golden={len(expected)})."
    )


# ---------------------------------------------------------------------------
# The two probe constants are parameterised so raw can fill them without
# leaking slayer wording; slayer fills reproduce today's bytes.
# ---------------------------------------------------------------------------
def test_table_set_probe_parameterised_for_mode() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import _TABLE_SET_PROBE

    assert {"knowledge_label", "schema_source"} <= _fields(_TABLE_SET_PROBE)
    for fills in (_SLAYER_TABLE_SET, _RAW_TABLE_SET):
        assert _fields(_TABLE_SET_PROBE.format(**fills)) == set()


def test_grader_zero_parameterised_for_mode() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import (
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC,
    )

    assert {"attempt_noun", "apply_verb"} <= _fields(_GRADER_ZERO_VS_ONE_DIAGNOSTIC)
    for fills in (_SLAYER_GRADER, _RAW_GRADER):
        assert _fields(_GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(**fills)) == set()


# ---------------------------------------------------------------------------
# The two probes are composed into RAW v0 with raw-mode fills.
# ---------------------------------------------------------------------------
def test_probes_composed_into_raw_v0() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import (
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC,
        _TABLE_SET_PROBE,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    assert _TABLE_SET_PROBE.format(**_RAW_TABLE_SET) in RAW_OTF_AINTERACT
    assert _GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(**_RAW_GRADER) in RAW_OTF_AINTERACT


def test_raw_v0_block_order() -> None:
    """USER_SIM -> TABLE_SET -> GRADER_ZERO -> ... -> host principle.

    Markers are the rendered shared constants (mechanical), not anchor phrases.
    """
    from bird_interact_agents.agents._shared_otf_prompts import (
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC,
        _RAW_HOST_PATH_PRINCIPLE,
        _TABLE_SET_PROBE,
        _USER_SIM_TRUST_CALIBRATION,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    i_usersim = RAW_OTF_AINTERACT.index(
        _USER_SIM_TRUST_CALIBRATION.format(knowledge_label="knowledge definition")
    )
    i_table = RAW_OTF_AINTERACT.index(_TABLE_SET_PROBE.format(**_RAW_TABLE_SET))
    i_grader = RAW_OTF_AINTERACT.index(
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(**_RAW_GRADER)
    )
    i_host = RAW_OTF_AINTERACT.index(_RAW_HOST_PATH_PRINCIPLE)
    assert i_usersim < i_table < i_grader < i_host, (
        f"unexpected order: usersim={i_usersim} table={i_table} "
        f"grader={i_grader} host={i_host}"
    )


def test_raw_ainteract_v0_matches_golden() -> None:
    """Byte-preservation lock: raw v0 == today's raw + exactly the planned
    inserted blocks (golden authored during implementation, eyeball-diffed
    against the pre-change raw)."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    expected = _RAW_GOLDEN.read_text()
    assert RAW_OTF_AINTERACT == expected, (
        f"RAW_OTF_AINTERACT drifted from golden "
        f"(len now={len(RAW_OTF_AINTERACT)}, golden={len(expected)})."
    )


# ---------------------------------------------------------------------------
# New condensed raw host/join-path principle: exists, concat-safe, no slayer
# vocab, and composed into BOTH raw v0 and raw v1.
# ---------------------------------------------------------------------------
def test_raw_host_principle_clean() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import _RAW_HOST_PATH_PRINCIPLE

    assert isinstance(_RAW_HOST_PATH_PRINCIPLE, str) and _RAW_HOST_PATH_PRINCIPLE.strip()
    assert _fields(_RAW_HOST_PATH_PRINCIPLE) == set(), "must be concat-safe (no fields)"
    for term in (
        "SLayer",
        "submit_query",
        "create_model",
        "edit_model",
        "[kb=",
        "mcp__slayer__",
        "inspect_model",
        "source_model",
        "memory:",
    ):
        assert term not in _RAW_HOST_PATH_PRINCIPLE, (
            f"slayer vocab {term!r} in raw host principle"
        )


def test_raw_host_principle_in_raw_v0_and_v1() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import _RAW_HOST_PATH_PRINCIPLE
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT as RAW_V0,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
        RAW_OTF_AINTERACT as RAW_V1,
    )

    assert _RAW_HOST_PATH_PRINCIPLE in RAW_V0
    assert _RAW_HOST_PATH_PRINCIPLE in RAW_V1


# ---------------------------------------------------------------------------
# Scope guard: the probes must NOT appear in either v1 prompt (v1 stays lean).
# ---------------------------------------------------------------------------
def test_probes_absent_from_v1_both_modes() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import (
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC,
        _TABLE_SET_PROBE,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
        RAW_OTF_AINTERACT as RAW_V1,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT as SLAYER_V1,
    )

    renders = [
        _TABLE_SET_PROBE.format(**_SLAYER_TABLE_SET),
        _TABLE_SET_PROBE.format(**_RAW_TABLE_SET),
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(**_SLAYER_GRADER),
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(**_RAW_GRADER),
    ]
    for prompt in (SLAYER_V1, RAW_V1):
        for block in renders:
            assert block not in prompt, "v1 must stay probe-free (DEV-1603 scope)"


# ---------------------------------------------------------------------------
# Every a-interact prompt's only runtime format fields are budget/db_name/
# user_query (proves all new params are supplied at compose time).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "modpath,name",
    [
        ("claude_sdk_otf_ainteract.prompts", "SLAYER_OTF_AINTERACT"),
        ("claude_sdk_otf_ainteract_raw.prompts", "RAW_OTF_AINTERACT"),
        ("claude_sdk_otf_ainteract_v1.prompts", "SLAYER_OTF_AINTERACT"),
        ("claude_sdk_otf_ainteract_raw_v1.prompts", "RAW_OTF_AINTERACT"),
    ],
)
def test_only_runtime_format_fields(modpath: str, name: str) -> None:
    mod = importlib.import_module(f"bird_interact_agents.agents.{modpath}")
    prompt = getattr(mod, name)
    extra = _fields(prompt) - _RUNTIME_FIELDS
    assert extra == set(), f"{name} has unexpected format fields: {extra}"


# ---------------------------------------------------------------------------
# Raw v0 carries no slayer vocab and no real eval-set names (synthetic only).
# ---------------------------------------------------------------------------
def test_raw_v0_no_slayer_vocab_and_synthetic() -> None:
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    for term in (
        "SLayer",
        "submit_query",
        "create_model",
        "edit_model",
        "[kb=",
        "mcp__slayer__",
        "inspect_model",
        "source_model",
        "memory:",
    ):
        assert term not in RAW_OTF_AINTERACT, f"slayer vocab {term!r} leaked into raw v0"
    banned = [
        "households", "tenure_type", "income_bracket", "dwelling_class",
        "socsupport", "service_types", "stellardist", "photo_band", "taguatinga",
    ]
    low = RAW_OTF_AINTERACT.lower()
    for name in banned:
        assert name not in low, f"real eval-set name {name!r} leaked into raw v0"
