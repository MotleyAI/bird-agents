"""DEV-1907: the slayer-mode claude_sdk prompts stop restating guidance the
0.10.2 tool descriptions / SlayerQuery schema / help.intro memory already
carry, teach the single-query capability, keep the harness-specific
discipline, and leave the raw prompts byte-identical.

Covers spec `slayer-agent-prompts` (all requirements).

Prompt text is read from the UN-formatted constants/builders (the markers
never live inside `{…}` placeholders). Families whose package still imports a
removed SLayer symbol (the un-migrated v1 / encoder OTF agents) fail to import
here — that is the import-compat feature gap, RED until the change lands.
"""

from __future__ import annotations

import pathlib

import pytest

_DATA = pathlib.Path(__file__).parent / "data" / "dev1907"

# Guidance the 0.10.2 tool docs / schema / help.intro carry — must NOT appear.
_HELP_TOOL = "`help`"
_REMOVED_MARKERS = [
    "query_nested",
    _HELP_TOOL,
    "colon-aggregation",
    "*:count",
    "revenue:sum",
    "DEDUP vs RAW ROWS",
    "SANITY-CHECK THE GENERATED SQL FOR SLAYER ARTIFACTS",
    "SEARCH-vs-INSPECT DISCIPLINE",
]


# --- slayer-mode family accessors (lazy so import breaks localise) ----------

def _v0_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import SLAYER_OTF_ONE_SHOT_V0
    return SLAYER_OTF_ONE_SHOT_V0


def _v0_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import SLAYER_OTF_AINTERACT_V0
    return SLAYER_OTF_AINTERACT_V0


def _v0_one_shot_readonly():
    from bird_interact_agents.agents._shared_otf_prompts import build_slayer_otf_one_shot_v0
    return build_slayer_otf_one_shot_v0(readonly_mode=True)


def _v0_ainteract_readonly():
    from bird_interact_agents.agents._shared_otf_prompts import build_slayer_otf_ainteract_v0
    return build_slayer_otf_ainteract_v0(readonly_mode=True)


def _v1_one_shot():
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import SLAYER_OTF_ONE_SHOT
    return SLAYER_OTF_ONE_SHOT


def _v1_ainteract():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import SLAYER_OTF_AINTERACT
    return SLAYER_OTF_AINTERACT


def _pre_one_shot():
    from bird_interact_agents.agents._pre_encoded_prompts import SLAYER_PRE_ENCODED_ONE_SHOT
    return SLAYER_PRE_ENCODED_ONE_SHOT


def _pre_ainteract():
    from bird_interact_agents.agents._pre_encoded_prompts import SLAYER_PRE_ENCODED_AINTERACT
    return SLAYER_PRE_ENCODED_AINTERACT


def _pre_one_shot_v1():
    from bird_interact_agents.agents._pre_encoded_prompts import SLAYER_PRE_ENCODED_ONE_SHOT_V1
    return SLAYER_PRE_ENCODED_ONE_SHOT_V1


def _pre_ainteract_v1():
    from bird_interact_agents.agents._pre_encoded_prompts import SLAYER_PRE_ENCODED_AINTERACT_V1
    return SLAYER_PRE_ENCODED_AINTERACT_V1


def _encoder():
    from bird_interact_agents.agents.claude_sdk_otf_encode.prompts import ENCODER_PROMPT
    return ENCODER_PROMPT


def _a_interact():
    from bird_interact_agents.agents.claude_sdk.prompts import SLAYER_A_INTERACT
    return SLAYER_A_INTERACT


def _c_interact():
    from bird_interact_agents.agents.claude_sdk.prompts import SLAYER_C_INTERACT
    return SLAYER_C_INTERACT


# Every slayer-mode prompt must be free of the removed tool-covered guidance.
_ALL_SLAYER = [
    ("v0_one_shot", _v0_one_shot),
    ("v0_ainteract", _v0_ainteract),
    ("v0_one_shot_readonly", _v0_one_shot_readonly),
    ("v0_ainteract_readonly", _v0_ainteract_readonly),
    ("v1_one_shot", _v1_one_shot),
    ("v1_ainteract", _v1_ainteract),
    ("pre_one_shot", _pre_one_shot),
    ("pre_ainteract", _pre_ainteract),
    ("pre_one_shot_v1", _pre_one_shot_v1),
    ("pre_ainteract_v1", _pre_ainteract_v1),
    ("encoder", _encoder),
    ("a_interact", _a_interact),
    ("c_interact", _c_interact),
]

_ONE_SHOT = ("v0_one_shot", _v0_one_shot)
_AINTERACT = ("v0_ainteract", _v0_ainteract)
_ONE_SHOT_RO = ("v0_one_shot_readonly", _v0_one_shot_readonly)
_AINTERACT_RO = ("v0_ainteract_readonly", _v0_ainteract_readonly)
_V1_ONE_SHOT = ("v1_one_shot", _v1_one_shot)
_V1_AINTERACT = ("v1_ainteract", _v1_ainteract)
_PRE_ONE_SHOT = ("pre_one_shot", _pre_one_shot)
_PRE_AINTERACT = ("pre_ainteract", _pre_ainteract)
_PRE_ONE_SHOT_V1 = ("pre_one_shot_v1", _pre_one_shot_v1)
_PRE_AINTERACT_V1 = ("pre_ainteract_v1", _pre_ainteract_v1)
_ENCODER = ("encoder", _encoder)

# Kept-block retention guards — each list is exactly the set of families that
# CARRY that block today (verified against the current source), so the guard is
# green now and RED only if the cleanup accidentally over-trims it.
_MUTATION_CHECK_FAMILIES = [
    _ONE_SHOT, _AINTERACT, _ONE_SHOT_RO, _AINTERACT_RO, _V1_ONE_SHOT, _V1_AINTERACT,
    _PRE_ONE_SHOT, _PRE_AINTERACT, _PRE_ONE_SHOT_V1, _PRE_AINTERACT_V1,
]
_KB_TRACE_FAMILIES = _MUTATION_CHECK_FAMILIES + [_ENCODER]
_GRADING_FAMILIES = _MUTATION_CHECK_FAMILIES
_GATE_FAMILIES = [_ONE_SHOT, _AINTERACT, _ONE_SHOT_RO, _AINTERACT_RO]
_READONLY_FAMILIES = [_ONE_SHOT_RO, _AINTERACT_RO]
_ASK_AGAIN_FAMILIES = [_AINTERACT, _AINTERACT_RO, _V1_AINTERACT, _PRE_AINTERACT, _PRE_AINTERACT_V1]
# Query-writing families that must carry the new filter-normalisation note.
_NORMALIZE_NOTE_FAMILIES = [
    _ONE_SHOT, _AINTERACT, _V1_ONE_SHOT, _V1_AINTERACT,
    _PRE_ONE_SHOT, _PRE_AINTERACT, _PRE_ONE_SHOT_V1, _PRE_AINTERACT_V1,
]


def _has_single_query_nudge(text: str) -> bool:
    low = text.lower()
    one_query = any(s in low for s in ("one query", "single query", "single, one query"))
    exception = any(s in low for s in ("multi-stage", "multistage", "multi stage"))
    return one_query and exception

# Raw (non-slayer) prompts must stay byte-identical to the captured goldens.
def _raw_v0_one_shot():
    from bird_interact_agents.agents._shared_otf_prompts import RAW_OTF_ONE_SHOT_V0
    return RAW_OTF_ONE_SHOT_V0


def _raw_v0_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import RAW_OTF_AINTERACT_V0
    return RAW_OTF_AINTERACT_V0


def _raw_v1_one_shot():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import RAW_OTF_ONE_SHOT
    return RAW_OTF_ONE_SHOT


def _raw_v1_ainteract():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import RAW_OTF_AINTERACT
    return RAW_OTF_AINTERACT


def _raw_a_interact():
    from bird_interact_agents.agents.claude_sdk.prompts import RAW_A_INTERACT
    return RAW_A_INTERACT


def _raw_c_interact():
    from bird_interact_agents.agents.claude_sdk.prompts import RAW_C_INTERACT
    return RAW_C_INTERACT


_RAW_FAMILIES = [
    ("raw_one_shot_v0.golden.txt", _raw_v0_one_shot),
    ("raw_ainteract_v0.golden.txt", _raw_v0_ainteract),
    ("raw_one_shot_v1.golden.txt", _raw_v1_one_shot),
    ("raw_ainteract_v1.golden.txt", _raw_v1_ainteract),
    ("raw_a_interact.golden.txt", _raw_a_interact),
    ("raw_c_interact.golden.txt", _raw_c_interact),
]


# ===========================================================================


@pytest.mark.parametrize("fam_id,accessor", _ALL_SLAYER)
def test_slayer_prompt_free_of_removed_tool_guidance(fam_id, accessor):
    text = accessor()
    present = [m for m in _REMOVED_MARKERS if m in text]
    assert not present, f"{fam_id} still restates tool-covered guidance: {present}"


def test_single_query_nudge_constant_defined_and_semantic():
    """The shared nudge constant exists, is non-empty, and names both the
    single-query capability and the multi-stage-list exception (so a family
    that merely embeds an EMPTY constant cannot pass vacuously)."""
    from bird_interact_agents.agents._shared_otf_prompts import _SINGLE_QUERY_NUDGE

    assert _SINGLE_QUERY_NUDGE.strip(), "shared single-query nudge constant is empty"
    assert _has_single_query_nudge(_SINGLE_QUERY_NUDGE), (
        "nudge must name single-query + the multi-stage-list exception"
    )


@pytest.mark.parametrize("fam_id,accessor", _ALL_SLAYER)
def test_slayer_prompt_teaches_single_query_capability(fam_id, accessor):
    """Every slayer-mode prompt carries the ONE shared nudge constant. Using
    the constant (not just the semantic tokens, which some prompts already
    contain incidentally) means the specific nudge must actually be spliced in;
    the companion test above guarantees the constant itself is non-vacuous."""
    from bird_interact_agents.agents._shared_otf_prompts import _SINGLE_QUERY_NUDGE

    assert _SINGLE_QUERY_NUDGE in accessor(), f"{fam_id} is missing the shared single-query nudge"


@pytest.mark.parametrize("fam_id,accessor", _GRADING_FAMILIES)
def test_positional_grading_rule_retained(fam_id, accessor):
    assert "COLUMN HEADERS DO NOT AFFECT GRADING" in accessor(), (
        f"{fam_id} dropped the positional-grading rule"
    )


@pytest.mark.parametrize("fam_id,accessor", _GATE_FAMILIES)
def test_query_before_submit_gate_retained(fam_id, accessor):
    assert "RUN THE FINAL QUERY BEFORE YOU SUBMIT" in accessor(), (
        f"{fam_id} dropped the query-before-submit gate"
    )


@pytest.mark.parametrize("fam_id,accessor", _MUTATION_CHECK_FAMILIES)
def test_pre_submit_mutation_check_retained(fam_id, accessor):
    assert "PRE-SUBMIT MUTATION CHECK" in accessor(), f"{fam_id} dropped the pre-submit mutation check"


@pytest.mark.parametrize("fam_id,accessor", _KB_TRACE_FAMILIES)
def test_kb_traceability_rule_retained(fam_id, accessor):
    assert "[kb=" in accessor(), f"{fam_id} dropped the [kb=N] traceability rule"


@pytest.mark.parametrize("fam_id,accessor", _ASK_AGAIN_FAMILIES)
def test_interactive_ask_again_discipline_retained(fam_id, accessor):
    assert "ASK AGAIN IF NEEDED" in accessor(), f"{fam_id} dropped the ask-again discipline"


@pytest.mark.parametrize("fam_id,accessor", _READONLY_FAMILIES)
def test_readonly_no_rederive_rule_retained(fam_id, accessor):
    assert "DO NOT RE-DERIVE AN ALREADY-ENCODED QUANTITY" in accessor(), (
        f"{fam_id} dropped the readonly no-rederive rule"
    )


def test_normalize_filters_note_defined_and_accurate():
    """The new filter-normalisation note exists, is non-empty, and describes
    the ACTUAL normaliser behaviour (the lower(trim(...)) wrap + the opt-out),
    not a vague "every filter" claim."""
    from bird_interact_agents.agents._shared_otf_prompts import _NORMALIZE_FILTERS_NOTE

    assert _NORMALIZE_FILTERS_NOTE.strip(), "filter-normalisation note constant is empty"
    low = _NORMALIZE_FILTERS_NOTE.lower()
    assert "normalize_filters" in low, "note must name the normalize_filters opt-out"
    assert "lower" in low and "trim" in low, "note must describe the lower(trim(...)) behaviour"


@pytest.mark.parametrize("fam_id,accessor", _NORMALIZE_NOTE_FAMILIES)
def test_normalize_filters_note_present(fam_id, accessor):
    from bird_interact_agents.agents._shared_otf_prompts import _NORMALIZE_FILTERS_NOTE

    assert _NORMALIZE_FILTERS_NOTE in accessor(), f"{fam_id} is missing the filter-normalisation note"


@pytest.mark.parametrize("golden_name,accessor", _RAW_FAMILIES)
def test_raw_prompt_byte_identical(golden_name, accessor):
    expected = (_DATA / golden_name).read_text(encoding="utf-8")
    assert accessor() == expected, f"raw prompt {golden_name} changed (must stay byte-identical)"
