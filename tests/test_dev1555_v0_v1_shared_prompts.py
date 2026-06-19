"""DEV-1555 v0/v1 split — shared OTF prompt constants (post-unification).

The earlier hard byte-identity pin to origin/main was dropped when the
v0 + v1 query tool was unified (CR r1 / O1): both agents now use the
same ``query`` tool that accepts a single object OR a list of stage
objects, and ``query_nested`` is no longer in any agent's allowlist.
v0 prompts had to be rewritten to teach the unified contract, so the
SHA-256 snapshot pinning to origin/main no longer fits — what survives
are presence + difference tests:

* ``_shared_otf_prompts.py`` still exports the four ``*_V0`` constants
  (the v0 snapshot the v0 agents render).
* v0 and v1 unsuffixed re-exports load cleanly via the respective
  agent dirs' ``prompts.py`` modules.
* v0 and v1 SLAYER prompts continue to differ as strings (otherwise
  the A/B has nothing to compare).
* Neither v0 nor v1 prompts mention ``query_nested`` or the legacy
  ``query_json`` parameter anymore.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# V0 prompt constants exist in _shared_otf_prompts and re-export
# cleanly via each v0 agent dir's prompts.py.
# ---------------------------------------------------------------------------


def test_shared_module_exports_slayer_one_shot_v0():
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_ONE_SHOT_V0,
    )

    assert isinstance(SLAYER_OTF_ONE_SHOT_V0, str)
    assert SLAYER_OTF_ONE_SHOT_V0.strip()


def test_shared_module_exports_slayer_ainteract_v0():
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_AINTERACT_V0,
    )

    assert isinstance(SLAYER_OTF_AINTERACT_V0, str)
    assert SLAYER_OTF_AINTERACT_V0.strip()


def test_shared_module_exports_raw_one_shot_v0():
    from bird_interact_agents.agents._shared_otf_prompts import (
        RAW_OTF_ONE_SHOT_V0,
    )

    assert isinstance(RAW_OTF_ONE_SHOT_V0, str)
    assert RAW_OTF_ONE_SHOT_V0.strip()


def test_shared_module_exports_raw_ainteract_v0():
    from bird_interact_agents.agents._shared_otf_prompts import (
        RAW_OTF_AINTERACT_V0,
    )

    assert isinstance(RAW_OTF_AINTERACT_V0, str)
    assert RAW_OTF_AINTERACT_V0.strip()


# ---------------------------------------------------------------------------
# v0 prompts re-export the *_V0 constants under their origin/main name
# (the v0 agent imports `SLAYER_OTF_ONE_SHOT` etc., unchanged from the
# pre-split call sites).
# ---------------------------------------------------------------------------


def test_v0_slayer_one_shot_re_export_matches_shared():
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_ONE_SHOT_V0,
    )
    from bird_interact_agents.agents.claude_sdk_otf.prompts import (
        SLAYER_OTF_ONE_SHOT,
    )

    assert SLAYER_OTF_ONE_SHOT == SLAYER_OTF_ONE_SHOT_V0


def test_v0_slayer_ainteract_re_export_matches_shared():
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_AINTERACT_V0,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    assert SLAYER_OTF_AINTERACT == SLAYER_OTF_AINTERACT_V0


def test_v0_raw_one_shot_re_export_matches_shared():
    from bird_interact_agents.agents._shared_otf_prompts import (
        RAW_OTF_ONE_SHOT_V0,
    )
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import (
        RAW_OTF_ONE_SHOT,
    )

    assert RAW_OTF_ONE_SHOT == RAW_OTF_ONE_SHOT_V0


def test_v0_raw_ainteract_re_export_matches_shared():
    from bird_interact_agents.agents._shared_otf_prompts import (
        RAW_OTF_AINTERACT_V0,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    assert RAW_OTF_AINTERACT == RAW_OTF_AINTERACT_V0


# ---------------------------------------------------------------------------
# v1 prompt constants stay reachable through the v1 agent dirs.
# ---------------------------------------------------------------------------


def test_v1_slayer_one_shot_re_export_present():
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import (
        SLAYER_OTF_ONE_SHOT,
    )

    assert isinstance(SLAYER_OTF_ONE_SHOT, str) and SLAYER_OTF_ONE_SHOT.strip()


def test_v1_slayer_ainteract_re_export_present():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    assert isinstance(SLAYER_OTF_AINTERACT, str) and SLAYER_OTF_AINTERACT.strip()


def test_v1_raw_one_shot_re_export_present():
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import (
        RAW_OTF_ONE_SHOT,
    )

    assert isinstance(RAW_OTF_ONE_SHOT, str) and RAW_OTF_ONE_SHOT.strip()


def test_v1_raw_ainteract_re_export_present():
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
        RAW_OTF_AINTERACT,
    )

    assert isinstance(RAW_OTF_AINTERACT, str) and RAW_OTF_AINTERACT.strip()


# ---------------------------------------------------------------------------
# v0 ≠ v1 (the prompts must actually diverge — otherwise the A/B is empty).
# ---------------------------------------------------------------------------


def test_v0_and_v1_slayer_one_shot_differ():
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_ONE_SHOT_V0,
    )
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import (
        SLAYER_OTF_ONE_SHOT,
    )

    assert SLAYER_OTF_ONE_SHOT != SLAYER_OTF_ONE_SHOT_V0


def test_v0_and_v1_slayer_ainteract_differ():
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_AINTERACT_V0,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    assert SLAYER_OTF_AINTERACT != SLAYER_OTF_AINTERACT_V0


# ---------------------------------------------------------------------------
# Neither v0 nor v1 SLAYER prompts mention the obsolete `query_nested`
# tool name or the obsolete `query_json` parameter. The single tool is
# ``query`` and accepts a JSON object OR an array of stages.
# ---------------------------------------------------------------------------


def test_no_v0_slayer_prompt_mentions_query_nested():
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_ONE_SHOT_V0,
        SLAYER_OTF_AINTERACT_V0,
    )

    for name, val in [
        ("SLAYER_OTF_ONE_SHOT_V0", SLAYER_OTF_ONE_SHOT_V0),
        ("SLAYER_OTF_AINTERACT_V0", SLAYER_OTF_AINTERACT_V0),
    ]:
        assert "query_nested" not in val, (
            f"{name} still mentions `query_nested`; the unified query "
            "tool replaces it."
        )


def test_no_v1_slayer_prompt_mentions_query_nested():
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import (
        SLAYER_OTF_ONE_SHOT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    for name, val in [
        ("v1 SLAYER_OTF_ONE_SHOT", SLAYER_OTF_ONE_SHOT),
        ("v1 SLAYER_OTF_AINTERACT", SLAYER_OTF_AINTERACT),
    ]:
        assert "query_nested" not in val, (
            f"{name} still mentions `query_nested`."
        )


def test_no_v0_slayer_prompt_mentions_query_json():
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_ONE_SHOT_V0,
        SLAYER_OTF_AINTERACT_V0,
    )

    for name, val in [
        ("SLAYER_OTF_ONE_SHOT_V0", SLAYER_OTF_ONE_SHOT_V0),
        ("SLAYER_OTF_AINTERACT_V0", SLAYER_OTF_AINTERACT_V0),
    ]:
        assert "query_json" not in val, (
            f"{name} still mentions the obsolete `query_json` single-string "
            "parameter; the unified query tool takes named fields."
        )
