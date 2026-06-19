"""DEV-1555 v0/v1 split — shared OTF prompt constants.

Asserts that ``agents/_shared_otf_prompts.py`` carries BOTH the v1
(unsuffixed, current-branch) prompt constants and the v0 (``_V0``-suffixed,
origin/main snapshot) prompt constants, and that each pins a byte-for-byte
SHA snapshot.

The four ``*_V0`` SHA values below were computed from the origin/main
revision by importing the v0 prompt modules in an isolated package tree:

    git show origin/main:src/.../prompts.py  >  /tmp/v0_pkg/.../prompts.py
    PYTHONPATH=/tmp/v0_pkg python -c "
        import hashlib
        from bird_interact_agents.agents.claude_sdk_otf.prompts import \\
            SLAYER_OTF_ONE_SHOT
        print(hashlib.sha256(SLAYER_OTF_ONE_SHOT.encode()).hexdigest())
    "

This was done once at spec time and the resulting digests are baked in.
Any future intentional re-baseline of the V0 constants must re-baseline
these too.
"""

from __future__ import annotations

import hashlib


# ---------------------------------------------------------------------------
# Origin/main (v0) SHA snapshots — these are the bytes that v0 agents must
# render their prompts from. The implementation lands the strings under
# ``*_V0``-suffixed names in ``_shared_otf_prompts.py`` and re-exports them
# via the v0 agents' ``prompts.py``.
# ---------------------------------------------------------------------------

_SLAYER_OTF_ONE_SHOT_V0_SHA256 = (
    "2ef7a1fc6abacc9a8e8efc52701a74ddc6559793be2fad52421c1c50bbe7d6ef"
)
_SLAYER_OTF_AINTERACT_V0_SHA256 = (
    "b35e3e5454bb37b2028515c917100d12d5d35ce3e0fed82abbabf6c5970a8708"
)
_RAW_OTF_ONE_SHOT_V0_SHA256 = (
    "7db7a9a2bdd99edd5f3377f2fe405834bd8a2d035cf504cedbfe045025e0fbda"
)
_RAW_OTF_AINTERACT_V0_SHA256 = (
    "ed9c10065201cdd8c4139629fcd84548bca371678563e9239bb946b5d26c4343"
)


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# V0 prompt constants must exist in _shared_otf_prompts and match
# origin/main byte-for-byte.
# ---------------------------------------------------------------------------


def test_shared_module_exports_slayer_one_shot_v0():
    """``SLAYER_OTF_ONE_SHOT_V0`` exported from ``_shared_otf_prompts``."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_ONE_SHOT_V0,
    )

    assert isinstance(SLAYER_OTF_ONE_SHOT_V0, str)
    assert SLAYER_OTF_ONE_SHOT_V0.strip()


def test_shared_module_exports_slayer_ainteract_v0():
    """``SLAYER_OTF_AINTERACT_V0`` exported from ``_shared_otf_prompts``."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_AINTERACT_V0,
    )

    assert isinstance(SLAYER_OTF_AINTERACT_V0, str)
    assert SLAYER_OTF_AINTERACT_V0.strip()


def test_shared_module_exports_raw_one_shot_v0():
    """``RAW_OTF_ONE_SHOT_V0`` exported from ``_shared_otf_prompts``."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        RAW_OTF_ONE_SHOT_V0,
    )

    assert isinstance(RAW_OTF_ONE_SHOT_V0, str)
    assert RAW_OTF_ONE_SHOT_V0.strip()


def test_shared_module_exports_raw_ainteract_v0():
    """``RAW_OTF_AINTERACT_V0`` exported from ``_shared_otf_prompts``."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        RAW_OTF_AINTERACT_V0,
    )

    assert isinstance(RAW_OTF_AINTERACT_V0, str)
    assert RAW_OTF_AINTERACT_V0.strip()


def test_slayer_otf_one_shot_v0_matches_origin_main():
    """v0 slayer one-shot template is byte-identical to origin/main."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_ONE_SHOT_V0,
    )

    actual = _digest(SLAYER_OTF_ONE_SHOT_V0)
    assert actual == _SLAYER_OTF_ONE_SHOT_V0_SHA256, (
        "SLAYER_OTF_ONE_SHOT_V0 must be byte-for-byte origin/main "
        f"(len={len(SLAYER_OTF_ONE_SHOT_V0)}):\n"
        f"  expected: {_SLAYER_OTF_ONE_SHOT_V0_SHA256}\n"
        f"  actual:   {actual}"
    )


def test_slayer_otf_ainteract_v0_matches_origin_main():
    """v0 slayer a-interact template is byte-identical to origin/main."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_AINTERACT_V0,
    )

    actual = _digest(SLAYER_OTF_AINTERACT_V0)
    assert actual == _SLAYER_OTF_AINTERACT_V0_SHA256, (
        "SLAYER_OTF_AINTERACT_V0 must be byte-for-byte origin/main "
        f"(len={len(SLAYER_OTF_AINTERACT_V0)}):\n"
        f"  expected: {_SLAYER_OTF_AINTERACT_V0_SHA256}\n"
        f"  actual:   {actual}"
    )


def test_raw_otf_one_shot_v0_matches_origin_main():
    """v0 raw one-shot template is byte-identical to origin/main."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        RAW_OTF_ONE_SHOT_V0,
    )

    actual = _digest(RAW_OTF_ONE_SHOT_V0)
    assert actual == _RAW_OTF_ONE_SHOT_V0_SHA256, (
        "RAW_OTF_ONE_SHOT_V0 must be byte-for-byte origin/main "
        f"(len={len(RAW_OTF_ONE_SHOT_V0)}):\n"
        f"  expected: {_RAW_OTF_ONE_SHOT_V0_SHA256}\n"
        f"  actual:   {actual}"
    )


def test_raw_otf_ainteract_v0_matches_origin_main():
    """v0 raw a-interact template is byte-identical to origin/main."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        RAW_OTF_AINTERACT_V0,
    )

    actual = _digest(RAW_OTF_AINTERACT_V0)
    assert actual == _RAW_OTF_AINTERACT_V0_SHA256, (
        "RAW_OTF_AINTERACT_V0 must be byte-for-byte origin/main "
        f"(len={len(RAW_OTF_AINTERACT_V0)}):\n"
        f"  expected: {_RAW_OTF_AINTERACT_V0_SHA256}\n"
        f"  actual:   {actual}"
    )


# ---------------------------------------------------------------------------
# V0 prompts are re-exported by each v0 agent dir's ``prompts.py``.
# (This is the symbol the v0 agent.py actually imports — keep parity.)
# ---------------------------------------------------------------------------


def test_v0_slayer_one_shot_re_export_matches():
    """v0 ``claude_sdk_otf.prompts.SLAYER_OTF_ONE_SHOT`` matches V0 bytes."""
    from bird_interact_agents.agents.claude_sdk_otf.prompts import (
        SLAYER_OTF_ONE_SHOT,
    )

    assert _digest(SLAYER_OTF_ONE_SHOT) == _SLAYER_OTF_ONE_SHOT_V0_SHA256


def test_v0_slayer_ainteract_re_export_matches():
    """v0 ``claude_sdk_otf_ainteract.prompts.SLAYER_OTF_AINTERACT`` matches V0 bytes."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    assert _digest(SLAYER_OTF_AINTERACT) == _SLAYER_OTF_AINTERACT_V0_SHA256


def test_v0_raw_one_shot_re_export_matches():
    """v0 ``claude_sdk_otf_raw.prompts.RAW_OTF_ONE_SHOT`` matches V0 bytes."""
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import (
        RAW_OTF_ONE_SHOT,
    )

    assert _digest(RAW_OTF_ONE_SHOT) == _RAW_OTF_ONE_SHOT_V0_SHA256


def test_v0_raw_ainteract_re_export_matches():
    """v0 ``claude_sdk_otf_ainteract_raw.prompts.RAW_OTF_AINTERACT`` matches V0 bytes."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw.prompts import (
        RAW_OTF_AINTERACT,
    )

    assert _digest(RAW_OTF_AINTERACT) == _RAW_OTF_AINTERACT_V0_SHA256


# ---------------------------------------------------------------------------
# V1 unsuffixed constants stay accessible and re-exported by the v1 agent
# dirs (renamed packages). They're ASSEMBLED inside each v1 prompts.py
# from shared internals (the existing structure, unchanged from this
# branch). The existing SHA snapshot test in
# tests/test_shared_otf_prompts.py already pins the v1 bytes.
# ---------------------------------------------------------------------------


def test_v1_slayer_one_shot_re_export_present():
    """v1 ``claude_sdk_otf_v1.prompts.SLAYER_OTF_ONE_SHOT`` is exported."""
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import (
        SLAYER_OTF_ONE_SHOT,
    )

    assert isinstance(SLAYER_OTF_ONE_SHOT, str) and SLAYER_OTF_ONE_SHOT.strip()


def test_v1_slayer_ainteract_re_export_present():
    """v1 ``claude_sdk_otf_ainteract_v1.prompts.SLAYER_OTF_AINTERACT`` is exported."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    assert isinstance(SLAYER_OTF_AINTERACT, str) and SLAYER_OTF_AINTERACT.strip()


def test_v1_raw_one_shot_re_export_present():
    """v1 ``claude_sdk_otf_raw_v1.prompts.RAW_OTF_ONE_SHOT`` is exported."""
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import (
        RAW_OTF_ONE_SHOT,
    )

    assert isinstance(RAW_OTF_ONE_SHOT, str) and RAW_OTF_ONE_SHOT.strip()


def test_v1_raw_ainteract_re_export_present():
    """v1 ``claude_sdk_otf_ainteract_raw_v1.prompts.RAW_OTF_AINTERACT`` is exported."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts import (
        RAW_OTF_AINTERACT,
    )

    assert isinstance(RAW_OTF_AINTERACT, str) and RAW_OTF_AINTERACT.strip()


# ---------------------------------------------------------------------------
# v0 and v1 slayer constants must DIFFER (the whole point of the split).
# Catches a regression where someone accidentally points both names at the
# same underlying string and the A/B becomes a no-op.
# ---------------------------------------------------------------------------


def test_v0_and_v1_slayer_one_shot_differ():
    """V1 (assembled in v1 prompts.py) and V0 (snapshot) must NOT be aliases."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_ONE_SHOT_V0,
    )
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import (
        SLAYER_OTF_ONE_SHOT,
    )

    assert SLAYER_OTF_ONE_SHOT != SLAYER_OTF_ONE_SHOT_V0


def test_v0_and_v1_slayer_ainteract_differ():
    """V1 (assembled in v1 prompts.py) and V0 (snapshot) must NOT be aliases."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        SLAYER_OTF_AINTERACT_V0,
    )
    from bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    assert SLAYER_OTF_AINTERACT != SLAYER_OTF_AINTERACT_V0
