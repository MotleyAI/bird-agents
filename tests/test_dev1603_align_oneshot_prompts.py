"""DEV-1603 (one-shot extension) — align raw one-shot with slayer one-shot.

One-shot has no user-sim, so:
- `_GRADER_ZERO_VS_ONE_DIAGNOSTIC` is N/A and stays absent from BOTH modes.
- the ALTERNATIVE-JOIN-PATH PROBE gets a user-sim-free variant
  (`_TABLE_SET_PROBE_ONESHOT`) shared by both one-shot modes; this also fixes
  the latent user-sim wart that slayer one-shot v0 used to carry.
- `_RAW_HOST_PATH_PRINCIPLE` is added to raw one-shot v0 + v1.

Probes are v0-only (v1 stays the curated lean set). Mechanical contracts only.
"""

from __future__ import annotations

import importlib
import pathlib
import string

import pytest

_DATA = pathlib.Path(__file__).parent / "data" / "dev1603"
_SLAYER_GOLDEN = _DATA / "slayer_one_shot_v0.golden.txt"
_RAW_GOLDEN = _DATA / "raw_one_shot_v0.golden.txt"

_SLAYER_FILL = dict(knowledge_label="KB", schema_source="SLayer's schema lookup")
_RAW_FILL = dict(knowledge_label="knowledge definition", schema_source="the schema")
_RUNTIME_FIELDS = {"budget", "db_name", "user_query"}


def _fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def test_oneshot_probe_variant_is_user_sim_free() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import _TABLE_SET_PROBE_ONESHOT

    assert {"knowledge_label", "schema_source"} <= _fields(_TABLE_SET_PROBE_ONESHOT)
    assert "user-sim" not in _TABLE_SET_PROBE_ONESHOT
    assert "ask_user" not in _TABLE_SET_PROBE_ONESHOT
    for fill in (_SLAYER_FILL, _RAW_FILL):
        assert _fields(_TABLE_SET_PROBE_ONESHOT.format(**fill)) == set()


def test_ainteract_probe_still_references_user_sim() -> None:
    # The a-interact constant is unchanged (still asks the user-sim).
    from bird_interact_agents.agents._shared_otf_prompts import _TABLE_SET_PROBE

    assert "user-sim" in _TABLE_SET_PROBE


def test_oneshot_and_ainteract_probes_share_head() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import (
        _TABLE_SET_PROBE,
        _TABLE_SET_PROBE_HEAD,
        _TABLE_SET_PROBE_ONESHOT,
    )

    assert _TABLE_SET_PROBE.startswith(_TABLE_SET_PROBE_HEAD)
    assert _TABLE_SET_PROBE_ONESHOT.startswith(_TABLE_SET_PROBE_HEAD)


def test_slayer_one_shot_v0_uses_clean_probe() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import (
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC,
        _TABLE_SET_PROBE,
        _TABLE_SET_PROBE_ONESHOT,
    )
    from bird_interact_agents.agents.claude_sdk_otf.prompts import SLAYER_OTF_ONE_SHOT

    # clean (user-sim-free) probe present; warty a-interact probe gone
    assert _TABLE_SET_PROBE_ONESHOT.format(**_SLAYER_FILL) in SLAYER_OTF_ONE_SHOT
    assert _TABLE_SET_PROBE.format(**_SLAYER_FILL) not in SLAYER_OTF_ONE_SHOT
    # grader-zero stays absent (no user to ask in one-shot)
    assert (
        _GRADER_ZERO_VS_ONE_DIAGNOSTIC.format(attempt_noun="encoding", apply_verb="encode")
        not in SLAYER_OTF_ONE_SHOT
    )


def test_raw_one_shot_v0_gains_probe_and_host() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import (
        _RAW_HOST_PATH_PRINCIPLE,
        _TABLE_SET_PROBE_ONESHOT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import RAW_OTF_ONE_SHOT

    assert _TABLE_SET_PROBE_ONESHOT.format(**_RAW_FILL) in RAW_OTF_ONE_SHOT
    assert _RAW_HOST_PATH_PRINCIPLE in RAW_OTF_ONE_SHOT
    # ordering: probe before the trailing host principle
    assert RAW_OTF_ONE_SHOT.index(
        _TABLE_SET_PROBE_ONESHOT.format(**_RAW_FILL)
    ) < RAW_OTF_ONE_SHOT.index(_RAW_HOST_PATH_PRINCIPLE)


def test_raw_one_shot_v1_gains_host_only() -> None:
    from bird_interact_agents.agents._shared_otf_prompts import _RAW_HOST_PATH_PRINCIPLE
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import RAW_OTF_ONE_SHOT

    assert _RAW_HOST_PATH_PRINCIPLE in RAW_OTF_ONE_SHOT
    assert "ALTERNATIVE-JOIN-PATH PROBE." not in RAW_OTF_ONE_SHOT


def test_probe_absent_from_one_shot_v1_both_modes() -> None:
    from bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts import (
        RAW_OTF_ONE_SHOT as RAW_V1,
    )
    from bird_interact_agents.agents.claude_sdk_otf_v1.prompts import (
        SLAYER_OTF_ONE_SHOT as SLAYER_V1,
    )

    for prompt in (SLAYER_V1, RAW_V1):
        assert "ALTERNATIVE-JOIN-PATH PROBE." not in prompt


@pytest.mark.parametrize(
    "modpath,name",
    [
        ("claude_sdk_otf.prompts", "SLAYER_OTF_ONE_SHOT"),
        ("claude_sdk_otf_raw.prompts", "RAW_OTF_ONE_SHOT"),
        ("claude_sdk_otf_v1.prompts", "SLAYER_OTF_ONE_SHOT"),
        ("claude_sdk_otf_raw_v1.prompts", "RAW_OTF_ONE_SHOT"),
    ],
)
def test_only_runtime_format_fields(modpath: str, name: str) -> None:
    mod = importlib.import_module(f"bird_interact_agents.agents.{modpath}")
    extra = _fields(getattr(mod, name)) - _RUNTIME_FIELDS
    assert extra == set(), f"{name} has unexpected format fields: {extra}"


def test_raw_one_shot_v0_no_slayer_vocab() -> None:
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import RAW_OTF_ONE_SHOT

    for term in (
        "SLayer", "submit_query", "create_model", "edit_model", "[kb=",
        "mcp__slayer__", "inspect_model", "source_model", "memory:",
    ):
        assert term not in RAW_OTF_ONE_SHOT, f"slayer vocab {term!r} leaked into raw one-shot"


def test_one_shot_v0_goldens() -> None:
    # Assert the PUBLIC re-exports (what the v0 agents actually consume), so a
    # wiring drift in claude_sdk_otf*.prompts is caught alongside the constant.
    from bird_interact_agents.agents.claude_sdk_otf.prompts import (
        SLAYER_OTF_ONE_SHOT,
    )
    from bird_interact_agents.agents.claude_sdk_otf_raw.prompts import (
        RAW_OTF_ONE_SHOT,
    )

    assert SLAYER_OTF_ONE_SHOT == _SLAYER_GOLDEN.read_text()
    assert RAW_OTF_ONE_SHOT == _RAW_GOLDEN.read_text()
