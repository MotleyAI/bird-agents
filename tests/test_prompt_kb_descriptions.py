"""Verify the c-interact prompt builder renders KB descriptions inline.

Step-7 follow-up to DEV-1362: after the description-refresh script
populates each KB-bearing entity's `description` with a canonical
[kb=<id>]…[/kb=<id>] block, the prompt builder should surface those
descriptions inline next to each entity name so the c-interact agent
sees the KB context without calling `inspect_model`.

This test exercises the v3-shape exports under
`bird-interact-agents/slayer_models/<db>/` directly, side-stepping the
v1-shape `slayer_storage/<db>/` directory used by the cross-framework
parity tests (whose v1↔v3 collisions are a separate issue).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SLAYER_MODELS_ROOT = REPO_ROOT / "slayer_models"


@pytest.mark.asyncio
async def test_prompt_renders_one_line_per_model_with_columns_and_measures():
    """The new format is multi-line per model: name + indented columns/measures."""
    from bird_interact_agents.agents import _prompt_builders as pb

    db = "polar"
    storage_dir = str(SLAYER_MODELS_ROOT / db)

    with patch.object(pb, "_filter_knowledge_for_agent", return_value={}), \
         patch.object(pb, "_schema_cache", {db: ""}):
        prompt = await pb.build_slayer_c_interact_prompt(
            budget=2.0,
            user_query="dummy",
            slayer_storage_dir=storage_dir,
            db_name=db,
            task_data={},
        )

    # Each model gets a top-level "- <name>" line; columns/measures live
    # under indented "columns:" / "measures:" sub-sections.
    assert "\n- equipment" in prompt or prompt.startswith("- equipment")
    assert "    columns:" in prompt


@pytest.mark.asyncio
async def test_prompt_includes_known_kb_description_inline():
    """A KB-bearing entity's description should appear inline."""
    from bird_interact_agents.agents import _prompt_builders as pb

    db = "polar"
    storage_dir = str(SLAYER_MODELS_ROOT / db)

    # `extreme_weather_ready` is a polar model carrying meta.kb_id=10
    # after Step 2's redirect; its description mentions "Extreme Weather"
    # in either the W4c-era prose or the post-refresh canonical block.
    from slayer.storage.yaml_storage import YAMLStorage
    storage = YAMLStorage(base_dir=storage_dir)
    target = await storage.get_model("extreme_weather_ready")
    assert target is not None and target.description, (
        "test fixture broken: polar.extreme_weather_ready has no description"
    )
    expected_token = "Extreme Weather"
    assert expected_token in target.description, (
        "test fixture broken: extreme_weather_ready.description doesn't "
        "mention 'Extreme Weather' anymore"
    )

    with patch.object(pb, "_filter_knowledge_for_agent", return_value={}), \
         patch.object(pb, "_schema_cache", {db: ""}):
        prompt = await pb.build_slayer_c_interact_prompt(
            budget=2.0,
            user_query="dummy",
            slayer_storage_dir=storage_dir,
            db_name=db,
            task_data={},
        )

    # The description should appear in the rendered prompt (possibly
    # truncated, but the leading 'Extreme Weather' phrase is well within
    # the truncation budget).
    assert expected_token in prompt, (
        "extreme_weather_ready description didn't make it into the rendered "
        "c-interact prompt"
    )


@pytest.mark.asyncio
async def test_truncation_caps_long_descriptions():
    """Per-entity descriptions are capped to keep the prompt manageable."""
    from bird_interact_agents.agents import _prompt_builders as pb

    long_desc = "x" * 1000
    short = pb._short(long_desc)
    assert len(short) <= pb._DESCRIPTION_TRUNCATE_AT + 1  # +1 for the '…'
    assert short.endswith("…")
