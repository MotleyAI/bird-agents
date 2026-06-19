"""V0 (origin/main) prompts for the one-shot SLayer OTF agent.

This module re-exports ``SLAYER_OTF_ONE_SHOT_V0`` from
``_shared_otf_prompts`` under its origin/main name
(``SLAYER_OTF_ONE_SHOT``). The actual prompt body lives in
``_shared_otf_prompts.py`` as a frozen byte-for-byte snapshot of
origin/main (pinned by SHA in ``tests/test_dev1555_v0_v1_shared_prompts.py``).

Format params: ``budget``, ``db_name``, ``user_query`` — substituted by
the v0 agent when building the system prompt.
"""

from __future__ import annotations

from bird_interact_agents.agents._shared_otf_prompts import (
    SLAYER_OTF_ONE_SHOT_V0 as SLAYER_OTF_ONE_SHOT,
)


__all__ = ["SLAYER_OTF_ONE_SHOT"]
