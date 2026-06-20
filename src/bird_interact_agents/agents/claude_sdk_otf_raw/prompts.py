"""V0 (origin/main) prompts for the raw-SQL OTF agent (one-shot flavor).

Re-exports ``RAW_OTF_ONE_SHOT_V0`` from ``_shared_otf_prompts`` under
its origin/main name (``RAW_OTF_ONE_SHOT``). The prompt body is the
frozen byte-for-byte snapshot of origin/main; SHA pinned in
``tests/test_dev1555_v0_v1_shared_prompts.py``.

Format params: ``budget``, ``db_name``, ``user_query``.
"""

from __future__ import annotations

from bird_interact_agents.agents._shared_otf_prompts import (
    RAW_OTF_ONE_SHOT_V0 as RAW_OTF_ONE_SHOT,
)


__all__ = ["RAW_OTF_ONE_SHOT"]
