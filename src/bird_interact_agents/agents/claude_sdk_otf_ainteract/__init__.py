"""Claude Agent SDK on-the-fly KB-encode adapter, a-interact flavor (DEV-1507).

Bound to ``--dataset mini_interact --mode a-interact``. Shares the OTF
cache pipeline + slayer write tools with the sibling ``claude_sdk_otf``
flavor, but adds a native ``ask_user`` tool and three hook guards that
enforce a hard ``ask_user``-before-``submit_query`` discipline. See
``agent.py`` for the entry class and ``prompts.py`` for the role prompt.

Public re-export: ``ClaudeSDKOtfAInteractAgent``.
"""

from bird_interact_agents.agents.claude_sdk_otf_ainteract.agent import (
    ClaudeSDKOtfAInteractAgent,
)

__all__ = ["ClaudeSDKOtfAInteractAgent"]
