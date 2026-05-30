"""Claude Agent SDK on-the-fly KB-encode adapter, one-shot flavor (DEV-1505).

After DEV-1507 this package is bound to ``--dataset livesqlbench --mode
one-shot``. The mini-interact / a-interact flavor lives in the sibling
``claude_sdk_otf_ainteract`` package.

A single Claude-SDK agent (no forced stages, no recursion) that encodes
the relevant knowledge-base items into the per-task SLayer store off the
deterministic OTF cache, then queries the named entities it created
instead of inlining everything. See ``agent.py`` for the entry class and
``prompts.py`` for the role prompt.

Public re-export: ``ClaudeSDKOtfAgent``.
"""

from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent

__all__ = ["ClaudeSDKOtfAgent"]
