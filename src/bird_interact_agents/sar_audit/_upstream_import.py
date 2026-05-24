"""Locate upstream SAR-Agent artefacts (prompt template + tool-schema getter).

Path B integration: we don't import upstream's `SARAgent` class. Instead we
reuse its BIRD prompt template (`prompts/prompt_user_bird.txt`) verbatim and
its function/tool schemas (`get_function_call_bird()`), then drive the loop
ourselves via the Anthropic SDK.
"""

from __future__ import annotations

from pathlib import Path

# Submodules live in each worktree separately, so we resolve via the
# package's own __file__ rather than `paths.main_checkout_root()` (which
# returns the canonical checkout). `parents[3]` walks
# `_upstream_import.py` → sar_audit → bird_interact_agents → src → worktree.
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
_SAR_AGENT_DIR = _WORKTREE_ROOT / "third_party" / "sar_agent" / "SAR-Agent"
PROMPT_TEMPLATE_PATH = _SAR_AGENT_DIR / "prompts" / "prompt_user_bird.txt"


def load_prompt_template() -> str:
    """Return the verbatim upstream BIRD prompt template (with `{...}` placeholders)."""
    if not PROMPT_TEMPLATE_PATH.exists():
        raise RuntimeError(
            "SAR-Agent submodule not initialised: "
            "run `git submodule update --init third_party/sar_agent`"
        )
    return PROMPT_TEMPLATE_PATH.read_text()
