"""DEV-1649: the SAVE hook must be wired into every on-the-fly slayer interact
agent (scope D9). This is a mechanical wiring contract — each agent's module
must reference ``maybe_save_edited_models`` at its success return — NOT a
behavioural assertion (the helper's behaviour is covered in test_edited_models).

Mirrors how other structural-wiring contracts are pinned; it guards against the
feature silently not firing for one of the five agents.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

_AGENT_MODULES = [
    "bird_interact_agents.agents.claude_sdk_otf.agent",
    "bird_interact_agents.agents.claude_sdk_otf_v1.agent",
    "bird_interact_agents.agents.claude_sdk_otf_ainteract.agent",
    "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent",
    "bird_interact_agents.agents.pydantic_ai_recursive.agent",
]


@pytest.mark.parametrize("modname", _AGENT_MODULES)
def test_agent_wires_edited_models_save(modname):
    """Each agent routes its success return through the shared
    ``finalize_with_edited_models_save`` hook (which calls
    ``maybe_save_edited_models``), so --save-edited-models actually persists
    the edited store for every on-the-fly slayer interact agent."""
    mod = importlib.import_module(modname)
    src = inspect.getsource(mod)
    assert "finalize_with_edited_models_save" in src, (
        f"{modname} must route its success return through "
        f"finalize_with_edited_models_save so --save-edited-models persists "
        f"the edited store."
    )
