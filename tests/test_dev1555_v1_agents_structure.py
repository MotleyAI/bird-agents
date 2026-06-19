"""DEV-1555 v0/v1 split — v1 agent packages exist and are self-contained.

The four v1 packages (``claude_sdk_otf_v1``, ``claude_sdk_otf_raw_v1``,
``claude_sdk_otf_ainteract_v1``, ``claude_sdk_otf_ainteract_raw_v1``)
carry this branch's improvements. Their sibling-inheritance follows the
v1 root (NOT the v0 root).
"""

from __future__ import annotations

import importlib
import re

import pytest


_V1_AGENT_PACKAGES = (
    "bird_interact_agents.agents.claude_sdk_otf_v1",
    "bird_interact_agents.agents.claude_sdk_otf_raw_v1",
    "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1",
    "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1",
)


def _agent_source(package: str) -> str:
    mod = importlib.import_module(f"{package}.agent")
    path = mod.__file__
    assert path is not None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize(
    "pkg,cls_name",
    [
        ("bird_interact_agents.agents.claude_sdk_otf_v1", "ClaudeSDKOtfAgent"),
        (
            "bird_interact_agents.agents.claude_sdk_otf_raw_v1",
            "ClaudeSDKOtfRawAgent",
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1",
            "ClaudeSDKOtfAInteractAgent",
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1",
            "ClaudeSDKOtfAInteractRawAgent",
        ),
    ],
)
def test_v1_package_exports_class(pkg: str, cls_name: str):
    """Each v1 package re-exports its agent class."""
    mod = importlib.import_module(pkg)
    assert hasattr(mod, cls_name), (
        f"{pkg} does not export {cls_name} "
        f"(public: {[n for n in dir(mod) if not n.startswith('_')]})"
    )


@pytest.mark.parametrize(
    "pkg",
    [
        "bird_interact_agents.agents.claude_sdk_otf_raw_v1",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1",
    ],
)
def test_v1_siblings_import_from_v1_root(pkg: str):
    """v1 raw/ainteract/ainteract_raw import from ``claude_sdk_otf_v1.agent``."""
    src = _agent_source(pkg)
    assert re.search(
        r"from\s+bird_interact_agents\.agents\.claude_sdk_otf_v1\.agent\s+import\b",
        src,
        re.DOTALL,
    ), (
        f"{pkg}/agent.py must import from claude_sdk_otf_v1.agent (the v1 root)."
    )


@pytest.mark.parametrize(
    "pkg",
    [
        "bird_interact_agents.agents.claude_sdk_otf_raw_v1",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1",
    ],
)
def test_v1_siblings_do_not_import_v0_root(pkg: str):
    """v1 siblings must NOT cross-import the unsuffixed v0 root."""
    src = _agent_source(pkg)
    # Use a word boundary that excludes `_v1` suffix at the end.
    pattern = re.compile(
        r"from\s+bird_interact_agents\.agents\.claude_sdk_otf\.agent\s+import\b",
        re.DOTALL,
    )
    assert not pattern.search(src), (
        f"{pkg}/agent.py imports from the v0 root claude_sdk_otf.agent; "
        "v1 siblings must import from claude_sdk_otf_v1.agent. This "
        "would silently make v1 inherit v0 behaviour."
    )


@pytest.mark.parametrize(
    "pkg,prompts_const",
    [
        (
            "bird_interact_agents.agents.claude_sdk_otf_v1.prompts",
            "SLAYER_OTF_ONE_SHOT",
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_raw_v1.prompts",
            "RAW_OTF_ONE_SHOT",
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.prompts",
            "SLAYER_OTF_AINTERACT",
        ),
        (
            "bird_interact_agents.agents.claude_sdk_otf_ainteract_raw_v1.prompts",
            "RAW_OTF_AINTERACT",
        ),
    ],
)
def test_v1_prompts_export_unsuffixed_constant(pkg: str, prompts_const: str):
    """v1 prompts module exports the unsuffixed (current-branch) constant."""
    mod = importlib.import_module(pkg)
    assert hasattr(mod, prompts_const), (
        f"{pkg} must export {prompts_const} (the v1 / current-branch text)."
    )
    val = getattr(mod, prompts_const)
    assert isinstance(val, str) and val.strip()
