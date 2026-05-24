"""Smoke test for the upstream submodule contract.

Path B integration: we don't import upstream's `SARAgent` class. We only
need (a) the BIRD prompt template, (b) `db_interface.get_function_call_bird`
(consumed inside our audit loop's tool schemas — currently inlined, but
this test guards against the upstream schema drifting).

Skipped when the submodule is not initialised; required in any environment
that runs the pilot.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_TEMPLATE_FILE = REPO_ROOT / "third_party" / "sar_agent" / "SAR-Agent" / "prompts" / "prompt_user_bird.txt"
DB_INTERFACE_FILE = REPO_ROOT / "third_party" / "sar_agent" / "SAR-Agent" / "db_interface.py"


@pytest.mark.skipif(
    not PROMPT_TEMPLATE_FILE.exists(),
    reason="SAR-Agent submodule not initialised (run `git submodule update --init`)",
)
def test_upstream_template_present_and_has_placeholders():
    assert PROMPT_TEMPLATE_FILE.exists()
    text = PROMPT_TEMPLATE_FILE.read_text()
    # We .format() this template — these placeholders MUST exist.
    for placeholder in ("{question}", "{schema}", "{external_knowledge}", "{gold_query}"):
        assert placeholder in text, (
            f"upstream prompt template lost placeholder {placeholder}; "
            f"our `prompt_wrapper._format_template` will KeyError"
        )


@pytest.mark.skipif(
    not PROMPT_TEMPLATE_FILE.exists(),
    reason="SAR-Agent submodule not initialised",
)
def test_load_prompt_template_returns_upstream_text():
    from bird_interact_agents.sar_audit import _upstream_import

    text = _upstream_import.load_prompt_template()
    assert "{question}" in text
    assert "{schema}" in text
    assert "Analyze Result Format" in text or "Correctness" in text


@pytest.mark.skipif(
    not DB_INTERFACE_FILE.exists(),
    reason="SAR-Agent submodule not initialised",
)
def test_upstream_db_interface_defines_expected_tool_names():
    """Sanity-check upstream's `db_interface.py` still references the
    same two tool names our audit loop uses (`read_sqlite_query` +
    `terminate`). We inline our own Anthropic-shape schemas (upstream's
    are OpenAI-shape and the module imports snowflake at top-level so we
    can't safely import it), but we DO want a drift detector for the
    tool names themselves.
    """
    text = DB_INTERFACE_FILE.read_text()
    assert "read_sqlite_query" in text
    assert "terminate" in text
