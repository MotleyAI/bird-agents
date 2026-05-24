"""Verify the BIRD-Interact harness imports and basic helpers work."""

import subprocess
import sys
import textwrap

from bird_interact_agents.config import settings


def test_harness_imports():
    """All harness re-exports import successfully."""
    from bird_interact_agents.harness import (
        execute_env_action,
        execute_submit_action,
        load_db_data_if_needed,
        SampleStatus,
        build_user_encoder_prompt,
        build_user_decoder_prompt,
        parse_encoder_response,
        calculate_budget,
        load_tasks,
    )
    assert callable(execute_env_action)
    assert callable(execute_submit_action)
    assert callable(load_db_data_if_needed)
    assert SampleStatus is not None


def test_load_tasks():
    """Loading mini_interact.jsonl produces task dicts with the expected fields."""
    from bird_interact_agents.harness import load_tasks

    tasks = load_tasks(settings.data_path, limit=3)
    assert len(tasks) == 3
    for t in tasks:
        assert "instance_id" in t
        assert "amb_user_query" in t
        assert "selected_database" in t
        assert "sol_sql" in t
        assert isinstance(t["sol_sql"], list)
        assert len(t["sol_sql"]) > 0  # GT was merged in


def test_calculate_budget():
    """Budget formula: 6 + 2*ambiguities + 2*patience."""
    from bird_interact_agents.harness import calculate_budget, load_tasks

    tasks = load_tasks(settings.data_path, limit=1)
    budget = calculate_budget(tasks[0], patience=3)
    assert isinstance(budget, (int, float))
    assert budget >= 12  # 6 + 0 + 6 minimum


def test_pydantic_ai_agent_imports_without_claude_sdk():
    """The pydantic_ai adapter must not pull in `claude_agent_sdk` at import
    time. Run a child Python with `claude_agent_sdk` masked from sys.modules
    via a meta-path finder, then assert the adapter still imports.

    Regression for the previous `from ...claude_sdk.agent import MAX_MODEL_TURNS`
    edge, which forced every pydantic_ai user to install the Claude SDK extra.
    """
    program = textwrap.dedent(
        """
        import importlib.abc
        import importlib.machinery
        import sys

        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
                    raise ImportError(
                        f"claude_agent_sdk is masked for this isolation test "
                        f"(attempted to import {name})"
                    )
                return None

        sys.meta_path.insert(0, _Block())

        import bird_interact_agents.agents.pydantic_ai.agent  # noqa: F401
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"pydantic_ai adapter import pulled in claude_agent_sdk:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert result.stdout.strip().endswith("ok")
