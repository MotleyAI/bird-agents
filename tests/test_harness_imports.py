"""Verify the BIRD-Interact harness imports and basic helpers work."""

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


def test_pydantic_ai_agent_imports_without_claude_sdk(
    import_isolation_results,
):
    """The pydantic_ai adapter must not pull in `claude_agent_sdk` at import
    time. Regression for the previous
    `from ...claude_sdk.agent import MAX_MODEL_TURNS` edge, which forced
    every pydantic_ai user to install the Claude SDK extra.

    Drives the consolidated `import_isolation_results` subprocess fixture
    (DEV-1508 perf) — see ``tests/conftest.py`` for the rationale; this
    test plus the two sibling ``test_import_does_not_pull_pydantic_ai_
    adapter_packages`` tests previously each spawned a fresh Python
    interpreter (~5-7s × 3) just to inspect ``sys.modules``.
    """
    r = import_isolation_results["pydantic_ai_without_claude_sdk"]
    assert r["ok"], (
        f"pydantic_ai adapter import failed under claude_agent_sdk mask: "
        f"{r.get('error')}"
    )
