"""Oracle mode: submit ground-truth SQL and verify it passes evaluation."""

import pytest

from bird_interact_agents.config import settings


@pytest.mark.asyncio
async def test_oracle_single_task():
    """Submitting ground-truth SQL for one task should pass phase 1."""
    from bird_interact_agents.harness import load_tasks
    from bird_interact_agents.run import run_oracle_task

    tasks = load_tasks(settings.data_path, limit=1)
    result = await run_oracle_task(tasks[0], settings.db_path)
    assert result["phase1_passed"] is True
    assert result["total_reward"] > 0


@pytest.mark.asyncio
async def test_oracle_multiple_tasks():
    """Ground-truth SQL should pass for the first 5 tasks."""
    from bird_interact_agents.harness import load_tasks
    from bird_interact_agents.run import run_oracle_task

    tasks = load_tasks(settings.data_path, limit=5)
    failures = []
    for task in tasks:
        result = await run_oracle_task(task, settings.db_path)
        if not result["phase1_passed"]:
            failures.append(task["instance_id"])
    assert not failures, f"Oracle failed for: {failures}"
