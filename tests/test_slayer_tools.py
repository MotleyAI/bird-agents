"""Verify the SLayer-mode native tools (only `submit_query` remains; the
discovery tools come from the actual `slayer mcp` server, not us)."""

import json
import shutil
from pathlib import Path

import pytest

from bird_interact_agents.config import settings


def _tmp_models_copy(tmp_path, db_name: str) -> str:
    """Copy the committed ``slayer_models/<db>/`` into a tmp dir and return
    its path. Opening a ``YAMLStorage`` on a committed dir lazily creates an
    empty ``embeddings.db`` sidecar there; operating on a copy keeps the
    committed tree pristine."""
    src = Path(__file__).resolve().parent.parent / "slayer_models" / db_name
    dst = tmp_path / db_name
    shutil.copytree(src, dst)
    return str(dst)


@pytest.mark.asyncio
async def test_submit_query_tool_with_valid_slayer_query(tmp_path):
    """`submit_query` translates a SLayer query JSON to SQL and submits it."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import (
        SampleStatus,
        load_db_data_if_needed,
        load_tasks,
    )

    # Pick a task whose database actually exposes the SLayer model the
    # hard-coded query below uses. Don't rely on `tasks[0]` — fixture order
    # is not part of the contract and the test would otherwise fail for
    # unrelated reasons if `mini_interact.jsonl` is reshuffled.
    target_db = "alien"  # has the `observatories` SLayer model in slayer_models
    all_tasks = load_tasks(settings.data_path)
    task = next((t for t in all_tasks if t["selected_database"] == target_db), None)
    assert task is not None, f"No task found for db={target_db}"
    db_name = task["selected_database"]
    load_db_data_if_needed(db_name, settings.db_path)

    agent_mod._ctx_var.set({
        "status": SampleStatus(idx=0, original_data=task),
        "data_path_base": settings.db_path,
        "slayer_storage_dir": _tmp_models_copy(tmp_path, db_name),
        "_slayer_client": None,
        "_slayer_storage": None,
        "result": None,
    })

    # Trivial valid SLayer query — exercises sql_sync + execute_submit_action.
    # Likely won't match the gold answer but should not error during translate.
    query = json.dumps({
        "source_model": "observatories",
        "dimensions": ["observstation"],
        "limit": 1,
    })
    result = await agent_mod.submit_query.handler({"query_json": query})
    text = result["content"][0]["text"]
    assert "Generated SQL:" in text
    assert "SELECT" in text


@pytest.mark.asyncio
async def test_submit_query_tool_with_invalid_json(tmp_path):
    """`submit_query` rejects invalid JSON cleanly."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import (
        SampleStatus,
        load_db_data_if_needed,
    )

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    load_db_data_if_needed("alien", settings.db_path)

    agent_mod._ctx_var.set({
        "status": SampleStatus(idx=0, original_data=task_data),
        "data_path_base": settings.db_path,
        "slayer_storage_dir": _tmp_models_copy(tmp_path, "alien"),
        "_slayer_client": None,
        "_slayer_storage": None,
        "result": None,
    })

    result = await agent_mod.submit_query.handler({"query_json": "not json"})
    text = result["content"][0]["text"]
    assert "Invalid JSON" in text or "submission aborted" in text


def test_otf_slayer_one_shot_tool_list_includes_dev1534_wrappers():
    """DEV-1534 Fix C migration target for the (now-deleted) SLAYER_A_TOOLS
    assertion: the production OTF slayer one-shot agent must register
    `query` AND `query_nested` on bird-interact-tools (not allowlist
    SLayer's subprocess `query`/`query_nested`) so the agent can pass
    `normalize_filters=false` mid-flight."""
    from bird_interact_agents.agents.claude_sdk_otf import agent as otf_mod

    knowledge_names = {t.name for t in otf_mod._KNOWLEDGE_TOOLS}
    assert "query" in knowledge_names
    assert "query_nested" in knowledge_names
    # The slayer subprocess allowlist must NOT carry query/query_nested
    # (they're served by our wrappers now).
    assert "query" not in otf_mod.SLAYER_MCP_TOOLS
    assert "query_nested" not in otf_mod.SLAYER_MCP_TOOLS


def test_otf_slayer_ainteract_tool_list_includes_dev1534_wrappers():
    """DEV-1534 Fix C migration target for the (now-deleted) SLAYER_C_TOOLS
    assertion: same contract for the OTF slayer a-interact agent."""
    from bird_interact_agents.agents.claude_sdk_otf_ainteract import (
        agent as otf_mod,
    )

    knowledge_names = {t.name for t in otf_mod._KNOWLEDGE_TOOLS}
    assert "query" in knowledge_names
    assert "query_nested" in knowledge_names
