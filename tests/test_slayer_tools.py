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


def test_slayer_a_tools_include_knowledge_for_parity():
    """SLAYER_A_TOOLS exposes the bird-interact knowledge tools so SLayer
    agents have the same access to external domain knowledge that raw
    agents do (slayer MCP handles SLayer schema discovery)."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    names = {t.name for t in agent_mod.SLAYER_A_TOOLS}
    assert names == {
        "ask_user",
        "submit_query",
        "get_all_external_knowledge_names",
        "get_knowledge_definition",
        "get_all_knowledge_definitions",
    }


def test_slayer_c_tools_only_native():
    """SLAYER_C_TOOLS stays minimal — knowledge is injected upfront in the
    c-interact prompt, no fetch tool needed."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    names = {t.name for t in agent_mod.SLAYER_C_TOOLS}
    assert names == {"ask_user", "submit_query"}
