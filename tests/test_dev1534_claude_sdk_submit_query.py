"""DEV-1907: claude_sdk `submit_query` takes the unified `query` argument
(one SlayerQuery object or a multi-stage list) plus a `normalize_filters`
sibling knob, never embedded in the payload.

The wrapper serialises the `query` value to JSON and forwards it to
`submit_slayer_query`; the opt-out flag flows to
`normalize_query_payload(parsed, normalize=normalize_filters)`.
`submitted_query` in the trajectory is that JSON string; `normalize_filters`
never appears inside it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from bird_interact_agents.config import settings
from bird_interact_agents.harness import ACTION_COSTS


def _seed_ctx(monkeypatch, *, remaining_budget=20.0):
    """Mirror tests/test_claude_sdk_tools._seed_ctx."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import SampleStatus

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    agent_mod._ctx_var.set({
        "status": SampleStatus(
            idx=0, original_data=task_data,
            remaining_budget=remaining_budget, total_budget=remaining_budget,
        ),
        "data_path_base": settings.db_path,
        "slayer_storage_dir": "",
        "_slayer_client": None,
        "_slayer_storage": None,
        "result": None,
    })
    return agent_mod


@pytest.mark.asyncio
async def test_submit_query_default_normalizes(monkeypatch):
    """When the tool is invoked WITHOUT `normalize_filters` (the
    default), the underlying compile path normalizes filters."""
    from bird_interact_agents.agents import _submit

    agent_mod = _seed_ctx(monkeypatch)
    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: ("ok", 1.0, True, True, True),
    )
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)

    received = {}
    def fake_sql_sync(payload):
        received["payload"] = payload
        return "SELECT 1"

    agent_mod._ctx_var.get()["_slayer_client"] = SimpleNamespace(sql_sync=fake_sql_sync)
    await agent_mod.submit_query.handler({
        "query": {"source_model": "w", "filters": ["category == \"Gadgets\""]},
    })

    assert received["payload"]["filters"] == ["lower(trim(category)) == 'gadgets'"]


@pytest.mark.asyncio
async def test_submit_query_normalize_filters_false_passthrough(monkeypatch):
    """`{"query_json": ..., "normalize_filters": False}` reaches
    `submit_slayer_query` and the compile-side payload is the agent's
    VERBATIM filters (no lower/trim wrap)."""
    from bird_interact_agents.agents import _submit

    agent_mod = _seed_ctx(monkeypatch)
    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: ("ok", 1.0, True, True, True),
    )
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)

    received = {}
    def fake_sql_sync(payload):
        received["payload"] = payload
        return "SELECT 1"

    agent_mod._ctx_var.get()["_slayer_client"] = SimpleNamespace(sql_sync=fake_sql_sync)
    await agent_mod.submit_query.handler({
        "query": {"source_model": "w", "filters": ['category == "Gadgets"']},
        "normalize_filters": False,
    })

    # Opt-out path: filter literal preserves the verbatim string.
    assert received["payload"]["filters"] == ['category == "Gadgets"']


@pytest.mark.asyncio
async def test_submit_query_normalize_filters_true_explicit(monkeypatch):
    """Explicit `normalize_filters=True` matches the default."""
    from bird_interact_agents.agents import _submit

    agent_mod = _seed_ctx(monkeypatch)
    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: ("ok", 1.0, True, True, True),
    )
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)

    received = {}
    def fake_sql_sync(payload):
        received["payload"] = payload
        return "SELECT 1"

    agent_mod._ctx_var.get()["_slayer_client"] = SimpleNamespace(sql_sync=fake_sql_sync)
    await agent_mod.submit_query.handler({
        "query": {"source_model": "w", "filters": ["category == \"Gadgets\""]},
        "normalize_filters": True,
    })

    assert received["payload"]["filters"] == ["lower(trim(category)) == 'gadgets'"]


@pytest.mark.asyncio
async def test_submit_query_records_built_payload(monkeypatch):
    """`submitted_query` trajectory entry == the JSON the wrapper built
    from the agent's structured args (the wrapper serialises here, not
    the agent). The flag never appears inside that JSON."""
    from bird_interact_agents.agents import _submit
    import json as _json

    agent_mod = _seed_ctx(monkeypatch)
    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: ("ok", 1.0, True, True, True),
    )
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)

    agent_mod._ctx_var.get()["_slayer_client"] = SimpleNamespace(
        sql_sync=lambda p: "SELECT 1",
    )
    await agent_mod.submit_query.handler({
        "query": {"source_model": "w", "filters": ["category == \"Gadgets\""]},
        "normalize_filters": False,
    })
    rec = agent_mod._ctx_var.get().get("result")
    assert rec is not None
    # The wrapper built {"source_model": "w", "filters": [...]} — the
    # `normalize_filters` knob lives OUTSIDE this JSON.
    parsed = _json.loads(rec["submitted_query"])
    assert parsed["source_model"] == "w"
    assert parsed["filters"] == ["category == \"Gadgets\""]
    assert "normalize_filters" not in parsed
    # Budget charged exactly once.
    status = agent_mod._ctx_var.get()["status"]
    assert status.remaining_budget == 20.0 - ACTION_COSTS["submit_query"]


def test_submit_query_tool_schema_is_unified_query_shape():
    """DEV-1907: `submit_query` advertises the unified `query` argument (one
    SlayerQuery object or a multi-stage list) plus a boolean `normalize_filters`
    sibling — no structured `source_model`/`queries`/`query_json` args."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    schema = agent_mod.submit_query.input_schema
    assert isinstance(schema, dict) and "properties" in schema
    props = schema["properties"]
    assert set(props) == {"query", "normalize_filters"}
    nf = props["normalize_filters"]
    assert isinstance(nf, dict) and nf.get("type") == "boolean"
    for legacy in ("source_model", "queries", "query_json"):
        assert legacy not in props
    assert schema.get("required") == ["query"]
