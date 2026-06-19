"""DEV-1534 Fix C / DEV-1555 CR r1 unification: claude_sdk
`submit_query` exposes the SAME structured shape as `query` — single-
stage via `source_model` + projection fields, or nested-DAG via
`queries`. `normalize_filters` is a separate boolean knob, not embedded
in the payload.

The agent no longer passes a `query_json` JSON STRING; the wrapper
builds the JSON internally from the structured args and forwards it to
`submit_slayer_query`. The opt-out flag is still forwarded to
`normalize_query_payload(parsed, normalize=normalize_filters)`.

`submitted_query` in the trajectory is whatever the wrapper handed
to `submit_slayer_query` (the JSON string it built); `normalize_filters`
never appears inside that JSON.
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
        "source_model": "w",
        "filters": ["category == \"Gadgets\""],
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
        "source_model": "w",
        "filters": ['category == "Gadgets"'],
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
        "source_model": "w",
        "filters": ["category == \"Gadgets\""],
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
        "source_model": "w",
        "filters": ["category == \"Gadgets\""],
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


def test_submit_query_tool_schema_advertises_structured_args():
    """DEV-1555 CR r1 unification: `submit_query` exposes the SAME
    structured shape as `query` — `source_model` plus projection
    fields OR `queries` array. `normalize_filters` is a separate
    boolean knob. The schema's ``required`` is empty (the handler
    gates on source_model XOR queries at runtime)."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    tool = agent_mod.submit_query
    schema = None
    for attr in ("inputSchema", "input_schema", "schema", "args_schema"):
        s = getattr(tool, attr, None)
        if s is not None:
            schema = s
            break
    assert schema is not None, (
        f"submit_query tool exposes no recognised schema attribute "
        f"(checked inputSchema/input_schema/schema/args_schema); cannot "
        f"verify the `normalize_filters` parameter. Tool object: {tool!r}"
    )
    if isinstance(schema, dict) and "properties" in schema:
        props = schema["properties"]
        assert "source_model" in props
        assert "queries" in props
        assert "normalize_filters" in props
        nf = props["normalize_filters"]
        nf_type = nf.get("type") if isinstance(nf, dict) else None
        assert nf_type == "boolean", (
            f"normalize_filters must be declared as boolean in input schema; "
            f"got {nf!r}"
        )
        # `queries` must be an array (nested-DAG list of stage objects).
        qs = props["queries"]
        qs_type = qs.get("type") if isinstance(qs, dict) else None
        assert qs_type == "array"
        # Legacy `query_json` single-string parameter is gone.
        assert "query_json" not in props
        required = schema.get("required", [])
        assert required == [], (
            f"submit_query schema must have empty `required` so EITHER "
            "source_model OR queries can be supplied; the SDK marks "
            "every key required for flat-dict schemas, so the explicit "
            f"empty list is load-bearing. got: {required!r}"
        )
    else:
        pytest.fail(
            f"unrecognised schema shape: {type(schema).__name__} {schema!r}"
        )
