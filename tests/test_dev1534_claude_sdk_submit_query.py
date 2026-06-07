"""DEV-1534 Fix C: claude_sdk `submit_query` tool exposes
`normalize_filters: bool = True` as a SEPARATE tool parameter alongside
`query_json`.

The opt-out flag rides as a separate kwarg, NOT inside the JSON
payload. The flag is forwarded to `submit_slayer_query` which forwards
to `normalize_query_payload(parsed, normalize=normalize_filters)`.

The trajectory's `submitted_query` MUST be the agent's ORIGINAL
`query_json` string, byte-for-byte; the kwarg lives outside the JSON
so it never contaminates the recorded DSL.
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
    qjson = '{"source_model": "w", "filters": ["category == \\"Gadgets\\""]}'
    await agent_mod.submit_query.handler({"query_json": qjson})

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
    qjson = '{"source_model": "w", "filters": ["category == \\"Gadgets\\""]}'
    await agent_mod.submit_query.handler(
        {"query_json": qjson, "normalize_filters": False}
    )

    # Opt-out path: filter literal inside the JSON used double-quotes;
    # after json.loads + deep-copy passthrough, they're preserved.
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
    qjson = '{"source_model": "w", "filters": ["category == \\"Gadgets\\""]}'
    await agent_mod.submit_query.handler(
        {"query_json": qjson, "normalize_filters": True}
    )

    assert received["payload"]["filters"] == ["lower(trim(category)) == 'gadgets'"]


@pytest.mark.asyncio
async def test_submit_query_records_unmodified_query_json(monkeypatch):
    """`submitted_query` trajectory entry == original `query_json`,
    NEVER carrying the `normalize_filters` flag (it lives outside the
    JSON)."""
    from bird_interact_agents.agents import _submit

    agent_mod = _seed_ctx(monkeypatch)
    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: ("ok", 1.0, True, True, True),
    )
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)

    agent_mod._ctx_var.get()["_slayer_client"] = SimpleNamespace(
        sql_sync=lambda p: "SELECT 1",
    )
    qjson = '{"source_model": "w", "filters": ["category == \\"Gadgets\\""]}'
    await agent_mod.submit_query.handler(
        {"query_json": qjson, "normalize_filters": False}
    )
    rec = agent_mod._ctx_var.get().get("result")
    assert rec is not None
    assert rec["submitted_query"] == qjson
    # And: budget charged exactly once (normalize_filters does not
    # change pre-eval cost semantics).
    status = agent_mod._ctx_var.get()["status"]
    # remaining = total - submit_query cost
    assert status.remaining_budget == 20.0 - ACTION_COSTS["submit_query"]


def test_submit_query_tool_schema_advertises_normalize_filters_as_bool():
    """The Claude SDK `@tool` decorator's INPUT SCHEMA must declare
    `normalize_filters` as a separate parameter (NOT just mentioned in
    a description blurb). Plan: separate tool parameter alongside
    `query_json`. A description-only mention would mean the SDK doesn't
    introspect it as a real argument, defeating the purpose."""
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
    # Post-PR-review (CodeRabbit/Codex): the schema MUST be an explicit
    # JSON Schema dict declaring only `query_json` as required. A flat
    # `{key: type}` schema would make the SDK convert it to
    # `required: list(properties.keys())` (see claude_agent_sdk
    # `_build_schema`), forcing every caller to supply `normalize_filters`
    # despite the handler defaulting it to True.
    if isinstance(schema, dict) and "properties" in schema:
        props = schema["properties"]
        assert "query_json" in props
        assert "normalize_filters" in props
        nf = props["normalize_filters"]
        nf_type = nf.get("type") if isinstance(nf, dict) else None
        assert nf_type == "boolean", (
            f"normalize_filters must be declared as boolean in input schema; "
            f"got {nf!r}"
        )
        qj = props["query_json"]
        qj_type = qj.get("type") if isinstance(qj, dict) else None
        assert qj_type == "string"
        required = schema.get("required", [])
        assert required == ["query_json"], (
            f"submit_query schema must mark ONLY `query_json` as required "
            f"(the SDK marks every key required for flat-dict schemas, so "
            f"the explicit list is load-bearing). got: {required!r}"
        )
    else:
        pytest.fail(
            f"unrecognised schema shape: {type(schema).__name__} {schema!r}"
        )
