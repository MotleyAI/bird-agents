"""DEV-1555 (CR r1 / outside-diff O1): the v1 ``query`` MCP tool accepts
BOTH a single SlayerQuery object AND a list of stage objects for a
nested DAG — same as SLayer-side. The schema must mirror that or the
SDK rejects valid nested-DAG calls before reaching the implementation,
and the v1 prompt's claim "``query`` (single object OR list of stage
objects for nested DAG)" becomes a lie.
"""

from __future__ import annotations

from unittest import mock

import pytest


@pytest.fixture
def ctx_with_slayer_storage():
    """Populate ``_ctx_var`` with a non-None ``_slayer_storage`` so the
    wrapper skips the lazy ``_slayer_client()`` call. Mirrors the
    pattern in ``tests/test_dev1534_query_wrapper.py``."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    agent_mod._ctx_var.set({
        "_slayer_storage": object(),
    })
    return agent_mod


# ---------------------------------------------------------------------------
# Schema-level contract
# ---------------------------------------------------------------------------


def test_query_tool_schema_required_is_empty():
    """Neither ``source_model`` nor ``queries`` is unconditionally required;
    the wrapper picks the form at runtime."""
    from bird_interact_agents.agents.claude_sdk.agent import query

    schema = query.input_schema
    assert isinstance(schema, dict)
    assert schema.get("required", "missing") == [], (
        "v1 query schema must have empty `required` so EITHER source_model "
        "OR queries can be supplied. "
        f"Got required={schema.get('required')!r}."
    )


def test_query_tool_schema_has_both_source_model_and_queries():
    from bird_interact_agents.agents.claude_sdk.agent import query

    props = query.input_schema.get("properties", {})
    assert "source_model" in props, "missing single-stage `source_model` prop"
    assert "queries" in props, "missing nested-DAG `queries` prop"
    assert props["queries"].get("type") == "array", (
        f"queries prop must be type=array, got {props['queries']!r}"
    )


# ---------------------------------------------------------------------------
# Runtime routing — single-stage path stays unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_single_stage_routes_to_query_impl(
    ctx_with_slayer_storage, monkeypatch,
):
    """Caller passes ``source_model`` → wrapper builds a SlayerQuery
    JSON object from the structured args and calls
    ``_query.query_impl(query_json=<str>, …)`` (the post-DEV-1546
    signature).
    """
    import json as _json

    agent_mod = ctx_with_slayer_storage

    captured: dict = {}

    async def fake_query_impl(query_json, **kwargs):
        captured["query_json"] = query_json
        captured["parsed"] = _json.loads(query_json)
        captured.update(kwargs)
        return "single-stage result"

    async def fake_query_nested_impl(**kwargs):
        captured["unexpected_nested"] = kwargs
        return "should not be called"

    monkeypatch.setattr(agent_mod._query_mod, "attach_storage", lambda _s: None)
    monkeypatch.setattr(agent_mod._query_mod, "query_impl", fake_query_impl)
    monkeypatch.setattr(
        agent_mod._query_mod, "query_nested_impl", fake_query_nested_impl,
    )

    result = await agent_mod.query.handler({
        "source_model": "orders",
        "dimensions": ["status"],
        "measures": ["amount:sum"],
    })

    assert "unexpected_nested" not in captured
    parsed = captured["parsed"]
    assert parsed["source_model"] == "orders"
    assert parsed["dimensions"] == ["status"]
    assert parsed["measures"] == ["amount:sum"]
    assert "single-stage result" in str(result)


# ---------------------------------------------------------------------------
# Runtime routing — nested-DAG path is the new behaviour.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_nested_dag_routes_to_query_nested_impl(
    ctx_with_slayer_storage, monkeypatch,
):
    """Caller passes ``queries`` (array of stage objects) → wrapper calls
    ``query_nested_impl``. This is what the v1 prompt's "single object
    OR list of stage objects" claim requires."""
    agent_mod = ctx_with_slayer_storage

    captured: dict = {}

    async def fake_query_impl(**kwargs):
        captured["unexpected_single"] = kwargs
        return "should not be called"

    async def fake_query_nested_impl(**kwargs):
        captured.update(kwargs)
        return "nested DAG result"

    queries = [
        {"name": "stage1", "source_model": "orders",
         "dimensions": ["status"], "measures": ["amount:sum"]},
        {"source_model": "stage1", "dimensions": ["status"]},
    ]

    monkeypatch.setattr(agent_mod._query_mod, "attach_storage", lambda _s: None)
    monkeypatch.setattr(agent_mod._query_mod, "query_impl", fake_query_impl)
    monkeypatch.setattr(
        agent_mod._query_mod, "query_nested_impl", fake_query_nested_impl,
    )

    result = await agent_mod.query.handler({"queries": queries})

    assert "unexpected_single" not in captured
    assert captured["queries"] == queries
    assert "nested DAG result" in str(result)


# ---------------------------------------------------------------------------
# Defensive guards.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_rejects_both_source_model_and_queries(
    ctx_with_slayer_storage, monkeypatch,
):
    """A caller passing both forms gets a clear error rather than a
    silent precedence rule."""
    agent_mod = ctx_with_slayer_storage
    monkeypatch.setattr(agent_mod._query_mod, "attach_storage", lambda _s: None)

    result = await agent_mod.query.handler({
        "source_model": "orders",
        "queries": [{"source_model": "orders", "dimensions": ["x"]}],
    })

    rendered = str(result)
    assert "either" in rendered.lower() or "not both" in rendered.lower(), (
        f"expected mutual-exclusion error, got {rendered!r}"
    )


@pytest.mark.asyncio
async def test_query_rejects_neither_source_model_nor_queries(
    ctx_with_slayer_storage, monkeypatch,
):
    """An empty call (neither shape) gets a clear single-stage-required
    error — without this the wrapper would have raised a KeyError on
    ``args["source_model"]``."""
    agent_mod = ctx_with_slayer_storage
    monkeypatch.setattr(agent_mod._query_mod, "attach_storage", lambda _s: None)

    result = await agent_mod.query.handler({})

    rendered = str(result)
    assert "source_model" in rendered, (
        f"expected single-stage-required error mentioning source_model, "
        f"got {rendered!r}"
    )
