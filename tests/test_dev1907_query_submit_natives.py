"""DEV-1907: the claude_sdk `query` / `submit_query` natives adopt SLayer
0.10.2's unified `query` argument (a query object, a multi-stage list, or a
run-by-name string) plus `normalize_filters` as a sibling arg — no
source_model/measures/… top-level args, no `query_nested` native.

Covers spec `slayer-agent-tool-surface`:
  * "Unified query argument" (single object / multi-stage list / computed dim)
  * "Submission accepts the unified query shape"
  * "Filter normalisation is a sibling argument"

Imports of `claude_sdk.agent` are deferred into each test: under 0.10.2 the
un-migrated agent still imports removed SLayer symbols, so these fail (feature
missing: import-compat + native reshape) until the change lands.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import jsonschema
import pytest

# Representative payloads, valid against SLayer 0.10.2's advertised `query`
# schema (measures/dimensions are object forms; month(col) is a computed dim).
_SINGLE_OBJECT = {
    "source_model": "orders",
    "dimensions": [{"name": "status"}],
    "measures": [{"formula": "amount:sum"}],
}
_COMPUTED_DIMENSION = {
    "source_model": "orders",
    "dimensions": [{"expression": "revenue:sum partition_by region", "name": "region_total"}],
    "measures": [{"formula": "amount:sum"}],
}
_INLINE_MODEL_EXTENSION = {
    "source_model": {"source_name": "orders", "columns": [{"name": "big", "sql": "amount*2"}]},
    "measures": [{"formula": "big:sum"}],
}
_MULTI_STAGE_LIST = [
    {"name": "monthly", "source_model": "orders", "measures": [{"formula": "amount:sum"}]},
    {"source_model": "monthly", "measures": [{"formula": "amount:avg"}]},
]
_MONTH_GRANULARITY = {
    "source_model": "orders",
    "dimensions": [{"expression": "month(created_at)", "name": "m"}],
}
_REPRESENTATIVE = {
    "single_object": _SINGLE_OBJECT,
    "computed_dimension": _COMPUTED_DIMENSION,
    "inline_model_extension": _INLINE_MODEL_EXTENSION,
    "multi_stage_list": _MULTI_STAGE_LIST,
    "month_granularity": _MONTH_GRANULARITY,
    "run_by_name": "orders",
}

_LEGACY_TOP_LEVEL = ("source_model", "measures", "dimensions", "filters", "queries")


def _agent():
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    return agent_mod


@pytest.fixture
def ctx_with_storage(monkeypatch):
    """Populate `_ctx_var` so the `query` wrapper skips the lazy client attach.

    Mirrors tests/test_dev1555_query_unified_schema.py::ctx_with_slayer_storage.
    """
    agent_mod = _agent()
    agent_mod._ctx_var.set({"_slayer_storage": object()})
    monkeypatch.setattr(agent_mod._query_mod, "attach_storage", lambda _s: None)
    return agent_mod


def _seed_submit_ctx(agent_mod):
    from bird_interact_agents.harness import SampleStatus

    agent_mod._ctx_var.set({
        "status": SampleStatus(
            idx=0,
            original_data={"selected_database": "alien", "knowledge_ambiguity": [],
                           "instance_id": "alien_1"},
            remaining_budget=20.0, total_budget=20.0,
        ),
        "data_path_base": "/tmp/ignored", "slayer_storage_dir": "",
        "_slayer_client": None, "_slayer_storage": None, "result": None,
    })


def _forwarded_query(captured: dict):
    """The value the wrapper forwarded as SLayer's single `query` argument,
    whether passed positionally or by keyword."""
    if "query" in captured.get("kwargs", {}):
        return captured["kwargs"]["query"]
    args = captured.get("args") or ()
    return args[0] if args else None


# ===========================================================================
# Schema-level contract — the native schema IS SLayer's `query` schema + the
# `normalize_filters` sibling arg (no structured top-level query args).
# ===========================================================================


def test_query_native_schema_is_slayer_query_plus_normalize_filters():
    agent_mod = _agent()
    _desc, slayer_params = agent_mod._slayer_tool_metadata("query")
    schema = agent_mod.query.input_schema

    assert set(schema["properties"]) == set(slayer_params["properties"]) | {"normalize_filters"}
    assert schema.get("required") == ["query"]
    assert schema["properties"]["query"] == slayer_params["properties"]["query"]
    assert "$defs" in schema, "native must carry SLayer's $defs so query $refs resolve"
    for legacy in _LEGACY_TOP_LEVEL:
        assert legacy not in schema["properties"], f"legacy top-level arg {legacy!r} still advertised"


def test_query_native_schema_is_valid_json_schema():
    agent_mod = _agent()
    jsonschema.Draft202012Validator.check_schema(agent_mod.query.input_schema)


def test_query_native_schema_accepts_representative_payloads():
    agent_mod = _agent()
    validator = jsonschema.Draft202012Validator(agent_mod.query.input_schema)
    for name, payload in _REPRESENTATIVE.items():
        errs = sorted(validator.iter_errors({"query": payload}), key=lambda e: list(e.path))
        assert not errs, f"payload {name!r} rejected: {errs[0].message if errs else ''}"


def test_submit_query_native_schema_is_query_and_normalize_only():
    agent_mod = _agent()
    schema = agent_mod.submit_query.input_schema
    assert set(schema["properties"]) == {"query", "normalize_filters"}
    assert schema.get("required") == ["query"]
    jsonschema.Draft202012Validator.check_schema(schema)


def test_submit_query_native_schema_accepts_object_and_list():
    agent_mod = _agent()
    validator = jsonschema.Draft202012Validator(agent_mod.submit_query.input_schema)
    for payload in (_SINGLE_OBJECT, _MULTI_STAGE_LIST):
        errs = list(validator.iter_errors({"query": payload}))
        assert not errs, f"submit_query rejected {payload!r}: {errs[0].message if errs else ''}"


# ===========================================================================
# Runtime routing — `query` executes THROUGH SLayer (via the real query_impl,
# mocking only SLayer's own `query` fn) and returns SLayer's formatted result.
# ===========================================================================


def _patch_slayer_query_fn(agent_mod, monkeypatch, captured, ret="SLAYER_RESULT"):
    async def fake_fn(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ret

    monkeypatch.setattr(agent_mod._query_mod, "_get_slayer_query_fn", lambda: fake_fn)


async def test_query_single_object_executes_through_slayer(ctx_with_storage, monkeypatch):
    agent_mod = ctx_with_storage
    captured: dict = {}
    _patch_slayer_query_fn(agent_mod, monkeypatch, captured)
    result = await agent_mod.query.handler({"query": _SINGLE_OBJECT})
    assert _forwarded_query(captured) == _SINGLE_OBJECT
    assert "SLAYER_RESULT" in str(result)


async def test_query_multi_stage_list_executes_without_a_nested_tool(ctx_with_storage, monkeypatch):
    agent_mod = ctx_with_storage
    captured: dict = {}
    _patch_slayer_query_fn(agent_mod, monkeypatch, captured)
    result = await agent_mod.query.handler({"query": _MULTI_STAGE_LIST})
    assert _forwarded_query(captured) == _MULTI_STAGE_LIST
    assert "SLAYER_RESULT" in str(result)


async def test_query_computed_dimension_runs_as_a_single_query(ctx_with_storage, monkeypatch):
    """A single object with a computed (re-aggregating) dimension executes as
    one query, not a forced multi-stage list."""
    agent_mod = ctx_with_storage
    captured: dict = {}
    _patch_slayer_query_fn(agent_mod, monkeypatch, captured)
    result = await agent_mod.query.handler({"query": _COMPUTED_DIMENSION})
    forwarded = _forwarded_query(captured)
    assert isinstance(forwarded, dict), "computed-dim query was split into a multi-stage list"
    assert forwarded == _COMPUTED_DIMENSION
    assert "SLAYER_RESULT" in str(result)


# ===========================================================================
# Filter normalisation is a sibling argument (lift from a single object;
# strip — never lift — from list stages).
# ===========================================================================


async def test_misplaced_tool_key_in_single_object_is_lifted(ctx_with_storage, monkeypatch):
    agent_mod = ctx_with_storage
    captured: dict = {}

    async def fake_fn(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "OK"

    monkeypatch.setattr(agent_mod._query_mod, "_get_slayer_query_fn", lambda: fake_fn)
    await agent_mod.query.handler({"query": {"source_model": "orders", "show_sql": True}})

    assert captured["kwargs"].get("show_sql") is True, "misplaced show_sql not lifted to tool level"
    fwd = _forwarded_query(captured)
    assert isinstance(fwd, dict) and "show_sql" not in fwd, "show_sql not stripped from query object"


async def test_misplaced_tool_key_in_list_stage_is_stripped_not_lifted(ctx_with_storage, monkeypatch):
    agent_mod = ctx_with_storage
    captured: dict = {}

    async def fake_fn(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "OK"

    monkeypatch.setattr(agent_mod._query_mod, "_get_slayer_query_fn", lambda: fake_fn)
    stages = [
        {"name": "s1", "source_model": "orders", "measures": [{"formula": "amount:sum"}], "show_sql": True},
        {"source_model": "s1", "measures": [{"formula": "amount:avg"}]},
    ]
    await agent_mod.query.handler({"query": stages})

    fwd = _forwarded_query(captured)
    assert isinstance(fwd, list)
    for stage in fwd:
        assert "show_sql" not in stage, "stray tool-level key not stripped from list stage"
    assert captured["kwargs"].get("show_sql", False) is False, "list-stage key wrongly lifted to top level"


# ===========================================================================
# submit_query accepts the unified shape, compiles, and records the payload.
# ===========================================================================


async def test_submit_single_object_forwards_payload(monkeypatch):
    agent_mod = _agent()
    _seed_submit_ctx(agent_mod)
    captured: dict = {}

    def fake_submit(state, query_json, _factory, *, normalize_filters=True):
        captured["query_json"] = query_json
        captured["normalize_filters"] = normalize_filters
        return "Generated SQL:\nSELECT 1\n\nResult: ok"

    monkeypatch.setattr(agent_mod, "submit_slayer_query", fake_submit)
    result = await agent_mod.submit_query.handler({"query": _SINGLE_OBJECT})
    assert json.loads(captured["query_json"]) == _SINGLE_OBJECT
    assert captured["normalize_filters"] is True
    assert "Generated SQL" in str(result)


async def test_submit_multi_stage_list_forwards_payload(monkeypatch):
    agent_mod = _agent()
    _seed_submit_ctx(agent_mod)
    captured: dict = {}

    def fake_submit(state, query_json, _factory, *, normalize_filters=True):
        captured["query_json"] = query_json
        return "Generated SQL:\nSELECT 1\n\nResult: ok"

    monkeypatch.setattr(agent_mod, "submit_slayer_query", fake_submit)
    await agent_mod.submit_query.handler({"query": _MULTI_STAGE_LIST})
    assert json.loads(captured["query_json"]) == _MULTI_STAGE_LIST


async def test_submit_normalize_filters_optout_forwards_flag_and_exact_payload(monkeypatch):
    agent_mod = _agent()
    _seed_submit_ctx(agent_mod)
    captured: dict = {}

    def fake_submit(state, query_json, _factory, *, normalize_filters=True):
        captured["query_json"] = query_json
        captured["normalize_filters"] = normalize_filters
        return "Generated SQL:\nSELECT 1\n\nResult: ok"

    monkeypatch.setattr(agent_mod, "submit_slayer_query", fake_submit)
    payload = {"source_model": "orders", "filters": ["Tier == 'Premium'"]}
    await agent_mod.submit_query.handler({"query": payload, "normalize_filters": False})
    assert captured["normalize_filters"] is False
    assert json.loads(captured["query_json"]) == payload, "opt-out must forward the payload unchanged"


async def test_submit_rejects_scalar_payload(monkeypatch):
    """A payload that is neither an object nor a list is rejected with a
    message naming both valid shapes."""
    agent_mod = _agent()
    _seed_submit_ctx(agent_mod)
    result = str(await agent_mod.submit_query.handler({"query": 123}))
    low = result.lower()
    assert "object" in low and ("array" in low or "list" in low), (
        f"expected a two-shapes rejection message, got: {result!r}"
    )


# ===========================================================================
# submit_slayer_query compiles the unified shape, records the payload
# byte-identical, and honours the exact-case opt-out (spec scenarios
# "Submit a single/multi-stage query" + "normalize_filters opt-out").
# Driven directly (the proven _FakeState pattern) so compilation + recording
# are actually exercised, not mocked away.
# ===========================================================================


class _FakeState(SimpleNamespace):
    def __init__(self, **kw):
        defaults = dict(
            status=SimpleNamespace(
                original_data={"selected_database": "fake_db"},
                remaining_budget=100.0, total_budget=100.0, force_submit=False,
            ),
            data_path_base="/tmp/ignored",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            user_sim_prompt_version="v2", slayer_storage_dir="", result=None,
        )
        defaults.update(kw)
        super().__init__(**defaults)


def _patch_execute_submit(monkeypatch):
    from bird_interact_agents.agents import _submit

    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: (f"obs of {sql}", 1.0, True, True, True),
    )
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)
    return _submit


async def test_submit_slayer_query_optout_records_byte_identical_and_exact_case(monkeypatch):
    _submit = _patch_execute_submit(monkeypatch)
    received: dict = {}

    def fake_sql_sync(payload):
        received["payload"] = payload
        return "SELECT 1"

    state = _FakeState()
    original = '{"source_model": "orders", "filters": ["Tier == \\"Premium\\""]}'
    _submit.submit_slayer_query(
        state, query_json=original,
        slayer_client_factory=lambda s: SimpleNamespace(sql_sync=fake_sql_sync),
        normalize_filters=False,
    )
    # Exact-case survives compilation (no lower/trim wrap, literal unchanged).
    assert received["payload"]["filters"] == ['Tier == "Premium"']
    # The recorded submitted query is the agent's payload byte-for-byte.
    assert state.result["submitted_query"] == original


async def test_submit_slayer_query_compiles_multi_stage_list_via_sql_sync(monkeypatch):
    """A multi-stage list compiles through `client.sql_sync` (the 0.10.2
    path — no `query_nested` fallback)."""
    import json as _json

    _submit = _patch_execute_submit(monkeypatch)
    received: dict = {}

    def fake_sql_sync(payload):
        received["payload"] = payload
        return "SELECT 1"

    state = _FakeState()
    original = _json.dumps(_MULTI_STAGE_LIST)
    _submit.submit_slayer_query(
        state, query_json=original,
        slayer_client_factory=lambda s: SimpleNamespace(sql_sync=fake_sql_sync),
    )
    assert isinstance(received["payload"], list)
    assert len(received["payload"]) == len(_MULTI_STAGE_LIST)
    assert state.result["submitted_query"] == original
