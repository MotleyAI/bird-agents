"""DEV-1822: `submit_cube_query` — whitelist refusal (Codex C9), /v1/sql →
materialized SQL (C1), and storage of the SQL in the submit_sql slot + the Cube
query JSON alongside (C3)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bird_interact_agents.cube_local import submission as sub


# --- whitelist validation (C9) ---------------------------------------------

def test_valid_query_passes():
    # exercises every allowed top-level key so an over-strict validator fails
    sub.validate_cube_query({
        "measures": ["orders.count"],
        "dimensions": ["customers.region"],
        "filters": [{"member": "customers.region", "operator": "equals", "values": ["US"]}],
        "timeDimensions": [{"dimension": "orders.created", "granularity": "month",
                            "dateRange": ["2021-01-01", "2021-12-31"]}],
        "segments": ["orders.big"],
        "order": {"orders.count": "desc"}, "limit": 100, "offset": 5,
        "timezone": "UTC", "ungrouped": False,
    })


@pytest.mark.parametrize("query", [
    {"measures": ["orders.count"], "total": True},                 # post-processing
    {"measures": ["orders.count"], "bogusKey": 1},                 # unknown key
    {"dimensions": [{"sql": "1+1", "type": "number"}]},            # member expr in dimensions
    {"measures": [{"sql": "count(*)", "type": "number"}]},         # member expr in measures
    {"segments": [{"sql": "x > 0"}]},                              # member expr in segments
    {"timeDimensions": [{"dimension": "o.t", "compareDateRange": [["a", "b"]]}]},  # compareDateRange
    [{"measures": ["a.count"]}, {"measures": ["b.count"]}],         # blending (list)
])
def test_refused_queries(query):
    with pytest.raises(sub.CubeQueryRefused):
        sub.validate_cube_query(query)


def test_query_to_sql_materializes(monkeypatch):
    client = SimpleNamespace(sql=lambda q: ("SELECT $1", ["US"]))
    out = sub.cube_query_to_sql({"measures": ["orders.count"]}, client)
    assert out == "SELECT 'US'"


def test_query_to_sql_refuses_before_calling_client():
    called = {"n": 0}

    def _sql(q):
        called["n"] += 1
        return ("x", [])

    client = SimpleNamespace(sql=_sql)
    with pytest.raises(sub.CubeQueryRefused):
        sub.cube_query_to_sql({"total": True}, client)
    assert called["n"] == 0


# --- tool wiring (C1 + C3) --------------------------------------------------

@pytest.mark.asyncio
async def test_submit_cube_query_stores_sql_and_cube_json(monkeypatch):
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import SampleStatus

    captured = {}

    def fake_submit_raw_sql(state, sql, cost_action="submit_sql"):
        captured["sql"] = sql
        captured["cost_action"] = cost_action
        state.result = {"submitted_sql": sql, "phase1_passed": True, "total_reward": 1.0}
        return "Submitted OK"

    monkeypatch.setattr(agent_mod, "submit_raw_sql", fake_submit_raw_sql)

    query = {"measures": ["orders.count"],
             "filters": [{"member": "orders.region", "operator": "equals", "values": ["US"]}]}
    agent_mod._ctx_var.set({
        "status": SampleStatus(idx=0, original_data={"selected_database": "alien"},
                               remaining_budget=30.0, total_budget=30.0),
        "data_path_base": "/dev/null",
        "cube": SimpleNamespace(sql=lambda q: ("SELECT $1", ["US"])),
        "query_mode": "cube",
        "result": None,
    })

    out = await agent_mod.submit_cube_query.handler({"query": query})
    assert "content" in out
    assert captured["cost_action"] == "submit_cube_query"
    assert captured["sql"] == "SELECT 'US'"
    stored = agent_mod._ctx_var.get()["result"]
    assert stored["submitted_sql"] == "SELECT 'US'"
    assert json.loads(stored["submitted_query"]) == query


@pytest.mark.asyncio
async def test_submit_cube_query_refusal_does_not_submit(monkeypatch):
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import SampleStatus

    called = {"n": 0}
    monkeypatch.setattr(agent_mod, "submit_raw_sql",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))

    status = SampleStatus(idx=0, original_data={"selected_database": "alien"},
                          remaining_budget=30.0, total_budget=30.0)
    agent_mod._ctx_var.set({
        "status": status,
        "data_path_base": "/dev/null",
        "cube": SimpleNamespace(sql=lambda q: ("x", [])),
        "query_mode": "cube",
        "result": None,
    })

    out = await agent_mod.submit_cube_query.handler({"query": {"total": True}})
    assert "content" in out
    assert called["n"] == 0
    # refusal must not submit, not charge budget, not force submission (C9)
    assert agent_mod._ctx_var.get()["result"] is None
    assert status.remaining_budget == 30.0
    assert status.force_submit is False
