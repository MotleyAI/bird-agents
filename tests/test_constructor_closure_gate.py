"""Closure-bound count-check on the constructor's `submit_query` tool
(DEV-1432).

After Stage 2 (projection-resolver) produces a confirmed list of
user-facing column names, that list is closure-captured into the
constructor agent's `submit_query` tool wrapper. The wrapper rejects
submissions whose `len(dimensions) + len(measures)` doesn't match the
confirmed count.

Key invariants pinned here:

* The closure-check raises `pydantic_ai.ModelRetry` BEFORE the call
  reaches `submit_slayer_query` — therefore consumes NO bird-coin
  budget (per the broader rule: only `execute_submit_action` calls
  charge submit_query cost).
* For nested-DAG submissions (list shape from DEV-1435), the check
  uses the LAST stage's dims+measures count (the DAG root).
* `_projection_count` is total and non-raising: weird shapes return
  None, which lets the helper produce its canonical shape-error
  instead of crashing.
* Bad-JSON submissions skip the closure-check entirely — the helper
  owns the "Invalid JSON" message.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic_ai import RunContext

# Reuse the existing test helpers for state-shaping.
from tests.test_pydantic_ai_recursive_tools import (
    _make_deps,
    _make_shared,
)


def _ctx(deps):
    return RunContext(
        deps=deps, model=None, usage=None, prompt="", run_step=0,
    )


def _tools(agent) -> dict:
    return dict(agent._function_toolset.tools)


# ---------------------------------------------------------------------------
# 1. Closure passes through on count match (single-stage and nested)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closure_passes_through_on_count_match_single_stage(monkeypatch):
    """When dims+measures equal the confirmed-projection length, the
    closure must hand off to `submit_slayer_query` unchanged. Mock the
    helper so we can assert it received the same JSON."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories
    from bird_interact_agents.agents import _submit

    seen = {}

    def _fake_submit_slayer_query(state, query_json, factory):
        seen["state"] = state
        seen["query_json"] = query_json
        return f"helper saw: {query_json}"

    monkeypatch.setattr(
        factories, "submit_slayer_query", _fake_submit_slayer_query,
    )

    shared = _make_shared(remaining_budget=20.0)
    deps = _make_deps(shared=shared, depth=0, max_depth=0)
    constructor = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("col_a", "col_b"),
    )
    submit_query = _tools(constructor)["submit_query"]

    payload = json.dumps({
        "source_model": "orders",
        "dimensions": ["status"],
        "measures": ["amount:sum"],
    })
    out = await submit_query.function(_ctx(deps), query_json=payload)

    assert "helper saw" in out
    assert seen["query_json"] == payload


@pytest.mark.asyncio
async def test_closure_passes_through_on_count_match_nested(monkeypatch):
    """Nested-DAG payload: the closure must count dims+measures on the
    LAST stage only, not sum across all stages."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    seen = {}

    def _fake_submit_slayer_query(state, query_json, factory):
        seen["query_json"] = query_json
        return "ok"

    monkeypatch.setattr(
        factories, "submit_slayer_query", _fake_submit_slayer_query,
    )

    shared = _make_shared(remaining_budget=20.0)
    deps = _make_deps(shared=shared)
    constructor = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("stage_dim_x", "stage_measure_y"),
    )
    submit_query = _tools(constructor)["submit_query"]

    # Stage 1 has lots of dims/measures (irrelevant); stage 2 has
    # exactly 1 dim + 1 measure = 2 cols, matching confirmed_projection.
    payload = json.dumps([
        {
            "name": "stage1",
            "source_model": "encounters",
            "dimensions": ["a", "b", "c"],
            "measures": ["x:sum", "y:avg", "z:max"],
        },
        {
            "source_model": "stage1",
            "dimensions": ["dim_x"],
            "measures": ["measure_y:sum"],
        },
    ])
    out = await submit_query.function(_ctx(deps), query_json=payload)

    assert out == "ok"
    assert seen["query_json"] == payload


# ---------------------------------------------------------------------------
# 2. Closure rejects on count mismatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closure_raises_model_retry_on_count_mismatch_single_stage(monkeypatch):
    """Count mismatch must raise pydantic_ai.ModelRetry with a message
    naming the expected count, observed count, and the confirmed list."""
    from pydantic_ai import ModelRetry
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    def _must_not_be_called(*a, **kw):
        pytest.fail("submit_slayer_query must NOT be called on count mismatch")

    monkeypatch.setattr(
        factories, "submit_slayer_query", _must_not_be_called,
    )

    deps = _make_deps()
    constructor = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("col_a", "col_b", "col_c"),
    )
    submit_query = _tools(constructor)["submit_query"]

    # 1 dim + 1 measure = 2 cols, but confirmed_projection has 3.
    payload = json.dumps({
        "source_model": "orders",
        "dimensions": ["status"],
        "measures": ["amount:sum"],
    })

    with pytest.raises(ModelRetry) as exc:
        await submit_query.function(_ctx(deps), query_json=payload)

    msg = str(exc.value)
    assert "2" in msg, "ModelRetry must name the observed count"
    assert "3" in msg, "ModelRetry must name the expected count"
    # The confirmed list itself must appear in the message so the LLM
    # can see what to reconcile against.
    assert "col_a" in msg
    assert "col_b" in msg
    assert "col_c" in msg


@pytest.mark.asyncio
async def test_closure_raises_model_retry_on_count_mismatch_nested(monkeypatch):
    """Nested form: last-stage count, NOT sum across stages."""
    from pydantic_ai import ModelRetry
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    monkeypatch.setattr(
        factories, "submit_slayer_query",
        lambda *a, **kw: pytest.fail("helper must not run"),
    )

    deps = _make_deps()
    constructor = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("a", "b"),
    )
    submit_query = _tools(constructor)["submit_query"]

    # Last stage has 3 cols (1 dim + 2 measures); confirmed has 2.
    payload = json.dumps([
        {"name": "stage1", "source_model": "encounters",
         "dimensions": ["x"], "measures": ["y:sum"]},
        {"source_model": "stage1",
         "dimensions": ["a"], "measures": ["b:sum", "c:sum"]},
    ])

    with pytest.raises(ModelRetry):
        await submit_query.function(_ctx(deps), query_json=payload)


# ---------------------------------------------------------------------------
# 3. Budget is NOT charged on closure rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closure_rejection_charges_no_budget(monkeypatch):
    """The whole point of the closure-check raising BEFORE
    submit_slayer_query: it consumes no submit budget. The constructor
    can retry with a corrected projection without burning bird-coins."""
    from pydantic_ai import ModelRetry
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    monkeypatch.setattr(
        factories, "submit_slayer_query",
        lambda *a, **kw: pytest.fail("helper must not run on mismatch"),
    )

    shared = _make_shared(remaining_budget=20.0)
    deps = _make_deps(shared=shared)
    start_budget = shared.status.remaining_budget

    constructor = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("a", "b", "c"),
    )
    submit_query = _tools(constructor)["submit_query"]

    payload = json.dumps({
        "source_model": "orders",
        "dimensions": ["status"],   # 1 col vs 3 expected
    })

    with pytest.raises(ModelRetry):
        await submit_query.function(_ctx(deps), query_json=payload)

    assert shared.status.remaining_budget == start_budget, (
        f"Closure-check rejection must not charge submit_query budget; "
        f"started at {start_budget}, now at "
        f"{shared.status.remaining_budget}."
    )


# ---------------------------------------------------------------------------
# 4. Bad JSON bypasses the closure-check (helper owns that message)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closure_bypasses_check_for_invalid_json(monkeypatch):
    """The closure can't count columns in unparseable JSON. The helper
    owns the canonical 'Invalid JSON' message. Verify the closure hands
    off without raising ModelRetry."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    helper_calls = []

    def _fake_helper(state, query_json, factory):
        helper_calls.append(query_json)
        return f"helper handled: {query_json}"

    monkeypatch.setattr(
        factories, "submit_slayer_query", _fake_helper,
    )

    deps = _make_deps()
    constructor = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("a", "b"),
    )
    submit_query = _tools(constructor)["submit_query"]

    out = await submit_query.function(_ctx(deps), query_json="{not json")

    assert helper_calls == ["{not json"]
    assert "helper handled" in out


# ---------------------------------------------------------------------------
# 5. Unsupported top-level shapes also bypass the closure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [
    '"oops"',         # top-level string
    "42",             # top-level number
    "true",           # top-level bool
    "null",           # top-level null
    "[]",             # empty list (degenerate DAG)
])
@pytest.mark.asyncio
async def test_closure_bypasses_check_for_non_dict_non_list_shapes(
    monkeypatch, payload,
):
    """Any shape whose dims+measures count is undefined (scalar,
    empty list) must skip the closure-check and let the helper
    produce its shape-error from DEV-1435."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    helper_calls = []

    def _fake_helper(state, query_json, factory):
        helper_calls.append(query_json)
        return f"helper handled: {query_json}"

    monkeypatch.setattr(
        factories, "submit_slayer_query", _fake_helper,
    )

    deps = _make_deps()
    constructor = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("a", "b"),
    )
    submit_query = _tools(constructor)["submit_query"]

    out = await submit_query.function(_ctx(deps), query_json=payload)

    assert helper_calls == [payload], (
        "Unsupported shape must hand off to helper, not raise ModelRetry."
    )
    assert "helper handled" in out


# ---------------------------------------------------------------------------
# 6. _projection_count helper: total and non-raising
# ---------------------------------------------------------------------------


def test_projection_count_total_and_non_raising():
    """`_projection_count` is invoked on whatever `json.loads` returned.
    It must not raise on edge cases (missing keys, wrong-type values,
    empty list, etc.) — those return None so the closure-check skips
    and the helper produces its canonical error."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    pc = factories._projection_count

    # Happy path: dict with dims+measures.
    assert pc({"dimensions": ["a"], "measures": ["b:sum"]}) == 2
    # Dict missing keys → 0 (still a number; closure-check sees 0).
    assert pc({"source_model": "orders"}) == 0
    # Dict with one key.
    assert pc({"dimensions": ["a", "b"]}) == 2
    assert pc({"measures": ["x:sum"]}) == 1
    # Nested DAG: last stage counts.
    assert pc([
        {"dimensions": ["a", "b", "c"], "measures": []},
        {"dimensions": ["x"], "measures": ["y:sum"]},
    ]) == 2
    # Empty list → None (degenerate DAG, helper handles it).
    assert pc([]) is None
    # Top-level scalar / non-dict-non-list → None.
    assert pc("oops") is None
    assert pc(42) is None
    assert pc(None) is None
    # Defensive: dict with non-list dimensions value.
    assert pc({"dimensions": "not a list", "measures": []}) is None
    # JSON null for a key is semantically equivalent to "key absent" —
    # treat as empty rather than non-list.
    assert pc({"dimensions": [], "measures": None}) == 0
    assert pc({"dimensions": None, "measures": ["x:sum"]}) == 1
    # Last stage is a non-dict in a nested list → None.
    assert pc([{"dimensions": ["a"]}, "stage2 should be a dict"]) is None
    # Wrapped shapes (DEV-1435 — the four shapes mental_4 burned coins
    # guessing) MUST return None so they bypass the closure and hit
    # the helper's canonical shape-error path. Without this, a wrapped
    # `{"queries": [...]}` would have dims+measures = 0 and the
    # closure would mask the proper shape-error message.
    assert pc({"queries": [{"source_model": "x"}]}) is None
    assert pc({"nested_queries": [{"source_model": "x"}]}) is None
    assert pc({"source_model": "nested", "queries": [{"source_model": "x"}]}) is None


@pytest.mark.asyncio
async def test_closure_bypasses_check_for_wrapped_shapes(monkeypatch):
    """The wrapped shapes from DEV-1435 (`{queries: [...]}` and friends)
    must NOT raise ModelRetry from the closure. Their canonical error
    message comes from `submit_slayer_query`'s shape-error path; the
    closure must hand off, not intercept."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    helper_calls = []

    def _fake_helper(state, query_json, factory):
        helper_calls.append(query_json)
        return f"helper handled: {query_json}"

    monkeypatch.setattr(
        factories, "submit_slayer_query", _fake_helper,
    )

    deps = _make_deps()
    constructor = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("a", "b", "c", "d"),
    )
    submit_query = _tools(constructor)["submit_query"]

    for wrapped in [
        '{"queries": [{"source_model": "x"}]}',
        '{"nested_queries": [{"source_model": "x"}]}',
        '{"source_model": "nested", "queries": [{"source_model": "x"}]}',
    ]:
        helper_calls.clear()
        out = await submit_query.function(_ctx(deps), query_json=wrapped)
        assert helper_calls == [wrapped], (
            f"Wrapped shape {wrapped!r} should bypass closure check and "
            f"hit the helper; instead helper saw {helper_calls!r}."
        )
        assert "helper handled" in out


@pytest.mark.asyncio
async def test_closures_for_two_constructors_are_isolated(monkeypatch):
    """Two constructor agents built with DIFFERENT confirmed lists must
    each enforce ONLY their own list. A buggy implementation using
    module-level / class-level state instead of true per-agent closure
    would let one constructor see the other's expected_count."""
    from pydantic_ai import ModelRetry
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    monkeypatch.setattr(
        factories, "submit_slayer_query",
        lambda *a, **kw: "ok",  # success path for matched-count cases
    )

    deps = _make_deps()

    agent_a = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("a",),
    )
    agent_b = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("a", "b", "c"),
    )
    submit_a = _tools(agent_a)["submit_query"]
    submit_b = _tools(agent_b)["submit_query"]

    # Payload with 1 column → matches agent_a's expected_count=1,
    # mismatches agent_b's expected_count=3.
    one_col_payload = json.dumps({
        "source_model": "orders",
        "measures": ["amount:sum"],
    })

    # agent_a accepts it.
    out_a = await submit_a.function(_ctx(deps), query_json=one_col_payload)
    assert out_a == "ok"

    # agent_b rejects it — proves the closure is per-agent, not shared.
    with pytest.raises(ModelRetry) as exc:
        await submit_b.function(_ctx(deps), query_json=one_col_payload)
    msg = str(exc.value)
    assert "3" in msg
    # agent_b's list must appear in the message, not agent_a's.
    assert "a" in msg and "b" in msg and "c" in msg


@pytest.mark.asyncio
async def test_constructor_factory_signature_omits_shared_slayer_server_or_does_not_attach_it():
    """Defensive: a future refactor might wire the shared MCP server
    into the constructor. The closure test above proves the closure
    fires; this test pins that the constructor still receives ONLY
    the tools it should (no `search`/`query`/`inspect_model` from a
    silently-attached MCP toolset). Stage 2's resolver factory has
    its own equivalent check in test_projection_resolver.py."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories

    constructor = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("a",),
    )
    names = set(_tools(constructor))
    # Native tools the constructor SHOULD have:
    assert "ask_user" in names
    assert "submit_query" in names
    # MCP-server tools that should NOT be silently attached when
    # shared_slayer_server is None:
    for forbidden in ("search", "query", "inspect_model", "models_summary",
                      "list_datasources", "help"):
        assert forbidden not in names, (
            f"Constructor built with shared_slayer_server=None still has "
            f"{forbidden!r} in its native tool list."
        )


# ---------------------------------------------------------------------------
# 7. Constructor factory signature requires confirmed_projection
# ---------------------------------------------------------------------------


def test_constructor_factory_requires_confirmed_projection():
    """The confirmed_projection arg must be REQUIRED — there's no safe
    default. agent.py only builds the constructor after Stage 2 has
    produced (or guarded an empty) list, so confirmed_projection is
    always available at call time. A default would let a future
    refactor accidentally bypass the closure."""
    from bird_interact_agents.agents.pydantic_ai_recursive import factories
    import inspect

    sig = inspect.signature(factories._build_query_constructor)
    assert "confirmed_projection" in sig.parameters, (
        "_build_query_constructor must accept a confirmed_projection arg."
    )
    param = sig.parameters["confirmed_projection"]
    # The parameter must be REQUIRED (no default value).
    assert param.default is inspect.Parameter.empty, (
        "confirmed_projection must be a required argument (no default); "
        "a default value would let a future refactor accidentally skip "
        "the closure-bound count-check."
    )
