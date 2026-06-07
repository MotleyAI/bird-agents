"""DEV-1534 Fix C: `normalize_filters` opt-out plumbing.

Background: `slayer_pipeline.filter_normalization.normalize_query_payload`
unconditionally wraps every Mode-B text-equality filter LHS in
``lower(trim(...))`` and lowercases the literal. When the gold uses
exact-case equality (a real failure pattern in the 298-instance
mini-interact analysis), the agent currently has no opt-out.

Fix C exposes an opt-out as a SEPARATE PARAMETER on:
- `normalize_query_payload(parsed, *, normalize: bool = True)` (lib helper)
- A new public `normalize_filters_list(filters, *, normalize: bool = True)`
  helper for the bare filters list (used by the wrapper tool).
- `agents/_submit.py:submit_slayer_query(..., normalize_filters: bool = True)`
  kwarg.
- `agents/claude_sdk/agent.py:submit_query` exposes `normalize_filters: bool`
  as a separate tool parameter alongside `query_json`.
- The new `mcp__bird-interact-tools__query` wrapper exposes
  `normalize_filters: bool` (covered in
  `tests/test_dev1534_query_wrapper.py`).

The trajectory's `submitted_query` MUST be the agent's ORIGINAL DSL
string (no flag injection / mutation) — only the compiled SQL reflects
the normalization decision.

Scope: claude_sdk only. `normalize_tool_filters` is UNCHANGED
(pydantic_ai_* out of scope per interview decision).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# normalize_query_payload — `normalize` kwarg
# ---------------------------------------------------------------------------


def test_normalize_query_payload_default_normalizes_dict():
    from bird_interact_agents.slayer_pipeline.filter_normalization import (
        normalize_query_payload,
    )

    q = {"source_model": "w", "filters": ["category == 'Gadgets'"]}
    out = normalize_query_payload(q)
    assert out["filters"] == ["lower(trim(category)) == 'gadgets'"]


def test_normalize_query_payload_normalize_true_explicit():
    """Explicit `normalize=True` matches the default behavior."""
    from bird_interact_agents.slayer_pipeline.filter_normalization import (
        normalize_query_payload,
    )

    q = {"filters": ["category == 'X'"]}
    out = normalize_query_payload(q, normalize=True)
    assert out["filters"] == ["lower(trim(category)) == 'x'"]


def test_normalize_query_payload_normalize_false_passthrough_dict():
    """`normalize=False` returns a deep-copy of the input with the
    filters VERBATIM (no lower/trim wrap, no literal lowercasing)."""
    from bird_interact_agents.slayer_pipeline.filter_normalization import (
        normalize_query_payload,
    )

    q = {"source_model": "w", "filters": ['category == "Gadgets"', "age > 10"]}
    out = normalize_query_payload(q, normalize=False)
    assert out["filters"] == ['category == "Gadgets"', "age > 10"]
    # Deep-copy invariant: returned object is NOT the same object as the input.
    assert out is not q
    assert out["filters"] is not q["filters"]


def test_normalize_query_payload_normalize_false_passthrough_list_shape():
    """List-shape (nested DAG) payload: `normalize=False` returns each stage
    verbatim too. One flag covers ALL stages (top-level semantics).

    Deep-copy invariant: NEITHER the outer list NOR any inner stage dict
    NOR any inner filter list is shared with the input. The wrapper /
    submit path must be free to mutate the returned object (e.g. strip
    transient keys) without ever reaching back through to the agent's
    original payload."""
    from bird_interact_agents.slayer_pipeline.filter_normalization import (
        normalize_query_payload,
    )

    stages = [
        {"name": "s1", "source_model": "w", "filters": ["category == 'A'"]},
        {"source_model": "s1", "filters": ["region == 'B'"]},
    ]
    out = normalize_query_payload(stages, normalize=False)
    assert out[0]["filters"] == ["category == 'A'"]
    assert out[1]["filters"] == ["region == 'B'"]
    # Outer list — distinct.
    assert out is not stages
    # Inner stage dicts — distinct.
    assert out[0] is not stages[0]
    assert out[1] is not stages[1]
    # Inner filter lists — distinct.
    assert out[0]["filters"] is not stages[0]["filters"]
    assert out[1]["filters"] is not stages[1]["filters"]


def test_normalize_query_payload_input_not_mutated_under_optout():
    """Like the default path, opt-out must NOT mutate the input."""
    from bird_interact_agents.slayer_pipeline.filter_normalization import (
        normalize_query_payload,
    )

    q = {"filters": ["category == 'Gadgets'"]}
    _ = normalize_query_payload(q, normalize=False)
    assert q["filters"] == ["category == 'Gadgets'"]


# ---------------------------------------------------------------------------
# normalize_filters_list — new public helper used by the wrapper tool
# ---------------------------------------------------------------------------


def test_normalize_filters_list_default_normalizes():
    from bird_interact_agents.slayer_pipeline.filter_normalization import (
        normalize_filters_list,
    )

    out = normalize_filters_list(['category == "Gadgets"', "age > 10"])
    assert out == ["lower(trim(category)) == 'gadgets'", "age > 10"]


def test_normalize_filters_list_normalize_false_passthrough():
    from bird_interact_agents.slayer_pipeline.filter_normalization import (
        normalize_filters_list,
    )

    inp = ['category == "Gadgets"', "age > 10"]
    out = normalize_filters_list(inp, normalize=False)
    assert out == ['category == "Gadgets"', "age > 10"]
    # Deep-copy invariant.
    assert out is not inp


def test_normalize_filters_list_handles_none():
    """Bare-list helper accepts None (the wrapper forwards
    `filters: Optional[List[str]] = None`)."""
    from bird_interact_agents.slayer_pipeline.filter_normalization import (
        normalize_filters_list,
    )

    assert normalize_filters_list(None) is None
    assert normalize_filters_list(None, normalize=False) is None


# ---------------------------------------------------------------------------
# submit_slayer_query — `normalize_filters` kwarg
# ---------------------------------------------------------------------------


class _FakeState(SimpleNamespace):
    """Mirrors tests/test_submit_helpers._FakeState."""

    def __init__(self, **kw):
        defaults = dict(
            status=SimpleNamespace(
                original_data={"selected_database": "fake_db"},
                remaining_budget=100.0,
                total_budget=100.0,
                force_submit=False,
            ),
            data_path_base="/tmp/ignored",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            user_sim_prompt_version="v2",
            slayer_storage_dir="",
            result=None,
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


def test_submit_slayer_query_default_normalizes(monkeypatch):
    """Default `normalize_filters=True` (or unsupplied) → SQL the slayer
    client receives carries the normalized payload."""
    from bird_interact_agents.agents import _submit

    _patch_execute_submit(monkeypatch)

    received = {}
    def fake_sql_sync(payload):
        received["payload"] = payload
        return "SELECT 1"

    state = _FakeState()
    _submit.submit_slayer_query(
        state,
        query_json='{"source_model": "w", "filters": ["category == \\"Gadgets\\""]}',
        slayer_client_factory=lambda s: SimpleNamespace(sql_sync=fake_sql_sync),
    )
    payload = received["payload"]
    assert payload["filters"] == ["lower(trim(category)) == 'gadgets'"]


def test_submit_slayer_query_normalize_filters_false_passthrough(monkeypatch):
    """With `normalize_filters=False`, the payload passed to the client
    carries the agent's ORIGINAL filter string (no lower/trim wrap, no
    literal lowercasing)."""
    from bird_interact_agents.agents import _submit

    _patch_execute_submit(monkeypatch)

    received = {}
    def fake_sql_sync(payload):
        received["payload"] = payload
        return "SELECT 1"

    state = _FakeState()
    _submit.submit_slayer_query(
        state,
        query_json='{"source_model": "w", "filters": ["category == \\"Gadgets\\""]}',
        slayer_client_factory=lambda s: SimpleNamespace(sql_sync=fake_sql_sync),
        normalize_filters=False,
    )
    payload = received["payload"]
    # Filter literal inside the JSON used double-quotes; after json.loads
    # the parsed value carries them verbatim under the opt-out path.
    assert payload["filters"] == ['category == "Gadgets"']


def test_submit_slayer_query_submitted_query_is_unmodified_original(monkeypatch):
    """The trajectory's `submitted_query` is the agent's ORIGINAL JSON
    string, byte-for-byte. The opt-out kwarg is OUTSIDE the JSON and
    must not contaminate the recorded DSL."""
    from bird_interact_agents.agents import _submit

    _patch_execute_submit(monkeypatch)

    state = _FakeState()
    original = '{"source_model": "w", "filters": ["category == \\"Gadgets\\""]}'
    _submit.submit_slayer_query(
        state,
        query_json=original,
        slayer_client_factory=lambda s: SimpleNamespace(sql_sync=lambda p: "SELECT 1"),
        normalize_filters=False,
    )
    assert state.result["submitted_query"] == original


def test_submit_slayer_query_normalize_filters_true_default_matches_no_kwarg(monkeypatch):
    """Explicit `normalize_filters=True` and the unsupplied default
    behave identically."""
    from bird_interact_agents.agents import _submit

    _patch_execute_submit(monkeypatch)

    payloads = []
    def fake_sql_sync(payload):
        payloads.append(payload)
        return "SELECT 1"

    state1 = _FakeState()
    _submit.submit_slayer_query(
        state1,
        query_json='{"source_model": "w", "filters": ["category == \\"X\\""]}',
        slayer_client_factory=lambda s: SimpleNamespace(sql_sync=fake_sql_sync),
    )
    state2 = _FakeState()
    _submit.submit_slayer_query(
        state2,
        query_json='{"source_model": "w", "filters": ["category == \\"X\\""]}',
        slayer_client_factory=lambda s: SimpleNamespace(sql_sync=fake_sql_sync),
        normalize_filters=True,
    )
    assert payloads[0] == payloads[1]
    assert payloads[0]["filters"] == ["lower(trim(category)) == 'x'"]


# ---------------------------------------------------------------------------
# normalize_tool_filters is UNCHANGED — regression smoke
# ---------------------------------------------------------------------------


def test_normalize_tool_filters_signature_unchanged_no_normalize_kwarg():
    """`normalize_tool_filters` is OUT OF SCOPE for Fix C (pydantic_ai_*
    only; user said don't touch). Its signature must not gain a
    `normalize` kwarg from this fix. If it does, this test fails so we
    catch unintended scope creep."""
    import inspect

    from bird_interact_agents.slayer_pipeline.filter_normalization import (
        normalize_tool_filters,
    )

    sig = inspect.signature(normalize_tool_filters)
    # Existing positional args: tool_name, tool_args
    assert list(sig.parameters) == ["tool_name", "tool_args"]
