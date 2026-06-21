"""DEV-1577 — the ``query`` / ``submit_query`` wrappers must accept the
shorthands ``SlayerQuery`` already coerces, instead of pre-rejecting them
at the tool boundary.

Two shapes from the PR #50 query-syntax-rejection analysis are pinned here:

1. **String measures/dimensions** (``measures: ["count"]``) — ``SlayerQuery``'s
   ``_coerce_measures`` / ``_coerce_dimensions`` BeforeValidators turn these into
   ``[{"formula": "count"}]`` / ``[{"name": ...}]``. Both wrappers route through
   ``SlayerQuery.model_validate`` (``query_impl`` via slayer's MCP ``query.fn``,
   ``submit_slayer_query`` via ``client.sql_sync``), so the wrapper must forward
   the strings untouched rather than choke on them.
2. **Misplaced tool-level passthrough kwargs** (``show_sql`` / ``dry_run`` /
   ``explain`` / ``format`` / ``normalize_filters`` nested INSIDE ``query_json``):
   * ``query_impl`` lifts them out into the corresponding wrapper kwarg.
   * ``submit_slayer_query`` strips them (a submission always evaluates, so the
     preview/directive options are meaningless there) so ``SlayerQuery``'s
     ``extra="forbid"`` doesn't reject the whole submission.

Genuinely unknown fields still raise in both paths.

Per ``feedback_no_prompt_content_tests``: behaviour assertions only, no
prompt-substring checks.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# query_impl — string-measure shorthand rides through to slayer's query.fn
# ---------------------------------------------------------------------------


def _patch_slayer_query_fn(monkeypatch):
    """Install a fake slayer MCP ``query.fn`` and return the captured-kwarg
    dict (mirrors the helper in test_dev1546_distinct_dim_values.py)."""
    from bird_interact_agents.agents import _query

    forwarded: dict[str, Any] = {}

    async def fake_slayer_query(**kwargs):
        forwarded.update(kwargs)
        return "<formatted>"

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)
    return forwarded


@pytest.mark.asyncio
async def test_query_impl_forwards_string_measure_shorthand(monkeypatch):
    """``measures: ["count"]`` (a bare string) is forwarded verbatim to
    slayer's ``query.fn`` — slayer's ``_coerce_measures`` turns it into
    ``{"formula": "count"}`` downstream. The wrapper must NOT reject the
    string at its own boundary."""
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    await query_impl(json.dumps({
        "source_model": "orders",
        "measures": ["count"],
        "dimensions": ["status"],
    }))
    assert forwarded["measures"] == ["count"]
    assert forwarded["dimensions"] == ["status"]


@pytest.mark.asyncio
async def test_query_impl_string_measure_validates_in_slayer(monkeypatch):
    """End-to-end coercion proof: the exact dict the wrapper hands slayer
    validates as a ``SlayerQuery`` with the string coerced to a formula —
    i.e. the shorthand the wrapper forwards is one SlayerQuery accepts."""
    from slayer.core.query import SlayerQuery

    q = SlayerQuery.model_validate({
        "source_model": "orders",
        "measures": ["count"],
        "dimensions": ["status"],
    })
    assert q.measures is not None
    assert q.measures[0].formula == "count"


# ---------------------------------------------------------------------------
# query_impl — misplaced tool-level kwargs are LIFTED, not rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_impl_lifts_misplaced_tool_level_keys(monkeypatch):
    """``show_sql`` / ``dry_run`` / ``explain`` / ``format`` nested inside
    ``query_json`` are lifted into slayer's ``query.fn`` kwargs (NOT
    rejected, NOT forwarded as SlayerQuery JSON fields)."""
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    await query_impl(json.dumps({
        "source_model": "orders",
        "dimensions": ["status"],
        "show_sql": True,
        "dry_run": True,
        "explain": True,
        "format": "json",
    }))
    assert forwarded["show_sql"] is True
    assert forwarded["dry_run"] is True
    assert forwarded["explain"] is True
    assert forwarded["format"] == "json"


@pytest.mark.asyncio
async def test_query_impl_lifts_nested_normalize_filters(monkeypatch):
    """``normalize_filters`` is OUR directive; misplaced inside the JSON it
    is lifted and applied (and never forwarded to slayer's ``query.fn``).
    With ``normalize_filters: false`` the filter survives case-sensitively."""
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    await query_impl(json.dumps({
        "source_model": "orders",
        "filters": ["category == 'Gadgets'"],
        "normalize_filters": False,
    }))
    assert forwarded["filters"] == ["category == 'Gadgets'"]
    assert "normalize_filters" not in forwarded


@pytest.mark.asyncio
async def test_query_impl_nested_format_overrides_default(monkeypatch):
    """Precedence sanity: a nested ``format`` beats the wrapper default
    (``markdown``) when the caller didn't pass the kwarg explicitly."""
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    await query_impl(json.dumps({
        "source_model": "orders",
        "format": "csv",
    }))
    assert forwarded["format"] == "csv"


@pytest.mark.parametrize("offending_field,value", [
    ("main_time_dimension", "created_at"),
    ("name", "my_query"),
    ("version", 3),
    ("totally_unknown", "x"),
])
@pytest.mark.asyncio
async def test_query_impl_still_rejects_genuinely_unknown_field(
    monkeypatch, offending_field, value,
):
    """Lifting tool-level kwargs must not loosen the allowlist for real
    SlayerQuery fields slayer's MCP ``query`` doesn't expose, nor for
    nonsense keys."""
    from bird_interact_agents.agents._query import query_impl

    _patch_slayer_query_fn(monkeypatch)
    with pytest.raises(ValueError) as ei:
        await query_impl(json.dumps({
            "source_model": "orders",
            offending_field: value,
        }))
    assert offending_field in str(ei.value)


# ---------------------------------------------------------------------------
# submit_slayer_query — string measures + misplaced kwargs are tolerated
# ---------------------------------------------------------------------------


class _FakeState(SimpleNamespace):
    """Minimal TaskState stand-in (mirrors the one in
    test_dev1546_distinct_dim_values.py)."""

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


def _validating_client(captured: dict | None = None,
                       return_sql: str = "SELECT 1") -> Any:
    """Fake slayer client whose ``sql_sync`` round-trips the dict through
    ``SlayerQuery.model_validate`` so the real coercion + ``extra=forbid``
    rules fire. Records the post-strip dict it received into
    ``captured["parsed"]`` when provided."""
    from slayer.core.query import SlayerQuery

    def sql_sync(parsed):
        if isinstance(parsed, list):
            for stage in parsed:
                SlayerQuery.model_validate(stage)
        else:
            SlayerQuery.model_validate(parsed)
        if captured is not None:
            captured["parsed"] = parsed
        return return_sql

    return SimpleNamespace(sql_sync=sql_sync)


def _patch_submit_side_effects(monkeypatch):
    """Bypass the real SQL evaluator + snapshot + dry-run gate."""
    from bird_interact_agents.agents import _submit

    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: ("ok", 1.0, True, True, True),
    )
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(_submit, "_dry_run_sql", lambda *a, **kw: None)


def test_submit_slayer_accepts_string_measure_shorthand(monkeypatch):
    """``measures: ["count"]`` round-trips through ``SlayerQuery.model_validate``
    (real coercion) and the submission reaches phase eval (p1=True)."""
    from bird_interact_agents.agents import _submit

    _patch_submit_side_effects(monkeypatch)
    captured: dict = {}
    state = _FakeState()
    _submit.submit_slayer_query(
        state,
        query_json=json.dumps({
            "source_model": "orders",
            "measures": ["count"],
            "dimensions": ["status"],
        }),
        slayer_client_factory=lambda s: _validating_client(captured),
    )
    assert state.result["phase1_passed"] is True
    assert state.result["submitted_sql"] == "SELECT 1"
    # The string shorthand reached slayer untouched (it coerces downstream).
    assert captured["parsed"]["measures"] == ["count"]


def test_submit_slayer_strips_misplaced_tool_level_keys(monkeypatch):
    """``show_sql`` (and friends) nested in the JSON are stripped before
    ``SlayerQuery.model_validate`` (``extra=forbid``) — the submission
    proceeds instead of failing translation."""
    from bird_interact_agents.agents import _submit

    _patch_submit_side_effects(monkeypatch)
    captured: dict = {}
    state = _FakeState()
    out = _submit.submit_slayer_query(
        state,
        query_json=json.dumps({
            "source_model": "orders",
            "dimensions": ["status"],
            "show_sql": True,
            "dry_run": True,
            "explain": True,
            "format": "json",
            "normalize_filters": False,
        }),
        slayer_client_factory=lambda s: _validating_client(captured),
    )
    assert state.result["phase1_passed"] is True
    assert state.result["submitted_sql"] == "SELECT 1"
    # None of the stripped keys reached SlayerQuery.
    for k in ("show_sql", "dry_run", "explain", "format", "normalize_filters"):
        assert k not in captured["parsed"]
    assert "Generated SQL" in out


def test_submit_slayer_strips_tool_level_keys_in_nested_stages(monkeypatch):
    """Nested-DAG submission: each stage gets the misplaced tool-level keys
    stripped so every stage validates against ``extra=forbid``."""
    from bird_interact_agents.agents import _submit

    _patch_submit_side_effects(monkeypatch)
    captured: dict = {}
    state = _FakeState()
    _submit.submit_slayer_query(
        state,
        query_json=json.dumps([
            {
                "source_model": "orders",
                "name": "stage1",
                "dimensions": ["status"],
                "show_sql": True,
            },
            {"source_model": "stage1", "dimensions": ["status"], "format": "csv"},
        ]),
        slayer_client_factory=lambda s: _validating_client(captured),
    )
    assert state.result["phase1_passed"] is True
    for stage in captured["parsed"]:
        assert "show_sql" not in stage
        assert "format" not in stage


def test_submit_slayer_still_rejects_genuinely_unknown_field(monkeypatch):
    """A field that is neither a SlayerQuery field nor a tool-level kwarg
    still surfaces SlayerQuery's ``extra=forbid`` rejection as a
    translation failure (submitted_sql stays None)."""
    from bird_interact_agents.agents import _submit

    _patch_submit_side_effects(monkeypatch)
    state = _FakeState()
    out = _submit.submit_slayer_query(
        state,
        query_json=json.dumps({
            "source_model": "orders",
            "dimensions": ["status"],
            "totally_unknown": "x",
        }),
        slayer_client_factory=lambda s: _validating_client(None),
    )
    assert state.result["phase1_passed"] is False
    assert state.result["submitted_sql"] is None
    assert "submission aborted" in out.lower() or "could not generate" in out.lower()
