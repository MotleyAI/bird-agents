"""DEV-1546 — ``distinct_dimension_values`` flag wired through the OTF
slayer-mode preview + submit path.

Slayer 0.7.2 added ``distinct_dimension_values: bool = True`` as a
``SlayerQuery`` field (and as a top-level kwarg on its MCP ``query``
tool). Setting ``False`` on a dimension-only query disables the auto-dedup
``GROUP BY`` and emits flat ``SELECT <dims/td>`` rows.

This file pins the wiring on OUR side of slayer:

* ``query_impl`` accepts ``query_json: str`` as a positional arg plus
  tool-level kwargs — the wrapper unpacks the JSON DSL through an
  explicit field allowlist to slayer's MCP ``query`` function. Sharp
  errors for invalid JSON, missing ``source_model``, nested-DAG list
  shape, and unknown top-level keys.
* DEV-1555 CR r1 (unification): the @tool wrapper for the ``query`` MCP
  tool no longer takes a single ``query_json`` string at the SDK layer
  — it accepts the SAME structured shape as ``submit_query`` (single-
  stage via ``source_model`` + projection fields OR nested-DAG via
  ``queries``), builds the SlayerQuery JSON internally, and forwards
  it to DEV-1546's ``query_impl(query_json: str, …)``. The
  ``distinct_dimension_values`` field rides through as a named
  property on the schema instead of a JSON-string field. See
  ``test_claude_sdk_query_tool_schema_uses_structured_shape_after_unification``.
* ``query_nested_impl`` forwards stage dicts verbatim — the new field
  rides through with no change. Pinned here as a regression net.
* ``submit_slayer_query`` forwards the JSON DSL through
  ``normalize_query_payload`` → ``_compile_sql``. Pinned that the new
  field survives that round-trip and that slayer's 0.7.2 rejection
  rules surface as observation errors.
* ``SLAYER_OTF_ONE_SHOT_V0`` (the V0 snapshot — see
  ``_shared_otf_prompts.py``) inlines the full ``_DEDUP_VS_RAW_ROWS``
  body byte-for-byte and is rendered by the v0 agents; the V1
  prompts teach the same guidance in a shorter inlined form.

Per the project convention (``feedback_no_prompt_content_tests``), no
substring / anchor-phrase assertions on prompt copy — mechanical
contracts only.
"""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# query_impl — signature pin (single query_json + tool-level kwargs)
# ---------------------------------------------------------------------------


def test_query_impl_signature_is_query_json_plus_tool_opts():
    """``query_impl(query_json, *, show_sql, dry_run, explain, format,
    normalize_filters)`` — the single-DSL-arg shape. Slayer-native
    SlayerQuery fields ride INSIDE ``query_json``; only tool-level
    options (and our ``normalize_filters`` directive) live outside it.
    """
    from bird_interact_agents.agents._query import query_impl

    sig = inspect.signature(query_impl)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == [
        "query_json",
        "show_sql",
        "dry_run",
        "explain",
        "format",
        "normalize_filters",
    ], f"unexpected param order: {[p.name for p in params]}"

    # Positional-or-keyword for the JSON; keyword-only for tool opts.
    assert params[0].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    for p in params[1:]:
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{p.name} must be keyword-only"
        )

    # Defaults: tool opts default to slayer's MCP `query` defaults;
    # normalize_filters defaults True (our directive).
    assert params[0].default is inspect.Parameter.empty
    defaults = {p.name: p.default for p in params[1:]}
    assert defaults == {
        "show_sql": False,
        "dry_run": False,
        "explain": False,
        "format": "markdown",
        "normalize_filters": True,
    }


# ---------------------------------------------------------------------------
# query_impl — distinct_dimension_values rides as a SlayerQuery field
# inside the JSON DSL and is forwarded as a kwarg to slayer's MCP query
# ---------------------------------------------------------------------------


def _patch_slayer_query_fn(monkeypatch):
    """Install a fake slayer MCP `query.fn` and return the captured-kwarg dict."""
    from bird_interact_agents.agents import _query

    forwarded: dict[str, Any] = {}

    async def fake_slayer_query(**kwargs):
        forwarded.update(kwargs)
        return "<formatted>"

    monkeypatch.setattr(_query, "_get_slayer_query_fn", lambda: fake_slayer_query)
    return forwarded


@pytest.mark.asyncio
async def test_query_impl_forwards_distinct_dimension_values_false(monkeypatch):
    """``distinct_dimension_values`` inside the JSON DSL becomes a kwarg
    on slayer's MCP ``query`` function call."""
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    await query_impl(
        json.dumps({
            "source_model": "orders",
            "dimensions": ["status"],
            "distinct_dimension_values": False,
        }),
    )
    assert forwarded["distinct_dimension_values"] is False


@pytest.mark.asyncio
async def test_query_impl_defaults_distinct_dimension_values_true(monkeypatch):
    """JSON without the field → forwarded kwarg defaults to ``True``
    (matches slayer's MCP query signature default).

    Either implementation is acceptable: (a) always explicitly forward
    ``distinct_dimension_values=parsed.get(..., True)``, or (b) only
    forward when the field is present and lean on slayer's MCP default.
    Both reach the same SQL output; the test accepts either.
    """
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    await query_impl(json.dumps({
        "source_model": "orders",
        "dimensions": ["status"],
    }))
    # Slayer's MCP `query` kwarg default is True; either we forward
    # the explicit True, or we omit and let slayer's default fire.
    assert forwarded.get("distinct_dimension_values", True) is True


# ---------------------------------------------------------------------------
# query_impl — filter normalization preserved on the new wrapper shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_impl_default_normalizes_filters_inside_query_json(monkeypatch):
    """``filters`` inside the JSON DSL still get the default
    ``lower(trim(...)) == 'x'`` Mode-B normalization before being
    forwarded to slayer."""
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    await query_impl(json.dumps({
        "source_model": "orders",
        "filters": ["category == 'Gadgets'"],
    }))
    assert forwarded["filters"] == ["lower(trim(category)) == 'gadgets'"]


@pytest.mark.asyncio
async def test_query_impl_normalize_filters_false_passes_verbatim(monkeypatch):
    """``normalize_filters=False`` (tool-level kwarg) leaves filters
    inside the JSON verbatim — case-sensitive equality semantics."""
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    await query_impl(
        json.dumps({
            "source_model": "orders",
            "filters": ["category == 'Gadgets'"],
        }),
        normalize_filters=False,
    )
    assert forwarded["filters"] == ["category == 'Gadgets'"]


@pytest.mark.asyncio
async def test_query_impl_does_not_forward_normalize_filters(monkeypatch):
    """``normalize_filters`` is OUR directive and must never reach
    slayer's ``query.fn`` kwargs."""
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    await query_impl(
        json.dumps({"source_model": "orders"}),
        normalize_filters=False,
    )
    assert "normalize_filters" not in forwarded


# ---------------------------------------------------------------------------
# query_impl — tool-level kwargs (show_sql / dry_run / explain / format)
# forward to slayer's MCP query under their natively-named slots
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_impl_forwards_tool_level_kwargs(monkeypatch):
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    await query_impl(
        json.dumps({"source_model": "orders"}),
        show_sql=True,
        dry_run=True,
        explain=True,
        format="json",
    )
    assert forwarded["show_sql"] is True
    assert forwarded["dry_run"] is True
    assert forwarded["explain"] is True
    assert forwarded["format"] == "json"


# ---------------------------------------------------------------------------
# query_impl — all SlayerQuery DSL fields ride through the JSON unpack
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_impl_unpacks_all_allowlisted_dsl_fields(monkeypatch):
    """Every allowlisted ``SlayerQuery`` field present in the JSON is
    forwarded under the same name to slayer's MCP ``query``. Missing
    fields default to ``None`` / slayer's documented defaults so the
    upstream signature is happy."""
    from bird_interact_agents.agents._query import query_impl

    forwarded = _patch_slayer_query_fn(monkeypatch)
    payload = {
        "source_model": "orders",
        "measures": [{"formula": "*:count"}],
        "dimensions": ["status"],
        "filters": [],  # empty → no normalization to compare against
        "time_dimensions": [{"dimension": "created_at", "granularity": "month"}],
        "order": [{"column": "status", "direction": "asc"}],
        "limit": 25,
        "offset": 3,
        "whole_periods_only": True,
        "variables": {"k": "v"},
        "distinct_dimension_values": False,
    }
    await query_impl(json.dumps(payload))
    for field, expected in payload.items():
        assert forwarded[field] == expected, (
            f"{field}: forwarded={forwarded.get(field)!r} expected={expected!r}"
        )


# ---------------------------------------------------------------------------
# query_impl — sharp errors for malformed inputs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_impl_rejects_invalid_json(monkeypatch):
    """Unparseable ``query_json`` surfaces a sharp error pointing at the
    DSL shape (mirrors ``submit_slayer_query``)."""
    from bird_interact_agents.agents._query import query_impl

    _patch_slayer_query_fn(monkeypatch)
    with pytest.raises(ValueError) as ei:
        await query_impl("{not json}")
    msg = str(ei.value)
    assert "json" in msg.lower()


@pytest.mark.asyncio
async def test_query_impl_rejects_missing_source_model(monkeypatch):
    """``source_model`` is the only mandatory ``SlayerQuery`` field; a
    JSON object without it should fail at the wrapper layer (not at
    slayer's Pydantic validator) so the error message stays sharp."""
    from bird_interact_agents.agents._query import query_impl

    _patch_slayer_query_fn(monkeypatch)
    with pytest.raises(ValueError) as ei:
        await query_impl(json.dumps({"dimensions": ["status"]}))
    msg = str(ei.value)
    assert "source_model" in msg


@pytest.mark.parametrize("offending_field,value", [
    # Unsupported SlayerQuery fields (not exposed as MCP `query` kwargs):
    ("main_time_dimension", "created_at"),
    ("name", "my_query"),
    ("version", 3),
    # NOTE (DEV-1577): tool-level kwargs misplaced inside the JSON
    # (`show_sql`/`dry_run`/`explain`/`format`/`normalize_filters`) are
    # NO LONGER rejected — they're lifted out into their wrapper kwargs.
    # See test_query_impl_lifts_misplaced_tool_level_keys.
])
@pytest.mark.asyncio
async def test_query_impl_rejects_unknown_top_level_field(
    monkeypatch, offending_field, value,
):
    """Codex round-1: blindly forwarding ``parsed`` to slayer's MCP
    ``query`` would TypeError on ``SlayerQuery`` fields the MCP tool
    doesn't accept as kwargs (``main_time_dimension``, ``name``,
    ``version``). The wrapper raises a sharp error naming the offending
    key and the supported-field allowlist instead.

    The allowlist is intentionally narrow — every key the agent puts
    in ``query_json`` must be a SlayerQuery DSL field that slayer's MCP
    ``query`` natively exposes as a kwarg.
    """
    from bird_interact_agents.agents._query import query_impl

    _patch_slayer_query_fn(monkeypatch)
    payload = {
        "source_model": "orders",
        "dimensions": ["status"],
        offending_field: value,
    }
    with pytest.raises(ValueError) as ei:
        await query_impl(json.dumps(payload))
    msg = str(ei.value)
    assert offending_field in msg, (
        f"error message must name the offending field {offending_field!r}; "
        f"got: {msg!r}"
    )
    # Error must point to alternative tools so the agent self-corrects.
    assert "submit_query" in msg or "query_nested" in msg


@pytest.mark.asyncio
async def test_query_impl_rejects_list_shape(monkeypatch):
    """Codex round-1: a nested-DAG list arrives at ``submit_query``
    legitimately, but the single-stage preview wrapper rejects it with
    a sharp error pointing at ``query_nested``."""
    from bird_interact_agents.agents._query import query_impl

    _patch_slayer_query_fn(monkeypatch)
    with pytest.raises(ValueError) as ei:
        await query_impl(json.dumps([
            {"source_model": "orders", "name": "stage1"},
            {"source_model": "stage1"},
        ]))
    msg = str(ei.value)
    assert "query_nested" in msg


# ---------------------------------------------------------------------------
# query_nested — stage dict with distinct_dimension_values rides through
# untouched. Regression pin (no impl change expected here).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_nested_stage_with_distinct_dim_passes_through(monkeypatch):
    from bird_interact_agents.agents import _query

    forwarded: dict[str, Any] = {}
    async def fake(**kwargs):
        forwarded.update(kwargs)
        return ""

    monkeypatch.setattr(
        _query, "_get_slayer_tool_fn",
        lambda name: fake if name == "query_nested" else None,
    )
    stages = [
        {
            "source_model": "orders",
            "dimensions": ["status"],
            "distinct_dimension_values": False,
        },
        {"source_model": "orders", "dimensions": ["region"]},
    ]
    await _query.query_nested_impl(queries=stages)
    out_stages = forwarded["queries"]
    assert out_stages[0]["distinct_dimension_values"] is False
    # Default (omitted) stays omitted, so slayer's per-stage default fires.
    assert "distinct_dimension_values" not in out_stages[1]


# ---------------------------------------------------------------------------
# submit_slayer_query — the JSON DSL field survives normalize_query_payload
# → _compile_sql, and slayer's 0.7.2 rejection rules surface as observation
# errors.
# ---------------------------------------------------------------------------


class _FakeState(SimpleNamespace):
    """Minimal stand-in for TaskState — only the fields ``submit_slayer_query``
    touches. Mirrors ``tests/test_submit_helpers.py::_FakeState``."""

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


def _capturing_client(captured: dict, return_sql: str = "SELECT 1") -> Any:
    """A fake slayer client whose ``sql_sync`` records the parsed DSL it
    receives and returns a static SQL string."""
    def sql_sync(parsed):
        captured["parsed"] = parsed
        return return_sql
    return SimpleNamespace(sql_sync=sql_sync)


def _validating_client(captured: dict | None = None,
                       return_sql: str = "SELECT 1") -> Any:
    """A fake slayer client whose ``sql_sync`` round-trips the dict
    through ``SlayerQuery.model_validate`` so 0.7.2's structural
    rejection rules (``DistinctDimensionValuesError``) actually fire.

    When ``captured`` is provided, the parsed dict (or list of stage
    dicts) is recorded into ``captured["parsed"]`` AFTER validation —
    so the test can both assert the field reaches slayer AND know
    slayer would accept it. Without ``captured`` the call still
    short-circuits to ``return_sql`` on a clean validation.
    """
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
    """Bypass the actual SQL-evaluator + snapshot capture for these
    field-roundtrip-only assertions."""
    from bird_interact_agents.agents import _submit

    monkeypatch.setattr(
        _submit, "execute_submit_action",
        lambda sql, status, dpb: ("ok", 1.0, True, True, True),
    )
    monkeypatch.setattr(_submit, "capture_result_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(_submit, "_dry_run_sql", lambda *a, **kw: None)


def test_submit_slayer_preserves_distinct_false_through_compile(monkeypatch):
    """The ``distinct_dimension_values: false`` field survives
    ``normalize_query_payload`` and arrives in the dict that
    ``client.sql_sync`` receives — i.e. nothing on our side strips it
    before slayer renders SQL."""
    from bird_interact_agents.agents import _submit

    _patch_submit_side_effects(monkeypatch)
    captured: dict = {}

    state = _FakeState()
    _submit.submit_slayer_query(
        state,
        query_json=json.dumps({
            "source_model": "orders",
            "dimensions": ["status"],
            "distinct_dimension_values": False,
        }),
        slayer_client_factory=lambda s: _capturing_client(captured),
    )
    assert captured["parsed"]["distinct_dimension_values"] is False


def test_submit_slayer_distinct_field_omitted_by_default(monkeypatch):
    """When the agent doesn't set the field, the parsed dict reaching
    ``client.sql_sync`` doesn't carry it (slayer falls back to its
    documented ``True`` default)."""
    from bird_interact_agents.agents import _submit

    _patch_submit_side_effects(monkeypatch)
    captured: dict = {}

    state = _FakeState()
    _submit.submit_slayer_query(
        state,
        query_json=json.dumps({
            "source_model": "orders",
            "dimensions": ["status"],
        }),
        slayer_client_factory=lambda s: _capturing_client(captured),
    )
    assert "distinct_dimension_values" not in captured["parsed"]


def test_submit_slayer_rejects_distinct_false_with_measures(monkeypatch):
    """Slayer 0.7.2 raises ``DistinctDimensionValuesError`` when
    ``distinct_dimension_values=False`` is combined with non-empty
    measures. Our submit path surfaces it as a translation-failed
    observation (not a crash)."""
    from bird_interact_agents.agents import _submit

    _patch_submit_side_effects(monkeypatch)
    state = _FakeState()
    out = _submit.submit_slayer_query(
        state,
        query_json=json.dumps({
            "source_model": "orders",
            "dimensions": ["status"],
            "measures": [{"formula": "*:count"}],
            "distinct_dimension_values": False,
        }),
        slayer_client_factory=lambda s: _validating_client(None),
    )
    assert state.result["phase1_passed"] is False
    assert state.result["phase2_passed"] is False
    assert state.result["submitted_sql"] is None
    # Slayer's error mentions the rule name; the observation surfaces it.
    assert "distinct_dimension_values" in out


def test_submit_slayer_rejects_distinct_false_with_empty_dims_and_tds(monkeypatch):
    """Slayer 0.7.2 also rejects ``distinct_dimension_values=False`` when
    both ``dimensions`` AND ``time_dimensions`` are empty (nothing to
    project)."""
    from bird_interact_agents.agents import _submit

    _patch_submit_side_effects(monkeypatch)
    state = _FakeState()
    out = _submit.submit_slayer_query(
        state,
        query_json=json.dumps({
            "source_model": "orders",
            "distinct_dimension_values": False,
        }),
        slayer_client_factory=lambda s: _validating_client(None),
    )
    assert state.result["phase1_passed"] is False
    assert state.result["phase2_passed"] is False
    assert "distinct_dimension_values" in out


def test_submit_slayer_distinct_false_with_time_dim_only(monkeypatch):
    """Time-dimension-only raw-rows query (no plain dims, no measures,
    flag false) is a VALID 0.7.2 combination — passes
    ``SlayerQuery.model_validate`` and rides through ``_compile_sql``.

    Uses ``_validating_client`` (NOT ``_capturing_client``) so the
    test actually exercises slayer's structural validation rules —
    a future regression that, say, swaps ``time_dimensions`` and
    ``dimensions`` rules would be caught here.
    """
    from bird_interact_agents.agents import _submit

    _patch_submit_side_effects(monkeypatch)
    captured: dict = {}

    state = _FakeState()
    _submit.submit_slayer_query(
        state,
        query_json=json.dumps({
            "source_model": "orders",
            "time_dimensions": [
                {"dimension": "created_at", "granularity": "month"},
            ],
            "distinct_dimension_values": False,
        }),
        slayer_client_factory=lambda s: _validating_client(captured),
    )
    assert captured["parsed"]["distinct_dimension_values"] is False
    assert captured["parsed"]["time_dimensions"] == [
        {"dimension": "created_at", "granularity": "month"},
    ]
    # Slayer accepted the combination → submission reached phase eval
    # (the fake `execute_submit_action` returns p1=True).
    assert state.result["phase1_passed"] is True


# ---------------------------------------------------------------------------
# Dependency floor — slayer must be >= 0.7.2 so the new field exists.
# ---------------------------------------------------------------------------


def test_motley_slayer_floor_is_at_least_0_8_3():
    """``pyproject.toml`` must pin ``motley-slayer >= 0.8.3`` in BOTH
    the ``slayer`` and ``all`` extras.

    0.7.2 introduced ``SlayerQuery.distinct_dimension_values`` (a floor
    regression below it silently strips the field via Pydantic). 0.8.3 is the
    binding floor now: it ships the ``inspect()`` single-entity point-lookup
    (DEV-1588) the claude_sdk OTF encoder's deps block calls directly (no
    fallback) — a downgrade below it would break the encoder at import.

    Mechanical text-level check (no version-comparator dependency).
    """
    from pathlib import Path

    import re

    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    # Both extras must use a >= floor at least 0.8.3 (allow patch bumps).
    floors = re.findall(
        r'"motley-slayer\[advanced-search\]>=([0-9.]+)"', text,
    )
    # BOTH the `slayer` and `all` extras must carry the pin (the docstring
    # contract) — assert at least two matches so dropping one extra's pin fails
    # instead of passing vacuously (CodeRabbit).
    assert len(floors) >= 2, (
        "motley-slayer floor must be pinned in BOTH the `slayer` and `all` "
        f"extras of pyproject.toml; found {len(floors)}: {floors}"
    )
    for raw in floors:
        major, minor, patch, *_ = [*raw.split("."), "0", "0"]
        version = (int(major), int(minor), int(patch))
        assert version >= (0, 8, 3), (
            f"motley-slayer floor {raw!r} is below 0.8.3 — the claude_sdk "
            "encoder's deps block calls inspect() (DEV-1588) directly."
        )


# ---------------------------------------------------------------------------
# claude-sdk @tool("query", ...) registration: input schema collapsed to
# a single `query_json` string field + tool-level options.
# ---------------------------------------------------------------------------


def _extract_tool_schema(tool_obj: Any) -> dict:
    for attr in ("inputSchema", "input_schema", "schema", "args_schema"):
        s = getattr(tool_obj, attr, None)
        if s is not None:
            return s
    raise AssertionError(
        f"could not find input schema on {tool_obj!r} via "
        "inputSchema/input_schema/schema/args_schema"
    )


def test_claude_sdk_query_tool_schema_uses_structured_shape_after_unification():
    """DEV-1555 CR r1 / O1 unification: the single ``query_json`` string
    parameter is gone; the slayer ``query`` tool now exposes the same
    structured shape as ``submit_query`` (single-stage via
    ``source_model`` + projection fields OR nested-DAG via ``queries``).
    The wrapper builds the JSON string internally and forwards it to
    DEV-1546's ``query_impl(query_json: str, …)``.

    The full per-property pin lives in
    ``tests/test_dev1534_query_wrapper.py::test_query_wrapper_tool_schema_accepts_unified_shape``;
    here we just lock the change from the original DEV-1546 schema."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod

    schema = _extract_tool_schema(agent_mod.query)
    assert isinstance(schema, dict) and "properties" in schema
    props = schema["properties"]
    # The unified surface adds the structured-form keys; `query_json` is gone.
    assert "query_json" not in props
    assert "source_model" in props
    assert "queries" in props
    # `required` is empty — the handler gates on source_model XOR queries.
    assert schema.get("required") == []


# ---------------------------------------------------------------------------
# Prompts — _DEDUP_VS_RAW_ROWS is composed into both slayer-flavor
# prompts, and the post-insertion str.format placeholders are unchanged.
# ---------------------------------------------------------------------------


def test_dedup_vs_raw_rows_constant_exists_and_nonempty():
    from bird_interact_agents.agents._shared_otf_prompts import _DEDUP_VS_RAW_ROWS

    assert isinstance(_DEDUP_VS_RAW_ROWS, str) and _DEDUP_VS_RAW_ROWS.strip()


def test_dedup_vs_raw_rows_present_in_slayer_v0_one_shot():
    """Structural composition pin: the live ``_DEDUP_VS_RAW_ROWS``
    appears in the V0 snapshot byte-for-byte (V0 is the
    origin/main-shape prompt the v0 agents render). The V1 prompts
    teach the same guidance in a shorter inlined form rather than
    composing the constant; that contract is covered by
    ``test_dev1555_v0_v1_shared_prompts``."""
    from bird_interact_agents.agents._shared_otf_prompts import (
        _DEDUP_VS_RAW_ROWS,
        SLAYER_OTF_ONE_SHOT_V0,
    )

    assert _DEDUP_VS_RAW_ROWS in SLAYER_OTF_ONE_SHOT_V0


def test_dedup_vs_raw_rows_present_in_slayer_v0_ainteract():
    from bird_interact_agents.agents._shared_otf_prompts import (
        _DEDUP_VS_RAW_ROWS,
        SLAYER_OTF_AINTERACT_V0,
    )

    assert _DEDUP_VS_RAW_ROWS in SLAYER_OTF_AINTERACT_V0


def test_slayer_otf_prompts_format_placeholders_unchanged():
    """Both slayer-flavor prompts still take exactly ``{budget}``,
    ``{db_name}``, ``{user_query}`` via ``str.format`` — no new
    placeholders introduced, none removed."""
    from bird_interact_agents.agents.claude_sdk_otf.prompts import SLAYER_OTF_ONE_SHOT
    from bird_interact_agents.agents.claude_sdk_otf_ainteract.prompts import (
        SLAYER_OTF_AINTERACT,
    )

    # Sentinel values are unique substrings so a regression that drops
    # one of them won't be masked by an unrelated literal.
    args = dict(budget="<BUDGET-SENTINEL-7>", db_name="<DB-SENTINEL-9>",
                user_query="<UQ-SENTINEL-11>")
    one_shot = SLAYER_OTF_ONE_SHOT.format(**args)
    ainter = SLAYER_OTF_AINTERACT.format(**args)
    for rendered in (one_shot, ainter):
        for sentinel in args.values():
            assert sentinel in rendered

    # Passing an unexpected placeholder should NOT change the rendering
    # (mechanical: str.format ignores extras, raises on missing). The
    # raise-on-missing case is covered by the dict above being complete.
    with pytest.raises(KeyError):
        SLAYER_OTF_ONE_SHOT.format(budget="x", db_name="y")  # missing user_query
    with pytest.raises(KeyError):
        SLAYER_OTF_AINTERACT.format(budget="x", db_name="y")
