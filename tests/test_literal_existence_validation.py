"""DEV-1478: literal-existence pre-save validation hook.

These tests are written BEFORE the implementation (TDD). They exercise the
new helper ``_collect_literal_problems`` in
``bird_interact_agents.agents.pydantic_ai_otf_encode.agent`` plus the wiring
inside ``_validate_and_call`` that combines the literal-existence feedback
with the existing SQL-execution feedback.

The validator extension is the structural fix for DEV-1478 failure modes
#1 (loses audited sample-value caveats) and #2 (acts on KB text without
checking the column). The encoder produces silently-broken predicates like
``socioeconomic__Income_Bracket IN ('High Income', 'Very High Income')``
because the encoder copied KB 2's prose enum verbatim while the column
actually holds R$ amount ranges. The validator catches this before the
write persists by comparing each `=` / `IN` literal against the column's
stored distinct values.

Walking the KB 42 regression target end-to-end (Codex review walkthrough
+ MINOR 1):

  agent SQL (after the new STYLE_GUIDE forces LOWER(TRIM(...))):
    LOWER(TRIM(socioeconomic__Income_Bracket)) IN ('high income',
                                                   'very high income')
    AND service_types.socsupport = 'no'

  walker peels Lower(Trim(Column)) -> bare Column socioeconomic__Income_Bracket
    -> resolves on host households -> sampled_values are R$ ranges ->
       BOTH 'high income' and 'very high income' miss -> problems recorded.
  walker on service_types.socsupport (bare; no wrapper) -> alias 'service_types'
    matches host households's join.target_model 'service_types' -> column
    looked up on service_types -> 'no' membership check fires.

Out-of-scope per plan §C.1.d: multi-hop ``<path>__<alias>.<col>`` references.
"""

from __future__ import annotations

import types
from typing import Any

import pytest
from slayer.core.models import Column, ModelJoin, SlayerModel


DB = "households"


# ---------------------------------------------------------------------------
# Fixture helpers — build an in-memory fake storage shaped like the
# households mini-interact DB. The fake stores SlayerModel instances by
# name and returns them by reference (no YAML serialization round-trip),
# so attributes attached via ``object.__setattr__`` — including the
# forward-compat ``Column.sampled_values`` that DEV-1480 will add to
# upstream SLayer — survive. The validator only reads
# ``storage.get_model(name)`` and ``storage.list_models(data_source=db)``;
# the fake provides exactly those two methods.
# ---------------------------------------------------------------------------


class _FakeStorage:
    """Minimal storage stub: stashes SlayerModel objects in a dict and
    returns them by reference. Mirrors the surface
    `_collect_literal_problems` reads (`get_model` / `list_models`)."""

    def __init__(self) -> None:
        self._models: dict[str, SlayerModel] = {}

    async def save_model(self, model: SlayerModel) -> None:
        self._models[model.name] = model

    async def get_model(self, name: str, data_source: str | None = None) -> Any:
        return self._models.get(name)

    async def list_models(self, data_source: str | None = None) -> list[str]:
        if data_source is None:
            return list(self._models.keys())
        return [
            n for n, m in self._models.items()
            if m.data_source == data_source
        ]


async def _new_storage(tmp_path) -> _FakeStorage:
    # tmp_path is accepted for symmetry with the existing setup_encoder
    # tests but unused — _FakeStorage is purely in-memory.
    return _FakeStorage()


async def _add_model(
    storage: _FakeStorage,
    *,
    name: str,
    columns: list[Column],
    joins: list[ModelJoin] | None = None,
) -> None:
    await storage.save_model(SlayerModel(
        name=name, data_source=DB, sql_table=name,
        columns=columns, joins=joins or [],
    ))


def _col(
    name: str,
    *,
    sampled: str | None = None,
    sampled_values: list[str] | None = None,
    type_: str = "string",
) -> Column:
    """Build a Column with optional sampled / sampled_values. Pydantic v2
    Columns don't yet declare `sampled_values` upstream (DEV-1480 adds it);
    we set it via object.__setattr__ so the validator can read it via
    ``getattr(col, "sampled_values", None)``. Survives only because the
    fake storage above does NOT YAML-round-trip these models."""
    c = Column(name=name, sql=name, type=type_, sampled=sampled)
    if sampled_values is not None:
        object.__setattr__(c, "sampled_values", sampled_values)
    return c


def _edit_args(
    model_name: str,
    *,
    columns: list[dict] | None = None,
    measures: list[dict] | None = None,
) -> dict:
    """Mimic the agent's edit_model call shape."""
    args: dict[str, Any] = {"model_name": model_name, "data_source": DB}
    if columns:
        args["columns"] = columns
    if measures:
        args["measures"] = measures
    return args


# ---------------------------------------------------------------------------
# Direct unit tests on `_collect_literal_problems` — these fail with
# ImportError until plan §C is implemented.
# ---------------------------------------------------------------------------


async def test_eq_literal_in_sampled_values_no_problem(tmp_path):
    """`<col> = 'Yes'` against sampled_values=['yes','no','Yes'] → no problem
    (case-insensitive membership)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="service_types", columns=[
        Column(name="serviceref", primary_key=True),
        _col("socsupport", sampled_values=["yes", "no", "Yes"]),
    ])

    args = _edit_args("service_types", columns=[
        {"name": "is_supported", "sql": "socsupport = 'Yes'", "type": "boolean"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    assert problems == []


async def test_eq_literal_missing_records_problem_with_full_sampled(tmp_path):
    """KB 21 / 42 regression target: `Income_Bracket = 'High Income'` against
    sampled values that are R$ ranges → problem feedback includes the full
    sampled list."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    sampled = ["R$ 0-1000", "R$ 1000-3000", "R$ 3000-6000",
               "R$ 6000-10000", "R$ 10000+"]
    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="households", columns=[
        Column(name="housenum", primary_key=True),
        _col("socioeconomic__Income_Bracket", sampled_values=sampled),
    ])

    args = _edit_args("households", columns=[
        {"name": "is_affluent", "type": "boolean",
         "sql": "socioeconomic__Income_Bracket = 'High Income'"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1
    p = problems[0]
    assert "socioeconomic__Income_Bracket" in p
    assert "High Income" in p
    # The full sampled list must be exposed so the agent can pick a real value.
    for v in sampled:
        assert v in p


async def test_in_literals_partial_miss_records_only_missing(tmp_path):
    """`<col> IN ('a','b')` where 'a' is missing → problem names only 'a'."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("status", sampled_values=["b", "c", "d"]),
    ])

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "status IN ('a', 'b')", "type": "boolean"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1
    # The "literal(s) ..." prefix names only the MISSING literal — 'a'.
    # ('b' will still appear later in the "Known values: 'b', 'c', 'd'"
    # section, which is the helpful list the agent should pick from.)
    missing_section = problems[0].split("Known values:", 1)[0]
    assert "'a'" in missing_section
    assert "'b'" not in missing_section


# ---------------------------------------------------------------------------
# Codex tests-vs-plan GAP: all THREE input SQL fields must be scanned —
# `columns[].sql`, `columns[].filter`, and `measures[].formula`. And the
# walker must also fire on `create_model` payloads (top-level columns).
# ---------------------------------------------------------------------------


async def test_columns_filter_predicate_is_scanned(tmp_path):
    """`Column.filter` is the per-row predicate applied inside CASE WHEN at
    aggregation time (R-FILTER recipe in the encoder). Its literals must
    pass the same membership check as `Column.sql`."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    args = _edit_args("m", columns=[
        {"name": "x", "filter": "col = 'maybe'", "type": "boolean"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1
    assert "'maybe'" in problems[0]


async def test_measures_formula_literals_are_scanned(tmp_path):
    """`measures[].formula` can carry expression strings (e.g.
    `case_when(col = 'x'):sum`). Their string literals must be checked
    against the column's sampled set."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    args = _edit_args("m", measures=[
        {"name": "yes_count",
         "formula": "CASE WHEN col = 'maybe' THEN 1 ELSE 0 END:sum"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1
    assert "'maybe'" in problems[0]


async def test_create_model_columns_are_scanned(tmp_path):
    """`create_model` carries top-level columns (table-backed model creation).
    Walking these is the same path as edit_model.columns[].sql."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    args = {
        "name": "m",  # creating into the existing host shape (test only)
        "data_source": DB, "sql_table": "m",
        "columns": [
            {"name": "x", "sql": "col = 'maybe'", "type": "boolean"},
        ],
    }
    problems = await _collect_literal_problems(
        "create_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1
    assert "'maybe'" in problems[0]


async def test_create_model_validation_runs_when_host_not_in_storage(tmp_path):
    """CodeRabbit PR #4 thread 1: a fresh `create_model` payload — host
    name is the NEW model that doesn't exist in storage yet — must still
    have its literal-existence checks run against the proposal's own
    columns. The synthetic host's columns inherit ``sampled_values`` from
    the proposal payload when supplied (the agent may copy upstream
    metadata into the create payload for R-MULTISTAGE / R-EXISTS stages).
    """
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    # NB: no model called `brand_new` is in storage.

    args = {
        "name": "brand_new",
        "data_source": DB, "sql_table": "src_table",
        "columns": [
            # This column carries explicit sampled_values, so the walker
            # has a concrete set to membership-check against.
            {"name": "status", "sql": "src_status",
             "sampled_values": ["yes", "no"]},
            {"name": "is_maybe", "type": "boolean",
             "sql": "status = 'maybe'"},  # 'maybe' is NOT in sampled
        ],
    }
    problems = await _collect_literal_problems(
        "create_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1, (
        f"create_model literal-walker must fire even when the host model "
        f"isn't in storage yet (CodeRabbit thread 1). got: {problems!r}"
    )
    assert "'maybe'" in problems[0]


async def test_create_model_silently_skips_when_payload_has_no_columns(tmp_path):
    """A degenerate create_model with no columns to resolve against AND a
    query whose source_model isn't in storage → walker returns no
    problems (best-effort skip; no false positive)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    args = {
        "name": "empty_model",
        "data_source": DB,
        # No columns / measures, and a backing-query stage whose
        # source_model `src` doesn't exist in storage either → the walker
        # has nothing to resolve names against.
        "query": {"source_model": "src", "dimensions": [],
                  "measures": [{"formula": "x = 'maybe':count"}], "limit": 0},
    }
    problems = await _collect_literal_problems(
        "create_model", args, storage=storage, db=DB,
    )
    assert problems == []


# ---------------------------------------------------------------------------
# Codex follow-up: `create_model(query=[...])` backing-query stages —
# R-MULTISTAGE / R-EXISTS write predicates inside query stages, whose
# columns resolve against each stage's own `source_model` (NOT the outer
# create_model's name). The walker must fire on these too.
# ---------------------------------------------------------------------------


async def test_create_model_single_stage_query_filter_is_scanned(tmp_path):
    """`create_model(query={source_model: 'src', filters: [...], ...})`:
    walker resolves the filter's columns against the stage's source_model
    and flags missing literals."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    # The stage's source_model exists in storage with a known sampled set.
    await _add_model(storage, name="src", columns=[
        Column(name="id", primary_key=True),
        _col("status", sampled_values=["yes", "no"]),
    ])

    args = {
        "name": "agg",
        "data_source": DB,
        "query": {
            "source_model": "src",
            "dimensions": [{"name": "id"}],
            "filters": ["status = 'maybe'"],
            "limit": 100,
        },
    }
    problems = await _collect_literal_problems(
        "create_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1, (
        f"query-stage filter must be walked against the stage's "
        f"source_model. got: {problems!r}"
    )
    assert "'maybe'" in problems[0]


async def test_create_model_multistage_query_stages_each_resolve_per_source(tmp_path):
    """Two-stage query — stage 1's filter references columns on
    `stage_a`, stage 2's measure formula references columns on
    `stage_b`. Each must resolve against its own source_model."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="stage_a", columns=[
        Column(name="id", primary_key=True),
        _col("a_status", sampled_values=["red", "blue"]),
    ])
    await _add_model(storage, name="stage_b", columns=[
        Column(name="id", primary_key=True),
        _col("b_kind", sampled_values=["alpha", "beta"]),
    ])

    args = {
        "name": "agg",
        "data_source": DB,
        "query": [
            {"source_model": "stage_a", "dimensions": [{"name": "id"}],
             "filters": ["a_status = 'green'"], "limit": 0},
            {"source_model": "stage_b", "dimensions": [{"name": "id"}],
             "measures": [{"formula": "CASE WHEN b_kind = 'gamma' THEN 1 "
                           "ELSE 0 END:sum"}],
             "limit": 0},
        ],
    }
    problems = await _collect_literal_problems(
        "create_model", args, storage=storage, db=DB,
    )
    # Both stages' literals are caught, one problem each.
    assert len(problems) == 2
    joined = " | ".join(problems)
    assert "'green'" in joined
    assert "'gamma'" in joined


async def test_create_model_query_stage_string_measures_are_scanned(tmp_path):
    """Codex finding J (PR #4 third-pass): SlayerQuery `measures` accepts
    BOTH dict form `{"formula": "..."}` AND bare-string form like
    `"col:sum"` or `"CASE WHEN col = 'x' THEN 1 ELSE 0 END:sum"`. The
    string form must be walked too — otherwise literal-bearing measures
    inside `create_model(query=[...])` stages slip through."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="src", columns=[
        Column(name="id", primary_key=True),
        _col("status", sampled_values=["good", "bad"]),
    ])

    args = {
        "name": "agg",
        "data_source": DB,
        "query": {
            "source_model": "src",
            "dimensions": [{"name": "id"}],
            # measures here is a list of strings, not dicts — the agent's
            # bare-formula shape.
            "measures": [
                "CASE WHEN status = 'maybe' THEN 1 ELSE 0 END:sum",
            ],
            "limit": 0,
        },
    }
    problems = await _collect_literal_problems(
        "create_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1, (
        f"string-shaped query-stage measure must be walked. got: "
        f"{problems!r}"
    )
    assert "'maybe'" in problems[0]


async def test_create_model_query_stage_dimension_filter_is_scanned(tmp_path):
    """A dimension-level `filter` clause inside a query stage carries the
    same literal-existence risk as the top-level `filters` list."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="src", columns=[
        Column(name="id", primary_key=True),
        _col("flag", sampled_values=["on", "off"]),
    ])

    args = {
        "name": "agg",
        "data_source": DB,
        "query": {
            "source_model": "src",
            "dimensions": [{"name": "id",
                            "filter": "flag = 'maybe'"}],
            "limit": 0,
        },
    }
    problems = await _collect_literal_problems(
        "create_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1
    assert "'maybe'" in problems[0]


async def test_create_model_query_stage_with_inline_source_model_skipped(tmp_path):
    """When the stage's `source_model` is an inline ModelExtension dict
    (not a model name string), the walker silently skips that stage —
    the dict-shaped source can't be looked up in storage. No false
    positives."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)

    args = {
        "name": "agg",
        "data_source": DB,
        "query": {
            # inline source — a ModelExtension dict, not a string lookup.
            "source_model": {"source_name": "src",
                             "columns": [{"name": "z", "sql": "z"}]},
            "filters": ["z = 'literal'"],
            "limit": 0,
        },
    }
    problems = await _collect_literal_problems(
        "create_model", args, storage=storage, db=DB,
    )
    assert problems == []


async def test_host_qualified_column_resolves_like_bare(tmp_path):
    """`households.col = 'literal'` where the alias IS the host's own name
    must resolve to the same column as bare `col`."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="households", columns=[
        Column(name="housenum", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    args = _edit_args("households", columns=[
        {"name": "x", "sql": "households.col = 'maybe'", "type": "boolean"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1
    assert "'maybe'" in problems[0]


# ---------------------------------------------------------------------------
# BLOCKER 1 (Codex): wrapper peeling. The STYLE_GUIDE forces
# LOWER(TRIM(<col>)) on every string comparison — without peeling these the
# walker would skip the literals entirely and the fix is useless.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sql, expected_miss", [
    ("LOWER(col) = 'x'", True),
    ("UPPER(col) = 'X'", True),
    ("TRIM(col) = 'x'", True),
    ("LOWER(TRIM(col)) = 'x'", True),
    ("TRIM(LOWER(col)) = 'x'", True),
    ("LOWER(col) = 'yes'", False),  # 'yes' IS in sampled
])
async def test_walker_peels_lower_trim_upper(tmp_path, sql, expected_miss):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    args = _edit_args("m", columns=[
        {"name": "out", "sql": sql, "type": "boolean"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    if expected_miss:
        assert len(problems) == 1, (
            f"expected wrapper-peel to find the bare col and flag the miss "
            f"for {sql!r}; got problems={problems!r}"
        )
    else:
        assert problems == []


async def test_walker_peels_for_in_with_lower_trim(tmp_path):
    """Both literals in LOWER(TRIM(col)) IN ('a', 'b') must be checked."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    args = _edit_args("m", columns=[
        {"name": "out", "type": "boolean",
         "sql": "LOWER(TRIM(col)) IN ('high income', 'very high income')"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1  # one column, two missing literals folded
    p = problems[0]
    assert "high income" in p
    assert "very high income" in p


# ---------------------------------------------------------------------------
# BLOCKER 2 (Codex): join-aware resolution. KB 42's predicate references
# `service_types.socsupport` while writing the entity on `households`. The
# walker must follow the host's joins to the target model.
# ---------------------------------------------------------------------------


async def test_alias_dot_col_resolves_via_host_join(tmp_path):
    """Writing on `households`, predicate `service_types.socsupport = 'no'` —
    alias matches a join target_model; the validator must follow the join
    and check `socsupport.sampled_values` on `service_types`."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="service_types", columns=[
        Column(name="serviceref", primary_key=True),
        _col("socsupport", sampled_values=["yes", "no"]),
    ])
    await _add_model(
        storage, name="households",
        columns=[
            Column(name="housenum", primary_key=True),
            Column(name="serviceplan"),
        ],
        joins=[ModelJoin(
            target_model="service_types",
            join_pairs=[["serviceplan", "serviceref"]],
        )],
    )

    # Hit: 'no' IS in sampled → no problem.
    args = _edit_args("households", columns=[
        {"name": "without_support", "type": "boolean",
         "sql": "service_types.socsupport = 'no'"},
    ])
    assert await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    ) == []

    # Miss: 'maybe' is NOT in sampled → problem.
    args2 = _edit_args("households", columns=[
        {"name": "ambig", "type": "boolean",
         "sql": "service_types.socsupport = 'maybe'"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args2, storage=storage, db=DB,
    )
    assert len(problems) == 1
    assert "service_types.socsupport" in problems[0] or "socsupport" in problems[0]
    assert "'maybe'" in problems[0]


async def test_alias_dot_col_unknown_alias_silent_skip(tmp_path):
    """`<alias>.<col>` where `<alias>` is neither the host nor a join target —
    no problem reported (best-effort; never false positive)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
    ])

    args = _edit_args("m", columns=[
        {"name": "x", "type": "boolean",
         "sql": "ghost_alias.col = 'literal'"},
    ])
    assert await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    ) == []


# ---------------------------------------------------------------------------
# Sampled-data fallback semantics (skip when not parseable).
# ---------------------------------------------------------------------------


async def test_sampled_missing_skip(tmp_path):
    """Column with sampled=None and no sampled_values → check skipped."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled=None),  # no sampled_values either
    ])

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'literal'", "type": "boolean"},
    ])
    assert await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    ) == []


@pytest.mark.parametrize("sampled", [
    "> 20 distinct",
    "> 50 distinct",
    "> 100 distinct",
])
async def test_sampled_overflow_skip(tmp_path, sampled):
    """Overflow marker → cannot check membership → skipped."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled=sampled),
    ])

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'literal'", "type": "boolean"},
    ])
    assert await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    ) == []


@pytest.mark.parametrize("sampled", [
    "0 .. 100",
    "1.5 .. 9.8",
    "2024-01-01 .. 2024-12-31",
])
async def test_sampled_numeric_or_date_range_skip(tmp_path, sampled):
    """Numeric / date range form → not a categorical set → skipped."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled=sampled),
    ])

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'literal'", "type": "boolean"},
    ])
    assert await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    ) == []


async def test_sampled_text_only_skips_membership_check(tmp_path):
    """Codex PR #4 finding L: when `Column.sampled_values` is None and
    only the human-readable `Column.sampled` text is present (older
    SLayer / not-yet-profiled column), the validator MUST skip the
    membership check. Naive comma-split would produce false positives on
    any value containing a comma (R$ ranges, "Springfield, IL", etc.) —
    violating the no-false-positive contract. DEV-1480 will ship
    `sampled_values` and make the check effective without this risk."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled="alpha, beta, gamma"),  # NO sampled_values
    ])

    # Agent writes a literal that ISN'T in the sampled text — but we skip
    # the check anyway because the parse is unsafe. No false-positive.
    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'delta'", "type": "boolean"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    assert problems == []


@pytest.mark.parametrize("sampled", [
    "R$ 1,000-3,000, R$ 3,000-6,000",   # numeric thousands-separator commas
    "Springfield, IL, Boston, MA",       # city/state non-numeric commas
    "Lastname, Firstname; OtherName",    # general embedded commas
])
async def test_sampled_text_only_never_false_positives_on_comma_values(
    tmp_path, sampled,
):
    """Codex finding L (specific): a real value containing commas in
    the human-readable `sampled` text MUST NOT trigger a false-positive
    rejection when the agent's literal matches a real value verbatim.
    The fix is total: skip whenever `sampled_values` is unavailable."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled=sampled),
    ])

    # Pick the first real value from the sampled text; the agent's
    # predicate exactly matches it. The validator must NOT flag this.
    real_value = sampled.split(",", 1)[0].strip()
    args = _edit_args("m", columns=[
        {"name": "x", "type": "boolean",
         "sql": f"col = '{real_value}'"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    assert problems == [], (
        f"validator must skip when sampled_values is unavailable, even "
        f"for sampled text containing comma-bearing values. got: {problems!r}"
    )


async def test_sampled_values_empty_list_flags_every_literal(tmp_path):
    """sampled_values == [] (all-NULL column / no distinct values) → any
    literal misses; feedback should advise the agent to defer."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=[]),
    ])

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'anything'", "type": "boolean"},
    ])
    problems = await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    )
    assert len(problems) == 1
    p = problems[0].lower()
    assert "no values" in p or "all null" in p or "defer" in p


# ---------------------------------------------------------------------------
# Patterns the walker should NOT pretend to handle.
# ---------------------------------------------------------------------------


async def test_storage_get_model_calls_are_scoped_to_data_source(tmp_path):
    """Codex finding K (PR #4 third-pass): `storage.get_model` calls
    must scope by the proposed write's `data_source` so a same-named
    model in another datasource (YAMLStorage's priority resolution)
    doesn't shadow the right one. Verify by recording every call and
    asserting `data_source=<DB>` was passed."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    calls: list[dict[str, Any]] = []

    class _RecordingStorage:
        def __init__(self, real: _FakeStorage):
            self._real = real

        async def get_model(self, name, data_source=None):
            calls.append({"name": name, "data_source": data_source})
            return await self._real.get_model(name, data_source=data_source)

        async def list_models(self, data_source=None):
            return await self._real.list_models(data_source=data_source)

    real = await _new_storage(tmp_path)
    await _add_model(real, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])
    recording = _RecordingStorage(real)

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'maybe'", "type": "boolean"},
    ])
    await _collect_literal_problems(
        "edit_model", args, storage=recording, db=DB,
    )
    assert calls, "the validator must consult storage for the host model"
    # Every consult must carry `data_source=DB` (from tool_args).
    for c in calls:
        assert c["data_source"] == DB, (
            f"unscoped get_model call leaked into the validator: {c}"
        )


async def test_storage_get_model_unscoped_fallback_when_data_source_missing(
    tmp_path,
):
    """When `tool_args` omits `data_source` (the corner case the existing
    `test_write_gate_resolves_datasource_via_storage_fallback` pins),
    the validator falls back to an unscoped `get_model(name)` call —
    preserves backward compatibility with the SQL gate's fallback path."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    calls: list[dict[str, Any]] = []

    class _RecordingStorage:
        def __init__(self, real: _FakeStorage):
            self._real = real

        async def get_model(self, name, data_source=None):
            calls.append({"name": name, "data_source": data_source})
            return await self._real.get_model(name, data_source=data_source)

        async def list_models(self, data_source=None):
            return await self._real.list_models(data_source=data_source)

    real = await _new_storage(tmp_path)
    await _add_model(real, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])
    recording = _RecordingStorage(real)

    # No data_source in args — matches the existing fallback test setup.
    args = {"model_name": "m", "columns": [
        {"name": "x", "sql": "col = 'maybe'", "type": "boolean"},
    ]}
    await _collect_literal_problems(
        "edit_model", args, storage=recording,
    )
    assert calls
    for c in calls:
        assert c["data_source"] is None, (
            f"without args data_source, validator should fall back to "
            f"unscoped lookup; got: {c}"
        )


async def test_unparseable_sql_returns_no_literal_problems(tmp_path):
    """sqlglot.parse_one raises → no literal problems collected (the existing
    SQL-execution gate still catches the real failure)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "THIS IS NOT SQL", "type": "boolean"},
    ])
    # Should not raise, should return empty list.
    assert await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    ) == []


async def test_numeric_equality_is_not_a_string_literal(tmp_path):
    """`<col> = 42` (numeric literal) is not in scope — only string literals
    are checked. Numeric ranges are conveyed via sampled, not sampled_values,
    anyway."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", type_="number", sampled="0 .. 100"),
    ])

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 42", "type": "boolean"},
    ])
    assert await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    ) == []


async def test_unresolvable_bare_column_skip(tmp_path):
    """Bare column reference that doesn't exist on the host → silent skip."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _collect_literal_problems,
    )

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
    ])

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "nonexistent = 'literal'", "type": "boolean"},
    ])
    assert await _collect_literal_problems(
        "edit_model", args, storage=storage, db=DB,
    ) == []


# ---------------------------------------------------------------------------
# Integration with `_validate_and_call`: combined SQL + literal feedback.
# ---------------------------------------------------------------------------


async def test_literal_problem_blocks_persist_and_returns_feedback(tmp_path):
    """When `_collect_literal_problems` finds a problem, the wired
    `_validate_and_call` must NOT call the underlying tool (no persist) and
    must return a feedback string that mentions the offending literal."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as _agent

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    called: list[Any] = []

    async def fake_call_tool(name: str, args: dict, _ctx: Any) -> str:
        called.append((name, args))
        return "WROTE"

    # Fake engine: every query passes (so SQL-execution gate finds nothing).
    class _OkEngine:
        async def execute(self, q):
            return types.SimpleNamespace(data=[])

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'maybe'", "type": "boolean"},
    ])
    result = await _agent._validate_and_call(
        fake_call_tool, "edit_model", args,
        storage=storage, engine=_OkEngine(),
    )
    assert called == [], (
        "literal-existence rejection must block the underlying write"
    )
    assert isinstance(result, str)
    assert "'maybe'" in result
    # The combined feedback string must clearly signal a failure to the agent
    # (the existing VALIDATION FAILED banner is fine, but the literal must be
    # named).


async def test_combined_sql_and_literal_problems_appear_together(
    monkeypatch, tmp_path,
):
    """If BOTH the existing SQL-execution gate AND the literal-existence
    gate find problems, the feedback must surface both — not one or the
    other — so the agent sees the full picture on one tool result.

    Monkeypatch `_collect_validation_problems` to deterministically return
    a stub SQL problem so we don't depend on engine internals; rely on the
    real `_collect_literal_problems` (against a known-missing literal) to
    produce the literal problem. Combined feedback must contain BOTH.
    """
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as _agent

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    called: list[Any] = []

    async def fake_call_tool(name, args, _ctx):
        called.append((name, args))
        return "WROTE"

    class _OkEngine:
        async def execute(self, q):
            return types.SimpleNamespace(data=[])

    async def stub_sql_problems(name, tool_args, *, storage, engine):
        return ["column m.x: SQL-SIDE STUB FAILURE"]

    monkeypatch.setattr(
        _agent, "_collect_validation_problems", stub_sql_problems,
    )

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'maybe'", "type": "boolean"},
    ])
    result = await _agent._validate_and_call(
        fake_call_tool, "edit_model", args,
        storage=storage, engine=_OkEngine(),
    )
    assert called == [], "validation failure must block the underlying write"
    assert isinstance(result, str)
    # BOTH problems must appear in the single tool result the agent receives.
    assert "SQL-SIDE STUB FAILURE" in result
    assert "'maybe'" in result


async def test_no_problems_lets_write_proceed(tmp_path):
    """All literals exist + SQL parses + engine is healthy → write proceeds."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as _agent

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    called: list[Any] = []

    async def fake_call_tool(name, args, _ctx):
        called.append((name, args))
        return "WROTE"

    class _OkEngine:
        async def execute(self, q):
            return types.SimpleNamespace(data=[])

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'yes'", "type": "boolean"},
    ])
    result = await _agent._validate_and_call(
        fake_call_tool, "edit_model", args,
        storage=storage, engine=_OkEngine(),
    )
    assert called and called[0][0] == "edit_model"
    assert result == "WROTE"


async def test_literal_collector_harness_error_fails_open(
    monkeypatch, tmp_path,
):
    """If `_collect_literal_problems` itself raises (a harness bug), the
    write must proceed (fail OPEN) WHEN THE SQL GATE IS ALSO CLEAN — never
    block valid work because of a validator bug. Mirrors the existing
    fail-open pattern in `_validate_and_call` for
    `_collect_validation_problems`.

    Monkeypatch the literal collector to raise directly so we're testing
    the wrapper's fail-open behavior, not an implementation detail that
    might swallow errors internally.
    """
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as _agent

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    called: list[Any] = []

    async def fake_call_tool(name, args, _ctx):
        called.append((name, args))
        return "WROTE"

    class _OkEngine:
        async def execute(self, q):
            return types.SimpleNamespace(data=[])

    async def exploding_collector(name, tool_args, *, storage, db=""):
        raise RuntimeError("simulated literal-collector bug")

    monkeypatch.setattr(
        _agent, "_collect_literal_problems", exploding_collector,
    )

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'yes'", "type": "boolean"},
    ])
    result = await _agent._validate_and_call(
        fake_call_tool, "edit_model", args,
        storage=storage, engine=_OkEngine(),
    )
    # Despite the harness error in the literal collector, the write proceeds
    # because the SQL gate found no problems either.
    assert called and called[0][0] == "edit_model"
    assert result == "WROTE"


async def test_literal_collector_error_still_surfaces_sql_problems(
    monkeypatch, tmp_path,
):
    """CodeRabbit PR #4 thread 2 + Codex: when the literal collector
    raises BUT the SQL gate has already collected blocking problems, the
    `except` branch must NOT short-circuit to `call_tool` — the existing
    SQL problems must still block the write. Earlier fix-open code
    swallowed sql_problems and allowed the write through, turning a
    literal-collector bug into a bypass for known-invalid SQL."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as _agent

    storage = await _new_storage(tmp_path)
    await _add_model(storage, name="m", columns=[
        Column(name="id", primary_key=True),
        _col("col", sampled_values=["yes", "no"]),
    ])

    called: list[Any] = []

    async def fake_call_tool(name, args, _ctx):
        called.append((name, args))
        return "WROTE"

    class _OkEngine:
        async def execute(self, q):
            return types.SimpleNamespace(data=[])

    # SQL gate finds a problem AND literal collector raises.
    async def stub_sql_problems(name, tool_args, *, storage, engine):
        return ["column m.x: SQL-SIDE STUB FAILURE — known invalid"]

    async def exploding_collector(name, tool_args, *, storage, db=""):
        raise RuntimeError("simulated literal-collector bug")

    monkeypatch.setattr(
        _agent, "_collect_validation_problems", stub_sql_problems,
    )
    monkeypatch.setattr(
        _agent, "_collect_literal_problems", exploding_collector,
    )

    args = _edit_args("m", columns=[
        {"name": "x", "sql": "col = 'yes'", "type": "boolean"},
    ])
    result = await _agent._validate_and_call(
        fake_call_tool, "edit_model", args,
        storage=storage, engine=_OkEngine(),
    )
    # The literal collector raised, but the SQL gate's problem must still
    # block the write — a literal-collector bug must NOT bypass the SQL
    # gate.
    assert called == [], (
        "literal-collector failure must NOT bypass pre-existing SQL "
        "validation failures (CodeRabbit thread 2 + Codex regression)"
    )
    assert isinstance(result, str)
    assert "SQL-SIDE STUB FAILURE" in result


# ---------------------------------------------------------------------------
# Async test plumbing — bird-agents uses pytest-asyncio; mirror existing
# tests' implicit auto-mode (no per-test @asyncio).
# ---------------------------------------------------------------------------


# Note: pytest-asyncio is configured project-wide; existing tests like
# tests/test_pydantic_ai_otf_encode_setup_encoder.py use `async def`
# functions directly without a per-test marker, so we follow the same
# convention here.
