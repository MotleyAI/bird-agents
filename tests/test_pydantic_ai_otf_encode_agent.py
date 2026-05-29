"""Tests for the `PydanticAIOtfEncodeAgent` class.

Three categories:

1. Init-time validation (must require `slayer_setup='on-the-fly'`).
2. `run_task` mode validation (slayer + a-interact only).
3. Smoke flow with mocked model — root → sub-clarifier → kb_to_slayer
   → constructor — to prove the spawn tree wires up.

No LLM calls, no real MCP server. The harness's `load_db_data_if_needed`
is monkeypatched because no real DB is loaded in the test environment.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.exc import SQLAlchemyError

from slayer.core.query import SlayerQuery


# ---------------------------------------------------------------------------
# __init__ validates `slayer_setup`
# ---------------------------------------------------------------------------


def test_init_requires_on_the_fly_setup():
    """Codex-validated: the agent's whole reason for existing is the
    on-the-fly mode. Any other setup mode is a misconfiguration that
    must surface immediately, not silently produce broken result rows."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )

    with pytest.raises(ValueError, match="on-the-fly"):
        PydanticAIOtfEncodeAgent(slayer_setup="pre-encoded")


def test_init_accepts_on_the_fly_setup():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )

    inst = PydanticAIOtfEncodeAgent(slayer_setup="on-the-fly")
    assert inst.slayer_setup == "on-the-fly"


# ---------------------------------------------------------------------------
# run_task mode validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_task_rejects_raw_query_mode():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )

    inst = PydanticAIOtfEncodeAgent(slayer_setup="on-the-fly")
    with pytest.raises(ValueError, match="slayer"):
        await inst.run_task(
            {"selected_database": "x", "instance_id": "i",
             "amb_user_query": "?"},
            data_path_base="/tmp", budget=10.0,
            query_mode="raw", eval_mode="a-interact",
        )


@pytest.mark.asyncio
async def test_run_task_rejects_c_interact_mode():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )

    inst = PydanticAIOtfEncodeAgent(slayer_setup="on-the-fly")
    with pytest.raises(ValueError, match="a-interact"):
        await inst.run_task(
            {"selected_database": "x", "instance_id": "i",
             "amb_user_query": "?"},
            data_path_base="/tmp", budget=10.0,
            query_mode="slayer", eval_mode="c-interact",
        )


@pytest.mark.asyncio
async def test_run_task_rejects_oracle_mode():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )

    inst = PydanticAIOtfEncodeAgent(slayer_setup="on-the-fly")
    with pytest.raises(ValueError, match="a-interact"):
        await inst.run_task(
            {"selected_database": "x", "instance_id": "i",
             "amb_user_query": "?"},
            data_path_base="/tmp", budget=10.0,
            query_mode="slayer", eval_mode="oracle",
        )


# ---------------------------------------------------------------------------
# End-to-end smoke with mocked model + mocked MCP setup
# ---------------------------------------------------------------------------


@pytest.fixture
def _mock_otf_setup(monkeypatch, tmp_path):
    """Bypass the real `_resolve_otf_task_storage_dir` (which would
    need a real mini-interact DB) by pointing at an empty tmp storage
    dir. The agent's phases are exercised via stub factories."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as _agent

    async def fake_resolve(
        *, db_name, task_data, data_path_base, build_encoder=None,
        benchmark=None,
    ):
        # DEV-1462: real `_resolve_otf_task_storage_dir` gained a
        # `benchmark` kwarg; stub must accept it (default None = legacy).
        return str(tmp_path), []

    monkeypatch.setattr(
        _agent, "_resolve_otf_task_storage_dir", fake_resolve,
        raising=False,
    )
    monkeypatch.setattr(
        _agent, "load_db_data_if_needed", lambda *a, **kw: None,
    )
    # Prevent the agent from spinning up a real MCP server.
    monkeypatch.setattr(
        _agent, "_build_shared_slayer_server", lambda _: None,
        raising=False,
    )
    return tmp_path


class _StubAgent:
    """A minimal stand-in for a pydantic-ai Agent.run() result. The agent
    factories under test get replaced with builders that return one of
    these instances; `.run` returns a SimpleNamespace mimicking what
    `_fill_record_from_run` reads off the real pydantic_ai RunResult."""

    def __init__(self, output: Any):
        self._output = output

    async def run(self, *a, **kw):
        from types import SimpleNamespace
        # DEV-1454: the projection resolver now delivers its columns via
        # submit_projection into per-run deps (not run.output). When this stub
        # stands in for the resolver (list output), mirror that by populating
        # deps.projection_submission so _run_projection_resolver reads it.
        deps = kw.get("deps")
        if deps is not None and isinstance(self._output, list):
            deps.projection_submission = self._output
        return SimpleNamespace(
            output=self._output,
            usage=lambda: SimpleNamespace(
                input_tokens=1, output_tokens=1,
                cache_read_tokens=0, cache_write_tokens=0,
            ),
            all_messages=lambda: [],
        )


@pytest.mark.asyncio
async def test_run_task_smoke_flow_root_to_constructor(
    _mock_otf_setup, monkeypatch,
):
    """End-to-end happy path: root_clarifier spawns one sub_clarifier,
    sub-clarifier's stub run returns a slice spec, projection-resolver
    returns one column, constructor 'submits' by setting submitter_result
    on the shared state via the stub. Verify the final result row carries:

    * `kb_encoded` field (may be empty in this stub run — sub_clarifier
      didn't actually call kb_to_slayer in the stub).
    * `agent_records` includes one entry per phase (root, sub,
      projection_resolver, query_constructor).
    * `submission_status` reflects the stubbed submitter result.
    """
    from bird_interact_agents.agents.pydantic_ai_otf_encode import (
        agent as _agent, factories,
    )

    monkeypatch.setattr(
        _agent, "_build_root_clarifier",
        lambda **kw: _StubAgent("root output: one slice"),
    )
    monkeypatch.setattr(
        _agent, "_build_projection_resolver",
        lambda **kw: _StubAgent(["score"]),
    )
    # The constructor stub mutates SharedTaskState via the deps adapter.
    class _ConstructorStub:
        async def run(self, *a, deps, **kw):
            deps.shared.submitter_result = {
                "phase1_passed": True,
                "phase2_passed": False,
                "total_reward": 0.5,
                "submitted_sql": "SELECT 1",
                "submission_status": "submitted",
            }
            from types import SimpleNamespace
            return SimpleNamespace(
                output="constructor done",
                usage=lambda: SimpleNamespace(
                    input_tokens=1, output_tokens=1,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    monkeypatch.setattr(
        _agent, "_build_query_constructor",
        lambda **kw: _ConstructorStub(),
    )

    inst = _agent.PydanticAIOtfEncodeAgent(slayer_setup="on-the-fly")
    row = await inst.run_task(
        {"selected_database": "x", "instance_id": "i",
         "amb_user_query": "?", "knowledge_ambiguity": []},
        data_path_base="/tmp", budget=100.0,
        query_mode="slayer", eval_mode="a-interact",
    )

    assert row["instance_id"] == "i"
    assert row["submission_status"] == "submitted"
    assert row["phase1_passed"] is True
    # kb_encoded field is present (even if empty for this stub run).
    assert "kb_encoded" in row
    # Trajectory carries the expected number of agent records.
    roles = [r["role"] for r in row["trajectory"]["agents"]]
    assert "root_clarifier" in roles
    assert "projection_resolver" in roles
    assert "query_constructor" in roles


@pytest.mark.asyncio
async def test_run_task_smoke_with_real_kb_to_slayer_invocation(
    _mock_otf_setup, monkeypatch, tmp_path,
):
    """Codex test-review finding 4: smoke must exercise the REAL
    `kb_to_slayer` tool path (not just manually append a record). The
    sub-clarifier stub here invokes the registered tool on its own
    deps; the tool runs the stub encoder via the `_encoder_runner`
    seam; the resulting `EncoderResult` lands on shared.kb_encoded
    via the production code path."""
    import yaml as _yaml
    from pydantic_ai import RunContext

    from bird_interact_agents.agents.pydantic_ai_otf_encode import (
        agent as _agent, factories,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )
    from bird_interact_agents.slayer_otf.kb_memory_encoder import (
        encode_kb_as_memories,
    )
    from slayer.storage.yaml_storage import YAMLStorage

    # Seed a real per-task storage with one KB memory so the loader
    # can resolve kb_id 5.
    storage_path = _mock_otf_setup  # tmp_path from the fixture
    mems = encode_kb_as_memories(
        "x",
        [{"id": 5, "knowledge": "k", "description": "d",
          "definition": "x + y", "type": "calc",
          "children_knowledge": -1}],
        deleted_kb_ids=set(),
    )
    (storage_path / "memories.yaml").write_text(
        _yaml.safe_dump(mems, sort_keys=False),
    )

    # Stub encoder writes a column (claim) — verification step will
    # downgrade because the column doesn't physically exist, BUT the
    # AgentRecord + kb_encoded entry still land on shared state.
    async def stub_encoder_runner(*, kb_id, row, deps_map, ctx):
        return EncoderResult(
            kb_id=kb_id, status="error",  # verification will downgrade anyway
            entities=[EncodedEntity(
                kind="column", host_model="m", name="c5",
                entity_ref="x.m.c5",
            )],
            notes="stub encoder",
            error="stub",
        )

    # Build a sub-clarifier stub that invokes the registered tool.
    class _SubClarifierStub:
        def __init__(self, sub_agent):
            self._sub = sub_agent

        async def run(self, *a, deps, **kw):
            tool = dict(self._sub._function_toolset.tools)["kb_to_slayer"]
            ctx = RunContext(
                deps=deps, model=None, usage=None, prompt="", run_step=0,
            )
            await tool.function(ctx, kb_ids=[5])
            from types import SimpleNamespace
            return SimpleNamespace(
                output="sub output",
                usage=lambda: SimpleNamespace(
                    input_tokens=1, output_tokens=1,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    # Build a bare Agent (NOT through `_build_sub_clarifier`, which
    # would register the default kb_to_slayer) and register the tool
    # with the stub runner.
    from pydantic_ai import Agent
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        TaskDeps,
    )
    real_sub = Agent(model="test", deps_type=TaskDeps, retries=2)
    factories._register_kb_to_slayer(
        real_sub,
        model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
        _encoder_runner=stub_encoder_runner,
    )

    class _RootSpawnStub:
        async def run(self, *a, deps, **kw):
            # Attach the YAMLStorage so the loader works.
            deps.shared._slayer_storage = YAMLStorage(
                base_dir=str(storage_path),
            )
            await _SubClarifierStub(real_sub).run(deps=deps)
            from types import SimpleNamespace
            return SimpleNamespace(
                output="root output",
                usage=lambda: SimpleNamespace(
                    input_tokens=1, output_tokens=1,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    monkeypatch.setattr(
        _agent, "_build_root_clarifier",
        lambda **kw: _RootSpawnStub(),
    )
    monkeypatch.setattr(
        _agent, "_build_projection_resolver",
        lambda **kw: _StubAgent(["c5"]),
    )

    class _ConstructorStub:
        async def run(self, *a, deps, **kw):
            deps.shared.submitter_result = {
                "phase1_passed": False, "phase2_passed": False,
                "total_reward": 0.0,
                "submitted_sql": None,
                "submission_status": "submitted",
            }
            from types import SimpleNamespace
            return SimpleNamespace(
                output="ok",
                usage=lambda: SimpleNamespace(
                    input_tokens=1, output_tokens=1,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    monkeypatch.setattr(
        _agent, "_build_query_constructor",
        lambda **kw: _ConstructorStub(),
    )

    inst = _agent.PydanticAIOtfEncodeAgent(slayer_setup="on-the-fly")
    row = await inst.run_task(
        {"selected_database": "x", "instance_id": "i",
         "amb_user_query": "?", "knowledge_ambiguity": []},
        data_path_base="/tmp", budget=100.0,
        query_mode="slayer", eval_mode="a-interact",
    )

    # kb_encoded reaches the result row via the REAL kb_to_slayer
    # invocation path:
    assert len(row["kb_encoded"]) == 1
    assert row["kb_encoded"][0]["kb_id"] == 5


# ---------------------------------------------------------------------------
# Validate-before-persist write gate (DEV-1454)
#
# The shared MCP server runs a `process_tool_call` hook that, for
# `edit_model`/`create_model`, validates the agent's PROPOSED entity against an
# in-memory copy via a client-side `SlayerQueryEngine` (a `query(limit=0)` that
# RAISES on bad SQL) BEFORE letting the real write happen. An invalid write is
# never persisted — the error is fed back so the agent self-corrects. These
# tests drive `_validate_and_call` directly with a fake engine + fake call_tool
# (no real DB), and pin `_inline_validation_queries` (pure construction).
# ---------------------------------------------------------------------------


class _CallToolLog:
    """Fake MCP ``call_tool(name, args, meta)`` recording every call."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, name, args, meta=None):
        self.calls.append((name, args))
        return f"OK: {name}"

    @property
    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class _FakeEngine:
    """Fake ``SlayerQueryEngine``. ``execute`` (a) asserts it always receives a
    real ``SlayerQuery`` (so the impl can't smuggle raw dicts past validation),
    (b) distinguishes the entity-validation query from the SELECT-1 health probe
    by ``query.source_model``, and (c) raises the configured exceptions."""

    def __init__(self, *, entity_error=None, probe_error=None):
        self.entity_error = entity_error  # Exception instance or None
        self.probe_error = probe_error
        self.executed: list[tuple[str, Any]] = []  # ("entity"|"probe", query)

    async def execute(self, query, **kw):
        assert isinstance(query, SlayerQuery), "validation must build a SlayerQuery"
        sm = query.source_model
        is_probe = isinstance(sm, dict) and sm.get("name") == "__healthcheck__"
        self.executed.append(("probe" if is_probe else "entity", query))
        if is_probe and self.probe_error is not None:
            raise self.probe_error
        if not is_probe and self.entity_error is not None:
            raise self.entity_error
        return object()

    @property
    def kinds(self) -> list[str]:
        return [k for k, _ in self.executed]

    def probe_query(self):
        return next((q for k, q in self.executed if k == "probe"), None)


class _FakeStorage:
    """Records get_model lookups so the datasource-resolution fallback is
    pinnable."""

    def __init__(self):
        self.got: list[str] = []

    async def get_model(self, name, data_source=None):
        from types import SimpleNamespace
        self.got.append(name)
        return SimpleNamespace(data_source="db")


@pytest.mark.asyncio
async def test_write_gate_blocks_invalid_and_does_not_persist():
    """An invalid column SQL (ILIKE) must NOT be persisted: the real
    edit_model call_tool is never made, and the agent gets the error back with
    the SQLite dialect hint + an explicit retry instruction."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _validate_and_call,
    )

    ct = _CallToolLog()
    eng = _FakeEngine(entity_error=SQLAlchemyError('near "ILIKE": syntax error'))
    res = await _validate_and_call(
        ct, "edit_model",
        {"model_name": "properties", "data_source": "db",
         "columns": [{"name": "bad", "sql": "x ILIKE 'y'", "type": "string"}]},
        storage=_FakeStorage(), engine=eng,
    )
    # The real write was NEVER persisted.
    assert "edit_model" not in ct.names
    # Feedback returned to the agent: not-saved + the SQL error + dialect hint
    # + an explicit retry instruction naming the tool.
    assert "VALIDATION FAILED" in res
    assert "NOT saved" in res
    assert "ILIKE" in res
    assert "DOUBLE PRECISION" in res          # SQLite dialect hint present
    assert "call `edit_model` again" in res   # explicit retry instruction
    # The gate is STRICT: an entity-query failure blocks directly — no health
    # probe / fail-open escape (that escape silently persisted broken refs).
    assert eng.kinds == ["entity"]


@pytest.mark.asyncio
async def test_write_gate_blocks_on_value_error():
    """A ValueError (e.g. a measure referencing an unknown column) is an
    expected validation failure too — block + feed back, don't persist."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _validate_and_call,
    )

    ct = _CallToolLog()
    eng = _FakeEngine(entity_error=ValueError("measure references unknown column 'zzz'"))
    res = await _validate_and_call(
        ct, "edit_model",
        {"model_name": "properties", "data_source": "db",
         "measures": [{"name": "m", "formula": "zzz:sum"}]},
        storage=_FakeStorage(), engine=eng,
    )
    assert "edit_model" not in ct.names
    assert "VALIDATION FAILED" in res
    assert "unknown column 'zzz'" in res
    assert eng.kinds == ["entity"]  # strict: block directly, no probe


@pytest.mark.asyncio
async def test_write_gate_blocks_when_data_source_omitted():
    """Even when the edit_model call omits ``data_source``, an invalid write is
    blocked and never persisted. (The strict gate no longer runs a health
    probe, so there is no datasource-resolution-for-probe step to assert.)"""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _validate_and_call,
    )

    ct = _CallToolLog()
    storage = _FakeStorage()
    eng = _FakeEngine(entity_error=SQLAlchemyError('near "ILIKE": syntax error'))
    res = await _validate_and_call(
        ct, "edit_model",
        {"model_name": "properties",  # NO data_source
         "columns": [{"name": "bad", "sql": "x ILIKE 'y'", "type": "string"}]},
        storage=storage, engine=eng,
    )
    assert "edit_model" not in ct.names
    assert "VALIDATION FAILED" in res
    assert eng.probe_query() is None  # strict gate: no health probe at all


@pytest.mark.asyncio
async def test_write_gate_commits_valid_write():
    """A column whose SQL executes is persisted exactly once, unaugmented; no
    health probe is needed on the success path."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _validate_and_call,
    )

    ct = _CallToolLog()
    eng = _FakeEngine()
    args = {"model_name": "properties", "data_source": "db",
            "columns": [{"name": "good", "sql": "beds + baths", "type": "number"}]}
    res = await _validate_and_call(
        ct, "edit_model", args, storage=_FakeStorage(), engine=eng,
    )
    assert ct.calls == [("edit_model", args)]
    assert res == "OK: edit_model"
    assert eng.kinds == ["entity"]  # validated, then committed; no probe


@pytest.mark.asyncio
async def test_write_gate_ignores_read_only_tools():
    """Read-only tools pass straight through; the validation engine is never
    touched."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _validate_and_call,
    )

    ct = _CallToolLog()
    eng = _FakeEngine(entity_error=SQLAlchemyError("boom"))
    res = await _validate_and_call(
        ct, "search", {"question": "x"}, storage=_FakeStorage(), engine=eng,
    )
    assert ct.calls == [("search", {"question": "x"})]
    assert res == "OK: search"
    assert eng.executed == []


@pytest.mark.asyncio
async def test_write_gate_blocks_on_infra_error_and_warns(caplog):
    """A connection/infra-class validation failure now BLOCKS too (strict gate —
    no silent fail-open) so a broken entity is never persisted. But because the
    encoder cannot fix infra, the gate logs a WARNING distinguishing it from a
    referential miss, keeping infra problems visible."""
    import logging
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _validate_and_call,
    )

    ct = _CallToolLog()
    eng = _FakeEngine(
        entity_error=SQLAlchemyError("could not connect to database"),
    )
    with caplog.at_level(logging.WARNING):
        res = await _validate_and_call(
            ct, "edit_model",
            {"model_name": "properties", "data_source": "db",
             "columns": [{"name": "c", "sql": "anything", "type": "string"}]},
            storage=_FakeStorage(), engine=eng,
        )
    # Strict: the write is NOT persisted and the agent gets feedback.
    assert "edit_model" not in ct.names
    assert "VALIDATION FAILED" in res
    assert eng.kinds == ["entity"]  # no probe
    # Infra visibility: a WARNING from the agent gate, naming the tool and the
    # connection/engine nature (so an infra defer isn't mistaken for a normal
    # referential miss).
    agent_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and r.name.startswith("bird_interact_agents")
    ]
    assert agent_warnings, "expected a WARNING from the agent write-gate"
    joined = " ".join(r.getMessage().lower() for r in agent_warnings)
    assert "edit_model" in joined and ("connect" in joined or "engine" in joined)


@pytest.mark.asyncio
async def test_write_gate_fails_open_on_harness_bug():
    """An UNEXPECTED error inside validation (a harness bug, not a SQL error)
    must never block a write."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _validate_and_call,
    )

    class _BoomEngine:
        async def execute(self, query, **kw):
            raise RuntimeError("validation harness exploded")

    ct = _CallToolLog()
    res = await _validate_and_call(
        ct, "edit_model",
        {"model_name": "properties", "data_source": "db",
         "columns": [{"name": "c", "sql": "beds", "type": "number"}]},
        storage=_FakeStorage(), engine=_BoomEngine(),
    )
    assert "edit_model" in ct.names
    assert res == "OK: edit_model"


# --- _inline_validation_queries (pure construction) ------------------------


def test_inline_validation_queries_column_uses_one_off_alias():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _inline_validation_queries,
    )

    qs = _inline_validation_queries("edit_model", {
        "model_name": "m",
        "columns": [{"name": "a", "sql": "a+1", "type": "number"},
                    {"name": "b", "sql": "b*2", "type": "number"}],
    })
    assert len(qs) == 2
    label0, q0 = qs[0]
    assert "m.a" in label0
    sm = q0["source_model"]
    assert sm["source_name"] == "m"
    alias = sm["columns"][0]["name"]
    assert alias != "a"                       # one-off, dodges append-collision
    assert sm["columns"][0]["sql"] == "a+1"   # the agent's SQL is what's tested
    assert q0["dimensions"] == [alias]
    assert q0["limit"] == 0


def test_inline_validation_queries_measure_carries_same_call_columns():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _inline_validation_queries,
    )

    qs = _inline_validation_queries("edit_model", {
        "model_name": "m",
        "columns": [{"name": "rev", "sql": "rev", "type": "number"}],
        "measures": [{"name": "aov", "formula": "rev:sum / *:count"}],
    })
    # one column query + one measure query
    assert len(qs) == 2
    meas_label, meas_q = qs[-1]
    assert "m.aov" in meas_label
    sm = meas_q["source_model"]
    # the same-call column is present under its REAL name so the formula resolves
    assert any(c["name"] == "rev" for c in sm["columns"])
    malias = sm["measures"][0]["name"]
    assert malias != "aov"
    # the AGENT's formula is what's validated (not mangled)
    assert sm["measures"][0]["formula"] == "rev:sum / *:count"
    assert meas_q["measures"] == [{"formula": malias}]
    assert meas_q["limit"] == 0


def test_inline_validation_queries_create_model_table_form():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _inline_validation_queries,
    )

    # sql_table variant — EVERY column AND measure must be validated, not just
    # the first column (Codex finding: a bad later column/measure must not slip
    # past the gate and persist).
    qs = _inline_validation_queries("create_model", {
        "name": "newm", "sql_table": "t", "data_source": "db",
        "columns": [{"name": "x", "sql": "x", "type": "string"},
                    {"name": "y", "sql": "y * 2", "type": "number"}],
        "measures": [{"name": "tot", "formula": "y:sum"}],
    })
    labels = [l for l, _ in qs]
    assert any("newm.x" in l for l in labels)
    assert any("newm.y" in l for l in labels)
    assert any("newm.tot" in l for l in labels)
    # a column query selects that column against the (full) inline model
    col_x = next(q for l, q in qs if "newm.x" in l)
    sm = col_x["source_model"]
    assert sm["name"].startswith("__v")          # ephemeral, never persisted
    assert sm.get("sql_table") == "t"
    assert sm.get("data_source") == "db"
    assert [c["name"] for c in sm["columns"]] == ["x", "y"]   # all cols defined
    assert col_x["dimensions"] == ["x"]
    assert col_x["limit"] == 0
    # the measure query validates the formula against the same inline model
    meas_q = next(q for l, q in qs if "newm.tot" in l)
    assert meas_q["measures"] == [{"formula": "y:sum"}]
    assert meas_q["limit"] == 0

    # sql variant: `sql` is set, `sql_table` absent
    qs2 = _inline_validation_queries("create_model", {
        "name": "m2", "sql": "SELECT 1 AS y", "data_source": "db",
        "columns": [{"name": "y", "sql": "y", "type": "number"}],
    })
    sm2 = qs2[0][1]["source_model"]
    assert sm2.get("sql") == "SELECT 1 AS y"
    assert "sql_table" not in sm2


def test_inline_validation_queries_create_model_backing_query():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _inline_validation_queries,
    )

    qs = _inline_validation_queries("create_model", {
        "name": "agg",
        "query": {"source_model": "orders", "measures": [{"formula": "*:count"}]},
    })
    assert len(qs) == 1
    _, q = qs[0]
    assert q["source_model"] == "orders"
    assert q["limit"] == 0


def test_inline_validation_queries_skips_aggregations_scalar_and_source_swaps():
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _inline_validation_queries,
    )

    # scalar-only edit → nothing to execute-validate
    assert _inline_validation_queries(
        "edit_model", {"model_name": "m", "description": "d"}) == []
    # aggregations are a documented limitation (inline ModelExtension has no
    # aggregations field) → not gated here
    assert _inline_validation_queries(
        "edit_model", {"model_name": "m",
                       "aggregations": [{"name": "wa", "formula": "SUM({v})"}]}) == []
    # a SOURCE swap in the same call would make the inline copy validate the new
    # column against the OLD source → skip rather than mis-validate
    assert _inline_validation_queries(
        "edit_model", {"model_name": "m", "sql_table": "newt",
                       "columns": [{"name": "a", "sql": "a", "type": "number"}]}) == []
    # everything we DO emit must validate against the real SlayerQuery schema
    for _, q in _inline_validation_queries("edit_model", {
            "model_name": "m",
            "columns": [{"name": "a", "sql": "a+1", "type": "number"}]}):
        SlayerQuery.model_validate(q)
