"""Tests for the `kb_to_slayer` tool body.

The tool orchestrates: dep walk → topo sort → per-id encoder runs →
registry dedup → return per-kb result. The tests inject a stub
encoder via `_register_kb_to_slayer`'s factory parameter so no
pydantic-ai Agent is constructed and no LLM call is made.

Codex review findings folded into these tests:

* Finding 3 (verification): encoder claims that name an absent entity
  are downgraded to `status="error"`.
* Finding 5 (cycles): cycle members short-circuit to `status="error",
  error="dependency cycle: ..."` without ever invoking the encoder.
* Finding 8 (cached failures): a failed encoder result is cached and
  returned on subsequent calls (no retry).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic_ai import RunContext

from bird_interact_agents.slayer_otf.kb_memory_encoder import (
    encode_kb_as_memories,
)


DB = "tinydb"


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _kb(kb_id: int, **overrides: Any) -> dict[str, Any]:
    row = {
        "id": kb_id,
        "knowledge": f"KB {kb_id}",
        "description": "d", "definition": "x", "type": "calc",
        "children_knowledge": -1,
    }
    row.update(overrides)
    return row


def _write_storage(tmp_path: Path, rows: list[dict]) -> Path:
    mems = encode_kb_as_memories(DB, rows, deleted_kb_ids=set())
    (tmp_path / "memories.yaml").write_text(
        yaml.safe_dump(mems, sort_keys=False),
    )
    return tmp_path


def _shared(tmp_path: Path):
    """SharedTaskState with a YAMLStorage that's wrapped in a
    permissive verifier — i.e., the loader sees the real storage but
    the post-encode verifier (which checks claimed entity_refs) is
    bypassed via a sentinel attribute so topo/dedup tests don't need
    to also write the entities the stub encoder claims."""
    from slayer.storage.yaml_storage import YAMLStorage

    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        SharedTaskState,
    )
    from bird_interact_agents.harness import SampleStatus

    status = SampleStatus(
        idx=0,
        original_data={
            "selected_database": DB, "instance_id": "t",
            "amb_user_query": "?", "knowledge_ambiguity": [],
        },
        remaining_budget=100.0, total_budget=100.0,
    )
    shared = SharedTaskState(
        status=status, data_path_base="/tmp",
        db_name=DB, amb_user_query="?",
        slayer_storage_dir=str(tmp_path),
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        user_sim_prompt_version="v2",
    )
    # Wrap the real storage so memory-loading works AND entity
    # verification always returns True. Tests that exercise the
    # verifier itself use the bare YAMLStorage via _shared_real.
    real_storage = YAMLStorage(base_dir=str(tmp_path))
    shared._slayer_storage = _PermissiveStorage(real_storage)
    return shared


class _PermissiveStorage:
    """Forwards `list_memories` to the real storage; pretends every
    `get_model(name)` returns a SlayerModel that contains the queried
    entity. Used by topo/dedup tests that don't care about the
    post-encode verifier."""

    def __init__(self, real):
        self._real = real

    async def list_memories(self, *, entities=None):
        return await self._real.list_memories(entities=entities)

    async def get_model(self, name):
        # Return a sentinel object that satisfies every existence
        # probe in `_entity_exists`. Pyright/runtime-wise we only
        # need: `model is not None`, `model.data_source == db_name`,
        # and the relevant `model.columns/measures/aggregations`
        # iterables to contain something matching `.name`.
        from types import SimpleNamespace
        match_all = _MatchAll()
        return SimpleNamespace(
            data_source=DB,
            columns=match_all, measures=match_all,
            aggregations=match_all,
        )


class _MatchAll:
    """Iterable whose `any(c.name == X for c in ...)` is always True."""
    def __iter__(self):
        from types import SimpleNamespace

        class _Anyname:
            def __eq__(self, _other):
                return True
        yield SimpleNamespace(name=_Anyname())


def _deps(shared, self_record_idx: int = 0):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        TaskDeps,
    )

    return TaskDeps(
        shared=shared, depth=1, max_depth=3,
        self_record_idx=self_record_idx,
    )


def _ctx(deps):
    return RunContext(
        deps=deps, model=None, usage=None, prompt="", run_step=0,
    )


def _build_sub_clarifier_with_stub_encoder(stub_encoder):
    """Build a bare pydantic-ai Agent (NOT the full sub-clarifier with
    its default `kb_to_slayer` already registered) and register
    `kb_to_slayer` with the provided stub encoder. This isolates the
    tool-body tests from `_build_sub_clarifier`'s default wiring."""
    from pydantic_ai import Agent
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        TaskDeps,
    )

    agent = Agent(model="test", deps_type=TaskDeps, retries=2)
    factories._register_kb_to_slayer(
        agent,
        model="test", model_settings=None,
        shared_slayer_server=None, self_model_id="test",
        _encoder_runner=stub_encoder,
    )
    return agent


def _tool(agent, name):
    return dict(agent._function_toolset.tools)[name]


# ---------------------------------------------------------------------------
# Happy path — topo order respected, results cached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_to_slayer_encodes_dependencies_in_topo_order(tmp_path):
    """Sub-clarifier asks for [5] but 5 depends on 3. The tool resolves
    the dep DAG, topo-sorts to [3, 5], and the stub encoder records the
    invocation order. Result returned to the LLM only mentions the
    requested id (5)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    _write_storage(tmp_path, rows=[_kb(3), _kb(5, children_knowledge=[3])])
    shared = _shared(tmp_path)

    invocation_order: list[int] = []

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        invocation_order.append(kb_id)
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="m",
                name=f"c_{kb_id}", entity_ref=f"{DB}.m.c_{kb_id}",
            )],
            notes=f"encoded {kb_id}",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[5])

    # Topo order respected:
    assert invocation_order == [3, 5]

    # Registry holds BOTH entries (dep + requested):
    assert {r.kb_id for r in shared.kb_encoded} == {3, 5}

    # Returned JSON only mentions the requested id:
    payload = json.loads(out)
    assert set(payload) == {"5"}
    assert payload["5"]["status"] == "encoded"
    assert payload["5"]["entities"][0]["entity_ref"] == f"{DB}.m.c_5"


@pytest.mark.asyncio
async def test_kb_to_slayer_second_call_hits_registry(tmp_path):
    """Codex finding 8: second call with the same id reuses the cached
    EncoderResult; the encoder is NOT invoked again."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    _write_storage(tmp_path, rows=[_kb(5)])
    shared = _shared(tmp_path)

    call_count = 0

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        nonlocal call_count
        call_count += 1
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="m", name="c",
                entity_ref=f"{DB}.m.c",
            )],
            notes="ok",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[5])
    await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[5])

    assert call_count == 1


@pytest.mark.asyncio
async def test_kb_to_slayer_transitive_child_cached_too(tmp_path):
    """If first call for [5] encoded 3 transitively, a second call for
    [3] hits the cache. The encoder is NOT re-run for 3."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    _write_storage(tmp_path, rows=[_kb(3), _kb(5, children_knowledge=[3])])
    shared = _shared(tmp_path)

    invocation_order: list[int] = []

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        invocation_order.append(kb_id)
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="m",
                name=f"c_{kb_id}", entity_ref=f"{DB}.m.c_{kb_id}",
            )],
            notes="ok",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[5])
    await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[3])

    assert invocation_order == [3, 5]


@pytest.mark.asyncio
async def test_kb_to_slayer_batch_with_shared_dep_encodes_dep_once(tmp_path):
    """[5, 7] where both depend on 3: 3 encodes once, then 5 and 7."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    _write_storage(
        tmp_path,
        rows=[
            _kb(3),
            _kb(5, children_knowledge=[3]),
            _kb(7, children_knowledge=[3]),
        ],
    )
    shared = _shared(tmp_path)

    invocation_order: list[int] = []

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        invocation_order.append(kb_id)
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="m",
                name=f"c{kb_id}", entity_ref=f"{DB}.m.c{kb_id}",
            )],
            notes="",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[5, 7])

    # 3 first, then 5 and 7 (tie broken by ascending id):
    assert invocation_order == [3, 5, 7]


# ---------------------------------------------------------------------------
# Failure isolation + cycle handling (Codex findings 5, 8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_to_slayer_failed_encoder_does_not_poison_independent_kb(
    tmp_path,
):
    """If 3 fails to encode, an independent 7 (no dep on 3) still
    encodes successfully. The failure is isolated."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    _write_storage(tmp_path, rows=[_kb(3), _kb(7)])
    shared = _shared(tmp_path)

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        if kb_id == 3:
            return EncoderResult(
                kb_id=3, status="error", entities=[], notes="",
                error="boom",
            )
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="m", name=f"c{kb_id}",
                entity_ref=f"{DB}.m.c{kb_id}",
            )],
            notes="",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(
        _ctx(deps), kb_ids=[3, 7],
    )
    payload = json.loads(out)

    assert payload["3"]["status"] == "error"
    assert payload["7"]["status"] == "encoded"


@pytest.mark.asyncio
async def test_kb_to_slayer_caches_failure_no_retry(tmp_path):
    """Codex finding 8: failed encoder result IS cached. Second call
    for the same id returns the cached error without re-running the
    encoder."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    _write_storage(tmp_path, rows=[_kb(3)])
    shared = _shared(tmp_path)

    call_count = 0

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        nonlocal call_count
        call_count += 1
        return EncoderResult(
            kb_id=kb_id, status="error", entities=[], notes="",
            error="boom",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out1 = await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[3])
    out2 = await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[3])

    assert call_count == 1
    p1 = json.loads(out1)
    p2 = json.loads(out2)
    assert p1["3"]["status"] == "error" == p2["3"]["status"]
    assert p1["3"]["error"] == p2["3"]["error"] == "boom"


@pytest.mark.asyncio
async def test_kb_to_slayer_cycle_short_circuits_without_calling_encoder(
    tmp_path,
):
    """Codex finding 5: a dep cycle marks the SCC members as errors
    BEFORE invoking the encoder. No encoder call is made for cycle
    members."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    # 2 -> 3 -> 2 cycle, 4 standalone.
    _write_storage(
        tmp_path,
        rows=[
            _kb(2, children_knowledge=[3]),
            _kb(3, children_knowledge=[2]),
            _kb(4),
        ],
    )
    shared = _shared(tmp_path)

    invoked: list[int] = []

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        invoked.append(kb_id)
        return EncoderResult(
            kb_id=kb_id, status="encoded", entities=[], notes="",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(
        _ctx(deps), kb_ids=[2, 3, 4],
    )
    payload = json.loads(out)

    # Cycle members → error, encoder NOT called for them.
    assert payload["2"]["status"] == "error"
    assert payload["3"]["status"] == "error"
    assert "cycle" in payload["2"]["error"].lower()
    assert 2 not in invoked
    assert 3 not in invoked

    # 4 encoded normally:
    assert payload["4"]["status"] == "encoded"
    assert 4 in invoked


# ---------------------------------------------------------------------------
# Concurrency — same kb_id encoded once even under gather() (Codex 2's shim)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_to_slayer_concurrent_calls_encode_once_per_kb(tmp_path):
    """Even with `sequential=True` removed, the per-kb lock keeps a
    single encoder run per id under `asyncio.gather`. This proves the
    defensive shim works."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    _write_storage(tmp_path, rows=[_kb(3)])
    shared = _shared(tmp_path)

    call_count = 0
    enter_event = asyncio.Event()
    proceed_event = asyncio.Event()

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        nonlocal call_count
        call_count += 1
        enter_event.set()
        # Block so the second concurrent caller hits the lock.
        await proceed_event.wait()
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="m", name="c",
                entity_ref=f"{DB}.m.c",
            )],
            notes="",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    tool = _tool(agent, "kb_to_slayer").function

    async def caller():
        return await tool(_ctx(deps), kb_ids=[3])

    task_a = asyncio.create_task(caller())
    # Wait until the first encoder is inside the critical section.
    await asyncio.wait_for(enter_event.wait(), timeout=1.0)
    task_b = asyncio.create_task(caller())
    # Let the first encoder finish.
    proceed_event.set()
    out_a = await task_a
    out_b = await task_b

    assert call_count == 1
    # Both callers see the same encoded result.
    assert json.loads(out_a)["3"]["status"] == "encoded"
    assert json.loads(out_b)["3"]["status"] == "encoded"


# ---------------------------------------------------------------------------
# Empty input + tool wiring (sequential=True on the tool itself)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_to_slayer_empty_list_is_noop(tmp_path):
    """`kb_to_slayer([])` returns an empty JSON map; encoder never runs."""
    _write_storage(tmp_path, rows=[_kb(3)])
    shared = _shared(tmp_path)

    invoked: list[int] = []

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        invoked.append(kb_id)
        raise AssertionError("encoder must not run for empty kb_ids")

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[])

    assert json.loads(out) == {}
    assert invoked == []


def test_kb_to_slayer_tool_is_sequential(tmp_path):
    """Codex finding 2: `kb_to_slayer` itself carries `sequential=True`
    so a model batch emitting multiple calls runs them serially. The
    per-kb lock is defensive on top of this."""
    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        raise AssertionError("not called")

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    tool = _tool(agent, "kb_to_slayer")
    assert tool.sequential is True


# ---------------------------------------------------------------------------
# Tool placement: sub-clarifier YES, constructor NO, root NO
# ---------------------------------------------------------------------------


def test_sub_clarifier_registers_kb_to_slayer():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    agent = factories._build_sub_clarifier(
        model="test", model_settings=None, shared_slayer_server=None,
    )
    # The default sub-clarifier in this package registers kb_to_slayer
    # (the helper is part of the factory by default).
    assert "kb_to_slayer" in dict(agent._function_toolset.tools)


def test_root_clarifier_does_not_register_kb_to_slayer():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    agent = factories._build_root_clarifier(
        model="test", model_settings=None,
        shared_slayer_server=None, max_depth=3,
    )
    assert "kb_to_slayer" not in dict(agent._function_toolset.tools)


def test_query_constructor_does_not_register_kb_to_slayer():
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    agent = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("a", "b"),
    )
    assert "kb_to_slayer" not in dict(agent._function_toolset.tools)


# ---------------------------------------------------------------------------
# Verification step (Codex finding 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_to_slayer_downgrades_to_error_when_claimed_entity_missing(
    tmp_path,
):
    """Codex finding 3: SLayer MCP write tools return error STRINGS,
    not exceptions, so `output_type=EncoderResult` doesn't prove the
    writes succeeded. The wrapper must verify each claimed entity_ref
    exists in storage; if any is missing, downgrade the result to
    `status='error'`.

    Uses REAL YAMLStorage (not the test's permissive wrapper) so the
    verification step can actually fail."""
    from slayer.storage.yaml_storage import YAMLStorage
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    _write_storage(tmp_path, rows=[_kb(3)])
    shared = _shared(tmp_path)
    # Replace permissive storage with the real one so verification
    # can fail on missing entity refs.
    shared._slayer_storage = YAMLStorage(base_dir=str(tmp_path))

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        # Encoder claims success with an entity_ref that doesn't exist
        # in the storage. The wrapper must catch this.
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="nonexistent_model",
                name="ghost", entity_ref=f"{DB}.nonexistent_model.ghost",
            )],
            notes="claimed but never written",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[3])
    payload = json.loads(out)

    assert payload["3"]["status"] == "error"
    assert "ghost" in payload["3"]["error"] or "not found" in payload["3"]["error"].lower()


# ---------------------------------------------------------------------------
# Codex test-review finding 3 — dep edges come from memory `entities`,
# NOT from the parsed body's `children_knowledge`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_to_slayer_dep_walk_uses_memory_entities_not_body(tmp_path):
    """A KB memory whose body says `children_knowledge: [99]` (the
    encoder's `_resolve_cross_refs` would have filtered 99 out if 99
    were deleted) but whose `entities` does NOT carry `memory:db_kb_99`
    must NOT trigger a dep on 99. The implementation reads the
    surviving `entities` refs, not the raw body."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )

    # Build a memory manually whose body lies about its children.
    # `entities` reflects the true (DEV-1455-filtered) state: only the
    # bare datasource entity, no `memory:` refs.
    mems = encode_kb_as_memories(
        DB,
        [
            # 99 deleted; 5's body still references 99 in
            # children_knowledge, but entities was scrubbed.
            _kb(5, children_knowledge=[99]),
        ],
        deleted_kb_ids={99},
    )
    (tmp_path / "memories.yaml").write_text(
        yaml.safe_dump(mems, sort_keys=False),
    )
    shared = _shared(tmp_path)

    invoked: list[int] = []

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        invoked.append(kb_id)
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="m", name=f"c{kb_id}",
                entity_ref=f"{DB}.m.c{kb_id}",
            )],
            notes="",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[5])

    # 99 is NOT walked into; only 5 encodes. If the implementation
    # mistakenly read `children_knowledge` from the body, it would have
    # tried to resolve 99 (which isn't in the row map) and either
    # raised or encoded it.
    assert invoked == [5]
    payload = json.loads(out)
    assert payload["5"]["status"] == "encoded"


# ---------------------------------------------------------------------------
# Codex test-review finding 1 — encoder agent tool surface
# ---------------------------------------------------------------------------


def test_kb_encoder_agent_tool_surface_matches_plan(tmp_path):
    """`_build_kb_encoder` must register `ask_user` and NOT register
    `submit_query`, `spawn_subagent`, or `kb_to_slayer`. (The MCP
    write tools — edit_model, create_model, etc. — come via the shared
    MCP toolset and are not visible on `_function_toolset.tools`, so
    this test asserts the native @agent.tool surface only.)"""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    encoder = factories._build_kb_encoder(
        model="test", model_settings=None, shared_slayer_server=None,
        self_model_id="test",
    )
    tools = dict(encoder._function_toolset.tools)
    assert "ask_user" in tools
    assert "submit_query" not in tools
    assert "spawn_subagent" not in tools
    assert "kb_to_slayer" not in tools


# ---------------------------------------------------------------------------
# Codex test-review finding 2 — 10-round ask_user cap + error fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encoder_run_wraps_pydantic_ai_request_limit_at_10(
    tmp_path, monkeypatch,
):
    """Plan finding #4: encoder runs are capped at 10 ask_user rounds
    on top of the bird-coin budget gate. `_run_kb_encoder` (the
    default encoder runner the kb_to_slayer tool invokes) must pass
    pydantic-ai's `UsageLimits(request_limit=10*2)` (or equivalent
    cap) to the encoder agent's `.run` call so a stuck encoder cannot
    loop indefinitely.

    The cap is asserted by patching the encoder's `.run` to capture
    the `usage_limits` kwarg.
    """
    from pydantic_ai.usage import UsageLimits
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    # Disable litellm cost lookup for the "unknown" stub model id —
    # mirrors the pattern in test_pydantic_ai_recursive_tools.py.
    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    _write_storage(tmp_path, rows=[_kb(3)])
    shared = _shared(tmp_path)

    captured = {}

    class _CapturingAgent:
        async def run(self, *a, **kw):
            captured["usage_limits"] = kw.get("usage_limits")
            from types import SimpleNamespace
            return SimpleNamespace(
                output=EncoderResult(
                    kb_id=3, status="encoded", entities=[], notes="",
                ),
                usage=lambda: SimpleNamespace(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    # Replace the encoder factory so the default `_encoder_runner`
    # uses our capturing agent.
    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as m:
        m.setattr(
            factories, "_build_kb_encoder",
            lambda **kw: _CapturingAgent(),
        )

        # `_build_sub_clarifier` registers `kb_to_slayer` with the
        # default `_encoder_runner` (which goes through
        # `_build_kb_encoder`, now patched).
        agent = factories._build_sub_clarifier(
            model="test", model_settings=None, shared_slayer_server=None,
        )
        deps = _deps(shared)
        await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[3])

    assert isinstance(captured["usage_limits"], UsageLimits)
    # Cap pinned in the plan ("10 ask_user rounds"); the implementation
    # may translate this to `request_limit=10 * 2 = 20` (one model
    # request per ask + one per reply) but MUST be bounded.
    assert captured["usage_limits"].request_limit is not None
    assert captured["usage_limits"].request_limit <= 20


@pytest.mark.asyncio
async def test_encoder_runner_default_invokes_built_encoder(
    tmp_path, monkeypatch,
):
    """The default `_encoder_runner` MUST go through `_build_kb_encoder`
    so an integration run actually constructs the encoder agent.
    Asserted by injecting a build-time spy."""
    from bird_interact_agents import usage as usage_mod
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    monkeypatch.setattr(usage_mod, "_cost_per_token", lambda **_: (0.0, 0.0))

    _write_storage(tmp_path, rows=[_kb(3)])
    shared = _shared(tmp_path)

    build_calls = []

    class _OkAgent:
        async def run(self, *a, **kw):
            from types import SimpleNamespace
            return SimpleNamespace(
                output=EncoderResult(
                    kb_id=3, status="encoded", entities=[], notes="",
                ),
                usage=lambda: SimpleNamespace(
                    input_tokens=0, output_tokens=0,
                    cache_read_tokens=0, cache_write_tokens=0,
                ),
                all_messages=lambda: [],
            )

    def _spy_build(**kw):
        build_calls.append(kw)
        return _OkAgent()

    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as m:
        m.setattr(factories, "_build_kb_encoder", _spy_build)

        # `_build_sub_clarifier` registers `kb_to_slayer` with the
        # default `_encoder_runner` — that runner goes through
        # `_build_kb_encoder` (now spied).
        agent = factories._build_sub_clarifier(
            model="test", model_settings=None, shared_slayer_server=None,
        )
        deps = _deps(shared)
        await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[3])

    assert len(build_calls) >= 1


# ---------------------------------------------------------------------------
# Codex test-review finding 6 — malformed memory contract aligned to plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_to_slayer_returns_error_for_id_with_no_memory(
    tmp_path,
):
    """DEV-1454: a requested id whose memory is absent (e.g. HARD-8-deleted,
    so it never appears in `list_memories`) must surface as a per-kb error,
    NOT silently disappear from the batch and NOT invoke the encoder."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncoderResult,
    )

    # Only kb 3 has a memory; kb 4 was "deleted" (no memory row).
    mems = encode_kb_as_memories(DB, [_kb(3)], deleted_kb_ids=set())
    (tmp_path / "memories.yaml").write_text(
        yaml.safe_dump(mems, sort_keys=False),
    )
    shared = _shared(tmp_path)

    invoked: list[int] = []

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        invoked.append(kb_id)
        return EncoderResult(
            kb_id=kb_id, status="encoded", entities=[], notes="",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    out = await _tool(agent, "kb_to_slayer").function(
        _ctx(deps), kb_ids=[3, 4],
    )
    payload = json.loads(out)

    # 3 encodes normally:
    assert payload["3"]["status"] == "encoded"
    # 4 surfaces as error; the encoder is never called for it.
    assert payload["4"]["status"] == "error"
    assert 4 not in invoked
    assert "unknown" in payload["4"]["error"].lower() or \
        "no memory" in payload["4"]["error"].lower()


# ---------------------------------------------------------------------------
# Codex test-review finding 12 — encoder draws from the shared budget pool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encoder_ask_user_draws_from_shared_budget_pool(tmp_path):
    """Plan decision #10: encoder uses the shared `status.remaining_budget`
    pool with no separate reserve. An ask_user call inside the encoder
    must decrement the same shared budget the sub-clarifier sees."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        EncodedEntity, EncoderResult,
    )
    from bird_interact_agents.harness import ACTION_COSTS

    _write_storage(tmp_path, rows=[_kb(3)])
    shared = _shared(tmp_path)
    initial_budget = shared.status.remaining_budget

    async def stub_encoder(*, kb_id, row, deps_map, ctx):
        # Simulate an ask_user inside the encoder by decrementing
        # the budget via the harness's normal path. We can do this
        # by directly invoking the budget gate (mirrors what
        # ask_user_impl does internally).
        from bird_interact_agents.harness import update_budget
        update_budget(ctx.deps.shared.status, "ask_user")
        return EncoderResult(
            kb_id=kb_id, status="encoded",
            entities=[EncodedEntity(
                kind="column", host_model="m", name=f"c{kb_id}",
                entity_ref=f"{DB}.m.c{kb_id}",
            )],
            notes="",
        )

    agent = _build_sub_clarifier_with_stub_encoder(stub_encoder)
    deps = _deps(shared)
    await _tool(agent, "kb_to_slayer").function(_ctx(deps), kb_ids=[3])

    # Budget decreased by exactly one ask_user cost (no other actions
    # in the stub):
    assert shared.status.remaining_budget == pytest.approx(
        initial_budget - ACTION_COSTS["ask_user"],
    )
