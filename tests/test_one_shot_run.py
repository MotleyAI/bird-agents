"""DEV-1462 — `agent.run_task(eval_mode="one-shot")` for BOTH SLayer agents.

These mirror `test_projection_resolver.py::test_run_task_wires_*` for the
one-shot branch: all three stage builders monkeypatched, so the test
exercises the orchestration without touching real LLMs or MCP.

Pins:
* call order root → resolver → constructor (just like a-interact).
* `eval_mode="one-shot"` accepted; `c-interact` / `oracle` still raise.
* `task["dataset"]=="livesqlbench"` required (Codex #1 programmatic-bypass close).
* user-sim never constructed; no `ask_user` tool in the spawn tree.
* constructor reserve = `submit_query` only (no `2*ask_user`).
* benchmark-scoped `cache_root`/`reference_root` derived from `task["dataset"]`.
* `empty_after_guard` resolver result skips the constructor →
  `never_submitted`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Shared mock factories — identical pattern across both packages.
# ---------------------------------------------------------------------------


def _make_stub_run(output):
    return SimpleNamespace(
        output=output,
        usage=lambda: SimpleNamespace(
            input_tokens=0, output_tokens=0,
            cache_read_tokens=0, cache_write_tokens=0,
        ),
        all_messages=lambda: [],
    )


def _make_stub_agent(output):
    class _Stub:
        async def run(self, *a, **kw):
            return _make_stub_run(output)
    return _Stub()


def _spy_materialize(monkeypatch, agent_mod) -> list:
    """Patch `materialize_task_db` with a spy that records each call.

    Per plan B4c, one-shot `run_task` MUST call `materialize_task_db`
    before the first submit (so concurrent eval-resets land in a per-task
    dir, never on the stable dataset sqlite). Codex flagged that a no-op
    stub here can hide a regression where run_task skips the call
    entirely — the spy + non-empty assertion at the end of each test
    closes that gap.
    """
    calls: list = []

    def _spy(task, base):
        calls.append((task, base))
        return None

    monkeypatch.setattr(agent_mod, "materialize_task_db", _spy)
    return calls


# ===========================================================================
# pydantic_ai_recursive
# ===========================================================================


@pytest.mark.asyncio
async def test_recursive_one_shot_wires_root_resolver_constructor(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as agent_mod
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    call_order: list[str] = []
    captured_confirmed: list = []
    used_oneshot_builders: list[str] = []

    def _root(**kw):
        used_oneshot_builders.append("root")
        return _make_stub_agent("spec from root")

    def _resolver(**kw):
        used_oneshot_builders.append("resolver")
        class _R:
            async def run(self, *a, **kw):
                call_order.append("resolver")
                return _make_stub_run(["col_a", "col_b", "col_c"])
        return _R()

    def _constructor(confirmed_projection, **kw):
        used_oneshot_builders.append("constructor")
        captured_confirmed.append(confirmed_projection)
        class _C:
            async def run(self, *a, **kw):
                call_order.append("constructor")
                return _make_stub_run("constructor done")
        return _C()

    class _RootRunner:
        async def run(self, *a, **kw):
            call_order.append("root")
            return _make_stub_run("spec from root")

    def _root_builds(**kw):
        used_oneshot_builders.append("root")
        return _RootRunner()

    monkeypatch.setattr(agent_mod, "_build_root_clarifier", _root_builds)
    monkeypatch.setattr(
        agent_mod, "_build_projection_resolver_oneshot", _resolver,
    )
    monkeypatch.setattr(
        agent_mod, "_build_query_constructor_oneshot", _constructor,
    )
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed", lambda *a, **kw: None)

    async def _no_storage(**kw):
        return ("", [])

    monkeypatch.setattr(
        agent_mod, "_resolve_otf_task_storage_dir", _no_storage,
    )
    monkeypatch.setattr(
        agent_mod, "_build_shared_slayer_server", lambda *a, **kw: None,
    )
    # No-op materialize: B0 lives in harness; we don't want the test to
    # rely on a real sqlite template.
    materialize_calls = _spy_materialize(monkeypatch, agent_mod)

    inst = PydanticAIRecursiveAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    await inst.run_task(
        task_data={
            "selected_database": "alien",
            "instance_id": "alien_1",
            "amb_user_query": "show me X",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench",
        },
        data_path_base=str(tmp_path),
        budget=30.0,
        query_mode="slayer",
        eval_mode="one-shot",
    )
    assert call_order == ["root", "resolver", "constructor"]
    assert tuple(captured_confirmed[0]) == ("col_a", "col_b", "col_c")
    assert materialize_calls, (
        "one-shot run_task MUST call materialize_task_db before submitting"
    )
    # And it received the live task dict (so db_file_path lands on this run's task).
    assert materialize_calls[0][0]["instance_id"] == "alien_1"


@pytest.mark.asyncio
async def test_recursive_one_shot_rejects_missing_livesqlbench_marker(
    monkeypatch, tmp_path,
):
    """Codex #1: programmatic-bypass close. A one-shot task that arrived
    without the loader's `dataset='livesqlbench'` marker must raise
    BEFORE any stage runs."""
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    inst = PydanticAIRecursiveAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    with pytest.raises(ValueError) as exc_info:
        await inst.run_task(
            task_data={
                "selected_database": "alien",
                "instance_id": "alien_1",
                "amb_user_query": "x",
                # No "dataset" key.
            },
            data_path_base=str(tmp_path),
            budget=30.0,
            query_mode="slayer",
            eval_mode="one-shot",
        )
    msg = str(exc_info.value).lower()
    assert "livesqlbench" in msg or "dataset" in msg


@pytest.mark.asyncio
async def test_recursive_one_shot_rejects_pre_encoded_slayer_setup(
    monkeypatch, tmp_path,
):
    """CodeRabbit close: ``_validate_slayer_setup`` enforces one-shot ⟹
    on-the-fly at the CLI / run_evaluation / make_runner / run_one_task
    boundaries, but a caller that instantiates
    ``PydanticAIRecursiveAgent(slayer_setup="pre-encoded")`` and calls
    ``.run_task(eval_mode="one-shot", ...)`` directly would otherwise
    route LiveSQLBench through the legacy pre-encoded
    ``slayer_models/`` path. The defensive in-run_task check closes
    that bypass."""
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    inst = PydanticAIRecursiveAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="pre-encoded",
    )
    with pytest.raises(ValueError) as exc_info:
        await inst.run_task(
            task_data={
                "selected_database": "alien",
                "instance_id": "alien_1",
                "amb_user_query": "x",
                "dataset": "livesqlbench",
            },
            data_path_base=str(tmp_path),
            budget=30.0,
            query_mode="slayer",
            eval_mode="one-shot",
        )
    msg = str(exc_info.value).lower()
    assert "one-shot" in msg and "on-the-fly" in msg


@pytest.mark.asyncio
async def test_recursive_one_shot_rejects_c_interact_and_oracle(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    inst = PydanticAIRecursiveAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    for bad_mode in ("c-interact", "oracle"):
        with pytest.raises(ValueError):
            await inst.run_task(
                task_data={
                    "selected_database": "alien",
                    "instance_id": "alien_1",
                    "amb_user_query": "x",
                    "dataset": "livesqlbench",
                },
                data_path_base=str(tmp_path),
                budget=30.0,
                query_mode="slayer",
                eval_mode=bad_mode,
            )


@pytest.mark.asyncio
async def test_recursive_one_shot_reserve_is_submit_query_only(
    monkeypatch, tmp_path,
):
    """One-shot constructor reserve = `submit_query` (3), not
    `2*ask_user + submit_query` (7) — there is no ask_user."""
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as agent_mod
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )
    from bird_interact_agents.harness import ACTION_COSTS

    observed_remaining_before_root: list[float] = []

    def _root_builds(**kw):
        class _R:
            async def run(self, *a, **kw):
                # `deps.shared.status.remaining_budget` at root-start = budget - reserve.
                observed_remaining_before_root.append(
                    kw["deps"].shared.status.remaining_budget,
                )
                return _make_stub_run("spec")
        return _R()

    monkeypatch.setattr(agent_mod, "_build_root_clarifier", _root_builds)
    monkeypatch.setattr(
        agent_mod, "_build_projection_resolver_oneshot",
        lambda **kw: _make_stub_agent(["col_a"]),
    )
    monkeypatch.setattr(
        agent_mod, "_build_query_constructor_oneshot",
        lambda **kw: _make_stub_agent("done"),
    )
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed", lambda *a, **kw: None)

    async def _no_storage(**kw):
        return ("", [])

    monkeypatch.setattr(
        agent_mod, "_resolve_otf_task_storage_dir", _no_storage,
    )
    monkeypatch.setattr(
        agent_mod, "_build_shared_slayer_server", lambda *a, **kw: None,
    )
    materialize_calls = _spy_materialize(monkeypatch, agent_mod)

    inst = PydanticAIRecursiveAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    await inst.run_task(
        task_data={
            "selected_database": "alien",
            "instance_id": "alien_1",
            "amb_user_query": "x",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench",
        },
        data_path_base=str(tmp_path), budget=30.0,
        query_mode="slayer", eval_mode="one-shot",
    )
    # remaining_at_root_start = 30 - submit_query (3) = 27
    expected = 30.0 - ACTION_COSTS["submit_query"]
    assert observed_remaining_before_root == [expected], (
        f"one-shot reserve must be submit_query (3) only; "
        f"got remaining={observed_remaining_before_root}, expected {expected}"
    )
    assert materialize_calls, (
        "one-shot run_task MUST call materialize_task_db (reserve test)"
    )


@pytest.mark.asyncio
async def test_recursive_one_shot_passes_benchmark_scoped_cache_root(
    monkeypatch, tmp_path,
):
    """One-shot `run_task` must derive `benchmark` from `task['dataset']`
    and pass it through `_resolve_otf_task_storage_dir`."""
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as agent_mod
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    captured_kw: list = []

    async def _capture_storage(**kw):
        captured_kw.append(kw)
        return ("", [])

    monkeypatch.setattr(
        agent_mod, "_resolve_otf_task_storage_dir", _capture_storage,
    )
    monkeypatch.setattr(
        agent_mod, "_build_root_clarifier",
        lambda **kw: _make_stub_agent("spec"),
    )
    monkeypatch.setattr(
        agent_mod, "_build_projection_resolver_oneshot",
        lambda **kw: _make_stub_agent(["c1"]),
    )
    monkeypatch.setattr(
        agent_mod, "_build_query_constructor_oneshot",
        lambda **kw: _make_stub_agent("done"),
    )
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(
        agent_mod, "_build_shared_slayer_server", lambda *a, **kw: None,
    )
    materialize_calls = _spy_materialize(monkeypatch, agent_mod)

    inst = PydanticAIRecursiveAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    await inst.run_task(
        task_data={
            "selected_database": "alien",
            "instance_id": "alien_1",
            "amb_user_query": "x",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench",
        },
        data_path_base=str(tmp_path), budget=30.0,
        query_mode="slayer", eval_mode="one-shot",
    )
    assert captured_kw, "_resolve_otf_task_storage_dir must have been called"
    assert captured_kw[0].get("benchmark") == "livesqlbench", (
        f"_resolve_otf_task_storage_dir must receive benchmark='livesqlbench'; "
        f"got kwargs={captured_kw[0]}"
    )
    assert materialize_calls, (
        "one-shot run_task MUST call materialize_task_db (benchmark-scope test)"
    )


@pytest.mark.asyncio
async def test_recursive_one_shot_empty_resolver_skips_constructor(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as agent_mod
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        PydanticAIRecursiveAgent,
    )

    constructor_calls: list = []

    def _fail_if_built(**kw):
        constructor_calls.append(kw)
        pytest.fail("constructor must NOT be built on empty_after_guard")

    monkeypatch.setattr(
        agent_mod, "_build_root_clarifier",
        lambda **kw: _make_stub_agent("spec"),
    )
    # Resolver always returns []
    class _AlwaysEmpty:
        async def run(self, *a, **kw):
            return _make_stub_run([])
    monkeypatch.setattr(
        agent_mod, "_build_projection_resolver_oneshot",
        lambda **kw: _AlwaysEmpty(),
    )
    monkeypatch.setattr(
        agent_mod, "_build_query_constructor_oneshot", _fail_if_built,
    )
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed", lambda *a, **kw: None)

    async def _no_storage(**kw):
        return ("", [])

    monkeypatch.setattr(
        agent_mod, "_resolve_otf_task_storage_dir", _no_storage,
    )
    monkeypatch.setattr(
        agent_mod, "_build_shared_slayer_server", lambda *a, **kw: None,
    )
    materialize_calls = _spy_materialize(monkeypatch, agent_mod)

    inst = PydanticAIRecursiveAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    row = await inst.run_task(
        task_data={
            "selected_database": "alien",
            "instance_id": "alien_1",
            "amb_user_query": "x",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench",
        },
        data_path_base=str(tmp_path), budget=30.0,
        query_mode="slayer", eval_mode="one-shot",
    )
    assert constructor_calls == []
    assert row["submission_status"] == "never_submitted"
    assert row.get("projection_resolver_status") == "empty_after_guard"
    assert materialize_calls, (
        "one-shot run_task MUST call materialize_task_db even on the "
        "empty-resolver short-circuit path (the eval would otherwise "
        "have raced on the next call's reset)"
    )


# ===========================================================================
# pydantic_ai_otf_encode — parity with the recursive cases above.
# ===========================================================================


@pytest.mark.asyncio
async def test_otf_encode_one_shot_wires_root_resolver_constructor(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as agent_mod
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )

    call_order: list[str] = []
    captured_confirmed: list = []

    def _root_builds(**kw):
        class _R:
            async def run(self, *a, **kw):
                call_order.append("root")
                return _make_stub_run("spec from root")
        return _R()

    monkeypatch.setattr(agent_mod, "_build_root_clarifier", _root_builds)

    # otf_encode resolver in one-shot DELIVERS the projection via
    # submit_projection into deps.projection_submission, then `_run_projection_resolver`
    # reads from deps. Mock the resolver to write to deps and return a stub run.
    def _resolver_builds(**kw):
        class _R2:
            async def run(self, *a, **kw):
                kw["deps"].projection_submission = ["c1", "c2"]
                call_order.append("resolver")
                return _make_stub_run("did submit")
        return _R2()

    monkeypatch.setattr(
        agent_mod, "_build_projection_resolver_oneshot", _resolver_builds,
    )

    def _constructor_builds(confirmed_projection, **kw):
        captured_confirmed.append(confirmed_projection)
        class _C:
            async def run(self, *a, **kw):
                call_order.append("constructor")
                return _make_stub_run("done")
        return _C()

    monkeypatch.setattr(
        agent_mod, "_build_query_constructor_oneshot", _constructor_builds,
    )
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed", lambda *a, **kw: None)

    async def _no_storage(**kw):
        return ("", [])

    monkeypatch.setattr(
        agent_mod, "_resolve_otf_task_storage_dir", _no_storage,
    )
    monkeypatch.setattr(
        agent_mod, "_build_shared_slayer_server", lambda *a, **kw: None,
    )
    materialize_calls = _spy_materialize(monkeypatch, agent_mod)
    monkeypatch.setattr(
        agent_mod, "make_setup_build_encoder",
        lambda **kw: lambda *a, **kw2: None,
    )
    # YAMLStorage init touches disk; stub it out.
    monkeypatch.setattr(
        agent_mod, "YAMLStorage", lambda **kw: SimpleNamespace(),
    )

    inst = PydanticAIOtfEncodeAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    await inst.run_task(
        task_data={
            "selected_database": "alien",
            "instance_id": "alien_1",
            "amb_user_query": "x",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench",
        },
        data_path_base=str(tmp_path), budget=30.0,
        query_mode="slayer", eval_mode="one-shot",
    )
    assert call_order == ["root", "resolver", "constructor"]
    assert tuple(captured_confirmed[0]) == ("c1", "c2")
    assert materialize_calls, (
        "otf_encode one-shot run_task MUST call materialize_task_db too"
    )
    assert materialize_calls[0][0]["instance_id"] == "alien_1"


@pytest.mark.asyncio
async def test_otf_encode_one_shot_passes_benchmark_scoped_reference_root(
    monkeypatch, tmp_path,
):
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as agent_mod
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )

    captured_kw: list = []

    async def _capture_storage(**kw):
        captured_kw.append(kw)
        return ("", [])

    monkeypatch.setattr(
        agent_mod, "_resolve_otf_task_storage_dir", _capture_storage,
    )
    monkeypatch.setattr(
        agent_mod, "_build_root_clarifier",
        lambda **kw: _make_stub_agent("spec"),
    )

    def _resolver_builds(**kw):
        class _R2:
            async def run(self, *a, **kw):
                kw["deps"].projection_submission = ["c1"]
                return _make_stub_run("submitted")
        return _R2()

    monkeypatch.setattr(
        agent_mod, "_build_projection_resolver_oneshot", _resolver_builds,
    )
    monkeypatch.setattr(
        agent_mod, "_build_query_constructor_oneshot",
        lambda **kw: _make_stub_agent("done"),
    )
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(
        agent_mod, "_build_shared_slayer_server", lambda *a, **kw: None,
    )
    materialize_calls = _spy_materialize(monkeypatch, agent_mod)
    monkeypatch.setattr(
        agent_mod, "make_setup_build_encoder",
        lambda **kw: lambda *a, **kw2: None,
    )
    monkeypatch.setattr(
        agent_mod, "YAMLStorage", lambda **kw: SimpleNamespace(),
    )

    inst = PydanticAIOtfEncodeAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    await inst.run_task(
        task_data={
            "selected_database": "alien",
            "instance_id": "alien_1",
            "amb_user_query": "x",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench",
        },
        data_path_base=str(tmp_path), budget=30.0,
        query_mode="slayer", eval_mode="one-shot",
    )
    assert captured_kw[0].get("benchmark") == "livesqlbench"
    assert materialize_calls, (
        "otf_encode one-shot run_task MUST call materialize_task_db "
        "(benchmark-scope test)"
    )


# ---------------------------------------------------------------------------
# Deeper benchmark-scope check (Codex Medium #4): the previous tests stub
# `_resolve_otf_task_storage_dir` and verify it receives benchmark="livesqlbench"
# — but they can't catch a regression where the real resolver ignores that
# kwarg and calls `paths.*_root()` with no benchmark, silently falling back
# to mini-interact. Drive the REAL resolver with spies on the path helpers.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recursive_real_resolver_passes_benchmark_to_paths(monkeypatch, tmp_path):
    from bird_interact_agents.agents.pydantic_ai_recursive import agent as agent_mod

    seen_benchmark: list = []

    real_cache_root = agent_mod._paths.slayer_otf_cache_root

    def spy_cache_root(*, benchmark=None):
        seen_benchmark.append(("cache_root", benchmark))
        return tmp_path / "cache" / (benchmark or "default")

    monkeypatch.setattr(
        agent_mod._paths, "slayer_otf_cache_root", spy_cache_root,
    )

    async def fake_ensure_db_cache(db, *, cache_root, mini_interact_root, benchmark=None, force=False):
        from bird_interact_agents.slayer_otf import cache as cache_mod
        Path(cache_root).mkdir(parents=True, exist_ok=True)
        (Path(cache_root) / db).mkdir(parents=True, exist_ok=True)
        return cache_mod.CacheEntry(
            cache_dir=Path(cache_root) / db, fingerprint="dead", kb_rows=[],
        )

    async def fake_prepare(*, db, deleted_kb_ids, cache_entry, work_dir,
                           mini_interact_root, db_root=None):
        scratch = Path(work_dir) / db
        scratch.mkdir(parents=True, exist_ok=True)
        return scratch

    monkeypatch.setattr(agent_mod, "ensure_db_cache", fake_ensure_db_cache)
    monkeypatch.setattr(agent_mod, "prepare_task_storage", fake_prepare)

    task = {"instance_id": "alien_1", "selected_database": "alien"}
    await agent_mod._resolve_otf_task_storage_dir(
        db_name="alien", task_data=task,
        data_path_base=str(tmp_path / "data"),
        benchmark="livesqlbench",
    )
    assert ("cache_root", "livesqlbench") in seen_benchmark, (
        "real _resolve_otf_task_storage_dir MUST pass benchmark='livesqlbench' "
        f"to paths.slayer_otf_cache_root; got: {seen_benchmark!r}"
    )


@pytest.mark.asyncio
async def test_otf_encode_real_resolver_passes_benchmark_to_paths(monkeypatch, tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as agent_mod

    seen_benchmark: list = []

    def spy_cache_root(*, benchmark=None):
        seen_benchmark.append(("cache_root", benchmark))
        return tmp_path / "cache" / (benchmark or "default")

    def spy_ref_root(*, benchmark=None):
        seen_benchmark.append(("ref_root", benchmark))
        return tmp_path / "ref" / (benchmark or "default")

    monkeypatch.setattr(
        agent_mod._paths, "slayer_otf_cache_root", spy_cache_root,
    )
    monkeypatch.setattr(
        agent_mod._paths, "slayer_models_otf_root", spy_ref_root,
    )

    async def fake_ensure_db_reference(db, *, reference_root, cache_root,
                                        mini_interact_root, build_encoder,
                                        force=False, db_root=None, benchmark=None):
        from bird_interact_agents.slayer_otf.reference_build import ReferenceEntry
        Path(reference_root).mkdir(parents=True, exist_ok=True)
        (Path(reference_root) / db).mkdir(parents=True, exist_ok=True)
        return ReferenceEntry(
            reference_dir=Path(reference_root) / db,
            fingerprint="dead",
        )

    async def fake_build_variant(*, canonical_storage_root, db_name,
                                  deleted_kb_ids, work_dir, mini_interact_root,
                                  db_root=None):
        # DEV-1462 round-2: build_task_variant_storage gained a `db_root`
        # kwarg (Codex finding); stub must accept it.
        out = Path(work_dir) / db_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(
        agent_mod, "ensure_db_reference", fake_ensure_db_reference,
    )
    monkeypatch.setattr(
        agent_mod, "build_task_variant_storage", fake_build_variant,
    )

    task = {"instance_id": "alien_1", "selected_database": "alien"}
    await agent_mod._resolve_otf_task_storage_dir(
        db_name="alien", task_data=task,
        data_path_base=str(tmp_path / "data"),
        build_encoder=lambda *a, **kw: None,
        benchmark="livesqlbench",
    )
    assert ("cache_root", "livesqlbench") in seen_benchmark, (
        "otf_encode _resolve_otf_task_storage_dir MUST pass benchmark to "
        f"paths.slayer_otf_cache_root; got: {seen_benchmark!r}"
    )
    assert ("ref_root", "livesqlbench") in seen_benchmark, (
        "otf_encode _resolve_otf_task_storage_dir MUST pass benchmark to "
        f"paths.slayer_models_otf_root; got: {seen_benchmark!r}"
    )


@pytest.mark.asyncio
async def test_otf_encode_one_shot_rejects_c_interact_and_oracle(monkeypatch, tmp_path):
    """Parity with the recursive case: otf_encode one-shot still rejects
    the interactive modes — `eval_mode` is one of {a-interact, one-shot}."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )

    inst = PydanticAIOtfEncodeAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    for bad_mode in ("c-interact", "oracle"):
        with pytest.raises(ValueError):
            await inst.run_task(
                task_data={
                    "selected_database": "alien",
                    "instance_id": "alien_1",
                    "amb_user_query": "x",
                    "dataset": "livesqlbench",
                },
                data_path_base=str(tmp_path), budget=30.0,
                query_mode="slayer", eval_mode=bad_mode,
            )


@pytest.mark.asyncio
async def test_otf_encode_one_shot_reserve_is_submit_query_only(monkeypatch, tmp_path):
    """Parity with the recursive case: otf_encode one-shot also drops
    the `2*ask_user` part of the reserve (no ask_user anywhere)."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as agent_mod
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )
    from bird_interact_agents.harness import ACTION_COSTS

    observed_remaining_before_root: list[float] = []

    def _root_builds(**kw):
        class _R:
            async def run(self, *a, **kw):
                observed_remaining_before_root.append(
                    kw["deps"].shared.status.remaining_budget,
                )
                return _make_stub_run("spec")
        return _R()

    def _resolver_builds(**kw):
        class _R2:
            async def run(self, *a, **kw):
                kw["deps"].projection_submission = ["c1"]
                return _make_stub_run("submitted")
        return _R2()

    monkeypatch.setattr(agent_mod, "_build_root_clarifier", _root_builds)
    monkeypatch.setattr(
        agent_mod, "_build_projection_resolver_oneshot", _resolver_builds,
    )
    monkeypatch.setattr(
        agent_mod, "_build_query_constructor_oneshot",
        lambda **kw: _make_stub_agent("done"),
    )
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed", lambda *a, **kw: None)

    async def _no_storage(**kw):
        return ("", [])

    monkeypatch.setattr(
        agent_mod, "_resolve_otf_task_storage_dir", _no_storage,
    )
    monkeypatch.setattr(
        agent_mod, "_build_shared_slayer_server", lambda *a, **kw: None,
    )
    materialize_calls = _spy_materialize(monkeypatch, agent_mod)
    monkeypatch.setattr(
        agent_mod, "make_setup_build_encoder",
        lambda **kw: lambda *a, **kw2: None,
    )
    monkeypatch.setattr(
        agent_mod, "YAMLStorage", lambda **kw: SimpleNamespace(),
    )

    inst = PydanticAIOtfEncodeAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    await inst.run_task(
        task_data={
            "selected_database": "alien",
            "instance_id": "alien_1",
            "amb_user_query": "x",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench",
        },
        data_path_base=str(tmp_path), budget=30.0,
        query_mode="slayer", eval_mode="one-shot",
    )
    expected = 30.0 - ACTION_COSTS["submit_query"]
    assert observed_remaining_before_root == [expected]
    assert materialize_calls, (
        "otf_encode one-shot run_task MUST call materialize_task_db "
        "(reserve test)"
    )


@pytest.mark.asyncio
async def test_otf_encode_one_shot_empty_resolver_skips_constructor(monkeypatch, tmp_path):
    """Parity with the recursive case: empty submit_projection (both
    attempts) → skip constructor, finalize as never_submitted."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import agent as agent_mod
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )

    constructor_calls: list = []

    def _fail_if_built(**kw):
        constructor_calls.append(kw)
        pytest.fail("constructor must NOT be built on empty_after_guard")

    monkeypatch.setattr(
        agent_mod, "_build_root_clarifier",
        lambda **kw: _make_stub_agent("spec"),
    )

    # Resolver never writes to deps.projection_submission → wrapper
    # interprets that as empty and falls into empty_after_guard.
    class _NeverSubmits:
        async def run(self, *a, **kw):
            return _make_stub_run("text, but no submit")
    monkeypatch.setattr(
        agent_mod, "_build_projection_resolver_oneshot",
        lambda **kw: _NeverSubmits(),
    )
    monkeypatch.setattr(
        agent_mod, "_build_query_constructor_oneshot", _fail_if_built,
    )
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed", lambda *a, **kw: None)

    async def _no_storage(**kw):
        return ("", [])

    monkeypatch.setattr(
        agent_mod, "_resolve_otf_task_storage_dir", _no_storage,
    )
    monkeypatch.setattr(
        agent_mod, "_build_shared_slayer_server", lambda *a, **kw: None,
    )
    materialize_calls = _spy_materialize(monkeypatch, agent_mod)
    monkeypatch.setattr(
        agent_mod, "make_setup_build_encoder",
        lambda **kw: lambda *a, **kw2: None,
    )
    monkeypatch.setattr(
        agent_mod, "YAMLStorage", lambda **kw: SimpleNamespace(),
    )

    inst = PydanticAIOtfEncodeAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    row = await inst.run_task(
        task_data={
            "selected_database": "alien",
            "instance_id": "alien_1",
            "amb_user_query": "x",
            "knowledge_ambiguity": [],
            "dataset": "livesqlbench",
        },
        data_path_base=str(tmp_path), budget=30.0,
        query_mode="slayer", eval_mode="one-shot",
    )
    assert constructor_calls == []
    assert row["submission_status"] == "never_submitted"
    assert row.get("projection_resolver_status") == "empty_after_guard"
    assert materialize_calls, (
        "otf_encode one-shot run_task MUST call materialize_task_db on the "
        "empty_after_guard short-circuit path too"
    )


@pytest.mark.asyncio
async def test_otf_encode_one_shot_rejects_missing_marker(monkeypatch, tmp_path):
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        PydanticAIOtfEncodeAgent,
    )

    inst = PydanticAIOtfEncodeAgent(
        model="anthropic/claude-sonnet-4-5", slayer_setup="on-the-fly",
    )
    with pytest.raises(ValueError) as exc_info:
        await inst.run_task(
            task_data={
                "selected_database": "alien", "instance_id": "x",
                "amb_user_query": "x",  # no `dataset`
            },
            data_path_base=str(tmp_path), budget=30.0,
            query_mode="slayer", eval_mode="one-shot",
        )
    msg = str(exc_info.value).lower()
    assert "livesqlbench" in msg or "dataset" in msg
