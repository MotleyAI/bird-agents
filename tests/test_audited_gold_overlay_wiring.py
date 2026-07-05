"""DEV-1510: wiring tests for `apply_audited_gold_overlay` invocation.

Acceptance criterion for DEV-1510 requires the audited-gold overlay to fire
for livesqlbench (so cloud reruns evaluate against `audited_sol_sql`). Both
the LOCAL runner (`run.run_evaluation`) and the CLOUD actor
(`cloud.ray_app._load_task_data`) used to short-circuit the overlay with
`if use_audited_gold_sql and not b.gold_required:` — i.e. they SKIPPED
livesqlbench because mini-interact was the only benchmark with an
audited_gold sidecar. After this change:

* Both call sites drop the `not b.gold_required` half of the gate.
* Both pass `benchmark=b` so `apply_audited_gold_overlay` can dispatch
  on `benchmark.audited_gold_layout` (per_db for mini-interact, single_file
  for livesqlbench).

These tests pin both halves of that wiring. The dispatch behaviour itself
is covered by `test_dual_eval.py` (the per_db and single_file unit tests
under "DEV-1510: apply_audited_gold_overlay learns a `benchmark` kwarg").
"""

from __future__ import annotations

import json

import pytest

# DEV-1640: these tests pin the LOCAL in-process per-task wiring / grading by
# monkeypatching agents + graders + loaders, which a spawned worker process
# cannot see. The process pool is now the default, so route run_evaluation
# through the retained legacy single-loop path (identical per-task wiring).
@pytest.fixture(autouse=True)
def _dev1640_force_legacy_inprocess(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_LOCAL_INPROCESS", "1")


# ---------------------------------------------------------------------------
# Local runner — `run.run_evaluation`
# ---------------------------------------------------------------------------


class _OverlayCalled(Exception):
    """Sentinel: the overlay was called with the captured args."""

    def __init__(self, args: tuple, kwargs: dict) -> None:
        super().__init__(repr((args, kwargs)))
        self.args = args
        self.kwargs = kwargs


def _patch_overlay_to_raise_sentinel(monkeypatch) -> None:
    """Replace `apply_audited_gold_overlay` (in BOTH places it can be looked
    up: as a module attr on harness AND as a re-export on run) with a stub
    that records the call and raises `_OverlayCalled`. Tests use the raise
    to short-circuit out of `run_evaluation` immediately after the gate so
    the rest of the function doesn't have to be set up."""

    def _raise(*args, **kwargs):
        raise _OverlayCalled(args, kwargs)

    # `run.py` imports `apply_audited_gold_overlay` at module load:
    #     `from .harness import apply_audited_gold_overlay`
    # so patching the harness symbol alone is NOT enough — we also need to
    # repoint the `run` module-level reference.
    monkeypatch.setattr(
        "bird_interact_agents.harness.apply_audited_gold_overlay", _raise,
    )
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(run_mod, "apply_audited_gold_overlay", _raise)


def _patch_loader_returns(monkeypatch, rows: list[dict]) -> None:
    """Patch `load_benchmark_tasks` so `run_evaluation` doesn't need real
    data on disk to reach the overlay gate."""
    monkeypatch.setattr(
        "bird_interact_agents.harness.load_benchmark_tasks",
        lambda *a, **kw: rows,
    )
    import bird_interact_agents.run as run_mod
    monkeypatch.setattr(run_mod, "load_benchmark_tasks", lambda *a, **kw: rows)


@pytest.mark.asyncio
async def test_run_evaluation_invokes_overlay_for_livesqlbench(
    monkeypatch, tmp_path,
):
    """Pre-fix the local runner skipped the overlay whenever `gold_required`
    (livesqlbench's discriminator). Acceptance criterion for DEV-1510
    requires the overlay to fire for livesqlbench too. The kwargs MUST
    carry `benchmark=LIVESQLBENCH` so the per_db/single_file dispatch
    works."""
    from bird_interact_agents import run as run_mod
    from bird_interact_agents.benchmark import get_benchmark

    _patch_loader_returns(monkeypatch, [
        {"instance_id": "museum_1", "selected_database": "museum",
         "sol_sql": ["SELECT 1"]},
    ])
    _patch_overlay_to_raise_sentinel(monkeypatch)
    # `_maybe_force_wipe_otf` can fail without real data — neuter it.
    monkeypatch.setattr(run_mod, "_maybe_force_wipe_otf", lambda **kw: None)

    output_path = tmp_path / "eval.json"

    with pytest.raises(_OverlayCalled) as exc:
        await run_mod.run_evaluation(
            framework="claude_sdk_otf",
            query_mode="slayer",
            mode="one-shot",
            data_path="ignored",
            data_dir=str(tmp_path / "ignored_dir"),
            output_path=str(output_path),
            concurrency=1,
            limit=None,
            agent_model="anthropic/claude-haiku-4-5-20251001",
            strict=False,
            prompt_cache=False,
            max_depth=1,
            slayer_storage_root=str(tmp_path / "slayer_models"),
            slayer_setup="on-the-fly",
            reasoning_effort=None,
            use_audited_gold_sql=True,
            dataset="livesqlbench-base-lite-sqlite",
            filter_ids=None,
        )

    assert exc.value.kwargs.get("benchmark") is get_benchmark("livesqlbench-base-lite-sqlite"), (
        f"overlay must be called with benchmark=LIVESQLBENCH; got "
        f"{exc.value.kwargs!r}"
    )


@pytest.mark.asyncio
async def test_run_evaluation_does_not_invoke_overlay_when_flag_off(
    monkeypatch, tmp_path,
):
    """Symmetry: when `use_audited_gold_sql=False`, the overlay must NOT
    be invoked for either benchmark (otherwise the runs without the
    audited-gold gate would still get the overlay's `original_sol_sql`
    stamping, which costs no time but changes the dual-eval shape)."""
    from bird_interact_agents import run as run_mod

    _patch_loader_returns(monkeypatch, [
        {"instance_id": "museum_1", "selected_database": "museum",
         "sol_sql": ["SELECT 1"]},
    ])
    _patch_overlay_to_raise_sentinel(monkeypatch)
    monkeypatch.setattr(run_mod, "_maybe_force_wipe_otf", lambda **kw: None)

    # Plant a sentinel at `_make_runner` to abort `run_evaluation` AFTER
    # the overlay gate is decided. The two sentinels disambiguate which
    # path the function took: `_OverlayCalled` = gate fired (wrong here);
    # `_RunnerCalled` = gate correctly passed.
    def _runner_sentinel(**kw):
        raise _RunnerCalled

    monkeypatch.setattr(run_mod, "_make_runner", _runner_sentinel)

    output_path = tmp_path / "eval.json"

    with pytest.raises(_RunnerCalled):
        await run_mod.run_evaluation(
            framework="claude_sdk_otf",
            query_mode="slayer",
            mode="one-shot",
            data_path="ignored",
            data_dir=str(tmp_path / "ignored_dir"),
            output_path=str(output_path),
            concurrency=1,
            limit=None,
            agent_model="anthropic/claude-haiku-4-5-20251001",
            strict=False,
            prompt_cache=False,
            max_depth=1,
            slayer_storage_root=str(tmp_path / "slayer_models"),
            slayer_setup="on-the-fly",
            reasoning_effort=None,
            use_audited_gold_sql=False,
            dataset="livesqlbench-base-lite-sqlite",
            filter_ids=None,
        )


class _RunnerCalled(Exception):
    """Sentinel: control reached `_make_runner`, i.e. past the overlay gate."""


# ---------------------------------------------------------------------------
# Cloud actor — `cloud.ray_app._load_task_data`
# ---------------------------------------------------------------------------


def test_cloud_load_task_data_invokes_overlay_for_livesqlbench(
    monkeypatch, tmp_path,
):
    """Pre-fix: `_load_task_data` had `not get_benchmark(dataset).gold_required`
    in the same gate as `run.py`, so livesqlbench actors silently used the
    raw upstream gold even when the manifest said `use_audited_gold_sql=true`.
    After this change the overlay fires for both layouts; cloud + local
    agree on the agent-visible gold."""
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud import ray_app

    # The dataset jsonl is required by the loader even though we override
    # `load_benchmark_tasks` below — `benchmark_data_file()` must resolve.
    lsb_root = tmp_path / "livesqlbench-base-lite-sqlite"
    lsb_root.mkdir()
    dataset_file = lsb_root / "livesqlbench_data_sqlite.jsonl"
    dataset_file.write_text(
        json.dumps({
            "instance_id": "museum_7",
            "selected_database": "museum",
            "query": "Showcase Failure Risk query",
            "category": "Query",
        }) + "\n"
    )
    monkeypatch.setattr(
        _paths, "benchmark_data_file", lambda *a, **k: dataset_file,
    )

    overlay_calls: list[dict] = []

    def fake_overlay(rows, audited_root, *, benchmark, **extra):
        overlay_calls.append({
            "rows": list(rows),
            "audited_root": audited_root,
            "benchmark": benchmark,
            "extra": extra,
        })
        # Mimic the real overlay's mutation so the rest of `_load_task_data`
        # carries the swapped gold to the caller.
        for r in rows:
            if r["instance_id"] == "museum_7":
                r["sol_sql"] = ["SELECT audited"]

    monkeypatch.setattr(
        "bird_interact_agents.harness.apply_audited_gold_overlay",
        fake_overlay,
    )
    # `_load_task_data` calls `load_benchmark_tasks` to merge the gated gold;
    # short-circuit it for this test so we don't need a real gold sidecar.
    monkeypatch.setattr(
        "bird_interact_agents.harness.load_benchmark_tasks",
        lambda *a, **k: [
            {"instance_id": "museum_7", "selected_database": "museum",
             "sol_sql": ["SELECT raw"]},
        ],
    )

    out = ray_app._load_task_data(
        ["museum_7"],
        dataset="livesqlbench-base-lite-sqlite",
        use_audited_gold_sql=True,
    )

    # Overlay was called (gate flipped) with the explicit benchmark.
    assert overlay_calls, "overlay was not invoked for livesqlbench"
    assert overlay_calls[0]["benchmark"] is get_benchmark("livesqlbench-base-lite-sqlite")
    # And the mutation propagated to the actor's task dict.
    assert out["museum_7"]["sol_sql"] == ["SELECT audited"]


def test_cloud_load_task_data_skips_overlay_when_flag_off(monkeypatch, tmp_path):
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.cloud import ray_app

    lsb_root = tmp_path / "livesqlbench-base-lite-sqlite"
    lsb_root.mkdir()
    dataset_file = lsb_root / "livesqlbench_data_sqlite.jsonl"
    dataset_file.write_text(
        json.dumps({
            "instance_id": "museum_7",
            "selected_database": "museum",
        }) + "\n"
    )
    monkeypatch.setattr(
        _paths, "benchmark_data_file", lambda *a, **k: dataset_file,
    )

    overlay_calls: list[dict] = []

    def fake_overlay(rows, audited_root, *, benchmark, **extra):
        overlay_calls.append({"benchmark": benchmark})

    monkeypatch.setattr(
        "bird_interact_agents.harness.apply_audited_gold_overlay",
        fake_overlay,
    )
    monkeypatch.setattr(
        "bird_interact_agents.harness.load_benchmark_tasks",
        lambda *a, **k: [
            {"instance_id": "museum_7", "selected_database": "museum",
             "sol_sql": ["SELECT raw"]},
        ],
    )

    out = ray_app._load_task_data(
        ["museum_7"],
        dataset="livesqlbench-base-lite-sqlite",
        use_audited_gold_sql=False,
    )

    assert overlay_calls == [], (
        f"overlay must not run when use_audited_gold_sql=False; got "
        f"{overlay_calls!r}"
    )
    assert out["museum_7"]["sol_sql"] == ["SELECT raw"]


def test_cloud_load_task_data_mini_interact_overlay_still_works(
    monkeypatch, tmp_path,
):
    """Regression guard: the fix flips the gate for ALL benchmarks, but
    mini-interact's invocation must keep working (this is the path tested
    by `test_ray_app::test_load_task_data_applies_audited_gold_overlay`)."""
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud import ray_app

    mi_root = tmp_path / "mini-interact"
    mi_root.mkdir()
    dataset_file = mi_root / "mini_interact.jsonl"
    dataset_file.write_text(
        json.dumps({"instance_id": "db_a_1", "selected_database": "db_a"})
        + "\n"
    )
    monkeypatch.setattr(
        _paths, "benchmark_data_file", lambda *a, **k: dataset_file,
    )

    overlay_calls: list[dict] = []

    def fake_overlay(rows, audited_root, *, benchmark, **extra):
        overlay_calls.append({"benchmark": benchmark})

    monkeypatch.setattr(
        "bird_interact_agents.harness.apply_audited_gold_overlay",
        fake_overlay,
    )
    monkeypatch.setattr(
        "bird_interact_agents.harness.load_benchmark_tasks",
        lambda *a, **k: [
            {"instance_id": "db_a_1", "selected_database": "db_a",
             "sol_sql": ["SELECT raw"]},
        ],
    )

    ray_app._load_task_data(
        ["db_a_1"],
        dataset="mini-interact",
        use_audited_gold_sql=True,
    )

    assert overlay_calls, "mini-interact overlay regressed"
    assert overlay_calls[0]["benchmark"] is get_benchmark("mini-interact")
