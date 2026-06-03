"""The Benchmark registry — one descriptor per benchmark, consumed symmetrically
by the local runner and the cloud submit/actor/image. Adding a benchmark is a
single BENCHMARKS entry; these tests pin the registry's resolution contract and
the per-benchmark facts the rest of the harness keys off."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bird_interact_agents.benchmark import (
    Benchmark,
    all_benchmarks,
    benchmark_names,
    cli_dataset_tokens,
    get_benchmark,
)


def test_resolves_canonical_name():
    assert get_benchmark("mini_interact").name == "mini_interact"
    assert get_benchmark("livesqlbench").name == "livesqlbench"


def test_resolves_hyphen_alias_to_underscore_canonical():
    """`--dataset mini-interact` (hyphen) is a back-compat alias for the
    underscore-canonical `mini_interact`."""
    assert get_benchmark("mini-interact").name == "mini_interact"


def test_resolves_dataset_marker():
    """The task-data marker stamped by the loaders must resolve back to its
    benchmark (the agents derive their benchmark from `task_data['dataset']`)."""
    for b in all_benchmarks():
        assert get_benchmark(b.dataset_marker).name == b.name


def test_unknown_token_raises():
    with pytest.raises(ValueError) as exc:
        get_benchmark("not-a-benchmark")
    assert "benchmark" in str(exc.value).lower()


_SQLITE_BENCHMARKS = {"mini_interact", "livesqlbench"}
_POSTGRES_BENCHMARKS = {"livesqlbench_postgres", "mini_interact_postgres"}
_ALL_BENCHMARKS = _SQLITE_BENCHMARKS | _POSTGRES_BENCHMARKS


def test_registry_membership_and_helpers():
    assert _ALL_BENCHMARKS <= set(benchmark_names())
    assert _ALL_BENCHMARKS <= {b.name for b in all_benchmarks()}
    # argparse `--dataset` choices: canonical names + aliases, all resolvable.
    tokens = set(cli_dataset_tokens())
    assert {"mini_interact", "mini-interact", "livesqlbench"} | _POSTGRES_BENCHMARKS <= tokens
    for t in tokens:
        get_benchmark(t)  # every advertised token resolves


def test_benchmark_is_frozen():
    b = get_benchmark("mini_interact")
    with pytest.raises(ValidationError):  # pydantic frozen → ValidationError
        b.name = "mutated"  # type: ignore[misc]


def test_mini_interact_facts():
    b = get_benchmark("mini_interact")
    assert b.dataset_marker == "mini_interact"
    assert b.data_subdir == "mini-interact"
    assert b.data_file == "mini_interact.jsonl"
    assert b.one_shot is False
    assert b.gold_required is False
    assert b.per_task_db_isolation is False
    assert b.container_data_dir == "/data/mini-interact"
    assert "a-interact" in b.supported_modes and "oracle" in b.supported_modes
    assert "one-shot" not in b.supported_modes


def test_livesqlbench_facts():
    b = get_benchmark("livesqlbench")
    assert b.dataset_marker == "livesqlbench"
    assert b.data_subdir == "livesqlbench-base-lite-sqlite"
    assert b.data_file == "livesqlbench_data_sqlite.jsonl"
    assert b.one_shot is True
    assert b.gold_required is True
    assert b.per_task_db_isolation is True
    assert b.container_data_dir == "/data/livesqlbench"
    assert set(b.supported_modes) == {"one-shot", "oracle"}


def test_distinct_container_dirs_and_data_files():
    """No two benchmarks may share a container data dir or data filename — that
    would re-introduce the cross-benchmark collision this registry prevents."""
    bs = all_benchmarks()
    assert len({b.container_data_dir for b in bs}) == len(bs)
    assert len({(b.data_subdir, b.data_file) for b in bs}) == len(bs)


def test_type_is_benchmark():
    assert isinstance(get_benchmark("mini_interact"), Benchmark)


# ---------------------------------------------------------------------------
# DEV-1510: audited_gold_layout — mini-interact stays per-db (each db gets its
# own sidecar dir); livesqlbench uses a single jsonl across all dbs (its dbs
# share names with mini-interact, so a per-db dir under audited_gold/ would
# collide). A forgotten `audited_gold_layout` on a new benchmark must NOT
# silently land in the per-db root and mix artifacts.
# ---------------------------------------------------------------------------


def test_audited_gold_layout_default_is_per_db():
    """Default preserves the existing on-disk shape (mini-interact contract)."""
    assert Benchmark.model_fields["audited_gold_layout"].default == "per_db"


def test_mini_interact_audited_gold_layout_single_file():
    """DEV-1515: mini-interact moved to single_file (matches livesqlbench).
    Per-DB JSONLs were consolidated into ``audited_gold/mini_interact_audited.jsonl``."""
    assert get_benchmark("mini_interact").audited_gold_layout == "single_file"


def test_livesqlbench_audited_gold_layout_single_file():
    """DEV-1510: livesqlbench routes audited gold through one consolidated
    `audited_gold/livesqlbench_audited.jsonl`. Distinct from mini-interact
    so DBs with shared names (alien, museum, etc.) don't collide."""
    assert get_benchmark("livesqlbench").audited_gold_layout == "single_file"


def test_audited_gold_layout_rejects_unknown_value():
    """`audited_gold_layout` is a closed Literal — typos must fail at
    descriptor construction time, not at lookup time inside the overlay."""
    with pytest.raises(ValidationError):
        Benchmark(
            name="x",
            dataset_marker="x",
            data_subdir="x",
            data_file="x.jsonl",
            data_root_env="X_ROOT",
            data_file_env="X_DATA",
            supported_modes=("one-shot",),
            one_shot=True,
            gold_required=False,
            per_task_db_isolation=False,
            container_data_dir="/data/x",
            audited_gold_layout="bogus",  # type: ignore[arg-type]
        )
