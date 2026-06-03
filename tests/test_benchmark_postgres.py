"""Benchmark registry tests for the two new postgres-backed benchmarks.

Pins the resolution contract (canonical name, dataset marker, cli tokens)
and the per-benchmark facts the harness keys off (db_backend, modes,
isolation, gold requirements, etc.).
"""

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


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolves_livesqlbench_postgres():
    b = get_benchmark("livesqlbench_postgres")
    assert b.name == "livesqlbench_postgres"


def test_resolves_mini_interact_postgres():
    b = get_benchmark("mini_interact_postgres")
    assert b.name == "mini_interact_postgres"


def test_resolves_dataset_markers():
    assert get_benchmark("livesqlbench_postgres").name == "livesqlbench_postgres"
    assert get_benchmark("mini_interact_postgres").name == "mini_interact_postgres"


def test_all_benchmarks_includes_postgres():
    names = {b.name for b in all_benchmarks()}
    assert "livesqlbench_postgres" in names
    assert "mini_interact_postgres" in names


def test_benchmark_names_includes_postgres():
    names = set(benchmark_names())
    assert "livesqlbench_postgres" in names
    assert "mini_interact_postgres" in names


def test_cli_dataset_tokens_includes_postgres():
    tokens = set(cli_dataset_tokens())
    assert "livesqlbench_postgres" in tokens
    assert "mini_interact_postgres" in tokens
    # every advertised token resolves
    for t in tokens:
        get_benchmark(t)


# ---------------------------------------------------------------------------
# db_backend field
# ---------------------------------------------------------------------------


def test_sqlite_benchmarks_have_sqlite_backend():
    assert get_benchmark("mini_interact").db_backend == "sqlite"
    assert get_benchmark("livesqlbench").db_backend == "sqlite"


def test_postgres_benchmarks_have_postgres_backend():
    assert get_benchmark("livesqlbench_postgres").db_backend == "postgres"
    assert get_benchmark("mini_interact_postgres").db_backend == "postgres"


def test_db_backend_default_is_sqlite():
    """A new Benchmark without an explicit db_backend must default to sqlite
    so existing descriptors are unaffected."""
    assert Benchmark.model_fields["db_backend"].default == "sqlite"


def test_db_backend_rejects_unknown_value():
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
            db_backend="mysql",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# livesqlbench_postgres facts
# ---------------------------------------------------------------------------


def test_livesqlbench_postgres_facts():
    b = get_benchmark("livesqlbench_postgres")
    assert b.db_backend == "postgres"
    assert b.data_subdir == "livesqlbench-base-lite-postgres"
    assert b.data_file == "livesqlbench_data.jsonl"
    assert b.one_shot is True
    assert b.gold_required is True
    assert b.per_task_db_isolation is False
    assert set(b.supported_modes) == {"one-shot"}
    assert b.audited_gold_layout == "single_file"
    assert b.container_data_dir == "/data/livesqlbench-postgres"
    assert b.dataset_marker == "livesqlbench_postgres"
    assert b.gold_root_env != ""


# ---------------------------------------------------------------------------
# mini_interact_postgres facts
# ---------------------------------------------------------------------------


def test_mini_interact_postgres_facts():
    b = get_benchmark("mini_interact_postgres")
    assert b.db_backend == "postgres"
    assert b.data_subdir == "mini-interact-postgres"
    assert b.data_file == "mini_interact.jsonl"
    assert b.one_shot is False
    assert b.gold_required is False
    assert b.per_task_db_isolation is False
    assert "a-interact" in b.supported_modes
    assert "one-shot" not in b.supported_modes
    assert b.audited_gold_layout == "single_file"
    assert b.container_data_dir == "/data/mini-interact-postgres"
    assert b.dataset_marker == "mini_interact_postgres"


# ---------------------------------------------------------------------------
# Uniqueness contract (extends test_benchmark.py)
# ---------------------------------------------------------------------------


def test_all_benchmarks_have_distinct_container_dirs():
    bs = list(all_benchmarks())
    dirs = [b.container_data_dir for b in bs]
    assert len(dirs) == len(set(dirs)), f"duplicate container_data_dir: {dirs}"


def test_all_benchmarks_have_distinct_data_subdir_file_pairs():
    bs = list(all_benchmarks())
    pairs = [(b.data_subdir, b.data_file) for b in bs]
    assert len(pairs) == len(set(pairs))


def test_all_benchmarks_have_distinct_dataset_markers():
    bs = list(all_benchmarks())
    markers = [b.dataset_marker for b in bs]
    assert len(markers) == len(set(markers))


# ---------------------------------------------------------------------------
# Loader dataset-marker threading (DEV-1523 Codex finding)
# ---------------------------------------------------------------------------


def _make_minimal_livesqlbench_jsonl(path, instance_ids):
    """Write a minimal livesqlbench JSONL with the given instance_ids."""
    import json
    with open(path, "w") as f:
        for iid in instance_ids:
            f.write(json.dumps({
                "instance_id": iid,
                "selected_database": "alien",
                "category": "Query",
                "query": "How many rows?",
            }) + "\n")


def _make_gold_jsonl(path, instance_ids):
    """Write a minimal gold sidecar with the given instance_ids."""
    import json
    with open(path, "w") as f:
        for iid in instance_ids:
            f.write(json.dumps({
                "instance_id": iid,
                "sol_sql": ["SELECT COUNT(*) FROM t"],
            }) + "\n")


def test_load_benchmark_tasks_postgres_stamps_correct_marker(tmp_path):
    """load_benchmark_tasks for livesqlbench_postgres must stamp
    row['dataset'] = 'livesqlbench_postgres', not 'livesqlbench'.
    The wrong marker causes execute_submit_action to route postgres tasks
    through the SQLite path (Codex finding, DEV-1523)."""
    from bird_interact_agents.harness import load_benchmark_tasks

    data_file = tmp_path / "data.jsonl"
    gold_file = tmp_path / "gold.jsonl"
    _make_minimal_livesqlbench_jsonl(data_file, ["alien_pg_1"])
    _make_gold_jsonl(gold_file, ["alien_pg_1"])

    tasks = load_benchmark_tasks(
        "livesqlbench_postgres",
        str(data_file),
        gold_file=str(gold_file),
        filter_ids=["alien_pg_1"],
    )
    assert tasks, "expected at least one task"
    assert tasks[0]["dataset"] == "livesqlbench_postgres", (
        f"expected dataset_marker='livesqlbench_postgres', got {tasks[0]['dataset']!r}; "
        "harness dispatch keys off this marker to route postgres vs sqlite"
    )


def test_load_benchmark_tasks_sqlite_livesqlbench_still_stamps_livesqlbench(tmp_path):
    """Backward-compat: the SQLite livesqlbench still gets dataset='livesqlbench'."""
    from bird_interact_agents.harness import load_benchmark_tasks

    data_file = tmp_path / "data.jsonl"
    gold_file = tmp_path / "gold.jsonl"
    _make_minimal_livesqlbench_jsonl(data_file, ["alien_1"])
    _make_gold_jsonl(gold_file, ["alien_1"])

    tasks = load_benchmark_tasks(
        "livesqlbench",
        str(data_file),
        gold_file=str(gold_file),
        filter_ids=["alien_1"],
    )
    assert tasks
    assert tasks[0]["dataset"] == "livesqlbench"
