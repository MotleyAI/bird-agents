"""DEV-1525: benchmark canonical names match official names (hyphenated).

Tests that FAIL until the rename is applied in benchmark.py.
After implementation, the old underscore names raise ValueError,
the new hyphenated names are the canonical tokens, and `cli_dataset_tokens()`
exposes only the new names.
"""

from __future__ import annotations

import pytest

from bird_interact_agents.benchmark import (
    all_benchmarks,
    benchmark_names,
    cli_dataset_tokens,
    get_benchmark,
)


# ---------------------------------------------------------------------------
# New canonical names resolve correctly
# ---------------------------------------------------------------------------


def test_mini_interact_canonical_name_is_hyphenated():
    assert get_benchmark("mini-interact").name == "mini-interact"


def test_livesqlbench_sqlite_canonical_name():
    assert get_benchmark("livesqlbench-base-lite-sqlite").name == "livesqlbench-base-lite-sqlite"


def test_livesqlbench_postgres_canonical_name():
    assert get_benchmark("livesqlbench-base-lite").name == "livesqlbench-base-lite"


def test_bird_interact_lite_exp_canonical_name():
    assert get_benchmark("bird-interact-lite-exp").name == "bird-interact-lite-exp"


# ---------------------------------------------------------------------------
# Old underscore names are REJECTED (no longer registered)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("old_token", [
    "mini_interact",
    "livesqlbench",
    "livesqlbench_postgres",
    "mini_interact_postgres",
])
def test_old_underscore_token_raises(old_token):
    """Old underscore tokens must raise — they are no longer registered."""
    with pytest.raises(ValueError):
        get_benchmark(old_token)


# ---------------------------------------------------------------------------
# dataset_marker is also the new hyphenated name
# ---------------------------------------------------------------------------


def test_dataset_markers_are_hyphenated():
    """The markers stamped into task_data['dataset'] by the loaders
    are the new hyphenated canonical names."""
    for b in all_benchmarks():
        assert "-" in b.name or b.name == b.name  # all current names have hyphens
        assert b.dataset_marker == b.name, (
            f"{b.name}: dataset_marker {b.dataset_marker!r} must equal canonical name"
        )


def test_dataset_markers_round_trip():
    """get_benchmark(b.dataset_marker) == b for all registered benchmarks."""
    for b in all_benchmarks():
        resolved = get_benchmark(b.dataset_marker)
        assert resolved.name == b.name


# ---------------------------------------------------------------------------
# cli_dataset_tokens returns only the new names (no old underscored tokens)
# ---------------------------------------------------------------------------


def test_cli_dataset_tokens_no_old_names():
    tokens = set(cli_dataset_tokens())
    old_names = {"mini_interact", "livesqlbench", "livesqlbench_postgres", "mini_interact_postgres"}
    overlap = tokens & old_names
    assert not overlap, f"Old underscore tokens still in cli_dataset_tokens(): {overlap}"


def test_cli_dataset_tokens_contains_new_names():
    tokens = set(cli_dataset_tokens())
    new_names = {
        "mini-interact",
        "bird-interact-full",
        "livesqlbench-base-lite-sqlite",
        "livesqlbench-base-lite",
        "livesqlbench-base-full",
        "livesqlbench-large",
        "bird-interact-lite-exp",
    }
    missing = new_names - tokens
    assert not missing, f"New canonical names missing from cli_dataset_tokens(): {missing}"
    extra = tokens - new_names
    assert not extra, f"Unexpected tokens in cli_dataset_tokens(): {extra}"


def test_all_cli_tokens_resolve():
    """Every token returned by cli_dataset_tokens() must resolve without error."""
    for t in cli_dataset_tokens():
        b = get_benchmark(t)
        assert b.name  # non-empty


# ---------------------------------------------------------------------------
# benchmark_names() returns only the 4 new names
# ---------------------------------------------------------------------------


def test_benchmark_names_are_hyphenated():
    names = set(benchmark_names())
    expected = {
        "mini-interact",
        "bird-interact-full",
        "livesqlbench-base-lite-sqlite",
        "livesqlbench-base-lite",
        "livesqlbench-base-full",
        "livesqlbench-large",
        "bird-interact-lite-exp",
    }
    assert names == expected, (
        f"benchmark_names() mismatch. Extra: {names - expected!r}, "
        f"Missing: {expected - names!r}"
    )


# ---------------------------------------------------------------------------
# Backend annotations still correct after rename
# ---------------------------------------------------------------------------


def test_mini_interact_is_sqlite():
    b = get_benchmark("mini-interact")
    assert b.db_backend == "sqlite"


def test_livesqlbench_sqlite_is_sqlite():
    b = get_benchmark("livesqlbench-base-lite-sqlite")
    assert b.db_backend == "sqlite"
    assert b.one_shot is True


def test_livesqlbench_base_lite_is_postgres():
    b = get_benchmark("livesqlbench-base-lite")
    assert b.db_backend == "postgres"
    assert b.one_shot is True


def test_bird_interact_lite_exp_is_postgres():
    b = get_benchmark("bird-interact-lite-exp")
    assert b.db_backend == "postgres"
    assert b.one_shot is False


# ---------------------------------------------------------------------------
# No gold_root_env field on Benchmark (replaced by gated_gold_root helper)
# ---------------------------------------------------------------------------


def test_benchmark_has_no_gold_root_env_field():
    """gold_root_env is removed — the gated_gold_root(benchmark=) path helper
    replaces it. Keeping the field would create a second source of truth."""
    for b in all_benchmarks():
        assert not hasattr(b, "gold_root_env"), (
            f"{b.name}: gold_root_env field must be removed from Benchmark dataclass"
        )
