"""Benchmark registry — one descriptor per benchmark, the single source of truth
consumed symmetrically by the local runner (`run.py`/`harness.py`/`paths.py`) and
the cloud runner (`cloud/{cli,driver,ray_app,image}.py`).

Adding a benchmark is one `BENCHMARKS` entry; no consumer edits. Tokens are
underscore-canonical (`mini_interact`, `livesqlbench`); the hyphenated
`mini-interact` is kept as a back-compat CLI alias. The dataset marker (stamped
into `task_data["dataset"]` by the loaders) also resolves back to its benchmark,
so an agent that derives its benchmark from task data and a CLI that parses
`--dataset` both funnel through `get_benchmark()`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Benchmark(BaseModel):
    """Immutable per-benchmark descriptor. Every per-benchmark fact the harness
    branches on lives here (no scattered `"livesqlbench"` literals)."""

    model_config = ConfigDict(frozen=True)

    name: str
    """Canonical underscore token, e.g. ``mini_interact`` / ``livesqlbench``."""
    cli_aliases: tuple[str, ...] = ()
    """Extra `--dataset` spellings that resolve to this benchmark (e.g. the
    hyphenated ``mini-interact``)."""
    dataset_marker: str
    """Stamped into ``task_data["dataset"]`` by the loaders; drives the agents'
    benchmark derivation and the one-shot guard."""

    # Data location (resolved by paths.benchmark_data_root / _data_file).
    data_subdir: str
    """Sibling dir name next to the main checkout, e.g. ``mini-interact``."""
    data_file: str
    """Tasks JSONL filename within the data root."""
    data_root_env: str
    """Env var that overrides the data root (cloud actor / tests set it)."""
    data_file_env: str
    """Env var that overrides the data file path."""

    # Modes / gating.
    supported_modes: tuple[str, ...]
    one_shot: bool
    """Whether this benchmark uses the one-shot (no user-sim) eval path."""

    # Gold.
    gold_required: bool
    """True when scoring needs a gated gold sidecar (the data JSONL ships
    ``sol_sql`` empty); False when gold is inline in the data JSONL."""
    gold_root_env: str = ""
    """Env var that overrides the gated gold-sidecar location (when required)."""

    # Per-task DB isolation (livesqlbench copies a stable per-task sqlite).
    per_task_db_isolation: bool

    # Cloud.
    container_data_dir: str
    """Where the actor materialises this benchmark's data in-container, e.g.
    ``/data/mini-interact``; also the cloud ``BIRD_DB_PATH``."""

    # DEV-1523: database backend.
    db_backend: Literal["sqlite", "postgres"] = "sqlite"
    """SQL execution backend. ``sqlite`` uses sqlite3 file-based DBs;
    ``postgres`` connects to a running postgres server via env vars
    (BIRD_PG_HOST / BIRD_PG_PORT / BIRD_PG_USER / BIRD_PG_PASSWORD)."""

    # DEV-1510: audited-gold sidecar layout.
    audited_gold_layout: Literal["per_db", "single_file"] = "per_db"
    """Where ``apply_audited_gold_overlay`` (and the cloud submit-time
    ``require_audited_gold`` guard) looks for this benchmark's audited
    rows.

    * ``per_db`` (default, mini-interact's historical contract) → one
      sidecar per DB at ``<audited_gold>/<db>/<db>_audited.jsonl``.
    * ``single_file`` (livesqlbench) → one consolidated JSONL at
      ``<audited_gold>/<benchmark>_audited.jsonl``, with the
      ``selected_database`` field on each row as the per-DB discriminator.
      Used when the benchmark's DBs collide on name with another
      benchmark's (e.g. ``alien`` / ``museum`` exist in both
      mini-interact and livesqlbench, so a per-db dir under
      ``audited_gold/`` would clash)."""


MINI_INTERACT = Benchmark(
    name="mini_interact",
    cli_aliases=("mini-interact",),
    dataset_marker="mini_interact",
    data_subdir="mini-interact",
    data_file="mini_interact.jsonl",
    data_root_env="BIRD_DB_PATH",
    data_file_env="BIRD_DATA_PATH",
    supported_modes=("a-interact", "c-interact", "oracle"),
    one_shot=False,
    gold_required=False,
    per_task_db_isolation=False,
    container_data_dir="/data/mini-interact",
    # DEV-1515: switched to the single-file layout (matches livesqlbench).
    # Per-DB JSONLs at audited_gold/<db>/<db>_audited.jsonl have been
    # consolidated into audited_gold/mini_interact_audited.jsonl with
    # variant_id + primary fields added per row (DEV-1515 multi-variant
    # support). See scripts/consolidate_mini_interact_audited.py.
    audited_gold_layout="single_file",
)

LIVESQLBENCH = Benchmark(
    name="livesqlbench",
    cli_aliases=(),
    dataset_marker="livesqlbench",
    data_subdir="livesqlbench-base-lite-sqlite",
    data_file="livesqlbench_data_sqlite.jsonl",
    data_root_env="BIRD_LIVESQLBENCH_ROOT",
    data_file_env="BIRD_LIVESQLBENCH_DATA_FILE",
    supported_modes=("one-shot", "oracle"),
    one_shot=True,
    gold_required=True,
    gold_root_env="BIRD_LIVESQLBENCH_GOLD_FILE",
    per_task_db_isolation=True,
    container_data_dir="/data/livesqlbench",
    # DEV-1510: a per-db sidecar dir under audited_gold/ would clash with
    # mini-interact (overlapping DB names: alien, museum, …). The
    # consolidated single-file layout uses the row's `selected_database`
    # as the per-db discriminator.
    audited_gold_layout="single_file",
)

LIVESQLBENCH_POSTGRES = Benchmark(
    name="livesqlbench_postgres",
    db_backend="postgres",
    dataset_marker="livesqlbench_postgres",
    data_subdir="livesqlbench-base-lite-postgres",
    data_file="livesqlbench_data.jsonl",
    data_root_env="BIRD_LIVESQLBENCH_POSTGRES_ROOT",
    data_file_env="BIRD_LIVESQLBENCH_POSTGRES_DATA_FILE",
    supported_modes=("one-shot",),
    one_shot=True,
    gold_required=True,
    gold_root_env="BIRD_LIVESQLBENCH_POSTGRES_GOLD_FILE",
    per_task_db_isolation=False,
    container_data_dir="/data/livesqlbench-postgres",
    audited_gold_layout="single_file",
)

MINI_INTERACT_POSTGRES = Benchmark(
    name="mini_interact_postgres",
    db_backend="postgres",
    dataset_marker="mini_interact_postgres",
    data_subdir="mini-interact-postgres",
    data_file="mini_interact.jsonl",
    data_root_env="BIRD_MINI_INTERACT_POSTGRES_ROOT",
    data_file_env="BIRD_MINI_INTERACT_POSTGRES_DATA_FILE",
    supported_modes=("a-interact",),
    one_shot=False,
    gold_required=False,
    per_task_db_isolation=False,
    container_data_dir="/data/mini-interact-postgres",
    audited_gold_layout="single_file",
)

BENCHMARKS: dict[str, Benchmark] = {
    b.name: b for b in (MINI_INTERACT, LIVESQLBENCH, LIVESQLBENCH_POSTGRES, MINI_INTERACT_POSTGRES)
}

# Every token that resolves to a benchmark: canonical name + cli aliases +
# dataset marker. Built once; collisions across benchmarks would be a bug.
_BY_TOKEN: dict[str, Benchmark] = {}
for _b in BENCHMARKS.values():
    for _tok in (_b.name, _b.dataset_marker, *_b.cli_aliases):
        if _tok in _BY_TOKEN and _BY_TOKEN[_tok] is not _b:
            raise RuntimeError(
                f"benchmark token collision: {_tok!r} maps to both "
                f"{_BY_TOKEN[_tok].name!r} and {_b.name!r}"
            )
        _BY_TOKEN[_tok] = _b


def get_benchmark(token: str) -> Benchmark:
    """Resolve a canonical name, CLI alias, or dataset marker to its Benchmark.
    Raises ValueError on an unknown token (no silent fallback)."""
    b = _BY_TOKEN.get(token)
    if b is None:
        raise ValueError(
            f"unknown benchmark token {token!r}; expected one of "
            f"{sorted(_BY_TOKEN)}"
        )
    return b


def all_benchmarks() -> tuple[Benchmark, ...]:
    return tuple(BENCHMARKS.values())


def benchmark_names() -> tuple[str, ...]:
    """Canonical names only."""
    return tuple(BENCHMARKS)


def cli_dataset_tokens() -> tuple[str, ...]:
    """All tokens valid as a `--dataset` value (canonical names + aliases),
    de-duplicated, for argparse `choices`."""
    seen: list[str] = []
    for b in BENCHMARKS.values():
        for tok in (b.name, *b.cli_aliases):
            if tok not in seen:
                seen.append(tok)
    return tuple(seen)
