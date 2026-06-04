"""Benchmark registry — one descriptor per benchmark, the single source of truth
consumed symmetrically by the local runner (`run.py`/`harness.py`/`paths.py`) and
the cloud runner (`cloud/{cli,driver,ray_app,image}.py`).

Adding a benchmark is one `BENCHMARKS` entry; no consumer edits. Canonical names
are the official hyphenated forms (mini-interact, livesqlbench-base-lite-sqlite, …).
The dataset marker (stamped into `task_data["dataset"]` by the loaders) equals
the canonical name so that `get_benchmark()` is the single resolution path from
any context.

Data directories follow a uniform layout: all benchmarks are checked out as sibling
directories of the main repo, named by their canonical benchmark name. The single
`BIRD_BENCHMARKS_ROOT` env var overrides the parent for all benchmarks at once.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Benchmark(BaseModel):
    """Immutable per-benchmark descriptor. Every per-benchmark fact the harness
    branches on lives here (no scattered ``"livesqlbench"`` literals)."""

    model_config = ConfigDict(frozen=True)

    name: str
    """Canonical hyphenated token, e.g. ``mini-interact`` / ``livesqlbench-base-lite-sqlite``."""
    cli_aliases: tuple[str, ...] = ()
    """Extra `--dataset` spellings that resolve to this benchmark."""
    dataset_marker: str
    """Stamped into ``task_data["dataset"]`` by the loaders; must equal ``name``."""

    # Data location (resolved by paths.benchmark_data_root / _data_file).
    # The data directory is <BIRD_BENCHMARKS_ROOT>/<name>/ — no per-benchmark
    # env vars. ``data_file`` is the tasks JSONL filename within that directory.
    data_file: str
    """Tasks JSONL filename within the data root."""

    # Modes / gating.
    supported_modes: tuple[str, ...]
    one_shot: bool
    """Whether this benchmark uses the one-shot (no user-sim) eval path."""

    # Gold.
    gold_required: bool
    """True when scoring needs a gated gold sidecar (the data JSONL ships
    ``sol_sql`` empty); False when gold is inline in the data JSONL."""

    # Per-task DB isolation (livesqlbench copies a stable per-task sqlite).
    per_task_db_isolation: bool

    # Cloud.
    container_data_dir: str
    """Where the actor materialises this benchmark's data in-container, e.g.
    ``/data/mini-interact``; also the cloud data root for this benchmark."""

    # DEV-1523: database backend.
    db_backend: Literal["sqlite", "postgres"] = "sqlite"
    """SQL execution backend. ``sqlite`` uses sqlite3 file-based DBs;
    ``postgres`` connects to a running postgres server via env vars
    (BIRD_PG_HOST / BIRD_PG_PORT / BIRD_PG_USER / BIRD_PG_PASSWORD)."""

    # One-shot SELECT task count (for full-run assertion in loader).
    select_full_run_count: int | None = None
    """Expected number of SELECT tasks on a full unfiltered one-shot run.
    ``None`` skips the assertion (use for benchmarks whose exact size isn't
    yet fixed or when the full dataset isn't yet available locally)."""

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
    name="mini-interact",
    cli_aliases=(),
    dataset_marker="mini-interact",
    data_file="mini_interact.jsonl",
    supported_modes=("a-interact", "c-interact", "oracle"),
    one_shot=False,
    gold_required=True,
    per_task_db_isolation=False,
    container_data_dir="/data/mini-interact",
    audited_gold_layout="single_file",
)

BIRD_INTERACT_FULL = Benchmark(
    name="bird-interact-full",
    cli_aliases=(),
    dataset_marker="bird-interact-full",
    data_file="bird_interact_full.jsonl",
    supported_modes=("a-interact", "c-interact", "oracle"),
    one_shot=False,
    gold_required=True,
    per_task_db_isolation=False,
    container_data_dir="/data/bird-interact-full",
    audited_gold_layout="single_file",
)

LIVESQLBENCH = Benchmark(
    name="livesqlbench-base-lite-sqlite",
    cli_aliases=(),
    dataset_marker="livesqlbench-base-lite-sqlite",
    data_file="livesqlbench_data_sqlite.jsonl",
    supported_modes=("one-shot", "oracle"),
    one_shot=True,
    gold_required=True,
    per_task_db_isolation=True,
    container_data_dir="/data/livesqlbench-base-lite-sqlite",
    audited_gold_layout="single_file",
    select_full_run_count=180,
)

LIVESQLBENCH_POSTGRES = Benchmark(
    name="livesqlbench-base-lite",
    db_backend="postgres",
    dataset_marker="livesqlbench-base-lite",
    data_file="livesqlbench_data.jsonl",
    supported_modes=("one-shot",),
    one_shot=True,
    gold_required=True,
    per_task_db_isolation=False,
    container_data_dir="/data/livesqlbench-base-lite",
    audited_gold_layout="single_file",
)

LIVESQLBENCH_BASE_FULL = Benchmark(
    name="livesqlbench-base-full",
    cli_aliases=(),
    dataset_marker="livesqlbench-base-full",
    data_file="livesqlbench_base_full_sqlite.jsonl",
    supported_modes=("one-shot", "oracle"),
    one_shot=True,
    gold_required=True,
    per_task_db_isolation=True,
    container_data_dir="/data/livesqlbench-base-full",
    audited_gold_layout="single_file",
)

LIVESQLBENCH_LARGE = Benchmark(
    name="livesqlbench-large",
    cli_aliases=(),
    dataset_marker="livesqlbench-large",
    data_file="livesqlbench_large_sqlite.jsonl",
    supported_modes=("one-shot", "oracle"),
    one_shot=True,
    gold_required=True,
    per_task_db_isolation=True,
    container_data_dir="/data/livesqlbench-large",
    audited_gold_layout="single_file",
)

MINI_INTERACT_POSTGRES = Benchmark(
    name="bird-interact-lite-exp",
    db_backend="postgres",
    dataset_marker="bird-interact-lite-exp",
    data_file="mini_interact.jsonl",
    supported_modes=("a-interact",),
    one_shot=False,
    gold_required=True,
    per_task_db_isolation=False,
    container_data_dir="/data/bird-interact-lite-exp",
    audited_gold_layout="single_file",
)

BENCHMARKS: dict[str, Benchmark] = {
    b.name: b for b in (
        MINI_INTERACT,
        BIRD_INTERACT_FULL,
        LIVESQLBENCH,
        LIVESQLBENCH_POSTGRES,
        LIVESQLBENCH_BASE_FULL,
        LIVESQLBENCH_LARGE,
        MINI_INTERACT_POSTGRES,
    )
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
    """Resolve any accepted token (canonical name, alias, or dataset marker)
    to its ``Benchmark`` descriptor. Raises ``ValueError`` for unknown tokens."""
    b = _BY_TOKEN.get(token)
    if b is None:
        raise ValueError(
            f"Unknown benchmark token {token!r}. "
            f"Known tokens: {sorted(_BY_TOKEN)}"
        )
    return b


def benchmark_names() -> list[str]:
    """Return the canonical names of all registered benchmarks."""
    return list(BENCHMARKS)


def all_benchmarks() -> list[Benchmark]:
    """Return all registered benchmark descriptors."""
    return list(BENCHMARKS.values())


def cli_dataset_tokens() -> tuple[str, ...]:
    """All tokens valid as a ``--dataset`` value (canonical names + aliases),
    de-duplicated, for argparse ``choices``."""
    seen: list[str] = []
    for b in BENCHMARKS.values():
        for tok in (b.name, *b.cli_aliases):
            if tok not in seen:
                seen.append(tok)
    return tuple(seen)
