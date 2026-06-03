"""Central path helper — resolves sibling data and output sinks from any checkout.

The repo expects `mini-interact/` to sit at `<main_checkout>/../mini-interact/`
and writes outputs (`audited_gold/`, `slayer_models/`, `results/`,
`.benchmarks/`) under the checkout root. From a `git worktree`, both
assumptions break: relative paths point into the `.worktrees/` container, and
benchmark output would scatter across throwaway worktrees.

`main_checkout_root()` short-circuits that by asking git for the *common* git
dir (shared across all worktrees of a repo) and returning its parent — i.e.
the original checkout. All other helpers anchor off that. Env vars override
the default for the sibling data dir (`BIRD_DB_PATH`, `BIRD_DATA_PATH`) and
for the results dir (`BIRD_RESULTS_ROOT`).
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

from bird_interact_agents.benchmark import (
    Benchmark,
    benchmark_names,
    get_benchmark,
)

# Module-level so tests can monkeypatch it. In production, this is the dir
# containing this file — `<checkout>/src/bird_interact_agents/`. Production
# code that imports from a worktree's editable install sees its own copy of
# this module, but `git rev-parse --git-common-dir` traces through the
# worktree's `.git` file back to the main checkout regardless.
_LOOKUP_DIR: Path = Path(__file__).resolve().parent


@functools.cache
def _main_checkout_root_cached() -> Path:
    """Cached worker — runs `git rev-parse --git-common-dir` exactly once per
    process. Tests reset the cache via the autouse fixture in conftest /
    test_paths.
    """
    lookup_dir = _LOOKUP_DIR
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(lookup_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # No git, not a git tree, or git CLI missing — fall back to the
        # source-relative repo root. Works for editable installs; for wheel
        # installs outside a checkout the caller must set explicit env vars.
        return Path(__file__).resolve().parents[2]

    common_dir = Path(result.stdout.strip())
    if not common_dir.is_absolute():
        # git returns "<lookup_dir>/.git" relative to cwd when inside the
        # main checkout (no worktree indirection).
        common_dir = (lookup_dir / common_dir).resolve()
    return common_dir.parent.resolve()


def main_checkout_root() -> Path:
    """Return the path of the canonical checkout, even when called from a
    worktree spawned via `git worktree add`."""
    return _main_checkout_root_cached()


# ---------------------------------------------------------------------------
# Benchmark data roots — registry-driven, so ANY benchmark resolves its data
# dir / tasks file the same way. Each benchmark declares its sibling subdir +
# data filename + env-override names in `benchmark.Benchmark`; these generic
# accessors read those. Worktree-safe by construction (anchored at the main
# checkout). The per-benchmark `mini_interact_*` / `livesqlbench_*` helpers
# below are thin back-compat shims over these.
# ---------------------------------------------------------------------------


def _as_benchmark(benchmark: str | Benchmark) -> Benchmark:
    return benchmark if isinstance(benchmark, Benchmark) else get_benchmark(benchmark)


def benchmark_data_root(benchmark: str | Benchmark) -> Path:
    """Sibling data dir for ``benchmark`` (per-DB SQLite + KB + tasks JSONL).
    Honours the benchmark's data-root env override; else sits next to the main
    checkout at ``<parent>/<data_subdir>``."""
    b = _as_benchmark(benchmark)
    override = os.environ.get(b.data_root_env)
    if override:
        return Path(override).expanduser()
    return main_checkout_root().parent / b.data_subdir


def benchmark_data_file(benchmark: str | Benchmark) -> Path:
    """Tasks JSONL for ``benchmark`` — honours the benchmark's data-file env
    override; else ``<data_root>/<data_file>``."""
    b = _as_benchmark(benchmark)
    override = os.environ.get(b.data_file_env)
    if override:
        return Path(override).expanduser()
    return benchmark_data_root(b) / b.data_file


def mini_interact_root() -> Path:
    """Back-compat shim → ``benchmark_data_root("mini_interact")``."""
    return benchmark_data_root("mini_interact")


def mini_interact_data_file() -> Path:
    """Back-compat shim → ``benchmark_data_file("mini_interact")``."""
    return benchmark_data_file("mini_interact")


def livesqlbench_root() -> Path:
    """Back-compat shim → ``benchmark_data_root("livesqlbench")``."""
    return benchmark_data_root("livesqlbench")


def livesqlbench_data_file() -> Path:
    """Back-compat shim → ``benchmark_data_file("livesqlbench")``."""
    return benchmark_data_file("livesqlbench")


# Set of benchmark identifiers the per-benchmark OTF roots accept. `benchmark`
# is REQUIRED and explicit on those helpers (no `None` default) so a forgotten
# or typo'd benchmark can never silently fall back to — and mix artifacts with
# — mini-interact. `"mini_interact"` is itself an explicit value that maps to
# the legacy dirs/env vars (so on-disk layout + the cloud contract are
# unchanged). New benchmarks (e.g. Base-Full) extend this set.
_KNOWN_BENCHMARKS: frozenset[str] = frozenset(benchmark_names())


def _validate_benchmark(benchmark: str | None) -> None:
    """Require an explicit, known benchmark. `None`/unknown raises ValueError —
    there is no silent mini-interact fallback."""
    if benchmark not in _KNOWN_BENCHMARKS:
        raise ValueError(
            f"benchmark must be one of {sorted(_KNOWN_BENCHMARKS)}; got "
            f"{benchmark!r}. It is required (no default) so a forgotten or "
            f"typo'd benchmark cannot silently mix per-benchmark artifacts."
        )


def audited_gold_root() -> Path:
    """Audited-gold SQL sidecars — committed, must live in the main checkout.

    Honours `BIRD_AUDITED_GOLD_ROOT` override (used by SAR-audit tests).

    Always creates the directory if missing — the dir is gitignored so a
    fresh checkout doesn't have one, and downstream callers (notably
    ``cloud/image.build_and_push``'s BuildKit ``--build-context``) need
    at least an empty dir on disk before they can mount it.
    """
    override = os.environ.get("BIRD_AUDITED_GOLD_ROOT")
    if override:
        return Path(override).expanduser()
    path = main_checkout_root() / "audited_gold"
    path.mkdir(parents=True, exist_ok=True)
    return path


def audited_gold_file(*, benchmark: str) -> Path:
    """DEV-1510: resolve the audited-gold JSONL for a `single_file` benchmark.

    Mini-interact uses ``per_db`` (each DB gets its own
    ``<audited_root>/<db>/<db>_audited.jsonl``) and has no single-file
    path — call ``audited_gold_root()`` directly there.

    Livesqlbench uses ``single_file`` because its DB names collide with
    mini-interact's (alien, museum, …); the file is one consolidated
    ``audited_gold/livesqlbench_audited.jsonl`` keyed by ``instance_id``
    with ``selected_database`` as the per-DB discriminator on each row.

    Raises ``ValueError`` for ``per_db`` benchmarks or unknown tokens —
    same posture as ``slayer_otf_cache_root`` so a forgotten / typo'd
    benchmark cannot silently land at a wrong path.
    """
    _validate_benchmark(benchmark)
    b = get_benchmark(benchmark)
    if b.audited_gold_layout != "single_file":
        raise ValueError(
            f"audited_gold_file({benchmark!r}) is only defined for "
            f"`single_file` layouts; got layout={b.audited_gold_layout!r}. "
            f"For `per_db` benchmarks, use `audited_gold_root() / <db> / "
            f"<db>_audited.jsonl` via `apply_audited_gold_overlay`."
        )
    return audited_gold_root() / f"{b.name}_audited.jsonl"


def annotations_root() -> Path:
    """DEV-1515: per-task / per-submission annotations, committed to the
    main checkout (gitignored locally — the human-judgment content is
    licensed for local use only, same posture as ``audited_gold/``).

    Honours ``BIRD_ANNOTATIONS_ROOT`` for tests / forks that mount the
    annotations from a parallel repo.

    Always creates the directory if missing — same posture as
    ``audited_gold_root``: the dir is gitignored so a fresh checkout
    doesn't have one, and ``cloud/image.build_and_push`` mounts it via
    BuildKit ``--build-context``, which needs at least an empty dir on
    disk before docker build can resolve it.
    """
    override = os.environ.get("BIRD_ANNOTATIONS_ROOT")
    if override:
        return Path(override).expanduser()
    path = main_checkout_root() / "annotations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sar_audited_gold_root() -> Path:
    """SAR-Agent audit JSONL output root.

    Honours `BIRD_SAR_AUDITED_GOLD_ROOT` override.
    """
    override = os.environ.get("BIRD_SAR_AUDITED_GOLD_ROOT")
    if override:
        return Path(override).expanduser()
    return main_checkout_root() / "sar_audited_gold"


def slayer_models_root() -> Path:
    """Per-DB SLayer model YAML — committed, must live in the main checkout.

    Honours ``BIRD_SLAYER_MODELS_ROOT`` (harmless locally; the cloud actor
    sets it to ``/data/slayer_models`` so the uploaded pre-encoded models
    are found there).
    """
    override = os.environ.get("BIRD_SLAYER_MODELS_ROOT")
    if override:
        return Path(override).expanduser()
    return main_checkout_root() / "slayer_models"


def slayer_otf_cache_root(*, benchmark: str) -> Path:
    """Per-DB phase-1-3 ingest cache root for the on-the-fly setup path.

    Single source of truth (replaces the duplicated per-agent
    ``_otf_cache_root`` helpers). ``benchmark`` is REQUIRED and explicit so two
    benchmarks with overlapping DB names (e.g. mini-interact and LiveSQLBench
    both have an ``alien`` DB) keep disjoint caches and a forgotten benchmark
    can never silently mix them:

    * ``benchmark="mini_interact"`` → ``<main_checkout>/slayer_otf_cache/``
      (the LEGACY dir + LEGACY env var — on-disk layout & the dev-1470 cloud
      contract are unchanged).
    * ``benchmark="livesqlbench"`` →
      ``<main_checkout>/slayer_otf_cache_livesqlbench/``.

    Env overrides (the cloud actor sets these to ``/data/<...>`` so the
    uploaded cache is found there) are PARALLEL — one per benchmark, so a
    livesqlbench override does not steer the mini-interact root and vice versa:

    * mini_interact: ``BIRD_OTF_CACHE_ROOT``
    * livesqlbench:  ``BIRD_OTF_CACHE_ROOT_LIVESQLBENCH``
    """
    _validate_benchmark(benchmark)
    if benchmark == "mini_interact":
        override = os.environ.get("BIRD_OTF_CACHE_ROOT")
        if override:
            return Path(override).expanduser()
        return main_checkout_root() / "slayer_otf_cache"
    env_var = f"BIRD_OTF_CACHE_ROOT_{benchmark.upper()}"
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    return main_checkout_root() / f"slayer_otf_cache_{benchmark}"


def slayer_models_otf_root(*, benchmark: str) -> Path:
    """Per-DB reference models built by the DEV-1454 on-the-fly KB-encode
    setup pass — a sibling of ``slayer_models`` under the main checkout (so
    it is git-committable, and ``hard8_preprocessor.build_task_variant_storage``
    resolves the ``mini-interact`` dataset path identically). NEVER the same
    dir as ``slayer_models`` — hand-built committed models are never touched.

    ``benchmark`` is REQUIRED and explicit, same semantics as
    :func:`slayer_otf_cache_root`:

    * ``benchmark="mini_interact"`` → ``<main_checkout>/slayer_models_otf/``
      (legacy dir + legacy env var).
    * ``benchmark="livesqlbench"`` →
      ``<main_checkout>/slayer_models_otf_livesqlbench/``.

    Env overrides (one per benchmark):

    * mini_interact: ``BIRD_SLAYER_MODELS_OTF_ROOT``
    * livesqlbench:  ``BIRD_SLAYER_MODELS_OTF_ROOT_LIVESQLBENCH``
    """
    _validate_benchmark(benchmark)
    if benchmark == "mini_interact":
        override = os.environ.get("BIRD_SLAYER_MODELS_OTF_ROOT")
        if override:
            return Path(override).expanduser()
        return main_checkout_root() / "slayer_models_otf"
    env_var = f"BIRD_SLAYER_MODELS_OTF_ROOT_{benchmark.upper()}"
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser()
    return main_checkout_root() / f"slayer_models_otf_{benchmark}"


def results_root() -> Path:
    """Benchmark output root. Honours `BIRD_RESULTS_ROOT`; otherwise main
    checkout's `results/` so all worktree runs aggregate in one place.
    """
    override = os.environ.get("BIRD_RESULTS_ROOT")
    if override:
        return Path(override).expanduser()
    return main_checkout_root() / "results"


def benchmarks_root() -> Path:
    """pytest-benchmark output (committed)."""
    return main_checkout_root() / ".benchmarks"
