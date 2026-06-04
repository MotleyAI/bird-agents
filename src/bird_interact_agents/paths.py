"""Central path helper — resolves sibling data and output sinks from any checkout.

Benchmark data directories sit next to the main checkout, each named by the
benchmark's canonical hyphenated name (e.g. ``mini-interact/``,
``livesqlbench-base-lite-sqlite/``). The single ``BIRD_BENCHMARKS_ROOT`` env var
overrides the parent directory for ALL benchmarks at once — no per-benchmark
env vars.

From a ``git worktree``, relative paths would point into the ``.worktrees/``
container and benchmark outputs would scatter. ``main_checkout_root()`` prevents
that by asking git for the *common* git dir (shared across all worktrees) and
returning its parent — the original checkout. All helpers anchor off that.
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

    Layout: ``<BIRD_BENCHMARKS_ROOT>/<benchmark.name>/``.
    Default parent: the directory next to the main checkout.
    ``BIRD_BENCHMARKS_ROOT`` overrides the parent for ALL benchmarks at once.
    """
    b = _as_benchmark(benchmark)
    parent_override = os.environ.get("BIRD_BENCHMARKS_ROOT")
    if parent_override:
        return Path(parent_override).expanduser() / b.name
    return main_checkout_root().parent / b.name


def benchmark_data_file(benchmark: str | Benchmark) -> Path:
    """Tasks JSONL for ``benchmark``: ``<benchmark_data_root>/<data_file>``."""
    b = _as_benchmark(benchmark)
    return benchmark_data_root(b) / b.data_file


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
        p = Path(override).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    path = main_checkout_root() / "audited_gold"
    path.mkdir(parents=True, exist_ok=True)
    return path


def audited_gold_file(*, benchmark: str) -> Path:
    """Resolve the audited-gold JSONL for a ``single_file`` benchmark.

    Layout: ``audited_gold/<benchmark>/<benchmark>_audited.jsonl``.

    Raises ``ValueError`` for ``per_db`` benchmarks or unknown tokens so
    a forgotten / typo'd benchmark cannot silently land at a wrong path.
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
    return audited_gold_root() / b.name / f"{b.name}_audited.jsonl"


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
        p = Path(override).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
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

    All benchmarks share a single parent dir ``slayer_otf_cache/`` under the
    main checkout; each benchmark gets its own subdirectory:
    ``slayer_otf_cache/<benchmark>/``.

    ``BIRD_OTF_CACHE_ROOT`` overrides the parent for ALL benchmarks —
    benchmark subdir is still appended under it.
    """
    _validate_benchmark(benchmark)
    parent_override = os.environ.get("BIRD_OTF_CACHE_ROOT")
    if parent_override:
        return Path(parent_override).expanduser() / benchmark
    return main_checkout_root() / "slayer_otf_cache" / benchmark


def slayer_models_otf_root(*, benchmark: str) -> Path:
    """Per-DB reference models built by the on-the-fly KB-encode setup pass.

    All benchmarks share a single parent dir ``slayer_models_otf/`` under the
    main checkout; each benchmark gets its own subdirectory:
    ``slayer_models_otf/<benchmark>/``.

    ``BIRD_SLAYER_MODELS_OTF_ROOT`` overrides the parent for ALL benchmarks —
    benchmark subdir is still appended under it.
    """
    _validate_benchmark(benchmark)
    parent_override = os.environ.get("BIRD_SLAYER_MODELS_OTF_ROOT")
    if parent_override:
        return Path(parent_override).expanduser() / benchmark
    return main_checkout_root() / "slayer_models_otf" / benchmark


def gated_gold_root(*, benchmark: str | None) -> Path:
    """Canonical location for gated gold sidecar files (gitignored).

    Each benchmark gets its own subdirectory:
    ``gated_gold/<benchmark>/``.

    ``BIRD_GATED_GOLD_ROOT`` overrides the parent for ALL benchmarks —
    benchmark subdir is still appended under it.

    Raises ``ValueError`` for ``None`` or unknown benchmark.
    """
    if benchmark is None:
        raise ValueError("benchmark must not be None")
    _validate_benchmark(benchmark)
    parent_override = os.environ.get("BIRD_GATED_GOLD_ROOT")
    if parent_override:
        return Path(parent_override).expanduser() / benchmark
    return main_checkout_root() / "gated_gold" / benchmark


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
