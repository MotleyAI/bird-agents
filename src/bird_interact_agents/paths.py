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


def mini_interact_root() -> Path:
    """Path to the sibling `mini-interact/` data dir (JSONL + per-DB SQLite + KB).

    Honours `BIRD_DB_PATH` env override; otherwise sits next to the main
    checkout.
    """
    override = os.environ.get("BIRD_DB_PATH")
    if override:
        return Path(override).expanduser()
    return main_checkout_root().parent / "mini-interact"


def mini_interact_data_file() -> Path:
    """Path to `mini_interact.jsonl` — honours `BIRD_DATA_PATH` env override."""
    override = os.environ.get("BIRD_DATA_PATH")
    if override:
        return Path(override).expanduser()
    return mini_interact_root() / "mini_interact.jsonl"


def audited_gold_root() -> Path:
    """Audited-gold SQL sidecars — committed, must live in the main checkout.

    Honours `BIRD_AUDITED_GOLD_ROOT` override (used by SAR-audit tests).
    """
    override = os.environ.get("BIRD_AUDITED_GOLD_ROOT")
    if override:
        return Path(override).expanduser()
    return main_checkout_root() / "audited_gold"


def sar_audited_gold_root() -> Path:
    """SAR-Agent audit JSONL output root.

    Honours `BIRD_SAR_AUDITED_GOLD_ROOT` override.
    """
    override = os.environ.get("BIRD_SAR_AUDITED_GOLD_ROOT")
    if override:
        return Path(override).expanduser()
    return main_checkout_root() / "sar_audited_gold"


def slayer_models_root() -> Path:
    """Per-DB SLayer model YAML — committed, must live in the main checkout."""
    return main_checkout_root() / "slayer_models"


def slayer_models_otf_root() -> Path:
    """Per-DB reference models built by the DEV-1454 on-the-fly KB-encode
    setup pass — a sibling of ``slayer_models`` under the main checkout (so
    it is git-committable, and ``hard8_preprocessor.build_task_variant_storage``
    resolves the ``mini-interact`` dataset path identically). NEVER the same
    dir as ``slayer_models`` — hand-built committed models are never touched.
    """
    return main_checkout_root() / "slayer_models_otf"


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
