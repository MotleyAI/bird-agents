#!/usr/bin/env python
"""DEV-1672: migrate existing stores so raw JSON columns are ``hidden=True``.

The phase-3 encoder change hides raw JSON columns in FRESHLY-built OTF caches,
and the ``_PHASE3_IMPL_TOKEN`` bump forces warm ``slayer_otf_cache`` to rebuild.
This one-shot, idempotent migration covers the stores that rebuild does NOT
touch:

* ``slayer_models_otf/<db>`` — OTF reference stores (merged, not cache-rebuilt);
* saved ``runs/<bench>/<db>/<iid>/edited_models.tar.gz`` archives that
  ``--apply-edited-models`` reuses (snapshotted from the OLD cache; the
  apply-time self-heal fixes them at query time, but migrating the on-disk
  archive makes the win visible without a run and keeps the file honest).

``slayer_otf_cache/<db>`` is ALSO swept when ``--cache`` is passed, but note the
token already rebuilds it on next use, so it is off by default.

All operations reuse the single ``hide_expanded_jsonb_columns`` predicate via
``hide_store_dir`` / ``hide_archive`` and preserve archive ``cache_fp`` meta.

Examples:
    uv run python scripts/migrate_hide_raw_jsonb.py --benchmark livesqlbench-large
    uv run python scripts/migrate_hide_raw_jsonb.py --benchmark mini-interact --reference
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.slayer_otf.hide_jsonb_stores import (
    hide_archive,
    hide_store_dir,
)


def _store_dirs(root: Path) -> list[Path]:
    """Immediate ``<db>/`` sub-dirs of a per-benchmark store root."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


async def _sweep_store_root(root: Path, label: str) -> int:
    total = 0
    for db_dir in _store_dirs(root):
        n = await hide_store_dir(db_dir)
        total += n
        print(f"  [{label}] {db_dir.name}: {n} column(s) hidden")
    return total


async def _sweep_archives(benchmark: str, instance_ids: set[str] | None) -> int:
    # Scope to this benchmark's runs subtree — archives live at
    # runs/<benchmark>/<db>/<iid>/edited_models.tar.gz. This also excludes
    # runs/_edited_models_backups/ (a sibling of <benchmark>/), so a
    # benchmark-targeted migration never rewrites backups or other benchmarks.
    root = paths.runs_root() / benchmark
    total = 0
    for archive in sorted(root.rglob("edited_models.tar.gz")):
        if instance_ids is not None and archive.parent.name not in instance_ids:
            continue
        n = await hide_archive(archive)
        total += n
        rel = archive.relative_to(paths.runs_root())
        print(f"  [archive] {rel}: {n} column(s) hidden")
    return total


async def _run(args: argparse.Namespace) -> int:
    # Default: reference stores + archives (the two the token does not rebuild).
    do_reference = args.reference or args.all or not (args.cache or args.archives)
    do_archives = args.archives or args.all or not (args.cache or args.reference)
    do_cache = args.cache or args.all

    ids = set(args.instance_ids.split(",")) if args.instance_ids else None
    grand = 0
    if do_reference:
        print(f"slayer_models_otf ({args.benchmark}):")
        grand += await _sweep_store_root(
            paths.slayer_models_otf_root(benchmark=args.benchmark), "reference"
        )
    if do_cache:
        print(f"slayer_otf_cache ({args.benchmark}):")
        grand += await _sweep_store_root(
            paths.slayer_otf_cache_root(benchmark=args.benchmark), "cache"
        )
    if do_archives:
        print(f"saved edited-model archives ({args.benchmark}):")
        grand += await _sweep_archives(args.benchmark, ids)
    print(f"\nDone: {grand} raw JSON column(s) newly hidden.")
    return grand


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True,
                    choices=sorted(paths._KNOWN_BENCHMARKS),
                    help="canonical benchmark name; scopes the cache/reference "
                         "roots AND the archive sweep (e.g. livesqlbench-large, "
                         "mini-interact)")
    ap.add_argument("--reference", action="store_true",
                    help="sweep slayer_models_otf/<db> (default on)")
    ap.add_argument("--archives", action="store_true",
                    help="sweep saved edited_models.tar.gz under runs/ (default on)")
    ap.add_argument("--cache", action="store_true",
                    help="also sweep slayer_otf_cache/<db> (token already rebuilds it)")
    ap.add_argument("--all", action="store_true", help="reference + cache + archives")
    ap.add_argument("--instance-ids", default=None,
                    help="comma-separated instance ids to limit the archive sweep to")
    asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    main()
