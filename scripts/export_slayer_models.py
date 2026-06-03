#!/usr/bin/env python3
"""Export SLayer models for one mini-interact DB from the live SLayer
storage to ``bird-interact-agents/slayer_models/<db>/``.

Reads from the SLayer storage path the MCP server uses (default:
``~/.local/share/slayer``) and writes a YAMLStorage-shaped tree at
``bird-interact-agents/slayer_models/<db>/`` containing the
datasource, every model whose ``data_source`` matches, AND every
deferred-KB memory whose ``linked_entities`` resolves under the
target DB (at least one canonical ref starting with ``<db>.``).
The memory filter matches the verifier's own scope rule
(``scripts/verify_kb_coverage.py``'s ``load_documented_ids``), so
the committed per-DB tree never carries cross-DB memory noise.

The output tree is exactly what ``YAMLStorage(base_dir=…)`` expects,
so the verifier (and any future runner that loads the per-DB YAML)
can pick it up directly.

Usage:
    python scripts/export_slayer_models.py --db <db>
    python scripts/export_slayer_models.py --db <db> --source <slayer_storage_path>

Idempotent: re-runs overwrite the destination directory.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

from slayer.storage.base import resolve_storage
from slayer.storage.yaml_storage import YAMLStorage

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from bird_interact_agents import paths  # noqa: E402
from bird_interact_agents.slayer_pipeline.portable_connection import (  # noqa: E402
    to_portable_connection_string,
)

DEST_ROOT = paths.slayer_models_root()
DEFAULT_SLAYER_STORAGE = Path(os.environ.get(
    "SLAYER_STORAGE",
    str(Path.home() / ".local" / "share" / "slayer"),
))
DEFAULT_MINI_INTERACT_ROOT = paths.benchmark_data_root("mini-interact")

_SQLITE_URI_PREFIXES = ("sqlite://", "yaml://")


def _open_source(path_or_uri: str):
    """Open the right storage backend for *path_or_uri*.

    Delegates to ``slayer.storage.base.resolve_storage`` so we pick up
    every backend SLayer's own CLI accepts: ``.db`` / ``.sqlite`` /
    ``.sqlite3`` files, ``sqlite://`` / ``yaml://`` URIs, third-party
    schemes registered via ``register_storage``, and YAMLStorage for
    directories.
    """
    return resolve_storage(path_or_uri)


async def _export_async(db: str, source: str) -> int:
    src = _open_source(source)
    dest_dir = DEST_ROOT / db

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = YAMLStorage(base_dir=str(dest_dir))

    ds = await src.get_datasource(db)
    if ds is None:
        print(
            f"[ERROR] Datasource '{db}' not found in {source}.",
            file=sys.stderr,
        )
        return 1
    # Strip the absolute mini-interact path from the connection_string
    # so the committed YAML doesn't bake in `/home/<user>/...`.
    portable = to_portable_connection_string(
        ds.connection_string or "", DEFAULT_MINI_INTERACT_ROOT
    )
    if portable != ds.connection_string:
        ds = ds.model_copy(update={"connection_string": portable})
    await dest.save_datasource(ds)

    n_models = 0
    for name in await src.list_models(data_source=db):
        try:
            model = await src.get_model(name, data_source=db)
        except Exception as exc:
            # Other DBs may have v1-shape models (e.g. demo data with
            # dimension/measure name collisions) that fail v3 migration.
            # We only care about models whose data_source matches *db*;
            # log and skip the rest.
            print(
                f"  skip '{name}': {exc.__class__.__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        if model is None or model.data_source != db:
            continue
        await dest.save_model(model)
        n_models += 1

    n_memories = await _export_memories(src=src, dest=dest, db=db)

    print(
        f"[OK] exported {n_models} model(s) and {n_memories} memory/memories "
        f"for '{db}' to {dest_dir}"
    )
    return 0


async def _export_memories(*, src, dest, db: str) -> int:
    """Copy memories whose ``linked_entities`` resolves under ``db``.

    A memory is in scope when at least one entry of its
    ``entities`` list starts with ``"<db>."``. This matches the
    verifier's scope rule in ``scripts/verify_kb_coverage.py``
    (``load_documented_ids``) so the committed per-DB tree never
    carries cross-DB noise.

    Preserves ``id`` and ``created_at`` from the source so re-exports
    are bit-identical when source memories are unchanged
    (``dest.save_memory`` would otherwise allocate fresh values).

    Returns the number of memories copied.
    """
    db_prefix = f"{db}."
    n = 0
    for mem in await src.list_memories():
        if not any(e.startswith(db_prefix) for e in mem.entities):
            continue
        await dest._save_memory_row(mem)
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export SLayer models for one mini-interact DB to slayer_models/<db>/.",
    )
    p.add_argument("--db", required=True, help="DB / datasource name (e.g. 'households').")
    p.add_argument(
        "--source",
        default=str(DEFAULT_SLAYER_STORAGE),
        help=(
            f"Path to the SLayer storage to read from "
            f"(default: {DEFAULT_SLAYER_STORAGE}). Override with $SLAYER_STORAGE."
        ),
    )
    args = p.parse_args()
    # Resolve filesystem paths to absolute; pass URIs straight through.
    source = (
        args.source
        if args.source.startswith(_SQLITE_URI_PREFIXES)
        else str(Path(args.source).resolve())
    )
    return asyncio.run(_export_async(args.db, source))


if __name__ == "__main__":
    sys.exit(main())
