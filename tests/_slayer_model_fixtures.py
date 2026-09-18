"""Shared helper: copy a committed ``slayer_models/<db>`` to tmp before loading.

Opening a ``YAMLStorage`` on a committed dir lazily creates an ``embeddings.db``
sidecar and (on a storage-schema bump) rewrites the tracked YAML; loading also
opens the datasource for on-load type refinement and hard-fails on a
machine-specific data path. Copying to tmp and repointing every sqlite
datasource at an empty scratch DB keeps every load off the tracked tree and
machine-independent. The single place the load-only reference tests get a
throwaway copy.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
from pathlib import Path

from bird_interact_agents import paths


def copy_committed_db(
    db: str, dest_root: Path, *, models_root: Path | None = None,
) -> Path:
    """Copy ``<models_root>/<db>`` into ``dest_root/<db>`` and repoint each sqlite
    datasource at an empty scratch sqlite. Returns the copy's path.
    ``models_root`` defaults to :func:`paths.slayer_models_root`.
    """
    root = models_root or paths.slayer_models_root()
    dst = dest_root / db
    shutil.copytree(root / db, dst)
    ds_dir = dst / "datasources"
    if ds_dir.exists():
        scratch = dst / "_scratch.sqlite"
        sqlite3.connect(str(scratch)).close()
        for dsf in ds_dir.glob("*.yaml"):
            txt = dsf.read_text(encoding="utf-8")
            txt = re.sub(
                r"connection_string:.*",
                f"connection_string: sqlite:///{scratch}",
                txt,
            )
            dsf.write_text(txt, encoding="utf-8")
    return dst
