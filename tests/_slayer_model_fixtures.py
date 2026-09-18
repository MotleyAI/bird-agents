"""Shared helper: copy a committed ``slayer_models/<db>`` to tmp before loading,
repointing its sqlite datasource at an empty scratch DB — so load-only reference
tests never touch the tracked tree nor hard-fail on a machine-specific path."""

from __future__ import annotations

import re
import shutil
import sqlite3
from pathlib import Path

from bird_interact_agents import paths


def copy_committed_db(
    db: str, dest_root: Path, *, models_root: Path | None = None,
) -> Path:
    """Copy ``<models_root>/<db>`` (default ``paths.slayer_models_root()``) into
    ``dest_root/<db>``, repointing each sqlite datasource at an empty scratch DB."""
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
