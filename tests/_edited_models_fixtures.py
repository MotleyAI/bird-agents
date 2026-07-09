"""Shared helpers for the DEV-1649 edited-models save/apply tests.

Builds a minimal-but-faithful per-task SLayer storage scratch dir on disk
(no real ``slayer ingest`` — these are unit tests). The layout mirrors what
``slayer_otf.prepare_task_storage`` materialises:

    <base>/<db>/
        datasources/<db>.yaml     # version/name/type/connection_string
        models/<db>/<model>.yaml
        memories.yaml
        embeddings.db             # sqlite
        _kb_rows.json
        _cache_fp.txt
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

DEFAULT_ABS_CONN = "sqlite:////nonexistent/build/machine/alien/alien.sqlite"


def make_fake_store(
    base: Path,
    db: str = "alien",
    *,
    conn_string: str = DEFAULT_ABS_CONN,
    kb_rows: list | None = None,
    cache_fp: str = "fp0",
    model_body: str = "name: foo\ncolumns: []\n",
    memories: list | None = None,
    with_wal: bool = False,
) -> Path:
    """Materialise a fake SLayer scratch under ``<base>/<db>/`` and return it."""
    scratch = base / db
    (scratch / "datasources").mkdir(parents=True)
    (scratch / "datasources" / f"{db}.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "name": db,
                "type": "sqlite",
                "connection_string": conn_string,
            },
            sort_keys=False,
        )
    )
    (scratch / "models" / db).mkdir(parents=True)
    (scratch / "models" / db / "foo.yaml").write_text(model_body)
    if memories is None:
        memories = [{"version": 1, "id": f"{db}_kb_0", "entities": [db]}]
    (scratch / "memories.yaml").write_text(yaml.safe_dump(memories, sort_keys=False))

    # Match SLayer's SidecarEmbeddingStore schema exactly so the real
    # YAMLStorage / reanchor path can open the sidecar without a migration
    # collision (CREATE TABLE IF NOT EXISTS + index on embedding_model_name).
    emb = scratch / "embeddings.db"
    con = sqlite3.connect(emb)
    try:
        if with_wal:
            con.execute("PRAGMA journal_mode=WAL")
        con.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "canonical_id TEXT NOT NULL, embedding_model_name TEXT NOT NULL, "
            "entity_kind TEXT NOT NULL, content_hash TEXT NOT NULL, "
            "embedding TEXT NOT NULL, created_at TEXT NOT NULL, "
            "PRIMARY KEY (canonical_id, embedding_model_name))"
        )
        con.execute(
            "INSERT INTO embeddings VALUES (?,?,?,?,?,?)",
            (f"memory:{db}_kb_0", "test-model", "memory", "h0", "[0.0]",
             "1970-01-01T00:00:00+00:00"),
        )
        con.commit()
    finally:
        con.close()

    (scratch / "_kb_rows.json").write_text(
        json.dumps(kb_rows if kb_rows is not None else [{"id": 0}])
    )
    (scratch / "_cache_fp.txt").write_text(cache_fp)
    return scratch


def edit_a_model(scratch: Path, db: str = "alien") -> None:
    """Simulate an agent editing a model YAML in the scratch."""
    (scratch / "models" / db / "foo.yaml").write_text(
        "name: foo\ncolumns:\n  - name: agent_added\n    sql: 1\n"
    )
