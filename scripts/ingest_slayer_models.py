"""Ingest SLayer models for every mini-interact SQLite DB.

For each subdirectory of the mini-interact data dir that contains a
`<name>.sqlite` file, create a per-DB SLayer YAML store at
`slayer_storage/<name>/` with one datasource and auto-generated models.

After auto-ingestion, this script also overlays bird-interact's
`<db>_column_meaning_base.json` onto the generated YAML so dimensions
and measures carry their semantic descriptions — without this step,
SLayer-mode agents see only column names while raw-mode agents see
full descriptions, breaking information parity.

Usage:
    python scripts/ingest_slayer_models.py \
        --db-path /path/to/mini-interact \
        --storage-root ./slayer_storage
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def ingest_one(db_name: str, sqlite_path: Path, storage_dir: Path) -> bool:
    storage_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SLAYER_STORAGE"] = str(storage_dir)

    conn_str = f"sqlite:////{sqlite_path}"

    # Step 1: create the datasource (with fixed 4-slash absolute URL)
    create_cmd = [
        "slayer", "datasources", "create", conn_str,
        "--name", db_name,
    ]
    create = subprocess.run(create_cmd, env=env, capture_output=True, text=True)
    if create.returncode != 0 and "already exists" not in create.stderr:
        # Fall back to direct yaml write if CLI parsing dropped a slash
        ds_dir = storage_dir / "datasources"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / f"{db_name}.yaml").write_text(
            f"name: {db_name}\ntype: sqlite\nconnection_string: {conn_str}\n"
        )

    # Always rewrite the connection string to ensure 4 slashes
    ds_yaml = storage_dir / "datasources" / f"{db_name}.yaml"
    if ds_yaml.exists():
        text = ds_yaml.read_text()
        text = text.replace(
            f"sqlite:///{sqlite_path}",
            f"sqlite:////{sqlite_path}",
        )
        ds_yaml.write_text(text)

    # Step 2: ingest models
    ingest_cmd = ["slayer", "ingest", "--datasource", db_name]
    ingest = subprocess.run(ingest_cmd, env=env, capture_output=True, text=True)
    if ingest.returncode != 0:
        print(f"  FAILED: {ingest.stderr.splitlines()[-1] if ingest.stderr else ''}")
        return False

    print(f"  {ingest.stdout.strip().splitlines()[-1] if ingest.stdout else 'ingested'}")
    return True


def overlay_column_meanings(
    db_name: str, db_dir: Path, storage_dir: Path
) -> int:
    """Set `description` on dimensions/measures from `<db>_column_meaning_base.json`.

    Keys in the meanings JSON are `<db>|<table>|<column>`. For every
    matching dim/measure in `<storage>/<db>/models/<table>.yaml`, we
    write the meaning into the `description` field. Returns the number
    of fields updated (for logging).
    """
    meanings_file = db_dir / f"{db_name}_column_meaning_base.json"
    models_dir = storage_dir / "models"
    if not meanings_file.is_file() or not models_dir.is_dir():
        return 0

    meanings: dict = json.loads(meanings_file.read_text())
    # Mini-interact mixes PascalCase (Observatories.ObservStation) with the
    # lowercased identifiers SLayer ingests, so we index on lowercase keys.
    by_table: dict[str, dict[str, str]] = {}
    for key, meaning in meanings.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _db, table, col = parts
        by_table.setdefault(table.lower(), {})[col.lower()] = meaning

    updated = 0
    for yaml_file in models_dir.glob("*.yaml"):
        model = yaml.safe_load(yaml_file.read_text()) or {}
        table = (model.get("sql_table") or yaml_file.stem).lower()
        col_to_meaning = by_table.get(table, {})
        if not col_to_meaning:
            continue
        for field_kind in ("dimensions", "measures"):
            for entry in model.get(field_kind) or []:
                col = (entry.get("sql") or entry.get("name") or "").lower()
                meaning = col_to_meaning.get(col)
                if meaning and not entry.get("description"):
                    entry["description"] = meaning
                    updated += 1
        yaml_file.write_text(yaml.safe_dump(model, sort_keys=False))
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, help="Path to mini-interact/")
    parser.add_argument(
        "--storage-root",
        default="./slayer_storage",
        help="Root dir for per-DB SLayer stores",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path).resolve()
    storage_root = Path(args.storage_root).resolve()

    if not db_path.is_dir():
        print(f"DB path not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    failed = []
    for sub in sorted(db_path.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        sqlite_file = sub / f"{sub.name}.sqlite"
        if not sqlite_file.is_file():
            continue
        print(f"Ingesting {sub.name}...")
        ok = ingest_one(sub.name, sqlite_file, storage_root / sub.name)
        if not ok:
            failed.append(sub.name)
            continue
        updated = overlay_column_meanings(sub.name, sub, storage_root / sub.name)
        print(f"  attached {updated} column meanings")

    if failed:
        print(f"\nFAILED: {failed}", file=sys.stderr)
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
