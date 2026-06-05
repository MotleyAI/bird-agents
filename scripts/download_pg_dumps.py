"""Stage PostgreSQL database dumps for cloud benchmark runs.

== RECOMMENDED: extract dumps from the official Docker images ==

The official LiveSQLBench PostgreSQL databases are published as Docker Hub images.
This is the fastest and most reliable way to get the dumps:

  # base-lite (18 DBs) — also used by bird-interact-lite-exp
  docker run -d --name livesql_lite -e POSTGRES_USER=root -e POSTGRES_PASSWORD=123123 \\
    shawnxxh/bird-interact-postgresql:latest
  # wait for "Done creating real DBs" in `docker logs livesql_lite`
  for db in alien archeology credit cross_db crypto cybermarket disaster fake gaming \\
             insider mental museum news polar robot solar vaccine virtual; do
    mkdir -p <livesqlbench-base-lite>/pg_dumps/$db
    mkdir -p <bird-interact-lite-exp>/pg_dumps/$db
    docker exec livesql_lite pg_dump -U root -d $db --no-owner --no-acl -F p \\
      > <livesqlbench-base-lite>/pg_dumps/$db/$db.sql
    cp <livesqlbench-base-lite>/pg_dumps/$db/$db.sql \\
       <bird-interact-lite-exp>/pg_dumps/$db/$db.sql
  done
  docker rm -f livesql_lite

  # base-full (22 DBs) — also used by bird-interact-full
  docker run -d --name livesql_full -e POSTGRES_USER=root -e POSTGRES_PASSWORD=123123 \\
    shawnxxh/bird-interact-postgresql-full:latest
  # wait, then dump each db (archeology_scan cold_chain_pharma_compliance
  # cross_border crypto_exchange ...) into livesqlbench-base-full/pg_dumps/
  # and copy to bird-interact-full/pg_dumps/.

  # large-v1 (21 DBs) — no pre-built Docker image; use the Google Drive zip below.

== FALLBACK: extract from Google Drive zips ==

Usage:
    uv run python scripts/download_pg_dumps.py \\
        --benchmark livesqlbench-base-lite \\
        --zip /path/to/downloaded_dumps.zip

Downloads are available from:
  livesqlbench-base-lite:  https://drive.google.com/file/d/1QIGQlRKbkqApAOrQXPqFJgUg8rQ7HRRZ
  livesqlbench-base-full:  https://drive.google.com/file/d/1V9SFIWebi27JtaDUAScG1xE9ELbYcWLR
  livesqlbench-large-v1:   https://drive.google.com/file/d/1u1L-SvJtOZGfcIST-dINw8DnGEQDMu6C

bird-interact-lite-exp and bird-interact-full share the same upstream
database dumps as livesqlbench-base-lite / livesqlbench-base-full respectively
(same DBs, postgres backend).

The script extracts the zip and reorganises the dumps into:
    <benchmark_data_root>/pg_dumps/<db>/<db>.sql

That layout is what `ray_app._ensure_postgres_loaded` expects at runtime.
The pg_dumps/ tree is automatically included in the benchmark data GCS
upload by `benchmark_data.ensure_uploaded` (uploads everything under the
data root), so no separate upload step is needed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def _find_sql_files(extracted: Path) -> list[Path]:
    return sorted(extracted.rglob("*.sql"))


def stage_dumps(benchmark: str, zip_path: Path, *, dry_run: bool = False) -> None:
    from bird_interact_agents import paths

    data_root = paths.benchmark_data_root(benchmark)
    dest = data_root / "pg_dumps"

    if not zip_path.exists():
        print(f"ERROR: zip file not found: {zip_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmark data root : {data_root}")
    print(f"Destination         : {dest}")
    print(f"Source zip          : {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    sql_entries = [n for n in names if n.endswith(".sql")]
    if not sql_entries:
        print("ERROR: zip contains no .sql files.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(sql_entries)} .sql file(s) in zip.")

    if dry_run:
        print("Dry run — not extracting.")
        return

    tmp = zip_path.parent / f"_pg_extract_{benchmark}"
    tmp.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        # Walk extracted tree and move <db>.sql files into pg_dumps/<db>/<db>.sql
        # Strip a trailing "_template" suffix: the upstream zips use
        # e.g. alien_template/alien_template.sql, but the benchmark instances
        # reference the DB as "alien".
        placed = 0
        seen_targets: set[Path] = set()
        for sql_file in _find_sql_files(tmp):
            raw_db = sql_file.parent.name if sql_file.parent != tmp else sql_file.stem
            db = raw_db.removesuffix("_template")
            db_dir = dest / db
            if not dry_run:
                db_dir.mkdir(parents=True, exist_ok=True)
            target = db_dir / f"{db}.sql"
            if target in seen_targets:
                continue  # duplicate after suffix-stripping — first one wins
            seen_targets.add(target)
            if target.exists():
                print(f"  skip (exists): {target.relative_to(data_root)}")
            else:
                if not dry_run:
                    shutil.copy2(sql_file, target)
                print(f"  placed: {target.relative_to(data_root)}")
                placed += 1

        print(f"\nDone — {placed} dump(s) staged under {dest}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True,
                   help="Benchmark name, e.g. livesqlbench-base-lite")
    p.add_argument("--zip", required=True, type=Path,
                   help="Path to the downloaded SQL dumps zip file")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be done without extracting")
    args = p.parse_args()
    stage_dumps(args.benchmark, args.zip, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
