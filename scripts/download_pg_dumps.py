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
    return sorted(
        p for p in extracted.rglob("*.sql")
        if "__MACOSX" not in p.parts  # skip macOS resource-fork sidecars
    )


def _count_create_tables(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.lstrip().startswith("CREATE TABLE"))


def _iter_statements(text: str):
    """Yield SQL statements from a pg_dump `--inserts` dump.

    Handles `COPY ... FROM stdin;` … `\\.` blocks as single statements and
    drops psql meta-commands (`\\restrict` / `\\unrestrict`) that only guard the
    client and repeat per-file. Naive `;`-at-end-of-line termination is safe for
    pg_dump output (one INSERT per line, no embedded unquoted semicolons).
    """
    buf: list[str] = []
    in_copy = False
    for line in text.splitlines():
        stripped = line.strip()
        if in_copy:
            buf.append(line)
            if stripped == r"\.":
                in_copy = False
                yield "\n".join(buf)
                buf = []
            continue
        if stripped.startswith("\\restrict") or stripped.startswith("\\unrestrict"):
            continue  # client-side guard; not needed for a trusted load
        buf.append(line)
        if stripped.upper().startswith("COPY ") and "FROM stdin" in stripped:
            in_copy = True
            continue
        if stripped.endswith(";"):
            yield "\n".join(buf)
            buf = []
    if buf:
        yield "\n".join(buf)


def _first_sql_line(stmt: str) -> str:
    """The first non-comment, non-blank line — pg_dump prefixes each statement
    with `-- Name: …; Type: …` comment lines."""
    for line in stmt.splitlines():
        s = line.strip()
        if s and not s.startswith("--"):
            return s
    return ""


def _is_fk_statement(stmt: str) -> bool:
    s = _first_sql_line(stmt).upper()
    return s.startswith("ALTER TABLE") and "FOREIGN KEY" in stmt.upper()


def _is_type_statement(stmt: str) -> bool:
    # CREATE TYPE / CREATE DOMAIN must precede the tables that use them.
    s = _first_sql_line(stmt).upper()
    return s.startswith("CREATE TYPE") or s.startswith("CREATE DOMAIN")


def combine_per_table_dumps(sql_files: list[Path]) -> str:
    """Build one loadable dump from large-v1's per-table files.

    The large-v1 zip ships one file per table (each: CREATE TABLE + INSERTs +
    its own FK constraints) plus partial `*_full.sql` aggregates. Loading a
    single file (the old "pick largest" heuristic) yields a 1-table DB; the
    `*_full.sql` aggregates are themselves incomplete for some DBs (e.g.
    disaster_relief: 29/49 tables). So concatenate the SINGLE-table files (the
    authoritative complete set, as the upstream loader uses) and DEFER every
    foreign-key constraint to the end, so the combined dump loads in ONE clean
    pass regardless of table order (all tables exist before any FK is added).
    """
    texts = [(f, f.read_text(encoding="utf-8", errors="replace")) for f in sql_files]
    per_table = [(f, t) for f, t in texts if _count_create_tables(t) == 1]
    # Fall back to the largest-by-table-count aggregate if there are no
    # per-table files (e.g. a benchmark that only ships a combined dump).
    chosen = per_table or [max(texts, key=lambda ft: _count_create_tables(ft[1]))]

    head: list[str] = [
        "-- Combined by scripts/download_pg_dumps.py "
        "(per-table; types first, FKs deferred).",
        "CREATE EXTENSION IF NOT EXISTS hstore;",
        "CREATE EXTENSION IF NOT EXISTS citext;",
    ]
    types: list[str] = []   # CREATE TYPE/DOMAIN — before any table that uses them
    body: list[str] = []    # tables + data + inline PK/unique constraints
    fks: list[str] = []     # foreign keys — after every table exists
    for _f, text in sorted(chosen, key=lambda ft: ft[0].name):
        for stmt in _iter_statements(text):
            if not stmt.strip():
                continue
            bucket = (types if _is_type_statement(stmt)
                      else fks if _is_fk_statement(stmt)
                      else body)
            bucket.append(stmt.rstrip())
    return "\n".join(
        head
        + ["-- Custom types."] + types
        + ["-- Tables, data, and inline constraints."] + body
        + ["-- Deferred foreign-key constraints."] + fks
    ) + "\n"


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

        # Group extracted SQL files by target DB and stage pg_dumps/<db>/<db>.sql.
        # Strip a trailing "_template" suffix: the upstream zips use e.g.
        # alien_template/alien_template.sql, but instances reference "alien".
        #
        # large-v1 ships MANY files per DB (one per table + partial `*_full.sql`
        # aggregates). Concatenating the single-table files with FKs deferred
        # (combine_per_table_dumps) is the only complete option — the aggregate
        # `*_full.sql` is itself missing tables for some DBs (disaster_relief:
        # 29/49). A DB with a single source file (base-lite/full docker dumps) is
        # copied verbatim.
        by_target: dict[Path, list[Path]] = {}
        for sql_file in _find_sql_files(tmp):
            raw_db = sql_file.parent.name if sql_file.parent != tmp else sql_file.stem
            db = raw_db.removesuffix("_template")
            target = dest / db / f"{db}.sql"
            by_target.setdefault(target, []).append(sql_file)

        placed = 0
        for target, sources in sorted(by_target.items()):
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                print(f"  skip (exists): {target.relative_to(data_root)}")
                continue
            if not dry_run:
                if len(sources) == 1:
                    shutil.copy2(sources[0], target)
                else:
                    target.write_text(combine_per_table_dumps(sources),
                                      encoding="utf-8")
            n = 1 if len(sources) == 1 else len(sources)
            print(f"  placed: {target.relative_to(data_root)} "
                  f"({'copied' if len(sources) == 1 else f'combined {n} files'})")
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
