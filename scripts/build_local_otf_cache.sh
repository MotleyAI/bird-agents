#!/usr/bin/env bash
# Build the deterministic OTF slayer cache for all DBs in a postgres benchmark,
# using a user-space postgres instance (no sudo needed).
#
# Usage:
#   scripts/build_local_otf_cache.sh <benchmark-name>
#
# Examples:
#   scripts/build_local_otf_cache.sh livesqlbench-base-lite
#   scripts/build_local_otf_cache.sh bird-interact-lite-exp
#
# The script:
#   1. Starts a user-space postgres (or reuses an existing one) at $PGDATA
#      on port $PGPORT (default 5435, to avoid clashing with any system postgres)
#   2. Loads all pg_dumps/<db>/<db>.sql for the benchmark (skips existing DBs)
#   3. Builds the deterministic slayer OTF cache for every DB
#   4. Leaves postgres running for subsequent runs (re-run the script for other
#      benchmarks — dump loading is idempotent)
#
# Environment overrides:
#   PGDATA    postgres data directory  (default: ~/.local/share/bird-agents/pgdata)
#   PGPORT    postgres port            (default: 5435)
#   PGHOST    postgres socket/host     (default: localhost)

set -euo pipefail

BENCHMARK="${1:?Usage: $0 <benchmark-name>  e.g. livesqlbench-base-lite}"

# ---------------------------------------------------------------------------
# Postgres settings (user-space, no sudo)
# ---------------------------------------------------------------------------
PGDATA="${PGDATA:-$HOME/.local/share/bird-agents/pgdata}"
# NOTE (DEV-1685): the OTF cache impl fingerprint NO LONGER embeds the postgres
# connection (host:port:user) — the connection is runtime-reanchored from
# BIRD_PG_* and persisted references are portabilised to canonical defaults, so
# the port you build on here does not affect cache reuse across machines/cloud.
# That removed the old reason to pin 5432; default to 5435 instead (matching the
# usage doc above) because :5432 is RESERVED for the IAP tunnel — defaulting the
# builder there would block or collide with it. Override PGPORT if 5435 is taken.
PGPORT="${PGPORT:-5435}"
PGHOST="${PGHOST:-localhost}"
ADMIN_USER="postgres"  # role created by initdb; used only for admin operations
CACHE_USER="${CACHE_USER:-bird_interact}"  # role used for cache build — must match cloud

# Locate pg binaries (prefer 16 → 14 → PATH).
for PG_BIN in /usr/lib/postgresql/16/bin /usr/lib/postgresql/14/bin ""; do
    if [[ -n "$PG_BIN" && -x "$PG_BIN/pg_ctl" ]]; then
        export PATH="$PG_BIN:$PATH"
        break
    fi
done
command -v pg_ctl >/dev/null || { echo "ERROR: pg_ctl not found"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Initialize data dir if needed
# ---------------------------------------------------------------------------
if [[ ! -d "$PGDATA/global" ]]; then
    echo "==> Initialising postgres data dir at $PGDATA"
    mkdir -p "$PGDATA"
    initdb -D "$PGDATA" -U "$ADMIN_USER" --auth=trust --no-instructions
fi

# ---------------------------------------------------------------------------
# 2. Start postgres if not already running
# ---------------------------------------------------------------------------
if ! pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
    echo "==> Starting postgres on port $PGPORT"
    pg_ctl -D "$PGDATA" -l "$PGDATA/pg.log" \
        -o "-p $PGPORT -h $PGHOST -k /tmp" start
else
    echo "==> Postgres already running"
fi

# Wait until postgres is actually accepting connections (up to 30 s).
for i in $(seq 1 30); do
    pg_isready -h "$PGHOST" -p "$PGPORT" -U "$ADMIN_USER" -q && break
    sleep 1
done
pg_isready -h "$PGHOST" -p "$PGPORT" -U "$ADMIN_USER" -q || {
    echo "ERROR: postgres is not reachable at $PGHOST:$PGPORT after 30 s (PGDATA=$PGDATA)"
    exit 1
}

# Create the application role used by the cache builder (matches cloud default).
# Validate CACHE_USER first — it's used both as an identifier AND in a string
# literal AND as the password, so any non-[A-Za-z_][A-Za-z0-9_]* character
# would break the SQL or open an injection vector. Mirrors the _SAFE_DB_NAME
# check in ray_app.py::_ensure_postgres_loaded.
if ! [[ "$CACHE_USER" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
    echo "ERROR: CACHE_USER must match [A-Za-z_][A-Za-z0-9_]* (got: '$CACHE_USER')"
    exit 1
fi
psql -h "$PGHOST" -p "$PGPORT" -U "$ADMIN_USER" -d postgres -c \
    "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$CACHE_USER') \
     THEN CREATE ROLE $CACHE_USER LOGIN SUPERUSER PASSWORD '$CACHE_USER'; \
     END IF; END \$\$;" -q

# ---------------------------------------------------------------------------
# 3. Load dumps for this benchmark (idempotent — skips existing DBs)
# ---------------------------------------------------------------------------
DATA_ROOT=$(uv run python - "$BENCHMARK" <<'PYEOF'
import sys
from bird_interact_agents import paths
print(paths.benchmark_data_root(sys.argv[1]))
PYEOF
)
DUMPS_DIR="$DATA_ROOT/pg_dumps"

if [[ ! -d "$DUMPS_DIR" ]]; then
    echo "ERROR: $DUMPS_DIR not found — run scripts/download_pg_dumps.py first"
    exit 1
fi

echo "==> Loading dumps from $DUMPS_DIR"
loaded=0
skipped=0
for db_dir in "$DUMPS_DIR"/*/; do
    db=$(basename "$db_dir")
    sql_file="$db_dir/$db.sql"
    [[ -f "$sql_file" ]] || continue

    if psql -h "$PGHOST" -p "$PGPORT" -U "$ADMIN_USER" -lqt \
            | cut -d'|' -f1 | grep -qw "$db"; then
        echo "  skip (exists): $db"
        ((skipped++)) || true
    else
        echo "  loading: $db"
        createdb -h "$PGHOST" -p "$PGPORT" -U "$ADMIN_USER" "$db"
        # Drop the partial DB if psql fails — otherwise next run sees the
        # DB name and skips it, building the cache against incomplete data.
        # Matches the cloud loader's dropdb-before-retry pattern in
        # ray_app.py::_ensure_postgres_loaded.
        if ! psql -h "$PGHOST" -p "$PGPORT" -U "$ADMIN_USER" -d "$db" \
                -f "$sql_file" -q --output /dev/null; then
            echo "    load FAILED — dropping partial $db so the next run reloads"
            dropdb -h "$PGHOST" -p "$PGPORT" -U "$ADMIN_USER" --if-exists "$db" || true
            exit 1
        fi
        ((loaded++)) || true
    fi
done
echo "    $loaded loaded, $skipped skipped"

# ---------------------------------------------------------------------------
# 4. Build the OTF cache for every DB in the benchmark
# ---------------------------------------------------------------------------
echo "==> Building OTF slayer cache for $BENCHMARK"

BIRD_PG_HOST="$PGHOST" \
BIRD_PG_PORT="$PGPORT" \
BIRD_PG_USER="$CACHE_USER" \
BIRD_PG_PASSWORD="$CACHE_USER" \
uv run python - "$BENCHMARK" <<'PYEOF'
import asyncio, sys
from bird_interact_agents import paths
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.slayer_otf.cache import ensure_db_cache

benchmark_name = sys.argv[1]
bench = get_benchmark(benchmark_name)
cache_root = paths.slayer_otf_cache_root(benchmark=benchmark_name)
data_root = paths.benchmark_data_root(benchmark_name)

# Collect unique DBs from the benchmark data file
import json
data_file = data_root / bench.data_file
dbs = sorted({
    json.loads(line)["selected_database"]
    for line in data_file.read_text().splitlines()
    if line.strip()
})

print(f"Building cache for {len(dbs)} DBs: {dbs}")
for db in dbs:
    print(f"  {db} ...", flush=True)
    asyncio.run(ensure_db_cache(
        db,
        cache_root=cache_root,
        mini_interact_root=data_root,
        benchmark=bench,
        force=False,
    ))
    print(f"  {db} done", flush=True)

print(f"\nCache built at: {cache_root}")
PYEOF

echo ""
echo "Done. To use this postgres for future runs, add to your shell:"
echo "  export BIRD_PG_HOST=$PGHOST"
echo "  export BIRD_PG_PORT=$PGPORT"
echo "  export BIRD_PG_USER=$CACHE_USER"
echo "  export BIRD_PG_PASSWORD=$CACHE_USER"
