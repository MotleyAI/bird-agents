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
PGPORT="${PGPORT:-5435}"
PGHOST="${PGHOST:-localhost}"
PG_USER="postgres"   # the role created by initdb inside this user-space cluster

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
    initdb -D "$PGDATA" -U "$PG_USER" --auth=trust --no-instructions
fi

# ---------------------------------------------------------------------------
# 2. Start postgres if not already running
# ---------------------------------------------------------------------------
if ! pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
    echo "==> Starting postgres on port $PGPORT"
    pg_ctl -D "$PGDATA" -l "$PGDATA/pg.log" \
        -o "-p $PGPORT -h $PGHOST -k /tmp" start
    # Give it a moment to accept connections
    sleep 2
else
    echo "==> Postgres already running"
fi

# ---------------------------------------------------------------------------
# 3. Load dumps for this benchmark (idempotent — skips existing DBs)
# ---------------------------------------------------------------------------
DATA_ROOT=$(uv run python -c "
from bird_interact_agents import paths
print(paths.benchmark_data_root('$BENCHMARK'))
")
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

    if psql -h "$PGHOST" -p "$PGPORT" -U "$PG_USER" -lqt \
            | cut -d'|' -f1 | grep -qw "$db"; then
        echo "  skip (exists): $db"
        ((skipped++)) || true
    else
        echo "  loading: $db"
        createdb -h "$PGHOST" -p "$PGPORT" -U "$PG_USER" "$db"
        psql -h "$PGHOST" -p "$PGPORT" -U "$PG_USER" -d "$db" -f "$sql_file" \
            -q --output /dev/null
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
BIRD_PG_USER="$PG_USER" \
BIRD_PG_PASSWORD="" \
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
echo "  export BIRD_PG_USER=$PG_USER"
echo "  export BIRD_PG_PASSWORD=''"
