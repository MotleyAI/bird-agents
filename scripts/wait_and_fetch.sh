#!/usr/bin/env bash
# Wait for a cloud run to terminate, then fetch its results and tear down
# the cluster. The canonical watcher pattern — kick this off in the
# background immediately after `bird-interact-cloud submit`/`annotate`
# so the cluster shuts down the moment the run terminates.
#
# Usage:
#   scripts/wait_and_fetch.sh <run-id>
#
# Polls every 30 s via driver.wait_until_done (handles stall detection,
# no-progress deadline, headless cluster), then `bird-interact-cloud
# fetch <run-id>` (downloads results + tears down the cluster).
#
# The two operations are intentionally chained inside this script:
# wait → fetch → cluster teardown is one logical operation.
# `submit` is NOT chained here — submit first, then invoke this script
# in the background. See ~/.claude/projects/.../memory/MEMORY.md
# (no-cloud-command-chains).

set -euo pipefail

RUN_ID="${1:?Usage: $0 <run-id>}"

env -u SSH_AUTH_SOCK uv run python - "$RUN_ID" <<'PYEOF'
import sys
from bird_interact_agents.cloud import driver, gcs
run_id = sys.argv[1]
client = gcs.default_gcs_client()
manifest = gcs.read_manifest(run_id, client=client)
print(f"Waiting for {run_id} ({manifest.get('dataset')})...", flush=True)
result = driver.wait_until_done(run_id, manifest, poll_interval_s=30.0)
print(f"TERMINAL: {result.terminal_state}  {result.hint!r}", flush=True)
PYEOF

echo "Fetching $RUN_ID..."
env -u SSH_AUTH_SOCK uv run bird-interact-cloud fetch "$RUN_ID"
echo "Done: $RUN_ID"
