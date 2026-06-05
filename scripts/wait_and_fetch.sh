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

env -u SSH_AUTH_SOCK uv run python -c "
from bird_interact_agents.cloud import driver, gcs
client = gcs.default_gcs_client()
manifest = gcs.read_manifest('$RUN_ID', client=client)
print(f'Waiting for $RUN_ID ({manifest.get(\"dataset\")})...', flush=True)
result = driver.wait_until_done('$RUN_ID', manifest, poll_interval_s=30.0)
print(f'TERMINAL: {result.terminal_state}  {result.hint!r}', flush=True)
"

echo "Fetching $RUN_ID..."
env -u SSH_AUTH_SOCK uv run bird-interact-cloud fetch "$RUN_ID"
echo "Done: $RUN_ID"
