#!/usr/bin/env bash
# Monitor live annotator clusters: fetch + kill when done, report progress when not.
#
# Usage:
#   bash scripts/check_annotation_runs.sh [--dry-run]
#
# Options:
#   --dry-run  Print what would happen without fetching or killing anything
#
# Only looks at runs that are currently LIVE (cluster still up) and are annotator runs.
# Runs whose cluster is already gone (state=done) are ignored — fetch them manually if needed.
#
# For each live annotator run:
#   - all tasks done  → fetch results, then kill the cluster
#   - tasks remaining → report progress
#
# Exit code: 0 if any runs were fetched, 1 if none (including no live runs found).
#
# This script is auto-approved in ~/.claude/settings.json so Claude Code can
# run it without a user confirmation prompt.

set -euo pipefail

CLOUD="env -u SSH_AUTH_SOCK uv run bird-interact-cloud"
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
    esac
done

LIST_OUTPUT=$($CLOUD list 2>&1) || true
if [[ -z "$LIST_OUTPUT" ]]; then
    echo "No runs found."
    exit 1
fi

FETCHED=0
LIVE_FOUND=0

while IFS= read -r line; do
    [[ -z "$line" ]] && continue

    run_id=$(awk '{print $1}' <<< "$line")
    combo=$(awk '{print $2}' <<< "$line")
    state=$(awk '{print $3}' <<< "$line")
    progress=$(awk '{print $4}' <<< "$line")

    # Only care about live annotator runs
    [[ "$combo" != annotator/* ]] && continue
    [[ "$state" != "live" ]] && continue

    LIVE_FOUND=$((LIVE_FOUND + 1))
    done_count=$(cut -d/ -f1 <<< "$progress")
    total_count=$(cut -d/ -f2 <<< "$progress")

    if [[ "$total_count" -gt 0 && "$done_count" -eq "$total_count" ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[dry-run] $run_id ($combo): $progress — would fetch + kill"
        else
            echo "Fetching $run_id ($combo, $progress) ..."
            # fetch auto-kills the cluster when all tasks are done (default behaviour)
            $CLOUD fetch "$run_id" || echo "  WARNING: fetch failed for $run_id (continuing)"
            echo ""
        fi
        FETCHED=$((FETCHED + 1))
    else
        echo "In progress: $run_id ($combo)  $done_count / $total_count tasks done"
    fi
done <<< "$LIST_OUTPUT"

echo ""
if [[ "$LIVE_FOUND" -eq 0 ]]; then
    echo "No live annotator clusters found."
elif [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete — $FETCHED run(s) would be fetched."
else
    echo "Done — $FETCHED run(s) fetched."
fi

[[ "$FETCHED" -gt 0 ]]
