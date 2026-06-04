#!/usr/bin/env bash
# Check all cloud annotation runs for completion and fetch those whose tasks are all done.
#
# Usage:
#   bash scripts/check_annotation_runs.sh [--all] [--dry-run]
#
# Options:
#   --all      Check every run type (default: annotation runs only, combo starts with "annotator/")
#   --dry-run  Print what would be fetched without actually fetching
#
# Exit code: 0 if any runs were fetched (or would be in --dry-run), 1 otherwise.
#
# A run is considered "done" when either:
#   - its state column is "done"  (cluster has shut down cleanly), OR
#   - its done/total count has done == total and total > 0
#     (all submitted tasks have produced results, even while cluster is still live)
#
# This script is auto-approved in ~/.claude/settings.json so Claude Code can
# run it without a user confirmation prompt.

set -euo pipefail

CLOUD="env -u SSH_AUTH_SOCK uv run bird-interact-cloud"
ALL=0
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --all)     ALL=1 ;;
        --dry-run) DRY_RUN=1 ;;
    esac
done

LIST_OUTPUT=$($CLOUD list 2>&1) || true
if [[ -z "$LIST_OUTPUT" ]]; then
    echo "No runs found."
    exit 1
fi

FETCHED=0

while IFS= read -r line; do
    [[ -z "$line" ]] && continue

    run_id=$(awk '{print $1}' <<< "$line")
    combo=$(awk '{print $2}' <<< "$line")
    state=$(awk '{print $3}' <<< "$line")
    progress=$(awk '{print $4}' <<< "$line")

    # Filter to annotation runs unless --all
    if [[ "$ALL" -eq 0 && "$combo" != annotator/* ]]; then
        continue
    fi

    done_count=$(cut -d/ -f1 <<< "$progress")
    total_count=$(cut -d/ -f2 <<< "$progress")

    should_fetch=0
    reason=""
    if [[ "$state" == "done" ]]; then
        should_fetch=1
        reason="state=done"
    elif [[ "$total_count" -gt 0 && "$done_count" -eq "$total_count" ]]; then
        should_fetch=1
        reason="all $total_count task(s) complete"
    fi

    if [[ "$should_fetch" -eq 1 ]]; then
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[dry-run] would fetch $run_id ($combo, $reason)"
        else
            echo "Fetching $run_id ($combo, $reason) ..."
            $CLOUD fetch "$run_id" || echo "  WARNING: fetch failed for $run_id (continuing)"
            echo ""
        fi
        FETCHED=$((FETCHED + 1))
    else
        echo "Still running: $run_id ($combo, state=$state, $progress)"
    fi
done <<< "$LIST_OUTPUT"

echo ""
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete — $FETCHED run(s) would be fetched."
else
    echo "Done — $FETCHED run(s) fetched."
fi

[[ "$FETCHED" -gt 0 ]]
