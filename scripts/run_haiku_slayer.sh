#!/usr/bin/env bash
# Reproducible single-DB benchmark run: pydantic_ai + slayer + Anthropic
# Claude on a-interact mode. Each invocation creates a timestamped output
# dir under results/, filters mini_interact.jsonl to one DB, and runs
# the bird-interact CLI with the canonical-only-info prompt set.
#
# Default agent + user-sim model: claude-haiku-4-5-20251001. Override
# either with --agent-model / --user-sim-model. Script name keeps the
# original "haiku" suffix for git history continuity, but the wrapper
# is now model-agnostic.
#
# Usage:
#   scripts/run_haiku_slayer.sh --db households --concurrency 5
#   scripts/run_haiku_slayer.sh --db households \
#       --agent-model anthropic/claude-sonnet-4-5
#   scripts/run_haiku_slayer.sh --db credit --limit 3 --concurrency 1
#
# Required env: ANTHROPIC_API_KEY.
# Optional env: BIRD_DATA_PATH (default: <main_checkout>/../mini-interact/mini_interact.jsonl),
#               BIRD_DB_PATH   (default: <main_checkout>/../mini-interact).
#
# When invoked from a `git worktree`, output and slayer_models still resolve
# to the main checkout (so a deleted worktree doesn't lose results).

set -euo pipefail

DB="households"
DB_EXPLICIT=0
LIMIT=""
INSTANCE_ID=""
CONCURRENCY=5
PATIENCE=3
AGENT_MODEL="anthropic/claude-haiku-4-5-20251001"
USER_SIM_MODEL="anthropic/claude-haiku-4-5-20251001"
USE_AUDITED_GOLD=""
NO_PROMPT_CACHE=""

usage() {
  cat <<EOF
Usage: $0 [--db NAME | --instance-id ID[,ID,...]] [--limit N] [--concurrency K]
          [--patience P] [--agent-model PROVIDER/MODEL] [--user-sim-model PROVIDER/MODEL]
  --db NAME             target database (default: households). Mutually
                        exclusive with --instance-id.
  --limit N             cap on tasks to run from the filtered list
  --instance-id ID[,ID,...]
                        run one task (debug) or a comma-separated set
                        of tasks (cross-DB allowed). Mutually exclusive
                        with --db and --limit. Output dir is tagged
                        '_multi_' for multi-id runs.
  --concurrency K       concurrent task workers (default: 5)
  --patience P          user_patience_budget (default: 3, matches canonical)
  --agent-model M       LiteLLM-style model id for the system agent
                        (default: anthropic/claude-haiku-4-5-20251001)
  --user-sim-model M    LiteLLM-style model id for the user simulator
                        (default: anthropic/claude-haiku-4-5-20251001)
  --use-audited-gold-sql  Swap each task's gold sol_sql for the audited
                          version from audited_gold/<db>/<db>_audited.jsonl
                          when available. Off by default.
  --no-prompt-cache       Disable Anthropic prompt caching on the agent.
                          On by default for anthropic/* models.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db) DB="$2"; DB_EXPLICIT=1; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --instance-id) INSTANCE_ID="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --agent-model) AGENT_MODEL="$2"; shift 2 ;;
    --user-sim-model) USER_SIM_MODEL="$2"; shift 2 ;;
    --use-audited-gold-sql) USE_AUDITED_GOLD="--use-audited-gold-sql"; shift ;;
    --no-prompt-cache) NO_PROMPT_CACHE="--no-prompt-cache"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -n "$INSTANCE_ID" && -n "$LIMIT" ]]; then
  echo "--instance-id and --limit are mutually exclusive" >&2; exit 1
fi
# --db selects all tasks for one database; --instance-id selects an
# explicit cross-DB id list. Allowing both forces a silent choice
# between them since they're alternative selection strategies.
if [[ -n "$INSTANCE_ID" && $DB_EXPLICIT -eq 1 ]]; then
  echo "--db and --instance-id are mutually exclusive" >&2; exit 1
fi

# Anchor everything to the main checkout so worktree runs share results.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_GIT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --git-common-dir)" || {
  echo "Failed to resolve git common dir from $SCRIPT_DIR" >&2; exit 1
}
[[ "$COMMON_GIT_DIR" = /* ]] || COMMON_GIT_DIR="$SCRIPT_DIR/$COMMON_GIT_DIR"
MAIN_CHECKOUT="$(cd "$(dirname "$COMMON_GIT_DIR")" && pwd)" || exit 1
cd "$MAIN_CHECKOUT" || exit 1

: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running.}"
: "${BIRD_DATA_PATH:=$MAIN_CHECKOUT/../mini-interact/mini_interact.jsonl}"
: "${BIRD_DB_PATH:=$MAIN_CHECKOUT/../mini-interact}"

if [[ ! -f "$BIRD_DATA_PATH" ]]; then
  echo "Dataset jsonl not found: $BIRD_DATA_PATH" >&2; exit 1
fi
if [[ ! -d "$BIRD_DB_PATH" ]]; then
  echo "DB path not found: $BIRD_DB_PATH" >&2; exit 1
fi

# Compact tag for the output dir: strip provider prefix + Anthropic
# date suffix. `claude-haiku-4-5-20251001` -> `haiku-4-5`,
# `claude-sonnet-4-5` -> `sonnet-4-5`. Falls through to the bare model
# id for non-Claude models.
MODEL_TAG="$(printf '%s' "$AGENT_MODEL" \
  | sed -E 's|^[^/]+/||; s|^claude-||; s|-202[0-9]{5,}$||')"

TS=$(date +%Y%m%d-%H%M)
OUT_SUFFIX=""
# --instance-id accepts a single id or a comma-separated list. Single id
# gets the historical `__task-<id>` suffix; multiple ids get
# `__tasks-<first>-and-N-more` so the output dir stays short.
if [[ -n "$INSTANCE_ID" ]]; then
  if [[ "$INSTANCE_ID" == *,* ]]; then
    first="${INSTANCE_ID%%,*}"
    total=$(awk -F, '{print NF}' <<<"$INSTANCE_ID")
    rest=$((total - 1))
    OUT_SUFFIX="__tasks-${first//\//_}-and-${rest}-more"
  else
    OUT_SUFFIX="__task-${INSTANCE_ID//\//_}"
  fi
fi
# When --instance-id is used, ids can span databases; tag the OUT dir
# with 'multi' instead of the (irrelevant) default $DB.
DB_TAG="$DB"
if [[ -n "$INSTANCE_ID" ]]; then DB_TAG="multi"; fi
OUT="$MAIN_CHECKOUT/results/${TS}_pa_${MODEL_TAG}_${DB_TAG}_slayer_a${OUT_SUFFIX}"
mkdir -p "$OUT"

if [[ -n "$INSTANCE_ID" ]]; then
  # Split comma-separated ids onto one-per-line, trim whitespace.
  printf '%s' "$INSTANCE_ID" \
    | tr ',' '\n' \
    | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
    | awk 'NF' \
    > "$OUT/instance_ids.txt"
  n_ids=$(wc -l < "$OUT/instance_ids.txt")
  echo "task-subset mode: ${n_ids} instance_id(s) -> $OUT/instance_ids.txt"
else
  # Filter the dataset to the chosen DB.
  python3 - "$BIRD_DATA_PATH" "$DB" "$OUT/instance_ids.txt" <<'PY'
import json, sys
data, db, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = 0
with open(data) as f, open(out, "w") as g:
    for line in f:
        if not line.strip():
            continue
        t = json.loads(line)
        if t.get("selected_database") == db:
            g.write(t["instance_id"] + "\n")
            n += 1
print(f"selected {n} tasks for db={db!r} -> {out}")
PY

  if [[ ! -s "$OUT/instance_ids.txt" ]]; then
    echo "No tasks matched db=$DB; aborting." >&2; exit 1
  fi

  # `--limit` is a CLI-side post-filter: it truncates AFTER --filter-ids
  # would normally apply. Applying it via the bird-interact CLI takes the
  # first N rows of the dataset (across all DBs) before --filter-ids runs,
  # which usually matches zero tasks. Instead, truncate our per-DB id list
  # ourselves so the CLI sees only the IDs we actually want to run.
  if [[ -n "$LIMIT" ]]; then
    head -n "$LIMIT" "$OUT/instance_ids.txt" > "$OUT/instance_ids.txt.tmp"
    mv "$OUT/instance_ids.txt.tmp" "$OUT/instance_ids.txt"
  fi
fi

# Stash the invocation for replay.
{
  echo "DB=$DB"
  echo "LIMIT=$LIMIT"
  echo "INSTANCE_ID=$INSTANCE_ID"
  echo "CONCURRENCY=$CONCURRENCY"
  echo "PATIENCE=$PATIENCE"
  echo "BIRD_DATA_PATH=$BIRD_DATA_PATH"
  echo "BIRD_DB_PATH=$BIRD_DB_PATH"
  echo "MAIN_CHECKOUT=$MAIN_CHECKOUT"
  echo "AGENT_MODEL=$AGENT_MODEL"
  echo "USER_SIM_MODEL=$USER_SIM_MODEL"
  date -Iseconds
} > "$OUT/invocation.txt"

uv run bird-interact \
  --dataset mini_interact \
  --framework pydantic_ai \
  --query-mode slayer \
  --mode a-interact \
  --agent-model "$AGENT_MODEL" \
  --user-sim-model "$USER_SIM_MODEL" \
  --slayer-storage-root "$MAIN_CHECKOUT/slayer_models" \
  --data "$BIRD_DATA_PATH" \
  --db-path "$BIRD_DB_PATH" \
  --filter-ids "$OUT/instance_ids.txt" \
  --concurrency "$CONCURRENCY" \
  --patience "$PATIENCE" \
  --output "$OUT/eval.json" \
  $USE_AUDITED_GOLD \
  $NO_PROMPT_CACHE \
  2>&1 | tee "$OUT/run.log"

echo
echo "Run complete. Output dir: $OUT"
echo "  - eval.json   (canonical metrics + per-task results dump)"
echo "  - results.db  (per-task SQLite with diagnostic columns)"
echo "  - run.log     (full stdout/stderr)"
echo "  - invocation.txt + instance_ids.txt (for replay)"
