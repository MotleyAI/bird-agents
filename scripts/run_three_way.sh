#!/usr/bin/env bash
# Run all three benchmark versions (original BIRD-Interact harness, our raw
# flavor, our slayer flavor) on the same task subset and emit a comparison.
#
# Versions are run sequentially by default (less rate-limit risk, cleaner
# logs); pass --parallel to run them concurrently with `&` + `wait`.
#
# Required env: ANTHROPIC_API_KEY (or whichever provider's key the chosen
# --agent-model / --user-sim-model needs).

set -euo pipefail

# ---- Defaults ----------------------------------------------------------------
MODE="a-interact"
LIMIT=10
CONCURRENCY=3
FRAMEWORK="claude_sdk"
AGENT_MODEL="anthropic/claude-sonnet-4-5"
USER_SIM_MODEL="anthropic/claude-haiku-4-5-20251001"
OUTPUT_DIR=""
PARALLEL=false
STRICT=false

usage() {
  cat <<EOF
Usage: $0 [options]
  --mode {a-interact,c-interact}    (default: a-interact)
  --limit N                         (default: 10)
  --concurrency K                   (default: 3)
  --output-dir DIR                  (default: results/3way_<timestamp>)
  --framework FRAMEWORK             (default: claude_sdk — note: cannot run
                                    from inside an active Claude Code session
                                    due to stdio contention with the spawned
                                    `claude` subprocess; use a plain terminal
                                    or Codex in that case)
  --agent-model MODEL               LiteLLM-style PROVIDER/MODEL_ID
                                    (default: anthropic/claude-sonnet-4-5).
                                    Examples: cerebras/zai-glm-4.7,
                                    openrouter/z-ai/glm-4.7-flash,
                                    fireworks_ai/glm-4p7. The matching
                                    API-key env var must be set.
  --user-sim-model MODEL            (default: anthropic/claude-haiku-4-5-20251001)
  --parallel                        Run the three versions concurrently
  --strict / --no-strict            Force every tool definition to carry
                                    strict=True (default: --no-strict).
                                    Only honoured by pydantic_ai/smolagents/agno;
                                    claude_sdk ignores; mcp_agent errors out.
  -h | --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --framework) FRAMEWORK="$2"; shift 2 ;;
    --agent-model) AGENT_MODEL="$2"; shift 2 ;;
    --user-sim-model) USER_SIM_MODEL="$2"; shift 2 ;;
    --parallel) PARALLEL=true; shift ;;
    --strict) STRICT=true; shift ;;
    --no-strict) STRICT=false; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

# ---- Env validation ----------------------------------------------------------
# Check the API key for whichever provider the agent + user-sim models use.
# (LiteLLM resolves the env var name from the prefix.)
provider_env() {
  case "${1%%/*}" in
    anthropic)    echo ANTHROPIC_API_KEY ;;
    cerebras)     echo CEREBRAS_API_KEY ;;
    openrouter)   echo OPENROUTER_API_KEY ;;
    fireworks_ai) echo FIREWORKS_API_KEY ;;
    zhipu)        echo ZHIPU_API_KEY ;;
    *)            echo "" ;;
  esac
}
for m in "$AGENT_MODEL" "$USER_SIM_MODEL"; do
  env_var="$(provider_env "$m")"
  if [[ -n "$env_var" && -z "${!env_var:-}" ]]; then
    echo "Error: $env_var is not set (needed for model '$m')." >&2
    exit 1
  fi
done

# Worktree-aware: anchor all paths to the main checkout so output, ingest,
# and SLayer storage land in one place regardless of which worktree we run
# from. `git rev-parse --git-common-dir` returns <main>/.git from any
# worktree of this repo; absolutise via `cd && pwd` because git may return
# a relative path (e.g. `../.git`) when invoked from inside the repo.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_GIT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --git-common-dir)" || {
  echo "Failed to resolve git common dir from $SCRIPT_DIR" >&2; exit 1
}
[[ "$COMMON_GIT_DIR" = /* ]] || COMMON_GIT_DIR="$SCRIPT_DIR/$COMMON_GIT_DIR"
MAIN_CHECKOUT="$(cd "$(dirname "$COMMON_GIT_DIR")" && pwd)" || exit 1
cd "$MAIN_CHECKOUT" || exit 1

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$MAIN_CHECKOUT/results/3way_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$OUTPUT_DIR"
# Resolve to absolute so subprocesses that cd elsewhere still find it.
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# mini-interact data sibling. Honour env-var overrides for unusual layouts.
DATA_PATH="${BIRD_DATA_PATH:-$MAIN_CHECKOUT/../mini-interact/mini_interact.jsonl}"
DB_PATH="${BIRD_DB_PATH:-$MAIN_CHECKOUT/../mini-interact}"

# ---- Step 1: lock task subset ------------------------------------------------
# Emit both an IDs file (consumed by `bird_interact_agents.run --filter-ids`)
# and a filtered JSONL containing exactly the selected rows. The upstream
# `batch_run_bird_interact.main` has no --filter-ids flag, so we feed it the
# pre-filtered JSONL via --data_path to keep all three runners on the same
# row set even if select_tasks.py later picks a non-prefix slice.
IDS_FILE="$OUTPUT_DIR/instance_ids.txt"
FILTERED_DATA="$OUTPUT_DIR/tasks.jsonl"
uv run python scripts/select_tasks.py \
  --data "$DATA_PATH" --limit "$LIMIT" \
  --out "$IDS_FILE" --out-jsonl "$FILTERED_DATA"
echo "Selected $(wc -l < "$IDS_FILE") tasks (logged in $IDS_FILE)"

# ---- Step 2: ensure SLayer storage exists ------------------------------------
SLAYER_STORAGE="$MAIN_CHECKOUT/slayer_storage"
if [[ ! -d "$SLAYER_STORAGE" ]] || [[ -z "$(ls -A "$SLAYER_STORAGE" 2>/dev/null)" ]]; then
  echo "slayer_storage not found at $SLAYER_STORAGE — running ingest..."
  uv run python scripts/ingest_slayer_models.py \
    --db-path "$DB_PATH" --storage-root "$SLAYER_STORAGE"
fi

# ---- Step 3: run each version ------------------------------------------------
# We invoke the upstream batch_run_bird_interact.main directly rather than its
# bash wrapper: the wrapper had a bug where --agent_models was assigned to a
# variable the loop never read (we patched it upstream too, but bypassing the
# wrapper means fewer moving parts and uses our uv-managed venv consistently).
run_original() {
  local out_dir="$OUTPUT_DIR/original"
  mkdir -p "$out_dir"
  echo "[original] Running upstream main.py via uv run..."
  local started ended total
  started=$(date +%s)
  uv run python -m batch_run_bird_interact.main \
    --data_path "$FILTERED_DATA" \
    --output_path "$out_dir/results.jsonl" \
    --agent_model "$AGENT_MODEL" \
    --user_model "$USER_SIM_MODEL" \
    --user_sim_mode encoder_decoder \
    --user_sim_prompt_version v2 \
    --user_patience_budget 6 \
    --max_turns 60 \
    --num_threads "$CONCURRENCY" \
    --mini_interact \
    --log_file "$out_dir/experiment.log" \
    --log_level WARNING \
    > "$out_dir/run.log" 2>&1 || {
      echo "[original] returned non-zero — see $out_dir/run.log" >&2
      return 1
    }
  ended=$(date +%s)
  total=$((ended - started))
  # Per-task duration isn't surfaced by the upstream runner. Use the
  # actual selected-task count (some datasets are shorter than --limit,
  # and select_tasks.py may emit fewer rows than requested).
  local n_tasks
  n_tasks=$(wc -l < "$IDS_FILE")
  python3 -c "
import json, sys
total = $total
n = int(sys.argv[2])
avg = total / n if n else 0.0
json.dump(
    {
        'total_duration_s': float(total),
        'n_tasks': n,
        'avg_duration_s': avg,
        'agent_model': sys.argv[3],
    },
    open(sys.argv[1], 'w'), indent=2,
)
" "$out_dir/duration.json" "$n_tasks" "$AGENT_MODEL"

  # Aggregate per-turn token_usage blobs into a single
  # token_usage_<model>.json that compare_results.py can read. Slashes
  # and dots in the model id make the filename ugly but we don't have
  # to parse it later — the comparison script globs token_usage_*.json.
  echo "[original] Running analyze_tokens to summarise token_usage..."
  uv run python -m batch_run_bird_interact.analyze_tokens \
    --output_dir "$out_dir" \
    --model "$AGENT_MODEL" \
    > "$out_dir/token_usage.log" 2>&1 || \
      echo "[original] analyze_tokens returned non-zero — see $out_dir/token_usage.log (compare_results.py will fall back to per-turn JSONLs)"

  # Backfill `original/results.db` so the original leg sits in the same
  # SQLite schema raw/slayer use after `run.py` writes their per-task
  # rows. compare_results.py then has a uniform query surface across
  # all three legs.
  echo "[original] Ingesting upstream results.jsonl into results.db..."
  uv run python -m bird_interact_agents.original_ingest \
    --orig-dir "$out_dir" \
    --db-path "$out_dir/results.db" \
    --run-id "$(basename "$OUTPUT_DIR")" \
    --mode "$MODE" \
    > "$out_dir/ingest.log" 2>&1 || \
      echo "[original] ingest returned non-zero — see $out_dir/ingest.log"
}

run_ours() {
  local query_mode="$1"
  local out_dir="$OUTPUT_DIR/$query_mode"
  mkdir -p "$out_dir"
  echo "[$query_mode] Running bird-interact (--framework $FRAMEWORK --query-mode $query_mode --mode $MODE)..."
  local strict_flag
  if $STRICT; then
    strict_flag="--strict"
  else
    strict_flag="--no-strict"
  fi
  # DEV-1586: --slayer-setup is retired; slayer mode encodes on the fly by
  # default (pass --pre-encoded-models to consume a pre-encoded reference).
  uv run python -m bird_interact_agents.run \
    --dataset mini-interact \
    --framework "$FRAMEWORK" \
    --mode "$MODE" \
    --query-mode "$query_mode" \
    --data "$DATA_PATH" \
    --db-path "$DB_PATH" \
    --output "$out_dir/eval.json" \
    --concurrency "$CONCURRENCY" \
    --filter-ids "$IDS_FILE" \
    --agent-model "$AGENT_MODEL" \
    --user-sim-model "$USER_SIM_MODEL" \
    "$strict_flag" \
    > "$out_dir/run.log" 2>&1 || {
      echo "[$query_mode] returned non-zero — see $out_dir/run.log" >&2
      return 1
    }
}

if $PARALLEL; then
  pids=()
  run_original & pids+=("$!")
  run_ours raw & pids+=("$!")
  run_ours slayer & pids+=("$!")
  rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  if (( rc != 0 )); then
    echo "One or more variants failed; aborting before comparison." >&2
    exit "$rc"
  fi
else
  run_original
  run_ours raw
  run_ours slayer
fi

# ---- Step 4: compare ---------------------------------------------------------
echo
uv run python scripts/compare_results.py "$OUTPUT_DIR"
