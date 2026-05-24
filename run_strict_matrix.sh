#!/usr/bin/env bash
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_GIT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --git-common-dir)" || {
  echo "Failed to resolve git common dir from $SCRIPT_DIR" >&2; exit 1
}
[[ "$COMMON_GIT_DIR" = /* ]] || COMMON_GIT_DIR="$SCRIPT_DIR/$COMMON_GIT_DIR"
MAIN_CHECKOUT="$(cd "$(dirname "$COMMON_GIT_DIR")" && pwd)" || exit 1
cd "$MAIN_CHECKOUT" || exit 1

DATA="${BIRD_DATA_PATH:-$MAIN_CHECKOUT/../mini-interact/mini_interact.jsonl}"
DB="${BIRD_DB_PATH:-$MAIN_CHECKOUT/../mini-interact}"

for fw in pydantic_ai smolagents agno mcp_agent claude_sdk; do
  for strict in "--strict" "--no-strict"; do
    label=${strict#--}
    out=$MAIN_CHECKOUT/results/strict_matrix/${fw}_${label}
    mkdir -p "$out"
    echo "=== $fw $strict ==="
    uv run python -m bird_interact_agents.run \
      --framework "$fw" --query-mode raw --mode a-interact \
      --agent-model cerebras/zai-glm-4.7 \
      --user-sim-model cerebras/zai-glm-4.7 \
      --limit 1 --concurrency 1 \
      $strict \
      --data "$DATA" --db-path "$DB" \
      --output "$out/eval.json" \
      >"$out/run.log" 2>&1
    echo "exit=$?"
  done
done
echo "=== DONE ==="
