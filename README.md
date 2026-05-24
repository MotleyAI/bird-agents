# bird-interact-agents

Pluggable agent benchmarks for [BIRD-Interact](https://github.com/bird-bench/BIRD-Interact) mini (300 SQLite tasks). Evaluates query generation via [SLayer](https://github.com/MotleyAI/slayer) semantic layer and raw SQL, across multiple agent frameworks.

See [ROADMAP.md](ROADMAP.md) for the multi-framework plan.

## Quick Start

### Prerequisites

1. Clone [BIRD-Interact](https://github.com/bird-bench/BIRD-Interact) **as a sibling of this checkout**:
   ```bash
   # cd into the parent directory of where you cloned bird-interact-agents
   git clone https://github.com/bird-bench/BIRD-Interact.git
   ```
   The `original` and `all` extras (used by `scripts/run_three_way.sh`) wire up
   the upstream `mini-interact-agent` package via a relative path
   (`../BIRD-Interact/mini_interact/knowledge_based/mini_interact_agent`) — so
   the directory layout matters. If you work from a `git worktree` directory
   the relative path won't resolve; either symlink `BIRD-Interact` next to the
   worktree or `uv pip install -e <absolute-path>/mini_interact/knowledge_based/mini_interact_agent`
   into the worktree's venv after `uv sync`.

2. Get the mini-interact dataset (SQLite DBs + metadata) from [HuggingFace](https://huggingface.co/datasets/birdsql/mini-interact) or use a local copy.

3. Set environment variables:
   ```bash
   export BIRD_BIRD_INTERACT_ROOT=/path/to/BIRD-Interact
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

### Install

```bash
pip install -e ".[claude-sdk,dev]"
```

### Run

```bash
# Validate eval pipeline (submits ground-truth SQL, no LLM)
bird-interact --mode oracle \
  --data /path/to/mini_interact.jsonl \
  --db-path /path/to/mini-interact/

# Run with Claude Agent SDK, raw SQL mode
bird-interact --framework claude_sdk --query-mode raw --mode a-interact \
  --data /path/to/mini_interact.jsonl \
  --db-path /path/to/mini-interact/ \
  --limit 10 --concurrency 3

# Run with SLayer mode (requires SLayer models to be ingested)
bird-interact --framework claude_sdk --query-mode slayer --mode a-interact \
  --data /path/to/mini_interact.jsonl \
  --db-path /path/to/mini-interact/
```

## 3-way comparison (original ↔ raw ↔ slayer)

`scripts/run_three_way.sh` runs the upstream BIRD-Interact harness, our raw-SQL flavour, and our SLayer flavour on the same `instance_id` slice and emits a side-by-side `comparison.json`.

Prerequisites:
```bash
export BIRD_BIRD_INTERACT_ROOT=/path/to/BIRD-Interact
export BIRD_DATA_PATH=/path/to/mini_interact.jsonl
export BIRD_DB_PATH=/path/to/mini-interact
export ANTHROPIC_API_KEY=sk-ant-...
uv sync --extra all --extra dev   # brings in the upstream harness via tool.uv.sources
```

Run:
```bash
bash scripts/run_three_way.sh --mode a-interact --limit 30 --concurrency 4
```

Defaults to `--framework pydantic_ai` because `claude_sdk` cannot run from inside an active Claude Code session (stdio collision with the spawned `claude` subprocess). `--parallel` runs the three versions concurrently. The output directory contains:

- `original/results.jsonl`, `raw/eval.json`, `slayer/eval.json` — raw per-version outputs
- `comparison.json` — `{summary: {<version>: {n, phase1_rate, phase2_rate, avg_reward, errors}}, per_task: {<id>: {<version>: ...}}}` for direct row-by-row comparison
- A Markdown table is also printed to stdout

### Choosing a model

Pass any LiteLLM-style provider/model string via `--agent-model` (and optionally `--user-sim-model`). LiteLLM auto-resolves the base URL and reads the matching API-key env var:

```bash
--agent-model cerebras/zai-glm-4.7              # GLM-4.7 on Cerebras (preview, fast tool calling)
--agent-model anthropic/claude-sonnet-4-5       # Default; required for claude_sdk framework
--agent-model openrouter/z-ai/glm-4.7-flash     # GLM-4.7 Flash via OpenRouter
--agent-model fireworks_ai/glm-4p7              # GLM-4.7 on Fireworks
--agent-model cerebras/llama3.1-8b              # Llama 3.1 8B on Cerebras
```

Set the corresponding env var: `CEREBRAS_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `FIREWORKS_API_KEY`, `ZHIPU_API_KEY`.

Caveats:
- `claude_sdk` is locked to Anthropic by SDK design — passing a non-Anthropic `--agent-model` causes that single framework to skip with a warning (other frameworks run normally).
- `mcp_agent` ships only Anthropic + OpenAI augmented LLMs, so non-Anthropic models route through OpenAI-compatible endpoints (configured via `_build_settings`).
- The user-sim model defaults to `anthropic/claude-haiku-4-5-20251001`. Swap with `--user-sim-model cerebras/llama3.1-8b` for fully-non-Anthropic runs.

## Query Modes

- **`raw`**: Agent gets direct DB tools (`execute_sql`, `get_schema`, `get_column_meaning`, etc.) and writes SQL.
- **`slayer`**: Agent uses SLayer MCP tools (`models_summary`, `inspect_model`, `query`). Doesn't know about SQL/SQLite.

## Agent Frameworks

| Framework | Status | Install extra |
|-----------|--------|--------------|
| Claude Agent SDK | Active | `claude-sdk` |
| PydanticAI | Planned | — |
| smolagents | Planned | — |
| Agno | Planned | — |
| mcp-agent | Planned | — |

## License

MIT
