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
bird-interact --dataset mini-interact --framework claude_sdk --mode oracle \
  --data /path/to/mini_interact.jsonl \
  --db-path /path/to/mini-interact/

# Run with Claude Agent SDK, raw SQL mode
bird-interact --dataset mini-interact --framework claude_sdk --query-mode raw --mode a-interact \
  --data /path/to/mini_interact.jsonl \
  --db-path /path/to/mini-interact/ \
  --limit 10 --concurrency 3

# Run with SLayer mode (requires SLayer models to be ingested)
bird-interact --dataset mini-interact --framework claude_sdk --query-mode slayer --mode a-interact \
  --slayer-setup on-the-fly \
  --data /path/to/mini_interact.jsonl \
  --db-path /path/to/mini-interact/
```

### On-the-fly KB encoding — two flavors

Both flavors are Claude Agent SDK agents that **encode the knowledge-base
on the fly**: they run off the deterministic OTF cache (base models + KB
items pre-loaded as SLayer memories), are given the SLayer MCP with
**write** tools, and are instructed to encode the KB items they need as
named columns/measures (in dependency order, referencing earlier entities
through declared joins) and then build the final query off those named
entities instead of inlining everything. Unlike `pydantic_ai_otf_encode`
there are no forced stages or recursion, and there is no save-time
validator yet (tracked in DEV-1506). Both are **slayer-query-mode only**
and **always** `--slayer-setup on-the-fly`.

DEV-1507 split this framework along the benchmark boundary because a
single prompt that had to decide "ask user vs don't ask" based on mode
was the wrong shape — trace analysis on Opus-high mini-interact runs
showed the agent never called `ask_user` despite the prompt instruction.
The mini-interact flavor now bakes the ask-then-submit discipline into
PreToolUse hooks, not prompt text.

> **Local runs must be launched from a non-Claude-Code shell** (a plain
> terminal, or via Codex) — the SDK spawns the `claude` CLI, which fails
> when nested inside an active Claude Code session (stdio collision). Cloud
> runs are unaffected (Ray actors don't nest).

#### `claude_sdk` — LiveSQLBench / one-shot

The `claude_sdk` framework auto-selects the correct internal agent based
on benchmark and mode — `--framework claude_sdk` is the only name you
need. For LiveSQLBench one-shot, pass `--dataset livesqlbench-base-lite-sqlite
--mode one-shot`. Recommended `--reasoning-effort high` (the smoke / baseline default).

```bash
bird-interact --framework claude_sdk --query-mode slayer \
  --slayer-setup on-the-fly --dataset livesqlbench-base-lite-sqlite --mode one-shot \
  --gold-file ../livesqlbench-base-lite-sqlite/livesqlbench_sqlite_gt_kg_testcases_0528.jsonl \
  --agent-model anthropic/claude-opus-4-7 \
  --reasoning-effort high \
  --data ../livesqlbench-base-lite-sqlite/livesqlbench_data_sqlite.jsonl \
  --db-path ../livesqlbench-base-lite-sqlite/
```

Cloud smoke (the deterministic OTF cache is uploaded from local if
present, else built locally first; no reference build or upload-back).
`museum_6` is the canonical single-task smoke (passes under these defaults):

```bash
env -u SSH_AUTH_SOCK uv run bird-interact-cloud submit \
  --framework claude_sdk --query-mode slayer --slayer-setup on-the-fly \
  --mode one-shot --dataset livesqlbench-base-lite-sqlite \
  --gold-file ../livesqlbench-base-lite-sqlite/livesqlbench_sqlite_gt_kg_testcases_0528.jsonl \
  --agent-model anthropic/claude-opus-4-7 \
  --user-sim-model anthropic/claude-sonnet-4-6 \
  --reasoning-effort high \
  --instance-ids museum_6 \
  --workers 1 --actors-per-worker 1 \
  --worker-type e2-standard-4 --max-runtime-hours 2 --detach
```

For multi-instance runs bump `--actors-per-worker` so tasks run in
parallel — `1` is correct for a 1-task smoke but serialises everything
larger. `e2-standard-4` (4 vCPU / 16 GB) comfortably runs 4 concurrent
Opus + Sonnet pairs (mostly network-bound on LLM APIs), so 4 is a safe
default for 10-50 task runs:

```bash
env -u SSH_AUTH_SOCK uv run bird-interact-cloud submit \
  ... \
  --instance-ids museum_1,museum_2,museum_3,museum_4,museum_5,museum_6,museum_7,museum_8,museum_9,museum_10 \
  --workers 1 --actors-per-worker 4 \
  --worker-type e2-standard-4 --max-runtime-hours 3 --detach
```

#### `claude_sdk` — mini-interact / a-interact

For mini-interact a-interact, pass `--dataset mini-interact --mode a-interact`.
The framework adds a native `ask_user` tool plus three PreToolUse/PostToolUse guards:
the agent CANNOT call `submit_query` until it has called `ask_user` at
least once (the user-sim holds masked-KB ground truth unrecoverable from
the visible KB alone), and a nag fires every 10 tool calls until the
first ask. Recommended `--reasoning-effort high`.

```bash
bird-interact --framework claude_sdk --query-mode slayer \
  --slayer-setup on-the-fly --dataset mini-interact --mode a-interact \
  --agent-model anthropic/claude-opus-4-7 \
  --reasoning-effort high \
  --data /path/to/mini_interact.jsonl \
  --db-path /path/to/mini-interact/ --instance-id households_1
```

Cloud smoke. `households_16` is the canonical single-task smoke (passes
both audited and original gold under these defaults):

```bash
env -u SSH_AUTH_SOCK uv run bird-interact-cloud submit \
  --framework claude_sdk --query-mode slayer \
  --slayer-setup on-the-fly --dataset mini-interact --mode a-interact \
  --agent-model anthropic/claude-opus-4-7 \
  --user-sim-model anthropic/claude-sonnet-4-6 \
  --reasoning-effort high \
  --instance-ids households_16 \
  --workers 1 --actors-per-worker 1 \
  --worker-type e2-standard-4 --max-runtime-hours 2 --detach
```

For multi-instance runs bump `--actors-per-worker` to 4 (see the
sizing rule above). With the default 1×1 a 53-task batch serialises
through one actor and takes ~5 hours; at 1×4 it finishes in ~80 minutes
for the same per-task wallclock and identical cluster cost.

## Annotator Agent (DEV-1518)

The annotator agent audits benchmark tasks and produces `TaskAnnotation` records
(metadata sufficiency verdict, gold SQL correctness, audited gold variants).
It uses the `bird-interact-cloud annotate` subcommand — **not** `submit`.

### Cloud smoke

```bash
env -u SSH_AUTH_SOCK uv run bird-interact-cloud annotate \
  --benchmark mini-interact \
  --agent-model anthropic/claude-opus-4-7 \
  --instance-ids households_1,households_2,households_3 \
  --workers 1 --actors-per-worker 3 \
  --worker-type e2-standard-4 --max-runtime-hours 1 \
  --override --detach
```

`--override` re-annotates even when stable blobs already exist in GCS.
Omit it on production runs to skip already-annotated tasks.

Bump `--actors-per-worker` to match the number of instance IDs for maximum
parallelism (each actor handles one task). `e2-standard-4` comfortably runs
4 concurrent Opus actors.

### Fetching results

Annotation outputs land under `results/cloud/<RUN-ID>/` after:

```bash
env -u SSH_AUTH_SOCK uv run bird-interact-cloud fetch <RUN-ID>
```

Each annotated task produces a `<instance_id>.task.json` (the `TaskAnnotation`)
and, if the gold was wrong, one or more `<instance_id>.<variant_id>.gold.json`
entries in `audited_gold/mini-interact_audited.jsonl`.

### Waiting for a run to finish

Use `driver.wait_until_done` — do **not** poll `bird-interact-cloud list`
in a shell loop (column layout is positional and fragile):

```python
from bird_interact_agents.cloud import driver, gcs
client = gcs.default_gcs_client()
manifest = gcs.read_manifest("<RUN-ID>", client=client)
result = driver.wait_until_done("<RUN-ID>", manifest, poll_interval_s=30.0)
print(f"TERMINAL: {result.terminal_state}  {result.hint!r}")
```

Or as a one-liner:

```bash
env -u SSH_AUTH_SOCK uv run python -c "
from bird_interact_agents.cloud import driver, gcs
client = gcs.default_gcs_client()
manifest = gcs.read_manifest('<RUN-ID>', client=client)
result = driver.wait_until_done('<RUN-ID>', manifest, poll_interval_s=30.0)
print(f'TERMINAL: {result.terminal_state}  {result.hint!r}')
"
```

### Reading results: cascade aggregation and autopsy

After `bird-interact-cloud fetch <RUN-ID>`, results land under
`results/cloud/<RUN-ID>/`. The **cascading phase-1 block** breaks the
pass rate across 9 tolerance tiers (N1 = strict original gold → N9 =
case-folded):

```python
import json
from pathlib import Path
from bird_interact_agents.eval.cascading_report import aggregate_cascading_phase1

rows_dir = Path("results/cloud/<RUN-ID>/rows")
print(json.dumps(aggregate_cascading_phase1(rows_dir), indent=2))
```

Key fields: `counts.n1` (original gold), `counts.n2` (audited primary),
`deltas` (incremental gains at each tier). The `eval.json` written by
fetch already includes the full `cascading_phase1` block.

**Autopsies for `agent_miss` failures** — each
`rows/<instance_id>/submission_annotation.json` carries
`failure_classification.primary` and `evaluation.miss_diagnostics`:

```python
import json
from pathlib import Path

rows_dir = Path("results/cloud/<RUN-ID>/rows")
for sub in sorted(rows_dir.iterdir()):
    ann = json.loads((sub / "submission_annotation.json").read_text())
    fc = ann["failure_classification"]
    if fc["primary"] == "agent_miss":
        print(sub.name, fc["details"])
        print(json.dumps(ann["evaluation"]["miss_diagnostics"], indent=2))
```

`miss_diagnostics` includes row counts, overlap fraction, structural
diffs (GROUP BY / predicate count / join count), and the `miss_patterns`
list (`disjoint_rowset`, `predicate_count_mismatch`,
`aggregation_shape_mismatch`, `agent_undercount`, etc.).

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

Defaults to `--framework claude_sdk`. Note that `claude_sdk` cannot run from inside an active Claude Code session (stdio collision with the spawned `claude` subprocess) — run from a plain terminal or Codex in that case. `--parallel` runs the three versions concurrently. The output directory contains:

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

## LiveSQLBench (one-shot, DEV-1462)

LiveSQLBench-Base-Lite-SQLite is the unambiguous, one-shot sibling of
mini-interact (270 tasks / 18 DBs). The harness evaluates the **180
SELECT** tasks (`category=="Query"`) under `--mode one-shot` — no user
simulator, no `ask_user` anywhere in the spawn tree.

### Setup

1. Clone the dataset as a sibling of this checkout (matches the
   `paths.benchmark_data_root("livesqlbench-base-lite-sqlite")` default):

   ```bash
   cd <parent of bird-agents>
   git clone https://huggingface.co/datasets/birdsql/livesqlbench-base-lite-sqlite
   ```

2. **Materialise the LFS-managed sqlite templates.** The repo ships
   each `<db>_template.sqlite` as a 132-byte LFS pointer; the OTF
   cache reads real sqlite files, so this step is mandatory:

   ```bash
   (cd livesqlbench-base-lite-sqlite && git lfs pull)
   ```

3. From the **bird-agents** checkout (where `scripts/` lives — not
   the dataset dir), materialise the stable per-DB `<db>.sqlite`
   (copies template → stable; idempotent; refuses LFS pointers loudly):

   ```bash
   cd bird-agents
   uv run python scripts/prepare_livesqlbench.py
   ```

4. Obtain the **gated gold sidecar** (e.g.
   `livesqlbench_gt_kg_testcases_*.jsonl`) — the public dataset ships
   `sol_sql` / `external_knowledge` / `test_cases` empty; the gated
   file carries them keyed by `instance_id`. See the LiveSQLBench repo
   for access. The harness merges the gold in via `--gold-file`; it is
   never committed here.

### Oracle validation (first thing to land/verify)

```bash
uv run bird-interact --dataset livesqlbench-base-lite-sqlite --mode oracle \
  --gold-file <gated.jsonl> \
  --data ../livesqlbench-base-lite-sqlite/livesqlbench_data_sqlite.jsonl \
  --db-path ../livesqlbench-base-lite-sqlite/
```

Submits the gold SQL directly and scores it — no LLM. Expected
`phase1_passed=True` for every kept task.

### One-shot run (both SLayer flavors)

```bash
uv run bird-interact --dataset livesqlbench-base-lite-sqlite --mode one-shot --query-mode slayer \
  --framework claude_sdk --slayer-setup on-the-fly \
  --gold-file <gated.jsonl> \
  --data ../livesqlbench-base-lite-sqlite/livesqlbench_data_sqlite.jsonl \
  --db-path ../livesqlbench-base-lite-sqlite/
```

`--mode one-shot` is gated to `--dataset livesqlbench-base-lite-sqlite` (and conversely
`--dataset livesqlbench-base-lite-sqlite` accepts `--mode {one-shot, oracle}` only).
`--slayer-setup on-the-fly` is required for one-shot.

### Per-benchmark artifact roots

OTF artifacts are kept **disjoint per benchmark** so DBs that share
names across benchmarks (e.g. `alien` appears in both mini-interact and
livesqlbench) never collide:

| Benchmark                      | OTF cache                                          | OTF reference                                        |
|--------------------------------|----------------------------------------------------|------------------------------------------------------|
| `mini-interact`                | `slayer_otf_cache/mini-interact/<db>/`             | `slayer_models_otf/mini-interact/<db>/`              |
| `livesqlbench-base-lite-sqlite`| `slayer_otf_cache/livesqlbench-base-lite-sqlite/<db>/` | `slayer_models_otf/livesqlbench-base-lite-sqlite/<db>/` |

All roots live under the main checkout (worktree-safe via
`paths.slayer_otf_cache_root(benchmark=...)` /
`paths.slayer_models_otf_root(benchmark=...)`). A single env var per
helper overrides the parent for all benchmarks: `BIRD_OTF_CACHE_ROOT`
and `BIRD_SLAYER_MODELS_OTF_ROOT`. `--otf-rebuild` purges only the
run's benchmark roots — never the other benchmark's.

## Query Modes

- **`raw`**: Agent gets direct DB tools (`execute_sql`, `get_schema`, `get_column_meaning`, etc.) and writes SQL.
- **`slayer`**: Agent uses SLayer MCP tools (`models_summary`, `inspect_model`, `query`). Doesn't know about SQL/SQLite.

## Agent Frameworks

| Framework | Status | Install extra |
|-----------|--------|--------------|
| Claude Agent SDK (`claude_sdk`) | Active — dispatches the right internal agent based on benchmark and mode | `claude-sdk` |
| PydanticAI | Planned | — |
| smolagents | Planned | — |
| Agno | Planned | — |
| mcp-agent | Planned | — |

## License

MIT
