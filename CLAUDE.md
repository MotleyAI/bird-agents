# Project notes for bird-interact-agents

## Always run project tools via `uv run` (never conda/system)

The shell auto-activates conda env `motley3.11`, whose
`~/miniconda3/envs/motley3.11/bin` shadows the project `.venv`. Bare `ray`,
`python`, `pytest`, etc. resolve to the WRONG environment (e.g. ray 2.41 vs
the venv's 2.55). Always invoke as `uv run <cmd>` (or `.venv/bin/<cmd>`)
from the worktree root. Mixing ray versions against one cluster corrupted
the autoscaler SSH keypair (`private key contents do not match public`) —
a self-inflicted bug from a bare `ray exec`.

## Running the test suite (the exact command)

Run the FULL non-integration suite from the worktree root with:

```bash
env -u SSH_AUTH_SOCK uv run --extra all --extra dev --extra pydantic-ai pytest
```

- The trailing `--extra pydantic-ai` pulls `httpx[socks]`. The `all` extra lists
  the bare `pydantic-ai` package but NOT the `pydantic-ai` *extra*, so it misses
  `httpx[socks]`. In a shell with a SOCKS proxy env var (this sandbox sets
  `ALL_PROXY=socks5h://...`), ~6 model/agent-factory tests then die with
  `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`.
  Adding `--extra pydantic-ai` (or unsetting the proxy vars) fixes it.

- `pytest` + `pytest-asyncio` live in the **`dev`** extra; the agent/SLayer
  tests import `slayer`, `pydantic_ai`, etc. from the **`all`** extra. You need
  BOTH. Bare `uv run pytest` (no extras) installs neither — it then falls back
  to the conda `pytest`, which can't see the project and dies with
  `ModuleNotFoundError: No module named 'bird_interact_agents'` (the wrong-env
  trap from the note above). `.venv/bin/pytest` does not exist until the `dev`
  extra is synced.
- The default run excludes `integration` tests (pyproject `addopts =
  -m 'not integration'` — they spawn real `slayer ingest` subprocesses and
  dominate wall-time). Run integration only with `... pytest -m integration`,
  or everything with `... pytest -o addopts=""`.
- First run does a one-time `uv sync --extra all --extra dev` (pulls the
  upstream `mini-interact-agent` from GitHub via `tool.uv.sources`); subsequent
  runs are cached. Equivalent to the README's `uv sync --extra all --extra dev`.

## Launching a job / smoke test (the exact command)

The CLI is the `bird-interact-cloud` console script (entry point
`bird_interact_agents.cloud.cli:main`). A cheap 2-task smoke, haiku for BOTH
the agent and the user-sim, detached so the cluster survives for inspection:

```bash
env -u SSH_AUTH_SOCK uv run bird-interact-cloud submit \
  --framework pydantic_ai --query-mode raw \
  --agent-model anthropic/claude-haiku-4-5-20251001 \
  --user-sim-model anthropic/claude-haiku-4-5-20251001 \
  --mode c-interact \
  --instance-ids alien_1,alien_2 \
  --workers 1 --actors-per-worker 2 \
  --worker-type e2-standard-4 --max-runtime-hours 1 \
  --detach
```

- `env -u SSH_AUTH_SOCK` + `uv run` are mandatory (see the two notes above).
- Instance IDs are the `instance_id` field of
  `<mini-interact root>/mini_interact.jsonl` (e.g. `alien_1`, `alien_2`);
  `uv run python -c "from bird_interact_agents import paths; print(paths.mini_interact_root())"`
  prints the root.
- `--detach` and `--allow-dirty` are mutually exclusive; commit first so the
  worktree is clean and you don't need `--allow-dirty`. Only `src/`,
  `slayer_models/`, `audited_gold/`, `uv.lock`, `pyproject.toml`,
  `Dockerfile.cloud` count toward the dirty check — a staged `CLAUDE.md` or
  untracked dotfiles don't block it.
- The rendered cluster YAML lands at `~/.bird-interact-cloud/<run-id>.yaml`;
  pass that to the `ray exec` inspection commands below. Use
  `uv run bird-interact-cloud list` / `fetch <run-id>` / `kill <run-id>` /
  `resubmit <run-id>` for the rest of the lifecycle.

## Debugging the cloud runner: pull live state, don't guess

When a `bird-interact-cloud` run fails on GCE, **do not iterate by editing
code and re-running full submits** — each real-cloud round is 10–60 min and
costs money. Several hours were burned this way before the actual root
causes (worker SA cannot `actAs` itself; actor OOM on the head node) were
found in seconds by reading the live cluster state. Instead:

1. **Run detached** (`--detach`) so the cluster survives for inspection
   instead of being torn down on failure.
2. **Pull the authoritative live state** from the head:
   - `uv run ray exec <yaml> "ray status"` — node states + `Recent
     failures` + `Pending Demands` (shows actors stuck PENDING, workers
     `LaunchFailed`, etc.).
   - `uv run ray exec <yaml> "cat /tmp/ray/session_latest/logs/monitor.log"`
     — the autoscaler's per-attempt VM-launch errors (this is where
     `SERVICE_ACCOUNT_ACCESS_DENIED`, quota, image-not-found surface).
   - `uv run ray exec <yaml> "ray job logs --address http://localhost:8265 <job_id>"`
     — the in-cluster `ray_app.py` driver traceback (PicklingError, OOM,
     import errors).
   The laptop-side `ray up` stdout does NOT show worker-launch failures —
   those happen on the head's autoscaler. You must read the head.
3. **Pressure-test assumptions with Codex (`mcp__codex__codex`) and
   Perplexity (`mcp__perplexity__search`)** before committing to a fix —
   but verify their claims against the actual installed Ray source under
   `.venv/.../ray/autoscaler/` (Perplexity confidently gave a wrong answer
   about `min_workers` in autoscaler v2; the source said otherwise).
4. Only after the live logs name the failure, change code/config + add a
   regression test, then re-run.

## Don't edit source files while a submit is in flight

`image.code_hash` refuses a dirty worktree by default, so editing any
`src/`/`Dockerfile.cloud`/`pyproject.toml` while a `submit` is mid-flight
makes the NEXT submit fail with `DirtyWorktreeError`. Commit first, then
launch, and don't touch tracked image-input files until it returns.

## Cloud runner ops quick-reference

- Deployment identifiers are in `src/bird_interact_agents/cloud/config.py`
  (all `BIRD_INTERACT_CLOUD_*` env-overridable; default project
  `motley-team-475011`).
- One-time GCP provisioning is an 8-command checklist in PR #24's
  description; `prereqs.check` validates every item (including the
  easy-to-miss worker-SA `actAs`-self grant) and prints the exact
  remediation `gcloud` command.
- The ssh-agent on this machine has been seen wedged ("agent refused
  operation"); run cloud submits with `env -u SSH_AUTH_SOCK` so ssh reads
  the pinned key file directly.
