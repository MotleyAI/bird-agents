# Project notes for bird-interact-agents

## Worktree-safe by default — never depend on a path inside `repo_root`

Every gitignored input or output (data dirs, encoded models, results,
caches, references) lives in the MAIN checkout, NEVER in the worktree.
The worktree only carries tracked source code; any reachable-from-Python
data path is resolved via `bird_interact_agents.paths.*_root()` helpers
(`audited_gold_root`, `slayer_models_otf_root`, `slayer_otf_cache_root`,
`results_root`, …), which anchor at the main checkout via git's common dir.

When adding ANY new code that reads or writes gitignored data, follow this
rule:

1. Add (or reuse) a `paths.*_root()` helper that returns the main-checkout
   path. Never `repo_root / "<some-dir>"` and never `Path(__file__).parents[…]`.
2. If the path needs to enter a docker build, wire it via BuildKit
   `--build-context <name>=<paths.x_root()>` (mirroring how `mini-interact`
   and `audited_gold` are passed), and `COPY --from=<name>` in
   `Dockerfile.cloud`. Do NOT `COPY <dir>/` from the build context root —
   that breaks the moment someone runs `bird-interact-cloud submit` from a
   worktree.
3. If you change a `build_and_push`-side signature, hash the content via
   the explicit root arg in `image.data_hash`, not via the worktree path.

The smoke for this contract is `tests/cloud/test_image.py::test_*audited_gold*`
(worktree-safety regression tests). Symlinking gitignored data into a
worktree is a workaround, not a fix — the next created worktree won't
have the symlink and the build will break for the same reason.

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

## Waiting on a cloud run to finish — use `driver.wait_until_done`, never bash

Polling for a run's terminal state from a bash loop is a maintenance trap.
The output of `bird-interact-cloud list` is positional (`run_id  combo
state  done/total`), so an awk-column off-by-one (`$4` vs `$3`) makes the
loop silently never terminate — once burned in DEV-1470 because the loop
compared `done|error` against `0/1` instead of against `live`.

The canonical primitive is already in the driver and was used by
`driver.submit` (non-detached path) from day one:

```bash
env -u SSH_AUTH_SOCK uv run python -c "
from bird_interact_agents.cloud import driver, gcs
client = gcs.default_gcs_client()
manifest = gcs.read_manifest('<RUN-ID>', client=client)
result = driver.wait_until_done('<RUN-ID>', manifest, poll_interval_s=30.0)
print(f'TERMINAL: {result.terminal_state}  {result.hint!r}')
"
```

It handles stall detection, no-progress deadline, headless cluster, and
returns a single `WaitResult` with the terminal state. No bash, no column
juggling. Use this (or a future `bird-interact-cloud wait <run-id>`
subcommand wrapping it) instead of an ad-hoc shell poll.

## Cheap end-to-end smoke for `pydantic_ai_otf_encode + on-the-fly` (DEV-1470)

Validates the full upload-back + merge round-trip — the deterministic
cache stays local, the LLM stage runs in the cloud, the encoded reference
comes back into the warm cache. Pick a DB with no committed reference (or
nuke a local one); the cache for that DB MUST be present locally because
`_check_slayer_setup_present` enforces `_cache_fp.txt` as REQUIRED:

```bash
# 1. Nuke local artefacts for the test DB.
rm -rf /path/to/main-checkout/slayer_models_otf/<db>
rm -rf /path/to/main-checkout/slayer_otf_cache/<db>

# 2. Build the deterministic cache locally (no LLMs, fast).
env -u SSH_AUTH_SOCK uv run python -c "
import asyncio
from bird_interact_agents.slayer_otf.cache import ensure_db_cache
from bird_interact_agents import paths
asyncio.run(ensure_db_cache('<db>',
    cache_root=paths.slayer_otf_cache_root(),
    mini_interact_root=paths.mini_interact_root(), force=False))
"

# 3. Submit otf_encode (cloud does the LLM stage).
env -u SSH_AUTH_SOCK uv run bird-interact-cloud submit \
  --framework pydantic_ai_otf_encode --query-mode slayer \
  --agent-model anthropic/claude-haiku-4-5-20251001 \
  --user-sim-model anthropic/claude-haiku-4-5-20251001 \
  --mode a-interact --slayer-setup on-the-fly \
  --instance-ids <db>_1 \
  --workers 1 --actors-per-worker 1 \
  --worker-type e2-standard-4 --max-runtime-hours 2 --detach

# 4. Wait via `driver.wait_until_done` (see note above), then:
env -u SSH_AUTH_SOCK uv run bird-interact-cloud fetch <run-id>
```

The fetch prints `merged slayer_models_otf/<db>: N files updated, 0 skipped`
and `<main-checkout>/slayer_models_otf/<db>/_reference_fp.txt` reappears
with the cloud-built fingerprint and content. Per-run merge audit lives at
`<results>/cloud/<run-id>/merge_report.json`.

## DEV-1478 smoke: OTF encoder vs hand-audited reference (`households`)

After landing the DEV-1478 prompt + validator fixes (style guide, peer-KB
dedup block, defensive defer, literal-existence pre-save check), this
smoke validates that the seven failure modes from the audit no longer
silently slip through. Manual, **Opus encoder / Sonnet user-sim** (haiku
was too weak to repro the original failures or distinguish the fix).

```bash
# 1. Nuke local artefacts for households so the reference rebuilds from
# scratch under the new prompts + validator.
rm -rf /path/to/main-checkout/slayer_models_otf/households
rm -rf /path/to/main-checkout/slayer_otf_cache/households

# 2. Build the deterministic cache locally (no LLMs, fast).
env -u SSH_AUTH_SOCK uv run python -c "
import asyncio
from bird_interact_agents.slayer_otf.cache import ensure_db_cache
from bird_interact_agents import paths
asyncio.run(ensure_db_cache('households',
    cache_root=paths.slayer_otf_cache_root(),
    mini_interact_root=paths.mini_interact_root(), force=False))
"

# 3. Submit otf_encode — Opus encoder, Sonnet user-sim, households only.
env -u SSH_AUTH_SOCK uv run bird-interact-cloud submit \
  --framework pydantic_ai_otf_encode --query-mode slayer \
  --agent-model anthropic/claude-opus-4-7 \
  --user-sim-model anthropic/claude-sonnet-4-6 \
  --mode a-interact --slayer-setup on-the-fly \
  --instance-ids households_1 \
  --workers 1 --actors-per-worker 1 \
  --worker-type e2-standard-4 --max-runtime-hours 2 --detach

# 4. Wait via driver.wait_until_done (see note above), then:
env -u SSH_AUTH_SOCK uv run bird-interact-cloud fetch <run-id>
```

Eyeball assertions on the fetched `slayer_models_otf/households/` and
`_setup_results.json` (failures here are signals, not test-suite gates —
the user-sim is stochastic):

- **KB 13 / 20 / 29 chain**: at least KB 13 is `encoded` OR `deferred`
  with a citation in `clarifying_questions`/`notes` (not silently absent).
  If KB 13 is `encoded`, KB 20 should follow.
- **KB 21 / 42 (broken-predicate regression target)**: either `deferred`
  with notes naming the failing literal (e.g. `'High Income'`) and the
  validator's known sampled list, OR `encoded` with a CASE/mapping whose
  literals DO occur in the column. Never silently-broken.
- **KB 9 vs KB 37**: exactly one of them encodes a Column tagged
  `meta.kb_id` on `service_types.socsupport`. The other defers with
  notes "duplicate of KB <other>".
- **Sampled-value caveats** (post-S1-S5 cleanup, commit `3a05b06`):
  KB 1, 2 (and KB 9 if encoded) MUST carry a 1-2 sentence
  sampled-value caveat ABOVE the `[kb=N]` block when actual values
  diverge from KB-described enums. Hand-audited exemplars used as
  three-shot in `prompts.py::_STYLE_GUIDE`: `Tenure_Type` mixed case,
  `Income_Bracket` R$-ranges vs ordinal labels, `Dwelling_Class`
  mixed case. Deferred value-illustration KBs (6, 8) should
  enumerate sampled values in `clarifying_questions`.
- **Peer-KB block enrichment**: every kb-tagged peer entity in the
  rendered prompt block MUST carry labeled markers `entity_ref=<db>.
  <model>[.<leaf>]` AND `reachable_from_host: host(hops), …`. Use
  `_collect_peer_kb_entries` (factories.py) to dump for inspection.

## Full eval baseline — households (15 instances)

After landing S1/S2/S4/S5/S6 (commit `3a05b06`), the post-cleanup OTF
reference beats the hand-audited reference on the 15 households
instances (Opus encoder, Sonnet user-sim, patience 500, audited gold):

| Run                                          | P1 / 15 | Wallclock | Run ID |
| --                                           | --      | --        | --     |
| PRIOR hand-audited (May 22, 53-task batch)   | 8       | ~93m sum  | `20260522-1041_par_opus-4-7-agent_sonnet-sim_combined53_p500` |
| NEW pat=3, no audit (bad defaults)           | 3       | ~80m sum  | `20260527t1443-pydanti-slayer-70eead` |
| NEW pat=500, audited gold (pre-S1-S5)        | 7       | ~87m sum  | `20260527t1601-pydanti-slayer-dab346` |
| **NEW post-S1-S5 cleanup**                   | **9**   | ~92m sum  | **`20260527t1922-pydanti-slayer-45372d`** |

Movement vs hand-audited: +`households_3, _7` (improvements);
−`households_17` (lone regression). Movement vs pre-S1-S5 baseline:
+`households_2, _7`, zero new regressions.

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
