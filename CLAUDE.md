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

## Per-benchmark OTF artifact roots (DEV-1462)

LiveSQLBench and mini-interact share DB names (e.g. `alien`). The OTF
cache + reference path helpers therefore take a `benchmark` kwarg so
the artifacts stay disjoint:

* `paths.slayer_otf_cache_root()` → `<main_checkout>/slayer_otf_cache/`
  (mini-interact, legacy / default — no caller breakage).
* `paths.slayer_otf_cache_root(benchmark="livesqlbench")` →
  `<main_checkout>/slayer_otf_cache_livesqlbench/`.
* Same shape for `slayer_models_otf_root`. Env overrides are parallel
  (`BIRD_OTF_CACHE_ROOT_LIVESQLBENCH`,
  `BIRD_SLAYER_MODELS_OTF_ROOT_LIVESQLBENCH`).

When adding a new benchmark, extend `paths._KNOWN_BENCHMARKS` and pass
`benchmark=...` from each call site that resolves an OTF root (today:
the recursive + otf_encode agents' `_resolve_otf_task_storage_dir` and
`run._maybe_force_wipe_otf`). Tests in `tests/test_paths.py`,
`tests/test_otf_rebuild_per_benchmark.py`, and
`tests/test_cloud_paths_unchanged.py` pin the contract — the last one
asserts the dev-1470 cloud upload-back/merge keeps using the LEGACY
(no-kwarg) roots until cloud-side LiveSQLBench is filed.

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
- **Postgres benchmarks (livesqlbench-base-lite, livesqlbench-base-full,
  livesqlbench-large, bird-interact-lite-exp, bird-interact-full)**: at worker
  startup `_ensure_postgres_loaded` starts ONE PostgreSQL instance per VM
  (shared by all actors via `/tmp/pg_init.lock` + the per-data_dir marker)
  and loads all databases from `pg_dumps/`. This adds 1–3 GB of RAM overhead
  per worker, not per actor. e2-standard-4 (16 GB) is too small with
  concurrent Opus actors + postgres; use e2-standard-8 (32 GB) for runs of
  10–20 tasks until per-actor density on each tier is characterised.

## Running a postgres benchmark LOCALLY (sudo-free) — `scripts/run_local_postgres.py`

The DB-load logic (`_ensure_postgres_loaded`) lives ONLY in the cloud worker,
so a local `bird-interact --dataset livesqlbench-large` (or any
`db_backend == "postgres"` benchmark) has nowhere to connect — the system
postgres on `:5432` isn't provisioned with the `bird_interact` role or the
task DBs, and you have no sudo to it. You do NOT need the system instance:
`scripts/setup_local_postgres.py` runs an ENTIRELY SEPARATE PostgreSQL cluster
owned by the current user under `<main_checkout>/.local_pg/` (gitignored) on a
spare port (default `5544`), so no sudo and no collision with `:5432`.

The one-shot orchestrator ties the three local-only gaps together — load auth
env, provision+load postgres, sync annotations from GCS — then execs
`bird-interact` with `--data`/`--db-path` auto-derived:

```bash
env -u SSH_AUTH_SOCK uv run python scripts/run_local_postgres.py \
  --benchmark livesqlbench-large \
  --instance-ids solar_panel_6,fake_account_15,...  \
  -- \
  --framework claude_sdk --query-mode raw --mode one-shot \
  --agent-model anthropic/claude-opus-4-8 --subscription-auth \
  --use-audited-gold-sql --concurrency 1
```

Everything after `--` is passed straight to `bird-interact`. Notes / gotchas
(each cost real debugging the first time):

- **Auth env**: the orchestrator loads a dotenv (`--env-file`, default
  `~/Dropbox/PyCharmProjects/shared/.env.ubuntu`, or `BIRD_ENV_FILE`) for
  `CLAUDE_CODE_OAUTH_TOKEN` (`--subscription-auth`) / `ANTHROPIC_API_KEY`
  (`--no-subscription-auth`). Missing file → skipped, assumes the shell already
  has the creds. The token is the SAME one the cloud `submit`/`annotate` reads.
- **Dumps need a `root` role**: every livesqlbench dump does
  `ALTER TABLE … OWNER TO root`, so the provisioner creates BOTH `bird_interact`
  (the LOGIN role the harness connects as) and `root` (the owner), as SUPERUSER
  under loopback `trust` auth (transient single-tenant dev cluster → no password
  hardening).
- **`--require-annotation` (default ON)**: annotations are the authoritative
  grading source and live in GCS after a cloud `annotate` run; they are NOT
  local unless pulled. `fetch_local_annotations.py` syncs the stable
  `*.task.json` for the selected ids into `annotations/<benchmark>/<db>/`.
- **`--reasoning-effort` is a NO-OP for the base `claude_sdk`/`pydantic_ai`
  frameworks** — it only maps to `ClaudeAgentOptions.effort` for the
  `claude_sdk_otf*` agents. A raw run ignores it (opus still does default
  thinking).
- **Benign log line**: `could not translate host name "livesqlbench_postgresql"`
  is EXPECTED and harmless. The authoritative result-match grading path
  (`eval/upstream_ex_base.py`) opens a `BIRD_PG_*` conn and passes it explicitly;
  a secondary upstream pool (`bird_interact_env`'s own `db_config`, default
  docker hostname, not env-aware) logs that error and is unused for the verdict.
  The CLOUD hits the identical error for the same reason — do NOT "fix" it by
  patching the upstream config; that would diverge from cloud behaviour.
- **large-v1 dumps ship PER-TABLE**, not as one combined file. The Google
  Drive zip (`download_pg_dumps.py`) has `postgre_table_dumps_large/<db>_template/
  <table>.sql` (one per table, each with its own FK constraints) plus PARTIAL
  `*_full.sql` aggregates (e.g. `disaster_relief` full = 29/49 tables). The old
  "pick the largest file" heuristic grabbed a single big table → a 1-table DB.
  `download_pg_dumps.combine_per_table_dumps` now concatenates the single-table
  files with `CREATE TYPE` hoisted first and `FOREIGN KEY`s deferred last, so
  the combined dump loads in one clean pass. If you see a livesqlbench-large DB
  with implausibly few tables, re-stage from the zip with
  `download_pg_dumps.py --force` (plain re-runs skip existing staged files).
- Manage the cluster directly: `scripts/setup_local_postgres.py --benchmark X`
  (idempotent provision — all benchmark DBs by default, or `--instance-ids` for
  a subset), `--stop` to stop it, `--recreate` to wipe+rebuild. Pure helpers are
  pinned by `tests/scripts/test_local_postgres_helpers.py`.

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

## Cheap end-to-end smoke for the OTF encoder + on-the-fly (DEV-1470)

DEV-1609: `claude_sdk_otf_encode` is now the DEFAULT OTF encode framework
everywhere (local + cloud) and the recipe below uses it. It is build-only —
it constructs the claude_sdk reference encoder (DEV-1589) and merges the
canonical `slayer_models_otf/<benchmark>/<db>` reference back, behaviourally
identical to `scripts/build_otf_references.py`. The legacy
`pydantic_ai_otf_encode` is retained only for back-compat / `resubmit`.

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
#    claude_sdk_otf_encode is a claude_sdk* framework, so an Anthropic
#    agent model REQUIRES an explicit --subscription-auth / --no-subscription-auth
#    choice (DEV-1602). Use --subscription-auth for Anthropic; registry
#    open-weight models (zai/, moonshot/) take --no-subscription-auth.
env -u SSH_AUTH_SOCK uv run bird-interact-cloud submit \
  --framework claude_sdk_otf_encode --query-mode slayer \
  --subscription-auth \
  --agent-model anthropic/claude-haiku-4-5-20251001 \
  --user-sim-model anthropic/claude-haiku-4-5-20251001 \
  --mode a-interact \
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
#    Anthropic claude_sdk* framework → explicit --subscription-auth required.
env -u SSH_AUTH_SOCK uv run bird-interact-cloud submit \
  --framework claude_sdk_otf_encode --query-mode slayer \
  --subscription-auth \
  --agent-model anthropic/claude-opus-4-7 \
  --user-sim-model anthropic/claude-sonnet-4-6 \
  --mode a-interact \
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

## Cascade stats per `(benchmark, agent_model, mode)` — use the script

For "how is Opus doing on mini-interact in raw vs slayer right now?", the
canonical tool is `scripts/cascade_for_combo.py`. It walks
`runs/<benchmark>/<db>/<iid>/`, joins each annotation against its cloud
manifest (`results/<benchmark>/cloud/<run_id>/manifest.json`, with a GCS
fallback that caches locally), filters to the requested `--mode` (raw or
slayer) and `--agent-model` substring (`opus` matches
`anthropic/claude-opus-4-7`), then picks the **latest non-`eval_failed`**
run per task — chronologically-latest can be a stale regrade whose grader
infrastructure choked, which would override a genuine earlier verdict.

It prints BOTH the cumulative N1..N9 table (with `Δ vs prev` per tier; N1
is the headline original-gold pass rate) AND the mutually-exclusive
partition L1..L11 (each task → exactly one tier). `--json` emits the
machine-readable payload.

```bash
env -u SSH_AUTH_SOCK uv run python scripts/cascade_for_combo.py \
  --benchmark mini-interact --mode slayer --agent-model opus
env -u SSH_AUTH_SOCK uv run python scripts/cascade_for_combo.py \
  --benchmark mini-interact --mode raw    --agent-model opus
```

Reach for this script, not ad-hoc `rglob` + cascade computation. The
"pick latest non-eval_failed" rule is non-obvious and the existing
`aggregate_cascading_latest` in `cascading_report.py` does NOT apply
either of these two filters. Contract is pinned by
`tests/scripts/test_cascade_for_combo.py`.

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

## Codex MCP sandbox fails with `bwrap: setting up uid map: Permission denied`

Symptom: every `mcp__codex__codex` shell command dies before executing with
`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` and/or
`bwrap: setting up uid map: Permission denied`; `unshare -Ur true` fails the
same way. Codex's sandbox is bubblewrap, which needs an **unprivileged user
namespace**.

Root cause (a regression from an OS upgrade — "it used to work"): Ubuntu
23.10/24.04 ship `kernel.apparmor_restrict_unprivileged_userns=1`, and there
is no AppArmor profile granting `/usr/bin/bwrap` the `userns` capability, so
bwrap runs unconfined and is blocked from creating the namespace.

Do NOT "fix" this by calling Codex with `sandbox: danger-full-access`
(disables Codex's sandbox entirely) and do NOT globally set
`kernel.apparmor_restrict_unprivileged_userns=0` (drops the hardening for
everything). The least-invasive fix is a per-binary AppArmor profile that
whitelists bwrap (needs sudo — run on the host):

```bash
sudo tee /etc/apparmor.d/bwrap >/dev/null <<'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
  include if exists <local/bwrap>
}
EOF
sudo apparmor_parser -r /etc/apparmor.d/bwrap
# verify (prints ok):
bwrap --dev-bind / / --unshare-net echo ok
```

After this, `mcp__codex__codex` works with the normal `read-only` /
`workspace-write` sandbox again — no `danger-full-access` needed. Diagnosis:
check `cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns` (1 =
restricted) and `unshare -Ur true` (must succeed once the profile is loaded).
Ref: https://www.jdhodges.com/blog/codex-sandbox-ubuntu-24-04-fix/

## Every `claude_sdk*` agent MUST route through `hermetic_claude_sdk_session` (DEV-1579)

The Claude Agent SDK launches a bundled `claude` Node CLI. By default it
reads `~/.claude.json` and loads every claude.ai-synced MCP connector
(Linear, Notion, Figma, …) into the model's per-turn tool schema — so a
**local** `bird-interact` run sees a different tool surface than **cloud**
(which has no `~/.claude.json`): 11 servers / 151 tools vs 2 / 20, ~4×
cost, different verdicts. `setting_sources=[]` + `allowed_tools=` do NOT
stop this; only an empty `CLAUDE_CONFIG_DIR` does.

The single choke point is
`agents/claude_sdk/sdk_env.py::hermetic_claude_sdk_session(...)`. Every SDK
call site (the 8 OTF agents + the annotator) uses it, and **any new
`claude_sdk*` agent MUST too** — do NOT hand-build `ClaudeAgentOptions.env`
or `ClaudeSDKClient(...)` directly. The helper owns, in one place:

1. **Auth** (`assert_api_key_auth`): two paths, chosen by an EXPLICIT operator
   signal (`BIRD_INTERACT_SUBSCRIPTION_AUTH`), never inferred from which
   credential happens to be present. Default (API-key) path: authenticate via
   `ANTHROPIC_API_KEY` (Anthropic) or the registry provider token. Subscription
   path (DEV-1602, signal set): authenticate via a Claude.ai OAuth token
   (`CLAUDE_CODE_OAUTH_TOKEN`, `sk-ant-oat01-` prefix) — a missing/malformed
   token hard-fails with NO silent fall-back to the API key. The registry
   exemption is checked FIRST, so the signal (Anthropic-only) is inert for
   registry models. The cloud driver sets the signal on the actor env when
   `--subscription-auth` is chosen; the local `run.py --subscription-auth`
   (default off) sets it in-process. (DEV-1602 supersedes DEV-1582, which would
   have retired the `--subscription-auth` machinery; DEV-1583 still renames the
   registry Bearer var.)
2. **Hermetic `CLAUDE_CONFIG_DIR`**: a fresh empty temp dir whose
   `.claude.json` declares no `mcpServers` (so only the explicit
   `mcp_servers` load) and pre-accepts onboarding/trust so the
   non-interactive CLI never blocks. No `.credentials.json` is copied —
   auth always flows from the env (`ANTHROPIC_API_KEY` or, on the subscription
   path, `CLAUDE_CODE_OAUTH_TOKEN`), never from a copied credential file.
3. **Provider session env + thinking** for registry open-weight models
   (`provider_aware=True`, the default). Pass `provider_aware=False` for an
   Anthropic-only agent (the annotator) so a stray registry model can't
   silently gain registry behaviour.
4. **MCP parity assertion** (`assert_hermetic_mcp_servers`): after
   `__aenter__`, before the first tool call, asserts the loaded MCP servers
   contain NO server beyond the explicit `mcp_servers` keys
   (`loaded - expected == ∅`, a contamination check — not exact equality and
   not readiness). NB: `get_mcp_status` reports stdio/external servers but
   NOT in-process `create_sdk_mcp_server` servers (e.g. `bird-interact-tools`),
   so a loaded set that is a strict subset of expected is normal — only an
   EXTRA server means the host's `~/.claude.json` leaked in. A leak raises
   `HermeticEnvError`, failing the task loudly. (Verified live: the
   `zai/glm-5.2` local smoke loaded only `slayer`, `extra=[]` — isolation
   confirmed.)
5. **Config-dir cleanup** on exit (success or any failure).

Usage: pass `self.model`, the explicit `mcp_servers` dict, and a
`build_options(opt_kwargs)` closure that returns the agent's own
`ClaudeAgentOptions(**opt_kwargs, …its unique tools/hooks/agents…)` —
`opt_kwargs` carries the policy-owned `env` (+ `thinking`). Preserve a
DEV-1561 `otf_timer`-around-`__aenter__` by passing
`enter_cm_factory=lambda: otf_timer(...)`.

Because the registry layer is a no-op for Anthropic models, this is inert
for existing cloud Anthropic runs (same servers, same auth). The contract
is pinned by `tests/test_sdk_subprocess_hermetic.py` (helper units),
`tests/test_dev1579_v0_provider_support.py` (v0 registry support +
graceful skip), the per-agent `CLAUDE_CONFIG_DIR` assertions in
`tests/test_dev1561_disable_cli_telemetry.py` /
`tests/test_dev1555_stage2_sdk_env.py`, and the real-CLI isolation smoke
`tests/test_dev1579_hermetic_integration.py` (`@pytest.mark.integration`,
needs `ANTHROPIC_API_KEY`).
