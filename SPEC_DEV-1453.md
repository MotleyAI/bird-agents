# DEV-1453 — Cloud benchmark runner (Spec)

Owner: Egor Kraev  ·  Branch: `egor/dev-1453-cloud-benchmark-runner-run-bird-interact-on-gce-via-ray`  ·  Linear: [DEV-1453](https://linear.app/motley-ai/issue/DEV-1453/cloud-benchmark-runner-run-bird-interact-on-gce-via-ray-driven-from)

## 1. Goal

Run a single `(framework × query-mode × model)` BIRD-Interact mini evaluation over an arbitrary subset of instance IDs on a small Ray cluster on Google Compute Engine, **driven and torn down entirely from the local laptop**. Results survive crash of the driver, the cluster, or any individual task.

## 2. Scope

In:
- One submission = one `(framework, query-mode, model)` over a caller-specified list of `instance_id`s (or the whole 300-task set).
- Both `--query-mode raw` and `--query-mode slayer`.
- Local CLI: `submit`, `fetch`, `kill`, `list`, `build`, `resubmit`.
- Auto-rebuild of the worker Docker image when the local code or data hash differs from the latest pushed tag.

Out (deferred, not v1):
- Matrix sweeps in a single submit (callers loop submit instead).
- Cluster reuse across submits (always fresh cluster).
- Autoscaling, preemptible/spot VMs, GCP Secret Manager, multi-region, IAP-only SSH, custom VPCs.
- Automatic per-task retries from inside the cluster (record-and-move-on, matches current local behaviour). Missing tasks are recovered by an explicit `resubmit` from the laptop.

## 3. Architecture

```
┌──── LOCAL LAPTOP ─────────────────────────────────────────────────┐
│   bird-interact-cloud {submit | fetch | kill | list | build |    │
│                        resubmit}                                  │
│       │                                                           │
│       ├─ src/bird_interact_agents/cloud/                          │
│       │     ├── cli.py        (argparse, sub-commands)            │
│       │     ├── driver.py     (submit/fetch/kill/list/resubmit)   │
│       │     ├── image.py      (split hash, build, push)           │
│       │     ├── cluster.py    (ray up / ray down wrappers)        │
│       │     ├── gcs.py        (per-task writer, prefix reader)    │
│       │     ├── collation.py  (rows/* → results.db + eval.json)   │
│       │     ├── ray_app.py    (in-cluster: actor + ActorPool)     │
│       │     ├── prereqs.py    (env / gcloud / bucket validation)  │
│       │     └── cluster.yaml.j2 (Ray GCP node-provider template)  │
│       │                                                           │
│       ├─ Reads API keys from laptop env / .env at submit time     │
│       └─ Writes to results/cloud/<run-id>/ on attached completion │
│                                                                   │
└── gcloud / ray CLIs / docker ── over HTTPS ───────────────────────┘
                  │
┌──── GCP project = motley-team-475011, zone = us-central1-a ───────────┐
│  Artifact Registry: us-central1-docker.pkg.dev/.../runner:<tag>   │
│      tag = <data-hash>-<code-hash> (data layers stable across     │
│      code changes; see §6.2)                                      │
│                                                                   │
│  Ray cluster: 1 head (e2-small) + 4 workers (e2-standard-4)       │
│      Image: runner:<tag> via cluster.yaml's `docker.image`        │
│      Each worker hosts 4 Ray actors (16 in-flight tasks total)    │
│      VM startup script: `sudo shutdown -h +<max-runtime-minutes>` │
│                                                                   │
│  GCS bucket: gs://motley-team-birdbench/runs/<run-id>/         │
│      ├── manifest.json                                            │
│      ├── rows/<instance_id>.json    (one object per completed)    │
│      ├── logs/<instance_id>.log                                   │
│      └── (no aggregated results.jsonl in GCS; collation is local) │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## 4. Resolved design decisions (from interview)

| # | Decision | Implication |
|---|----------|-------------|
| 1 | **Results sink = per-task objects** at `runs/<run-id>/rows/<instance_id>.json`; no GCS-side append. | No concurrent-writer hazards; collation runs locally on `fetch`. |
| 2 | **Per-task entry point**: extract `async def run_one_task(...) -> TaskResultRow` out of `run.run_evaluation`'s loop body. Local CLI is unchanged behaviourally (still does asyncio.gather over this). | One source of truth for per-task semantics; actor calls it directly. |
| 3 | **API keys** reach the actors via a **per-actor** `runtime_env={'env_vars': {...}}`, delivered out-of-band: the submitter `ray rsync_up`s a temp JSON secrets file into the head container (secret in file content, never on a command line) and the job itself is submitted with an EMPTY runtime_env. The in-cluster `ray_app` reads the file (`--secrets-file`) and applies the keys per-actor, then deletes it. Transport is the Ray **Jobs API** (`ray job submit`), NOT the deprecated `ray submit`. **Amended**: keys are deliberately kept OUT of the *job* runtime_env because `ray job list`/the dashboard echo it back, leaking secrets into logs. | Lost when cluster torn down; correct API; no key in `ray job list`/dashboard. |
| 4 | **Worker GCS auth** = VM default service account with `roles/storage.objectUser` scoped to `gs://motley-team-birdbench` for read+write of objects. Container picks creds up from metadata server. Bucket *creation* and IAM binding require **submitter** permissions; see §6.8 for the split. | No key files. Bucket auto-created (if absent) by submit, which validates the *submitter*'s `roles/storage.admin` on the project first. |
| 5 | **Slayer mode — per-task isolated storage** (amended from interview). Each task copies `/data/slayer_dbs/<db>.sqlite` to a per-task tmp dir and boots an ephemeral SLayer server pointed at that copy; teardown after the task. Reason: HARD-8 deletions and model edits mutate the SQLite, so sharing one server across tasks would corrupt isolation. The actor reuses only immutable state (loaded task list, audited overlay). | Matches the per-task semantics of current local code. Cost: one server-boot per task (≈O(seconds)); acceptable. e2-standard-4 with 4 actors handles four concurrent boots. |
| 6 | **Local artifacts** post-fetch: `results.db` + `eval.json` + `rows/` + `logs/` + `manifest.json` + `run.log`. `eval.json` byte-compatible with `bird-interact run`. | Downstream tooling (`test_compare_results.py`, etc.) unchanged. |
| 7 | **Pass-through CLI flags**: `--use-audited-gold-sql`, `--max-depth`, `--prompt-cache / --no-prompt-cache`, `--mode {interactive,oracle}` in addition to the flags in the Linear plan. | All knobs from `bird-interact run` available cloud-side. |
| 8 | **Missing-task recovery**: `bird-interact-cloud resubmit <run-id>` lists `rows/`, diffs against `manifest.json:instance_ids`, runs only the missing IDs into the *same* `run-id` prefix on a fresh cluster pinned to the original image tag. | No in-cluster retry logic; deterministic from the laptop. |
| 9 | **Docker image** is **layered**: slow layers (mini-interact data, slayer_models, pre-ingested slayer_dbs) sit below the fast-changing code layer. Tag carries split hash `<data-hash>-<code-hash>`. | Code-only edits push a small delta layer; data changes push the whole image. |
| 10 | **Ray test mode**: `ray.init(local_mode=True, ignore_reinit_error=True)` in tests; real Ray in prod. If the pinned Ray version has removed `local_mode`, fall back to `ray.init(num_cpus=1)` and assert single-process behaviour. Actors carry `max_restarts=3` (see §6.5) so we must also test the `RayActorError` recovery path. | One code path, two init recipes; no Ray-stubbing brittleness. |
| 11 | **Attached-mode Ctrl-C**: first Ctrl-C ⇒ graceful `ray down --yes` blocking ≤ 60 s; second Ctrl-C ⇒ skip teardown, warn, exit — VM **self-delete** timer (§7) is the safety net (NOT `shutdown -h`, which only stops). | Matches muscle memory; no surprise lingering clusters; no surprise stopped-but-still-billed disks. |
| 12 | **Per-task logs**: full captured stdout + stderr per task via OS-level FD redirection (`os.dup2` of fds 1 and 2 to a per-task log file before each task, restored after), always uploaded to `logs/<instance_id>.log`. This captures SLayer/MCP subprocess output too. Actors process exactly one task at a time (enforced by single-method ActorPool). | Subprocess-aware capture; not just Python's `sys.stdout`. |
| 13 | **No cluster reuse in v1**: every `submit` = `ray up` + work + `ray down`. | Predictable, isolated; matches the Linear plan. |
| 14 | **Prereq validation = detect-and-fail-fast** at submit time, split across submitter and worker SA (see §6.8). | gcloud auth, project, region, docker, bucket existence + lifecycle, Artifact Registry repo, submitter's IAM rights, worker SA's IAM rights, required API keys for the requested `--agent-model` and `--user-sim-model`. |
| 15 | **GCS clients**: one SDK only — `google-cloud-storage`. `gsutil` is NOT used or required. `fetch` uses concurrent object downloads via the SDK. | One auth surface, one dependency, no second tool. |
| 16 | **Submit refuses dirty worktree by default**. `--allow-dirty` overrides and hashes the working tree as-is. Untracked files under image-input paths count as dirty unless gitignored. | Reproducibility: the pushed image's hash matches a real git commit unless the user opts out. |
| 17 | **Cluster recovery from any machine**: render-inputs (workers, worker_type, zone, image_uri, SA email, max-runtime-hours) live in `manifest.json`. `~/.bird-interact-cloud/<run-id>.yaml` is a perf cache only; `kill`/`resubmit` re-render from manifest if the cache is absent. | Matches "GCS is the source of truth" from the Linear plan. |
| 18 | **Heartbeat**: in-cluster driver writes `runs/<run-id>/status.json` every ~30 s with `{ray_job_id, last_heartbeat_ts, rows_done, rows_total, terminal_state}`. Local `wait_until_done` treats heartbeat older than 5 minutes (while head is alive) as a stalled job → fetch+exit with a hint to `resubmit`. | Catches "head alive but Ray job died" — closes a polling-forever gap. |

## 5. Container layout

```
/app/bird-interact-agents/      ← repo (checked out at build time)
/app/.venv/                     ← uv sync --extra all --extra dev
/data/mini-interact/            ← baked dataset (~234 MB)
/data/slayer_models/            ← committed YAMLs (~4.4 MB)
/data/slayer_dbs/<db>.sqlite    ← pre-ingested per-DB SLayer SQLite, built
                                 by `slayer ingest` during `docker build`
ENV BIRD_DB_PATH=/data/mini-interact
ENV BIRD_DATA_PATH=/data/mini-interact/mini_interact.jsonl
ENV BIRD_RESULTS_ROOT=/tmp/results  (per-actor scratch; flushed to GCS)
ENV BIRD_INTERACT_AGENTS_CLOUD=1   (signals to in-process code that
                                    `paths.main_checkout_root()` should
                                    return /app/bird-interact-agents)
```

The mini-interact-agent upstream harness arrives via `uv sync --extra all` from the pinned MotleyAI fork SHA (already in `pyproject.toml [tool.uv.sources]`). No separate clone.

## 6. Module-by-module spec

### 6.0 `cloud/config.py` — deployment identifiers

Every GCP-side identifier the runner uses is centralised here with an
env-var override. Defaults target the `motley-team-475011` deployment we
ship with; switching deployments needs zero code changes — just export
the relevant `BIRD_INTERACT_CLOUD_*` env var before running the CLI.

| Constant | Default | Override env var |
|---|---|---|
| `PROJECT` | `motley-team-475011` | `BIRD_INTERACT_CLOUD_PROJECT` |
| `REGION` | `us-central1` | `BIRD_INTERACT_CLOUD_REGION` |
| `ZONE` | `<REGION>-a` | `BIRD_INTERACT_CLOUD_ZONE` |
| `BUCKET_NAME` | `motley-team-birdbench` | `BIRD_INTERACT_CLOUD_BUCKET` |
| `AR_REPO` | `bird-interact-runner` | `BIRD_INTERACT_CLOUD_AR_REPO` |
| `IMAGE_NAME` | `runner` | `BIRD_INTERACT_CLOUD_IMAGE_NAME` |
| `WORKER_SA` | `<AR_REPO>@<PROJECT>.iam.gserviceaccount.com` | `BIRD_INTERACT_CLOUD_WORKER_SA` |

`image_uri_prefix()` re-reads env at call time so a test fixture can
override without re-importing the module.

The constants are also exposed by `cloud.prereqs` and `cloud.cluster` as
backwards-compatible aliases so old call sites and tests keep working.

`build_manifest` writes `project` into `render_inputs` so a fetched
manifest from any machine renders to the original cluster's project,
not the local environment's default.

### 6.1 `cloud/cli.py` — entry point `bird-interact-cloud`

Sub-commands and flags:

```
bird-interact-cloud submit
    --framework {pydantic_ai, claude_sdk, smolagents, agno, mcp_agent, raw_sql, ...}
    --query-mode {raw, slayer}
    --agent-model <litellm-model-string>
    (--instance-ids id1,id2,id3 | --instance-ids-file path)
    [--mode {a-interact, c-interact, oracle}] (new, pass-through; matches local CLI)
    [--user-sim-model <litellm-model-string>] (default anthropic/claude-haiku-4-5-20251001)
    [--patience N]                            (default 3)
    [--strict | --no-strict]                  (default --no-strict)
    [--use-audited-gold-sql]                  (new, pass-through)
    [--max-depth N]                           (new, pass-through, default 3)
    [--prompt-cache | --no-prompt-cache]      (new, pass-through, default --prompt-cache)
    [--workers 4] [--actors-per-worker 4] [--worker-type e2-standard-4]
    [--max-runtime-hours 8]
    [--run-id <override>]                     (default: ts-based slug)
    [--detach]                                (default: attached, blocking)
    [--allow-dirty]                           (override: hash working tree as-is even if `git status` is dirty under image-input paths)

bird-interact-cloud fetch  <run-id>
bird-interact-cloud kill   <run-id>
bird-interact-cloud list
bird-interact-cloud build  [--force]
bird-interact-cloud resubmit <run-id> [--workers N] [--actors-per-worker N]
```

`pyproject.toml`:

```
[project.scripts]
bird-interact-cloud = "bird_interact_agents.cloud.cli:main"

[project.optional-dependencies]
cloud = ["ray[default]>=2.30", "google-cloud-storage>=2.18", "jinja2>=3.1"]
all   = [...existing..., "ray[default]>=2.30", "google-cloud-storage>=2.18", "jinja2>=3.1"]
```

### 6.2 `cloud/image.py`

```python
def data_hash(repo_root: Path, mini_interact_root: Path) -> str: ...
def code_hash(repo_root: Path, allow_dirty: bool) -> str: ...
def image_tag(repo_root: Path, mini_interact_root: Path, allow_dirty: bool) -> str:
    return f"{data_hash(repo_root, mini_interact_root)[:12]}-{code_hash(repo_root, allow_dirty)[:12]}"
```

**Content-based hashing only — no mtimes.** Both hashes are SHA-256 over the bytes of the constituent files in sorted-path order. Identical content → identical hash, regardless of checkout history.

- `data_hash` covers, by content:
  - every regular file under `mini_interact_root` (sibling dataset; NOT under repo git — we walk the filesystem).
  - every regular file under `slayer_models/` in the repo (`git ls-files` filter).
  - every regular file under `audited_gold/` in the repo.
  - the `# DATA-LAYERS` section of `Dockerfile.cloud` (a sentinel-delimited block — see Dockerfile layout below).
  - the version-pin of the `slayer ingest` CLI (`slayer --version` baked at build time).
- `code_hash` covers, by content:
  - `uv.lock`
  - `pyproject.toml`
  - every tracked file under `src/`
  - the `# CODE-LAYERS` section of `Dockerfile.cloud`.
  - **Dirty-worktree handling**: by default, if `git status --porcelain` shows any uncommitted change touching the above paths, raise `DirtyWorktreeError`. With `--allow-dirty`, hash the working tree as-is and tag the image with a `-dirty` suffix; `submit` refuses `--detach --allow-dirty` (no reproducibility from a transient state if the laptop won't be around).
- `build_and_push(tag, repo_root, mini_interact_root, force: bool) -> str`:
  1. `docker manifest inspect us-central1-docker.pkg.dev/.../runner:<tag>` — skip if exists and not `force`.
  2. `docker build -t runner:<tag> -f Dockerfile.cloud .`
  3. `docker push`.
  4. Returns full image URI.
- **Dockerfile.cloud layer order — slow → fast** so code edits don't invalidate data ingestion. Docker layer cache: later layers depend on earlier layers, so put rarely-changing inputs first.
  ```
  # Base
  FROM python:3.11-slim
  RUN apt-get install -y ... && pip install uv
  # DATA-LAYERS  ────────────────────────────────────────────
  COPY mini-interact/ /data/mini-interact/
  COPY slayer_models/ /data/slayer_models/
  COPY audited_gold/  /app/bird-interact-agents/audited_gold/
  RUN  slayer ingest --root /data/slayer_models --out /data/slayer_dbs
  # CODE-LAYERS  ────────────────────────────────────────────
  COPY pyproject.toml uv.lock /app/bird-interact-agents/
  RUN  uv sync --extra all --extra dev
  COPY src/ /app/bird-interact-agents/src/
  ENV  BIRD_DB_PATH=/data/mini-interact BIRD_DATA_PATH=/data/mini-interact/mini_interact.jsonl BIRD_RESULTS_ROOT=/tmp/results
  ```
  The `# DATA-LAYERS` and `# CODE-LAYERS` sentinels are how `data_hash` / `code_hash` split the file.
- Bucket-side cache: on push, also write a manifest blob `gs://motley-team-birdbench/images/<tag>.json` recording the constituent hashes + the `git` commit (or `dirty:<wt-hash>`) for traceability.

### 6.3 `cloud/cluster.py` + `cluster.yaml.j2`

`cluster.yaml.j2` is the Ray GCP node-provider config rendered with:

- `image: us-central1-docker.pkg.dev/.../runner:<tag>`
- `project_id: motley-team-475011`
- `region: us-central1`, `availability_zone: us-central1-a`
- `service_account_email: <worker-sa>@motley-team-475011.iam.gserviceaccount.com` (the worker SA from §6.8)
- `head_node`: `e2-small`, no GPU
- `worker_nodes`: count = `<workers>`, `e2-standard-4`
- `labels`: `bird-interact-run-id=<run-id>` (≤63 chars, lower-kebab — our run-id slug already complies), `bird-interact-image-tag=<tag>`
- `initialization_commands`: schedule a **self-delete** (NOT `shutdown -h`, which only stops and leaves disks billing):
  ```bash
  ( sleep $((<max-runtime-minutes> * 60)) && \
    gcloud --quiet compute instances delete "$(hostname)" \
      --zone="us-central1-a" --delete-disks=all ) &
  ```
  This deletes the instance and its boot disk after the timer. The VM SA needs `roles/compute.instanceAdmin.v1` *restricted to its own instances via condition* (see §6.8) or the broader `compute.instances.delete` on the project. We use a delete-self IAM condition for tightness.
- Defence-in-depth: also set `idle_timeout_minutes: 30` in the Ray config so Ray's autoscaler tears workers down on idle.
- `docker.run_options`: `--env BIRD_DB_PATH=/data/mini-interact`, etc.

`cluster.py`:

```python
def render(run_id, image_uri, workers, worker_type, max_runtime_hours, zone, worker_sa, ...) -> Path
def up(yaml_path: Path) -> None              # ray up --yes <yaml>
def down(yaml_path: Path) -> None             # ray down --yes -f <yaml>
def submit_job(head_address: str, args: list[str], env_vars: dict[str, str],
               yaml_path: Path | None = None) -> str
    # Ray Jobs API, run INSIDE the head container via `ray exec <yaml>`:
    #   ray job submit --address=http://localhost:8265 \
    #                  --runtime-env-json='{}'   # NO secrets here \
    #                  --no-wait \
    #                  -- python /app/.../ray_app.py <args> --secrets-file <path>
    # Secrets delivery (API keys): NOT via the job runtime_env — `ray job
    # list`/the dashboard echo it back, leaking keys into logs. Instead
    # `_rsync_secrets` writes env_vars to a temp JSON file and `ray rsync_up`s
    # it into the head container (secret travels in file CONTENT, never on a
    # command line) at SECRETS_REMOTE_PATH. ray_app reads it via
    # --secrets-file and applies the keys as a PER-ACTOR runtime_env, then
    # deletes the file. Returns the ray_job_id. With --no-wait the call
    # returns immediately; the local driver then polls status.json (§6.7).
def head_address(yaml_path: Path) -> str
    # Resolves the head node's external IP via gcloud (using the run-id label)
    # without depending on the local YAML cache, so kill/resubmit work from
    # a YAML-less machine.
def render_from_manifest(manifest: dict) -> Path
    # Rebuilds cluster.yaml from manifest.json's render_inputs block.
def fallback_delete_by_label(run_id: str) -> None
    # gcloud compute instances delete --filter=labels.bird-interact-run-id=<id> --delete-disks=all
```

Rendered yaml is **cached** at `~/.bird-interact-cloud/<run-id>.yaml` purely as a performance hint. The source of truth for `kill`/`resubmit` is `manifest.json:render_inputs` in GCS (§6.7). If the cache is missing, `render_from_manifest` rebuilds it from the GCS manifest.

### 6.4 `cloud/gcs.py`

Single SDK only: `google-cloud-storage`. No `gsutil` shell-outs anywhere.

```python
def manifest_blob(run_id) -> str                # runs/<id>/manifest.json
def status_blob(run_id) -> str                  # runs/<id>/status.json
def row_blob(run_id, instance_id, attempt: int) -> str
    # runs/<id>/rows/<iid>/attempt-<n>.json
def log_blob(run_id, instance_id, attempt: int) -> str
    # runs/<id>/logs/<iid>/attempt-<n>.log
def write_row(run_id, instance_id, attempt: int, row: dict) -> None   # one PUT
def write_log(run_id, instance_id, attempt: int, text: bytes) -> None # one PUT
def list_attempts(run_id) -> dict[str, list[int]]
    # {iid: [1, 2, ...]} from listing rows/<iid>/attempt-*.json
def latest_attempt(run_id, instance_id) -> int | None
def read_row(run_id, instance_id, attempt: int) -> dict
def write_status(run_id, status: dict) -> None
def read_status(run_id) -> dict | None
def read_manifest(run_id) -> dict
def write_manifest(run_id, manifest: dict) -> None
def concurrent_download_prefix(run_id, dest: Path, max_workers: int = 32) -> None
    # SDK-based concurrent blob download — replaces `gsutil -m rsync`
def ensure_bucket(name: str, region: str, worker_sa_email: str) -> None
    # idempotent; sets 30-day lifecycle on runs/ if creating;
    # grants worker_sa roles/storage.objectUser on the bucket
```

**Attempt-aware writes**: row/log paths embed `attempt-<n>`. Each actor knows its own attempt counter (1 on first dispatch, incremented if the in-cluster driver re-dispatches after a `RayActorError`). `submit` writes attempt=1; `resubmit` writes attempt=`latest_attempt(iid) + 1`. Collation rule (see §6.6): for each `iid`, pick the **latest** attempt and treat its row as the canonical result. Error rows from earlier attempts are retained for forensics but excluded from `results.db` / `eval.json`.

No `If-Generation-Match` preconditions: with attempt-keyed paths, double-dispatch within the same attempt would PUT the same object (same key, same content if same task ran twice — idempotent by construction), and across attempts the path differs.

### 6.5 `cloud/ray_app.py` — in-cluster driver

Invoked via the Ray Jobs API (`ray job submit --no-wait -- python ray_app.py --run-id <id> --gcs-bucket ... --attempt <n>`).

```python
@ray.remote(max_restarts=3, max_task_retries=0)
class WorkerActor:
    def __init__(self, framework, query_mode, agent_model, mode, ..., gcs_bucket, run_id, attempt):
        # boot: load tasks file lazily; load the audited-gold overlay once;
        # immutable state only. SLayer state is NOT held across tasks — see
        # run_one for the per-task copy + ephemeral server.
        ...

    async def run_one(self, instance_id: str) -> str:
        # 1. fd-level capture: open logs/<iid>/attempt-<n>.tmp, dup2 over fd 1
        #    and 2 so Python AND any subprocess (SLayer server, MCP procs)
        #    write there. Save original fds; restore in `finally`.
        # 2. In slayer mode: copy /data/slayer_dbs/<db>.sqlite to a per-task
        #    tmp dir; boot an ephemeral SLayer server bound to that copy;
        #    tear down in `finally`.
        # 3. Call bird_interact_agents.run.run_one_task(task_data, ...).
        # 4. Restore fds, upload row + log to GCS *before* returning.
        # 5. On any exception: write an `error`-row, upload the captured log,
        #    return instance_id. Re-raise nothing — record-and-move-on.
        # 6. Each actor processes exactly one task at a time (enforced by
        #    ActorPool's single-method dispatch model + actor-local lock).

def main():
    ray.init(address="auto")  # in-cluster: picks up the existing head
    actors = [WorkerActor.remote(...) for _ in range(num_actors)]
    pool = ray.util.ActorPool(actors)
    pending = list(instance_ids)
    heartbeat = HeartbeatWriter(run_id=run_id, total=len(pending))
    heartbeat.start()  # daemon thread, status.json every 30s
    try:
        # Submit all ids, then drain. Catch RayActorError for `iid`s whose
        # actor died beyond max_restarts → record `actor-lost` error row,
        # carry on; do NOT abort the run.
        for fn, iid in zip(actors, pending):  # initial fill
            pool.submit(lambda a, x: a.run_one.remote(x), iid)
        leftover = pending[len(actors):]
        while pool.has_next():
            try:
                done_iid = pool.get_next_unordered()  # blocks
                heartbeat.tick_done()
                if leftover:
                    pool.submit(lambda a, x: a.run_one.remote(x), leftover.pop(0))
            except ray.exceptions.RayActorError as e:
                # The actor died with no more restarts; write a synthetic
                # row + log marking this iid as `actor-lost` and continue.
                lost_iid = e.actor_handle.<recover_iid_from_actor_state>()
                gcs.write_row(run_id, lost_iid, attempt, {"error": "actor-lost", ...})
    finally:
        heartbeat.stop_and_flush(terminal_state="done")  # status.json final write
```

`HeartbeatWriter` writes `runs/<run-id>/status.json` every 30 s with `{ray_job_id, last_heartbeat_ts, rows_done, rows_total, terminal_state, attempt}`. On clean exit, `terminal_state` becomes one of `done` / `error`. The local driver (§6.7) treats a stale heartbeat (> 5 min) while the head is alive as a stalled job.

### 6.6 `cloud/collation.py`

```python
def collate(run_dir: Path, manifest: dict) -> dict:
    # 1. Discover all attempts via local rows/<iid>/attempt-*.json after
    #    fetch. For each iid, pick the latest attempt as canonical.
    # 2. Open `results.db` next to run_dir; call insert_run_metadata(...) then
    #    insert_task_result(...) per canonical row, reusing existing helpers
    #    from bird_interact_agents.results_db. Note: insert_task_result is
    #    `INSERT OR REPLACE` keyed by (run_id, framework, mode, query_mode,
    #    instance_id) — re-collating after a fresh fetch *intentionally*
    #    overwrites stale rows with the latest attempt.
    # 3. Call the existing aggregator from run.py to produce eval.json.
    # 4. Return the eval.json dict (also written to disk).
```

**Idempotency contract**: re-running `collate` after a `fetch` that pulled new attempts replaces existing rows with the latest attempt — *intentional* replacement, not skip-on-conflict. This is the desired semantics so `resubmit` can correct earlier error rows.

**Non-canonical attempts** (older ones for an iid) remain in `rows/<iid>/attempt-*.json` on disk for forensic inspection; they are NOT written to `results.db`. `eval.json` carries `n_resubmitted` and a list of iids whose latest attempt differs from attempt 1.

### 6.7 `cloud/driver.py`

Orchestrates the lifecycle:

```python
def submit(args: SubmitArgs) -> RunId:
    prereqs.check(args)                 # split: submitter rights + worker SA rights
    tag = image.image_tag(repo_root, mini_interact_root, args.allow_dirty)
    image_uri = image.build_and_push(tag, repo_root, mini_interact_root, force=False)
    run_id = mint_run_id(args)
    manifest = build_manifest(args, image_uri, run_id)
    # manifest carries `render_inputs`: workers, worker_type, zone, image_uri,
    # worker_sa, max_runtime_hours, agent_model, query_mode, framework, mode,
    # instance_ids, attempt=1, git_commit_or_dirty_hash.
    gcs.write_manifest(run_id, manifest)
    yaml_path = cluster.render_from_manifest(manifest)  # also caches at ~/.bird-interact-cloud/<id>.yaml
    install_signal_handlers(run_id, yaml_path)  # idempotent ray down on first SIGINT
    try:
        cluster.up(yaml_path)
        head = cluster.head_address(yaml_path)
        ray_job_id = cluster.submit_job(
            head_address=head,
            args=["--run-id", run_id, "--gcs-bucket", BUCKET, "--attempt", "1", ...],
            env_vars=read_api_keys_from_local_env(args),  # Ray Jobs API runtime_env
        )
        gcs.write_status(run_id, {"ray_job_id": ray_job_id, "terminal_state": None, ...})
        if not args.detach:
            wait_until_done(run_id, manifest)
            fetch(run_id)
    finally:
        if not args.detach:
            cluster.down(yaml_path)
    return run_id

def fetch(run_id): ...
def kill(run_id):
    # 1. Look for ~/.bird-interact-cloud/<id>.yaml; if missing, read manifest
    #    from GCS and re-render via cluster.render_from_manifest.
    # 2. cluster.down(yaml).
    # 3. Fallback: cluster.fallback_delete_by_label(run_id) with --delete-disks=all.
def list_runs(): ...
def resubmit(run_id, **overrides):
    # 1. manifest = gcs.read_manifest(run_id)
    # 2. attempts = gcs.list_attempts(run_id)
    # 3. done = {iid for iid, atts in attempts.items()
    #            if any(not gcs.read_row(run_id, iid, a).get("error") for a in atts)}
    # 4. missing = set(manifest.instance_ids) - done
    # 5. If missing empty → exit 0.
    # 6. Re-render cluster.yaml from manifest (pinned to original image_uri).
    # 7. cluster.up; cluster.submit_job(..., --attempt N+1, --instance-ids missing)
    # 8. wait_until_done; fetch; cluster.down.
```

`wait_until_done` (rewritten):
- Polls `gcs.read_status(run_id)` every ~10 s. Prints `[N/Total]` with ETA.
- Treats the job as **stalled** if `status.terminal_state is None` AND `now - status.last_heartbeat_ts > 300s` AND the head node is still alive. On stall: fetch what's there, exit non-zero with hint to `resubmit`.
- Treats the job as **done** if `status.terminal_state in ("done", "error")` OR all attempts cover all `instance_ids`.
- Treats the job as **headless** if the head node has been deleted (label query) — fetch what's there, exit non-zero with hint to `resubmit`.

### 6.8 `cloud/prereqs.py`

Two principals, two permission sets — validated separately.

**Submitter (local user / ADC)** needs:
- `gcloud` / `docker` / `ray` on PATH (no `gsutil` — we use the SDK).
- `gcloud config get project` == `motley-team-475011`; region == `us-central1`.
- ADC available (`gcloud auth application-default print-access-token`).
- API keys for `--agent-model` and `--user-sim-model` (env vars).
- `roles/storage.admin` on the project (for bucket auto-create + lifecycle + IAM binding) — or, if the bucket already exists, `roles/storage.admin` on the bucket suffices.
- `roles/artifactregistry.writer` on `bird-interact-runner` repo (for `docker push`).
- `roles/iam.serviceAccountUser` on the worker SA (to launch VMs as that SA).
- `roles/compute.instanceAdmin.v1` on the project (for `ray up` to provision VMs and set labels).

**Worker SA** (`bird-interact-runner@motley-team-475011.iam.gserviceaccount.com`) needs, scoped tightly:
- `roles/storage.objectUser` on `gs://motley-team-birdbench` (read+write rows/, logs/, status.json; read manifest).
- `roles/artifactregistry.reader` on `bird-interact-runner` repo (image pull).
- `roles/compute.instanceAdmin.v1` on the project (the head's autoscaler — running as this SA — creates/deletes worker VMs, and the self-delete timer deletes instances + disks).
- **`roles/iam.serviceAccountUser` ON ITSELF** (`--member=serviceAccount:<SA>` on the SA's own IAM policy). The head autoscaler runs *as* the worker SA and launches worker VMs that also run as the worker SA, so the SA must be able to `actAs` itself. Without this, the head comes up but worker launches fail with `SERVICE_ACCOUNT_ACCESS_DENIED` and actors hang PENDING forever (silent — verified the hard way during the smoke). `check_worker_sa_iam` validates this via `_sa_can_act_as_self`.

```python
def check(args) -> None:
    check_python_version()
    check_local_tools()              # gcloud, docker, ray; NOT gsutil
    check_gcloud_config()             # project=motley-team-475011, region=us-central1
    check_adc()                       # application-default credentials present
    check_api_keys(args.agent_model, args.user_sim_model)
    check_submitter_iam()             # storage.admin OR bucket-level, AR writer,
                                      # SA-user on worker SA, compute.instanceAdmin
    # Create the bucket + AR repo BEFORE checking the worker SA's IAM:
    # `check_worker_sa_iam` reads the worker SA's *bucket-level* objectUser
    # binding, which can't exist until the bucket does (chicken-and-egg).
    ensure_bucket_and_artifact_repo()  # gs://motley-team-birdbench (us-central1)
                                       # + AR repo bird-interact-runner
    check_worker_sa_iam(WORKER_SA)    # storage.objectUser on bucket, AR reader,
                                      # compute.instanceAdmin.v1 (+ actAs self)
```

Failures raise `PrereqError` with a `remediation` field carrying the exact `gcloud` command to fix the gap; CLI prints the remediation and exits 2.

### 6.9 Extract `run_one_task` from `run.py`

New function with the per-task body that today lives inline in `run_evaluation`:

```python
async def run_one_task(
    task_data: dict,
    *,
    data_dir: str,
    framework: str,
    query_mode: str,
    mode: str,
    agent_model: str,
    user_sim_model: str,
    patience: int,
    strict: bool,
    use_audited_gold_sql: bool,
    prompt_cache: bool,
    max_depth: int,
    slayer_storage_root: str | None,
) -> TaskResultRow: ...
```

`run_evaluation` becomes a thin wrapper that loads tasks, applies the audited overlay once, opens `results.db`, then `asyncio.gather`s `run_one_task` with a concurrency semaphore. **No behavioural change to local CLI.** This is a refactor that lands as part of this PR because it's prerequisite for `ray_app`.

## 7. Run lifecycle (operational)

### submit (attached, default)

1. `prereqs.check`.
2. `image.image_tag` → `build_and_push` (skip if tag already in registry).
3. Mint run-id `<UTC-ts>-<frame>-<mode>-<6char>`, e.g. `20260521T1422-pydanticai-raw-a1b2c3`.
4. Upload `manifest.json` (full config + image URI + instance_ids).
5. Render & cache `cluster.yaml`.
6. Install SIGINT/SIGTERM/`atexit` handlers (see Ctrl-C semantics below). Single-use guard so teardown only fires once.
7. `cluster.up` → `cluster.submit_job` with `runtime_env.env_vars = {API keys from local env}`.
8. Poll GCS for row count → progress bar.
9. On completion: `fetch` → `cluster.down`.

### submit (detached, `--detach`)

Steps 1–7 — using Ray Jobs API with `--no-wait` so the job genuinely runs in the background — then print `submitted: <run-id>` and exit. No `atexit`. Cluster lifecycle = VM self-delete timer (§6.3); user runs `fetch` or `kill` later. `--detach --allow-dirty` is refused (a dirty image's source isn't reachable from any git commit, so detached reproducibility is broken).

### Ctrl-C semantics

```
first SIGINT  → print "tearing down cluster; Ctrl-C again to detach"
              → run `ray down --yes` with 60 s wall budget
              → exit 130
second SIGINT → print "abandoning teardown; VM self-delete timer will clean up"
              → exit 130 without teardown
```

`atexit` handler runs the same teardown on normal exception exit (idempotent via a thread-safe `_torn_down: bool` flag). The VM self-delete timer (`gcloud compute instances delete`, not `shutdown -h`) guarantees the cluster goes away even if both SIGINTs miss.

### fetch <run-id>

1. `gcs.concurrent_download_prefix(run_id, results/cloud/<run-id>/)` via the SDK — idempotent, replaces stale files with newer generations.
2. `collate(results/cloud/<run-id>/, manifest)` → updates `results.db` (INSERT OR REPLACE on the primary key), writes `eval.json`.
3. Safe to run mid-run, repeatedly.

### kill <run-id>

1. Look up `~/.bird-interact-cloud/<id>.yaml`; if missing, `gcs.read_manifest(run_id)` + `cluster.render_from_manifest` to rebuild it.
2. `cluster.down(yaml)`.
3. Fallback: `cluster.fallback_delete_by_label(run_id)` with `--delete-disks=all`.
4. Bucket data untouched.
5. Idempotent.

### list

```
storage.Client().bucket("motley-team-birdbench").list_blobs(prefix="runs/")
  → for each run-id:
      mf = gcs.read_manifest(run_id)
      n_done = len(gcs.list_attempts(run_id))
      cluster_status = live | done | orphaned    # GCE label check
      print row
```

(SDK-only — `gsutil` is NOT a dependency of `bird-interact-cloud`; see §4#15.)

### resubmit <run-id>

1. Read manifest from GCS → instance_ids, framework, query-mode, model, image_uri, render_inputs.
2. `attempts = gcs.list_attempts(run_id)`; `done = {iid for iid in attempts if non-error row exists at latest attempt}`.
3. `missing = set(manifest.instance_ids) - done`. (Iids with only error rows count as missing — that's the point of resubmit.)
4. If `missing` is empty: exit 0 with "nothing to do".
5. `next_attempt = max(attempts.values(), default=[0])[0] + 1` (cluster-wide bump, not per-iid).
6. Re-render cluster.yaml from `manifest.render_inputs` (pinned to the original `image_uri` so behaviour is reproducible) and submit a fresh Ray Jobs job whose args are `--instance-ids missing --attempt <next_attempt>`. Writes land in `runs/<run-id>/rows/<iid>/attempt-<n>.json`.

## 8. Crash-safety matrix

| Failure | What survives | How |
|---|---|---|
| One task's agent raises | All other tasks complete; failed task gets an `error` row in `rows/<iid>/attempt-<n>.json` plus its log | Try/except in `run_one`, GCS PUT before return |
| Actor process dies mid-task (1st–3rd time) | Ray restarts the actor (`max_restarts=3`); the in-cluster driver catches `RayActorError` for the in-flight task only, writes a synthetic `actor-lost` row, continues with remaining work | `@ray.remote(max_restarts=3)` + explicit `RayActorError` handling in the drain loop |
| Actor dies a 4th time | That actor is permanently lost; remaining actors carry the workload; iids that were on the dead actor when it gave up land as `actor-lost` error rows | as above |
| Worker VM dies | Ray reschedules to remaining actors on healthy VMs; finished iids safe in GCS | Ray's actor scheduler + GCS |
| Head node dies | In-cluster job dies; heartbeat goes stale; local driver detects stall (§6.7) and exits non-zero with `resubmit` hint | status.json heartbeat + head-alive check |
| Local driver Ctrl-C (attached) | Cluster torn down via `ray down`; GCS state intact | SIGINT handler |
| Local driver killed without cleanup (detached, OOM, SIGKILL) | Cluster fully removed (instance + disks) within `--max-runtime-hours` | VM `gcloud compute instances delete` self-delete timer |
| Network split laptop↔GCP | Cluster keeps running until self-delete timer | as above |
| Total laptop loss | All results recoverable from any machine: `kill`/`fetch`/`resubmit` re-render the cluster YAML from `manifest.json` in GCS | GCS = source of truth; no laptop-local state required |

## 9. Output layout

`gs://motley-team-birdbench/runs/<run-id>/`:
- `manifest.json` (config + render_inputs snapshot; source of truth for `kill`/`resubmit`)
- `status.json` (live heartbeat from the in-cluster driver)
- `rows/<instance_id>/attempt-<n>.json` (one per (iid, attempt); schema = `TaskResultRow` dict)
- `logs/<instance_id>/attempt-<n>.log` (captured stdout+stderr via fd-level redirection)

`results/cloud/<run-id>/` (after fetch+collate):
- `manifest.json` (mirror)
- `status.json` (mirror of last fetched state)
- `rows/` (mirror — all attempts preserved on disk for forensics)
- `logs/` (mirror)
- `results.db` (SQLite; identical schema to local `bird-interact run`; only canonical/latest-attempt rows present)
- `eval.json` (aggregated metrics; identical schema to local `bird-interact run`; adds `n_resubmitted` + `resubmitted_ids`)
- `run.log` (laptop-side driver log)

## 10. Test strategy (Step 4 — written before implementation)

Behavioural tests only. No assertions about prompt text, Ray internals, or third-party Docker/Ray library internals beyond what their public API guarantees.

| # | Module | Test | Notes |
|---|---|---|---|
| T1 | `image.py` | `data_hash` is **content-based**: identical bytes → identical hash regardless of mtime, checkout history, or untracked sibling files | tmp dir fixture; touch a file to change mtime, hash unchanged; modify bytes, hash changes |
| T2 | `image.py` | `code_hash` likewise for `src/`, `uv.lock`, `pyproject.toml`, and the `# CODE-LAYERS` block of `Dockerfile.cloud` | content-based |
| T3 | `image.py` | `image_tag` = `<data12>-<code12>`; idempotent | |
| T4 | `image.py` | Dirty worktree: by default `image_tag(allow_dirty=False)` raises `DirtyWorktreeError` if `git status --porcelain` shows changes touching image-input paths. With `allow_dirty=True`, hash succeeds and tag carries a `-dirty` suffix | use `subprocess`-fakery for git |
| T5 | `image.py` | `submit --detach --allow-dirty` is refused at CLI | argparse-level cross-validation |
| T6 | `cluster.py` | `render(...)` produces YAML with the expected image URI, worker count, run-id label, **self-delete** `gcloud compute instances delete` command (NOT `shutdown -h`), worker SA email | YAML round-trip via PyYAML; grep for `instances delete` |
| T7 | `cluster.py` | `render_from_manifest(manifest)` produces a YAML identical to what `render(...)` would for the same render_inputs | round-trip |
| T8 | `cluster.py` | `submit_job` shells out via Ray Jobs API (`ray job submit --no-wait --runtime-env-json '{"env_vars": {...}}'`), NOT `ray submit` | mocked subprocess; assert argv |
| T9 | `gcs.py` | `write_row` + `list_attempts` + `read_row` round-trip via a fake `google.cloud.storage.Client` | |
| T10 | `gcs.py` | Concurrent writes from N actors with distinct `(iid, attempt)` pairs produce N distinct objects, none lost | asyncio.gather of `write_row` |
| T11 | `gcs.py` | `write_row(iid, attempt=2)` adds a new object alongside `attempt=1`; `latest_attempt(iid)` returns 2 | attempt-aware naming |
| T12 | `gcs.py` | `concurrent_download_prefix` uses the SDK (no `gsutil` shell-out anywhere) | mocked subprocess: assert no `gsutil` invocation |
| T13 | `driver.py` | Given mocked `image`, `cluster`, `gcs`: `submit` calls build → push → up → submit (Jobs API) → (poll or detach) in order | `pytest.MonkeyPatch` |
| T14 | `driver.py` | `submit --detach` returns **before** the Ray job completes (the call to `submit_job` is `--no-wait` and the driver process exits without polling) | mock returns a job_id immediately |
| T15 | `driver.py` | `wait_until_done` detects a stalled job: stale `status.last_heartbeat_ts > 300s` while head label still resolves → returns non-zero with `resubmit` hint | inject a fake clock + GCS status |
| T16 | `driver.py` | `atexit` teardown fires exactly once on normal exit, exception, and SIGINT path | use `signal.raise_signal(signal.SIGINT)` |
| T17 | `driver.py` | Second SIGINT skips teardown and exits 130 | as above |
| T18 | `driver.py` | `kill` works with `~/.bird-interact-cloud/<id>.yaml` absent — falls back to `cluster.render_from_manifest(manifest_from_gcs)` | delete cache file in fixture |
| T19 | `driver.py` | `kill`'s fallback `gcloud compute instances delete` is called with `--delete-disks=all` | mocked subprocess; assert flag |
| T20 | `collation.py` | Given fixture `rows/<iid>/attempt-*.json` with multiple attempts per iid, collation picks the **latest** attempt per iid for `results.db`; non-canonical attempts remain on disk untouched | |
| T21 | `collation.py` | Re-running `collate` after a `fetch` that pulled a newer attempt **replaces** the existing row in `results.db` (per `INSERT OR REPLACE`); `eval.json`'s `n_resubmitted` increments | |
| T22 | `collation.py` | `eval.json` matches local `bird-interact run` output for the same canonical rows | reuses existing aggregator |
| T23 | `fetch` | Running `fetch` twice doesn't duplicate rows or corrupt `eval.json` | idempotency |
| T24 | `ray_app.py` | With `ray.init(local_mode=True)`, the ActorPool dispatches N `instance_id`s to K actors and each `iid` is processed exactly once | recording actor |
| T25 | `ray_app.py` | An actor that raises on `instance_id=X` does not corrupt `Y`, `Z`; `X` lands in `rows/X/attempt-<n>.json` with `error` set | deliberate raise |
| T26 | `ray_app.py` | The actor restart path: simulate `RayActorError` for one iid (mocked); the driver writes a synthetic `actor-lost` row for that iid and continues processing remaining iids | injected exception in the drain loop |
| T27 | `ray_app.py` | Slayer mode: **each task** boots a fresh ephemeral SLayer server pointed at a per-task SQLite copy and tears it down; assertion via a recording slayer stub | NOT one-per-actor (this is the §4#5 amendment) |
| T28 | `ray_app.py` | fd-level log capture: spawn a fake subprocess that writes to stdout from the task body; assert the bytes appear in `logs/<iid>/attempt-<n>.log` | tests that `os.dup2` is used, not `redirect_stdout` |
| T29 | `ray_app.py` | `HeartbeatWriter` writes `status.json` at the configured interval; `terminal_state` becomes `done` on clean exit, `error` on raise | thread + injected GCS client + fake clock |
| T30 | `ray_app.py` | Actor honours `BIRD_DB_PATH` / `BIRD_DATA_PATH` / `BIRD_RESULTS_ROOT` from env; `paths.mini_interact_root()` reflects them | monkeypatch env |
| T31 | `prereqs.py` | Each submitter-side and worker-SA-side prereq raises `PrereqError` with a `remediation` field carrying the exact remediation `gcloud` command when violated | unit per check; split table |
| T32 | `resubmit` | Given a manifest with N ids and attempts/ where M < N have non-error canonical rows, `resubmit` submits exactly the missing N−M with `attempt = max_attempt + 1`, pinned to the manifest's `image_uri` | mocked cluster + gcs |
| T33 | `resubmit` | An iid whose only existing attempt is an `error` row is treated as missing and re-run by `resubmit` | guards against the "error row hides resubmit" bug Codex flagged |
| T34 | `run.run_one_task` | Extracted function returns a `TaskResultRow` matching what the inline loop body produced previously (parity) | fixture-based |
| T35 | `cli.py` | All new flags (`--use-audited-gold-sql`, `--max-depth`, `--prompt-cache`, `--mode {a-interact,c-interact,oracle}`, `--allow-dirty`) are parsed and propagated into the manifest | argparse test; mode names match local CLI |

Integration tests (gated, NOT in CI; run manually against `motley-team-475011`):

| # | Probe | Status |
|---|---|---|
| I1 | `--detach` submit returns within ~30 s of `ray up` finishing — clearly before the Ray job finishes — and `gcs.read_status(run_id)` then shows a `ray_job_id` and `terminal_state is None` | Implemented in `tests/integration/test_cloud_smoke.py` |
| I2 | Inside the running worker container, the API key from local env is visible via `os.environ` (proved by inspecting the captured per-task log) | **Placeholder skip** — implement during phase 9 |
| I3 | Inside the running worker container, a GCS write to `runs/<run-id>/rows/probe/attempt-1.json` succeeds using the metadata-server credentials (no key files) | **Placeholder skip** |
| I4 | Image pull on cluster bring-up uses the worker SA's `roles/artifactregistry.reader` (not the user's docker login) | **Placeholder skip** |
| I5 | `kill <run-id>` from a machine without `~/.bird-interact-cloud/<run-id>.yaml` succeeds by re-rendering from `manifest.json` | **Placeholder skip** |
| I6 | The VM self-delete script (`gcloud compute instances delete`) fully removes the VM **and** its boot disk after the safety timer (not just stops it). Verified by listing instances and disks 5 min after timer + buffer | **Placeholder skip** |
| I7 | End-to-end: `--workers 1 --actors-per-worker 1`, tears down cleanly, produces per-task rows + `eval.json` | **✅ Verified manually** on `motley-team-475011` (run `20260524t1049-pydanticai-raw-2373bb`, haiku/haiku, `households_1,households_10`): both tasks ran (4 & 6 agent turns, $0.071 total), crash-safe rows written to GCS, collated to local `results.db`/`eval.json`, cluster + worker + disks torn down clean. Both `wrong_result` — expected model-quality outcome for haiku, not a pipeline failure. |

I2–I7 ship as `pytest.skip(...)` stubs documenting the probe and the implementation hook. They turn into real tests during the corresponding implementation phases (§11.9–§11.11).

## 11. Implementation phases

1. **Tests first** — all unit tests in §10 (T1–T35) land, fail for-the-right-reason, get Codex-reviewed.
2. **Refactor `run.py`** — extract `run_one_task`. Local CLI behaviour unchanged. T34 turns green.
3. **`image.py`** + `Dockerfile.cloud` + local `docker build` smoke. T1–T5 green.
4. **`cluster.py`** + `cluster.yaml.j2`. T6–T8 green.
5. **`gcs.py`** + bucket auto-create + attempt-aware paths. T9–T12 green.
6. **`collation.py`**. T20–T23 green.
7. **`ray_app.py`** + ActorPool + HeartbeatWriter + per-task SLayer + fd capture. T24–T30 green.
8. **`driver.py`** + `cli.py` + `prereqs.py`. T13–T19, T31–T33, T35 green.
9. **Integration probes I1–I6** on motley-team-475011 (without running the full benchmark).
10. **Smoke (I7)** — `--instance-ids households_5 --workers 1 --actors-per-worker 1`.
11. **Full run** — pydantic_ai × raw × 30 tasks end-to-end.

## 12. Files added / modified

Added:
- `Dockerfile.cloud`
- `src/bird_interact_agents/cloud/__init__.py`
- `src/bird_interact_agents/cloud/cli.py`
- `src/bird_interact_agents/cloud/driver.py`
- `src/bird_interact_agents/cloud/image.py`
- `src/bird_interact_agents/cloud/cluster.py`
- `src/bird_interact_agents/cloud/cluster.yaml.j2`
- `src/bird_interact_agents/cloud/gcs.py`
- `src/bird_interact_agents/cloud/collation.py`
- `src/bird_interact_agents/cloud/ray_app.py`
- `src/bird_interact_agents/cloud/prereqs.py`
- `tests/cloud/test_image.py`
- `tests/cloud/test_cluster.py`
- `tests/cloud/test_gcs.py`
- `tests/cloud/test_driver.py`
- `tests/cloud/test_collation.py`
- `tests/cloud/test_ray_app.py`
- `tests/cloud/test_prereqs.py`
- `tests/cloud/test_resubmit.py`
- `tests/cloud/test_cli.py`
- `tests/cloud/test_run_one_task.py`
- `tests/integration/test_cloud_smoke.py` (gated; not in CI)

Modified:
- `pyproject.toml` (cloud extra; new `bird-interact-cloud` script; bump `all`).
- `src/bird_interact_agents/run.py` (extract `run_one_task`; behaviour-preserving).

Not modified:
- `paths.py` (the `BIRD_*` env overrides we need are already in place).
- existing test files (modulo refactor breakage which T21 guards against).

## 13. Open / deferred

- Cluster reuse across submits — explicitly deferred.
- BigQuery streaming sink — explicitly deferred.
- Preemptible/spot VMs — deferred.
- Secret Manager — deferred.
- Per-task retries inside the cluster beyond `max_restarts=3` actor restarts — deferred (we have `resubmit` for everything else).
- Multi-region — deferred.

## 14. Audit trail — Codex review findings folded

The codex MCP server reviewed the v1 draft of this spec (before §10/§11 were renumbered). All 17 findings were folded in:

| Finding | Severity | Folded as |
|---|---|---|
| Ray submit vs Ray Jobs API + runtime_env transport | BLOCKER | §4#3 (amended), §6.3 (`submit_job` uses Jobs API), §6.5, §6.7 |
| `--detach` doesn't actually detach with `ray submit` | BLOCKER | §6.3 (`--no-wait`), §7 (detached steps), T14 |
| Ray actors don't auto-restart | BLOCKER | §6.5 (`max_restarts=3`), §8 (matrix rows), T26 |
| Per-actor SLayer server corrupts isolation under HARD-8 / model edits | BLOCKER | §4#5 (amended to per-task), §6.5 (per-task copy + ephemeral server), T27 |
| Dockerfile layer order inverted | MAJOR | §6.2 (flipped to slow → fast with sentinels) |
| `data_hash` uses mtimes; dataset is sibling, not git-tracked | MAJOR | §6.2 (content-based; explicit sibling walk), T1 |
| `If-Generation-Match: 0` blocks `resubmit` overwrites | MAJOR | §6.4 (attempt-aware paths, no precondition), §6.7 (`resubmit`), T11, T33 |
| stdout/stderr redirect misses subprocess output | MAJOR | §4#12 (amended to fd-level), §6.5 (`os.dup2`), T28 |
| `insert_task_result` is INSERT OR REPLACE | MAJOR | §6.6 (collation contract is replacement, not skip), T21 |
| `--mode` values don't match local CLI | MAJOR | §6.1 (`{a-interact, c-interact, oracle}`), T35 |
| `shutdown -h` stops, doesn't delete | MAJOR | §4#11/§6.3/§7/§8 (`gcloud compute instances delete` self-delete with `--delete-disks=all`), I6 |
| Submitter vs worker SA permissions conflated | MAJOR | §6.8 (split table), T31 |
| `kill`/`resubmit` need cached YAML — breaks recovery-from-anywhere | MAJOR | §4#17, §6.3 (`render_from_manifest`), §6.7 (`kill`/`resubmit` re-render), T18 |
| No heartbeat — polling-forever on dead Ray job | MAJOR | §4#18, §6.5 (`HeartbeatWriter`), §6.7 (`wait_until_done` stall detection), T15, T29 |
| Test coverage gaps for real-semantics gates | MAJOR | I1–I6 added as gated integration probes |
| Dirty worktree unspecified | MINOR | §4#16, §6.2 (`--allow-dirty` + tag suffix; refused with `--detach`), T4, T5 |
| Two GCS clients (`gsutil` + SDK) | MINOR | §4#15 (SDK-only), §6.4, §6.7, §6.8 (no `gsutil` prereq), T12 |
