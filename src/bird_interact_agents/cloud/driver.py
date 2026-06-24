"""Laptop-side driver: submit, fetch, kill, list, resubmit, build."""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import secrets
import signal
import sys
import time
from pathlib import Path

from bird_interact_agents import paths, provider_registry
from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.cloud import benchmark_data, cluster, config, gcs, image, prereqs
from bird_interact_agents.cloud import collation as _collation
from bird_interact_agents.cloud import post_run_merge as _post_run_merge
from bird_interact_agents.eval import cascading_report as _cascading_report
# Imported by NAME (not via the `gcs` module attr) so tests that mock
# `driver.gcs` still get the real pure mapping — only the I/O helpers
# (`gcs.upload_dir_prefix` etc.) need to be mockable.
# Imported by NAME so they survive tests that mock `driver.prereqs`. PrereqError
# must be the real class for the raise in `read_api_keys_from_local_env`, and
# `_required_api_keys` must be the REAL provider→key mapping so submit/resubmit
# tests that mock `driver.prereqs` still exercise the actual key selection
# (CodeRabbit) — otherwise the mock returns an empty MagicMock-iterable and
# key-selection silently no-ops in those tests.
from bird_interact_agents.cloud.prereqs import (
    PrereqError,
    _is_claude_sdk_framework,
    _required_api_keys,
)
# Imported by NAME so `_build_missing_otf_caches` can be exercised with a mock
# (`monkeypatch.setattr(driver, "ensure_db_cache", ...)`) without a real build.
from bird_interact_agents.slayer_otf.cache import ensure_db_cache


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local paths (overridable in tests)
# ---------------------------------------------------------------------------


def yaml_cache_dir() -> Path:
    return Path.home() / ".bird-interact-cloud"


def local_results_root() -> Path:
    return paths.results_root() / "cloud"


def default_gcs_client():
    return gcs.default_gcs_client()


def submitter_repo_root() -> Path:
    """The repo tree whose code we should hash + build + push.

    `paths.main_checkout_root()` returns the *canonical* checkout shared
    across worktrees — fine for results sinks, but wrong for cloud
    builds: when the user invokes `bird-interact-cloud` from a worktree,
    we want to ship the worktree's branch, not the main checkout's HEAD.

    Resolve via `git rev-parse --show-toplevel` against the cwd; fall
    back to `main_checkout_root()` when not in a git tree (e.g. installed
    wheel)."""
    import subprocess as _sp
    try:
        res = _sp.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return Path(res.stdout.strip())
    except (FileNotFoundError, OSError):
        pass
    return paths.main_checkout_root()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PG_FORWARD_VARS = (
    "BIRD_PG_HOST", "BIRD_PG_PORT", "BIRD_PG_USER",
    "BIRD_PG_PASSWORD", "BIRD_PG_STATEMENT_TIMEOUT",
)


def _maybe_forward_pg_vars(result: dict, dataset: str) -> None:
    """Copy BIRD_PG_* env vars into ``result`` for non-postgres benchmarks.

    Postgres benchmarks start a bundled local server on the worker —
    forwarding external BIRD_PG_* vars would override localhost with an
    unreachable external address.
    """
    if _is_postgres_benchmark(dataset):
        return
    for pg_var in _PG_FORWARD_VARS:
        val = os.environ.get(pg_var)
        if val:
            result[pg_var] = val


def _is_postgres_benchmark(dataset: str) -> bool:
    """True when ``dataset`` maps to a benchmark with ``db_backend="postgres"``."""
    if not dataset:
        return False
    try:
        return getattr(get_benchmark(dataset), "db_backend", "sqlite") == "postgres"
    except Exception:  # noqa: BLE001
        return False


# GCP's compute-instance-name regex caps the FULL name at 63 chars
# (`[a-z]([-a-z0-9]{0,61}[a-z0-9])?` → 1+61+1 = 63). Ray composes the name as
# `ray-<run_id>-worker-<uuid8>-compute` (worst case: "worker" 6 > "head" 4;
# "compute" 7 > "tpu" 3), so 4 + len(run_id) + 7 + 9 + 8 = 28 + len(run_id).
# Ray's INTERNAL assertion (`name_label <= 55`, INSTANCE_NAME_MAX_LEN-1-UUID)
# is off-by-one and looser than GCP's real 63 — observed: Ray accepted a
# 50-char label but GCP rejected the resulting 66-char name. So compute the
# slug cap from GCP's true 63 limit, not Ray's 55.
_INSTANCE_NAME_MAX = 63
_NAME_OVERHEAD = 28  # ray- + -worker + -<uuid8> + -compute


def mint_run_id(framework: str, query_mode: str) -> str:
    """Mint a run-id that's also a valid GCE label value AND instance-name
    fragment: lowercase letters / digits / `-` only. The timestamp uses
    `t` (lowercase) as the date/time separator because uppercase `T` is
    forbidden in GCE label values (`The value can only contain lowercase
    letters, numeric characters, underscores and dashes`). The framework slug
    is length-capped so Ray's `ray-<run_id>-worker-<uuid8>-compute` GCE name
    fits GCP's 63-char compute-instance-name limit."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dt%H%M")
    qm = query_mode.lower()
    token = secrets.token_hex(3)
    # run_id format: <ts>-<slug>-<qm>-<token> (3 dashes). Cap the slug so
    # the full GCE instance name fits 63.
    slug_max = max(
        1, _INSTANCE_NAME_MAX - _NAME_OVERHEAD - len(ts) - len(qm) - len(token) - 3
    )
    slug = framework.replace("_", "").lower()[:slug_max]
    return f"{ts}-{slug}-{qm}-{token}"


def read_api_keys_from_local_env(
    agent_model: str, user_sim_model: str, *, query_mode: str = "raw",
    framework: str = "", no_subscription_auth: bool = False,
    dataset: str = "",
) -> dict[str, str]:
    import os

    # claude_sdk* + subscription auth opted-in (no_subscription_auth=False)
    # → OAuth path. Ship the token and rename the user-sim Anthropic key so
    # the SDK cannot see ANTHROPIC_API_KEY and is forced to use the OAuth
    # token.
    # `--subscription-auth` is REQUIRED at the CLI (BooleanOptionalAction
    # with no default); when the operator opts in but the token is absent
    # or malformed, fail loudly here instead of silently falling through
    # to the legacy API-key path (which burns API credits — the failure
    # mode that bit DEV-1535).
    # CodeRabbit (DEV-1535 r2): the predicate is imported BY NAME at
    # module top, rather than dereferenced via the `prereqs` module
    # attribute. Some tests monkeypatch `driver.prereqs` to a generic
    # MagicMock; with the attribute access, `prereqs._is_claude_sdk_framework`
    # becomes a truthy mock and the OAuth path fires for every framework,
    # masking real test failures. The direct-name import is mock-safe.
    # DEV-1602 (registry-first): a registry open-weight agent model authenticates
    # via its provider key, NEVER OAuth — gate the OAuth branch on the agent
    # model not being a registry model so a programmatic/resubmit caller passing
    # no_subscription_auth=False for a registry model falls through to the
    # provider-key branch instead of demanding an OAuth token.
    if (
        _is_claude_sdk_framework(framework)
        and not no_subscription_auth
        and provider_registry.get_provider(agent_model) is None
    ):
        token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        if not token:
            raise PrereqError(
                "--subscription-auth was selected but CLAUDE_CODE_OAUTH_TOKEN "
                "is not set in the submitter's env.",
                remediation=(
                    "source the env file that exports it "
                    "(e.g. `set -a; source .env.ubuntu; set +a`), run "
                    "`claude setup-token`, or pass `--no-subscription-auth` "
                    "to use the ANTHROPIC_API_KEY path."
                ),
            )
        if not token.startswith("sk-ant-oat01-"):
            raise PrereqError(
                "CLAUDE_CODE_OAUTH_TOKEN does not look like a Claude.ai OAuth token "
                "(expected sk-ant-oat01- prefix).",
                remediation="claude setup-token",
            )
        # DEV-1602: ship the explicit subscription-path signal so sdk_env on the
        # worker takes the OAuth path (path chosen by operator intent, never
        # inferred from which credential happens to be present).
        result: dict[str, str] = {
            "CLAUDE_CODE_OAUTH_TOKEN": token,
            "BIRD_INTERACT_SUBSCRIPTION_AUTH": "1",
        }
        # Track the LOCAL env var names for error messages (the worker-side names
        # differ — e.g. ANTHROPIC_API_KEY → BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY).
        missing_local: list[str] = []
        # Rename the Anthropic key for user-sim; LiteLLM reads it via
        # _maybe_inject_anthropic_key in usage.acompletion_tracked.
        if user_sim_model.startswith("anthropic/"):
            val = os.environ.get("ANTHROPIC_API_KEY", "")
            result["BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY"] = val
            if not val:
                missing_local.append("ANTHROPIC_API_KEY")
        # Non-anthropic user-sim keys (OPENAI, CEREBRAS, GEMINI).
        for k in _required_api_keys(user_sim_model):
            if k != "ANTHROPIC_API_KEY":
                result[k] = os.environ.get(k, "")
                if not result[k]:
                    missing_local.append(k)
        # DEV-1468: slayer embeddings always need OPENAI_API_KEY.
        if query_mode == "slayer":
            result["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
            if not result["OPENAI_API_KEY"] and "OPENAI_API_KEY" not in missing_local:
                missing_local.append("OPENAI_API_KEY")
        # Fail fast on missing keys (resubmit has no prereq check).
        if missing_local:
            missing_local_sorted = sorted(missing_local)
            cmds = "\n".join(f"export {k}=<your-key>" for k in missing_local_sorted)
            raise PrereqError(
                f"missing API key env vars for job submission: {missing_local_sorted}",
                remediation=cmds,
            )
        _maybe_forward_pg_vars(result, dataset)
        return result

    # DEV-1555 Stage 2: claude_sdk framework + registry open-weight agent
    # model (e.g. moonshot/kimi-k2.7-code). Ship the provider key and
    # NEVER raw Anthropic credentials — the Claude Agent SDK auto-discovers
    # ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN and would silently route
    # the agent to Anthropic instead of the configured ANTHROPIC_BASE_URL
    # backend. An anthropic user-sim key travels under the renamed
    # BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY, same as the OAuth path.
    _agent_spec = provider_registry.get_provider(agent_model)
    if _is_claude_sdk_framework(framework) and _agent_spec is not None:
        result = {}
        missing_local = []
        result[_agent_spec.auth_env] = os.environ.get(_agent_spec.auth_env, "")
        if not result[_agent_spec.auth_env]:
            missing_local.append(_agent_spec.auth_env)
        if user_sim_model.startswith("anthropic/"):
            val = os.environ.get("ANTHROPIC_API_KEY", "")
            result["BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY"] = val
            if not val:
                missing_local.append("ANTHROPIC_API_KEY")
        else:
            for k in _required_api_keys(user_sim_model):
                if k == "ANTHROPIC_API_KEY":
                    continue
                result[k] = os.environ.get(k, "")
                if not result[k]:
                    missing_local.append(k)
        if query_mode == "slayer":
            result["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
            if not result["OPENAI_API_KEY"] and "OPENAI_API_KEY" not in missing_local:
                missing_local.append("OPENAI_API_KEY")
        # Forward an operator base-url override so workers hit the same
        # endpoint the submitter validated against.
        override = os.environ.get(_agent_spec.base_url_env, "")
        if override:
            result[_agent_spec.base_url_env] = override
        if missing_local:
            missing_local_sorted = sorted(missing_local)
            cmds = "\n".join(f"export {k}=<your-key>" for k in missing_local_sorted)
            raise PrereqError(
                f"missing API key env vars for job submission: {missing_local_sorted}",
                remediation=cmds,
            )
        _maybe_forward_pg_vars(result, dataset)
        return result

    needed: set[str] = set()
    for model in (agent_model, user_sim_model):
        needed.update(_required_api_keys(model))
    # DEV-1468: slayer mode needs OPENAI_API_KEY for channel-3 embeddings,
    # regardless of the agent/user-sim providers.
    if query_mode == "slayer":
        needed.add("OPENAI_API_KEY")
    # Fail fast on a missing required key instead of silently dropping it —
    # `resubmit` does NOT run prereq checks, so an absent key would otherwise
    # surface much later as an opaque per-actor auth failure (CodeRabbit).
    missing_keys = [k for k in sorted(needed) if not os.environ.get(k)]
    if missing_keys:
        cmds = "\n".join(f"export {k}=<your-key>" for k in missing_keys)
        raise PrereqError(
            f"missing API key env vars for job submission: {missing_keys}",
            remediation=cmds,
        )
    result = {k: os.environ[k] for k in needed}
    _maybe_forward_pg_vars(result, dataset)
    return result


def build_manifest(
    args, *, image_uri: str, run_id: str, benchmark_data_prefix: str | None = None,
) -> dict:
    """Build the manifest dict from a SubmitArgs-like object.

    ``benchmark_data_prefix`` is the content-hashed GCS prefix the dataset was
    uploaded to at submit; the actor downloads from it per node."""
    instance_ids = list(args.instance_ids)
    return {
        "run_id": run_id,
        "framework": args.framework,
        "mode": args.mode,
        "dataset": _submit_benchmark(args),
        "benchmark_data_prefix": benchmark_data_prefix,
        "query_mode": args.query_mode,
        "agent_model": args.agent_model,
        "user_sim_model": args.user_sim_model,
        "instance_ids": instance_ids,
        "patience": args.patience,
        "strict": bool(args.strict),
        "use_audited_gold_sql": bool(args.use_audited_gold_sql),
        "max_depth": args.max_depth,
        "prompt_cache": bool(args.prompt_cache),
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        # DEV-1545: snake_case key per existing convention; None when
        # the CLI flag was unset at submit (resubmit / replay logic
        # checks truthiness to decide whether to re-emit the flag).
        "user_sim_prompt_version": getattr(
            args, "user_sim_prompt_version", None
        ),
        "slayer_setup": getattr(args, "slayer_setup", "pre-encoded"),
        # DEV-1586: which pre-encoded reference feeds a pre-encoded run
        # (otf=encoding-agent output, custom=hand-curated; None on-the-fly).
        "pre_encoded_source": getattr(args, "pre_encoded_source", None),
        "slayer_storage_root": getattr(
            args, "slayer_storage_root", "/data/slayer_models"
        ),
        "no_subscription_auth": bool(getattr(args, "no_subscription_auth", False)),
        "render_inputs": {
            "workers": args.workers,
            "actors_per_worker": args.actors_per_worker,
            "worker_type": args.worker_type,
            "zone": cluster.DEFAULT_ZONE,
            "worker_sa": cluster.DEFAULT_WORKER_SA,
            "max_runtime_hours": args.max_runtime_hours,
            "image_uri": image_uri,
            "project": config.PROJECT,
            "region": config.REGION,
        },
    }


# ---------------------------------------------------------------------------
# SLayer setup delivery (DEV-1468)
# ---------------------------------------------------------------------------


def _dbs_for_instances(instance_ids, benchmark: str = "mini-interact") -> list[str]:
    """Map the selected instance_ids to their distinct ``selected_database``
    via the benchmark's tasks file (never string-split the id — DB names contain
    underscores, e.g. ``california_schools``). Returns a sorted, de-duplicated
    db list."""
    wanted = set(instance_ids)
    dbs: set[str] = set()
    with paths.benchmark_data_file(benchmark).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            td = json.loads(line)
            if td.get("instance_id") in wanted:
                db = td.get("selected_database")
                if db:
                    dbs.add(db)
    return sorted(dbs)


def _benchmark_for_dataset(dataset: str | None) -> str:
    """Map a run's ``dataset`` token (canonical name, CLI alias, or marker) to
    its canonical benchmark name for the OTF path roots + container data dir.

    An absent dataset (e.g. a pre-de-bake manifest being resubmitted) falls
    back to ``"mini-interact"``; otherwise the token is resolved through the
    registry so a third benchmark never silently aliases to mini-interact.
    """
    if not dataset:
        return "mini-interact"
    # Legacy: manifests submitted before DEV-1525 used underscore/short forms.
    _LEGACY: dict[str, str] = {
        "mini_interact": "mini-interact",
        "livesqlbench": "livesqlbench-base-lite-sqlite",
    }
    if dataset in _LEGACY:
        return _LEGACY[dataset]
    return get_benchmark(dataset).name


def _submit_benchmark(args) -> str:
    """Benchmark for a submit, derived from ``args.dataset`` (defaults to
    mini-interact when the CLI didn't set one).

    Annotator args use ``args.benchmark`` rather than ``args.dataset``; fall
    back to that when ``dataset`` is absent so gold-file path helpers work for
    both submit and annotate args.
    """
    return _benchmark_for_dataset(
        getattr(args, "dataset", None) or getattr(args, "benchmark", None)
    )


def _slayer_uploads_for(args) -> list[tuple[Path, str, bool]]:
    """Return the local-dir → GCS-artifact-name uploads to ship for this
    combo, each tagged ``required`` (must exist + non-empty) or optional
    (uploaded only when present, as a "skip this encode" seed for the cloud).

    DEV-1470: for ``pydantic_ai_otf_encode + on-the-fly`` the deterministic
    cache (``slayer_otf_cache``) is REQUIRED (it's the input to the LLM-driven
    setup encoder, kept local because it's free to build); the per-DB
    reference (``slayer_models_otf``) is OPTIONAL — if present, uploaded as a
    seed so the cloud skips re-encoding that DB; if absent, the cloud encodes
    lazily on first task.
    """
    fw = args.framework
    # DEV-1555 v0/v1: raw flavours of either version use no SLayer uploads.
    if fw in (
        "claude_sdk_otf_raw", "claude_sdk_otf_ainteract_raw",
        "claude_sdk_otf_raw_v1", "claude_sdk_otf_ainteract_raw_v1",
    ):
        return []
    setup = args.slayer_setup
    benchmark = _submit_benchmark(args)
    if setup == "pre-encoded":
        # DEV-1586: source selects which encoded reference is uploaded.
        # 'otf' = the benchmark-scoped encoding-agent output (marker
        # _reference_fp.txt); 'custom' (default / legacy) = the committed
        # hand-curated slayer_models dir.
        source = getattr(args, "pre_encoded_source", None) or "custom"
        if source == "otf":
            return [
                (paths.slayer_models_otf_root(benchmark=benchmark),
                 "slayer_models_otf", True),
            ]
        return [(submitter_repo_root() / "slayer_models", "slayer_models", True)]
    if fw == "pydantic_ai_otf_encode":
        return [
            (paths.slayer_otf_cache_root(benchmark=benchmark),
             "slayer_otf_cache", True),
            (paths.slayer_models_otf_root(benchmark=benchmark),
             "slayer_models_otf", False),
        ]
    # pydantic_ai_recursive + on-the-fly — cache only, no LLM-encoded reference.
    return [
        (paths.slayer_otf_cache_root(benchmark=benchmark),
         "slayer_otf_cache", True),
    ]


def _check_gold_present(benchmark_name: str) -> None:
    """Fail early if the benchmark requires gated gold but none is found locally.
    Called at submit time so the error surfaces before the image build / GCS upload."""
    from bird_interact_agents.benchmark import get_benchmark as _gb
    b = _gb(benchmark_name)
    if not b.gold_required:
        return
    gold_dir = paths.gated_gold_root(benchmark=benchmark_name)
    jsonls = list(gold_dir.glob("*.jsonl")) if gold_dir.is_dir() else []
    if not jsonls:
        raise FileNotFoundError(
            f"benchmark {benchmark_name!r} requires gated gold but no "
            f"*.jsonl found in {gold_dir}. "
            f"Place the gold sidecar there before submitting."
        )
    if len(jsonls) > 1:
        raise RuntimeError(
            f"benchmark {benchmark_name!r} gold dir has multiple *.jsonl "
            f"files — auto-discovery is ambiguous: "
            f"{[f.name for f in jsonls]}. Remove the stale copies."
        )


def _has_nonempty_embeddings(db_dir: Path) -> bool:
    """True iff ``<db_dir>/embeddings.db`` exists and is non-empty. DEV-1586:
    the submit-side pre-encoded preflight mirror of the runtime guard."""
    emb = db_dir / "embeddings.db"
    return emb.is_file() and emb.stat().st_size > 0


def _artifact_present(root: Path, db: str, artifact: str) -> bool:
    """Presence semantics per artifact: ``slayer_models`` = a NON-EMPTY
    committed dir (no marker); OTF layers = their completeness marker file is
    present."""
    db_dir = root / db
    if artifact == "slayer_models":
        return db_dir.is_dir() and any(db_dir.iterdir())
    marker = "_cache_fp.txt" if artifact == "slayer_otf_cache" else "_reference_fp.txt"
    return (db_dir / marker).is_file()


def _build_missing_otf_caches(
    cache_root: Path, dbs: list[str], benchmark: str = "mini-interact",
) -> None:
    """Build the deterministic OTF ingest cache locally for each DB in `dbs`.

    The cache is free to build (no LLMs) and fully deterministic, so a missing
    ``slayer_otf_cache`` is auto-created here — laptop-side, before upload —
    rather than aborting the submit. The cloud runner never builds REQUIRED
    setup in-cluster (it consumes the uploaded cache), so this is the only
    place it can be produced for the submit to proceed."""
    data_root = paths.benchmark_data_root(benchmark)
    for db in dbs:
        logger.info(
            "cloud slayer: deterministic cache missing for %s — building it "
            "locally (no LLMs) before upload", db,
        )
        asyncio.run(
            ensure_db_cache(
                db, cache_root=cache_root, mini_interact_root=data_root,
                benchmark=get_benchmark(benchmark),
                force=False,
            )
        )


def _check_slayer_setup_present(args) -> list[str]:
    """Validate (BEFORE build/push/cluster) that every REQUIRED artifact is
    present locally for every selected DB. Optional artifacts (e.g. the
    ``slayer_models_otf`` seed under ``otf_encode + on-the-fly``) are not
    required — the cloud will encode missing references on the fly.

    A missing ``slayer_otf_cache`` (the deterministic ingest cache) is NOT a
    hard error: it's free to build locally, so we build it here instead of
    aborting. Other REQUIRED artifacts (e.g. the hand-authored ``slayer_models``
    reference) can't be auto-built and still fail fast.

    Returns the db list to upload."""
    benchmark = _submit_benchmark(args)
    dbs = _dbs_for_instances(args.instance_ids, benchmark)
    uploads = _slayer_uploads_for(args)
    # DEV-1586: the claude_sdk pre-encoded agents start SLayer with
    # ingest_on_startup=False, so a present reference is unusable without a
    # built embeddings.db. Mirror the runtime guard
    # (agents/_pre_encoded._assert_reference_present) at submit so a doomed
    # pre-encoded run fails BEFORE the cluster spins up (Codex review). Gate on
    # `pre_encoded_source` (set ONLY for the claude_sdk consumers) — the legacy
    # pydantic committed-reference path ingests on startup and needs no
    # pre-built embeddings.
    pre_encoded = getattr(args, "pre_encoded_source", None) is not None
    for root, artifact, required in uploads:
        if not required:
            continue
        missing = [db for db in dbs if not _artifact_present(root, db, artifact)]
        if pre_encoded:
            no_emb = [
                db for db in dbs
                if db not in missing and not _has_nonempty_embeddings(root / db)
            ]
            if no_emb:
                raise FileNotFoundError(
                    f"cloud slayer: pre-encoded reference for {no_emb} under "
                    f"{root} has no usable embeddings.db (the pre-encoded "
                    f"agents run with ingest_on_startup=False). Build the "
                    f"embeddings before submitting "
                    f"(scripts/build_otf_references.py for the otf source)."
                )
        if not missing:
            continue
        if artifact == "slayer_otf_cache":
            # Deterministic + LLM-free → build it rather than erroring.
            _build_missing_otf_caches(root, missing, benchmark)
            continue
        marker_hint = {
            "slayer_models": "non-empty <db>/ dir",
            "slayer_otf_cache": "_cache_fp.txt",
            "slayer_models_otf": "_reference_fp.txt",
        }.get(artifact, artifact)
        raise FileNotFoundError(
            f"cloud slayer: required local setup missing for {missing} "
            f"under {root} (combo {args.slayer_setup}/{args.framework}, "
            f"artifact {artifact} expects {marker_hint}); build it "
            f"locally first — the cloud runner never builds REQUIRED "
            f"setup in-cluster."
        )
    return dbs


def _upload_slayer_setup(args, run_id: str, dbs: list[str]) -> None:
    """Upload every artifact's local dir PER selected DB to
    ``runs/<run_id>/slayer_setup/<artifact>/<db>/``. Optional artifacts whose
    local ``<db>/`` is absent are skipped — the cloud will encode the missing
    reference on the fly."""
    uploads = _slayer_uploads_for(args)
    client = default_gcs_client()
    for root, artifact, required in uploads:
        for db in dbs:
            if not required and not _artifact_present(root, db, artifact):
                # Optional seed absent for this db — cloud will encode.
                continue
            prefix = f"runs/{run_id}/slayer_setup/{artifact}/{db}"
            gcs.upload_dir_prefix(root / db, prefix, client=client)


def _instance_ids_sorted_by_db(
    instance_ids, benchmark: str = "mini-interact",
) -> list[str]:
    """DEV-1470: sort iids by ``(selected_database, instance_id)`` so same-db
    tasks are adjacent in the ActorPool dispatch order — a single actor then
    typically does all encoding for a given DB and the cross-actor encode
    race is rare. Unknown iids (absent from the dataset) sort to the end
    grouped by their iid string so they still appear deterministically."""
    wanted = list(instance_ids)
    if not wanted:
        return wanted
    data_file = paths.benchmark_data_file(benchmark)
    # De-bake: the dataset is GCS-delivered, so `resubmit` may run on a machine
    # that has no local copy. DB-grouping is only a dispatch optimization (it
    # makes same-db tasks adjacent so one actor tends to do all of a DB's
    # encoding), never a correctness gate — so a missing local data file falls
    # back to the input order rather than failing the resubmit (Codex).
    if not data_file.is_file():
        logger.info(
            "instance DB-grouping skipped: local data file %s absent "
            "(e.g. resubmit without the local dataset) — preserving input order",
            data_file,
        )
        return wanted
    db_by_iid: dict[str, str] = {}
    with data_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            td = json.loads(line)
            iid = td.get("instance_id")
            if iid in wanted:
                db = td.get("selected_database") or ""
                db_by_iid[iid] = db
    # Stable, deterministic sort by (db, iid). Unknown iids → ("", iid).
    return sorted(wanted, key=lambda iid: (db_by_iid.get(iid, ""), iid))


# ---------------------------------------------------------------------------
# Signal handlers / teardown
# ---------------------------------------------------------------------------


class _Handler:
    def __init__(self, *, run_id: str, yaml_path: Path):
        self.run_id = run_id
        self.yaml_path = yaml_path
        self._torn_down = False
        self._sigint_count = 0

    def teardown(self, *, reason: str) -> None:
        if self._torn_down:
            return
        self._torn_down = True
        try:
            cluster.down(self.yaml_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("teardown ray down failed: %s (reason=%s)", e, reason)
            try:
                cluster.fallback_delete_by_label(self.run_id)
            except Exception:  # noqa: BLE001
                pass

    def _on_sigint(self, _signum, _frame):
        self._sigint_count += 1
        if self._sigint_count == 1:
            sys.stderr.write(
                "tearing down cluster; Ctrl-C again to detach.\n"
            )
            self.teardown(reason="sigint-1")
            raise SystemExit(130)
        sys.stderr.write(
            "abandoning teardown; VM self-delete timer will clean up.\n"
        )
        raise SystemExit(130)


def install_signal_handlers(*, run_id: str, yaml_path: Path) -> _Handler:
    h = _Handler(run_id=run_id, yaml_path=yaml_path)
    signal.signal(signal.SIGINT, h._on_sigint)
    signal.signal(signal.SIGTERM, h._on_sigint)
    return h


# ---------------------------------------------------------------------------
# Instance-ID validation (fail fast before any cloud touch)
# ---------------------------------------------------------------------------


def _validate_instance_ids(instance_ids, benchmark: str) -> None:
    """Raise ValueError if any requested instance_id is not in the local
    benchmark data file.

    Runs before prereqs, image build, or any GCS/cluster call so typos and
    non-existent IDs surface immediately rather than after paying for a cluster
    spin-up that produces only dispatch-failure rows."""
    data_file = paths.benchmark_data_file(benchmark)
    if not data_file.is_file():
        # Data not present locally (e.g. resubmit on a machine without the
        # dataset) — skip the check rather than blocking.
        return
    known: set[str] = set()
    with data_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                td = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = td.get("instance_id")
            if iid:
                known.add(iid)
    requested = set(instance_ids)
    missing = sorted(requested - known)
    if missing:
        raise ValueError(
            f"instance_id(s) not found in {data_file.name} for benchmark "
            f"{benchmark!r}: {missing}. Check spelling or benchmark name."
        )


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class WaitResult:
    def __init__(self, *, terminal_state: str, hint: str = ""):
        self.terminal_state = terminal_state
        self.hint = hint


def submit(args) -> str:
    _validate_instance_ids(args.instance_ids, _submit_benchmark(args))
    prereqs.check(args)
    repo_root = submitter_repo_root()
    # DEV-1468: slayer fail-fast — verify the local setup is present BEFORE
    # building/pushing the image or bringing up a cluster (no in-cloud builds).
    slayer_dbs: list[str] = []
    if args.query_mode == "slayer":
        slayer_dbs = _check_slayer_setup_present(args)
    bird_interact_eval, livesqlbench_eval = image.default_grader_eval_roots()
    tag = image.image_tag(
        repo_root,
        paths.audited_gold_root(),
        allow_dirty=args.allow_dirty,
        annotations_root=paths.annotations_root(),
        bird_interact_evaluation_root=bird_interact_eval,
        livesqlbench_evaluation_root=livesqlbench_eval,
    )
    image_uri = image.build_and_push(
        tag, repo_root,
        audited_gold_root=paths.audited_gold_root(),
        annotations_root=paths.annotations_root(),
        bird_interact_evaluation_root=bird_interact_eval,
        livesqlbench_evaluation_root=livesqlbench_eval,
        force=False,
    )
    # De-bake: upload the benchmark dataset ONCE to its content-hashed GCS
    # prefix (skipped when the hash already exists). The actor downloads it
    # per node — the dataset is no longer baked into the image.
    _check_gold_present(_submit_benchmark(args))
    benchmark_data_prefix = benchmark_data.ensure_uploaded(_submit_benchmark(args))
    run_id = args.run_id or mint_run_id(args.framework, args.query_mode)
    manifest = build_manifest(
        args, image_uri=image_uri, run_id=run_id,
        benchmark_data_prefix=benchmark_data_prefix,
    )
    gcs.write_manifest(run_id, manifest)
    # Upload the slayer setup under the run prefix (per selected DB) so the
    # actor downloads it in-cluster. After the manifest so kill/resubmit work.
    if args.query_mode == "slayer":
        _upload_slayer_setup(args, run_id, slayer_dbs)
    yaml_path = cluster.render_from_manifest(manifest, cache_dir=yaml_cache_dir())
    h = install_signal_handlers(run_id=run_id, yaml_path=yaml_path)
    submit_succeeded = False
    try:
        cluster.up(yaml_path)
        head = cluster.head_address(yaml_path)
        env_vars = read_api_keys_from_local_env(
            args.agent_model, args.user_sim_model, query_mode=args.query_mode,
            framework=args.framework,
            no_subscription_auth=getattr(args, "no_subscription_auth", False),
            dataset=getattr(args, "dataset", ""),
        )
        job_args = _build_job_args(
            args, run_id, attempt=1,
            benchmark_data_prefix=benchmark_data_prefix,
        )
        # Write status BEFORE submit_job so the in-job HeartbeatWriter — which
        # starts asynchronously inside the Ray job — cannot have its first real
        # heartbeat clobbered by our None reset. ray_job_id is unknown at this
        # point; HeartbeatWriter writes the real value when it ticks.
        # last_heartbeat_ts stays None so wait_until_done's stall check skips
        # until the in-job writer takes over (long postgres dataset loads must
        # not trip HEARTBEAT_STALL_SECONDS).
        gcs.write_status(run_id, {
            "ray_job_id": None,
            "last_heartbeat_ts": None,
            "rows_done": 0,
            "rows_total": len(args.instance_ids),
            "terminal_state": None,
            "attempt": 1,
        })
        cluster.submit_job(
            head_address=head, args=job_args, env_vars=env_vars,
            yaml_path=yaml_path,
        )
        submit_succeeded = True
        if not args.detach:
            wait_until_done(run_id, manifest)
            fetch(run_id)
    finally:
        # Detach skips teardown ONLY on successful submit — if we never
        # got the Ray job in flight, the cluster is just burning money
        # without anything running. (CR#4 — without this, `--detach` +
        # `submit_job` failure orphans the cluster.)
        if (not args.detach) or (not submit_succeeded):
            h.teardown(reason="finally")
    return run_id


def _build_job_args(
    args, run_id: str, *, attempt: int, benchmark_data_prefix: str | None = None,
) -> list[str]:
    benchmark = _submit_benchmark(args)
    job_args = [
        "--run-id", run_id,
        "--attempt", str(attempt),
        "--framework", args.framework,
        "--query-mode", args.query_mode,
        "--mode", args.mode,
        "--dataset", benchmark,
        "--agent-model", args.agent_model,
        "--user-sim-model", args.user_sim_model,
        "--patience", str(args.patience),
        "--max-depth", str(args.max_depth),
        "--num-actors", str(args.workers * args.actors_per_worker),
        # DEV-1470: group same-db iids adjacently so a single actor typically
        # does all encoding for a given DB.
        "--instance-ids",
        ",".join(_instance_ids_sorted_by_db(args.instance_ids, benchmark)),
    ]
    if benchmark_data_prefix:
        job_args += ["--benchmark-data-prefix", benchmark_data_prefix]
    if args.strict:
        job_args.append("--strict")
    if args.use_audited_gold_sql:
        job_args.append("--use-audited-gold-sql")
    if args.prompt_cache:
        job_args.append("--prompt-cache")
    else:
        job_args.append("--no-prompt-cache")
    if getattr(args, "reasoning_effort", None):
        job_args += ["--reasoning-effort", args.reasoning_effort]
    # DEV-1545: conditional emission — never emit
    # `--user-sim-prompt-version None`, which the receiving argparse
    # would accept as the string "None" and silently shadow the v2
    # default.
    if getattr(args, "user_sim_prompt_version", None):
        job_args += [
            "--user-sim-prompt-version", args.user_sim_prompt_version,
        ]
    job_args += [
        "--slayer-setup", getattr(args, "slayer_setup", "pre-encoded"),
        "--slayer-storage-root",
        getattr(args, "slayer_storage_root", "/data/slayer_models"),
    ]
    # DEV-1586: forward the pre-encoded source so the in-cluster worker
    # routes to the read-only flavor. Conditional — never emit
    # `--pre-encoded-models None` (argparse would reject the choice).
    if getattr(args, "pre_encoded_source", None):
        job_args += ["--pre-encoded-models", args.pre_encoded_source]
    return job_args


# ---------------------------------------------------------------------------
# Annotator submit
# ---------------------------------------------------------------------------

_ANNOTATOR_RAY_APP_PATH = (
    "/app/bird-interact-agents/src/bird_interact_agents/cloud/ray_app_annotator.py"
)


def build_annotator_manifest(
    args, *, image_uri: str, run_id: str, benchmark_data_prefix: str | None = None,
) -> dict:
    """Build the manifest dict for an annotator run."""
    manifest: dict = {
        "run_id": run_id,
        "framework": "annotator",
        "mode": "annotate",
        "query_mode": "raw",
        "dataset": get_benchmark(args.benchmark).name,
        "benchmark_data_prefix": benchmark_data_prefix,
        "agent_model": args.agent_model,
        "effort": getattr(args, "effort", "medium"),
        "override": getattr(args, "override", False),
        "instance_ids": list(args.instance_ids),
        "no_subscription_auth": bool(getattr(args, "no_subscription_auth", False)),
        "render_inputs": {
            "workers": args.workers,
            "actors_per_worker": args.actors_per_worker,
            "worker_type": args.worker_type,
            "zone": cluster.DEFAULT_ZONE,
            "worker_sa": cluster.DEFAULT_WORKER_SA,
            "max_runtime_hours": args.max_runtime_hours,
            "image_uri": image_uri,
            "project": config.PROJECT,
            "region": config.REGION,
        },
    }
    return manifest


def _build_annotator_job_args(
    args, run_id: str, *, benchmark_data_prefix: str | None = None,
) -> list[str]:
    benchmark = get_benchmark(args.benchmark).name
    job_args = [
        "--run-id", run_id,
        "--benchmark", benchmark,
        "--model", args.agent_model,
        "--effort", getattr(args, "effort", "medium"),
        "--num-actors", str(args.workers * args.actors_per_worker),
        "--instance-ids", ",".join(args.instance_ids),
    ]
    if benchmark_data_prefix:
        job_args += ["--benchmark-data-prefix", benchmark_data_prefix]
    if getattr(args, "override", False):
        job_args.append("--override")
    return job_args


def submit_annotator(args) -> str:
    _validate_instance_ids(args.instance_ids, get_benchmark(args.benchmark).name)
    _prereq_args = argparse.Namespace(
        agent_model=args.agent_model,
        user_sim_model="",
        query_mode="raw",
        framework="annotator",
        no_subscription_auth=getattr(args, "no_subscription_auth", False),
    )
    prereqs.check(_prereq_args)
    repo_root = submitter_repo_root()
    bird_interact_eval, livesqlbench_eval = image.default_grader_eval_roots()
    tag = image.image_tag(
        repo_root,
        paths.audited_gold_root(),
        allow_dirty=args.allow_dirty,
        annotations_root=paths.annotations_root(),
        bird_interact_evaluation_root=bird_interact_eval,
        livesqlbench_evaluation_root=livesqlbench_eval,
    )
    image_uri = image.build_and_push(
        tag, repo_root,
        audited_gold_root=paths.audited_gold_root(),
        annotations_root=paths.annotations_root(),
        bird_interact_evaluation_root=bird_interact_eval,
        livesqlbench_evaluation_root=livesqlbench_eval,
        force=False,
    )
    _check_gold_present(get_benchmark(args.benchmark).name)
    benchmark_data_prefix = benchmark_data.ensure_uploaded(
        get_benchmark(args.benchmark).name
    )
    run_id = args.run_id or mint_run_id("annotator", "raw")
    manifest = build_annotator_manifest(
        args, image_uri=image_uri, run_id=run_id,
        benchmark_data_prefix=benchmark_data_prefix,
    )
    gcs.write_manifest(run_id, manifest)
    yaml_path = cluster.render_from_manifest(manifest, cache_dir=yaml_cache_dir())
    h = install_signal_handlers(run_id=run_id, yaml_path=yaml_path)
    submit_succeeded = False
    try:
        cluster.up(yaml_path)
        head = cluster.head_address(yaml_path)
        env_vars = read_api_keys_from_local_env(
            args.agent_model, "", query_mode="raw",
            framework="annotator",
            no_subscription_auth=getattr(args, "no_subscription_auth", False),
            dataset=getattr(args, "dataset", None) or getattr(args, "benchmark", ""),
        )
        job_args = _build_annotator_job_args(
            args, run_id, benchmark_data_prefix=benchmark_data_prefix,
        )
        # Write status BEFORE submit_job (see note in `submit`): the in-job
        # HeartbeatWriter races against this reset; writing first guarantees
        # the real heartbeat is not clobbered.
        gcs.write_status(run_id, {
            "ray_job_id": None,
            "last_heartbeat_ts": None,
            "rows_done": 0,
            "rows_total": len(args.instance_ids),
            "terminal_state": None,
            "attempt": 1,
        })
        cluster.submit_job(
            head_address=head, args=job_args, env_vars=env_vars,
            yaml_path=yaml_path,
            ray_app_path=_ANNOTATOR_RAY_APP_PATH,
        )
        submit_succeeded = True
        if not args.detach:
            wait_until_done(run_id, manifest)
            fetch(run_id)
    finally:
        if (not args.detach) or (not submit_succeeded):
            h.teardown(reason="finally")
    return run_id


# ---------------------------------------------------------------------------
# wait_until_done
# ---------------------------------------------------------------------------


HEARTBEAT_STALL_SECONDS = 300.0


def wait_until_done(run_id: str, manifest: dict, *,
                     poll_interval_s: float = 10.0,
                     no_progress_deadline_s: float = 3600.0,
                     min_attempt: int = 1) -> WaitResult:
    total = len(manifest.get("instance_ids", []))
    # `no_progress_deadline_s` is a NO-PROGRESS deadline, not a wall-clock
    # cap: it resets every time a new row lands. A wedged job (workers never
    # autoscale, actor sits PENDING, 0 rows) trips it; a healthy long run —
    # `max_runtime_hours` can be many hours — keeps resetting it and runs to
    # completion. (The VM self-delete timer + the headless check below bound
    # the absolute runtime.)
    #
    # `min_attempt` filters the completion + progress checks to rows whose
    # attempt number is >= min_attempt. Without this, resubmit() would return
    # `done` immediately because the manifest's instance_ids all already have
    # rows from the failed prior attempts. Defaults to 1 (counts everything),
    # preserving the original submit/annotate semantics.
    def _done_count(atts: dict[str, list[int]]) -> int:
        if min_attempt <= 1:
            return len(atts)
        return sum(1 for ns in atts.values() if any(n >= min_attempt for n in ns))

    last_progress = -1
    last_progress_ts = time.time()
    while True:
        status = gcs.read_status(run_id) or {}
        attempts = gcs.list_attempts(run_id)
        terminal = status.get("terminal_state")
        if terminal in ("done", "error"):
            return WaitResult(terminal_state=terminal)
        done_count = _done_count(attempts)
        if done_count >= total and total > 0:
            return WaitResult(terminal_state="done")
        # Progress = number of iids with at least one row at or after the
        # current attempt. Counting only current-attempt rows means the
        # no-progress deadline stays meaningful on a resubmit (it'd be
        # instantly maxed out if we counted prior-attempt rows here too).
        if done_count > last_progress:
            last_progress = done_count
            last_progress_ts = time.time()
        last = status.get("last_heartbeat_ts")
        if last is not None and (time.time() - float(last)) > HEARTBEAT_STALL_SECONDS:
            if cluster.head_is_alive(run_id):
                return WaitResult(
                    terminal_state="stalled",
                    hint="job appears stalled; fetch + resubmit",
                )
        if not cluster.head_is_alive(run_id) and total > 0:
            return WaitResult(
                terminal_state="headless",
                hint="head node is gone; fetch + resubmit",
            )
        # No-progress deadline: even with a fresh heartbeat, if NO new rows
        # have landed for this long the job is wedged (e.g. workers never
        # autoscaled up, so actors sit PENDING forever while the heartbeat
        # keeps writing `terminal_state: null`). Resets on every new row, so
        # a slow-but-progressing run is never falsely timed out.
        if (time.time() - last_progress_ts) > no_progress_deadline_s:
            return WaitResult(
                terminal_state="timed-out",
                hint=(
                    "no new rows within the no-progress deadline; check "
                    "`ray status` on the head (workers may not have scaled "
                    "up); resubmit"
                ),
            )
        if poll_interval_s <= 0:
            return WaitResult(terminal_state="poll-once-exit")
        time.sleep(poll_interval_s)


# ---------------------------------------------------------------------------
# fetch / kill / list / resubmit
# ---------------------------------------------------------------------------


def fetch(run_id: str, *, kill_after_fetch: bool = False) -> dict:
    client = default_gcs_client()
    manifest = gcs.read_manifest(run_id, client=client)
    benchmark = _benchmark_for_dataset(manifest.get("dataset"))
    dest = paths.results_root() / benchmark / "cloud" / run_id
    gcs.concurrent_download_prefix(run_id, dest, client=client)
    manifest_path = dest / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    metrics = _collation.collate(dest, manifest)
    # DEV-1470: promote per-DB cloud-encoded OTF references from
    # <run_dir>/post_run/slayer_models_otf/<shard>/<db>/ into the global
    # warm cache at paths.slayer_models_otf_root()/<db>/ (newest-mtime-wins
    # per file, with the marker invariant preserved under a per-DB flock).
    # No-op for runs without any post_run/ shards (raw, pre-encoded, recursive).
    merge_report = _post_run_merge.merge_post_run_into_warm_cache(
        run_dir=dest,
        reference_root=paths.slayer_models_otf_root(
            benchmark=_benchmark_for_dataset(manifest.get("dataset")),
        ),
    )
    metrics["merge_report"] = merge_report

    # DEV-1515: merge per-task submission_annotation.json files into
    # `<main_checkout>/annotations/<benchmark>/<db>/<inst>.submission.<run_id>.json`.
    # No-overwrite-if-present; schema-validated; audit log persisted.
    annotation_merge = _post_run_merge.merge_submission_annotations(
        downloaded_run_dir=dest,
        run_id=run_id,
        benchmark=_benchmark_for_dataset(manifest.get("dataset")),
    )
    metrics["annotation_merge_report"] = annotation_merge.model_dump()

    if manifest.get("framework") != "annotator":
        metrics = _emit_cascading_phase1_on_fetch(
            dest=dest, metrics=metrics,
            benchmark=_benchmark_for_dataset(manifest.get("dataset")),
            run_id=run_id,
        )

    # DEV-1518: for annotator runs, merge per-task annotation + gold variant
    # files from the downloaded rows into local stable storage.
    if manifest.get("framework") == "annotator":
        task_ann_report = _post_run_merge.merge_task_annotations(
            downloaded_run_dir=dest,
            benchmark=_benchmark_for_dataset(manifest.get("dataset")),
            annotations_root=paths.annotations_root(),
        )
        metrics["task_annotation_merge_report"] = task_ann_report.model_dump()
        variants_report = _post_run_merge.merge_audited_gold_variants(
            downloaded_run_dir=dest,
            benchmark=_benchmark_for_dataset(manifest.get("dataset")),
            audited_gold_root=paths.audited_gold_root(),
            override=manifest.get("override", False),
        )
        metrics["audited_gold_variants_merge_report"] = variants_report.model_dump()

    # Rewrite eval.json so annotation merge reports (added after collate()) are
    # persisted on disk — collate() writes eval.json before these keys exist.
    eval_path = dest / "eval.json"
    if eval_path.exists():
        eval_path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")

    if kill_after_fetch and cluster.head_is_alive(run_id):
        # Trust the caller's intent: ``kill_after_fetch=True`` (the CLI
        # default unless ``--no-kill`` was passed) means "I am done with
        # this cluster". Previously this branch guarded on
        # ``terminal_state in ('done','error')`` OR ``all_attempts_present``,
        # which silently skipped the kill whenever a waiter terminated
        # with a non-natural state — ``timed-out`` / ``stalled`` /
        # ``headless`` (the cluster only writes ``done``/``error`` itself;
        # waiter-side terminals never reach ``status.json``). The result
        # was leaked clusters on the runs `bird-interact-cloud fetch`
        # was clearly invoked to tear down. Inspecting a still-running
        # cluster without killing it now goes through ``fetch --no-kill``,
        # which is what the flag is for.
        logger.info("Shutting down cluster %s after fetch.", run_id)
        try:
            kill(run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto-kill after fetch failed for %s: %s", run_id, exc)
            metrics["kill_after_fetch_error"] = str(exc)

    return metrics


def _emit_cascading_phase1_on_fetch(
    *, dest: Path, metrics: dict,
    benchmark: "str | None" = None,
    run_id: "str | None" = None,
) -> dict:
    """Aggregate cascade metrics into the fetched run's eval.json.

    DEV-1533: reads from ``runs/<benchmark>/`` (the golden store populated
    by ``merge_submission_annotations``). Falls back to legacy rows_dir
    behaviour when ``benchmark`` / ``run_id`` are absent (older runs).
    """
    eval_path = dest / "eval.json"
    if not eval_path.exists():
        return metrics
    if benchmark and run_id:
        try:
            return _cascading_report.emit_cascading_eval_json(
                benchmark, run_id, eval_path, base_metrics=metrics,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("cascading_phase1 aggregation failed: %s", exc)
            metrics["cascading_phase1_error"] = str(exc)
            return metrics
    # Legacy fallback: rows_dir-based aggregation for older runs.
    rows_dir = dest / "rows"
    if not rows_dir.exists():
        return metrics
    has_per_row_anns = any(
        (row_dir / "submission_annotation.json").exists()
        for row_dir in rows_dir.iterdir() if row_dir.is_dir()
    )
    if not has_per_row_anns:
        return metrics
    try:
        from bird_interact_agents.eval.cascading_report import (
            _aggregate_cascading_phase1_legacy,
        )
        block = _aggregate_cascading_phase1_legacy(rows_dir)
        metrics["cascading_phase1"] = block
        metrics["phase1_count"] = block["counts"]["n1"]
        metrics["phase1_rate"] = block["rates"]["n1"]
        eval_path.write_text(
            __import__("json").dumps(metrics, indent=2, default=str) + "\n"
        )
        return metrics
    except FileNotFoundError as exc:
        logger.warning("cascading_phase1 aggregation failed: %s", exc)
        metrics["cascading_phase1_error"] = str(exc)
        return metrics


def kill(run_id: str) -> None:
    cache = yaml_cache_dir()
    yaml_path = cache / f"{run_id}.yaml"
    if not yaml_path.exists():
        manifest = gcs.read_manifest(run_id)
        yaml_path = cluster.render_from_manifest(manifest, cache_dir=cache)
    try:
        cluster.down(yaml_path)
    except Exception:  # noqa: BLE001
        cluster.fallback_delete_by_label(run_id)


def list_runs() -> list[dict]:
    client = default_gcs_client()
    bucket = client.bucket(gcs.BUCKET_NAME)
    seen: set[str] = set()
    for blob in bucket.list_blobs(prefix="runs/"):
        parts = blob.name.split("/", 2)
        if len(parts) >= 2:
            seen.add(parts[1])
    out: list[dict] = []
    for rid in sorted(seen):
        try:
            mf = gcs.read_manifest(rid, client=client)
        except Exception:  # noqa: BLE001
            continue
        attempts = gcs.list_attempts(rid, client=client)
        run_status = gcs.read_status(rid, client=client) or {}
        terminal = run_status.get("terminal_state")
        if terminal in ("done", "error"):
            status = terminal
        else:
            status = "live" if cluster.head_is_alive(rid) else "done"
        out.append({
            "run_id": rid,
            "framework": mf.get("framework"),
            "mode": mf.get("mode"),
            "query_mode": mf.get("query_mode"),
            "status": status,
            "done": len(attempts),
            "total": len(mf.get("instance_ids", [])),
        })
    return out


def resubmit(run_id: str) -> None:
    manifest = gcs.read_manifest(run_id)
    attempts = gcs.list_attempts(run_id)
    done: set[str] = set()
    max_attempt = 0
    for iid, lst in attempts.items():
        if not lst:
            continue
        max_attempt = max(max_attempt, max(lst))
        latest_n = lst[-1]
        try:
            row = gcs.read_row(run_id, iid, latest_n)
        except Exception:  # noqa: BLE001
            continue
        if not row.get("error"):
            done.add(iid)
    missing = [iid for iid in manifest.get("instance_ids", []) if iid not in done]
    if not missing:
        return
    next_attempt = max_attempt + 1
    yaml_path = cluster.render_from_manifest(manifest, cache_dir=yaml_cache_dir())
    h = install_signal_handlers(run_id=run_id, yaml_path=yaml_path)
    try:
        cluster.up(yaml_path)
        head = cluster.head_address(yaml_path)
        _framework = manifest.get("framework", "")
        # DEV-1535 r4 (Codex): legacy manifests (pre-DEV-1535) have NO
        # `no_subscription_auth` field at all. Pre-DEV-1535 the cloud
        # actor had a silent fall-back to the API-key path when the
        # OAuth token was absent; DEV-1535 killed that fall-back for
        # NEW submits (good — explicit-choice contract). For
        # RESUBMIT of legacy manifests, however, defaulting the
        # missing field to False (subscription-required) hard-fails
        # resubmits of every pre-DEV-1535 cloud run unless
        # CLAUDE_CODE_OAUTH_TOKEN is now in the env — a
        # backward-compat regression. Default to True (legacy API-key
        # path) when the field is absent so old runs stay
        # resubmittable. New manifests carry the field explicitly so
        # the strict-submit contract for the original submit is
        # unaffected.
        _no_subscription_auth = bool(manifest.get("no_subscription_auth", True))
        if "no_subscription_auth" not in manifest:
            logger.info(
                "resubmit: manifest has no 'no_subscription_auth' field "
                "(pre-DEV-1535); defaulting to legacy API-key path for "
                "backward compatibility"
            )
        if not _framework:
            logger.info(
                "resubmit: manifest has no 'framework' field (pre-DEV-1517); "
                "defaulting to legacy API-key path"
            )
        # Reset status BEFORE submit_job so the new attempt's in-job
        # HeartbeatWriter cannot have its first real heartbeat clobbered by
        # this reset. Mirrors submit/annotate. ray_job_id is unknown at this
        # point; the in-job HeartbeatWriter writes the real value.
        gcs.write_status(run_id, {
            "ray_job_id": None,
            "last_heartbeat_ts": None,
            "rows_done": 0,
            "rows_total": len(manifest.get("instance_ids", [])),
            "terminal_state": None,
            "attempt": next_attempt,
        })
        if _framework == "annotator":
            env_vars = read_api_keys_from_local_env(
                manifest["agent_model"], "",
                query_mode="raw", framework="annotator",
                no_subscription_auth=_no_subscription_auth,
                dataset=manifest.get("dataset", manifest.get("benchmark", "")),
            )
            job_args = _build_annotator_resubmit_args(manifest, run_id, missing, next_attempt)
            cluster.submit_job(
                head_address=head, args=job_args, env_vars=env_vars,
                yaml_path=yaml_path, ray_app_path=_ANNOTATOR_RAY_APP_PATH,
            )
        else:
            env_vars = read_api_keys_from_local_env(
                manifest["agent_model"], manifest["user_sim_model"],
                query_mode=manifest.get("query_mode", "raw"),
                framework=_framework,
                no_subscription_auth=_no_subscription_auth,
                dataset=manifest.get("dataset", ""),
            )
            job_args = _build_resubmit_args(manifest, run_id, missing, next_attempt)
            cluster.submit_job(
                head_address=head, args=job_args, env_vars=env_vars,
                yaml_path=yaml_path,
            )
        # Pass a derived manifest scoped to `missing`: resubmit only
        # dispatches the missing IIDs, so previously-succeeded IIDs will
        # never get a `next_attempt` row. Comparing the per-attempt
        # done-count against the FULL manifest length would prevent the
        # row-count completion fallback from ever firing on a partial
        # retry, leaving the run reliant on a single terminal-state write.
        # min_attempt=next_attempt: don't count failed prior-attempt rows
        # toward this retry's completion.
        retry_manifest = {**manifest, "instance_ids": missing}
        wait_until_done(run_id, retry_manifest, min_attempt=next_attempt)
        fetch(run_id)
    finally:
        h.teardown(reason="resubmit-finally")


def _build_resubmit_args(manifest: dict, run_id: str, missing: list[str],
                          attempt: int) -> list[str]:
    benchmark = _benchmark_for_dataset(manifest.get("dataset"))
    job_args = [
        "--run-id", run_id,
        "--attempt", str(attempt),
        "--framework", manifest.get("framework", ""),
        "--query-mode", manifest["query_mode"],
        "--mode", manifest["mode"],
        "--agent-model", manifest["agent_model"],
        "--user-sim-model", manifest["user_sim_model"],
        "--patience", str(manifest.get("patience", 3)),
        "--max-depth", str(manifest.get("max_depth", 3)),
        "--num-actors", str(
            manifest["render_inputs"]["workers"]
            * manifest["render_inputs"]["actors_per_worker"]
        ),
        # DEV-1470: db-grouped retries — same rule as submit.
        "--instance-ids", ",".join(_instance_ids_sorted_by_db(missing, benchmark)),
    ]
    # Pass --dataset / --benchmark-data-prefix ONLY when the manifest carries
    # them. A manifest with a `dataset` key was written by a driver new enough
    # to also bake `--dataset` support into its image; a pre-`--dataset`
    # manifest (no key) pins an OLDER image whose `ray_app` argparse rejects
    # `--dataset`, so resubmitting it MUST omit the flag and let the old image
    # use its baked mini-interact data (Codex). The prefix is independently
    # conditional for the same reason (de-bake came later than `--dataset`).
    if manifest.get("dataset"):
        job_args += ["--dataset", benchmark]
    prefix = manifest.get("benchmark_data_prefix")
    if prefix:
        job_args += ["--benchmark-data-prefix", prefix]
    if manifest.get("strict"):
        job_args.append("--strict")
    if manifest.get("use_audited_gold_sql"):
        job_args.append("--use-audited-gold-sql")
    if manifest.get("prompt_cache", True):
        job_args.append("--prompt-cache")
    else:
        job_args.append("--no-prompt-cache")
    if manifest.get("reasoning_effort"):
        job_args += ["--reasoning-effort", manifest["reasoning_effort"]]
    # DEV-1545: old manifests (pre-DEV-1545) lack the key. `.get`
    # returns None → falsy → flag omitted, so old manifests resubmit
    # under the v2 default the agent class already carries.
    if manifest.get("user_sim_prompt_version"):
        job_args += [
            "--user-sim-prompt-version",
            manifest["user_sim_prompt_version"],
        ]
    job_args += [
        "--slayer-setup", manifest.get("slayer_setup", "pre-encoded"),
        "--slayer-storage-root",
        manifest.get("slayer_storage_root", "/data/slayer_models"),
    ]
    # DEV-1586: forward the pre-encoded source. Back-compat: a pre-DEV-1586
    # manifest with slayer_setup="pre-encoded" but no source meant the
    # committed slayer_models reference → "custom".
    pre_src = manifest.get("pre_encoded_source")
    if not pre_src and manifest.get("slayer_setup") == "pre-encoded":
        pre_src = "custom"
    if pre_src:
        job_args += ["--pre-encoded-models", pre_src]
    return job_args


def _build_annotator_resubmit_args(
    manifest: dict, run_id: str, missing: list[str], attempt: int,
) -> list[str]:
    benchmark = _benchmark_for_dataset(manifest.get("dataset"))
    num_actors = (
        manifest["render_inputs"]["workers"]
        * manifest["render_inputs"]["actors_per_worker"]
    )
    job_args = [
        "--run-id", run_id,
        "--attempt", str(attempt),
        "--benchmark", benchmark,
        "--model", manifest["agent_model"],
        "--effort", manifest.get("effort", "medium"),
        "--num-actors", str(num_actors),
        "--instance-ids", ",".join(missing),
    ]
    prefix = manifest.get("benchmark_data_prefix")
    if prefix:
        job_args += ["--benchmark-data-prefix", prefix]
    if manifest.get("override"):
        job_args.append("--override")
    return job_args
