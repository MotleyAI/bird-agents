"""DEV-1470: per-task upload-back from cloud workers.

Three best-effort upload helpers called from
:func:`bird_interact_agents.cloud.ray_app._run_one_in_actor` after the row +
log writes, before the per-task tmp-dir cleanup:

* :func:`upload_per_task_debug` — ships ``<work_root>/<iid>/`` to
  ``runs/<run_id>/sessions/<iid>/attempt-<n>/`` (the per-task agent session
  log dir + the HARD-8 SLayer scratch storage).
* :func:`upload_per_task_setup_sessions` — for each
  ``<setup_sessions_root>/<db>/`` with any file mtime ≥ ``task_start_ts``,
  uploads to ``runs/<run_id>/sessions/_setup_sessions/<db>/``.
* :func:`upload_otf_reference_delta` — for each per-DB OTF reference whose
  on-disk fingerprint differs from the actor's initial seed snapshot AND
  isn't already in the actor's ``uploaded_dbs`` set, uploads the WHOLE
  ``<db>/`` subtree to ``runs/<run_id>/post_run/slayer_models_otf/<shard>/<db>/``.

All three swallow their own exceptions and log to stderr — the per-task row
already landed before this hook fires, and a logging-side failure must never
poison a run.

Upload ordering for the OTF reference delta (M1 / Codex round-1):

    bulk content files  →  _source_mtimes.json sidecar  →  _upload_complete

The sidecar carries the per-file LOCAL mtime so the laptop merger can apply
newest-mtime-wins faithfully even though GCS re-clocks `blob.updated` on
upload. ``_upload_complete`` is written LAST so a concurrent fetch can refuse
to merge an in-progress shard (see :mod:`bird_interact_agents.cloud.post_run_merge`).
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

from bird_interact_agents.benchmark import get_benchmark
from bird_interact_agents.cloud import gcs as _gcs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-task debug bundle
# ---------------------------------------------------------------------------


def upload_per_task_debug(
    *,
    run_id: str,
    iid: str,
    attempt: int,
    work_root: Path,
    client: Any,
) -> None:
    """Upload per-task scratch (``<work_root>/<iid>[-<uuid>]/``) to
    ``runs/<run_id>/sessions/<iid>/attempt-<n>/``. No-op when no matching
    dir exists (non-OTF runs never populate it). Best-effort.

    The OTF scratch resolver (``slayer_otf.runtime._otf_work_dir``) appends
    a UUID suffix to keep concurrent runs of the same instance from
    ``rmtree``-ing each other's live storage, so the on-disk layout is
    ``<work_root>/<iid>-<uuid>/`` rather than ``<work_root>/<iid>/``. We
    match both shapes via ``Path.glob`` so cloud debug uploads work for
    every OTF flavor (DEV-1505 + DEV-1507) without forcing the resolver
    back to colliding names.
    """
    try:
        work_root = Path(work_root)
        prefix = f"runs/{run_id}/sessions/{iid}/attempt-{attempt}"
        # Exact ``<iid>/`` (legacy, non-OTF callers) first, then any
        # ``<iid>-<suffix>/`` produced by the OTF scratch resolver. Both
        # are uploaded under the same destination prefix — concurrent
        # invocations of the same task on one actor are not expected, but
        # if they happen we'd want all scratch contents shipped.
        candidates: list[Path] = []
        legacy = work_root / iid
        if legacy.is_dir():
            candidates.append(legacy)
        candidates.extend(
            d for d in sorted(work_root.glob(f"{iid}-*")) if d.is_dir()
        )
        for d in candidates:
            _gcs.upload_dir_prefix(d, prefix, client=client)
    except Exception:  # noqa: BLE001
        sys.stderr.write(
            f"[upload_back] upload_per_task_debug failed for {iid}: "
            f"{traceback.format_exc()}\n"
        )


# ---------------------------------------------------------------------------
# Per-task setup-encoder sessions
# ---------------------------------------------------------------------------


def upload_per_task_setup_sessions(
    *,
    run_id: str,
    iid: str,
    attempt: int,  # noqa: ARG001 — kept for symmetry / future per-attempt sinks
    setup_sessions_root: Path,
    task_start_ts: float,
    client: Any,
) -> None:
    """For each ``<setup_sessions_root>/<db>/`` whose any file's mtime is
    ≥ ``task_start_ts``, upload to
    ``runs/<run_id>/sessions/_setup_sessions/<db>/``. The per-DB setup-encoder
    sessions are per-actor scratch — a task that doesn't trigger a build
    leaves no fresh files, so the mtime gate skips stale per-actor state from
    earlier tasks.

    Best-effort: a logging-side failure never poisons the run."""
    try:
        root = Path(setup_sessions_root)
        if not root.is_dir():
            return
        for db_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            # Any file under <db>/ newer-or-equal than task_start_ts → upload.
            fresh = any(
                p.is_file() and p.stat().st_mtime >= task_start_ts
                for p in db_dir.rglob("*")
            )
            if not fresh:
                continue
            prefix = f"runs/{run_id}/sessions/_setup_sessions/{db_dir.name}"
            _gcs.upload_dir_prefix(db_dir, prefix, client=client)
    except Exception:  # noqa: BLE001
        sys.stderr.write(
            f"[upload_back] upload_per_task_setup_sessions failed for {iid}: "
            f"{traceback.format_exc()}\n"
        )


# ---------------------------------------------------------------------------
# OTF reference delta (per-DB warm-cache promotion)
# ---------------------------------------------------------------------------


_REF_MARKER = "_reference_fp.txt"


def _read_fp(db_dir: Path) -> str | None:
    """Return the contents of ``<db_dir>/_reference_fp.txt`` or ``None`` when
    the marker is absent — the OTF reference is incomplete and must not be
    uploaded."""
    marker = db_dir / _REF_MARKER
    if not marker.is_file():
        return None
    try:
        return marker.read_text().strip()
    except OSError:
        return None


def _collect_source_mtimes(db_dir: Path) -> dict[str, float]:
    """Walk ``db_dir`` and return ``{rel_posix_path: float_mtime}`` for every
    file under it (used as the sidecar payload)."""
    out: dict[str, float] = {}
    for p in sorted(db_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(db_dir).as_posix()
            out[rel] = p.stat().st_mtime
    return out


def _upload_one_db_shard(
    *,
    db: str,
    db_dir: Path,
    run_id: str,
    shard: str,
    client: Any,
) -> None:
    """Upload everything under ``db_dir`` to
    ``runs/<run_id>/post_run/slayer_models_otf/<shard>/<db>/``.

    Ordering (M1): bulk content first → ``_source_mtimes.json`` →
    ``_upload_complete``. A concurrent fetch that observes the prefix WITHOUT
    ``_upload_complete`` treats it as in-progress and ignores the shard, so
    the marker-last write is load-bearing for fetch-time consistency."""
    base = f"runs/{run_id}/post_run/slayer_models_otf/{shard}/{db}"
    bucket = client.bucket(_gcs.BUCKET_NAME)

    # 1. Bulk content.
    _gcs.upload_dir_prefix(db_dir, base, client=client)

    # 2. Sidecar (per-file local mtime).
    sidecar = _collect_source_mtimes(db_dir)
    bucket.blob(f"{base}/_source_mtimes.json").upload_from_string(
        json.dumps(sidecar).encode(),
    )

    # 3. Completeness marker — LAST.
    bucket.blob(f"{base}/_upload_complete").upload_from_string(b"ok")


def upload_otf_reference_delta(
    *,
    run_id: str,
    cfg: dict[str, Any],
    shard: str,
    uploaded_dbs: set[str],
    initial_seed_fp_by_db: dict[str, str],
    client: Any,
) -> None:
    """Upload any newly-built per-DB OTF reference that this actor hasn't
    already shipped this run.

    Eligibility per ``<db>/`` under ``paths.slayer_models_otf_root()``:
        * ``_reference_fp.txt`` is present (build complete)  AND
        * ``<db>`` not in ``uploaded_dbs`` (per-actor retry tracker — H2)  AND
        * on-disk fp != ``initial_seed_fp_by_db.get(<db>)`` (truly new content)

    On full success, adds ``<db>`` to ``uploaded_dbs`` so future tasks on this
    actor don't re-upload. On failure, leaves ``uploaded_dbs`` untouched so
    the next task retries (H2 / M6 — otherwise an unlucky first-attempt
    failure permanently loses the warm-cache artifact).

    No-op for any combo that doesn't actually use ``paths.slayer_models_otf_root()``
    (raw mode, pre-encoded slayer, recursive on-the-fly). Restricting to
    ``otf_encode + on-the-fly`` is load-bearing: under the recursive combo
    ``initial_seed_fp_by_db`` is ``{}`` (the optional seed download isn't
    wired for that combo), so any stale ``slayer_models_otf/<db>/`` left on
    a shared worker filesystem would otherwise pass the fingerprint check
    and pollute the laptop warm cache (CodeRabbit)."""
    if (
        cfg.get("query_mode") != "slayer"
        or cfg.get("framework") != "pydantic_ai_otf_encode"
        or cfg.get("slayer_setup") != "on-the-fly"
    ):
        return
    try:
        from bird_interact_agents import paths  # local import: tests stub `paths`

        dataset = cfg.get("dataset")
        if not dataset:
            raise ValueError(
                "upload_otf_reference_delta requires cfg['dataset'] — "
                "task data is missing the required 'dataset' field"
            )
        benchmark = get_benchmark(dataset).name
        # DEV-1605: walk the version-scoped reference root the cloud encode
        # built into (encode_version, default = agent-model slug). The GCS
        # post_run prefix stays <db>-keyed — `runs/<run_id>/` already isolates
        # versions across runs, so the version need not enter the prefix.
        from bird_interact_agents.model_string import resolve_encode_version
        encode_version = resolve_encode_version(
            cfg.get("encode_version"), cfg.get("agent_model"),
        )
        ref_root = paths.slayer_models_otf_root(
            benchmark=benchmark, version=encode_version,
        )
        if not ref_root.is_dir():
            return
        for db_dir in sorted(p for p in ref_root.iterdir() if p.is_dir()):
            db = db_dir.name
            if db in uploaded_dbs:
                continue
            local_fp = _read_fp(db_dir)
            if local_fp is None:
                continue  # incomplete — never ship without the marker
            if initial_seed_fp_by_db.get(db) == local_fp:
                continue  # unchanged from initial seed — nothing new to ship
            try:
                _upload_one_db_shard(
                    db=db, db_dir=db_dir, run_id=run_id, shard=shard,
                    client=client,
                )
            except Exception:  # noqa: BLE001
                # H2 — per-db failure: log + skip; do NOT mark as uploaded so
                # a later task can retry. Continue with the next db.
                sys.stderr.write(
                    f"[upload_back] upload_otf_reference_delta failed for "
                    f"db={db}: {traceback.format_exc()}\n"
                )
                continue
            uploaded_dbs.add(db)
    except Exception:  # noqa: BLE001
        # Outer guard: walking paths.slayer_models_otf_root() itself could
        # raise (e.g. paths import wedged); log + carry on.
        sys.stderr.write(
            f"[upload_back] upload_otf_reference_delta outer failure: "
            f"{traceback.format_exc()}\n"
        )
