"""DEV-1591 stream 2 — code-version provenance for ``runs/`` records.

Per-task ``runs/`` records (``SubmissionAnnotation``) historically carried
no signal of WHICH agent code version produced them. The cascade tooling
buckets only on ``(query_mode, agent_model)`` from the cloud manifest and
never reads ``framework``, so a run from this branch's *modified* agent code
(same ``claude_sdk`` framework token as clean origin/main) silently overrode
the clean-v0 baselines per task.

This module owns the version taxonomy and the rules to resolve a record's
``version`` + ``agent_model``:

* ``v0`` = clean origin/main ``claude_sdk`` (single-agent).
* ``v1`` = clean ``claude_sdk_v1`` (DEV-1581 R2 two-stage ``ask_discovery``).
* ``v2`` = this branch's modified ``claude_sdk`` (the compact-search work).
* ``v3`` = this branch's modified ``claude_sdk_v1``.

The manifest records NO durable code-version signal (no git commit / code
hash; the image tag is a content hash that can't be mapped back). So the
three runs positively attributable to this branch can only be tagged via an
explicit run-id override table — a token→version mapping ALONE would fold
them back into clean v0/v1. See the ``project_dev1591_modified_run_versions``
note for the provenance of this list.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional, Tuple

from bird_interact_agents import paths

if TYPE_CHECKING:  # avoid an import cycle (annotation_io imports this module)
    from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation

logger = logging.getLogger(__name__)


DEFAULT_VERSION = "v0"
"""Version assigned when the framework is unknown/missing. In-flight clean
runs are submitted from origin/main, so a record whose manifest is absent
defaults to clean v0 (a deliberate decision — see DEV-1591)."""

_FRAMEWORK_TO_VERSION = {
    "claude_sdk": "v0",
    "claude_sdk_v1": "v1",
}

# This branch carries MODIFIED agent code (the DEV-1591 compact-search
# prompt changes), so every run PRODUCED BY IT is a modified variant: v2
# (modified-v0 / ``claude_sdk``) or v3 (modified-v1 / ``claude_sdk_v1``),
# never the clean v0/v1. ``_BRANCH_IS_MODIFIED`` is the durable, code-baked
# signal that the running process is this branch — it travels into the cloud
# image (unlike git state), so cloud runs are tagged correctly too.
#
# ┌─ IMPORTANT ──────────────────────────────────────────────────────────┐
# │ Reset to False when these modifications land on main / are retired.   │
# │ Once "modified" IS main, its runs ARE the clean v0/v1 baseline, and   │
# │ leaving this True would mis-tag every main run as v2/v3.              │
# └──────────────────────────────────────────────────────────────────────┘
#
# It governs ONLY live write-time stamping (``resolve_version(...,
# live_run=True)`` from ``stamp_provenance``), where we KNOW the running
# code is this branch. The BACKFILL of historical records uses the default
# ``live_run=False`` path: a historical record's branch is unknown, so its
# version is derived from its own manifest framework + the override table
# (assume clean unless explicitly overridden). Mixing the two would let the
# backfill re-tag genuinely-clean runs from other branches as v2/v3.
_BRANCH_IS_MODIFIED = True

_MODIFIED_FRAMEWORK_TO_VERSION = {
    "claude_sdk": "v2",
    "claude_sdk_v1": "v3",
}
_MODIFIED_DEFAULT_VERSION = "v2"

# Historical runs from this branch made BEFORE ``_BRANCH_IS_MODIFIED`` existed
# (their records were written by code that stamped clean v0/v1, then fixed by
# the backfill via this table). Keyed by the exact run-id (the ``runs/``
# filename stem); wins over every framework map. New live runs no longer need
# an entry here — they self-stamp v2/v3 via ``_BRANCH_IS_MODIFIED``.
_RUN_ID_VERSION_OVERRIDES = {
    "20260629t1209-claudes-slayer-cea364": "v2",  # modified-v0 (opus smoke)
    "20260624t0833-claudes-slayer-4df43f": "v2",  # modified-v0 (glm smoke)
    "20260624t0844-claudes-slayer-4246fd": "v3",  # modified-v1 (glm smoke)
}


def resolve_version(
    run_id: str, framework: Optional[str], *, live_run: bool = False,
) -> str:
    """Resolve a record's ``version`` from its run-id and framework token.

    Precedence:
      1. The run-id override table (authoritative for the listed runs).
      2. When ``live_run`` AND this is the modified branch: the MODIFIED
         framework map (``claude_sdk``→v2, ``claude_sdk_v1``→v3) and a
         modified default (v2) — every run this branch produces is tagged
         distinctly from clean main without needing a per-run override.
      3. The clean framework→version map (``claude_sdk``→v0,
         ``claude_sdk_v1``→v1).
      4. ``DEFAULT_VERSION`` (v0) when the framework is missing/unknown — the
         clean-run default for historical / origin/main runs.
      5. Otherwise the framework token verbatim, so a present-but-unmapped
         framework (e.g. an otf/encoder run) stays SEPARABLE from v0 rather
         than being silently folded into the clean-v0 bucket.

    ``live_run=True`` is passed ONLY by the write-time stamp of a run being
    produced by THIS process (``stamp_provenance``). The backfill and any
    historical re-derivation use the default (clean) path — see
    ``_BRANCH_IS_MODIFIED``.
    """
    if run_id in _RUN_ID_VERSION_OVERRIDES:
        return _RUN_ID_VERSION_OVERRIDES[run_id]
    if live_run and _BRANCH_IS_MODIFIED:
        if framework in _MODIFIED_FRAMEWORK_TO_VERSION:
            return _MODIFIED_FRAMEWORK_TO_VERSION[framework]
        if not framework:
            return _MODIFIED_DEFAULT_VERSION
        return framework
    if framework in _FRAMEWORK_TO_VERSION:
        return _FRAMEWORK_TO_VERSION[framework]
    if not framework:
        return DEFAULT_VERSION
    return framework


def provenance_from_manifest(
    manifest: Optional[dict],
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(framework, agent_model)`` from a cloud manifest dict."""
    if not manifest:
        return (None, None)
    return (manifest.get("framework"), manifest.get("agent_model"))


def load_local_manifest(benchmark: str, run_id: str) -> Optional[dict]:
    """Best-effort LOCAL-ONLY manifest read (no GCS) — keeps the write path
    cheap and side-effect-free. Returns None when absent/corrupt.

    Checks the benchmark-scoped run dir first, then the legacy-flat layout
    (``results/cloud/<run_id>/``) that ``eval.annotate`` / ``eval.regrade``
    still fall back to — without this, a legacy-flat run can't surface its
    framework and gets stamped default-v0 with no agent_model."""
    results_root = paths.results_root()
    for p in (
        results_root / benchmark / "cloud" / run_id / "manifest.json",
        results_root / "cloud" / run_id / "manifest.json",
    ):
        if not p.exists():
            continue
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover
            logger.warning("corrupt local manifest %s: %s", p, exc)
            return None
    return None


def stamp_provenance(
    ann: "SubmissionAnnotation",
    *,
    benchmark: str,
    run_id: str,
    manifest: Optional[dict] = None,
) -> None:
    """Stamp ``version`` + ``agent_model`` onto ``ann`` IN PLACE.

    This is the LIVE write path — the run is being produced by THIS process,
    so version resolution runs with ``live_run=True``: on the modified branch
    a clean framework is tagged as its modified twin (v2/v3). No-clobber:
    only fills a field that is currently ``None`` (never overwrites an
    explicitly-set value — so the backfill's clean resolution, pre-set before
    the write, is preserved). When ``manifest`` is omitted, a local manifest
    is loaded best-effort. The override table is keyed by ``run_id``, so an
    override run is tagged even with no manifest on disk.
    """
    if manifest is None:
        manifest = load_local_manifest(benchmark, run_id)
    framework, agent_model = provenance_from_manifest(manifest)
    if ann.version is None:
        ann.version = resolve_version(run_id, framework, live_run=True)
    if ann.agent_model is None and agent_model:
        ann.agent_model = agent_model
