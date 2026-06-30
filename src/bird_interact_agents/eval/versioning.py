"""DEV-1591 stream 2 — code-version provenance for ``runs/`` records.

Per-task ``runs/`` records (``SubmissionAnnotation``) carry a ``version`` so
runs from a modified-agent branch don't pollute the clean baselines: the
cascade tooling buckets on ``(query_mode, agent_model)`` and never read
``framework``, so a run from a branch's modified ``claude_sdk`` code (same
framework token as clean origin/main) would silently override the clean-v0
stats.

The version is written by the PRODUCER at creation time — never reconstructed
downstream:

* ``v0`` = clean origin/main ``claude_sdk`` (single-agent).
* ``v1`` = clean ``claude_sdk_v1`` (DEV-1581 R2 two-stage ``ask_discovery``).
* ``v2`` = a modified ``claude_sdk`` (the DEV-1591 prompt-experiment branch).
* ``v3`` = a modified ``claude_sdk_v1``.

``version_for_framework`` is THIS checkout's identity and is applied ONLY by the
producers — ``cloud.driver.build_manifest`` at submit and
``grade_in_place._build_submission_annotation`` at grade. Both run inside the
producing process/image, so they know the true code version. Every downstream
path (merge, regrade, annotate, backfill) only COPIES the literal the producer
wrote (from the record or the run's manifest); none of them re-map a framework
token, so a clean run merged/regraded while a modified branch is checked out
stays clean (and vice-versa).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional, Tuple

from bird_interact_agents import paths

if TYPE_CHECKING:  # avoid an import cycle (annotation_io imports this module)
    from bird_interact_agents.eval.annotation_schema import SubmissionAnnotation

logger = logging.getLogger(__name__)


# THIS checkout is clean origin/main, so its producers stamp v0/v1. Applied
# ONLY at production (build_manifest at submit, _build_submission_annotation at
# grade); downstream copies the literal, never re-derives.
#
# ┌─ IMPORTANT ──────────────────────────────────────────────────────────┐
# │ A modified-agent branch flips this to {"claude_sdk":"v2",             │
# │ "claude_sdk_v1":"v3"} so its experimental runs don't override these   │
# │ clean baselines (see the DEV-1591 prompt-experiment branch). On main  │
# │ it MUST stay {"claude_sdk":"v0","claude_sdk_v1":"v1"} — or make it a  │
# │ submit-time selection for the "support all 4 agent versions on one    │
# │ branch" work. Leaving v2/v3 here would mis-tag every clean run.       │
# └──────────────────────────────────────────────────────────────────────┘
VERSION_BY_FRAMEWORK = {
    "claude_sdk": "v0",
    "claude_sdk_v1": "v1",
}

# Manifest key the producer records (``build_manifest``); copied by downstream
# stampers. Absent on pre-DEV-1591 manifests.
MANIFEST_VERSION_KEY = "version"

# READ-SIDE ONLY. The cascade ``--version`` filter treats an UNTAGGED record
# (legacy, pre-stamping) as this version when deciding whether it matches the
# requested filter. This is a filtering convenience, NOT write-time
# reconstruction — the producer always writes the real version going forward.
DEFAULT_VERSION = "v0"


def version_for_framework(framework: Optional[str]) -> Optional[str]:
    """The version THIS checkout's producer assigns to a framework token, or
    ``None`` for a framework outside the v0–v3 taxonomy (e.g. an otf/encoder
    run). Called ONLY by producers; never to reconstruct an existing run."""
    return VERSION_BY_FRAMEWORK.get(framework or "")


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
    still fall back to — without this, a legacy-flat run can't surface the
    ``version`` its producer recorded in the manifest."""
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


def copy_provenance_from_manifest(
    ann: "SubmissionAnnotation",
    *,
    benchmark: str,
    run_id: str,
    manifest: Optional[dict] = None,
) -> None:
    """Fill ``version`` + ``agent_model`` on ``ann`` IN PLACE by COPYING the
    literal the producer recorded in the run's manifest — no framework→version
    mapping, no branch detection. For records that don't already carry the
    fields (a regrade/annotate rebuild, or a legacy merge); a producer-stamped
    record already has them, so this is a no-op (no-clobber). When ``manifest``
    is omitted it is loaded best-effort.
    """
    if manifest is None:
        manifest = load_local_manifest(benchmark, run_id)
    if not manifest:
        return
    if ann.version is None and manifest.get(MANIFEST_VERSION_KEY):
        ann.version = manifest[MANIFEST_VERSION_KEY]
    if ann.agent_model is None and manifest.get("agent_model"):
        ann.agent_model = manifest["agent_model"]
