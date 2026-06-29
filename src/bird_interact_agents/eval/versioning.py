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

# The ONLY durable v2/v3 signal. Keyed by the exact run-id (the ``runs/``
# filename stem). Override wins over the framework map because these runs
# carry the clean framework token (``claude_sdk`` / ``claude_sdk_v1``).
_RUN_ID_VERSION_OVERRIDES = {
    "20260629t1209-claudes-slayer-cea364": "v2",  # modified-v0 (opus smoke)
    "20260624t0833-claudes-slayer-4df43f": "v2",  # modified-v0 (glm smoke)
    "20260624t0844-claudes-slayer-4246fd": "v3",  # modified-v1 (glm smoke)
}


def resolve_version(run_id: str, framework: Optional[str]) -> str:
    """Resolve a record's ``version`` from its run-id and framework token.

    Precedence:
      1. The run-id override table (the only v2/v3 signal).
      2. The framework→version map (``claude_sdk``→v0, ``claude_sdk_v1``→v1).
      3. ``DEFAULT_VERSION`` (v0) when the framework is missing/unknown — the
         clean-run default for in-flight origin/main runs.
      4. Otherwise the framework token verbatim, so a present-but-unmapped
         framework (e.g. an otf/encoder run) stays SEPARABLE from v0 rather
         than being silently folded into the clean-v0 bucket.
    """
    if run_id in _RUN_ID_VERSION_OVERRIDES:
        return _RUN_ID_VERSION_OVERRIDES[run_id]
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
    cheap and side-effect-free. Returns None when absent/corrupt."""
    p = paths.results_root() / benchmark / "cloud" / run_id / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - rare
        logger.warning("corrupt local manifest %s: %s", p, exc)
        return None


def stamp_provenance(
    ann: "SubmissionAnnotation",
    *,
    benchmark: str,
    run_id: str,
    manifest: Optional[dict] = None,
) -> None:
    """Stamp ``version`` + ``agent_model`` onto ``ann`` IN PLACE.

    No-clobber: only fills a field that is currently ``None`` (never
    overwrites an explicitly-set value). When ``manifest`` is omitted, a
    local manifest is loaded best-effort. The override table is keyed by
    ``run_id``, so an override run is tagged even with no manifest on disk.
    """
    if manifest is None:
        manifest = load_local_manifest(benchmark, run_id)
    framework, agent_model = provenance_from_manifest(manifest)
    if ann.version is None:
        ann.version = resolve_version(run_id, framework)
    if ann.agent_model is None and agent_model:
        ann.agent_model = agent_model
