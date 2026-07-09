"""Read / write helpers + canonical paths for annotations (DEV-1515/1533).

Path conventions:

* Task annotation:
  ``<repo_root>/annotations/<benchmark>/<db>/<instance_id>.task.json``
* Run annotation (DEV-1533):
  ``<repo_root>/runs/<benchmark>/<db>/<instance_id>/<run_id>.json``
* Run trajectory sidecar:
  ``<repo_root>/runs/<benchmark>/<db>/<instance_id>/<run_id>.trajectory.json``

``<benchmark>`` is the canonical hyphenated name (``mini-interact``,
``livesqlbench-base-lite-sqlite``) as returned by
:func:`bird_interact_agents.benchmark.get_benchmark`. The name is used
as-is in path components; callers must pass the canonical form.

``<db>`` matches the task's ``selected_database`` field.

These helpers do NOT regenerate annotations from raw artefacts — that
belongs in a separate tool (``scripts/generate_annotation_skeleton.py``
or similar). Here we only do schema-validated JSON I/O.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_schema import (
    SubmissionAnnotation,
    TaskAnnotation,
)

logger = logging.getLogger(__name__)

ANNOTATIONS_DIRNAME = "annotations"
RUNS_DIRNAME = "runs"


def _canonical_benchmark(benchmark: str) -> str:
    """Return the benchmark name as-is; canonical names are now hyphenated
    (e.g. ``mini-interact``).  No normalization is applied so that
    annotations land in the correct directory after the DEV-1525 migration."""
    return benchmark


def _runs_root(repo_root: Optional[Path] = None) -> Path:
    """Anchor at the main checkout for the runs/ tree (DEV-1533).

    When ``repo_root`` is None, delegate to ``paths.runs_root()`` so
    ``BIRD_RUNS_ROOT`` is honoured. Passing an explicit ``repo_root``
    bypasses the override — the caller has already pinned the location.
    """
    if repo_root is None:
        return paths.runs_root()
    return Path(repo_root) / RUNS_DIRNAME


def _annotations_root(repo_root: Optional[Path] = None) -> Path:
    """Anchor at the main checkout (matches the worktree-safe contract
    used by ``audited_gold/`` and ``results/``).

    When ``repo_root`` is None, delegate to ``paths.annotations_root()``
    so the ``BIRD_ANNOTATIONS_ROOT`` env override is honoured (used by
    tests + forks that mount the annotations tree elsewhere). Passing
    an explicit ``repo_root`` bypasses the override on purpose — the
    caller has already pinned the location."""
    if repo_root is None:
        return paths.annotations_root()
    return Path(repo_root) / ANNOTATIONS_DIRNAME


def task_annotation_path(
    *,
    benchmark: str,
    selected_database: str,
    instance_id: str,
    repo_root: Optional[Path] = None,
) -> Path:
    return (
        _annotations_root(repo_root)
        / _canonical_benchmark(benchmark)
        / selected_database
        / f"{instance_id}.task.json"
    )


def submission_annotation_path(
    *,
    benchmark: str,
    selected_database: str,
    instance_id: str,
    run_id: str,
    repo_root: Optional[Path] = None,
) -> Path:
    return (
        _annotations_root(repo_root)
        / _canonical_benchmark(benchmark)
        / selected_database
        / f"{instance_id}.submission.{run_id}.json"
    )


def read_task_annotation(path: Path) -> TaskAnnotation:
    return TaskAnnotation.model_validate_json(path.read_text())


def write_task_annotation(ann: TaskAnnotation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ann.model_dump_json(indent=2, exclude_none=False) + "\n")


def read_submission_annotation(path: Path) -> SubmissionAnnotation:
    return SubmissionAnnotation.model_validate_json(path.read_text())


def write_submission_annotation(ann: SubmissionAnnotation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ann.model_dump_json(indent=2, exclude_none=False) + "\n")


def iter_task_annotations(
    *, benchmark: str, repo_root: Optional[Path] = None,
) -> "list[tuple[Path, TaskAnnotation]]":
    """Walk every task annotation under ``annotations/<benchmark>/``.

    Returns ``(path, model)`` pairs; raises if any file fails validation
    so corruption surfaces at scan time rather than at first downstream
    use.
    """
    root = _annotations_root(repo_root) / _canonical_benchmark(benchmark)
    out: list[tuple[Path, TaskAnnotation]] = []
    if not root.exists():
        return out
    for path in sorted(root.rglob("*.task.json")):
        out.append((path, read_task_annotation(path)))
    return out


def iter_submission_annotations(
    *, benchmark: str, run_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> "list[tuple[Path, SubmissionAnnotation]]":
    """Walk submission annotations. When ``run_id`` is set, filter."""
    root = _annotations_root(repo_root) / _canonical_benchmark(benchmark)
    out: list[tuple[Path, SubmissionAnnotation]] = []
    if not root.exists():
        return out
    pattern = f"*.submission.{run_id}.json" if run_id else "*.submission.*.json"
    for path in sorted(root.rglob(pattern)):
        out.append((path, read_submission_annotation(path)))
    return out


# ---------------------------------------------------------------------------
# DEV-1533: runs/ path helpers
# ---------------------------------------------------------------------------


def run_annotation_path(
    *,
    benchmark: str,
    selected_database: str,
    instance_id: str,
    run_id: str,
    repo_root: Optional[Path] = None,
) -> Path:
    """``runs/<benchmark>/<db>/<instance_id>/<run_id>.json``."""
    return (
        _runs_root(repo_root)
        / _canonical_benchmark(benchmark)
        / selected_database
        / instance_id
        / f"{run_id}.json"
    )


def run_trajectory_path(
    *,
    benchmark: str,
    selected_database: str,
    instance_id: str,
    run_id: str,
    repo_root: Optional[Path] = None,
) -> Path:
    """``runs/<benchmark>/<db>/<instance_id>/<run_id>.trajectory.json``."""
    return (
        _runs_root(repo_root)
        / _canonical_benchmark(benchmark)
        / selected_database
        / instance_id
        / f"{run_id}.trajectory.json"
    )


def run_edited_models_archive(
    *,
    benchmark: str,
    selected_database: str,
    instance_id: str,
    repo_root: Optional[Path] = None,
) -> Path:
    """``runs/<benchmark>/<db>/<instance_id>/edited_models.tar.gz`` (DEV-1649).

    Per-task (no ``run_id`` component — latest-wins overwrite). A ``.tar.gz``,
    never a ``*.json``, so the recursive ``rglob("*.json")`` annotation walkers
    (``iter_run_annotations`` / ``latest_run_per_instance``) ignore it."""
    return (
        _runs_root(repo_root)
        / _canonical_benchmark(benchmark)
        / selected_database
        / instance_id
        / "edited_models.tar.gz"
    )


def _infer_benchmark_run_id(
    path: Path,
) -> "tuple[Optional[str], Optional[str]]":
    """Best-effort ``(benchmark, run_id)`` from a ``runs/`` destination path
    (``<runs_root>/<benchmark>/<db>/<inst>/<run_id>.json``). Returns
    ``(None, None)`` when ``path`` is not under the default runs root — e.g.
    a tmp path in tests, or a writer that pinned an explicit ``repo_root``
    (those pass benchmark/run_id explicitly instead)."""
    try:
        rel = path.relative_to(_runs_root(None))
    except ValueError:
        return (None, None)
    parts = rel.parts
    if len(parts) >= 4 and parts[-1].endswith(".json"):
        return (parts[0], parts[-1][: -len(".json")])
    return (None, None)


def _copy_provenance(
    ann: SubmissionAnnotation,
    path: Path,
    *,
    benchmark: Optional[str],
    run_id: Optional[str],
    manifest: Optional[dict],
    allow_manifest_fallback: bool = True,
) -> None:
    """DEV-1591: fill ``version`` + ``agent_model`` on ``ann`` (if missing) by
    COPYING the literal the producer recorded in the run's manifest — never
    re-derived from the framework. Producer-stamped records already carry both
    (no-op). Explicit ``benchmark``/``run_id`` (passed by the runs/ writers)
    take precedence; otherwise infer from the path. A no-op when neither is
    resolvable (a tmp path in tests, etc.).

    ``allow_manifest_fallback=False`` (set by a caller that already read THIS
    run's manifest and found it unreadable) suppresses the best-effort
    self-load so a corrupt manifest can't draw provenance from an unrelated
    same-named run under the default results root."""
    if benchmark is None or run_id is None:
        b2, r2 = _infer_benchmark_run_id(path)
        benchmark = benchmark or b2
        run_id = run_id or r2
    if not benchmark or not run_id:
        return
    from bird_interact_agents.eval import versioning

    versioning.copy_provenance_from_manifest(
        ann, benchmark=benchmark, run_id=run_id, manifest=manifest,
        allow_manifest_fallback=allow_manifest_fallback,
    )


def write_run_annotation(
    ann: SubmissionAnnotation,
    path: Path,
    *,
    benchmark: Optional[str] = None,
    run_id: Optional[str] = None,
    manifest: Optional[dict] = None,
    allow_manifest_fallback: bool = True,
) -> None:
    _copy_provenance(
        ann, path, benchmark=benchmark, run_id=run_id, manifest=manifest,
        allow_manifest_fallback=allow_manifest_fallback,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ann.model_dump_json(indent=2, exclude_none=False) + "\n")


def _attempt_from_trajectory(trajectory_path: Optional[str]) -> Optional[int]:
    """Parse attempt number from ``rows/<inst>/attempt-N.json``."""
    import re
    if not trajectory_path:
        return None
    m = re.search(r"attempt-(\d+)\.json$", trajectory_path)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def write_run_annotation_no_overwrite(
    ann: SubmissionAnnotation,
    path: Path,
    *,
    benchmark: Optional[str] = None,
    run_id: Optional[str] = None,
    manifest: Optional[dict] = None,
    allow_manifest_fallback: bool = True,
) -> bool:
    """Write ``ann`` to ``path`` only if absent or the new attempt is strictly
    greater than the stored one (resubmit case). Returns True if written.

    When either attempt number is unparseable, the existing file is preserved
    (return False) rather than silently overwritten — unknown ordering is not
    a sufficient reason to discard a valid existing annotation. The only path
    that overwrites is: parseable new attempt > parseable existing attempt, OR
    the existing file is corrupt (validation raised).
    """
    if path.exists():
        try:
            existing = SubmissionAnnotation.model_validate_json(path.read_text())
            existing_attempt = _attempt_from_trajectory(
                existing.submission.trajectory_path
            )
            new_attempt = _attempt_from_trajectory(ann.submission.trajectory_path)
            # Preserve existing unless we can prove the new attempt is strictly
            # newer. Unknown ordering on either side → preserve.
            if (
                existing_attempt is None
                or new_attempt is None
                or new_attempt <= existing_attempt
            ):
                return False
        except Exception:  # noqa: BLE001
            pass  # corrupt destination — overwrite
    write_run_annotation(
        ann, path, benchmark=benchmark, run_id=run_id, manifest=manifest,
        allow_manifest_fallback=allow_manifest_fallback,
    )
    return True


def iter_run_annotations(
    *,
    benchmark: str,
    run_id: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> "list[tuple[Path, SubmissionAnnotation]]":
    """Walk run annotations under ``runs/<benchmark>/``.

    When ``run_id`` is set, only files named ``<run_id>.json`` are
    returned (i.e. a specific run). Otherwise all ``*.json`` files that
    are NOT trajectory sidecars are returned.
    """
    root = _runs_root(repo_root) / _canonical_benchmark(benchmark)
    out: list[tuple[Path, SubmissionAnnotation]] = []
    if not root.exists():
        return out
    pattern = f"{run_id}.json" if run_id else "*.json"
    for path in sorted(root.rglob(pattern)):
        if path.name.endswith(".trajectory.json"):
            continue
        try:
            out.append((path, read_submission_annotation(path)))
        except Exception as exc:  # noqa: BLE001
            # Surface corruption to operators: silent skip masks denominator
            # shrinkage in downstream cascade aggregation.
            logger.warning(
                "[iter_run_annotations] skipping unreadable %s: %s: %s",
                path, type(exc).__name__, exc,
            )
    return out


def latest_run_per_instance(
    *,
    benchmark: str,
    repo_root: Optional[Path] = None,
    version: Optional[str] = None,
) -> "dict[tuple[str, str], tuple[str, SubmissionAnnotation]]":
    """Return the latest run annotation per ``(selected_database, instance_id)``.

    "Latest" is determined by the ``annotated_at`` field of each annotation
    (ISO timestamp written at grading time), not by lexicographic run_id —
    this handles local run_ids that are not timestamp-formatted.

    When ``version`` is set (DEV-1591), records whose effective version (the
    stamped ``version``, defaulting a missing value to ``v0``) does not match
    are excluded BEFORE the latest-per-task pick — so a newer wrong-version
    run can't override the requested-version baseline.

    Returns ``{(db, instance_id): (run_id, annotation)}``.
    """
    root = _runs_root(repo_root) / _canonical_benchmark(benchmark)
    best: dict[tuple[str, str], tuple[str, str, SubmissionAnnotation]] = {}
    # Expected layout: runs/<benchmark>/<db>/<instance_id>/<run_id>.json
    if not root.exists():
        return {}
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".trajectory.json"):
            continue
        # path.parts from root: (<db>, <instance_id>, <run_id>.json)
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        db, iid, run_file = parts
        run_id = run_file[:-5]  # strip .json
        try:
            ann = read_submission_annotation(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[latest_run_per_instance] skipping unreadable %s: %s: %s",
                path, type(exc).__name__, exc,
            )
            continue
        if version is not None:
            from bird_interact_agents.eval.versioning import DEFAULT_VERSION
            if (ann.version or DEFAULT_VERSION) != version:
                continue
        key = (db, iid)
        annotated_at = ann.annotated_at or ""
        cur = best.get(key)
        if cur is None or annotated_at > cur[1]:
            best[key] = (run_id, annotated_at, ann)
    return {k: (v[0], v[2]) for k, v in best.items()}
