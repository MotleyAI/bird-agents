"""Read / write helpers + canonical paths for annotations (DEV-1515).

Path conventions:

* Task annotation:
  ``<repo_root>/annotations/<benchmark>/<db>/<instance_id>.task.json``
* Submission annotation:
  ``<repo_root>/annotations/<benchmark>/<db>/<instance_id>.submission.<run_id>.json``

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

from pathlib import Path
from typing import Optional

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_schema import (
    SubmissionAnnotation,
    TaskAnnotation,
)

ANNOTATIONS_DIRNAME = "annotations"


def _canonical_benchmark(benchmark: str) -> str:
    """Return the benchmark name as-is; canonical names are now hyphenated
    (e.g. ``mini-interact``).  No normalization is applied so that
    annotations land in the correct directory after the DEV-1525 migration."""
    return benchmark


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
