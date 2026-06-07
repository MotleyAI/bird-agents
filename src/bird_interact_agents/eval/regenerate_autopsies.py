"""DEV-1541: backfill missing / errored autopsies on persisted submission
annotations.

Scans the per-(instance, run) annotation store
(``runs/<benchmark>/<db>/<instance>/<run_id>.json``) for entries whose
``evaluation.verdict == "agent_miss"`` AND whose autopsy is either
absent or carries an ``AutopsyError``. For each work item, reloads the
saved trajectory (preferring the ``runs/`` sidecar, falling back to the
cloud-results attempt JSON) and re-invokes ``run_autopsy`` with the
benchmark's ``is_one_shot`` flag. Overwrites the annotation on disk.

Two failure modes are tracked separately:

* ``io_errors`` — missing trajectory, unparseable annotation, or any
  failure that prevents the autopsy from being attempted at all.
* ``autopsy_errors`` — autopsy completed but ``run_autopsy`` returned
  an ``AutopsyError`` (e.g. context_overflow). The annotation still
  gets updated with the new error payload.

``RegenerationReport.exit_code()`` returns nonzero only on
``io_errors`` so a cron driver does not loop on transient LLM errors
(Codex r1 #9).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

from pydantic import BaseModel

from bird_interact_agents.eval.annotation_schema import (
    SubmissionAnnotation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


class WorkItem(BaseModel):
    """One annotation that needs an autopsy re-run."""
    model_config = {"frozen": True}

    instance_id: str
    selected_database: str
    run_id: str
    annotation_path: Path


class RegenerationReport(BaseModel):
    """Per-call summary of a ``regenerate`` invocation."""
    model_config = {"arbitrary_types_allowed": True}

    work_items: int = 0
    regenerated: int = 0
    autopsy_errors: int = 0
    io_errors: int = 0

    def exit_code(self) -> int:
        """Codex r1 #9: io_errors are real failures (annotation file
        unreadable, trajectory missing, write failed). LLM-returned
        ``AutopsyError`` is data, not a backfill failure — the
        annotation got updated either way."""
        return 1 if self.io_errors > 0 else 0


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def _annotation_needs_backfill(payload: dict) -> bool:
    """True iff this annotation is a genuine miss AND its autopsy is
    missing or carries an error.

    DEV-1541 r2 (Codex): also picks up pre-DEV-1533 annotations whose
    raw verdict is ``"invalid"``. ``SubmissionAnnotation._migrate_invalid_verdict``
    normalises those to ``"agent_miss"`` on read when
    ``failure_classification.primary != "other"`` (i.e. a real agent
    miss, not an infra failure). The scan walks raw JSON before
    validation, so we mirror the same migration rule here."""
    ev = payload.get("evaluation") or {}
    fc = payload.get("failure_classification") or {}
    verdict = ev.get("verdict")
    if verdict == "invalid" and fc.get("primary") != "other":
        verdict = "agent_miss"
    if verdict != "agent_miss":
        return False
    autopsy = payload.get("autopsy")
    if autopsy is None:
        return True
    if isinstance(autopsy, dict) and autopsy.get("error") is not None:
        return True
    return False


def scan_work_list(
    *,
    runs_root: Path,
    benchmark: str,
    run_id: Optional[str] = None,
    instance_ids: Optional[Iterable[str]] = None,
    on_parse_error: Optional[Callable[[Path, Exception], None]] = None,
) -> list[WorkItem]:
    """Walk ``runs_root/<benchmark>/*/*/*.json`` and return the work
    items. The optional ``run_id`` and ``instance_ids`` filters narrow
    the scan to a subset; both are AND-combined.

    DEV-1541 r2 (Codex + CodeRabbit): an unreadable / unparseable
    annotation file invokes ``on_parse_error(path, exc)`` (when
    supplied) before being skipped, so callers can count it toward
    ``RegenerationReport.io_errors`` instead of dropping it silently."""
    inst_filter = set(instance_ids) if instance_ids else None
    bench_root = Path(runs_root) / benchmark
    if not bench_root.exists():
        return []
    items: list[WorkItem] = []
    for ann_path in sorted(bench_root.glob("*/*/*.json")):
        # Skip the trajectory sidecars: <run_id>.trajectory.json.
        if ann_path.name.endswith(".trajectory.json"):
            continue
        # Path layout: <benchmark>/<db>/<instance>/<run_id>.json
        # DEV-1541 r3 (Codex): derive the filter values from the path
        # BEFORE opening the file. A targeted `--instance-ids` /
        # `--run-id` backfill should not bump io_errors for unrelated
        # corrupt files outside the filter.
        path_inst = ann_path.parent.name
        path_run = ann_path.stem
        if run_id is not None and path_run != run_id:
            continue
        if inst_filter is not None and path_inst not in inst_filter:
            continue
        try:
            payload = json.loads(ann_path.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[regenerate] failed to parse %s; counting as io_error: %s",
                ann_path, exc,
            )
            if on_parse_error is not None:
                on_parse_error(ann_path, exc)
            continue
        if not _annotation_needs_backfill(payload):
            continue
        # Prefer the payload's authoritative instance_id / db; fall back
        # to the path-derived values when missing.
        inst = payload.get("instance_id") or path_inst
        db = payload.get("selected_database") or ann_path.parent.parent.name
        run = path_run
        items.append(WorkItem(
            instance_id=inst,
            selected_database=db,
            run_id=run,
            annotation_path=ann_path,
        ))
    return items


# ---------------------------------------------------------------------------
# Trajectory loader (sidecar → cloud-results fallback)
# ---------------------------------------------------------------------------


def _load_trajectory(
    *,
    annotation_path: Path,
    benchmark: str,
    selected_database: str,
    instance_id: str,
    run_id: str,
    results_root: Optional[Path],
    submission_trajectory_path: Optional[str] = None,
) -> Optional[list[dict]]:
    """Resolve the trajectory for a single work item.

    Preference order:
    1. ``<runs>/<benchmark>/<db>/<inst>/<run_id>.trajectory.json`` —
       the DEV-1533 sidecar.
    2. ``<results>/<benchmark>/cloud/<run_id>/<submission.trajectory_path>``
       — uses the actual ``submission.trajectory_path`` from the
       annotation (e.g. ``rows/<inst>/attempt-2.json`` for resubmits).
       Falls back to the legacy hardcoded ``rows/<inst>/attempt-1.json``
       only when the annotation has no trajectory_path.

    DEV-1541 r3 (Codex): the previous implementation hardcoded
    ``attempt-1.json`` regardless of which attempt the annotation
    actually referred to, so backfilling a resubmit would silently
    regenerate the autopsy from the wrong agent session.

    Returns ``None`` if neither exists (counts as an IO error)."""
    # `with_suffix("").with_suffix(...)` only strips ONE suffix; the
    # explicit path is more reliable for files like `r1.json`:
    sidecar = annotation_path.parent / f"{annotation_path.stem}.trajectory.json"
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text())
            traj = payload.get("trajectory") if isinstance(payload, dict) else None
            if isinstance(traj, list):
                return traj
        except Exception:  # noqa: BLE001
            logger.warning("[regenerate] sidecar parse failed at %s", sidecar)
    if results_root is not None:
        # Prefer the annotation's actual trajectory_path so resubmits
        # (attempt-2+) load the right session. Fall back to the legacy
        # hardcoded path only when the annotation doesn't record one.
        if submission_trajectory_path:
            fallback = (
                Path(results_root) / benchmark / "cloud" / run_id
                / submission_trajectory_path
            )
        else:
            fallback = (
                Path(results_root) / benchmark / "cloud" / run_id
                / "rows" / instance_id / "attempt-1.json"
        )
        if fallback.exists():
            try:
                payload = json.loads(fallback.read_text())
                traj = payload.get("trajectory") if isinstance(payload, dict) else None
                if isinstance(traj, list):
                    return traj
            except Exception:  # noqa: BLE001
                logger.warning("[regenerate] fallback parse failed at %s", fallback)
    return None


# ---------------------------------------------------------------------------
# Per-task autopsy re-run
# ---------------------------------------------------------------------------


def _resolve_slayer_storage_dir(benchmark: str, db_name: str) -> str:
    """Resolve the per-DB SLayer OTF model dir (the source of KB
    memories). Missing dirs degrade gracefully — ``_read_kb_text``
    returns ``""`` when the dir is absent, so a missing path doesn't
    crash the backfill."""
    try:
        from bird_interact_agents import paths
        return str(paths.slayer_models_otf_root(benchmark=benchmark) / db_name)
    except Exception:  # noqa: BLE001
        return ""


async def _regenerate_one(
    item: WorkItem,
    *,
    benchmark: str,
    model: str,
    results_root: Optional[Path],
) -> tuple[str, Optional[Exception]]:
    """Re-run autopsy for one item and overwrite the annotation.

    Returns a tuple ``(outcome, exc)`` where ``outcome`` is one of:

    * ``"regenerated"`` — autopsy succeeded; analysis written back.
    * ``"autopsy_error"`` — autopsy completed but returned an
      ``AutopsyError``; the error payload was written back.
    * ``"io_error"`` — couldn't load/save; nothing was written.
    """
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.eval.autopsy import run_autopsy

    try:
        ann_payload = json.loads(item.annotation_path.read_text())
        ann = SubmissionAnnotation.model_validate(ann_payload)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[regenerate] failed to load annotation at %s: %s",
            item.annotation_path, exc,
        )
        return ("io_error", exc)

    trajectory = _load_trajectory(
        annotation_path=item.annotation_path,
        benchmark=benchmark,
        selected_database=item.selected_database,
        instance_id=item.instance_id,
        run_id=item.run_id,
        results_root=results_root,
        # DEV-1541 r3 (Codex): use the annotation's own trajectory_path
        # so the cloud-results fallback resolves the correct attempt
        # (resubmits land at attempt-2+).
        submission_trajectory_path=(
            ann.submission.trajectory_path
            if ann.submission is not None
            else None
        ),
    )
    if trajectory is None:
        logger.error(
            "[regenerate] no trajectory available for %s/%s/%s",
            benchmark, item.instance_id, item.run_id,
        )
        return ("io_error", FileNotFoundError(
            f"no trajectory for {item.instance_id}/{item.run_id}"
        ))

    # Build the task annotation (used to seed the prompt). Falls back to
    # an implicit one if no on-disk task annotation exists. The helper
    # lives in grade_in_place.py (it's re-exported from there as the
    # only public surface).
    from bird_interact_agents.eval.grade_in_place import (
        load_task_annotation_or_implicit,
    )
    try:
        task_ann = load_task_annotation_or_implicit(
            instance_id=item.instance_id,
            selected_database=item.selected_database,
            benchmark=benchmark,
            amb_user_query=ann_payload.get("submission", {}).get(
                "amb_user_query", ""
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[regenerate] failed to load task annotation for %s: %s",
            item.instance_id, exc,
        )
        return ("io_error", exc)

    miss_diagnostics = ann.evaluation.miss_diagnostics
    slayer_dir = _resolve_slayer_storage_dir(benchmark, item.selected_database)
    is_one_shot = get_benchmark(benchmark).one_shot

    result = await run_autopsy(
        task_annotation=task_ann,
        trajectory=trajectory,
        slayer_storage_dir=slayer_dir,
        miss_diagnostics=miss_diagnostics,
        model=model,
        is_one_shot=is_one_shot,
    )

    ann.autopsy = result
    # DEV-1541 r2 (Codex): mirror grade_and_write's guarded promotion —
    # when the autopsy actually succeeded, propagate its
    # decision_point + user_sim_interaction up to the top-level
    # annotation fields so a backfilled annotation looks identical to
    # one written inline at grade time. On the error path we leave the
    # existing top-level fields alone.
    if result.analysis is not None:
        ann.decision_point = result.decision_point
        ann.user_sim_interaction = result.user_sim_interaction
    # Persist; preserve indent=2 like the original writers.
    try:
        item.annotation_path.write_text(
            ann.model_dump_json(indent=2, exclude_none=False) + "\n"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[regenerate] failed to write %s: %s", item.annotation_path, exc,
        )
        return ("io_error", exc)

    if result.error is not None:
        return ("autopsy_error", None)
    return ("regenerated", None)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def regenerate(
    *,
    runs_root: Path,
    benchmark: str,
    model: str,
    dry_run: bool = False,
    run_id: Optional[str] = None,
    instance_ids: Optional[Iterable[str]] = None,
    results_root: Optional[Path] = None,
) -> RegenerationReport:
    """Backfill missing or errored autopsies under ``runs_root/<benchmark>/``.

    ``--dry-run`` reports the work-list without instantiating the
    Anthropic client. ``--run-id`` and ``--instance-ids`` narrow the
    scan to a subset (AND-combined).

    DEV-1541 r2 (CodeRabbit): when ``results_root`` is ``None`` we
    resolve it from ``paths.results_root()`` so the advertised
    sidecar→cloud-results trajectory fallback is reachable for the
    normal CLI path (the CLI doesn't pass ``--results-root``)."""
    if results_root is None:
        try:
            from bird_interact_agents import paths
            results_root = paths.results_root()
        except Exception:  # noqa: BLE001
            results_root = None
    report = RegenerationReport()
    # DEV-1541 r2 (Codex + CodeRabbit): unreadable annotation files
    # bump io_errors via this callback instead of being dropped.
    def _on_parse_error(_path: Path, _exc: Exception) -> None:
        report.io_errors += 1
    items = scan_work_list(
        runs_root=runs_root, benchmark=benchmark,
        run_id=run_id, instance_ids=instance_ids,
        on_parse_error=_on_parse_error,
    )
    report.work_items = len(items)

    if dry_run:
        for it in items:
            print(
                f"[dry-run] would regenerate "
                f"{it.instance_id} / {it.run_id} "
                f"({it.annotation_path})"
            )
        return report

    async def _drive() -> None:
        for it in items:
            outcome, _ = await _regenerate_one(
                it,
                benchmark=benchmark,
                model=model,
                results_root=results_root,
            )
            if outcome == "regenerated":
                report.regenerated += 1
            elif outcome == "autopsy_error":
                report.autopsy_errors += 1
            elif outcome == "io_error":
                report.io_errors += 1

    asyncio.run(_drive())
    return report
