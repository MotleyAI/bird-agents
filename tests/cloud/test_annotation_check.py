"""Tests for the submit-time annotation-availability guard
(``cloud._annotation_check.missing_annotation_ids``).

The annotation file is the authoritative grading source post-DEV-1515.
The guard is default-on in ``bird-interact-cloud submit`` and runs
unconditionally — it does NOT key off ``--use-audited-gold-sql``.
"""

from __future__ import annotations

import json
from pathlib import Path


def _write_dataset(data_path: Path, rows: list[dict]) -> None:
    data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _write_annotation(
    ann_root: Path, benchmark: str, db: str, iid: str,
) -> Path:
    d = ann_root / benchmark / db
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{iid}.task.json"
    fp.write_text("{}")  # presence-only check; the loader is exercised elsewhere
    return fp


def test_present_annotation_passes(tmp_path: Path) -> None:
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        missing_annotation_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    ann_root = tmp_path / "annotations"
    _write_annotation(ann_root, "mini-interact", "alien", "alien_1")

    missing = missing_annotation_ids(
        ["alien_1"],
        benchmark=get_benchmark("mini-interact"),
        annotations_root=ann_root,
        data_path=data,
    )
    assert missing == []


def test_missing_annotation_file_is_reported(tmp_path: Path) -> None:
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        missing_annotation_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    ann_root = tmp_path / "annotations"
    # No annotation file written.

    missing = missing_annotation_ids(
        ["alien_1"],
        benchmark=get_benchmark("mini-interact"),
        annotations_root=ann_root,
        data_path=data,
    )
    assert missing == ["alien_1"]


def test_unknown_instance_id_is_reported(tmp_path: Path) -> None:
    """An id absent from the benchmark data file (typo / stale id) is
    reported missing even when an annotation file accidentally exists."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        missing_annotation_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    ann_root = tmp_path / "annotations"
    _write_annotation(ann_root, "mini-interact", "alien", "alien_1")

    missing = missing_annotation_ids(
        ["alien_9999"],
        benchmark=get_benchmark("mini-interact"),
        annotations_root=ann_root,
        data_path=data,
    )
    assert missing == ["alien_9999"]


def test_partial_batch_reports_only_missing_ids(tmp_path: Path) -> None:
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        missing_annotation_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
        {"instance_id": "alien_2", "selected_database": "alien"},
        {"instance_id": "alien_3", "selected_database": "alien"},
    ])
    ann_root = tmp_path / "annotations"
    _write_annotation(ann_root, "mini-interact", "alien", "alien_1")
    _write_annotation(ann_root, "mini-interact", "alien", "alien_3")
    # alien_2 has no annotation.

    missing = missing_annotation_ids(
        ["alien_1", "alien_2", "alien_3"],
        benchmark=get_benchmark("mini-interact"),
        annotations_root=ann_root,
        data_path=data,
    )
    assert missing == ["alien_2"]


def test_livesqlbench_path_uses_canonical_benchmark_name(tmp_path: Path) -> None:
    """The helper resolves
    ``<ann_root>/<benchmark.name>/<db>/<iid>.task.json`` — for livesqlbench
    that's the hyphenated `livesqlbench-base-lite-sqlite` subdirectory."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        missing_annotation_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_dataset(data, [
        {"instance_id": "museum_7", "selected_database": "museum"},
    ])
    ann_root = tmp_path / "annotations"
    _write_annotation(
        ann_root, "livesqlbench-base-lite-sqlite", "museum", "museum_7",
    )

    missing = missing_annotation_ids(
        ["museum_7"],
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
        annotations_root=ann_root,
        data_path=data,
    )
    assert missing == []


def test_livesqlbench_missing_annotation_is_reported(tmp_path: Path) -> None:
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        missing_annotation_ids,
    )

    data = tmp_path / "livesqlbench_data_sqlite.jsonl"
    _write_dataset(data, [
        {"instance_id": "museum_7", "selected_database": "museum"},
    ])
    ann_root = tmp_path / "annotations"
    # File at wrong benchmark subdir — must NOT count as present.
    _write_annotation(ann_root, "mini-interact", "museum", "museum_7")

    missing = missing_annotation_ids(
        ["museum_7"],
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
        annotations_root=ann_root,
        data_path=data,
    )
    assert missing == ["museum_7"]


def test_default_annotations_root_used_when_kwarg_omitted(
    tmp_path: Path, monkeypatch,
) -> None:
    """When the caller omits ``annotations_root``, the helper resolves
    via ``paths.annotations_root()`` — same path-override pattern as the
    rest of the cloud guards."""
    from bird_interact_agents import paths as _paths
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        missing_annotation_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    ann_root = tmp_path / "annotations"
    _write_annotation(ann_root, "mini-interact", "alien", "alien_1")

    monkeypatch.setattr(_paths, "annotations_root", lambda: ann_root)

    missing = missing_annotation_ids(
        ["alien_1"],
        benchmark=get_benchmark("mini-interact"),
        data_path=data,
    )
    assert missing == []
