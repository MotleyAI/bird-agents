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
    """Write a minimal schema-valid TaskAnnotation.

    DEV-1535 r4: the bare `{}` stub no longer passes the primary
    guard (which now schema-validates), so tests that want
    "annotation present + grader accepts" need a real annotation.
    `original_gold_is_correct=True` keeps the test out of the
    layered guard's audited-gold check too."""
    from bird_interact_agents.eval.annotation_io import write_task_annotation
    from bird_interact_agents.eval.annotation_schema import (
        MetadataSufficiency, Provenance, TaskAnnotation,
    )
    ann = TaskAnnotation(
        instance_id=iid, selected_database=db,
        annotated_by="test", annotated_at="2026-06-09",
        amb_user_query="q", external_knowledge=[], masked_terms=[],
        metadata_sufficiency=MetadataSufficiency(
            verdict="sufficient", rationale="r",
            evidence_sources_consulted=[],
        ),
        original_gold_is_correct=True, gold_variants=[],
        provenance=Provenance(
            task_jsonl_path="x.jsonl", task_jsonl_instance_id=iid,
        ),
    )
    d = ann_root / benchmark / db
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{iid}.task.json"
    write_task_annotation(ann, fp)
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


def test_directory_at_annotation_path_is_treated_as_missing(tmp_path: Path) -> None:
    """DEV-1535 r2 (CodeRabbit): `is_file()` not `exists()` — a
    directory accidentally created at the annotation path no longer
    bypasses the submit-time guard."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        missing_annotation_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    ann_root = tmp_path / "annotations"
    # Create a DIRECTORY where the annotation file should be.
    dirpath = ann_root / "mini-interact" / "alien" / "alien_1.task.json"
    dirpath.mkdir(parents=True)

    missing = missing_annotation_ids(
        ["alien_1"],
        benchmark=get_benchmark("mini-interact"),
        annotations_root=ann_root,
        data_path=data,
    )
    assert missing == ["alien_1"]


def test_malformed_annotation_is_reported_by_primary_guard(
    tmp_path: Path,
) -> None:
    """DEV-1535 r4 (Codex): the primary guard now schema-validates the
    annotation file too, not just checks presence. Pre-fix the
    validation only fired inside the layered audited-gold check
    (gated on `--use-audited-gold-sql`), so a malformed file passed
    submit under `--no-use-audited-gold-sql` and then failed later
    when the harness called `read_task_annotation`."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        missing_annotation_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    ann_root = tmp_path / "annotations"
    d = ann_root / "mini-interact" / "alien"
    d.mkdir(parents=True)
    # Garbage JSON — read_task_annotation raises.
    (d / "alien_1.task.json").write_text("{not valid json")

    missing = missing_annotation_ids(
        ["alien_1"],
        benchmark=get_benchmark("mini-interact"),
        annotations_root=ann_root,
        data_path=data,
    )
    assert missing == ["alien_1"]


def test_schema_invalid_annotation_is_reported_by_primary_guard(
    tmp_path: Path,
) -> None:
    """Same guard catches a parseable-but-schema-invalid annotation —
    e.g. JSON that's missing required fields."""
    import json as _json
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        missing_annotation_ids,
    )

    data = tmp_path / "mini_interact.jsonl"
    _write_dataset(data, [
        {"instance_id": "alien_1", "selected_database": "alien"},
    ])
    ann_root = tmp_path / "annotations"
    d = ann_root / "mini-interact" / "alien"
    d.mkdir(parents=True)
    # JSON parses but Pydantic validation fails (missing `provenance`,
    # `metadata_sufficiency`, etc).
    (d / "alien_1.task.json").write_text(_json.dumps({
        "instance_id": "alien_1", "selected_database": "alien",
    }))

    missing = missing_annotation_ids(
        ["alien_1"],
        benchmark=get_benchmark("mini-interact"),
        annotations_root=ann_root,
        data_path=data,
    )
    assert missing == ["alien_1"]


# ---------------------------------------------------------------------------
# DEV-1535 r2 (Codex) — layered guard for the annotation ↔ audited_gold
# sync gap.
# ---------------------------------------------------------------------------


def _write_full_annotation(
    ann_root: Path, benchmark: str, db: str, iid: str,
    *, original_gold_is_correct: bool,
    with_variant: bool = False,
) -> Path:
    """Write a schema-valid TaskAnnotation via the pydantic models. The
    layered guard reads `original_gold_is_correct`, so a bare `{}` stub
    won't do. `with_variant=True` adds the single primary variant the
    schema requires when `original_gold_is_correct=False`."""
    from bird_interact_agents.eval.annotation_schema import (
        AuditedGoldRef, GoldVariantRef, MetadataSufficiency, Provenance,
        TaskAnnotation,
    )
    from bird_interact_agents.eval.annotation_io import write_task_annotation

    variants = []
    if with_variant:
        variants = [GoldVariantRef(
            variant_id="primary", interpretation="p", primary=True,
            audited_gold_ref=AuditedGoldRef(
                file=f"audited_gold/{benchmark}/{benchmark}_audited.jsonl",
                instance_id=iid, variant_id="primary",
            ),
        )]
    ann = TaskAnnotation(
        instance_id=iid, selected_database=db,
        annotated_by="test", annotated_at="2026-06-09",
        amb_user_query="q", external_knowledge=[], masked_terms=[],
        metadata_sufficiency=MetadataSufficiency(
            verdict="sufficient", rationale="r",
            evidence_sources_consulted=[],
        ),
        original_gold_is_correct=original_gold_is_correct,
        gold_variants=variants,
        provenance=Provenance(
            task_jsonl_path="x.jsonl", task_jsonl_instance_id=iid,
        ),
    )
    d = ann_root / benchmark / db
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{iid}.task.json"
    write_task_annotation(ann, fp)
    return fp


def _write_audited_gold(audited_root: Path, benchmark: str,
                         iid: str, db: str) -> Path:
    """Drop a single-file `<bench>/<bench>_audited.jsonl` with one
    minimal row for `iid`."""
    import json as _json
    d = audited_root / benchmark
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{benchmark}_audited.jsonl"
    fp.write_text(_json.dumps({
        "instance_id": iid,
        "selected_database": db,
        "benchmark": benchmark,
        "variants": [{
            "variant_id": "primary",
            "primary": True,
            "audit_status": "edited",
            "audited_sol_sql": ["SELECT 1"],
        }],
    }) + "\n")
    return fp


def test_layered_check_passes_when_annotation_says_original_correct(
    tmp_path: Path, monkeypatch,
) -> None:
    """`original_gold_is_correct=True` → no audited row needed →
    layered guard returns empty list. Most common case."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        annotations_requiring_audited_gold_without_rows,
    )
    from bird_interact_agents import paths as _paths

    # Stub the dataset map.
    monkeypatch.setattr(_paths, "benchmark_data_file",
        lambda *a, **k: _write_mini_dataset(tmp_path, [
            {"instance_id": "alien_1", "selected_database": "alien"},
        ]),
    )
    ann_root = tmp_path / "annotations"
    _write_full_annotation(
        ann_root, "livesqlbench-base-lite-sqlite", "alien", "alien_1",
        original_gold_is_correct=True,
    )

    missing = annotations_requiring_audited_gold_without_rows(
        ["alien_1"],
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
        annotations_root=ann_root,
    )
    assert missing == []


def test_layered_check_flags_annotation_wrong_gold_no_audited_row(
    tmp_path: Path, monkeypatch,
) -> None:
    """`original_gold_is_correct=False` AND audited_gold sidecar
    missing → layered guard reports the iid. This is the silent-fallback
    gap Codex flagged in round 2."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        annotations_requiring_audited_gold_without_rows,
    )
    from bird_interact_agents import paths as _paths

    monkeypatch.setattr(_paths, "benchmark_data_file",
        lambda *a, **k: _write_mini_dataset(tmp_path, [
            {"instance_id": "museum_7", "selected_database": "museum"},
        ]),
    )
    ann_root = tmp_path / "annotations"
    _write_full_annotation(
        ann_root, "livesqlbench-base-lite-sqlite", "museum", "museum_7",
        original_gold_is_correct=False, with_variant=True,
    )
    # NO audited_gold sidecar — that's the gap.
    monkeypatch.setattr(_paths, "audited_gold_root",
                        lambda: tmp_path / "audited_gold_empty")

    missing = annotations_requiring_audited_gold_without_rows(
        ["museum_7"],
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
        annotations_root=ann_root,
    )
    assert missing == ["museum_7"]


def test_layered_check_passes_when_audited_row_present(
    tmp_path: Path, monkeypatch,
) -> None:
    """Annotation says wrong-gold, audited_gold has a matching row →
    no gap, no failure."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        annotations_requiring_audited_gold_without_rows,
    )
    from bird_interact_agents import paths as _paths

    monkeypatch.setattr(_paths, "benchmark_data_file",
        lambda *a, **k: _write_mini_dataset(tmp_path, [
            {"instance_id": "museum_7", "selected_database": "museum"},
        ]),
    )
    ann_root = tmp_path / "annotations"
    _write_full_annotation(
        ann_root, "livesqlbench-base-lite-sqlite", "museum", "museum_7",
        original_gold_is_correct=False, with_variant=True,
    )
    audited_root = tmp_path / "audited_gold"
    _write_audited_gold(
        audited_root, "livesqlbench-base-lite-sqlite", "museum_7", "museum",
    )
    monkeypatch.setattr(_paths, "audited_gold_root", lambda: audited_root)

    missing = annotations_requiring_audited_gold_without_rows(
        ["museum_7"],
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
        annotations_root=ann_root,
    )
    assert missing == []


def test_layered_check_reports_malformed_annotation(
    tmp_path: Path, monkeypatch,
) -> None:
    """DEV-1535 r3 (Codex): a malformed annotation file (parse error
    or schema validation failure) is REPORTED, not silently skipped.
    The submit guard exists to surface these before cluster startup;
    swallowing them recreates the delayed-failure mode the guard
    prevents."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        annotations_requiring_audited_gold_without_rows,
    )
    from bird_interact_agents import paths as _paths

    monkeypatch.setattr(_paths, "benchmark_data_file",
        lambda *a, **k: _write_mini_dataset(tmp_path, [
            {"instance_id": "museum_7", "selected_database": "museum"},
        ]),
    )
    ann_root = tmp_path / "annotations"
    d = ann_root / "livesqlbench-base-lite-sqlite" / "museum"
    d.mkdir(parents=True)
    # Garbage JSON — pydantic will raise on read_task_annotation.
    (d / "museum_7.task.json").write_text("{not valid json")

    missing = annotations_requiring_audited_gold_without_rows(
        ["museum_7"],
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
        annotations_root=ann_root,
    )
    assert missing == ["museum_7"]


def test_layered_check_reads_audited_gold_sidecar_once(
    tmp_path: Path, monkeypatch,
) -> None:
    """DEV-1535 r3 (CodeRabbit perf): the audited-gold sidecar is read
    ONCE into a presence set, not per-iid. Asserted via a spy on
    `_build_audited_gold_presence_index` (the new single-pass helper)
    — a 5-iid batch must call the sidecar reader exactly once."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud import _annotation_check as ann_mod
    from bird_interact_agents import paths as _paths

    iids = [f"museum_{i}" for i in range(1, 6)]
    monkeypatch.setattr(_paths, "benchmark_data_file",
        lambda *a, **k: _write_mini_dataset(tmp_path, [
            {"instance_id": iid, "selected_database": "museum"}
            for iid in iids
        ]),
    )
    ann_root = tmp_path / "annotations"
    for iid in iids:
        _write_full_annotation(
            ann_root, "livesqlbench-base-lite-sqlite", "museum", iid,
            original_gold_is_correct=False, with_variant=True,
        )
    monkeypatch.setattr(_paths, "audited_gold_root",
                        lambda: tmp_path / "audited_gold_empty")

    call_count = {"n": 0}
    original = ann_mod._build_audited_gold_presence_index
    def spy(bench_name: str) -> set[str]:
        call_count["n"] += 1
        return original(bench_name)
    monkeypatch.setattr(
        ann_mod, "_build_audited_gold_presence_index", spy,
    )

    missing = ann_mod.annotations_requiring_audited_gold_without_rows(
        iids,
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
        annotations_root=ann_root,
    )
    assert missing == iids
    assert call_count["n"] == 1, (
        f"audited-gold sidecar read {call_count['n']} times; expected 1 "
        f"(O(N_ids) regression)"
    )


def test_layered_check_skips_audited_index_when_no_iid_needs_it(
    tmp_path: Path, monkeypatch,
) -> None:
    """Tiny perf detail: when every annotation says
    `original_gold_is_correct=True`, we never even build the
    presence index. Asserted via the same spy as above — zero calls
    expected."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud import _annotation_check as ann_mod
    from bird_interact_agents import paths as _paths

    monkeypatch.setattr(_paths, "benchmark_data_file",
        lambda *a, **k: _write_mini_dataset(tmp_path, [
            {"instance_id": "alien_1", "selected_database": "alien"},
        ]),
    )
    ann_root = tmp_path / "annotations"
    _write_full_annotation(
        ann_root, "livesqlbench-base-lite-sqlite", "alien", "alien_1",
        original_gold_is_correct=True,
    )

    call_count = {"n": 0}
    def spy(bench_name: str) -> set[str]:
        call_count["n"] += 1
        return set()
    monkeypatch.setattr(
        ann_mod, "_build_audited_gold_presence_index", spy,
    )

    missing = ann_mod.annotations_requiring_audited_gold_without_rows(
        ["alien_1"],
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
        annotations_root=ann_root,
    )
    assert missing == []
    assert call_count["n"] == 0


def test_layered_check_skips_ids_without_annotation_file(
    tmp_path: Path, monkeypatch,
) -> None:
    """IIDs that lack an annotation entirely are skipped — the primary
    `missing_annotation_ids` guard catches them. The layered check
    only fires for the annotation ↔ audited_gold sync gap."""
    from bird_interact_agents.benchmark import get_benchmark
    from bird_interact_agents.cloud._annotation_check import (
        annotations_requiring_audited_gold_without_rows,
    )
    from bird_interact_agents import paths as _paths

    monkeypatch.setattr(_paths, "benchmark_data_file",
        lambda *a, **k: _write_mini_dataset(tmp_path, [
            {"instance_id": "alien_1", "selected_database": "alien"},
        ]),
    )
    # No annotation file at all.
    ann_root = tmp_path / "annotations"
    ann_root.mkdir()

    missing = annotations_requiring_audited_gold_without_rows(
        ["alien_1"],
        benchmark=get_benchmark("livesqlbench-base-lite-sqlite"),
        annotations_root=ann_root,
    )
    assert missing == []


def _write_mini_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    """Helper that lays down a livesqlbench-shaped dataset file.
    Used by the layered-check tests above for the
    `benchmark_data_file` stub."""
    import json as _json
    d = tmp_path / "lsb"
    d.mkdir(exist_ok=True)
    fp = d / "livesqlbench_data_sqlite.jsonl"
    fp.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    return fp
