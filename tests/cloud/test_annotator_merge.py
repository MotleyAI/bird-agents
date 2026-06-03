"""Tests for merge_task_annotations and merge_audited_gold_variants (DEV-1518).

Contract:
* merge_task_annotations: writes to correct local path, always overwrites
  (skip logic was upstream at worker), handles missing rows/, rejects malformed,
  normalises benchmark name dash↔underscore.
* merge_audited_gold_variants: deduplicates by (instance_id, variant_id),
  appends new entries, skips empty JSONL files, rejects entries missing
  required fields (selected_database, benchmark, audit_status, audited_sol_sql).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _minimal_task_annotation_dict(instance_id: str = "shop_1", db: str = "shop") -> dict:
    return {
        "schema_version": 1,
        "kind": "task_annotation",
        "instance_id": instance_id,
        "selected_database": db,
        "annotated_by": "annotator-agent/test",
        "annotated_at": "2026-06-02",
        "amb_user_query": "How many orders?",
        "external_knowledge": [],
        "masked_terms": [],
        "metadata_sufficiency": {
            "verdict": "sufficient",
            "rationale": "KB 1 directly answers the question.",
            "evidence_sources_consulted": ["kb:1"],
        },
        "original_gold_is_correct": True,
        "gold_variants": [],
        "provenance": {
            "task_jsonl_path": "mini_interact.jsonl",
            "task_jsonl_instance_id": instance_id,
        },
    }


def _audited_gold_variant(
    instance_id: str = "shop_1",
    variant_id: str = "primary",
    db: str = "shop",
) -> dict:
    return {
        "instance_id": instance_id,
        "variant_id": variant_id,
        "selected_database": db,
        "benchmark": "mini-interact",
        "audit_status": "clean",
        "original_sol_sql": ["SELECT COUNT(*) FROM orders;"],
        "audited_sol_sql": ["SELECT COUNT(*) FROM orders;"],
        "audited_sample_row": [42],
        "changes": [],
        "reasoning_summary": "Gold SQL is fully justified by KB 1.",
        "skill_version": "annotator-agent/1.0",
        "audited_at": "2026-06-02",
    }


def _write_task_annotation_file(rows_dir: Path, instance_id: str, db: str) -> None:
    task_dir = rows_dir / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task_annotation.json").write_text(
        json.dumps(_minimal_task_annotation_dict(instance_id, db))
    )


def _write_audited_gold_file(rows_dir: Path, instance_id: str, variants: list) -> None:
    task_dir = rows_dir / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(v) for v in variants)
    (task_dir / "audited_gold_variants.jsonl").write_text(
        content + "\n" if content else ""
    )


# ---------------------------------------------------------------------------
# merge_task_annotations
# ---------------------------------------------------------------------------

def test_merge_task_annotations_writes_correct_path(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_task_annotations

    downloaded = tmp_path / "downloaded"
    _write_task_annotation_file(downloaded / "rows", "shop_1", "shop")

    annotations_root = tmp_path / "annotations"
    report = merge_task_annotations(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        annotations_root=annotations_root,
    )

    dest = annotations_root / "mini_interact" / "shop" / "shop_1.task.json"
    assert dest.exists()
    assert json.loads(dest.read_text())["instance_id"] == "shop_1"
    assert report.merged == 1
    assert report.errors == 0


def test_merge_task_annotations_always_overwrites_existing(tmp_path):
    """Fetch is idempotent — always overwrites local file (skip logic was at worker)."""
    from bird_interact_agents.cloud.post_run_merge import merge_task_annotations

    downloaded = tmp_path / "downloaded"
    _write_task_annotation_file(downloaded / "rows", "shop_1", "shop")

    annotations_root = tmp_path / "annotations"
    dest = annotations_root / "mini_interact" / "shop" / "shop_1.task.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('{"stale": true}')

    merge_task_annotations(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        annotations_root=annotations_root,
    )

    data = json.loads(dest.read_text())
    assert "instance_id" in data  # replaced with fresh content
    assert "stale" not in data


def test_merge_task_annotations_normalises_dash_benchmark(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_task_annotations

    downloaded = tmp_path / "downloaded"
    _write_task_annotation_file(downloaded / "rows", "shop_1", "shop")

    annotations_root = tmp_path / "annotations"
    merge_task_annotations(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",  # dash form
        annotations_root=annotations_root,
    )

    dest = annotations_root / "mini_interact" / "shop" / "shop_1.task.json"
    assert dest.exists()


def test_merge_task_annotations_rejects_malformed_json(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_task_annotations

    downloaded = tmp_path / "downloaded"
    task_dir = downloaded / "rows" / "shop_1"
    task_dir.mkdir(parents=True)
    (task_dir / "task_annotation.json").write_text("not valid json{{{")

    annotations_root = tmp_path / "annotations"
    report = merge_task_annotations(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        annotations_root=annotations_root,
    )

    assert report.merged == 0
    assert report.errors >= 1


def test_merge_task_annotations_no_rows_dir_is_noop(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_task_annotations

    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()

    report = merge_task_annotations(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        annotations_root=tmp_path / "annotations",
    )

    assert report.merged == 0
    assert report.errors == 0


def test_merge_task_annotations_multiple_tasks(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_task_annotations

    downloaded = tmp_path / "downloaded"
    rows = downloaded / "rows"
    for iid, db in [("shop_1", "shop"), ("shop_2", "shop"), ("museum_1", "museum")]:
        _write_task_annotation_file(rows, iid, db)

    report = merge_task_annotations(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        annotations_root=tmp_path / "annotations",
    )

    assert report.merged == 3
    assert (tmp_path / "annotations" / "mini_interact" / "museum" / "museum_1.task.json").exists()


# ---------------------------------------------------------------------------
# merge_audited_gold_variants
# ---------------------------------------------------------------------------

def test_merge_audited_gold_variants_appends_to_jsonl(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    downloaded = tmp_path / "downloaded"
    _write_audited_gold_file(
        downloaded / "rows", "shop_1", [_audited_gold_variant("shop_1")]
    )

    audited_gold_root = tmp_path / "audited_gold"
    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        audited_gold_root=audited_gold_root,
    )

    consolidated = audited_gold_root / "mini_interact_audited.jsonl"
    assert consolidated.exists()
    rows = [json.loads(l) for l in consolidated.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "shop_1"
    assert report.added == 1
    assert report.skipped_duplicate == 0


def test_merge_audited_gold_variants_deduplicates_by_instance_and_variant(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    audited_gold_root = tmp_path / "audited_gold"
    audited_gold_root.mkdir()
    consolidated = audited_gold_root / "mini_interact_audited.jsonl"
    consolidated.write_text(json.dumps(_audited_gold_variant("shop_1", "primary")) + "\n")

    downloaded = tmp_path / "downloaded"
    _write_audited_gold_file(
        downloaded / "rows", "shop_1",
        [_audited_gold_variant("shop_1", "primary")],  # same key — should skip
    )

    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        audited_gold_root=audited_gold_root,
    )

    rows = [json.loads(l) for l in consolidated.read_text().splitlines() if l.strip()]
    assert len(rows) == 1  # not doubled
    assert report.added == 0
    assert report.skipped_duplicate == 1


def test_merge_audited_gold_variants_different_variant_id_not_duplicate(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    audited_gold_root = tmp_path / "audited_gold"
    audited_gold_root.mkdir()
    consolidated = audited_gold_root / "mini_interact_audited.jsonl"
    consolidated.write_text(json.dumps(_audited_gold_variant("shop_1", "primary")) + "\n")

    downloaded = tmp_path / "downloaded"
    _write_audited_gold_file(
        downloaded / "rows", "shop_1",
        [_audited_gold_variant("shop_1", "alt_reading")],  # different variant_id
    )

    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        audited_gold_root=audited_gold_root,
    )

    rows = [json.loads(l) for l in consolidated.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert report.added == 1


def test_merge_audited_gold_variants_empty_file_is_noop(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    downloaded = tmp_path / "downloaded"
    _write_audited_gold_file(downloaded / "rows", "shop_1", [])  # clean gold → empty

    audited_gold_root = tmp_path / "audited_gold"
    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        audited_gold_root=audited_gold_root,
    )

    assert report.added == 0
    assert report.errors == 0


def test_merge_audited_gold_variants_rejects_missing_required_fields(tmp_path):
    """Variants missing selected_database, benchmark, audit_status, or audited_sol_sql
    must be rejected — not silently stored (overlay/grader require these fields)."""
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    bad = {"instance_id": "shop_1", "variant_id": "primary"}  # missing everything

    downloaded = tmp_path / "downloaded"
    _write_audited_gold_file(downloaded / "rows", "shop_1", [bad])

    audited_gold_root = tmp_path / "audited_gold"
    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        audited_gold_root=audited_gold_root,
    )

    assert report.added == 0
    assert report.errors >= 1


def test_merge_audited_gold_variants_mixed_new_and_existing(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    audited_gold_root = tmp_path / "audited_gold"
    audited_gold_root.mkdir()
    consolidated = audited_gold_root / "mini_interact_audited.jsonl"
    consolidated.write_text(json.dumps(_audited_gold_variant("shop_1")) + "\n")

    downloaded = tmp_path / "downloaded"
    _write_audited_gold_file(
        downloaded / "rows", "shop_1",
        [_audited_gold_variant("shop_1")],  # duplicate
    )
    _write_audited_gold_file(
        downloaded / "rows", "shop_2",
        [_audited_gold_variant("shop_2", db="shop")],  # new
    )

    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        audited_gold_root=audited_gold_root,
    )

    rows = [json.loads(l) for l in consolidated.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert report.added == 1
    assert report.skipped_duplicate == 1


def test_merge_audited_gold_variants_no_rows_dir_is_noop(tmp_path):
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()

    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini-interact",
        audited_gold_root=tmp_path / "audited_gold",
    )

    assert report.added == 0
    assert report.errors == 0


def test_merge_audited_gold_variants_override_replaces_existing(tmp_path):
    """override=True: incoming rows replace existing rows with the same
    (instance_id, variant_id) key; the consolidated file is rewritten."""
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    audited_gold = tmp_path / "audited_gold"
    audited_gold.mkdir()
    consolidated = audited_gold / "mini_interact_audited.jsonl"
    old_row = {
        "instance_id": "db_a_1", "variant_id": "v0",
        "selected_database": "db_a", "benchmark": "mini_interact",
        "audit_status": "clean", "audited_sol_sql": ["SELECT 1"],
    }
    consolidated.write_text(json.dumps(old_row) + "\n")

    downloaded = tmp_path / "run"
    sub = downloaded / "rows" / "db_a_1"
    sub.mkdir(parents=True)
    new_row = {
        "instance_id": "db_a_1", "variant_id": "v0",
        "selected_database": "db_a", "benchmark": "mini_interact",
        "audit_status": "edited", "audited_sol_sql": ["SELECT 2"],
    }
    (sub / "audited_gold_variants.jsonl").write_text(json.dumps(new_row) + "\n")

    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini_interact",
        audited_gold_root=audited_gold,
        override=True,
    )

    assert report.added == 1
    lines = [ln for ln in consolidated.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["audit_status"] == "edited"


def test_merge_audited_gold_variants_override_purges_stale_rows_when_empty_file(tmp_path):
    """override=True: when a re-annotated task now produces an empty variants
    file (e.g. original_gold_is_correct=True), ALL existing rows for that
    instance must be removed from the consolidated file, not left stale."""
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    audited_gold = tmp_path / "audited_gold"
    audited_gold.mkdir()
    consolidated = audited_gold / "mini_interact_audited.jsonl"
    old_row = {
        "instance_id": "db_a_1", "variant_id": "v0",
        "selected_database": "db_a", "benchmark": "mini_interact",
        "audit_status": "edited", "audited_sol_sql": ["SELECT 1"],
    }
    unrelated_row = {
        "instance_id": "db_b_1", "variant_id": "v0",
        "selected_database": "db_b", "benchmark": "mini_interact",
        "audit_status": "clean", "audited_sol_sql": ["SELECT 2"],
    }
    consolidated.write_text(json.dumps(old_row) + "\n" + json.dumps(unrelated_row) + "\n")

    downloaded = tmp_path / "run"
    sub = downloaded / "rows" / "db_a_1"
    sub.mkdir(parents=True)
    # Empty variants file — re-annotation decided original is correct.
    (sub / "audited_gold_variants.jsonl").write_text("")

    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini_interact",
        audited_gold_root=audited_gold,
        override=True,
    )

    assert report.added == 0
    lines = [ln for ln in consolidated.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["instance_id"] == "db_b_1"


def test_merge_audited_gold_variants_override_preserves_rows_for_failed_tasks(tmp_path):
    """override=True: failed annotator tasks (no audited_gold_variants.jsonl)
    must NOT purge their existing consolidated rows — only dirs with a variants
    file (even empty) count as successful re-annotations to purge."""
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    audited_gold = tmp_path / "audited_gold"
    audited_gold.mkdir()
    consolidated = audited_gold / "mini_interact_audited.jsonl"
    old_row = {
        "instance_id": "db_a_1", "variant_id": "v0",
        "selected_database": "db_a", "benchmark": "mini_interact",
        "audit_status": "edited", "audited_sol_sql": ["SELECT 1"],
    }
    consolidated.write_text(json.dumps(old_row) + "\n")

    downloaded = tmp_path / "run"
    # Failed task: only attempt-1.json, no audited_gold_variants.jsonl.
    sub = downloaded / "rows" / "db_a_1"
    sub.mkdir(parents=True)
    (sub / "attempt-1.json").write_text('{"status": "error"}')

    merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini_interact",
        audited_gold_root=audited_gold,
        override=True,
    )

    lines = [ln for ln in consolidated.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1, "failed-task row must be preserved in consolidated file"
    assert json.loads(lines[0])["instance_id"] == "db_a_1"


def test_merge_audited_gold_variants_override_purges_all_rows_truncates_file(tmp_path):
    """override=True: when ALL consolidated rows belong to the re-run instance
    and the incoming variants file is empty, the consolidated file must be
    truncated rather than left stale."""
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    audited_gold = tmp_path / "audited_gold"
    audited_gold.mkdir()
    consolidated = audited_gold / "mini_interact_audited.jsonl"
    old_row = {
        "instance_id": "db_a_1", "variant_id": "v0",
        "selected_database": "db_a", "benchmark": "mini_interact",
        "audit_status": "edited", "audited_sol_sql": ["SELECT 1"],
    }
    # Only rows for the instance being re-annotated — no unrelated rows.
    consolidated.write_text(json.dumps(old_row) + "\n")

    downloaded = tmp_path / "run"
    sub = downloaded / "rows" / "db_a_1"
    sub.mkdir(parents=True)
    (sub / "audited_gold_variants.jsonl").write_text("")

    merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini_interact",
        audited_gold_root=audited_gold,
        override=True,
    )

    lines = [ln for ln in consolidated.read_text().splitlines() if ln.strip()]
    assert lines == [], "stale rows must be removed; consolidated file must be empty"


def test_merge_audited_gold_variants_override_false_does_not_replace(tmp_path):
    """override=False (default): existing rows are never replaced; the new row
    for an already-present key is skipped."""
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    audited_gold = tmp_path / "audited_gold"
    audited_gold.mkdir()
    consolidated = audited_gold / "mini_interact_audited.jsonl"
    old_row = {
        "instance_id": "db_a_1", "variant_id": "v0",
        "selected_database": "db_a", "benchmark": "mini_interact",
        "audit_status": "clean", "audited_sol_sql": ["SELECT 1"],
    }
    consolidated.write_text(json.dumps(old_row) + "\n")

    downloaded = tmp_path / "run"
    sub = downloaded / "rows" / "db_a_1"
    sub.mkdir(parents=True)
    new_row = {
        "instance_id": "db_a_1", "variant_id": "v0",
        "selected_database": "db_a", "benchmark": "mini_interact",
        "audit_status": "edited", "audited_sol_sql": ["SELECT 2"],
    }
    (sub / "audited_gold_variants.jsonl").write_text(json.dumps(new_row) + "\n")

    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini_interact",
        audited_gold_root=audited_gold,
    )

    assert report.skipped_duplicate == 1
    assert report.added == 0
    lines = [ln for ln in consolidated.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["audit_status"] == "clean"


def test_merge_audited_gold_variants_append_handles_missing_trailing_newline(tmp_path):
    """Append mode must not corrupt JSONL when the consolidated file has no
    trailing newline (e.g. hand-edited file); the new row should appear on its
    own line."""
    from bird_interact_agents.cloud.post_run_merge import merge_audited_gold_variants

    audited_gold = tmp_path / "audited_gold"
    audited_gold.mkdir()
    consolidated = audited_gold / "mini_interact_audited.jsonl"
    old_row = {
        "instance_id": "db_a_1", "variant_id": "v0",
        "selected_database": "db_a", "benchmark": "mini_interact",
        "audit_status": "clean", "audited_sol_sql": ["SELECT 1"],
    }
    # Intentionally write without trailing newline.
    consolidated.write_bytes(json.dumps(old_row).encode())

    downloaded = tmp_path / "run"
    sub = downloaded / "rows" / "db_a_2"
    sub.mkdir(parents=True)
    new_row = {
        "instance_id": "db_a_2", "variant_id": "v0",
        "selected_database": "db_a", "benchmark": "mini_interact",
        "audit_status": "clean", "audited_sol_sql": ["SELECT 2"],
    }
    (sub / "audited_gold_variants.jsonl").write_text(json.dumps(new_row) + "\n")

    report = merge_audited_gold_variants(
        downloaded_run_dir=downloaded,
        benchmark="mini_interact",
        audited_gold_root=audited_gold,
    )

    assert report.added == 1
    lines = [ln for ln in consolidated.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["instance_id"] == "db_a_1"
    assert json.loads(lines[1])["instance_id"] == "db_a_2"
