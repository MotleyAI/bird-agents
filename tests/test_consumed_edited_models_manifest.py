"""DEV-1778: per-run manifest aggregation — collect the per-task
`consumed_edited_models` records from a run dir and dedupe one-per-(db,
instance_id), first-seen wins, as a LIST (Codex #7)."""
from __future__ import annotations

import json

from bird_interact_agents.slayer_otf import edited_models as em


def _write_ann(run_dir, iid, payload):
    d = run_dir / "rows" / iid
    d.mkdir(parents=True)
    (d / "submission_annotation.json").write_text(json.dumps(payload))


def test_dedupe_first_seen_wins_and_drops_none_and_malformed():
    items = [
        {"db": "alien", "instance_id": "alien_1", "store_fp": "x"},
        {"db": "alien", "instance_id": "alien_1", "store_fp": "y"},  # dup -> dropped
        None,
        {"db": "alien"},  # malformed -> dropped
        {"db": "alien", "instance_id": "alien_2", "store_fp": "z"},
    ]
    assert em.dedupe_consumed_edited_models(items) == [
        {"db": "alien", "instance_id": "alien_1", "store_fp": "x"},
        {"db": "alien", "instance_id": "alien_2", "store_fp": "z"},
    ]


def test_collect_reads_and_dedupes_per_db_iid(tmp_path):
    _write_ann(tmp_path, "alien_1", {
        "consumed_edited_models": {"db": "alien", "instance_id": "alien_1", "store_fp": "fp1"},
    })
    _write_ann(tmp_path, "alien_2", {
        "consumed_edited_models": {"db": "alien", "instance_id": "alien_2", "store_fp": "fp2"},
    })
    out = em.collect_consumed_edited_models_from_run_dir(tmp_path)
    assert {(r["db"], r["instance_id"], r["store_fp"]) for r in out} == {
        ("alien", "alien_1", "fp1"), ("alien", "alien_2", "fp2"),
    }


def test_collect_skips_none_missing_and_unreadable(tmp_path):
    _write_ann(tmp_path, "no_field", {"instance_id": "x"})
    _write_ann(tmp_path, "null_field", {"consumed_edited_models": None})
    corrupt = tmp_path / "rows" / "corrupt"
    corrupt.mkdir(parents=True)
    (corrupt / "submission_annotation.json").write_text("{not json")
    assert em.collect_consumed_edited_models_from_run_dir(tmp_path) == []


def test_collect_dedupes_duplicate_identity_sorted_path_first_wins(tmp_path):
    """Two annotations with the SAME (db, instance_id) but different fingerprints
    under different nested paths — the sorted-first path wins deterministically."""
    for sub, fp in (("z_dir", "ZZZ"), ("a_dir", "AAA")):
        d = tmp_path / sub
        d.mkdir()
        (d / "submission_annotation.json").write_text(json.dumps({
            "consumed_edited_models": {"db": "alien", "instance_id": "alien_1", "store_fp": fp},
        }))
    out = em.collect_consumed_edited_models_from_run_dir(tmp_path)
    assert out == [{"db": "alien", "instance_id": "alien_1", "store_fp": "AAA"}]


def test_collect_is_deterministic_across_filesystem_order(tmp_path):
    """First-seen dedupe must be stable — the collector sorts candidate paths
    so the result does not depend on rglob() order."""
    _write_ann(tmp_path, "b_2", {
        "consumed_edited_models": {"db": "beta", "instance_id": "b_2", "store_fp": "s2"},
    })
    _write_ann(tmp_path, "b_1", {
        "consumed_edited_models": {"db": "beta", "instance_id": "b_1", "store_fp": "s1"},
    })
    out = em.collect_consumed_edited_models_from_run_dir(tmp_path)
    assert [r["instance_id"] for r in out] == ["b_1", "b_2"]
