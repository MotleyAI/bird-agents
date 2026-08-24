"""DEV-1778: the single finalize hook stamps `row["consumed_edited_models"]`
from the resolver's apply-time stash (fingerprint + identity)."""
from __future__ import annotations

import bird_interact_agents.agents._edited_models_hook as hook


def _isolate(monkeypatch):
    """Neutralise the hook's two collaborators so only its own stamping runs."""
    monkeypatch.setattr(hook, "finalize_result_row", lambda row, **_k: row)
    monkeypatch.setattr(
        hook._edited_models, "maybe_save_edited_models", lambda *_a, **_k: None,
    )


def _call(row, td):
    return hook.finalize_with_edited_models_save(
        row, deleted_kb_ids=[], slayer_storage_dir="/tmp/scratch/alien",
        benchmark="mini-interact", save_edited_models=False, task_data=td,
    )


def test_stamps_consumed_when_applied_and_fingerprinted(monkeypatch):
    _isolate(monkeypatch)
    out = _call(
        {"database": "alien", "instance_id": "alien_1"},
        {
            "_edited_models_applied_from": "/runs/x/edited_models.tar.gz",
            "_edited_models_consumed_store_fp": "fp123",
        },
    )
    assert out["edited_models_applied_from"] == "/runs/x/edited_models.tar.gz"
    assert out["consumed_edited_models"] == {
        "db": "alien", "instance_id": "alien_1", "store_fp": "fp123",
    }


def test_no_consumed_stamp_when_not_applied(monkeypatch):
    _isolate(monkeypatch)
    out = _call({"database": "alien", "instance_id": "alien_1"}, {})
    assert "consumed_edited_models" not in out


def test_no_consumed_stamp_when_fingerprint_missing(monkeypatch):
    """Apply succeeded but the fingerprint could not be computed — record the
    path (applied_from) but omit the unattributable consumed record."""
    _isolate(monkeypatch)
    out = _call(
        {"database": "alien", "instance_id": "alien_1"},
        {"_edited_models_applied_from": "/runs/x/edited_models.tar.gz"},
    )
    assert out["edited_models_applied_from"] == "/runs/x/edited_models.tar.gz"
    assert "consumed_edited_models" not in out


def test_no_stamp_when_db_missing_everywhere(monkeypatch):
    _isolate(monkeypatch)
    out = _call(
        {"instance_id": "alien_1"},  # no database, td has no selected_database
        {
            "_edited_models_applied_from": "/runs/x/edited_models.tar.gz",
            "_edited_models_consumed_store_fp": "fp123",
            "instance_id": "alien_1",
        },
    )
    assert "consumed_edited_models" not in out


def test_no_stamp_when_instance_missing_everywhere(monkeypatch):
    _isolate(monkeypatch)
    out = _call(
        {"database": "alien"},  # no instance_id, td has no instance_id
        {
            "_edited_models_applied_from": "/runs/x/edited_models.tar.gz",
            "_edited_models_consumed_store_fp": "fp123",
            "selected_database": "alien",
        },
    )
    assert "consumed_edited_models" not in out


def test_identity_falls_back_to_task_data(monkeypatch):
    """Row lacks database/instance_id — fall back to task_data so the record
    is never built with None fields (Codex #9)."""
    _isolate(monkeypatch)
    out = _call(
        {},
        {
            "_edited_models_applied_from": "/runs/x/edited_models.tar.gz",
            "_edited_models_consumed_store_fp": "fp123",
            "selected_database": "alien",
            "instance_id": "alien_1",
        },
    )
    assert out["consumed_edited_models"] == {
        "db": "alien", "instance_id": "alien_1", "store_fp": "fp123",
    }
