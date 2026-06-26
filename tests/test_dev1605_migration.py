"""DEV-1605: one-shot migration of legacy FLAT references
(``slayer_models_otf/<benchmark>/<db>/``) into the versioned layout
(``slayer_models_otf/<benchmark>/<version>/<db>/``).

The version label is derived from each flat ref's ``_setup_usage.json``
``setup_encoder::<model>`` breakdown; when that is absent/underivable the dir
is moved under ``unknown/`` with a warning. Idempotent + dry-run supported.
"""

from __future__ import annotations

import json

import pytest

from bird_interact_agents import paths
from tests.test_paths import _setup_main_and_worktree

BENCH = "mini-interact"


@pytest.fixture(autouse=True)
def _isolate_main_checkout_cache():
    """Clear the memoised main-checkout resolution before+after each test."""
    paths._main_checkout_root_cached.cache_clear()
    yield
    paths._main_checkout_root_cached.cache_clear()

# Imported lazily inside tests so collection still works before the script
# module exists (TDD: it fails for the right reason — ImportError/AttributeError
# surfaced as a test failure, not a collection error that hides the rest).


def _migrate():
    import importlib

    return importlib.import_module("scripts.migrate_otf_references")


def _flat_db(main, db, *, model=None, fp="fp"):
    d = main / "slayer_models_otf" / BENCH / db
    d.mkdir(parents=True, exist_ok=True)
    (d / "_reference_fp.txt").write_text(fp)
    (d / "embeddings.db").write_bytes(b"x")
    if model is not None:
        usage = {
            "n_calls": 1,
            "breakdown": [
                {"name": f"setup_encoder::{model}", "scope": "setup_encoder",
                 "model": model, "n_calls": 1},
            ],
        }
        (d / "_setup_usage.json").write_text(json.dumps(usage))
    return d


def test_migrate_derives_version_from_setup_usage(tmp_path, monkeypatch):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _flat_db(main, "alien", model="anthropic/claude-opus-4-7")

    _migrate().migrate_benchmark(benchmark=BENCH)

    moved = main / "slayer_models_otf" / BENCH / "opus-4-7" / "alien"
    assert (moved / "_reference_fp.txt").exists()
    # the flat dir no longer holds a direct marker
    assert not (main / "slayer_models_otf" / BENCH / "alien" / "_reference_fp.txt").exists()


def test_migrate_backfills_encoder_meta(tmp_path, monkeypatch):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _flat_db(main, "alien", model="zai/glm-5.2", fp="deadbeef")

    _migrate().migrate_benchmark(benchmark=BENCH)

    from bird_interact_agents.slayer_otf.encoder_types import EncoderMeta

    meta_fp = (
        main / "slayer_models_otf" / BENCH / "glm-5.2" / "alien" / "_encoder_meta.json"
    )
    assert meta_fp.exists()
    meta = EncoderMeta.model_validate_json(meta_fp.read_text())
    assert meta.version == "glm-5.2"
    assert meta.encoder_model == "zai/glm-5.2"
    assert meta.reference_fp == "deadbeef"


def test_migrate_unknown_when_no_usage(tmp_path, monkeypatch, capsys):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _flat_db(main, "alien", model=None)  # no _setup_usage.json

    _migrate().migrate_benchmark(benchmark=BENCH)

    assert (
        main / "slayer_models_otf" / BENCH / "unknown" / "alien" / "_reference_fp.txt"
    ).exists()
    # a warning was emitted naming the db
    captured = capsys.readouterr()
    assert "alien" in (captured.out + captured.err)


def test_migrate_is_idempotent(tmp_path, monkeypatch):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _flat_db(main, "alien", model="anthropic/claude-opus-4-7")

    mig = _migrate()
    mig.migrate_benchmark(benchmark=BENCH)
    # second run must be a no-op (no flat dirs left; version dirs untouched)
    report = mig.migrate_benchmark(benchmark=BENCH)
    assert report.moves == []
    moved = main / "slayer_models_otf" / BENCH / "opus-4-7" / "alien"
    assert (moved / "_reference_fp.txt").exists()


def test_migrate_dry_run_moves_nothing(tmp_path, monkeypatch):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _flat_db(main, "alien", model="anthropic/claude-opus-4-7")

    report = _migrate().migrate_benchmark(benchmark=BENCH, dry_run=True)

    # still flat, nothing moved
    assert (main / "slayer_models_otf" / BENCH / "alien" / "_reference_fp.txt").exists()
    assert not (main / "slayer_models_otf" / BENCH / "opus-4-7").exists()
    # but the report describes the intended move
    assert any(m.db == "alien" and m.version == "opus-4-7" for m in report.moves)


def test_migrate_malformed_usage_json_falls_back_unknown(tmp_path, monkeypatch):
    """A valid-but-wrong-shaped _setup_usage.json (a JSON list, or non-dict
    breakdown rows) must NOT crash — it falls back to 'unknown' (CodeRabbit)."""
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    d = _flat_db(main, "alien", model=None)
    # overwrite with a malformed (list) usage doc
    (d / "_setup_usage.json").write_text("[1, 2, 3]")

    _migrate().migrate_benchmark(benchmark=BENCH)
    assert (
        main / "slayer_models_otf" / BENCH / "unknown" / "alien" / "_reference_fp.txt"
    ).exists()


def test_migrate_non_string_model_falls_back_unknown(tmp_path, monkeypatch):
    """A setup_encoder row with a non-string `model` (e.g. an int) must fall
    back to 'unknown' rather than crash encoder_version_slug (Codex)."""
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    d = main / "slayer_models_otf" / BENCH / "alien"
    d.mkdir(parents=True)
    (d / "_reference_fp.txt").write_text("fp")
    (d / "embeddings.db").write_bytes(b"x")
    (d / "_setup_usage.json").write_text(json.dumps({
        "breakdown": [{"scope": "setup_encoder", "model": 123}],
    }))

    _migrate().migrate_benchmark(benchmark=BENCH)
    assert (
        main / "slayer_models_otf" / BENCH / "unknown" / "alien" / "_reference_fp.txt"
    ).exists()


def test_migrate_dry_run_reports_collision_skip(tmp_path, monkeypatch):
    """Dry-run must honor a destination collision the same way the real run
    does — it should NOT report a move whose target already exists."""
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _flat_db(main, "alien", model="anthropic/claude-opus-4-7")
    # pre-create the destination so the move would collide
    dest = main / "slayer_models_otf" / BENCH / "opus-4-7" / "alien"
    dest.mkdir(parents=True)
    (dest / "_reference_fp.txt").write_text("existing")

    report = _migrate().migrate_benchmark(benchmark=BENCH, dry_run=True)
    assert report.moves == []  # collision → not reported as a move
    # flat dir untouched
    assert (main / "slayer_models_otf" / BENCH / "alien" / "_reference_fp.txt").exists()


def test_migrate_skips_already_versioned(tmp_path, monkeypatch):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    # already versioned: <bench>/opus-4-7/alien
    d = main / "slayer_models_otf" / BENCH / "opus-4-7" / "alien"
    d.mkdir(parents=True)
    (d / "_reference_fp.txt").write_text("fp")

    report = _migrate().migrate_benchmark(benchmark=BENCH)
    assert report.moves == []
    assert (d / "_reference_fp.txt").exists()
