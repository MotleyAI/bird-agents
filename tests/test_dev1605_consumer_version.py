"""DEV-1605: consumer-side version resolution for ``--pre-encoded-models otf``.

``resolve_otf_version`` scans ``slayer_models_otf/<benchmark>/<version>/<db>/``
for a ``_reference_fp.txt`` marker and picks the version:

* exactly one version present for THIS db -> use it;
* an explicit ``requested`` version -> must exist (for this db) else error;
* 2+ versions present and none requested -> error listing them;
* 0 versions present for this db -> error.

There is NO legacy flat ``<db>/`` fallback (abolished in DEV-1605): a flat
``<benchmark>/<db>/_reference_fp.txt`` is NOT a valid reference anymore.
"""

from __future__ import annotations

import pytest

from bird_interact_agents import paths
from bird_interact_agents.agents._pre_encoded import (
    PreEncodedSetupError,
    resolve_otf_version,
)
from tests.test_paths import _setup_main_and_worktree

BENCH = "mini-interact"


@pytest.fixture(autouse=True)
def _isolate_main_checkout_cache():
    """Clear the memoised main-checkout resolution before+after each test."""
    paths._main_checkout_root_cached.cache_clear()
    yield
    paths._main_checkout_root_cached.cache_clear()


def _make_version(main, version, db, *, fp="fp"):
    d = main / "slayer_models_otf" / BENCH / version / db
    d.mkdir(parents=True, exist_ok=True)
    (d / "_reference_fp.txt").write_text(fp)
    # a non-empty embeddings.db so downstream presence checks pass
    (d / "embeddings.db").write_bytes(b"x")
    return d


def test_single_version_is_used(tmp_path, monkeypatch):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _make_version(main, "opus-4-7", "alien")
    assert resolve_otf_version(benchmark=BENCH, db="alien", requested=None) == "opus-4-7"


def test_explicit_version_used_when_present(tmp_path, monkeypatch):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _make_version(main, "opus-4-7", "alien")
    _make_version(main, "glm-5.2", "alien")
    assert (
        resolve_otf_version(benchmark=BENCH, db="alien", requested="glm-5.2")
        == "glm-5.2"
    )


def test_explicit_version_missing_raises(tmp_path, monkeypatch):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _make_version(main, "opus-4-7", "alien")
    with pytest.raises(PreEncodedSetupError) as ei:
        resolve_otf_version(benchmark=BENCH, db="alien", requested="does-not-exist")
    # error should name the available versions
    assert "opus-4-7" in str(ei.value)


def test_ambiguous_without_request_raises(tmp_path, monkeypatch):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _make_version(main, "opus-4-7", "alien")
    _make_version(main, "glm-5.2", "alien")
    with pytest.raises(PreEncodedSetupError) as ei:
        resolve_otf_version(benchmark=BENCH, db="alien", requested=None)
    msg = str(ei.value)
    assert "opus-4-7" in msg and "glm-5.2" in msg


def test_no_version_for_db_raises(tmp_path, monkeypatch):
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    # a version exists, but only for a DIFFERENT db
    _make_version(main, "opus-4-7", "other_db")
    with pytest.raises(PreEncodedSetupError):
        resolve_otf_version(benchmark=BENCH, db="alien", requested=None)


def test_requested_version_present_but_db_absent_raises(tmp_path, monkeypatch):
    """Codex MED#5: 'present' means the version dir contains THIS db with a
    marker, not merely that the version dir exists for some other db."""
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    _make_version(main, "opus-4-7", "other_db")
    with pytest.raises(PreEncodedSetupError):
        resolve_otf_version(benchmark=BENCH, db="alien", requested="opus-4-7")


def test_flat_layout_is_not_a_fallback(tmp_path, monkeypatch):
    """A legacy flat <benchmark>/<db>/_reference_fp.txt is no longer valid."""
    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    flat = main / "slayer_models_otf" / BENCH / "alien"
    flat.mkdir(parents=True)
    (flat / "_reference_fp.txt").write_text("fp")
    (flat / "embeddings.db").write_bytes(b"x")
    with pytest.raises(PreEncodedSetupError):
        resolve_otf_version(benchmark=BENCH, db="alien", requested=None)


def test_pre_encoded_source_root_accepts_version(tmp_path, monkeypatch):
    from bird_interact_agents.agents._pre_encoded import pre_encoded_source_root

    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    assert (
        pre_encoded_source_root("otf", benchmark=BENCH, version="opus-4-7")
        == paths.slayer_models_otf_root(benchmark=BENCH, version="opus-4-7")
    )


async def test_resolve_pre_encoded_storage_dir_returns_consumed(tmp_path, monkeypatch):
    """The resolver must return the consumed reference (db/version/
    encoder_model/reference_fp) read from the chosen version dir, alongside
    the per-task storage dir + deleted kb ids."""
    import bird_interact_agents.agents._pre_encoded as pe
    from bird_interact_agents.slayer_otf.encoder_types import (
        ConsumedReference, EncoderMeta, EncoderMetaSettings,
    )

    main, _ = _setup_main_and_worktree(tmp_path, monkeypatch)
    db_dir = _make_version(main, "opus-4-7", "alien", fp="fp-xyz")
    (db_dir / "_encoder_meta.json").write_text(
        EncoderMeta(
            version="opus-4-7", encoder_model="anthropic/claude-opus-4-7",
            encoder_framework="claude_sdk", benchmark=BENCH, db="alien",
            reference_fp="fp-xyz", built_at="2026-06-25T00:00:00+00:00",
            settings=EncoderMetaSettings(),
        ).model_dump_json()
    )

    async def fake_build(**kwargs):
        return tmp_path / "variant" / "alien"

    monkeypatch.setattr(pe, "build_task_variant_storage", fake_build)

    result = await pe.resolve_pre_encoded_storage_dir(
        db_name="alien",
        task_data={"instance_id": "alien_1"},
        data_path_base=str(tmp_path / "mini-interact"),
        benchmark=BENCH,
        source="otf",
        version="opus-4-7",
    )
    # storage dir + deleted ids still available (back-compat first two values)
    assert str(result.storage_dir) == str(tmp_path / "variant" / "alien")
    assert result.deleted_kb_ids == []
    assert result.consumed == ConsumedReference(
        db="alien", version="opus-4-7",
        encoder_model="anthropic/claude-opus-4-7", reference_fp="fp-xyz",
    )
