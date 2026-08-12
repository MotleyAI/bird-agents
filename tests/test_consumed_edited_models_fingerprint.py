"""DEV-1778: `store_content_fingerprint` — the content-manifest digest that
identifies which saved edited-models store state a run consumed."""
from __future__ import annotations

from bird_interact_agents.slayer_otf import edited_models as em
from tests._edited_models_fixtures import edit_a_model, make_fake_store


def test_fingerprint_is_hex_sha256(tmp_path):
    fp = em.store_content_fingerprint(make_fake_store(tmp_path / "a"))
    assert isinstance(fp, str)
    assert len(fp) == 64
    assert all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_deterministic_for_identical_content(tmp_path):
    a = make_fake_store(tmp_path / "a")
    b = make_fake_store(tmp_path / "b")
    assert em.store_content_fingerprint(a) == em.store_content_fingerprint(b)


def test_fingerprint_changes_with_content(tmp_path):
    a = make_fake_store(tmp_path / "a")
    fp0 = em.store_content_fingerprint(a)
    edit_a_model(a)
    assert em.store_content_fingerprint(a) != fp0


def test_fingerprint_ignores_transient_sqlite_sidecars(tmp_path):
    a = make_fake_store(tmp_path / "a")
    fp0 = em.store_content_fingerprint(a)
    (a / "embeddings.db-wal").write_bytes(b"transient")
    (a / "embeddings.db-shm").write_bytes(b"transient")
    assert em.store_content_fingerprint(a) == fp0


def test_fingerprint_ignores_meta_and_baseline(tmp_path):
    """The store-meta + baseline manifest are apply-time bookkeeping, not
    consumed definitions — content_manifest excludes them, so the digest must
    too."""
    a = make_fake_store(tmp_path / "a")
    fp0 = em.store_content_fingerprint(a)
    (a / em._STORE_META).write_text('{"benchmark": "x"}')
    (a / em._BASELINE_MANIFEST).write_text("{}")
    assert em.store_content_fingerprint(a) == fp0


def test_fingerprint_reflects_saved_datasource_anchor(tmp_path):
    """Documented semantic (Codex #2): store_fp identifies the persisted store
    content AS SAVED, including the saving machine's datasource anchor. Two
    otherwise-identical stores saved under different connection strings hash
    differently — attribution is to the exact archive, not a canonical form."""
    a = make_fake_store(tmp_path / "a", conn_string="sqlite:////m1/alien/alien.sqlite")
    b = make_fake_store(tmp_path / "b", conn_string="sqlite:////m2/alien/alien.sqlite")
    assert em.store_content_fingerprint(a) != em.store_content_fingerprint(b)
