"""DEV-1672 Cause 2: the deterministic OTF cache fingerprint must carry a
phase-3 behaviour token so that flipping the raw-JSON-column hiding behaviour
invalidates already-built ``slayer_otf_cache/<db>`` and forces a rebuild.

``fingerprint_of`` (the FULL ``_cache_fp.txt``) hashes only slayer version +
input content, NOT bird-agents code — so a phase-3 code change alone would not
invalidate an existing cache. We add the token to the IMPL half
(``_impl_fingerprint_of``), which ``ensure_db_cache`` recomputes and compares
against ``_impl_fp.txt`` on the reuse path. The FULL fingerprint is left
UNCHANGED so saved edited-model archives (stamped with the full fp) stay valid.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from bird_interact_agents.slayer_otf import cache

DB = "testdb"


def test_impl_fingerprint_includes_phase3_token(monkeypatch) -> None:
    """The phase-3 behaviour token is a component of the impl fingerprint:
    changing it changes the fingerprint (so caches rebuild)."""
    baseline = cache._impl_fingerprint_of()
    monkeypatch.setattr(cache, "_PHASE3_IMPL_TOKEN", "sentinel-different-value")
    assert cache._impl_fingerprint_of() != baseline


def test_phase3_token_is_a_nonempty_str() -> None:
    assert isinstance(cache._PHASE3_IMPL_TOKEN, str)
    assert cache._PHASE3_IMPL_TOKEN


def _build_data_root(tmp_path: Path) -> Path:
    """Minimal data root fingerprint_of needs: <db>/{sqlite,meanings,kb}."""
    root = tmp_path / "data"
    (root / DB).mkdir(parents=True)
    conn = sqlite3.connect(str(root / DB / f"{DB}.sqlite"))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()
    (root / DB / f"{DB}_column_meaning_base.json").write_text("{}", encoding="utf-8")
    (root / DB / f"{DB}_kb.jsonl").write_text("", encoding="utf-8")
    return root


def test_full_fingerprint_unchanged_by_phase3_token(tmp_path, monkeypatch) -> None:
    """The FULL cache fingerprint (``_cache_fp.txt``, stamped into saved
    edited-model archives) must NOT depend on the phase-3 token — otherwise
    bumping the token would invalidate every saved archive on apply. The token
    belongs to the IMPL half only."""
    root = _build_data_root(tmp_path)
    before = cache.fingerprint_of(db_name=DB, data_root=root)
    monkeypatch.setattr(cache, "_PHASE3_IMPL_TOKEN", "sentinel-different-value")
    after = cache.fingerprint_of(db_name=DB, data_root=root)
    assert before == after
