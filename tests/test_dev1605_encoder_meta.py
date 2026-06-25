"""DEV-1605: the reference build writes a self-describing
``_encoder_meta.json`` next to ``_reference_fp.txt`` recording which encoder
(model + framework + version + settings) produced the reference.

Mirrors the ``test_slayer_otf_reference_build`` fixture style: the slayer
ingest subprocess is monkeypatched away and the encoder is injected via the
``build_encoder`` seam, so no real LLM/MCP runs.
"""

from __future__ import annotations

import datetime as _dt
import json

import pytest

from tests.test_slayer_otf_reference_build import _encoded_build_encoder, _kb_rows

DB = "fakedb"


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.slayer_otf.cache import CacheEntry

    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    cache_dir = tmp_path / "cache" / DB
    (cache_dir / "datasources").mkdir(parents=True)
    (cache_dir / "models" / DB).mkdir(parents=True)
    abs_sqlite = tmp_path / "mini-interact" / DB / f"{DB}.sqlite"
    (cache_dir / "datasources" / f"{DB}.yaml").write_text(
        f"name: {DB}\ntype: sqlite\nconnection_string: sqlite:///{abs_sqlite}\n"
    )
    (cache_dir / "models" / DB / "households.yaml").write_text(
        "name: households\ndata_source: %s\nsql_table: households\n"
        "columns:\n  - name: id\n    primary_key: true\n  - name: income\n"
        "measures: []\naggregations: []\njoins: []\n" % DB
    )
    rows = _kb_rows()
    (cache_dir / "_kb_rows.json").write_text(json.dumps(rows))
    entry = CacheEntry(cache_dir=cache_dir, fingerprint="fp_abc123", kb_rows=rows)

    async def fake_ensure_db_cache(
        db, *, cache_root, mini_interact_root, force=False, benchmark=None,
    ):
        return entry

    monkeypatch.setattr(reference_build, "ensure_db_cache", fake_ensure_db_cache)
    return entry


async def _build_with_meta(tmp_path, *, version, model, framework, explicit):
    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.slayer_otf.encoder_types import EncoderMetaSettings

    return await reference_build.ensure_db_reference(
        DB,
        # Caller bakes the version into the reference_root (version ABOVE db).
        reference_root=tmp_path / "slayer_models_otf" / "mini-interact" / version,
        cache_root=tmp_path / "cache",
        mini_interact_root=tmp_path / "mini-interact",
        build_encoder=_encoded_build_encoder([]),
        encoder_model=model,
        encoder_framework=framework,
        version=version,
        benchmark="mini-interact",
        encoder_settings=EncoderMetaSettings(
            reasoning_effort="high", version_was_explicit=explicit,
        ),
    )


def _meta_path(tmp_path, version):
    return (
        tmp_path / "slayer_models_otf" / "mini-interact" / version / DB
        / "_encoder_meta.json"
    )


async def test_encoder_meta_written_into_versioned_dir(fake_cache, tmp_path):
    await _build_with_meta(
        tmp_path, version="opus-4-7",
        model="anthropic/claude-opus-4-7", framework="claude_sdk", explicit=False,
    )
    meta_fp = _meta_path(tmp_path, "opus-4-7")
    marker = meta_fp.parent / "_reference_fp.txt"
    assert marker.exists(), "marker must exist"
    assert meta_fp.exists(), "_encoder_meta.json must sit next to the marker"

    from bird_interact_agents.slayer_otf.encoder_types import EncoderMeta

    meta = EncoderMeta.model_validate_json(meta_fp.read_text())
    assert meta.version == "opus-4-7"
    assert meta.encoder_model == "anthropic/claude-opus-4-7"
    assert meta.encoder_framework == "claude_sdk"
    assert meta.benchmark == "mini-interact"
    assert meta.db == DB
    # The recorded fp must equal the marker content (consistency / written-with).
    assert meta.reference_fp == marker.read_text().strip()
    # built_at parses as ISO-8601.
    _dt.datetime.fromisoformat(meta.built_at)
    assert meta.settings.version_was_explicit is False
    assert meta.settings.reasoning_effort == "high"
    # schema_version present for forward-compat.
    assert meta.schema_version == 1


async def test_encoder_meta_carries_model_settings_json(fake_cache, tmp_path):
    from bird_interact_agents.slayer_otf import reference_build
    from bird_interact_agents.slayer_otf.encoder_types import (
        EncoderMeta, EncoderMetaSettings,
    )

    await reference_build.ensure_db_reference(
        DB,
        reference_root=tmp_path / "slayer_models_otf" / "mini-interact" / "opus-4-7",
        cache_root=tmp_path / "cache",
        mini_interact_root=tmp_path / "mini-interact",
        build_encoder=_encoded_build_encoder([]),
        encoder_model="anthropic/claude-opus-4-7",
        encoder_framework="claude_sdk",
        version="opus-4-7",
        benchmark="mini-interact",
        encoder_settings=EncoderMetaSettings(
            model_settings_json='{"temperature": 0.0}',
        ),
    )
    meta = EncoderMeta.model_validate_json(_meta_path(tmp_path, "opus-4-7").read_text())
    assert meta.settings.model_settings_json == '{"temperature": 0.0}'


async def test_encoder_meta_records_explicit_version_flag(fake_cache, tmp_path):
    await _build_with_meta(
        tmp_path, version="my-label",
        model="zai/glm-5.2", framework="claude_sdk", explicit=True,
    )
    from bird_interact_agents.slayer_otf.encoder_types import EncoderMeta

    meta = EncoderMeta.model_validate_json(_meta_path(tmp_path, "my-label").read_text())
    assert meta.version == "my-label"
    assert meta.encoder_model == "zai/glm-5.2"
    assert meta.settings.version_was_explicit is True


async def test_meta_is_optional_back_compat(fake_cache, tmp_path):
    """Callers that don't pass encoder_model (legacy/tests) still build, just
    without the meta sidecar."""
    from bird_interact_agents.slayer_otf import reference_build

    await reference_build.ensure_db_reference(
        DB,
        reference_root=tmp_path / "slayer_models_otf",
        cache_root=tmp_path / "cache",
        mini_interact_root=tmp_path / "mini-interact",
        build_encoder=_encoded_build_encoder([]),
    )
    ref = tmp_path / "slayer_models_otf" / DB
    assert (ref / "_reference_fp.txt").exists()
    assert not (ref / "_encoder_meta.json").exists()
