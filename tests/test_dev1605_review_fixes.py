"""DEV-1605 review-round regressions (Codex + CodeRabbit).

Mechanical contracts only — no prompt-content tests.
"""

from __future__ import annotations

import pytest

from bird_interact_agents import paths
from bird_interact_agents.model_string import (
    encoder_version_slug,
    resolve_encode_version,
)


@pytest.fixture(autouse=True)
def _isolate_main_checkout_cache():
    paths._main_checkout_root_cached.cache_clear()
    yield
    paths._main_checkout_root_cached.cache_clear()


# --- encoder_version_slug rejects dot-segment / backslash slugs (CR thread 7) -


@pytest.mark.parametrize("model", ["anthropic/.", "anthropic/..", "x/.."])
def test_slug_rejects_dot_segments(model):
    with pytest.raises(ValueError):
        encoder_version_slug(model)


# --- resolve_encode_version helper (CR dedup nitpick) ------------------------


def test_resolve_encode_version_explicit_wins():
    assert resolve_encode_version("my-label", "anthropic/claude-opus-4-7") == "my-label"


def test_resolve_encode_version_falls_back_to_slug():
    assert resolve_encode_version(None, "anthropic/claude-opus-4-7") == "opus-4-7"
    assert resolve_encode_version("", "zai/glm-5.2") == "glm-5.2"


def test_resolve_encode_version_none_model_is_none():
    assert resolve_encode_version(None, None) is None
    assert resolve_encode_version(None, "") is None


def test_resolve_encode_version_propagates_bad_slug():
    with pytest.raises(ValueError):
        resolve_encode_version(None, "anthropic/claude-")


# --- _validate_slayer_setup rejects pre_encoded_version w/o otf (CR run.py) ---


def test_validate_rejects_pre_encoded_version_without_otf():
    from bird_interact_agents.run import _validate_slayer_setup

    with pytest.raises(ValueError, match="requires --pre-encoded-models otf"):
        _validate_slayer_setup(
            slayer_setup="pre-encoded", framework="claude_sdk_otf",
            query_mode="slayer", mode="a-interact",
            pre_encoded_source="custom", pre_encoded_version="opus-4-7",
        )


def test_validate_rejects_pre_encoded_version_on_the_fly():
    from bird_interact_agents.run import _validate_slayer_setup

    with pytest.raises(ValueError, match="requires --pre-encoded-models otf"):
        _validate_slayer_setup(
            slayer_setup="on-the-fly", framework="claude_sdk_otf",
            query_mode="slayer", mode="a-interact",
            pre_encoded_source=None, pre_encoded_version="opus-4-7",
        )


def test_validate_allows_pre_encoded_version_with_otf():
    from bird_interact_agents.run import _validate_slayer_setup

    # Must not raise.
    _validate_slayer_setup(
        slayer_setup="pre-encoded", framework="claude_sdk_otf",
        query_mode="slayer", mode="a-interact",
        pre_encoded_source="otf", pre_encoded_version="opus-4-7",
    )


# --- _read_consumed_reference tolerates malformed _encoder_meta.json (CR 4) ---


def test_read_consumed_reference_tolerates_malformed_meta(tmp_path):
    from bird_interact_agents.agents._pre_encoded import _read_consumed_reference

    db_dir = tmp_path / "alien"
    db_dir.mkdir()
    (db_dir / "_reference_fp.txt").write_text("fp1")
    # a JSON LIST (not an object) — must not crash
    (db_dir / "_encoder_meta.json").write_text("[1, 2, 3]")
    cr = _read_consumed_reference(db_dir, db="alien", version="v1")
    assert cr.encoder_model == "unknown"
    assert cr.reference_fp == "fp1"


def test_read_consumed_reference_null_model(tmp_path):
    from bird_interact_agents.agents._pre_encoded import _read_consumed_reference

    db_dir = tmp_path / "alien"
    db_dir.mkdir()
    (db_dir / "_reference_fp.txt").write_text("fp1")
    (db_dir / "_encoder_meta.json").write_text('{"encoder_model": null}')
    cr = _read_consumed_reference(db_dir, db="alien", version="v1")
    assert cr.encoder_model == "unknown"


# --- pydantic_ai_otf_encode builds into the VERSIONED dir (Codex major) ------


async def test_otf_encode_resolver_builds_versioned_with_provenance(tmp_path, monkeypatch):
    """`_resolve_otf_task_storage_dir` must build into
    slayer_models_otf/<benchmark>/<version>/<db> and forward the encoder
    identity into ensure_db_reference (so cloud upload-back/merge find it)."""
    import bird_interact_agents.agents.pydantic_ai_otf_encode.agent as agent_mod

    captured: dict = {}

    async def fake_ensure(db, *, reference_root, **kwargs):
        captured["reference_root"] = reference_root
        captured.update(kwargs)

    async def fake_variant(**kwargs):
        return tmp_path / "variant" / "alien"

    monkeypatch.setattr(agent_mod, "ensure_db_reference", fake_ensure)
    monkeypatch.setattr(agent_mod, "build_task_variant_storage", fake_variant)
    monkeypatch.setattr(
        agent_mod._paths, "slayer_models_otf_root",
        lambda *, benchmark, version=None: tmp_path / "models_otf" / benchmark / (version or "_nov"),
    )
    monkeypatch.setattr(
        agent_mod._paths, "slayer_otf_cache_root",
        lambda *, benchmark: tmp_path / "cache",
    )

    await agent_mod._resolve_otf_task_storage_dir(
        db_name="alien",
        task_data={"instance_id": "alien_1"},
        data_path_base=str(tmp_path / "data"),
        build_encoder=lambda *a, **k: None,
        benchmark="mini-interact",
        version="opus-4-7",
        encoder_model="anthropic/claude-opus-4-7",
        encoder_framework="pydantic_ai",
    )
    assert captured["reference_root"].name == "opus-4-7"
    assert captured["version"] == "opus-4-7"
    assert captured["encoder_model"] == "anthropic/claude-opus-4-7"
    assert captured["encoder_framework"] == "pydantic_ai"
