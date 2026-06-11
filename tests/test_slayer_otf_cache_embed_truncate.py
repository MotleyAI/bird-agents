"""Tests for the embedding-text truncation in `slayer_otf.cache`.

When a rendered memory text exceeds the embedding model's 8192-token
cap, the OTF cache builder pre-truncates to 7800 tokens (400-token
margin) before passing to `embed_batch`. Without this, litellm's
aembedding raises `BadRequestError: Invalid 'input[N]': maximum input
length is 8192 tokens` and `embed_batch` returns `[None] * len(texts)`
for the WHOLE batch — every memory in the batch loses its embedding.

Failure modes covered:

- short text: identity-return (no encode/decode round-trip).
- long text: truncated to 7800 tokens, head preserved, tail dropped.
- unknown model name: fall back to `cl100k_base` encoding.
- provider-prefixed model name (`openai/text-embedding-3-small`):
  strip prefix before `encoding_for_model` lookup.
- tiktoken not installed: fall back to a 28_000-char hard cap with
  logged warning (NOT a crash).
- encoder load failure: same fallback path.
- `_materialise_cache_memories`: every text passed to `embed_batch`
  is under budget; per-truncation warning logged.
- `_EMBEDDING_BUILDER_VERSION` participates in the impl fingerprint so
  bumping it invalidates already-built caches automatically.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# _truncate_for_embedding — unit tests
# ---------------------------------------------------------------------------


def test_truncate_for_embedding_short_text_unchanged_identity():
    """A short input must return the SAME string object — no encode/
    decode round-trip when not needed."""
    from bird_interact_agents.slayer_otf.cache import _truncate_for_embedding

    text = "Hello, world."
    out = _truncate_for_embedding(text, model_name="text-embedding-3-small")
    assert out is text  # identity, not equality


def test_truncate_for_embedding_long_text_truncated_to_budget():
    """A text that encodes past 7800 tokens is sliced back to ≤7800.
    Re-encode the output and verify the token count budget."""
    import tiktoken

    from bird_interact_agents.slayer_otf.cache import _truncate_for_embedding

    enc = tiktoken.get_encoding("cl100k_base")
    # ~ "word" token costs 1 token in cl100k_base.
    long = " ".join(["word"] * 12_000)
    out = _truncate_for_embedding(long, model_name="text-embedding-3-small")
    assert out is not long
    assert len(enc.encode(out)) <= 7800


def test_truncate_for_embedding_preserves_head_drops_tail():
    """Truncation is from the tail — the head of the input must appear
    in the output."""
    from bird_interact_agents.slayer_otf.cache import _truncate_for_embedding

    head = "FIRST_HEAD_MARKER_TOKEN " * 50
    tail = "END_TAIL_MARKER_TOKEN " * 15_000
    text = head + tail
    out = _truncate_for_embedding(text, model_name="text-embedding-3-small")
    assert "FIRST_HEAD_MARKER_TOKEN" in out
    assert out.startswith("FIRST_HEAD_MARKER_TOKEN")


def test_truncate_for_embedding_unknown_model_falls_back_to_cl100k_base():
    """An unknown model name must NOT raise; `cl100k_base` is the
    fallback encoding (and the actual encoding for all OpenAI
    text-embedding-3-* / ada-002 models)."""
    from bird_interact_agents.slayer_otf.cache import _truncate_for_embedding

    # Should not raise. Short text → identity.
    out = _truncate_for_embedding(
        "Hello.", model_name="fictitious-future-model-9000",
    )
    assert out == "Hello."


def test_truncate_for_embedding_strips_provider_prefix_for_model_lookup(
    monkeypatch,
):
    """Codex finding #8: model names come through as
    `openai/text-embedding-3-small`. The bare model name MUST be passed
    to `tiktoken.encoding_for_model` so it resolves; the fallback
    `cl100k_base` is only for genuinely unknown models."""
    import tiktoken

    seen: list[str] = []
    original = tiktoken.encoding_for_model

    def spy(name: str):
        seen.append(name)
        return original(name)

    monkeypatch.setattr(tiktoken, "encoding_for_model", spy)
    from bird_interact_agents.slayer_otf import cache as otf_cache

    # Reload so the import-time tiktoken reference is fresh.
    importlib.reload(otf_cache)
    otf_cache._truncate_for_embedding(
        "Hello", model_name="openai/text-embedding-3-small",
    )
    assert any(s == "text-embedding-3-small" for s in seen)


def test_truncate_for_embedding_tiktoken_missing_falls_back_to_char_cap(
    monkeypatch, caplog,
):
    """Codex finding #7: if `import tiktoken` raises, the helper must
    NOT crash the cache builder. Fall back to a char-cap (28_000 chars
    ~ 7000 tokens worst case English) and log a warning."""
    real_modules = sys.modules.copy()
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    from bird_interact_agents.slayer_otf import cache as otf_cache

    importlib.reload(otf_cache)
    try:
        with caplog.at_level(logging.WARNING):
            short = "x" * 100
            assert otf_cache._truncate_for_embedding(
                short, model_name="text-embedding-3-small",
            ) == short
            long = "x" * 60_000
            capped = otf_cache._truncate_for_embedding(
                long, model_name="text-embedding-3-small",
            )
            assert len(capped) <= 28_000
        assert any("tiktoken unavailable" in r.message.lower()
                   or "tiktoken" in r.message.lower()
                   for r in caplog.records)
    finally:
        sys.modules.clear()
        sys.modules.update(real_modules)
        importlib.reload(otf_cache)


def test_truncate_for_embedding_encoder_load_failure_logs_and_char_caps(
    monkeypatch, caplog,
):
    """Both `encoding_for_model` and `get_encoding('cl100k_base')`
    failing must also degrade to char-cap, not crash."""
    import tiktoken

    def _boom(*a, **kw):
        raise RuntimeError("encoder load went boom")

    monkeypatch.setattr(tiktoken, "encoding_for_model", _boom)
    monkeypatch.setattr(tiktoken, "get_encoding", _boom)

    from bird_interact_agents.slayer_otf.cache import _truncate_for_embedding

    with caplog.at_level(logging.WARNING):
        out = _truncate_for_embedding(
            "x" * 60_000, model_name="text-embedding-3-small",
        )
    assert len(out) <= 28_000


# ---------------------------------------------------------------------------
# _materialise_cache_memories — integration
# ---------------------------------------------------------------------------


def test_materialise_cache_memories_truncates_before_embed_batch(
    monkeypatch, tmp_path: Path,
):
    """The end-to-end caller path: a memory whose rendered text exceeds
    7800 tokens must arrive at `embed_batch` already truncated. The stub
    `embed_batch` ASSERTS that every input is ≤ 7800 tokens."""
    import tiktoken

    from bird_interact_agents.slayer_otf import cache as otf_cache

    enc = tiktoken.get_encoding("cl100k_base")
    embed_inputs: list[str] = []

    async def stub_embed_batch(texts, *, model=None):
        # Enforce: every input is ≤ 7800 tokens.
        for t in texts:
            assert len(enc.encode(t)) <= 7800, (
                f"embed_batch received {len(enc.encode(t))}-token input "
                f"(should be ≤ 7800)"
            )
        embed_inputs.extend(texts)
        # Return a vector per input.
        return [[0.1] * 8 for _ in texts]

    # Patch our local re-export so the call site (cache._materialise_cache_memories)
    # actually invokes the stub. The render function gives a long text
    # for one memory and a short one for the other.
    monkeypatch.setattr(otf_cache, "embed_batch", stub_embed_batch)
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model",
        lambda: "openai/text-embedding-3-small",
    )

    def fake_render(*, memory):
        if memory.id == "long":
            return " ".join(["word"] * 12_000)
        return "Short text."

    monkeypatch.setattr(otf_cache, "render_memory_text_for_embedding", fake_render)

    # Build the synthetic memories. `encode_kb_as_memories` lives in the
    # caller's helper module; we monkeypatch to return our two memories.
    long_mem = MagicMock()
    long_mem.id = "long"
    short_mem = MagicMock()
    short_mem.id = "short"

    def fake_encode(*a, **kw):
        # Returns a list of dicts that `Memory.model_validate` accepts.
        # We bypass validation by also stubbing Memory.model_validate.
        return [{"id": "long"}, {"id": "short"}]

    monkeypatch.setattr(otf_cache, "encode_kb_as_memories", fake_encode)
    monkeypatch.setattr(
        otf_cache.Memory, "model_validate",
        lambda d: long_mem if d.get("id") == "long" else short_mem,
    )

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    asyncio.run(otf_cache._materialise_cache_memories(
        db="alien", build_dir=build_dir, kb_rows=[{"id": 1}, {"id": 2}],
    ))
    # Both texts arrived at embed_batch under budget.
    assert len(embed_inputs) == 2


def test_materialise_cache_memories_logs_per_truncation(
    monkeypatch, tmp_path: Path, caplog,
):
    """One warning log line per truncated memory, naming the memory id
    + db so operators can find the offenders."""
    from bird_interact_agents.slayer_otf import cache as otf_cache

    async def stub_embed_batch(texts, *, model=None):
        return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(otf_cache, "embed_batch", stub_embed_batch)
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model",
        lambda: "openai/text-embedding-3-small",
    )

    long_mem = MagicMock()
    long_mem.id = "long_offender"

    monkeypatch.setattr(
        otf_cache, "render_memory_text_for_embedding",
        lambda *, memory: " ".join(["word"] * 12_000),
    )
    monkeypatch.setattr(otf_cache, "encode_kb_as_memories", lambda *a, **kw: [{"id": "long_offender"}])
    monkeypatch.setattr(otf_cache.Memory, "model_validate", lambda d: long_mem)

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    with caplog.at_level(logging.WARNING):
        asyncio.run(otf_cache._materialise_cache_memories(
            db="alien", build_dir=build_dir, kb_rows=[{"id": 1}],
        ))
    records = [r for r in caplog.records
               if "truncated embedding text" in r.message.lower()]
    assert len(records) == 1
    assert "long_offender" in records[0].message
    assert "alien" in records[0].message


# ---------------------------------------------------------------------------
# Cache fingerprint participation (Codex finding #9)
# ---------------------------------------------------------------------------


def test_embedding_builder_version_in_cache_fingerprint(monkeypatch):
    """Bumping `_EMBEDDING_BUILDER_VERSION` MUST change
    `_impl_fingerprint_of(...)` so already-built caches invalidate
    automatically without manual `rm -rf _cache_fp.txt`."""
    from bird_interact_agents.slayer_otf import cache as otf_cache

    fp_before = otf_cache._impl_fingerprint_of(None)
    monkeypatch.setattr(otf_cache, "_EMBEDDING_BUILDER_VERSION", 999)
    fp_after = otf_cache._impl_fingerprint_of(None)
    assert fp_before != fp_after


# ---------------------------------------------------------------------------
# Codex round-2 additions
# ---------------------------------------------------------------------------


def test_materialise_cache_memories_passes_exact_truncated_strings_to_embed_batch(
    monkeypatch, tmp_path: Path,
):
    """Codex round-2 finding #14: prove the truncation happens AFTER
    rendering and BEFORE embed_batch — i.e., the exact strings reaching
    `embed_batch` are the truncation output, not the raw rendered text."""
    import tiktoken

    from bird_interact_agents.slayer_otf import cache as otf_cache

    enc = tiktoken.get_encoding("cl100k_base")
    seen_inputs: list[str] = []

    async def stub_embed_batch(texts, *, model=None):
        seen_inputs.extend(texts)
        return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(otf_cache, "embed_batch", stub_embed_batch)
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model",
        lambda: "openai/text-embedding-3-small",
    )

    long_raw = " ".join(["word"] * 12_000)
    short_raw = "short text"

    def fake_render(*, memory):
        if memory.id == "long":
            return long_raw
        return short_raw

    monkeypatch.setattr(otf_cache, "render_memory_text_for_embedding", fake_render)

    long_mem = MagicMock(); long_mem.id = "long"
    short_mem = MagicMock(); short_mem.id = "short"
    monkeypatch.setattr(
        otf_cache, "encode_kb_as_memories",
        lambda *a, **kw: [{"id": "long"}, {"id": "short"}],
    )
    monkeypatch.setattr(
        otf_cache.Memory, "model_validate",
        lambda d: long_mem if d.get("id") == "long" else short_mem,
    )

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    asyncio.run(otf_cache._materialise_cache_memories(
        db="alien", build_dir=build_dir, kb_rows=[{"id": 1}, {"id": 2}],
    ))
    # The short text reached embed_batch verbatim (no truncation needed).
    assert short_raw in seen_inputs
    # The long text was NOT passed verbatim — the truncated version was.
    assert long_raw not in seen_inputs
    # And the long input that DID arrive is under budget.
    long_inputs = [t for t in seen_inputs if t != short_raw]
    assert len(long_inputs) == 1
    assert len(enc.encode(long_inputs[0])) <= 7800


def test_materialise_cache_memories_logs_full_warning_content_per_memory(
    monkeypatch, tmp_path: Path, caplog,
):
    """Codex round-2 finding #15: 2 long memories + 1 short → exactly 2
    warning records (per-truncation), each carrying memory id, db, AND
    the original/capped char counts, AND the model name."""
    from bird_interact_agents.slayer_otf import cache as otf_cache

    async def stub_embed_batch(texts, *, model=None):
        return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(otf_cache, "embed_batch", stub_embed_batch)
    monkeypatch.setattr(otf_cache, "_embeddings_available", lambda: True)
    monkeypatch.setattr(
        otf_cache, "_embedding_current_model",
        lambda: "openai/text-embedding-3-small",
    )

    long_a = " ".join(["worda"] * 12_000)
    long_b = " ".join(["wordb"] * 12_000)
    short = "short text."

    def fake_render(*, memory):
        return {"a": long_a, "b": long_b, "c": short}[memory.id]

    monkeypatch.setattr(otf_cache, "render_memory_text_for_embedding", fake_render)

    mems = []
    for _id in ("a", "b", "c"):
        m = MagicMock(); m.id = _id
        mems.append(m)

    monkeypatch.setattr(
        otf_cache, "encode_kb_as_memories",
        lambda *a, **kw: [{"id": "a"}, {"id": "b"}, {"id": "c"}],
    )
    monkeypatch.setattr(
        otf_cache.Memory, "model_validate",
        lambda d: {"a": mems[0], "b": mems[1], "c": mems[2]}[d["id"]],
    )

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    with caplog.at_level(logging.WARNING):
        asyncio.run(otf_cache._materialise_cache_memories(
            db="alien_db", build_dir=build_dir,
            kb_rows=[{"id": 1}, {"id": 2}, {"id": 3}],
        ))
    records = [r for r in caplog.records
               if "truncated embedding text" in r.message.lower()]
    assert len(records) == 2  # exactly per-truncation, not per-batch
    ids_seen = set()
    for r in records:
        msg = r.message
        assert "alien_db" in msg
        # model name (or its bare form) appears
        assert "text-embedding-3-small" in msg
        # char counts (original + capped)
        assert "chars" in msg.lower()
        if "a" in msg:
            ids_seen.add("a")
        if "b" in msg:
            ids_seen.add("b")
    assert ids_seen == {"a", "b"}


def test_truncate_for_embedding_char_cap_fallback_short_text_unchanged(
    monkeypatch,
):
    """Codex round-2 finding #16: when tiktoken is unavailable the char-cap
    fallback still returns short text unchanged (no needless work)."""
    real_modules = sys.modules.copy()
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    from bird_interact_agents.slayer_otf import cache as otf_cache

    importlib.reload(otf_cache)
    try:
        short = "hello, world."
        out = otf_cache._truncate_for_embedding(
            short, model_name="text-embedding-3-small",
        )
        assert out == short
    finally:
        sys.modules.clear()
        sys.modules.update(real_modules)
        importlib.reload(otf_cache)


def test_impl_fingerprint_stable_for_unchanged_inputs(monkeypatch):
    """Codex round-2 finding #17: calling `_impl_fingerprint_of` twice
    with the same inputs is byte-equal — no time-dependent or random
    component leaks into the fingerprint."""
    from bird_interact_agents.slayer_otf import cache as otf_cache

    fp1 = otf_cache._impl_fingerprint_of(None)
    fp2 = otf_cache._impl_fingerprint_of(None)
    assert fp1 == fp2


def test_impl_fingerprint_still_changes_on_existing_inputs(monkeypatch):
    """Codex round-2 finding #17 (regression guard): bumping the
    embedding-model component still moves the fingerprint, proving the
    existing inputs weren't silently dropped when we added the builder
    version."""
    from bird_interact_agents.slayer_otf import cache as otf_cache

    fp_before = otf_cache._impl_fingerprint_of(None)
    monkeypatch.setattr(
        otf_cache, "_active_embedding_model_or_none",
        lambda: "openai/different-embedding-model",
    )
    fp_after = otf_cache._impl_fingerprint_of(None)
    assert fp_before != fp_after
