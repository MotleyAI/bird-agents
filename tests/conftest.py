"""Shared test fixtures and configuration.

Pre-seeds `BIRD_DATA_PATH` / `BIRD_DB_PATH` from the central `paths` helper
so tests work from any checkout (canonical or `git worktree`) without
requiring per-developer env-var setup.
"""

import os

# Force litellm to use its bundled model-cost map instead of fetching the
# latest over the network at import time — faster import and fully offline
# (no real network during tests). Must be set before litellm is imported.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest

from bird_interact_agents import paths

# Set defaults for environment variables before any bird_interact_agents
# imports that read them at module level (e.g. mini-interact-agent harness).
os.environ.setdefault("BIRD_DATA_PATH", str(paths.mini_interact_data_file()))
os.environ.setdefault("BIRD_DB_PATH", str(paths.mini_interact_root()))


@pytest.fixture(autouse=True)
def _disable_real_embeddings(monkeypatch):
    """Globally force SLayer's embedding channel OFF for the whole suite.

    SLayer auto-embeds on every write (``save_memory`` / ``edit_model`` /
    ``ingest``) and runs dense search whenever ``embeddings.client.is_available()``
    is True — which it is on any machine with an ``OPENAI_API_KEY`` in the env.
    Left unmocked, tests that touch a real SLayer storage would make real
    (paid, non-deterministic) OpenAI embedding calls AND write ``embeddings.db``
    sidecars into committed dirs. Forcing ``is_available`` False short-circuits
    every refresh hook + the dense search channel (BM25 + tantivy still work),
    keeping the suite offline and deterministic (the no-real-APIs rule).

    Tests that specifically exercise the embedding hook re-enable it locally
    (monkeypatch ``is_available`` True) AND stub ``EmbeddingService`` so they
    still never hit the wire.
    """
    try:
        from slayer.embeddings import client as _emb_client
    except Exception:  # noqa: BLE001 — embeddings extra not installed → nothing to do
        return
    monkeypatch.setattr(_emb_client, "is_available", lambda: False)
