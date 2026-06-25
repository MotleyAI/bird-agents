"""DEV-1605: ``model_string.encoder_version_slug`` derives the default
version label from the encoder model string.

Rule: strip the provider prefix, strip a leading ``claude-``, replace any
remaining ``/`` with ``__`` (filesystem-safe). Empty/unsafe results raise.
"""

from __future__ import annotations

import pytest

from bird_interact_agents import paths
from bird_interact_agents.model_string import encoder_version_slug


@pytest.fixture(autouse=True)
def _isolate_main_checkout_cache():
    """Clear the memoised main-checkout resolution before+after each test."""
    paths._main_checkout_root_cached.cache_clear()
    yield
    paths._main_checkout_root_cached.cache_clear()


@pytest.mark.parametrize(
    "model,expected",
    [
        ("anthropic/claude-opus-4-7", "opus-4-7"),
        ("anthropic/claude-sonnet-4-6", "sonnet-4-6"),
        ("zai/glm-5.2", "glm-5.2"),
        ("openai/gpt-5.2", "gpt-5.2"),
        ("openrouter/z-ai/glm-4.7-flash", "z-ai__glm-4.7-flash"),
        # No provider prefix at all — passes through (minus claude- if present).
        ("glm-5.2", "glm-5.2"),
        ("claude-opus-4-7", "opus-4-7"),
    ],
)
def test_slug_examples(model, expected):
    assert encoder_version_slug(model) == expected


def test_slug_is_filesystem_safe():
    # The slug must never contain a path separator after sanitisation.
    assert "/" not in encoder_version_slug("openrouter/z-ai/glm-4.7-flash")


@pytest.mark.parametrize("bad", ["", "anthropic/claude-", "/", "anthropic/"])
def test_slug_empty_result_raises(bad):
    with pytest.raises(ValueError):
        encoder_version_slug(bad)


def test_slug_output_accepted_by_path_validator(tmp_path, monkeypatch):
    """Every slug we emit must be a label the path helper accepts."""
    from bird_interact_agents import paths
    from tests.test_paths import _setup_main_and_worktree

    _setup_main_and_worktree(tmp_path, monkeypatch)
    for model in (
        "anthropic/claude-opus-4-7",
        "zai/glm-5.2",
        "openrouter/z-ai/glm-4.7-flash",
    ):
        slug = encoder_version_slug(model)
        # Must not raise.
        paths.slayer_models_otf_root(benchmark="mini-interact", version=slug)
