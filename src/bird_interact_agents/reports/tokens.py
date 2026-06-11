"""Anthropic-SDK token counter with envelope-baseline subtraction.

Counts tokens for a single string by wrapping it as a one-message
``user`` message and calling ``anthropic.Anthropic().messages.count_tokens``.
We subtract a once-per-process baseline = ``count_tokens("")`` so the
Section VI 250 / 1000 thresholds are contract-exact (no wrapper bias).

Tests monkeypatch ``count_tokens`` to a deterministic char-based fake;
production runs call the live API (free; no LLM rollout).
"""

from __future__ import annotations

import functools
import os
from typing import Any


DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_BASELINE_BY_MODEL: dict[str, int] = {}


def _count_tokens_via_api(messages: list[dict[str, Any]], *, model: str) -> Any:
    """Real Anthropic API call. Tests stub THIS via monkeypatching."""
    # Lazy-import so tests that fake the function never touch anthropic.
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return client.messages.count_tokens(model=model, messages=messages)


def _baseline(model: str) -> int:
    if model not in _BASELINE_BY_MODEL:
        result = _count_tokens_via_api(
            messages=[{"role": "user", "content": ""}], model=model
        )
        _BASELINE_BY_MODEL[model] = int(result.input_tokens)
    return _BASELINE_BY_MODEL[model]


def _reset_baseline_cache() -> None:
    """Test seam: re-prime the baseline next time count_tokens is called."""
    _BASELINE_BY_MODEL.clear()
    cache_clear = getattr(count_tokens, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


@functools.lru_cache(maxsize=10_000)
def count_tokens(s: str, *, model: str = DEFAULT_MODEL) -> int:
    """Return ``Anthropic.messages.count_tokens(s) - baseline(model)``.

    Cached by ``(s, model)`` — submit-SQL strings often repeat across
    retries.
    """
    if not s:
        # Trivial fast path: empty content is 0 by definition (envelope
        # subtracted).
        return 0
    result = _count_tokens_via_api(
        messages=[{"role": "user", "content": s}], model=model
    )
    return max(0, int(result.input_tokens) - _baseline(model))
