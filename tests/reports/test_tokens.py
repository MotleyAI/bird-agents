"""Tests for the token-counter wrapper.

Spec (DEV-1553 + Codex finding #4):
* ``count_tokens(s)`` returns Anthropic's token count for ``s`` AFTER
  subtracting a per-process baseline = ``count_tokens("")``, so the
  Section VI 250/1000 thresholds are contract-exact (no wrapper bias).
* The function is module-level so tests can monkeypatch it.
* An LRU cache keyed on ``(hash(s), model)`` keeps the network call count
  bounded when SQL strings repeat (retry submits).
"""

from __future__ import annotations

from unittest.mock import MagicMock


def test_count_tokens_subtracts_empty_message_baseline(monkeypatch):
    """The wrapper subtracts the empty-message token count so that
    ``count_tokens("")`` returns 0 and any short string returns the
    intrinsic token count without the user-message envelope overhead."""
    from bird_interact_agents.reports import tokens as _tokens

    # Stub the underlying Anthropic-SDK call so the test is offline.
    # The stub returns a fixed envelope of 4 tokens for empty content + 1
    # token per non-empty character (a deliberately weird shape to verify
    # the baseline is subtracted, not approximated).
    def _stub_api(messages, **_):
        body = messages[0]["content"]
        return MagicMock(input_tokens=4 + len(body))

    monkeypatch.setattr(_tokens, "_count_tokens_via_api", _stub_api)
    # Force the baseline to recompute.
    _tokens._reset_baseline_cache()

    # Empty string → 0 reported (envelope subtracted).
    assert _tokens.count_tokens("") == 0
    # A 7-char string → 7 reported (envelope 4 subtracted from raw 11).
    assert _tokens.count_tokens("abcdefg") == 7


def test_count_tokens_caches_repeated_calls(monkeypatch):
    """Repeated calls for the same (string, model) hit cache, not the API."""
    from bird_interact_agents.reports import tokens as _tokens

    calls: list[str] = []

    def _stub_api(messages, **_):
        body = messages[0]["content"]
        calls.append(body)
        return MagicMock(input_tokens=4 + len(body))

    monkeypatch.setattr(_tokens, "_count_tokens_via_api", _stub_api)
    _tokens._reset_baseline_cache()

    s = "SELECT 1 FROM x"
    a = _tokens.count_tokens(s)
    b = _tokens.count_tokens(s)
    assert a == b
    # API hit: once for baseline, once for the string. Second call to
    # count_tokens(s) MUST not hit the API again.
    assert calls.count(s) == 1


def test_count_tokens_function_is_module_level_for_monkeypatching():
    """The fake_count_tokens fixture relies on monkeypatching
    ``reports.tokens.count_tokens`` directly."""
    from bird_interact_agents.reports import tokens as _tokens

    assert hasattr(_tokens, "count_tokens")
    assert callable(_tokens.count_tokens)
