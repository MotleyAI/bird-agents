"""The autopsy Anthropic client must be built with a raised ``max_retries`` so
the SDK rides out transient 429s (exponential backoff + jitter + Retry-After)
instead of recording a `rate_limit_error` as an autopsy failure.
"""
from __future__ import annotations

from unittest.mock import patch

from bird_interact_agents.eval.autopsy import (
    _AUTOPSY_MAX_RETRIES,
    _build_anthropic_client,
)


def test_max_retries_above_sdk_default():
    # The whole point is to exceed the anthropic SDK default of 2.
    assert _AUTOPSY_MAX_RETRIES > 2


def test_api_key_path_passes_max_retries(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
    with patch("anthropic.AsyncAnthropic") as m:
        _build_anthropic_client("")
    assert m.call_args.kwargs.get("max_retries") == _AUTOPSY_MAX_RETRIES


def test_oauth_path_passes_max_retries(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
    # oauth takes precedence over api_key, but clear it anyway for clarity
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch("anthropic.AsyncAnthropic") as m:
        _build_anthropic_client("")
    assert m.call_args.kwargs.get("max_retries") == _AUTOPSY_MAX_RETRIES
