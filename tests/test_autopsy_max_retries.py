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


def test_max_retries_pins_the_contract():
    # Pin the exact contract (6), not just "> SDK default of 2": a regression
    # to 3-5 should fail here, since the whole point is to ride out 429 bursts.
    assert _AUTOPSY_MAX_RETRIES == 6


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


def test_provider_path_passes_max_retries():
    # Registry open-weight models route to the provider's Anthropic-compatible
    # endpoint, NOT the ambient-Anthropic env paths above. This branch must
    # also carry the raised retry budget.
    with patch(
        "bird_interact_agents.eval.autopsy.get_provider", return_value=object()
    ), patch(
        "bird_interact_agents.eval.autopsy.resolve_base_url",
        return_value="https://example.invalid",
    ), patch(
        "bird_interact_agents.eval.autopsy.provider_api_key",
        return_value="provider-token",
    ), patch("anthropic.AsyncAnthropic") as m:
        _build_anthropic_client("moonshot/kimi-k2.7-code")
    assert m.call_args.kwargs.get("max_retries") == _AUTOPSY_MAX_RETRIES
