"""Tests for the LiteLLM rate-limit retry shell wrapping
`acompletion_tracked`. The shell honours `retry-after` and falls back to
exp backoff with jitter; both paths must return the eventual successful
response while skipping the asyncio sleep delays in tests."""

from __future__ import annotations

import asyncio
import types
from unittest.mock import patch

import pytest

import litellm

from bird_interact_agents import usage as usage_mod


class _FakeResponse:
    """Minimal acompletion-shaped response for the tracker."""
    def __init__(self, prompt: int = 1, completion: int = 1):
        self.usage = types.SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            cache_read_input_tokens=0,
        )


def _make_rate_limit_error(retry_after: float | None) -> Exception:
    """Construct a minimally viable RateLimitError. Some LiteLLM versions
    require positional message + llm_provider + model — hand them in by
    name so we don't depend on the exact arity."""
    cls = litellm.exceptions.RateLimitError
    err = cls.__new__(cls)
    Exception.__init__(err, "rate limit")
    err.message = "rate limit"
    err.llm_provider = "anthropic"
    err.model = "claude-haiku-4-5-20251001"
    if retry_after is not None:
        # _extract_retry_after walks `response.headers`; the simplest fixture
        # is a dict-like header bag on a stand-in response.
        err.response = types.SimpleNamespace(
            headers={"retry-after": str(retry_after)},
        )
    return err


@pytest.mark.asyncio
async def test_retry_litellm_retries_on_rate_limit_then_succeeds():
    calls = {"n": 0}

    async def flaky_completion(**_):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _make_rate_limit_error(retry_after=None)
        return _FakeResponse()

    accum = usage_mod.TokenUsage()
    sleeps: list[float] = []

    async def _no_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with patch.object(usage_mod, "_acompletion", flaky_completion), \
         patch.object(usage_mod.asyncio, "sleep", _no_sleep):
        resp = await usage_mod.acompletion_tracked(
            accum, scope="user_sim", model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert isinstance(resp, _FakeResponse)
    assert calls["n"] == 3
    # Two retries were issued.
    assert len(sleeps) == 2


@pytest.mark.asyncio
async def test_retry_litellm_honours_retry_after_header():
    """When the 429's response carries `retry-after`, the wait must be
    that exact value (no exp-backoff drift), with no jitter applied."""
    calls = {"n": 0}

    async def flaky_completion(**_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_rate_limit_error(retry_after=4.5)
        return _FakeResponse()

    sleeps: list[float] = []

    async def _no_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with patch.object(usage_mod, "_acompletion", flaky_completion), \
         patch.object(usage_mod.asyncio, "sleep", _no_sleep):
        await usage_mod.acompletion_tracked(
            usage_mod.TokenUsage(),
            scope="user_sim",
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert sleeps == [4.5]


@pytest.mark.asyncio
async def test_retry_litellm_does_not_retry_non_retryable():
    """A bad-request / authentication error must surface immediately —
    retrying would just waste budget."""
    cls = litellm.exceptions.AuthenticationError
    err = cls.__new__(cls)
    Exception.__init__(err, "bad creds")
    err.message = "bad creds"
    err.llm_provider = "anthropic"
    err.model = "claude-haiku-4-5-20251001"

    async def always_fails(**_):
        raise err

    with patch.object(usage_mod, "_acompletion", always_fails):
        with pytest.raises(litellm.exceptions.AuthenticationError):
            await usage_mod.acompletion_tracked(
                usage_mod.TokenUsage(),
                scope="user_sim",
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "hi"}],
            )
