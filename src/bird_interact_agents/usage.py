"""Token-usage accumulator for benchmark runs.

Each `TokenUsage` records totals across many LLM calls plus a per-(scope,
model) breakdown. Cost is computed at `add_call` time via
`litellm.cost_per_token`, so callers always see a numeric cost — unpriced
models silently record 0.

Wire it through the per-task `deps`/state object, then `merge()` per-task
accumulators in `run.py` to produce the eval-level total.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Awaitable, Callable

import litellm
from pydantic import BaseModel, Field

from bird_interact_agents import provider_registry

logger = logging.getLogger(__name__)

# Indirection seams so tests can monkey-patch without touching litellm.
_acompletion = litellm.acompletion

# DEV-1517: dedicated env var for the user-sim Anthropic API key on OAuth runs.
# ANTHROPIC_API_KEY is absent from the worker env on OAuth runs (the SDK would
# silently prefer it over the OAuth token), so the driver ships the key under
# this non-standard name which LiteLLM does not auto-discover.
_LITELLM_ANTHROPIC_KEY_ENV = "BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY"


def _maybe_inject_anthropic_key(model: str, kwargs: dict) -> None:
    """Inject the dedicated Anthropic API key into LiteLLM kwargs when present.

    On OAuth runs, ANTHROPIC_API_KEY is absent from the worker env. LiteLLM
    reads the key via this explicit kwarg instead of auto-discovering the
    standard env var.  Does NOT mutate os.environ.
    """
    if not model.startswith("anthropic/"):
        return
    if "api_key" in kwargs:
        return
    dedicated = os.environ.get(_LITELLM_ANTHROPIC_KEY_ENV)
    if dedicated:
        kwargs["api_key"] = dedicated


def _maybe_inject_registry_kwargs(model: str, kwargs: dict) -> str:
    """Route registry open-weight models (DEV-1555) through litellm's
    OpenAI-compatible driver.

    Returns the model string to hand litellm — rewritten to
    ``openai/<native_id>`` with ``api_base``/``api_key`` injected for
    registry providers, unchanged otherwise. Caller-passed kwargs are
    never clobbered. Also registers the providers' official prices with
    litellm so cost rows stay accurate.
    """
    litellm_model, extra = provider_registry.litellm_route(model)
    if litellm_model == model:
        return model
    provider_registry.ensure_litellm_pricing()
    for k, v in extra.items():
        kwargs.setdefault(k, v)
    return litellm_model


# ---------------------------------------------------------------------------
# Rate-limit retry shell. Used by the user-simulator path.
#
# Anthropic returns 429 with a `retry-after` header (seconds) when ITPM/OTPM
# is breached. LiteLLM's default no-op retry policy means a concurrent burst
# can surface those as hard failures. We wrap each call with our own retry
# loop honouring `retry-after` first, falling back to jittered exp backoff.
# ---------------------------------------------------------------------------

_MAX_RETRY_ATTEMPTS = 6
_BASE_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0


def _extract_retry_after(exc: BaseException) -> float | None:
    """Best-effort: pull a numeric `retry-after` (seconds) off a litellm
    rate-limit exception. Different providers attach the header in
    different shapes (`response.headers`, `headers`, plain attribute) —
    try them in order and return None when nothing parses."""
    for attr in ("response", "raw_response", "http_response"):
        resp = getattr(exc, attr, None)
        if resp is None:
            continue
        headers = getattr(resp, "headers", None)
        if headers is None:
            continue
        try:
            value = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            value = None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    direct = getattr(exc, "retry_after", None)
    if direct is not None:
        try:
            return float(direct)
        except (TypeError, ValueError):
            return None
    return None


def _is_retryable(exc: BaseException) -> bool:
    """Treat rate-limit, transient API, timeout and connection errors as
    retryable. LiteLLM exposes provider exceptions through its own
    typed wrappers (`litellm.exceptions`)."""
    le = litellm.exceptions
    retryable_types: tuple[type[BaseException], ...] = tuple(
        t for t in (
            getattr(le, "RateLimitError", None),
            getattr(le, "APIConnectionError", None),
            getattr(le, "APIError", None),
            getattr(le, "Timeout", None),
            getattr(le, "ServiceUnavailableError", None),
            getattr(le, "InternalServerError", None),
        ) if t is not None
    )
    return isinstance(exc, retryable_types)


async def _retry_litellm(
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int = _MAX_RETRY_ATTEMPTS,
) -> Any:
    """Call `coro_factory()` and await the result, retrying transient
    failures. `coro_factory` must return a fresh coroutine on each call so
    we can re-await without reusing an exhausted coroutine.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 — we re-raise non-retryable
            if not _is_retryable(exc) or attempt == max_attempts:
                raise
            last_exc = exc
            wait = _extract_retry_after(exc)
            if wait is None:
                wait = min(
                    _MAX_BACKOFF_S,
                    _BASE_BACKOFF_S * (2 ** (attempt - 1)),
                )
                wait += random.uniform(0, wait * 0.25)
            logger.warning(
                "LiteLLM transient error (%s); retry %d/%d after %.1fs",
                type(exc).__name__, attempt, max_attempts - 1, wait,
            )
            await asyncio.sleep(wait)
    if last_exc is not None:  # pragma: no cover — defensive
        raise last_exc


def _cost_per_token(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> tuple[float, float]:
    """Return (prompt_cost_usd, completion_cost_usd) for this call.

    Wraps `litellm.cost_per_token`, which returns the same tuple for any
    provider it has pricing for. Cache read/write tokens are billed at
    Anthropic's discounted/premium rates (10%/125% of input cost
    respectively) when passed in — litellm reads the per-model
    `cache_*_input_token_cost` entries and folds them into the
    returned prompt cost. Raises if the model is unknown — callers
    must handle that.
    """
    return litellm.cost_per_token(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )


_warned_unpriced: set[str] = set()


def _safe_cost(
    *, model: str, prompt_tokens: int, completion_tokens: int,
    cache_read_input_tokens: int = 0, cache_creation_input_tokens: int = 0,
) -> tuple[float, float]:
    # DEV-1555: registry open-weight models are priced IN-HOUSE from the
    # registry's official prices. litellm's openai-provider math assumes
    # prompt_tokens INCLUDES cached tokens (OpenAI convention) while our
    # callers pass Anthropic-convention numbers (input excludes cache), so
    # it swallows the non-cached input whenever cache_read is present —
    # and it prices cache_creation at $0. Moonshot's model is hit/miss:
    # cache writes ARE misses, billed at the full input rate.
    spec = provider_registry.get_provider(model)
    if spec is not None:
        _, _, _native = model.partition("/")
        pricing = spec.model_pricing.get(_native)
        if pricing is not None:
            read_rate = (
                pricing.cache_read_input_token_cost
                if pricing.cache_read_input_token_cost is not None
                else pricing.input_cost_per_token
            )
            prompt_cost = (
                (prompt_tokens + cache_creation_input_tokens)
                * pricing.input_cost_per_token
                + cache_read_input_tokens * read_rate
            )
            return prompt_cost, completion_tokens * pricing.output_cost_per_token
    # Registry models WITHOUT a pricing entry still need litellm to see a
    # registration so cost_per_token degrades to the NotFoundError path
    # instead of a bare Exception. Idempotent, no-op after the first call.
    provider_registry.ensure_litellm_pricing()
    try:
        return _cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        )
    except litellm.exceptions.NotFoundError as e:
        # Only swallow the "no price entry for this model" case — that's
        # the expected, recoverable failure (warn once, record $0). Any
        # other exception type points at an integration bug we don't
        # want to silently mask.
        if model not in _warned_unpriced:
            _warned_unpriced.add(model)
            logger.warning(
                "No litellm price entry for model %r (%s); "
                "recording cost_usd=0 for this run.",
                model, e,
            )
        return 0.0, 0.0


class CallCost(BaseModel):
    """Per-(scope, model) aggregate.

    `name` is the canonical "{scope}::{model}" key used for merging.
    """

    name: str
    scope: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    n_calls: int = 0
    cost_usd: float = 0.0


class TokenUsage(BaseModel):
    """Cumulative token + cost accumulator across many LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    n_calls: int = 0
    cost_usd: float = 0.0
    agent_cost_usd: float = 0.0
    user_sim_cost_usd: float = 0.0
    partial: bool = False  # set when a framework can't expose all usage
    breakdown: list[CallCost] = Field(default_factory=list)

    def _row_for(self, *, scope: str, model: str) -> CallCost:
        name = f"{scope}::{model}"
        for row in self.breakdown:
            if row.name == name:
                return row
        row = CallCost(name=name, scope=scope, model=model)
        self.breakdown.append(row)
        return row

    def add_call(
        self,
        *,
        scope: str,
        model: str,
        prompt: int,
        completion: int,
        reasoning: int = 0,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> None:
        prompt_cost, completion_cost = _safe_cost(
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        )
        cost = prompt_cost + completion_cost

        row = self._row_for(scope=scope, model=model)
        row.prompt_tokens += prompt
        row.completion_tokens += completion
        row.reasoning_tokens += reasoning
        row.cache_read_tokens += cache_read
        row.cache_write_tokens += cache_write
        row.n_calls += 1
        row.cost_usd += cost

        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.reasoning_tokens += reasoning
        self.cache_read_tokens += cache_read
        self.cache_write_tokens += cache_write
        self.n_calls += 1
        self.cost_usd += cost
        if scope == "agent":
            self.agent_cost_usd += cost
        elif scope == "user_sim":
            self.user_sim_cost_usd += cost

    def merge(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.n_calls += other.n_calls
        self.cost_usd += other.cost_usd
        self.agent_cost_usd += other.agent_cost_usd
        self.user_sim_cost_usd += other.user_sim_cost_usd
        self.partial = self.partial or other.partial

        for incoming in other.breakdown:
            existing = next(
                (r for r in self.breakdown if r.name == incoming.name), None,
            )
            if existing is None:
                self.breakdown.append(incoming.model_copy(deep=True))
            else:
                existing.prompt_tokens += incoming.prompt_tokens
                existing.completion_tokens += incoming.completion_tokens
                existing.reasoning_tokens += incoming.reasoning_tokens
                existing.cache_read_tokens += incoming.cache_read_tokens
                existing.cache_write_tokens += incoming.cache_write_tokens
                existing.n_calls += incoming.n_calls
                existing.cost_usd += incoming.cost_usd


async def acompletion_tracked(
    accum: TokenUsage,
    *,
    scope: str,
    model: str,
    **kwargs: Any,
):
    """Drop-in replacement for `litellm.acompletion` that records usage.

    Returns the raw response object (so callers can read `.choices[0]`,
    `.usage`, etc. as before).

    Wrapped with `_retry_litellm` so transient rate-limit and connection
    errors don't surface as hard failures during concurrent benchmark runs.
    """
    _maybe_inject_anthropic_key(model, kwargs)
    litellm_model = _maybe_inject_registry_kwargs(model, kwargs)
    response = await _retry_litellm(
        lambda: _acompletion(model=litellm_model, **kwargs),
    )
    usage_obj = getattr(response, "usage", None)
    if usage_obj is not None:
        prompt = getattr(usage_obj, "prompt_tokens", 0) or 0
        completion = getattr(usage_obj, "completion_tokens", 0) or 0
        reasoning = getattr(usage_obj, "reasoning_tokens", 0) or 0
        cache_read = getattr(usage_obj, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage_obj, "cache_creation_input_tokens", 0) or 0
        accum.add_call(
            scope=scope,
            model=model,
            prompt=prompt,
            completion=completion,
            reasoning=reasoning,
            cache_read=cache_read,
            cache_write=cache_write,
        )
    return response
