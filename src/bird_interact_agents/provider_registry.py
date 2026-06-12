"""DEV-1555 Stage 2: open-weight provider registry.

Single source of truth for open-weight backends usable by the
``claude_sdk_otf*`` agents: endpoints, auth env var, API format, context
windows, and official pricing. Every key-handling site consumes this
registry (cloud prereqs, driver secret shipping, the SDK session env,
the user-sim litellm route, the autopsy client) so adding a provider is
one entry here, not a code change per site.

First provider: Moonshot (Kimi K2.7 Code) via its Anthropic-compatible
endpoint — the Claude Agent SDK connects directly through
``ANTHROPIC_BASE_URL``; no translation proxy. Providers whose only API is
OpenAI-format (``api_format="openai"``, e.g. Doubleword) will additionally
need the deferred LiteLLM sidecar before they can serve the SDK agents.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import litellm
from pydantic import BaseModel


class ModelPricing(BaseModel):
    """Official per-token USD prices (litellm cost-map shape)."""

    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: Optional[float] = None


class ProviderSpec(BaseModel):
    key: str
    # Anthropic-format endpoint (what ANTHROPIC_BASE_URL points at).
    base_url: str
    # Env var that overrides ``base_url`` at runtime.
    base_url_env: str
    api_format: Literal["anthropic", "openai"]
    auth_env: str
    default_context_window: int
    # Per-model overrides; provider default applies otherwise.
    model_context_windows: dict[str, int] = {}
    # OpenAI-compatible endpoint for the litellm (user-sim) route.
    openai_base_url: Optional[str] = None
    # Official pricing per native model id; registered with litellm so
    # cost reports are accurate instead of the $0 unpriced fallback.
    model_pricing: dict[str, ModelPricing] = {}


# kimi-k2.7-code numbers from the official pricing page (2026-06):
# context window 262,144 tokens; input $0.95/M (cache hit $0.19/M),
# output $4.00/M.
_KIMI_K27_CODE_PRICING = ModelPricing(
    input_cost_per_token=0.95e-6,
    output_cost_per_token=4.00e-6,
    cache_read_input_token_cost=0.19e-6,
)

REGISTRY: dict[str, ProviderSpec] = {
    "moonshot": ProviderSpec(
        key="moonshot",
        base_url="https://api.moonshot.ai/anthropic",
        base_url_env="BIRD_MOONSHOT_ANTHROPIC_BASE_URL",
        api_format="anthropic",
        auth_env="MOONSHOT_API_KEY",
        default_context_window=262_144,
        openai_base_url="https://api.moonshot.ai/v1",
        model_pricing={"kimi-k2.7-code": _KIMI_K27_CODE_PRICING},
    ),
}


def _split(model: str) -> tuple[str, str]:
    provider, sep, rest = model.partition("/")
    return (provider, rest) if sep else ("", model)


def get_provider(model: str) -> Optional[ProviderSpec]:
    """Registry spec for a LiteLLM-style ``provider/model`` string, else None.

    Anthropic is deliberately NOT a registry provider — it keeps its
    dedicated OAuth/API-key flows.
    """
    provider, _ = _split(model)
    return REGISTRY.get(provider)


def is_supported_agent_model(model: str) -> bool:
    """True iff the claude_sdk_otf* agents can run this model."""
    return model.split("/", 1)[0] == "anthropic" or get_provider(model) is not None


def required_env_for(model: str) -> tuple[str, ...]:
    spec = get_provider(model)
    return (spec.auth_env,) if spec else ()


def resolve_base_url(spec: ProviderSpec) -> str:
    return os.environ.get(spec.base_url_env) or spec.base_url


def provider_api_key(spec: ProviderSpec) -> str:
    key = os.environ.get(spec.auth_env, "")
    if not key:
        raise RuntimeError(
            f"{spec.auth_env} is not set — required for {spec.key} models. "
            f"export {spec.auth_env}=<your-key>"
        )
    return key


def sdk_session_env(model: str) -> dict[str, str]:
    """Per-run env for the Claude Agent SDK subprocess.

    ``ANTHROPIC_AUTH_TOKEN`` (Bearer) rather than ``ANTHROPIC_API_KEY``:
    the third-party Anthropic-compatible endpoints document Bearer auth,
    and keeping the API-key name out of the session env removes any
    ambiguity with ambient Anthropic credentials.
    """
    spec = get_provider(model)
    if spec is None:
        raise ValueError(f"{model!r} is not a registry open-weight model")
    return {
        "ANTHROPIC_BASE_URL": resolve_base_url(spec),
        "ANTHROPIC_AUTH_TOKEN": provider_api_key(spec),
    }


def litellm_route(model: str) -> tuple[str, dict[str, str]]:
    """(litellm_model, extra_kwargs) for the user-sim path.

    Registry providers route through litellm's generic OpenAI-compatible
    driver (``openai/<native_id>`` + ``api_base``) — robust regardless of
    whether litellm ships a dedicated provider for them.
    """
    spec = get_provider(model)
    if spec is None or not spec.openai_base_url:
        return model, {}
    _, native_id = _split(model)
    return (
        f"openai/{native_id}",
        {"api_base": spec.openai_base_url, "api_key": provider_api_key(spec)},
    )


_pricing_registered = False


def ensure_litellm_pricing() -> None:
    """Register official registry prices with litellm (idempotent).

    Registered under BOTH the canonical ``<provider>/<id>`` string (used
    by agent-side cost rows) and the rewritten ``openai/<id>`` string
    (used by the user-sim litellm route), so neither path falls back to
    the $0 unpriced warning.
    """
    global _pricing_registered
    if _pricing_registered:
        return
    cost_map: dict[str, dict] = {}
    for spec in REGISTRY.values():
        for native_id, pricing in spec.model_pricing.items():
            prices = {
                k: v for k, v in pricing.model_dump().items() if v is not None
            }
            # litellm validates the registered litellm_provider against the
            # provider it infers from the model string — each key needs its
            # own provider tag.
            cost_map[f"{spec.key}/{native_id}"] = {
                "litellm_provider": spec.key, "mode": "chat", **prices,
            }
            cost_map[f"openai/{native_id}"] = {
                "litellm_provider": "openai", "mode": "chat", **prices,
            }
    if cost_map:
        litellm.register_model(cost_map)
    _pricing_registered = True
