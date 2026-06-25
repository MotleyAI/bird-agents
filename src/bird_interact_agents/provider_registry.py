"""DEV-1555 Stage 2: open-weight provider registry.

Single source of truth for open-weight backends usable by the
``claude_sdk_otf*`` agents: endpoints, auth env var, API format, context
windows, and official pricing. Every key-handling site consumes this
registry (cloud prereqs, driver secret shipping, the SDK session env,
the user-sim litellm route, the autopsy client) so adding a provider is
one entry here, not a code change per site.

First provider: Moonshot (Kimi K2.7 Code) via its Anthropic-compatible
endpoint — the Claude Agent SDK connects directly through
``ANTHROPIC_BASE_URL``; no translation proxy. Second (DEV-1580): z.ai
(Zhipu GLM) via the same Anthropic-compatible shape; unlike Kimi, no GLM
model requires ``thinking``. Providers whose only API is OpenAI-format
(``api_format="openai"``, e.g. Doubleword) will additionally need the
deferred LiteLLM sidecar before they can serve the SDK agents.
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
    # Anthropic-format endpoint (what ANTHROPIC_BASE_URL points at). ``None``
    # for OpenAI-only providers (DEV-1604, e.g. Doubleword) that have NO
    # Anthropic ``/v1/messages`` surface — the bridge proxy supplies the URL via
    # ``base_url_env`` at runtime, and ``resolve_base_url`` fails fast if it is
    # still unset (so a ``None`` never leaks to any consumer).
    base_url: Optional[str]
    # Env var that overrides ``base_url`` at runtime. STAYS required — it is the
    # override carrier the bridge proxy + driver shipping read.
    base_url_env: str
    api_format: Literal["anthropic", "openai"]
    auth_env: str
    default_context_window: int
    # False marks pricing as a deliberate, unconfirmed placeholder (DEV-1604
    # Doubleword): ``model_pricing`` is left empty so cost rows are $0/absent
    # rather than WRONG, and this flag stops the placeholder silently shipping
    # as if real console numbers were entered.
    pricing_confirmed: bool = True
    # Per-model overrides; provider default applies otherwise.
    model_context_windows: dict[str, int] = {}
    # OpenAI-compatible endpoint for the litellm (user-sim) route.
    openai_base_url: Optional[str] = None
    # Official pricing per native model id; registered with litellm so
    # cost reports are accurate instead of the $0 unpriced fallback.
    model_pricing: dict[str, ModelPricing] = {}
    # Native model ids that REJECT /v1/messages requests without
    # thinking={"type": "enabled", ...} (probed live; kimi-k2.7-code does).
    # Drives the SDK session thinking config and the autopsy request shape
    # (forced tool_choice is incompatible with thinking — autopsy switches
    # to tool_choice=auto + the text-JSON fallback).
    models_requiring_thinking: list[str] = []


# kimi-k2.7-code numbers from the official pricing page (2026-06):
# context window 262,144 tokens; input $0.95/M (cache hit $0.19/M),
# output $4.00/M.
_KIMI_K27_CODE_PRICING = ModelPricing(
    input_cost_per_token=0.95e-6,
    output_cost_per_token=4.00e-6,
    cache_read_input_token_cost=0.19e-6,
)

# z.ai GLM pricing (z.ai pricing docs, 2026-06), USD per token. GLM-5.1 and
# GLM-5.2 share the same published row but get DISTINCT instances so a future
# edit to one cannot silently mutate the other (pydantic models are mutable).
# GLM-5.x: input $1.40/M, cache-hit $0.26/M, output $4.40/M.
_GLM52_PRICING = ModelPricing(
    input_cost_per_token=1.40e-6,
    output_cost_per_token=4.40e-6,
    cache_read_input_token_cost=0.26e-6,
)
_GLM51_PRICING = ModelPricing(
    input_cost_per_token=1.40e-6,
    output_cost_per_token=4.40e-6,
    cache_read_input_token_cost=0.26e-6,
)
# GLM-4.7: input $0.60/M, cache-hit $0.11/M, output $2.20/M.
_GLM47_PRICING = ModelPricing(
    input_cost_per_token=0.60e-6,
    output_cost_per_token=2.20e-6,
    cache_read_input_token_cost=0.11e-6,
)
# GLM-4.6: input $0.60/M, output $2.20/M (no separate cache-hit price — cache
# reads fall back to the input rate in usage._safe_cost).
_GLM46_PRICING = ModelPricing(
    input_cost_per_token=0.60e-6,
    output_cost_per_token=2.20e-6,
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
        models_requiring_thinking=["kimi-k2.7-code"],
    ),
    # DEV-1580: z.ai (Zhipu GLM). Anthropic-compatible endpoint + an
    # OpenAI-compatible endpoint for the user-sim route, same shape as
    # Moonshot. GLM-5.2 ships a 1M-token window (z.ai model card); the rest
    # use the 200K default (GLM-4.7/4.6 firm windows vary across sources —
    # default to 200K rather than bake in a suspect value). Unlike
    # kimi-k2.7-code, NO GLM model requires thinking on /v1/messages, so
    # models_requiring_thinking stays at its [] default.
    "zai": ProviderSpec(
        key="zai",
        base_url="https://api.z.ai/api/anthropic",
        base_url_env="BIRD_ZAI_ANTHROPIC_BASE_URL",
        api_format="anthropic",
        auth_env="ZAI_API_KEY",
        default_context_window=200_000,
        model_context_windows={"glm-5.2": 1_000_000},
        openai_base_url="https://api.z.ai/api/paas/v4",
        model_pricing={
            "glm-5.2": _GLM52_PRICING,
            "glm-5.1": _GLM51_PRICING,
            "glm-4.7": _GLM47_PRICING,
            "glm-4.6": _GLM46_PRICING,
        },
    ),
    # DEV-1604: Doubleword (api.doubleword.ai) — OpenAI-Chat-Completions ONLY,
    # no Anthropic endpoint. `claude_sdk*` agents reach it through the local
    # Anthropic⇄OpenAI bridge proxy, which sets ANTHROPIC_BASE_URL via
    # `base_url_env`. The GLM-5.2 native id is the FP8 build `zai-org/GLM-5.2-FP8`
    # (verified live via GET /v1/models); it flows through `_split` / the SDK as
    # a multi-slash native id (only the FIRST slash splits provider from id).
    #
    # MUST-CONFIRM (Doubleword console — not published): exact per-token price
    # and the true context window for GLM-5.2-FP8. Until then pricing is UNSET
    # (`pricing_confirmed=False`, cost rows $0 not wrong) and the window mirrors
    # Doubleword's published GLM-5.1 (~198K, max 202,752 → 200K).
    "doubleword": ProviderSpec(
        key="doubleword",
        base_url=None,  # no Anthropic endpoint — the proxy supplies it
        base_url_env="BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL",
        api_format="openai",
        auth_env="DOUBLEWORD_API_KEY",
        default_context_window=200_000,  # placeholder — MUST-CONFIRM console
        openai_base_url="https://api.doubleword.ai/v1",
        model_pricing={},  # UNSET — MUST-CONFIRM console (see pricing_confirmed)
        pricing_confirmed=False,
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


def requires_thinking(model: str) -> bool:
    spec = get_provider(model)
    if spec is None:
        return False
    _, native_id = _split(model)
    return native_id in spec.models_requiring_thinking


def resolve_base_url(spec: ProviderSpec) -> str:
    """The Anthropic base URL for this provider: the ``base_url_env`` override
    if set, else the static ``base_url``.

    DEV-1604: an OpenAI-only provider has ``base_url=None`` and reaches the SDK
    only through the bridge proxy, which sets ``base_url_env`` to its loopback
    URL. If neither is present (or the override is an empty string — a common
    misconfig), this raises rather than letting a falsy ``None``/`""`` leak to
    any consumer (``sdk_session_env`` AND ``eval/autopsy``)."""
    url = os.environ.get(spec.base_url_env) or spec.base_url
    if not url:
        raise RuntimeError(
            f"{spec.key}: no Anthropic base URL — {spec.base_url_env} is unset "
            f"and the provider has no direct Anthropic endpoint "
            f"(api_format={spec.api_format!r}). An OpenAI-only provider must run "
            f"behind the bridge proxy, which sets {spec.base_url_env} to its "
            f"loopback URL before the SDK session is built."
        )
    return url


def per_token_openai_target(model: str) -> tuple[str, str, str]:
    """``(openai_base_url, native_id, auth_env)`` for the bridge's upstream POST.

    The bridge proxy POSTs Anthropic-translated requests to
    ``<openai_base_url>/chat/completions`` as ``openai/<native_id>`` with the
    provider key from ``auth_env``. Raises if ``model`` is not a registry
    provider exposing an OpenAI-compatible endpoint."""
    spec = get_provider(model)
    if spec is None or not spec.openai_base_url:
        raise ValueError(
            f"{model!r} has no OpenAI-compatible endpoint for the bridge proxy"
        )
    _, native_id = _split(model)
    return spec.openai_base_url, native_id, spec.auth_env


def agent_needs_bridge(model: str, zai_billing: "str | None") -> bool:
    """True iff the ``claude_sdk`` agent for ``model`` must route through the
    Anthropic⇄OpenAI bridge proxy.

    * OpenAI-only providers (``api_format=="openai"``, e.g. Doubleword) have NO
      Anthropic endpoint — they ALWAYS need the bridge, billing-agnostic.
    * z.ai needs it ONLY when the operator opts into per-token billing
      (``zai_billing=="per-token"``); the default coding-plan path keeps its
      direct Anthropic endpoint.
    * Everything else (Moonshot anthropic-format, Anthropic-proper) never
      bridges."""
    spec = get_provider(model)
    if spec is None:
        return False
    if spec.api_format == "openai":
        return True
    return spec.key == "zai" and zai_billing == "per-token"


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

    Codex r2: the Claude SDK subprocess inherits the parent process's
    environment by default — an ambient ``ANTHROPIC_API_KEY`` or
    ``CLAUDE_CODE_OAUTH_TOKEN`` in the developer's shell would survive
    and the SDK would silently authenticate against Anthropic instead
    of the provider's ``ANTHROPIC_BASE_URL`` endpoint. Cloud actors get
    the ``_assert_actor_oauth_invariant`` guard; local SDK runs do not.
    Emit empty-string overrides for both env vars so the subprocess
    sees them as unset — anything truthy here loses to our explicit
    provider key on ``ANTHROPIC_AUTH_TOKEN``.
    """
    spec = get_provider(model)
    if spec is None:
        raise ValueError(f"{model!r} is not a registry open-weight model")
    return {
        "ANTHROPIC_API_KEY": "",
        "CLAUDE_CODE_OAUTH_TOKEN": "",
        "ANTHROPIC_BASE_URL": resolve_base_url(spec),
        "ANTHROPIC_AUTH_TOKEN": provider_api_key(spec),
    }


def litellm_route(
    model: str, *, caller_api_key: str | None = None,
) -> tuple[str, dict[str, str]]:
    """(litellm_model, extra_kwargs) for the user-sim path.

    Registry providers route through litellm's generic OpenAI-compatible
    driver (``openai/<native_id>`` + ``api_base``) — robust regardless of
    whether litellm ships a dedicated provider for them.

    CR r1: ``caller_api_key`` takes precedence over the env-var lookup.
    Without it, a caller threading an explicit ``api_key=`` through
    ``acompletion_tracked`` still hit ``provider_api_key(spec)`` first
    and could raise on an unset env var — even though the explicit key
    would have been preserved by ``kwargs.setdefault`` downstream.
    """
    spec = get_provider(model)
    if spec is None or not spec.openai_base_url:
        return model, {}
    _, native_id = _split(model)
    api_key = caller_api_key if caller_api_key else provider_api_key(spec)
    return (
        f"openai/{native_id}",
        {"api_base": spec.openai_base_url, "api_key": api_key},
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
