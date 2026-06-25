"""Tiny helpers for routing LiteLLM-style model strings to each framework's
preferred form.

We use LiteLLM's slash-separated `provider/model_id` convention as the
canonical form across our CLI and configs, because LiteLLM already maintains
the provider catalogue (Cerebras, OpenRouter, Anthropic, Z.ai, Fireworks,
DeepInfra, ...). PydanticAI uses the same set of providers but spells them
with a colon (`provider:model_id`), and OpenRouter model ids contain
embedded slashes — so we only swap the FIRST slash when converting.
"""

from __future__ import annotations


def is_anthropic(model: str) -> bool:
    """`anthropic/claude-sonnet-4-5` -> True. Anything else -> False.

    Used to short-circuit the Claude-locked claude_sdk framework when the
    user has selected a non-Anthropic model.
    """
    return model.split("/", 1)[0] == "anthropic"


def to_pydantic_ai(model: str) -> str:
    """Convert a LiteLLM-style string to PydanticAI's `provider:model_id` form.

    Idempotent: an input already in PydanticAI form (provider contains a
    colon before the first slash) is returned unchanged. Otherwise only the
    first slash is swapped — OpenRouter model ids contain a slash themselves
    (e.g. `z-ai/glm-4.7-flash`).

        cerebras/zai-glm-4.7              -> cerebras:zai-glm-4.7
        openrouter/z-ai/glm-4.7-flash     -> openrouter:z-ai/glm-4.7-flash
        anthropic/claude-sonnet-4-5       -> anthropic:claude-sonnet-4-5
        openrouter:z-ai/glm-4.7-flash     -> openrouter:z-ai/glm-4.7-flash  (unchanged)
    """
    provider, sep, rest = model.partition("/")
    if not sep:
        return model
    if ":" in provider:
        return model
    return f"{provider}:{rest}"


def encoder_version_slug(model: str) -> str:
    """Derive the default OTF-reference version label from a model string.

    DEV-1605: strip the provider prefix, strip a leading ``claude-`` (so the
    common Anthropic case reads ``opus-4-7`` not ``claude-opus-4-7``), and
    replace any remaining ``/`` with ``__`` so the result is a safe single
    filesystem path segment.

        anthropic/claude-opus-4-7      -> opus-4-7
        zai/glm-5.2                    -> glm-5.2
        openai/gpt-5.2                 -> gpt-5.2
        openrouter/z-ai/glm-4.7-flash  -> z-ai__glm-4.7-flash

    Raises ``ValueError`` when the transformation yields an empty label (e.g.
    ``anthropic/claude-``), which would otherwise be an unusable path segment.
    """
    rest = native_model_id(model)
    if rest.startswith("claude-"):
        rest = rest[len("claude-"):]
    slug = rest.replace("/", "__")
    if not slug or slug in (".", "..") or "\\" in slug:
        raise ValueError(
            f"encoder_version_slug({model!r}) produced an invalid label "
            f"{slug!r}; pass an explicit --version."
        )
    return slug


def resolve_encode_version(
    explicit: str | None, model: str | None,
) -> str | None:
    """DEV-1605: the single source of truth for an otf_encode run's version
    label — an explicit ``--version`` wins, else the encoder-model slug, else
    ``None`` (no model known). Centralised so the cloud actor, upload-back,
    laptop merge, and the in-cloud encoder all resolve the SAME version (a
    drift here would build into one dir and upload from another)."""
    if explicit:
        return explicit
    return encoder_version_slug(model) if model else None


def native_model_id(model: str) -> str:
    """Strip the provider prefix: `anthropic/claude-sonnet-4-5` -> `claude-sonnet-4-5`.

    Some framework-native classes (e.g. `agno.models.anthropic.Claude(id=...)`)
    expect the bare model id without a provider prefix.
    """
    _, sep, rest = model.partition("/")
    return rest if sep else model


# Providers whose models can be used with PydanticAI by passing them through
# OpenAI's chat-completion wire format. Each entry is the OpenAI-compatible
# base URL plus the env var holding the API key.
_OPENAI_COMPAT_PROVIDERS: list[tuple[str, str, str]] = [
    ("deepinfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
    ("doubleword", "https://api.doubleword.ai/v1", "DOUBLEWORD_API_KEY"),
]


def build_pydantic_ai_model(model: str):
    """Return either a model-id string or a PydanticAI Model instance.

    For pydantic-ai-native providers (anthropic, cerebras, openrouter, ...)
    the colon-form string is enough — Agent(model="cerebras:zai-glm-4.7")
    works out of the box. For OpenAI-compatible providers that PydanticAI
    doesn't ship a dedicated provider for (DeepInfra), build an
    `OpenAIChatModel` pointing at the provider's chat-completion endpoint.
    """
    import os

    provider, sep, rest = model.partition("/")
    if not sep:
        return model
    if ":" in provider:
        return model

    for prov, base_url, env_var in _OPENAI_COMPAT_PROVIDERS:
        if provider == prov:
            api_key = os.environ.get(env_var)
            if not api_key:
                raise RuntimeError(
                    f"{env_var} is not set; required to use {prov!r} models."
                )
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            return OpenAIChatModel(
                rest,
                provider=OpenAIProvider(base_url=base_url, api_key=api_key),
            )

    return f"{provider}:{rest}"
