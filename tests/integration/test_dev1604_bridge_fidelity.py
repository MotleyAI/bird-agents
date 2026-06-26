"""DEV-1604 BLOCKING fidelity gate: real Agent SDK ⇄ local bridge proxy.

The unit tests stub `litellm.anthropic_messages`, so they CANNOT prove the
bridge survives the full Claude Agent SDK tool loop against a real upstream.
This gate does, per target provider (Doubleword AND z.ai per-token):

1. native-id route selection (the SDK sends the bare native id; the proxy
   maps it to the provider's OpenAI endpoint);
2. streaming SSE (the SDK only streams `/v1/messages`);
3. ≥1 MCP tool call + a `tool_result` continuation (the in-process tool
   records its invocation; the model uses the result in its final answer);
4. multi-turn `tool_use`/`tool_result` (the loop runs to a final text turn);
5. terminal `ResultMessage.usage` non-zero (cost accounting works);
6. no z.ai `[1313]` Fair-Usage throttle on the per-token path.

Marked `integration` (excluded from the default suite). Needs the provider key
+ a non-zero balance. Each provider skips independently if its key is absent.
"""

from __future__ import annotations

import os

import pytest

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

from bird_interact_agents.cloud import bridge_proxy
from bird_interact_agents.model_string import native_model_id
from bird_interact_agents.agents.claude_sdk.sdk_env import (
    hermetic_claude_sdk_session,
)

pytestmark = [pytest.mark.integration]

_PROVIDERS = [
    pytest.param(
        # (model, no_subscription_auth, key_env). no_subscription_auth=True is
        # the per-token/bridge path for z.ai; Doubleword bridges regardless.
        "doubleword/zai-org/GLM-5.2-FP8", True, "DOUBLEWORD_API_KEY",
        id="doubleword",
    ),
    pytest.param(
        "zai/glm-5.2", True, "ZAI_API_KEY", id="zai-per-token",
    ),
]


@pytest.fixture(autouse=True)
def _clean_ambient_creds(monkeypatch):
    # A registry run must not silently authenticate against Anthropic.
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN",
                "ANTHROPIC_AUTH_TOKEN", "BIRD_INTERACT_SUBSCRIPTION_AUTH"):
        monkeypatch.delenv(var, raising=False)
    yield
    bridge_proxy.terminate_local_proxies()


@pytest.mark.parametrize("model,no_subscription_auth,key_env", _PROVIDERS)
@pytest.mark.asyncio
async def test_bridge_full_sdk_tool_loop(model, no_subscription_auth, key_env):
    if not os.environ.get(key_env):
        pytest.skip(f"needs {key_env} (+ balance) for the {model} bridge gate")

    weather_calls: list[str] = []

    @tool("get_weather", "Get the weather for a city.", {"city": str})
    async def _weather(args):
        weather_calls.append(args.get("city", ""))
        return {"content": [{"type": "text", "text": "18C and sunny"}]}

    server = create_sdk_mcp_server(
        name="weather-tools", version="1.0.0", tools=[_weather]
    )
    mcp_servers = {"weather-tools": server}

    # Bring up the real loopback proxy and point the registry override at it.
    url = bridge_proxy.ensure_bridge_proxy_for_actor(
        model, {"no_subscription_auth": no_subscription_auth}
    )
    assert url.startswith("http://127.0.0.1:")

    def _build_options(opt_kwargs: dict) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            **opt_kwargs,
            mcp_servers=mcp_servers,
            allowed_tools=["mcp__weather-tools__get_weather"],
            tools=[],
            setting_sources=[],
            model=native_model_id(model),
            max_turns=4,
        )

    texts: list[str] = []
    usage_total = 0
    cost_usd = 0.0
    async with hermetic_claude_sdk_session(
        model, mcp_servers=mcp_servers, build_options=_build_options,
    ) as client:
        await client.query(
            "What's the weather in Paris? You MUST call the get_weather tool, "
            "then tell me the result in one sentence."
        )
        async for msg in client.receive_response():
            name = type(msg).__name__
            blob = repr(msg)
            # (6) z.ai per-token must NOT hit the Fair-Usage throttle.
            assert "1313" not in blob, f"z.ai [1313] throttle surfaced: {blob}"
            if name == "AssistantMessage":
                for block in getattr(msg, "content", []) or []:
                    if getattr(block, "type", None) == "text" or hasattr(
                        block, "text"
                    ):
                        texts.append(getattr(block, "text", "") or "")
            if name == "ResultMessage":
                # The SDK's usage dict key names vary; sum every integer token
                # field rather than assume input_tokens/output_tokens.
                usage = getattr(msg, "usage", None)
                if isinstance(usage, dict):
                    usage_total = sum(
                        v for v in usage.values() if isinstance(v, int)
                    )
                cost_usd = getattr(msg, "total_cost_usd", 0.0) or 0.0

    # (3)(4) the tool was actually invoked and the loop continued past it.
    assert weather_calls, "the MCP get_weather tool was never called"
    final = " ".join(texts).lower()
    assert "sunny" in final or "18" in final, (
        f"model did not use the tool_result in its answer: {texts!r}"
    )
    # (5) cost accounting is live — non-zero token usage OR a non-zero cost.
    assert usage_total > 0 or cost_usd > 0, (
        f"terminal ResultMessage cost accounting was zero "
        f"(usage_total={usage_total}, cost_usd={cost_usd})"
    )
