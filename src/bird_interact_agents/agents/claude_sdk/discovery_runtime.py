"""DEV-1581 R2: drive a task with two persistent hermetic SDK clients.

R2 replaces the SDK-subagent split with two persistent ``ClaudeSDKClient``s in
ONE process — the main loop and a long-lived *warm* discovery client — bridged
by the in-process ``ask_discovery`` tool. This module owns the orchestration
the four ``claude_sdk_otf*_v1`` agents share verbatim: open both clients under
one exit stack (discovery FIRST so it closes LAST), publish the warm
:class:`DiscoveryChannel` into the per-task context for the ``ask_discovery``
native, then run the main receive loop.

Each client routes through :func:`hermetic_claude_sdk_session` (the DEV-1579
choke point) so the second CLI subprocess gets the same CLAUDE_CONFIG_DIR
isolation, API-key auth enforcement, MCP parity assertion, and cleanup.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import Callable

from bird_interact_agents.agents.claude_sdk.agent import _ctx
from bird_interact_agents.agents.claude_sdk.context_budget import (
    update_context_tokens,
)
from bird_interact_agents.agents.claude_sdk.discovery_channel import (
    DEFAULT_MAX_DISCOVERY_CALLS,
    DiscoveryChannel,
)
from bird_interact_agents.agents.claude_sdk.sdk_env import (
    hermetic_claude_sdk_session,
)

logger = logging.getLogger(__name__)


async def run_main_with_discovery(
    *,
    model: str,
    accum,
    usage_tracker,
    context_state: dict,
    main_mcp_servers: dict,
    discovery_mcp_servers: dict,
    build_main_options: Callable[[dict], object],
    build_discovery_options: Callable[[dict], object],
    initial_query: str,
    trajectory: list,
    max_discovery_calls: int = DEFAULT_MAX_DISCOVERY_CALLS,
    enter_cm_factory: Callable[[], object] | None = None,
    on_main_message: Callable[[object, int], None] | None = None,
) -> None:
    """Run one task across a warm discovery client and the main client.

    Discovery is entered FIRST under the shared ``AsyncExitStack`` so it tears
    down LAST — main closes before discovery, and an ``ask_discovery`` still in
    flight while main is shutting down never touches a closed discovery client.
    The :class:`DiscoveryChannel` is published into ``_ctx['_discovery']`` so
    the in-process ``ask_discovery`` native can reach it. Both clients' usage
    flows into the SAME ``accum`` — main via ``usage_tracker`` in the loop
    below; each discovery ask via its own fresh tracker inside the channel.

    ``enter_cm_factory`` is threaded to BOTH hermetic sessions'
    ``enter_cm_factory`` so the DEV-1561 ``otf_timer``-around-``__aenter__``
    instrumentation covers EACH ``claude`` CLI subprocess spawn (discovery is
    entered first, where an initialize-handshake hang surfaces first; main is
    entered second). Each enter emits its own ``run_task.sdk_client_enter``
    span. ``on_main_message(msg, seq)`` (1-based seq) is invoked per main
    message before usage accounting, for per-message logging.

    Mutates ``trajectory`` (appends one structured entry per main message) and
    ``accum`` (usage). Does not finalize ``usage_tracker`` — the caller does, so
    the crash path can finalize once.
    """
    enter_kwargs: dict = {}
    if enter_cm_factory is not None:
        enter_kwargs["enter_cm_factory"] = enter_cm_factory
    async with contextlib.AsyncExitStack() as stack:
        # Discovery FIRST → closes LAST (LIFO unwind). It is also the first
        # CLI subprocess spawned, so its enter-timing fires first.
        discovery_client = await stack.enter_async_context(
            hermetic_claude_sdk_session(
                model,
                mcp_servers=discovery_mcp_servers,
                build_options=build_discovery_options,
                **enter_kwargs,
            )
        )
        channel = DiscoveryChannel(
            client=discovery_client,
            usage_accum=accum,
            model=model,
            max_calls=max_discovery_calls,
        )
        # Publish the warm channel for the in-process ask_discovery native, and
        # restore the prior value on exit so a closed channel never lingers in a
        # reused context (CodeRabbit PR #56).
        prev_discovery = _ctx.get("_discovery")
        _ctx["_discovery"] = channel
        try:
            main_client = await stack.enter_async_context(
                hermetic_claude_sdk_session(
                    model,
                    mcp_servers=main_mcp_servers,
                    build_options=build_main_options,
                    **enter_kwargs,
                )
            )

            await main_client.query(initial_query)
            seq = 0
            async for msg in main_client.receive_response():
                seq += 1
                if on_main_message is not None:
                    on_main_message(msg, seq)
                try:
                    _data: object = dataclasses.asdict(msg)
                except Exception:  # noqa: BLE001
                    _data = str(msg)
                trajectory.append({"type": str(type(msg).__name__), "data": _data})
                usage_tracker.observe(msg)
                update_context_tokens(context_state, msg)
        finally:
            _ctx["_discovery"] = prev_discovery
