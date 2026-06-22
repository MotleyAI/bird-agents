"""DEV-1581 R2: the warm persistent-discovery bridge.

R2 replaces the SDK-subagent split with two persistent ``ClaudeSDKClient``s in
one process: the main loop, and a long-lived *discovery* client that main
reaches through an in-process ``ask_discovery(question)`` tool. ``DiscoveryChannel``
is that bridge — it forwards a question to the warm discovery client and
returns its text, while guaranteeing the operational invariants the Codex
review pinned:

* **Single-flight** — one ``asyncio.Lock`` serialises the whole
  query/receive cycle so two ``ask_discovery`` calls never drain the discovery
  stream concurrently (which would corrupt the warm session).
* **Call cap** — after ``max_calls`` the channel stops querying discovery and
  returns :data:`DISCOVERY_CALL_CAP_MESSAGE` (R2's bound on warm follow-ups —
  the analog of the old per-task ``Task`` cap). A *failed* call still consumes
  a slot so a broken discovery client can't be retried unboundedly.
* **Per-stream usage** — a FRESH usage tracker is created and finalised for
  every ``ask`` (the real :class:`SdkUsageTracker` is idempotent after one
  ``finalize``), so multi-call discovery usage is fully counted.
* **No raises into main** — a dead / empty / raising discovery stream yields a
  usable string, never a hang and never an exception into the main loop.

The lifecycle helpers (:func:`discovery_session`, :func:`open_main_and_discovery`)
guarantee the second CLI subprocess is always closed — and, when paired with
main, that discovery closes *after* main so an in-flight ``ask_discovery``
during main shutdown never touches a closed client.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


#: Returned by :meth:`DiscoveryChannel.ask` once the per-task call cap is hit.
#: A stable sentinel (asserted by tests) telling main to proceed on what it has.
DISCOVERY_CALL_CAP_MESSAGE = (
    "[discovery call limit reached for this task] Proceed with the handoff you "
    "already have: re-run your candidate query, pivot your operationalisation, "
    "or ask the user — do not wait for more discovery."
)

#: Default per-task cap on ``ask_discovery`` calls.
DEFAULT_MAX_DISCOVERY_CALLS = 10

#: Default per-ask wall-clock backstop (seconds). The single-flight lock is held
#: for the whole query/drain cycle, so a stalled discovery stream would block
#: every later ``ask`` indefinitely. This generous cap converts an indefinite
#: hang into a usable sentinel without cutting off a legitimately-long first
#: introspection sweep (CodeRabbit PR #56).
DEFAULT_DISCOVERY_ASK_TIMEOUT_S = 600.0


def _default_is_result(msg: Any) -> bool:
    return type(msg).__name__ == "ResultMessage"


def _default_tracker_factory(accum: Any, model: str) -> Any:
    # Lazy import to avoid a module-load cycle with the agent module (which
    # imports this module for the ``ask_discovery`` native).
    from bird_interact_agents.agents.claude_sdk.agent import SdkUsageTracker

    return SdkUsageTracker(accum, model)


def _extract_text(msg: Any) -> str:
    parts: list[str] = []
    for block in getattr(msg, "content", None) or []:
        if type(block).__name__ == "TextBlock":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


class DiscoveryChannel:
    """Bridge from the main loop's ``ask_discovery`` tool to the warm
    persistent discovery client."""

    def __init__(
        self,
        *,
        client: Any,
        usage_accum: Any,
        model: str,
        max_calls: int = DEFAULT_MAX_DISCOVERY_CALLS,
        tracker_factory: Callable[[Any, str], Any] | None = None,
        is_result: Callable[[Any], bool] | None = None,
        timeout_s: float | None = DEFAULT_DISCOVERY_ASK_TIMEOUT_S,
    ) -> None:
        self._client = client
        self._usage_accum = usage_accum
        self._model = model
        self._max_calls = max_calls
        self._tracker_factory = tracker_factory or _default_tracker_factory
        self._is_result = is_result or _default_is_result
        self._timeout_s = timeout_s
        self._lock = asyncio.Lock()
        self._calls = 0
        self._closed = False

    @property
    def calls(self) -> int:
        return self._calls

    async def ask(self, question: str) -> str:
        """Forward ``question`` to the warm discovery client and return its
        text answer (or a usable sentinel on cap / error / empty)."""
        async with self._lock:
            if self._calls >= self._max_calls:
                return DISCOVERY_CALL_CAP_MESSAGE
            # Count the attempt up-front so failures consume a slot too.
            self._calls += 1
            tracker = self._tracker_factory(self._usage_accum, self._model)
            parts: list[str] = []
            agen = None

            async def _drive() -> None:
                nonlocal agen
                await self._client.query(question)
                agen = self._client.receive_response()
                async for msg in agen:
                    tracker.observe(msg)
                    parts.append(_extract_text(msg))
                    if self._is_result(msg):
                        break

            try:
                if self._timeout_s is not None:
                    await asyncio.wait_for(_drive(), timeout=self._timeout_s)
                else:
                    await _drive()
            except (asyncio.TimeoutError, TimeoutError):
                # Backstop against a stalled stream holding the single-flight
                # lock forever. wait_for cancels _drive; the finally below
                # closes the half-open stream.
                logger.warning(
                    "ask_discovery timed out after %ss", self._timeout_s,
                )
                return f"[discovery timeout: no response within {self._timeout_s}s]"
            except Exception as exc:  # noqa: BLE001 - never raise into main
                logger.warning("ask_discovery stream failed: %s", exc)
                return f"[discovery error: {exc}]"
            finally:
                # Explicitly close the response stream so a break never leaves
                # a half-open receive cycle on the warm discovery session (and
                # so the single-flight invariant truly releases the client).
                aclose = getattr(agen, "aclose", None)
                if aclose is not None:
                    with contextlib.suppress(Exception):
                        await aclose()
                tracker.finalize()
            text = "".join(parts).strip()
            if not text:
                return "[discovery returned no content for this question]"
            return text

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._client.aclose()


@contextlib.asynccontextmanager
async def discovery_session(
    *,
    open_client: Callable[[], Any],
    usage_accum: Any,
    model: str,
    max_calls: int = DEFAULT_MAX_DISCOVERY_CALLS,
    tracker_factory: Callable[[Any, str], Any] | None = None,
    is_result: Callable[[Any], bool] | None = None,
):
    """Open the warm discovery client, yield a :class:`DiscoveryChannel`, and
    guarantee the client is closed on EVERY exit path (success / exception /
    cancellation) so the second CLI subprocess is never orphaned."""
    client = await open_client()
    channel = DiscoveryChannel(
        client=client,
        usage_accum=usage_accum,
        model=model,
        max_calls=max_calls,
        tracker_factory=tracker_factory,
        is_result=is_result,
    )
    try:
        yield channel
    finally:
        await channel.aclose()


@contextlib.asynccontextmanager
async def open_main_and_discovery(
    *,
    open_main: Callable[[], Any],
    open_discovery: Callable[[], Any],
    usage_accum: Any,
    model: str,
    max_calls: int = DEFAULT_MAX_DISCOVERY_CALLS,
    tracker_factory: Callable[[Any, str], Any] | None = None,
    is_result: Callable[[Any], bool] | None = None,
):
    """Enter the main client and the warm discovery client under ONE
    ``AsyncExitStack`` and yield ``(main_client, discovery_channel)``.

    Discovery is entered FIRST so it tears down LAST — i.e. main closes before
    discovery — so an ``ask_discovery`` still in flight while the main loop is
    shutting down never touches a closed discovery client.
    """
    async with contextlib.AsyncExitStack() as stack:
        discovery_client = await open_discovery()
        channel = DiscoveryChannel(
            client=discovery_client,
            usage_accum=usage_accum,
            model=model,
            max_calls=max_calls,
            tracker_factory=tracker_factory,
            is_result=is_result,
        )
        stack.push_async_callback(channel.aclose)
        main_client = await open_main()
        stack.push_async_callback(main_client.aclose)
        yield main_client, channel
