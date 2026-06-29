"""DEV-1589: `SdkUsageTracker` gains a back-compat `scope` param.

The build-time claude_sdk encoder reuses `SdkUsageTracker` per re-prompt cycle
(the DEV-1581 `DiscoveryChannel` pattern) but must record under
`scope="setup_encoder"` (the reference-build `_setup_usage.json` contract),
NOT the default `scope="agent"`. The param is optional and defaults to
"agent" so every existing caller is unchanged.
"""

from __future__ import annotations

from bird_interact_agents.agents.claude_sdk.agent import SdkUsageTracker
from bird_interact_agents.usage import TokenUsage


class ResultMessage:
    """Name MUST be exactly 'ResultMessage' — SdkUsageTracker.observe() commits
    only `type(msg).__name__ == 'ResultMessage'`."""

    def __init__(self, usage):
        self.usage = usage


_USAGE = {
    "input_tokens": 100,
    "output_tokens": 20,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 3,
}


def test_default_scope_is_agent():
    accum = TokenUsage()
    t = SdkUsageTracker(accum, "anthropic/claude-opus-4-7")
    t.observe(ResultMessage(_USAGE))
    t.finalize()
    rows = {r.scope for r in accum.breakdown}
    assert rows == {"agent"}


def test_explicit_setup_encoder_scope():
    accum = TokenUsage()
    t = SdkUsageTracker(accum, "anthropic/claude-opus-4-7", scope="setup_encoder")
    t.observe(ResultMessage(_USAGE))
    t.finalize()
    rows = {r.scope for r in accum.breakdown}
    assert rows == {"setup_encoder"}
    row = next(r for r in accum.breakdown if r.scope == "setup_encoder")
    assert row.prompt_tokens == 100
    assert row.completion_tokens == 20


def test_fresh_tracker_per_cycle_sums_across_cycles():
    """Two cycles, fresh tracker each, same accum → totals SUM (the warm-client
    re-prompt accounting; mirrors DiscoveryChannel.ask)."""
    accum = TokenUsage()
    for _ in range(3):
        t = SdkUsageTracker(accum, "anthropic/claude-opus-4-7", scope="setup_encoder")
        t.observe(ResultMessage(_USAGE))
        t.finalize()
    row = next(r for r in accum.breakdown if r.scope == "setup_encoder")
    assert row.prompt_tokens == 300
    assert row.completion_tokens == 60
