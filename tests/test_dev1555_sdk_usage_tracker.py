"""DEV-1555: Stage 2 follow-up — replace the per-AssistantMessage
accumulator with a tracker that uses the cumulative ResultMessage.

Diagnosed empirically: the claude-agent-sdk splits a single assistant
TURN into multiple ``AssistantMessage`` events (one per content block:
thinking, text, tool_use, …), every one of which carries the SAME
per-turn ``usage`` dict. Summing them double-counts. The truth lives in
the single ``ResultMessage`` emitted at stream end. (Verified against
both an Opus alien_1 run and an intercepted Kimi probe.)

Real-world impact at diagnosis time:

* Opus alien_1: ResultMessage cache_read = 1.39M; per-AssistantMessage
  sum = 2.21M (1.6× over).
* Opus alien_1: ResultMessage output = 17.6K; per-AssistantMessage sum =
  3.6K (5× under, because output_tokens only fills in on the LAST block
  of a turn while we summed all blocks).
* Kimi alien_1 (r4): ResultMessage cache_read = 3.66M; per-AssistantMessage
  sum = 0 (Moonshot's per-turn AssistantMessage.usage reports cache=0;
  the cumulative is in ResultMessage).

The new ``SdkUsageTracker``:

* prefers ResultMessage.usage at stream end (1 LLM call to ``add_call``);
* on crash (no ResultMessage), falls back to summing per-turn usage
  deduplicated by ``message_id`` / usage-dict identity, sets
  ``accum.partial=True``;
* preserves ``n_calls`` as the number of distinct turns the SDK
  delivered, not the number of streamed events.
"""

from __future__ import annotations

from bird_interact_agents.agents.claude_sdk.agent import SdkUsageTracker
from bird_interact_agents.usage import TokenUsage


class AssistantMessage:
    def __init__(self, usage, message_id=None, parent_tool_use_id=None):
        self.usage = usage
        self.message_id = message_id
        self.parent_tool_use_id = parent_tool_use_id


class ResultMessage:
    def __init__(self, usage):
        self.usage = usage


def _assistant(usage_dict, *, message_id=None):
    """Build a fake AssistantMessage. All blocks of a turn share the SAME
    usage dict on the live SDK — pass the same dict instance to model
    that."""
    return AssistantMessage(usage_dict, message_id=message_id)


def _result(usage_dict):
    return ResultMessage(usage_dict)


# ---------------------------------------------------------------------------
# Happy path: ResultMessage is the authoritative cumulative.
# ---------------------------------------------------------------------------

def test_result_message_overrides_per_turn_estimate(monkeypatch):
    """Live-Opus shape: 39 AssistantMessages (each duplicated for thinking +
    text/tool_use blocks of the same turn), 1 ResultMessage at end with the
    cumulative. After finalize, accum reflects the ResultMessage cumulative
    — not the per-block sum."""
    accum = TokenUsage()
    tracker = SdkUsageTracker(accum, "anthropic/claude-opus-4-7")

    turn1 = {
        "input_tokens": 6, "cache_creation_input_tokens": 27155,
        "cache_read_input_tokens": 0, "output_tokens": 0,
    }
    turn2 = {
        "input_tokens": 1, "cache_creation_input_tokens": 11763,
        "cache_read_input_tokens": 27276, "output_tokens": 44,
    }
    # 2 blocks per turn (thinking + tool_use), same usage dict shared.
    tracker.observe(_assistant(turn1, message_id="m1"))
    tracker.observe(_assistant(turn1, message_id="m1"))
    tracker.observe(_assistant(turn2, message_id="m2"))
    tracker.observe(_assistant(turn2, message_id="m2"))

    result = {
        "input_tokens": 7, "cache_creation_input_tokens": 38918,
        "cache_read_input_tokens": 27276, "output_tokens": 44,
    }
    tracker.observe(_result(result))

    assert accum.prompt_tokens == 7
    assert accum.cache_write_tokens == 38918
    assert accum.cache_read_tokens == 27276
    assert accum.completion_tokens == 44
    assert accum.n_calls == 2  # two distinct turns, not four AssistantMessages
    assert accum.partial is False


def test_dedup_via_usage_dict_identity_when_no_message_id():
    """The live SDK exposes one usage DICT instance per turn — blocks of
    the same turn share it by identity. Fallback dedup key when no
    message_id."""
    accum = TokenUsage()
    tracker = SdkUsageTracker(accum, "moonshot/kimi-k2.7-code")
    shared = {
        "input_tokens": 38139, "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0, "output_tokens": 0,
    }
    tracker.observe(_assistant(shared))
    tracker.observe(_assistant(shared))
    tracker.observe(_assistant(shared))
    result = {
        "input_tokens": 38139, "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0, "output_tokens": 110,
    }
    tracker.observe(_result(result))
    assert accum.prompt_tokens == 38139
    assert accum.completion_tokens == 110
    assert accum.n_calls == 1


# ---------------------------------------------------------------------------
# Crash path: no ResultMessage. Fall back to summed per-turn estimate;
# mark accum.partial=True so downstream knows.
# ---------------------------------------------------------------------------

def test_crash_path_falls_back_to_summed_per_turn_estimate():
    accum = TokenUsage()
    tracker = SdkUsageTracker(accum, "moonshot/kimi-k2.7-code")
    turn1 = {"input_tokens": 100, "output_tokens": 10,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    turn2 = {"input_tokens": 200, "output_tokens": 20,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 50}
    tracker.observe(_assistant(turn1, message_id="m1"))
    tracker.observe(_assistant(turn1, message_id="m1"))
    tracker.observe(_assistant(turn2, message_id="m2"))

    tracker.finalize()  # caller invokes this from the except-path

    assert accum.prompt_tokens == 300
    assert accum.completion_tokens == 30
    assert accum.cache_read_tokens == 50
    assert accum.n_calls == 2
    assert accum.partial is True


def test_finalize_with_no_messages_is_noop():
    accum = TokenUsage()
    tracker = SdkUsageTracker(accum, "moonshot/kimi-k2.7-code")
    tracker.finalize()
    assert accum.n_calls == 0
    assert accum.partial is False


def test_finalize_is_idempotent():
    accum = TokenUsage()
    tracker = SdkUsageTracker(accum, "moonshot/kimi-k2.7-code")
    result = {
        "input_tokens": 100, "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 50, "output_tokens": 10,
    }
    tracker.observe(_result(result))
    tracker.finalize()
    tracker.finalize()
    assert accum.prompt_tokens == 100
    assert accum.n_calls == 1


# ---------------------------------------------------------------------------
# Subagent (Task) messages still counted — DEV-1555 Stage-1 contract.
# ---------------------------------------------------------------------------

def test_subagent_messages_count_toward_total():
    """Stage-1 partition makes the discovery subagent emit AssistantMessages
    too (carrying ``parent_tool_use_id``). They contribute to the same
    cumulative the ResultMessage reports — no special handling needed."""
    accum = TokenUsage()
    tracker = SdkUsageTracker(accum, "anthropic/claude-sonnet-4-6")
    sub = {"input_tokens": 5, "output_tokens": 10,
           "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}
    main = {"input_tokens": 3, "output_tokens": 7,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 200}
    sub_msg = AssistantMessage(sub, message_id="sub-1", parent_tool_use_id="tu_42")
    tracker.observe(sub_msg)
    tracker.observe(_assistant(main, message_id="main-1"))
    result = {
        "input_tokens": 8, "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 200, "output_tokens": 17,
    }
    tracker.observe(_result(result))
    assert accum.completion_tokens == 17
    assert accum.cache_read_tokens == 1000
    assert accum.cache_write_tokens == 200
    assert accum.n_calls == 2
