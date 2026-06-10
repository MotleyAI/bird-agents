"""DEV-1535 follow-up — `extract_usage_costs` is the single source of truth
for `(cost_usd_agent, cost_usd_user_sim)` extraction from a usage dict.

Pre-fix the local writer never read the costs at all, and the cloud
writer used the WRONG key names (`cost_usd_agent` / `cost_usd_user_sim`)
while `TokenUsage.model_dump()` emits `agent_cost_usd` / `user_sim_cost_usd`.
Both paths wrote None to every submission annotation since DEV-1515.
"""

from __future__ import annotations

from bird_interact_agents.eval.grade_in_place import extract_usage_costs


def test_extracts_canonical_token_usage_keys():
    """`TokenUsage.model_dump()` happy shape — must read both keys."""
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "agent_cost_usd": 2.34,
        "user_sim_cost_usd": 0.12,
    }
    assert extract_usage_costs(usage) == (2.34, 0.12)


def test_none_input_returns_none_pair():
    """No usage dict at all → (None, None). Common for very-early errors."""
    assert extract_usage_costs(None) == (None, None)


def test_non_dict_input_returns_none_pair():
    """Defensive: usage stuffed with a string / list → (None, None) rather
    than crashing on `.get`."""
    assert extract_usage_costs("usage") == (None, None)
    assert extract_usage_costs(["agent_cost_usd", 1.0]) == (None, None)
    assert extract_usage_costs(42) == (None, None)


def test_missing_keys_return_none():
    """Empty dict OR dict with unrelated keys → (None, None)."""
    assert extract_usage_costs({}) == (None, None)
    assert extract_usage_costs({"prompt_tokens": 10}) == (None, None)


def test_wrong_key_names_return_none():
    """Pins the DEV-1535 contract: the pre-fix wrong-key shape
    (`cost_usd_agent` / `cost_usd_user_sim`) — emitted by NO live caller
    — returns (None, None). This guards against a regression where the
    bug is reintroduced by string-renaming the canonical keys."""
    usage = {
        "cost_usd_agent": 2.34,   # WRONG key — pre-DEV-1535 bug
        "cost_usd_user_sim": 0.12,
    }
    assert extract_usage_costs(usage) == (None, None)


def test_partial_keys_return_partial_pair():
    """One key present, one absent — the present one comes through, the
    other is None. (Realistic: a benchmark with no user-sim never emits
    `user_sim_cost_usd`.)"""
    usage = {"agent_cost_usd": 1.50}
    assert extract_usage_costs(usage) == (1.50, None)
    usage = {"user_sim_cost_usd": 0.05}
    assert extract_usage_costs(usage) == (None, 0.05)


def test_non_numeric_values_return_none():
    """A string in the cost slot → None for that slot (not the literal
    string). Defensive against a hypothetical adapter that stashes a
    label there."""
    usage = {"agent_cost_usd": "1.50", "user_sim_cost_usd": None}
    assert extract_usage_costs(usage) == (None, None)


def test_integer_costs_pass_through():
    """ints are accepted (e.g. zero) — the schema field is
    `Optional[float]` so any number is fine."""
    usage = {"agent_cost_usd": 0, "user_sim_cost_usd": 1}
    assert extract_usage_costs(usage) == (0, 1)
