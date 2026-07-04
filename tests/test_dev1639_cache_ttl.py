"""DEV-1639: configurable prompt-cache TTL (5m default / 1h) for claude_sdk*.

One env signal, ``BIRD_INTERACT_CACHE_TTL``, is read by BOTH the SDK-session env
layer (which injects the CLI's ENABLE/FORCE knobs, hermetically) and
``usage._safe_cost`` (which prices cache writes at the matching 5m/1h tier). A
separate ``BIRD_INTERACT_DISABLE_PROMPT_CACHE`` signal carries ``--no-prompt-cache``.
These are wired from ``--cache-ttl`` on both the local runner and the cloud CLI.
"""

from __future__ import annotations

import pytest

from bird_interact_agents import provider_registry as pr
from bird_interact_agents import usage
from bird_interact_agents.agents.claude_sdk import sdk_env


_DW = "doubleword/zai-org/GLM-5.2-FP8"
_ANTHROPIC = "anthropic/claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# selected_cache_ttl resolver
# ---------------------------------------------------------------------------


def test_selected_cache_ttl_default_is_5m(monkeypatch):
    monkeypatch.delenv("BIRD_INTERACT_CACHE_TTL", raising=False)
    assert pr.selected_cache_ttl() == "5m"


def test_selected_cache_ttl_1h(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "1h")
    assert pr.selected_cache_ttl() == "1h"


def test_selected_cache_ttl_explicit_5m(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "5m")
    assert pr.selected_cache_ttl() == "5m"


@pytest.mark.parametrize("junk", ["", "60m", "1hour", "garbage", "0"])
def test_selected_cache_ttl_unknown_falls_back_to_5m(monkeypatch, junk):
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", junk)
    assert pr.selected_cache_ttl() == "5m"


# ---------------------------------------------------------------------------
# cache_control_env — hermetic knob injection (all three set explicitly)
# ---------------------------------------------------------------------------

_KNOBS = ("ENABLE_PROMPT_CACHING_1H", "FORCE_PROMPT_CACHING_5M", "DISABLE_PROMPT_CACHING")


def test_cache_control_env_default_forces_5m(monkeypatch):
    monkeypatch.delenv("BIRD_INTERACT_CACHE_TTL", raising=False)
    monkeypatch.delenv("BIRD_INTERACT_DISABLE_PROMPT_CACHE", raising=False)
    env = sdk_env.cache_control_env()
    assert set(env) == set(_KNOBS)  # all three set, hermetic
    assert env["FORCE_PROMPT_CACHING_5M"] == "1"
    assert env["ENABLE_PROMPT_CACHING_1H"] == ""
    assert env["DISABLE_PROMPT_CACHING"] == ""


def test_cache_control_env_1h(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "1h")
    monkeypatch.delenv("BIRD_INTERACT_DISABLE_PROMPT_CACHE", raising=False)
    env = sdk_env.cache_control_env()
    assert env["ENABLE_PROMPT_CACHING_1H"] == "1"
    assert env["FORCE_PROMPT_CACHING_5M"] == ""
    assert env["DISABLE_PROMPT_CACHING"] == ""


def test_cache_control_env_disabled_wins_over_ttl(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "1h")  # ignored when disabled
    monkeypatch.setenv("BIRD_INTERACT_DISABLE_PROMPT_CACHE", "1")
    env = sdk_env.cache_control_env()
    assert env["DISABLE_PROMPT_CACHING"] == "1"
    assert env["ENABLE_PROMPT_CACHING_1H"] == ""
    assert env["FORCE_PROMPT_CACHING_5M"] == ""


def test_cache_control_env_masks_ambient_leak(monkeypatch):
    """An ambient ENABLE_PROMPT_CACHING_1H in the parent env must NOT survive
    when 5m is selected — the knob is set to "" (seen as unset) hermetically."""
    monkeypatch.delenv("BIRD_INTERACT_CACHE_TTL", raising=False)  # 5m
    monkeypatch.delenv("BIRD_INTERACT_DISABLE_PROMPT_CACHE", raising=False)
    monkeypatch.setenv("ENABLE_PROMPT_CACHING_1H", "1")  # ambient leak
    env = sdk_env.cache_control_env()
    assert env["ENABLE_PROMPT_CACHING_1H"] == ""
    assert env["FORCE_PROMPT_CACHING_5M"] == "1"


# ---------------------------------------------------------------------------
# build_hermetic_session_env merges the cache knobs (Anthropic AND registry)
# ---------------------------------------------------------------------------


def test_hermetic_env_includes_cache_knobs_for_anthropic(monkeypatch):
    monkeypatch.delenv("BIRD_INTERACT_CACHE_TTL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    env = sdk_env.build_hermetic_session_env(_ANTHROPIC, "/tmp/cfg")
    for k in _KNOBS:
        assert k in env
    assert env["FORCE_PROMPT_CACHING_5M"] == "1"


def test_hermetic_env_1h_for_doubleword(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "1h")
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-key-1")
    monkeypatch.setenv("BIRD_DOUBLEWORD_ANTHROPIC_BASE_URL", "https://api.doubleword.ai")
    env = sdk_env.build_hermetic_session_env(_DW, "/tmp/cfg")
    assert env["ENABLE_PROMPT_CACHING_1H"] == "1"
    # registry layer still present (base url + bearer)
    assert env["ANTHROPIC_BASE_URL"] == "https://api.doubleword.ai"


# ---------------------------------------------------------------------------
# _safe_cost prices cache writes by the selected TTL
# ---------------------------------------------------------------------------


def test_safe_cost_write_multiplier_5m_vs_1h(monkeypatch):
    base = dict(model=_DW, prompt_tokens=3020, completion_tokens=0,
                cache_creation_input_tokens=3001, cache_read_input_tokens=0)
    # base_input = 3020 - 3001 = 19 (DW input_tokens includes cache-creation)
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "5m")
    p5, _ = usage._safe_cost(**base)
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "1h")
    p1, _ = usage._safe_cost(**base)
    assert p5 == pytest.approx(19 * 1.40e-6 + 3001 * 1.40e-6 * 1.25)
    assert p1 == pytest.approx(19 * 1.40e-6 + 3001 * 1.40e-6 * 2.0)
    assert p1 > p5


def test_safe_cost_write_and_read_combined(monkeypatch):
    """Edge case: cache_creation AND cache_read both > 0 in one call. The
    DW-inclusive base_input subtracts ONLY cache_creation (reads are billed
    separately at read_rate), so nothing is double-counted."""
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "5m")
    # prompt_tokens (incl. creation) = 1000; 600 written, 300 read this call.
    p, _ = usage._safe_cost(
        model=_DW, prompt_tokens=1000, completion_tokens=0,
        cache_creation_input_tokens=600, cache_read_input_tokens=300,
    )
    base = 1000 - 600  # 400 genuinely-uncached
    assert p == pytest.approx(
        base * 1.40e-6 + 600 * 1.40e-6 * 1.25 + 300 * 0.14e-6
    )


def test_safe_cost_prompt_less_than_creation_clamps_to_zero(monkeypatch):
    """Defensive: if a provider ever reports prompt_tokens < cache_creation
    (shouldn't happen, but the max(...,0) guard prevents a negative base cost)."""
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "5m")
    p, _ = usage._safe_cost(
        model=_DW, prompt_tokens=100, completion_tokens=0,
        cache_creation_input_tokens=500, cache_read_input_tokens=0,
    )
    # base_input clamps to 0; only the write cost remains.
    assert p == pytest.approx(500 * 1.40e-6 * 1.25)


def test_safe_cost_zai_unchanged_by_ttl(monkeypatch):
    """Existing providers (multipliers default 1.0) are unaffected by the TTL —
    a regression guard that DEV-1639 didn't silently reprice z.ai/moonshot."""
    base = dict(model="zai/glm-5.2", prompt_tokens=1000, completion_tokens=0,
                cache_creation_input_tokens=500, cache_read_input_tokens=0)
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "5m")
    p5, _ = usage._safe_cost(**base)
    monkeypatch.setenv("BIRD_INTERACT_CACHE_TTL", "1h")
    p1, _ = usage._safe_cost(**base)
    assert p5 == p1 == pytest.approx((1000 + 500) * 1.40e-6)


# ---------------------------------------------------------------------------
# Local runner wiring: --cache-ttl / --no-prompt-cache -> env signals
# ---------------------------------------------------------------------------


def test_apply_cache_ttl_env_default(monkeypatch):
    from bird_interact_agents import run
    monkeypatch.delenv("BIRD_INTERACT_CACHE_TTL", raising=False)
    monkeypatch.setenv("BIRD_INTERACT_DISABLE_PROMPT_CACHE", "stale")
    run._apply_cache_ttl_env(cache_ttl="5m", prompt_cache=True)
    import os
    assert os.environ["BIRD_INTERACT_CACHE_TTL"] == "5m"
    # prompt_cache=True clears any ambient disable signal.
    assert "BIRD_INTERACT_DISABLE_PROMPT_CACHE" not in os.environ


def test_apply_cache_ttl_env_1h_and_disable(monkeypatch):
    from bird_interact_agents import run
    run._apply_cache_ttl_env(cache_ttl="1h", prompt_cache=False)
    import os
    assert os.environ["BIRD_INTERACT_CACHE_TTL"] == "1h"
    assert os.environ["BIRD_INTERACT_DISABLE_PROMPT_CACHE"] == "1"


# ---------------------------------------------------------------------------
# Cloud CLI + driver wiring
# ---------------------------------------------------------------------------


def _submit_ns(extra):
    from bird_interact_agents.cloud import cli
    return cli.parse_args([
        "submit", "--framework", "claude_sdk_v1", "--query-mode", "slayer",
        "--mode", "one-shot", "--agent-model", _DW,
        "--user-sim-model", "anthropic/claude-haiku-4-5-20251001",
        "--instance-ids", "alien_1", "--dataset", "livesqlbench-base-lite-sqlite",
        "--no-require-annotation", "--no-subscription-auth", *extra,
    ])


def test_cli_cache_ttl_default_5m():
    ns = _submit_ns([])
    assert ns.cache_ttl == "5m"


def test_cli_cache_ttl_1h():
    ns = _submit_ns(["--cache-ttl", "1h"])
    assert ns.cache_ttl == "1h"


def test_cli_cache_ttl_rejects_bad_value():
    with pytest.raises(SystemExit):
        _submit_ns(["--cache-ttl", "60m"])


def test_manifest_stamps_cache_ttl():
    from bird_interact_agents.cloud import driver
    ns = _submit_ns(["--cache-ttl", "1h"])
    manifest = driver.build_manifest(ns, image_uri="img:1", run_id="r1")
    assert manifest["cache_ttl"] == "1h"


def test_cache_ttl_env_vars_helper():
    from bird_interact_agents.cloud import driver
    ev = driver._cache_ttl_env_vars(cache_ttl="1h", prompt_cache=True)
    # Enabled: TTL shipped, disable signal EXPLICITLY masked ("") so a stale
    # ambient worker value cannot flip caching off (Ray applies env_vars additively).
    assert ev == {"BIRD_INTERACT_CACHE_TTL": "1h", "BIRD_INTERACT_DISABLE_PROMPT_CACHE": ""}
    ev2 = driver._cache_ttl_env_vars(cache_ttl="5m", prompt_cache=False)
    assert ev2["BIRD_INTERACT_CACHE_TTL"] == "5m"
    assert ev2["BIRD_INTERACT_DISABLE_PROMPT_CACHE"] == "1"
    # unknown value clamps to 5m (deterministic actor env)
    ev3 = driver._cache_ttl_env_vars(cache_ttl="bogus")
    assert ev3["BIRD_INTERACT_CACHE_TTL"] == "5m"
