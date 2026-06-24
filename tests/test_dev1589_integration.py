"""DEV-1589 integration tests (skipped in the default non-integration run).

Two real-SDK checks:

1. **SDK usage characterization (REQUIRED, Codex r2 #2):** a warm ClaudeSDKClient
   driven over TWO `query()` cycles must report PER-QUERY `ResultMessage.usage`
   (not session-cumulative) and a PER-`query()` `max_turns` budget — the
   assumption the per-cycle fresh-tracker accounting (and the shipped DEV-1581
   `DiscoveryChannel`) rests on.

2. **households KB encode smoke:** encode 1-2 households KBs against a real
   deterministic cache → models appear in storage tagged `meta.kb_id`, their
   descriptions carry the verbatim KB block, and the KB memories carry the
   entity refs.

Both need `ANTHROPIC_API_KEY` (or `ZAI_API_KEY`) and the `claude` CLI; the
households smoke also needs the local deterministic cache. Run with:
`env -u SSH_AUTH_SOCK ... pytest -m integration tests/test_dev1589_integration.py`.
"""

from __future__ import annotations

import os
import shutil

import pytest

pytestmark = pytest.mark.integration

_HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ZAI_API_KEY"))
# The encoder spawns the bundled `claude` CLI — gate on it too, so a box with a
# key but no CLI skips cleanly instead of failing (CodeRabbit).
_HAS_CLAUDE_CLI = shutil.which("claude") is not None
_CAN_RUN = _HAS_KEY and _HAS_CLAUDE_CLI
_SKIP_REASON = "needs ANTHROPIC_API_KEY/ZAI_API_KEY and the `claude` CLI"


@pytest.mark.skipif(not _CAN_RUN, reason=_SKIP_REASON)
async def test_result_message_usage_is_per_query():
    """Drive a warm client twice; assert the 2nd ResultMessage.usage is NOT the
    cumulative of both turns (i.e. per-query). If this ever fails, the
    fresh-tracker-per-cycle accounting in setup_encoder would over-count and must
    switch to delta tracking."""
    from claude_agent_sdk import ClaudeAgentOptions

    from bird_interact_agents.agents.claude_sdk.sdk_env import (
        hermetic_claude_sdk_session,
    )
    from bird_interact_agents.model_string import native_model_id

    model = "anthropic/claude-haiku-4-5-20251001" if os.environ.get(
        "ANTHROPIC_API_KEY") else "zai/glm-5.2"

    def _opts(opt_kwargs):
        return ClaudeAgentOptions(
            **opt_kwargs, tools=[], setting_sources=[],
            model=native_model_id(model), max_turns=2,
        )

    def _result_usage(msgs):
        for m in reversed(msgs):
            if type(m).__name__ == "ResultMessage":
                return getattr(m, "usage", None)
        return None

    async with hermetic_claude_sdk_session(
        model, mcp_servers={}, build_options=_opts,
    ) as client:
        await client.query("Reply with the single word: one")
        first = [m async for m in client.receive_response()]
        await client.query("Reply with the single word: two")
        second = [m async for m in client.receive_response()]

    u1 = _result_usage(first)
    u2 = _result_usage(second)
    assert u1 is not None and u2 is not None

    def _in(u):
        return (u.get("input_tokens") if isinstance(u, dict)
                else getattr(u, "input_tokens", 0)) or 0

    # Per-query: the 2nd cycle's input tokens are bounded by a single short
    # turn, NOT the running total of both turns. (A cumulative usage would make
    # u2 input >= u1 input + the 2nd prompt; per-query keeps them independent.)
    assert _in(u2) < _in(u1) + _in(u2) + 1  # trivially true; see strict check
    # Strict: the 2nd is in the same order of magnitude as the 1st, not ~2x+.
    assert _in(u2) <= _in(u1) * 1.8, (
        f"ResultMessage.usage looks cumulative (u1={_in(u1)}, u2={_in(u2)}); "
        "the per-cycle usage accounting assumption is broken"
    )


@pytest.mark.skipif(not _CAN_RUN, reason=_SKIP_REASON)
async def test_households_kb_encode_smoke(tmp_path):
    """End-to-end encode of a couple of households KBs against a real cache."""
    from slayer.storage.yaml_storage import YAMLStorage

    from bird_interact_agents import paths
    from bird_interact_agents.slayer_otf.cache import ensure_db_cache
    from bird_interact_agents.agents.claude_sdk_otf_encode.setup_encoder import (
        make_claude_sdk_build_encoder,
    )

    db = "households"
    cache_entry = await ensure_db_cache(
        db, cache_root=paths.slayer_otf_cache_root(),
        mini_interact_root=paths.mini_interact_root(), force=False,
    )
    build_dir = tmp_path / db
    shutil.copytree(cache_entry.cache_dir, build_dir, dirs_exist_ok=True)
    storage = YAMLStorage(base_dir=str(build_dir))

    model = "anthropic/claude-opus-4-7" if os.environ.get(
        "ANTHROPIC_API_KEY") else "zai/glm-5.2"
    be = make_claude_sdk_build_encoder(
        model=model, self_model_id=model,
    )
    run_one = be(storage, build_dir, db, tmp_path / "_sessions")
    results = []
    await run_one.aopen()
    try:
        for row in cache_entry.kb_rows[:3]:
            results.append(await run_one(int(row["id"]), row, []))
    finally:
        await run_one.aclose()

    # The encoder must actually DO something — not every row may be encodable,
    # but they can't all be hard errors (Codex test #8).
    assert any(r.status != "error" for r in results)

    # For every row it claims `encoded`, the storage invariants MUST hold:
    # entity present + tagged meta.kb_id, description carries the verbatim KB
    # row, and the KB memory carries each entity ref.
    s = YAMLStorage(base_dir=str(build_dir))
    encoded = [r for r in results if r.status == "encoded"]
    for r in encoded:
        mem = await s.get_memory_row(f"{db}_kb_{r.kb_id}")
        for ent in r.entities:
            model = await s.get_model(ent.host_model or ent.name, data_source=db)
            assert model is not None
            assert ent.entity_ref in mem.entities
            # description injection (HC-desc) — at least the [kb=N] tag present
            blob = ""
            for bucket in (model.columns or []) + (model.measures or []) + \
                    (getattr(model, "aggregations", None) or []):
                if getattr(bucket, "name", None) == ent.name:
                    blob = getattr(bucket, "description", "") or ""
            if ent.kind == "model":
                blob = model.description or ""
            assert f"[kb={r.kb_id}]" in blob
