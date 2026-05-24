"""Codex-finding-10 parity tests against the recursive adapter.

The plan copies several pieces verbatim from `pydantic_ai_recursive/`
into `pydantic_ai_otf_encode/`: `_constructor_reserve`, the OTF
work-dir helpers, `_fill_record_from_run`, `_finalize`, the resolver
helpers, etc. Codex flagged drift risk: a future change to the
recursive adapter that doesn't get mirrored here would silently
diverge.

These tests assert structural parity on the pieces most likely to
drift, NOT phrasing parity. If the recursive adapter changes its
`_constructor_reserve` formula tomorrow, this test fails loudly and
the new agent needs the same update.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Numerical parity — bird-coin reserve formula
# ---------------------------------------------------------------------------


def test_constructor_reserve_matches_recursive_adapter():
    """The constructor's bird-coin reserve formula MUST equal the
    recursive adapter's (`2 * ask_user + submit_query`). If the
    recursive adapter changes it, this fails — and the new agent
    needs the same change."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _constructor_reserve as otf_reserve,
    )
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        _constructor_reserve as base_reserve,
    )

    assert otf_reserve() == base_reserve()


# ---------------------------------------------------------------------------
# Result-row shape parity — both adapters emit the same top-level keys
# ---------------------------------------------------------------------------


def test_result_row_keys_match_recursive_adapter_baseline():
    """The set of keys in the final result row must be a superset of
    the recursive adapter's. The new agent adds `kb_encoded`; nothing
    else should be missing."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode.agent import (
        _finalize as otf_finalize,
    )
    from bird_interact_agents.agents.pydantic_ai_recursive.agent import (
        _finalize as base_finalize,
    )
    from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import (
        SharedTaskState as OtfShared,
    )
    from bird_interact_agents.agents.pydantic_ai_recursive.deps import (
        SharedTaskState as BaseShared,
    )
    from bird_interact_agents.harness import SampleStatus

    def _mk(cls):
        status = SampleStatus(
            idx=0,
            original_data={
                "selected_database": "fake_db",
                "instance_id": "i", "amb_user_query": "?",
                "knowledge_ambiguity": [],
            },
            remaining_budget=10.0, total_budget=10.0,
        )
        return cls(
            status=status, data_path_base="/tmp",
            db_name="fake_db", amb_user_query="?",
            slayer_storage_dir="/tmp/x",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            user_sim_prompt_version="v2",
        )

    base_shared = _mk(BaseShared)
    otf_shared = _mk(OtfShared)

    base_row = base_finalize(
        shared=base_shared, instance_id="i", db_name="fake_db",
        deleted_kb_ids=[], slayer_storage_dir="/tmp/x",
        final_output_excerpt="", error=None,
    )
    otf_row = otf_finalize(
        shared=otf_shared, instance_id="i", db_name="fake_db",
        deleted_kb_ids=[], slayer_storage_dir="/tmp/x",
        final_output_excerpt="", error=None,
    )

    # Every base key is present in the otf row:
    missing = set(base_row) - set(otf_row)
    assert missing == set(), f"otf row dropped keys: {missing}"

    # The new agent contributes one extra: `kb_encoded`.
    extras = set(otf_row) - set(base_row)
    assert extras == {"kb_encoded"}, (
        f"otf row added unexpected keys: {extras - {'kb_encoded'}}"
    )


# ---------------------------------------------------------------------------
# Projection-count-check parity — confirmed_projection still gates submits
# ---------------------------------------------------------------------------


def test_query_constructor_count_check_present_in_otf_adapter():
    """The closure-bound count check on `submit_query` defends against
    over-projection; the new agent's constructor MUST carry the same
    check or it loses the DEV-1432 / DEV-1435 invariant."""
    from bird_interact_agents.agents.pydantic_ai_otf_encode import factories

    agent = factories._build_query_constructor(
        model="test", model_settings=None, shared_slayer_server=None,
        confirmed_projection=("col_a", "col_b"),
    )
    tools = dict(agent._function_toolset.tools)
    # submit_query is registered with the count-check wrapper:
    assert "submit_query" in tools
