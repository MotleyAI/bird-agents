"""Parity invariants vs. the original BIRD-Interact harness + ADK reference.

These tests guard against regressions in:
- the budget formula (a-interact reproduces the original mini_interact_agent
  with user_patience_budget=6; c-interact reproduces ADK's turn budget)
- the budget bookkeeping (force_submit fires under cost)
- the SLayer-mode tool roster (knowledge tools present for parity)
- the SLayer ingest column-meaning overlay (descriptions populated)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bird_interact_agents.config import settings
from bird_interact_agents.harness import (
    ACTION_COSTS,
    SampleStatus,
    calculate_budget,
    update_budget,
)


# ---------------------------------------------------------------------------
# Budget formula
# ---------------------------------------------------------------------------

def _task(critical: int, knowledge: int) -> dict:
    return {
        "user_query_ambiguity": {
            "critical_ambiguity": [{}] * critical,
        },
        "knowledge_ambiguity": [{}] * knowledge,
    }


@pytest.mark.parametrize(
    "critical,knowledge",
    [(0, 0), (1, 2), (3, 1)],
)
def test_a_interact_budget_matches_original(critical: int, knowledge: int):
    """a-interact: 12 + 2*amb (matches mini_interact_agent main.py with
    ENV(3)+SUBMIT(3)+2*amb+user_patience_budget(6))."""
    td = _task(critical, knowledge)
    assert calculate_budget(td, patience=3, mode="a-interact") == 12 + 2 * (
        critical + knowledge
    )


@pytest.mark.parametrize(
    "critical,knowledge",
    [(0, 0), (1, 2), (3, 1)],
)
def test_c_interact_budget_matches_adk(critical: int, knowledge: int):
    """c-interact: ask_cost*(amb + patience) + submit_cost — reproduces
    ADK cinteract.py's discrete turn budget (n_amb+patience asks + 1 submit)
    in coin terms."""
    td = _task(critical, knowledge)
    expected = (
        ACTION_COSTS["ask_user"] * (critical + knowledge + 3)
        + ACTION_COSTS["submit_sql"]
    )
    assert calculate_budget(td, patience=3, mode="c-interact") == expected


def test_a_and_c_interact_budgets_diverge_for_amb_zero():
    """Sanity: with no ambiguities, the two budgets are deliberately different."""
    td = _task(0, 0)
    a = calculate_budget(td, patience=3, mode="a-interact")
    c = calculate_budget(td, patience=3, mode="c-interact")
    assert a != c


# ---------------------------------------------------------------------------
# Budget bookkeeping
# ---------------------------------------------------------------------------

def test_update_budget_decrements_and_sets_force_submit():
    status = SampleStatus(
        idx=0, original_data={}, remaining_budget=5.0, total_budget=5.0,
    )
    # At-or-below submit cost (3) -> force_submit fires (<= boundary).
    update_budget(status, "execute_sql")  # cost 1; remaining 4.0
    assert status.remaining_budget == 4.0
    assert status.force_submit is False
    update_budget(status, "execute_sql")  # cost 1; remaining 3.0 == submit_cost
    assert status.remaining_budget == 3.0
    assert status.force_submit is True


def test_update_budget_clamps_at_zero():
    status = SampleStatus(
        idx=0, original_data={}, remaining_budget=0.5, total_budget=10.0,
    )
    update_budget(status, "execute_sql")  # cost 1, but remaining 0.5
    assert status.remaining_budget == 0.0


# ---------------------------------------------------------------------------
# Tool gating in raw mode (R2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_raw_tool_gated_when_budget_exhausted():
    """A raw-mode exploration tool refuses to execute when remaining_budget
    is below its cost — mirrors ADK before_tool_callback."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import load_db_data_if_needed

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    load_db_data_if_needed("alien", settings.db_path)
    agent_mod._ctx_var.set({
        "status": SampleStatus(
            idx=0, original_data=task_data,
            remaining_budget=0.0, total_budget=20.0,
        ),
        "data_path_base": settings.db_path,
        "slayer_storage_dir": "",
        "_slayer_client": None,
        "_slayer_storage": None,
        "result": None,
        "eval_mode": "a-interact",
        "max_asks": 3,
        "asks_used": 0,
    })
    result = await agent_mod.get_schema.handler({})
    text = result["content"][0]["text"]
    assert "Budget exhausted" in text
    assert "submit_sql" in text


@pytest.mark.asyncio
async def test_raw_tool_appends_remaining_budget_note():
    """Successful raw-mode tool calls append a remaining-budget note —
    mirrors ADK after_tool_callback."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import load_db_data_if_needed

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    load_db_data_if_needed("alien", settings.db_path)
    agent_mod._ctx_var.set({
        "status": SampleStatus(
            idx=0, original_data=task_data,
            remaining_budget=12.0, total_budget=12.0,
        ),
        "data_path_base": settings.db_path,
        "slayer_storage_dir": "",
        "_slayer_client": None,
        "_slayer_storage": None,
        "result": None,
        "eval_mode": "a-interact",
        "max_asks": 3,
        "asks_used": 0,
    })
    result = await agent_mod.get_schema.handler({})
    assert "[Remaining budget: 11.0 / 12.0]" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# SLayer ingest overlays column meanings (S1)
# ---------------------------------------------------------------------------

def test_slayer_storage_carries_column_descriptions():
    """After running ingest_slayer_models with S1 overlay, at least one
    dim/measure in cybermarket/markets.yaml has a populated `description`
    derived from cybermarket_column_meaning_base.json.

    Skips if storage hasn't been built yet (CI / fresh checkout) — there's
    no automatic way to populate it without a network-bound `slayer ingest`
    run. Once present, this test guards against regressions in the overlay.
    """
    yaml_file = Path(
        "slayer_storage/cybermarket/models/markets.yaml"
    )
    if not yaml_file.is_file():
        pytest.skip("slayer_storage not built; run scripts/ingest_slayer_models.py")

    model = yaml.safe_load(yaml_file.read_text())
    described = [
        e for e in (model.get("dimensions") or []) + (model.get("measures") or [])
        if isinstance(e.get("description"), str) and e["description"].strip()
    ]
    assert described, (
        "Expected at least one dim/measure to carry a description after "
        "the ingest overlay (S1). Re-run scripts/ingest_slayer_models.py."
    )


def _load_compare_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compare_results",
        Path(__file__).parent.parent / "scripts" / "compare_results.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_results_normalises_original_jsonl_shape():
    """Upstream main.py writes the row as
    {original_data: {instance_id: ...}, phase1_completed: bool,
    task_finished: bool, last_reward: float}. compare_results must read
    those, not the bird-interact-agents-shaped {instance_id, phase1_passed,
    phase2_passed, total_reward} fields.
    """
    mod = _load_compare_module()
    upstream = {
        "original_data": {"instance_id": "alien_1"},
        "phase1_completed": True,
        "task_finished": False,
        "last_reward": 0.5,
    }
    out = mod._norm_orig(upstream)
    assert out["instance_id"] == "alien_1"
    assert out["phase1_passed"] is True
    assert out["phase2_passed"] is False
    assert out["total_reward"] == 0.5


def test_compare_results_handles_ours_shape_unchanged():
    """Smoke: our own eval.json shape still normalises through _norm_ours."""
    mod = _load_compare_module()
    ours = {
        "instance_id": "alien_2",
        "phase1_passed": False,
        "phase2_passed": False,
        "total_reward": 0.0,
        "error": None,
    }
    out = mod._norm_ours(ours)
    assert out["instance_id"] == "alien_2"
    assert out["phase1_passed"] is False


@pytest.mark.asyncio
async def test_claude_sdk_skips_non_anthropic_model_with_clear_error():
    """claude_sdk is locked to Anthropic by SDK design. When --agent-model
    is non-Anthropic, run_task must return a skip-shaped row (not crash) so
    the 3-way comparison still renders cleanly."""
    from bird_interact_agents.agents.claude_sdk import agent as agent_mod
    from bird_interact_agents.harness import load_db_data_if_needed

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
        "amb_user_query": "any",
    }
    load_db_data_if_needed("alien", settings.db_path)

    agent = agent_mod.ClaudeSDKAgent(
        slayer_storage_root=None, model="cerebras/zai-glm-4.7"
    )
    result = await agent.run_task(
        task_data, settings.db_path, budget=12.0, query_mode="raw",
        eval_mode="a-interact",
    )
    assert result["phase1_passed"] is False
    assert result["phase2_passed"] is False
    assert result["error"] is not None
    err = result["error"].lower()
    assert "claude_sdk" in err and "anthropic" in err


def test_overlay_helper_writes_descriptions(tmp_path: Path):
    """Unit-test the overlay function directly with a synthetic store +
    meanings file — guards the YAML round-trip independent of slayer."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ingest_module",
        Path(__file__).parent.parent / "scripts" / "ingest_slayer_models.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    db_dir = tmp_path / "fakedb"
    db_dir.mkdir()
    storage_dir = tmp_path / "store" / "fakedb"
    models_dir = storage_dir / "models"
    models_dir.mkdir(parents=True)

    (db_dir / "fakedb_column_meaning_base.json").write_text(
        json.dumps({
            "fakedb|orders|status": "Order lifecycle state (pending/paid/...).",
            "fakedb|orders|amount": "Total order value in USD.",
        })
    )
    (models_dir / "orders.yaml").write_text(yaml.safe_dump({
        "name": "orders",
        "sql_table": "orders",
        "data_source": "fakedb",
        "dimensions": [{"name": "status", "sql": "status", "type": "string"}],
        "measures": [{"name": "amount", "sql": "amount", "type": "number"}],
    }, sort_keys=False))

    updated = mod.overlay_column_meanings("fakedb", db_dir, storage_dir)
    assert updated == 2
    model = yaml.safe_load((models_dir / "orders.yaml").read_text())
    assert model["dimensions"][0]["description"].startswith("Order lifecycle")
    assert model["measures"][0]["description"].startswith("Total order value")
