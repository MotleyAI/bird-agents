"""DEV-1606 Defect 1 — in-task submit must accept best-of audited variants.

Under ``--use-audited-gold-sql`` the in-task grader compared the agent
ONLY against the audited PRIMARY variant (``sol_sql``). On an ambiguous
task the agent that commits to a valid NON-primary reading got
false-negative feedback and thrashed (reverse_logistics_17: 19 resubmits
against the primary while its result exactly matched a non-primary
variant the FINAL grader accepts).

Contract under test:

* ``apply_audited_gold_overlay`` attaches ``task["audited_variants"]`` (the
  WHOLE variant set) for any matching grouped row that passes the
  DB/benchmark guards — INDEPENDENT of whether the primary's status
  triggers a ``sol_sql`` swap (the defect's row has primary status
  ``original``).
* ``evaluate_best_of_audited_variants`` accepts the FIRST variant whose
  result matches; the agent-visible observation NEVER names the matched
  variant.
* ``_dispatch_eval`` falls back to best-of when the audited-primary eval
  fails, overrides the audited side to a pass, and surfaces the matched
  variant id via a NON-agent-visible channel.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# apply_audited_gold_overlay attaches the full variant set (single_file)
# ---------------------------------------------------------------------------


def _write_grouped(audited_root, benchmark_name, row: dict) -> None:
    d = audited_root / benchmark_name
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{benchmark_name}_audited.jsonl").open("w") as f:
        f.write(json.dumps(row) + "\n")


def test_overlay_attaches_all_variants_even_when_primary_is_original(tmp_path):
    """The defect's exact shape: primary audit_status='original' (so
    ``sol_sql`` is NOT swapped) but the row still carries edited
    non-primary variants. All three must reach the task dict."""
    from bird_interact_agents.harness import apply_audited_gold_overlay
    from bird_interact_agents.benchmark import get_benchmark

    b = get_benchmark("mini-interact")  # single_file layout
    _write_grouped(tmp_path, b.name, {
        "instance_id": "reverse_logistics_17",
        "selected_database": "reverse_logistics",
        "benchmark": b.name,
        "variants": [
            {"variant_id": "v1_gold_broad_costs", "primary": True,
             "audit_status": "original",
             "audited_sol_sql": ["SELECT broad FROM t"]},
            {"variant_id": "v2_kb_strict_trc", "primary": False,
             "audit_status": "edited",
             "audited_sol_sql": ["SELECT strict FROM t"]},
            {"variant_id": "v3_critical_ambiguity_snippet", "primary": False,
             "audit_status": "edited",
             "audited_sol_sql": ["SELECT snippet FROM t"]},
        ],
    })
    task = {
        "instance_id": "reverse_logistics_17",
        "selected_database": "reverse_logistics",
        "sol_sql": ["SELECT broad FROM t"],
    }
    apply_audited_gold_overlay([task], tmp_path, benchmark=b)

    # primary status 'original' → sol_sql unchanged, original_sol_sql set.
    assert task["sol_sql"] == ["SELECT broad FROM t"]
    assert task["original_sol_sql"] == ["SELECT broad FROM t"]
    # The WHOLE variant set is attached.
    variants = task["audited_variants"]
    ids = {v["variant_id"] for v in variants}
    assert ids == {
        "v1_gold_broad_costs", "v2_kb_strict_trc",
        "v3_critical_ambiguity_snippet",
    }
    v2 = next(v for v in variants if v["variant_id"] == "v2_kb_strict_trc")
    assert v2["audited_sol_sql"] == ["SELECT strict FROM t"]


# ---------------------------------------------------------------------------
# evaluate_best_of_audited_variants — first match wins, no id leak
# ---------------------------------------------------------------------------


def _status(sol_sql):
    return SimpleNamespace(original_data={
        "selected_database": "reverse_logistics",
        "instance_id": "reverse_logistics_17",
        "sol_sql": list(sol_sql),
    })


def _make_dispatcher(verdicts: dict[str, bool]):
    calls: list[str] = []

    def fake_eval(sol_sql, status, dpb):
        gold = status.original_data["sol_sql"]
        key = gold[0] if isinstance(gold, list) and gold else str(gold)
        calls.append(key)
        passed = verdicts.get(key, False)
        return (f"Submitted. Result match: {passed}",
                1.0 if passed else 0.0, passed, False, True)

    return fake_eval, calls


_VARIANTS = [
    {"variant_id": "v1_gold_broad_costs", "primary": True,
     "audit_status": "original", "audited_sol_sql": ["SELECT broad"]},
    {"variant_id": "v2_kb_strict_trc", "primary": False,
     "audit_status": "edited", "audited_sol_sql": ["SELECT strict"]},
    {"variant_id": "v3_critical_ambiguity_snippet", "primary": False,
     "audit_status": "edited", "audited_sol_sql": ["SELECT snippet"]},
]


def test_best_of_matches_non_primary_variant_without_id_leak(monkeypatch):
    from bird_interact_agents import harness

    fake_eval, _ = _make_dispatcher({"SELECT strict": True})
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    status = _status(["SELECT broad"])
    passed, matched_id, observation = harness.evaluate_best_of_audited_variants(
        pred_sql="SELECT 1",
        variants=_VARIANTS,
        status=status,
        data_path_base="/tmp/ignored",
    )
    assert passed is True
    assert matched_id == "v2_kb_strict_trc"
    # The matched variant id MUST NOT appear in the agent-visible string.
    assert "v2_kb_strict_trc" not in observation
    # status restored.
    assert status.original_data["sol_sql"] == ["SELECT broad"]


def test_best_of_returns_first_passing_among_multiple(monkeypatch):
    """When more than one variant passes, the FIRST in list order wins
    (deterministic short-circuit)."""
    from bird_interact_agents import harness

    fake_eval, _ = _make_dispatcher({"SELECT strict": True, "SELECT snippet": True})
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    passed, matched_id, _obs = harness.evaluate_best_of_audited_variants(
        pred_sql="SELECT 1",
        variants=_VARIANTS,
        status=_status(["SELECT broad"]),
        data_path_base="/tmp/ignored",
    )
    assert passed is True
    assert matched_id == "v2_kb_strict_trc"  # earlier than v3 in the list


def test_best_of_no_match_returns_false(monkeypatch):
    from bird_interact_agents import harness

    fake_eval, _ = _make_dispatcher({})  # nothing passes
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    passed, matched_id, _obs = harness.evaluate_best_of_audited_variants(
        pred_sql="SELECT 1",
        variants=_VARIANTS,
        status=_status(["SELECT broad"]),
        data_path_base="/tmp/ignored",
    )
    assert passed is False
    assert matched_id is None


def test_best_of_can_skip_already_tried_primary(monkeypatch):
    """When the caller already evaluated the primary, it can be skipped to
    avoid a redundant re-eval."""
    from bird_interact_agents import harness

    fake_eval, calls = _make_dispatcher({"SELECT strict": True})
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    harness.evaluate_best_of_audited_variants(
        pred_sql="SELECT 1",
        variants=_VARIANTS,
        status=_status(["SELECT broad"]),
        data_path_base="/tmp/ignored",
        skip_variant_sqls={("SELECT broad",)},
    )
    assert "SELECT broad" not in calls  # primary skipped


# ---------------------------------------------------------------------------
# _dispatch_eval override — best-of drives a pass, id stays hidden
# ---------------------------------------------------------------------------


def _state(audited_variants):
    return SimpleNamespace(
        status=SimpleNamespace(
            original_data={
                "selected_database": "reverse_logistics",
                "instance_id": "reverse_logistics_17",
                "sol_sql": ["SELECT broad"],          # audited primary
                "original_sol_sql": ["SELECT orig"],  # original gold
                "audited_variants": audited_variants,
            },
            remaining_budget=100.0,
            total_budget=100.0,
            force_submit=False,
            current_phase=1,
        ),
        data_path_base="/tmp/ignored",
        user_sim_model="x",
        user_sim_prompt_version="v2",
        slayer_storage_dir="",
        result=None,
    )


def test_dispatch_eval_best_of_overrides_primary_miss(monkeypatch):
    from bird_interact_agents.agents import _submit
    from bird_interact_agents import harness

    # primary + original both miss; non-primary v2 passes.
    fake_eval, _ = _make_dispatcher({
        "SELECT broad": False, "SELECT orig": False,
        "SELECT strict": True, "SELECT snippet": False,
    })
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    state = _state(_VARIANTS)
    result = _submit._dispatch_eval(state, "SELECT agent")
    observation, reward, p1, p2, finished = result[:5]
    matched_id = result[9]

    assert p1 is True
    assert reward == 1.0
    assert finished is True
    assert matched_id == "v2_kb_strict_trc"
    assert "v2_kb_strict_trc" not in observation


def test_dispatch_eval_primary_pass_skips_best_of(monkeypatch):
    """Regression: when the audited primary already passes, no best-of and
    no matched-variant id."""
    from bird_interact_agents.agents import _submit
    from bird_interact_agents import harness

    fake_eval, _ = _make_dispatcher({
        "SELECT broad": True, "SELECT orig": True,
    })
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    state = _state(_VARIANTS)
    result = _submit._dispatch_eval(state, "SELECT agent")
    p1 = result[2]
    matched_id = result[9]
    assert p1 is True
    assert matched_id is None


def test_dispatch_eval_total_miss_stays_failed(monkeypatch):
    """Regression: primary, original AND every variant miss → p1 False."""
    from bird_interact_agents.agents import _submit
    from bird_interact_agents import harness

    fake_eval, _ = _make_dispatcher({})  # nothing passes
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    state = _state(_VARIANTS)
    result = _submit._dispatch_eval(state, "SELECT agent")
    p1 = result[2]
    matched_id = result[9]
    assert p1 is False
    assert matched_id is None


# ---------------------------------------------------------------------------
# submit_raw_sql records the matched-variant id on state.result, NEVER in the
# agent-facing return string (Codex review of tests — handler persistence).
# ---------------------------------------------------------------------------


def test_submit_raw_sql_records_matched_variant_id_hidden_from_agent(monkeypatch):
    from bird_interact_agents.agents import _submit
    from bird_interact_agents import harness

    monkeypatch.setattr(_submit, "_dry_run_sql", lambda *a, **kw: None)
    # Keep the diagnostic snapshot path from touching a real DB.
    monkeypatch.setattr(_submit, "_diagnostic_payload", lambda **kw: {})
    fake_eval, _ = _make_dispatcher({
        "SELECT broad": False, "SELECT orig": False,
        "SELECT strict": True, "SELECT snippet": False,
    })
    monkeypatch.setattr(harness, "execute_submit_action", fake_eval)

    state = _state(_VARIANTS)
    agent_reply = _submit.submit_raw_sql(state, "SELECT agent")

    assert state.result["phase1_passed"] is True
    assert state.result["phase1_matched_audited_variant_id"] == "v2_kb_strict_trc"
    # The agent-facing reply must NOT reveal which variant matched.
    assert "v2_kb_strict_trc" not in agent_reply
