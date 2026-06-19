"""Tests for ``scripts/analyze_query_syntax_rejections.py``.

Pins the contract that matters for the analysis:

* the rejection taxonomy classifies each representative SLayer error string
  into the right bucket (``dsl`` headline vs the excluded ``semantic`` /
  ``gate`` / ``infra`` buckets),
* retry-to-recovery chains are counted as maximal consecutive ``dsl`` runs,
* the major-contributor-to-failure flags (``direct`` / ``never_clean`` /
  ``budget_burn``) fire only under their intended conditions,
* ``latest_trajectory_per_iid`` selects the chronologically-latest slayer
  trajectory per ``(db, iid)``,
* the budget-cost literals stay pinned to ``harness.ACTION_COSTS`` (drift guard).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_query_syntax_rejections.py"
)
_spec = importlib.util.spec_from_file_location("aqsr", SCRIPT)
aqsr = importlib.util.module_from_spec(_spec)
sys.modules["aqsr"] = aqsr
_spec.loader.exec_module(aqsr)


# ---------------------------------------------------------------------------
# Synthetic-trajectory builders
# ---------------------------------------------------------------------------
def _assistant(tool_use_id: str, name: str = "mcp__slayer__query") -> dict:
    return {
        "type": "AssistantMessage",
        "data": {"content": [{"id": tool_use_id, "name": name, "input": {}}]},
    }


def _result(tool_use_id: str, text: str, *, is_error: bool = False) -> dict:
    return {
        "type": "UserMessage",
        "data": {
            "content": [
                {
                    "tool_use_id": tool_use_id,
                    "is_error": is_error,
                    "content": text,
                }
            ]
        },
    }


def _trajectory(*pairs: tuple[str, str, bool]) -> list[dict]:
    """Each pair is ``(tool_name, result_text, is_error)``; emit an
    AssistantMessage + paired UserMessage per call, with unique ids."""
    items: list[dict] = []
    for i, (tool, text, is_err) in enumerate(pairs):
        tid = f"toolu_{i}"
        items.append(_assistant(tid, tool))
        items.append(_result(tid, text, is_error=is_err))
    return items


# representative real error strings, one per sub_tag
_OK = '{"result":"rows..."}'
_ORDER = "1 validation error for SlayerQuery\norder.0.column\n  Field required"
_AGG = "Error executing tool query: Aggregation 'countd' not allowed for primary-key column 'x'"
_FILTER = "Error executing tool query: Filter references unknown name 'fc' on model 'm'"
_SEMANTIC = '{"result":"Database error: no such column: per_dwelling.x"}'
_GATE = (
    "Your last tool call was `mcp__slayer__edit_model`, not `query`. Before "
    "submitting you MUST run the exact query JSON through `query`."
)
_PERM = "Claude requested permissions to use mcp__slayer__query, but you haven't granted it."


# ---------------------------------------------------------------------------
# classify_result — taxonomy buckets
# ---------------------------------------------------------------------------
def test_classify_dsl_subtags():
    assert aqsr.classify_result(_ORDER, is_error=False) == ("dsl", "order_shape")
    assert aqsr.classify_result(_AGG, is_error=True)[0] == "dsl"
    assert aqsr.classify_result(_AGG, is_error=True)[1] == "aggregation_rule"
    assert aqsr.classify_result(_FILTER, is_error=True) == (
        "dsl",
        "filter_construction",
    )
    assert aqsr.classify_result(
        "Error executing tool query: Unknown transform function 'ROUND'.",
        is_error=True,
    ) == ("dsl", "unknown_transform")
    assert aqsr.classify_result(
        "Bare measure name 'app_count' is not valid. Use colon syntax.",
        is_error=True,
    ) == ("dsl", "bare_measure")
    assert aqsr.classify_result(
        "Model 'agg' not found", is_error=True
    ) == ("dsl", "model_not_found")


def test_classify_excluded_buckets():
    # DB-execution failure → semantic, not DSL
    assert aqsr.classify_result(_SEMANTIC, is_error=False) == (
        "semantic",
        "no_such_column",
    )
    # submit-gate → gate
    assert aqsr.classify_result(_GATE, is_error=False)[0] == "gate"
    # permission-mode denial → infra (harness noise, not the agent's syntax)
    assert aqsr.classify_result(_PERM, is_error=True) == (
        "infra",
        "permission_denied",
    )
    # clean result → ok
    assert aqsr.classify_result(_OK, is_error=False) == ("ok", None)
    # is_error with no known pattern → other
    assert aqsr.classify_result("kaboom", is_error=True) == ("other", None)


def test_formula_syntax_wins_over_generated_sql_syntax():
    # "Invalid formula syntax" contains "syntax" but must classify as the
    # DSL formula error, not the semantic generated-SQL "syntax error".
    bucket, tag = aqsr.classify_result(
        "Error executing tool query: Invalid formula syntax: 'CAST(x)'",
        is_error=True,
    )
    assert (bucket, tag) == ("dsl", "formula_syntax")


# ---------------------------------------------------------------------------
# retry-to-recovery chains
# ---------------------------------------------------------------------------
def test_recovered_chain_length_one():
    seq = aqsr.query_call_sequence(
        _trajectory(
            ("mcp__slayer__query", _ORDER, False),
            ("mcp__slayer__query", _OK, False),
        )
    )
    chains = aqsr._dsl_chains(seq)
    assert len(chains) == 1
    assert chains[0]["length"] == 1
    assert chains[0]["recovered"] is True


def test_multi_retry_chain():
    seq = aqsr.query_call_sequence(
        _trajectory(
            ("mcp__slayer__query", _ORDER, False),
            ("mcp__slayer__query", _ORDER, False),
            ("mcp__slayer__query", _AGG, True),
            ("mcp__slayer__query", _OK, False),
        )
    )
    chains = aqsr._dsl_chains(seq)
    assert len(chains) == 1
    assert chains[0]["length"] == 3
    assert chains[0]["recovered"] is True
    assert chains[0]["sub_tags"] == ["order_shape", "order_shape", "aggregation_rule"]


def test_unrecovered_chain_runs_to_end():
    seq = aqsr.query_call_sequence(
        _trajectory(
            ("mcp__slayer__query", _OK, False),
            ("mcp__slayer__query", _ORDER, False),
            ("mcp__slayer__query", _ORDER, False),
        )
    )
    chains = aqsr._dsl_chains(seq)
    assert len(chains) == 1
    assert chains[0]["length"] == 2
    assert chains[0]["recovered"] is False


def test_semantic_and_gate_excluded_from_dsl_count():
    data = {
        "phase1_passed": True,
        "submission_status": "passed_phase1",
        "trajectory": _trajectory(
            ("mcp__slayer__query", _SEMANTIC, False),
            ("mcp__bird-interact-tools__submit_query", _GATE, False),
            ("mcp__slayer__query", _PERM, True),
            ("mcp__slayer__query", _OK, False),
        ),
    }
    rec = aqsr.analyze_trajectory(data, db="d", iid="d_1")
    assert rec["n_dsl"] == 0
    assert rec["n_semantic"] == 1
    assert rec["n_gate"] == 1
    assert rec["n_infra"] == 1
    assert rec["n_ok"] == 1


# ---------------------------------------------------------------------------
# major-contributor flags
# ---------------------------------------------------------------------------
def test_never_clean_flag():
    # failed task, every query call DSL-rejected, never an ok → never_clean
    data = {
        "phase1_passed": False,
        "submission_status": "wrong_result",
        "trajectory": _trajectory(
            ("mcp__slayer__query", _ORDER, False),
            ("mcp__slayer__query", _ORDER, False),
        ),
    }
    rec = aqsr.analyze_trajectory(data, db="d", iid="d_1")
    assert rec["flags"]["never_clean"] is True
    assert rec["flags"]["major_contributor"] is True


def test_direct_flag_on_translation_error():
    data = {
        "phase1_passed": False,
        "submission_status": "translation_error",
        "trajectory": _trajectory(("mcp__slayer__query", _OK, False)),
    }
    rec = aqsr.analyze_trajectory(data, db="d", iid="d_1")
    assert rec["flags"]["direct"] is True
    assert rec["flags"]["major_contributor"] is True


def test_budget_burn_flag():
    # tiny budget, exhausted, and DSL retries ate >=25% of it
    note = "\n\n[Remaining budget: 2.0 / 12.0]"
    data = {
        "phase1_passed": False,
        "submission_status": "wrong_result",
        "trajectory": _trajectory(
            ("mcp__slayer__query", _ORDER + note, False),
            ("mcp__slayer__query", _ORDER + note, False),
            ("mcp__slayer__query", _ORDER + note, False),
            ("mcp__slayer__query", _ORDER + note, False),
            ("mcp__slayer__query", _OK + note, False),
        ),
    }
    rec = aqsr.analyze_trajectory(data, db="d", iid="d_1")
    assert rec["budget"]["final_remaining"] == 2.0
    assert rec["budget"]["total"] == 12.0
    assert rec["est_dsl_coins"] == 4.0  # 4 DSL calls * 1 coin
    assert rec["flags"]["budget_burn"] is True


def test_no_flags_when_passed_with_high_budget():
    # passed task with DSL friction but huge budget → not a contributor
    note = "\n\n[Remaining budget: 968.0 / 1010.0]"
    data = {
        "phase1_passed": True,
        "submission_status": "passed_phase1",
        "trajectory": _trajectory(
            ("mcp__slayer__query", _ORDER + note, False),
            ("mcp__slayer__query", _OK + note, False),
        ),
    }
    rec = aqsr.analyze_trajectory(data, db="d", iid="d_1")
    assert rec["flags"]["major_contributor"] is False


# ---------------------------------------------------------------------------
# latest-trajectory selection
# ---------------------------------------------------------------------------
def _write_traj(runs_root: Path, db: str, iid: str, run_id: str, data: dict) -> Path:
    dest = runs_root / "mini-interact" / db / iid / f"{run_id}.trajectory.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data))
    return dest


def test_latest_trajectory_per_iid_picks_newest_slayer(tmp_path):
    runs_root = tmp_path / "runs"
    base = (runs_root / "mini-interact")  # noqa: F841 (clarity)
    _write_traj(
        runs_root, "alien", "alien_1",
        "20260101t1000-claudes-slayer-aaa", {"phase1_passed": True},
    )
    newest = _write_traj(
        runs_root, "alien", "alien_1",
        "20260605t1304-claudes-slayer-bbb", {"phase1_passed": False},
    )
    # a raw-mode run must be ignored under mode=slayer
    _write_traj(
        runs_root, "alien", "alien_1",
        "20260606t1304-claudesdk-raw-ccc", {"phase1_passed": True},
    )
    chosen = aqsr.latest_trajectory_per_iid(
        benchmark="mini-interact", mode="slayer",
        runs_root=runs_root / "mini-interact",
    )
    assert chosen == {("alien", "alien_1"): newest}


def test_build_payload_end_to_end(tmp_path):
    runs_root = tmp_path / "runs"
    _write_traj(
        runs_root, "alien", "alien_1", "20260605t1304-claudes-slayer-aaa",
        {
            "phase1_passed": True,
            "submission_status": "passed_phase1",
            "trajectory": _trajectory(
                ("mcp__slayer__query", _ORDER, False),
                ("mcp__slayer__query", _OK, False),
            ),
        },
    )
    payload = aqsr.build_payload(
        benchmark="mini-interact", runs_root=runs_root / "mini-interact"
    )
    agg = payload["aggregate"]
    assert agg["n_tasks"] == 1
    assert agg["bucket_totals"]["dsl"] == 1
    assert agg["dsl_subtag_counts"]["order_shape"] == 1
    assert agg["retry_histogram"] == {"1": 1}


# ---------------------------------------------------------------------------
# per-sub_tag examples + description coverage
# ---------------------------------------------------------------------------
def test_dsl_examples_captured_per_subtag():
    data = {
        "phase1_passed": True,
        "submission_status": "passed_phase1",
        "trajectory": _trajectory(
            ("mcp__slayer__query", _ORDER, False),
            ("mcp__slayer__query", _AGG, True),
            ("mcp__slayer__query", _OK, False),
        ),
    }
    rec = aqsr.analyze_trajectory(data, db="d", iid="d_1")
    assert set(rec["dsl_examples"]) == {"order_shape", "aggregation_rule"}
    assert rec["dsl_examples"]["order_shape"]["iid"] == "d_1"
    assert "order.0.column" in rec["dsl_examples"]["order_shape"]["text"]


def test_aggregate_dedups_and_caps_examples():
    # 5 tasks all hit order_shape with the SAME text → dedup to 1 example;
    # the cap is _MAX_EXAMPLES_PER_SUBTAG regardless.
    per_task = [
        aqsr.analyze_trajectory(
            {
                "phase1_passed": True,
                "trajectory": _trajectory(("mcp__slayer__query", _ORDER, False)),
            },
            db="d",
            iid=f"d_{i}",
        )
        for i in range(5)
    ]
    agg = aqsr.aggregate(per_task)
    assert len(agg["dsl_examples"]["order_shape"]) == 1  # identical text deduped
    assert len(agg["dsl_examples"]["order_shape"]) <= aqsr._MAX_EXAMPLES_PER_SUBTAG


def test_every_dsl_subtag_has_a_description():
    # Every DSL sub_tag the taxonomy can emit must be documented, so the
    # notebook's per-mode breakdown never shows a blank explanation.
    for tag in aqsr.DSL_SUBTAGS:
        assert tag in aqsr.SUBTAG_DESCRIPTIONS, tag
        assert aqsr.SUBTAG_DESCRIPTIONS[tag].strip()


# ---------------------------------------------------------------------------
# drift guard: the budget literals must match the harness source of truth
# ---------------------------------------------------------------------------
def test_budget_cost_literals_match_harness():
    from bird_interact_agents.harness import ACTION_COSTS

    assert aqsr._QUERY_COST == ACTION_COSTS["query"]
    assert aqsr._SUBMIT_COST == ACTION_COSTS["submit_query"]
