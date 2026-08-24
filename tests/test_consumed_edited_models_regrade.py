"""DEV-1778: a regrade rebuilds the annotation but must NOT drop the consumed
provenance — it re-stamps `consumed_edited_models` from the run manifest's
per-db aggregate, exactly as it already re-stamps version/agent_model."""
from __future__ import annotations

import json
from pathlib import Path


def _write_attempt(run_dir: Path, instance_id: str, *, consumed: dict | None = None):
    d = run_dir / "rows" / instance_id
    d.mkdir(parents=True, exist_ok=True)
    row = {
        "instance_id": instance_id,
        "selected_database": "alien",
        "submitted_sql": "SELECT 1",
        "trajectory": [],
        "usage": {"cost_usd_agent": 0.0, "cost_usd_user_sim": 0.0,
                  "n_agent_turns": 0, "n_ask_user_calls": 0},
        "duration_s": 0.0,
        "predicted_row_count": 0,
        "sol_sql": ["SELECT gold"],
        "original_sol_sql": ["SELECT gold"],
    }
    if consumed is not None:
        row["consumed_edited_models"] = consumed
    (d / "attempt-1.json").write_text(json.dumps(row))


class _StubGrader:
    def __call__(self, **_kw):
        from bird_interact_agents.eval.tolerant_grader import CascadeVerdict
        return CascadeVerdict(
            n1_original_gold=True, n2_audited_primary=True,
            n3_any_audited_variant=True, n4_tie_order=True,
            n5_llm_judge=True, n6_numeric_epsilon=True,
            n7_trailing_whitespace=True, n8_column_order=True,
            n9_case_fold=True, matched_variant_id="primary",
            novel_reading_judgment=None, variant_matches=[], rowset_relations=[],
        )


def test_regrade_restamps_consumed_from_manifest(tmp_path, monkeypatch):
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.regrade import regrade_run

    run_dir = tmp_path / "results" / "cloud" / "r1"
    _write_attempt(run_dir, "alien_1")
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": "r1", "framework": "claude_sdk", "agent_model": "anthropic/x",
        "consumed_edited_models": [
            {"db": "alien", "instance_id": "alien_1", "store_fp": "fpAAA"},
            {"db": "alien", "instance_id": "alien_2", "store_fp": "fpBBB"},
        ],
    }))

    regrade_run(
        run_id="r1", benchmark="mini-interact", run_dir=run_dir,
        instance_ids=None, grader=_StubGrader(), repo_root=tmp_path,
    )

    dest = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1" / "r1.json"
    ann = json.loads(dest.read_text())
    assert ann["consumed_edited_models"] == {
        "db": "alien", "instance_id": "alien_1", "store_fp": "fpAAA",
    }


def test_regrade_consumed_none_when_manifest_lacks_it(tmp_path, monkeypatch):
    """Local runs (no manifest aggregate) regrade with consumed=None, exactly
    as version is None there — no crash, no spurious record."""
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.regrade import regrade_run

    run_dir = tmp_path / "results" / "cloud" / "r2"
    _write_attempt(run_dir, "alien_1")
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "r2", "framework": "claude_sdk"}
    ))

    regrade_run(
        run_id="r2", benchmark="mini-interact", run_dir=run_dir,
        instance_ids=None, grader=_StubGrader(), repo_root=tmp_path,
    )
    dest = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1" / "r2.json"
    assert json.loads(dest.read_text())["consumed_edited_models"] is None


def test_regrade_falls_back_to_attempt_data_when_manifest_lacks_it(tmp_path, monkeypatch):
    """A local run has no manifest aggregate, but its attempt-N.json carries the
    consumed record — regrade must preserve it rather than write None."""
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.regrade import regrade_run

    run_dir = tmp_path / "results" / "cloud" / "r3"
    _write_attempt(run_dir, "alien_1", consumed={
        "db": "alien", "instance_id": "alien_1", "store_fp": "fpLOCAL",
    })
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "r3", "framework": "claude_sdk"}  # no consumed aggregate
    ))

    regrade_run(
        run_id="r3", benchmark="mini-interact", run_dir=run_dir,
        instance_ids=None, grader=_StubGrader(), repo_root=tmp_path,
    )
    dest = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1" / "r3.json"
    assert json.loads(dest.read_text())["consumed_edited_models"] == {
        "db": "alien", "instance_id": "alien_1", "store_fp": "fpLOCAL",
    }


def test_regrade_fallback_rejects_mismatched_identity(tmp_path, monkeypatch):
    """A stale/mismatched attempt record (wrong db/instance_id) is NOT stamped
    onto this task's annotation."""
    from bird_interact_agents import paths as paths_mod
    monkeypatch.setattr(paths_mod, "main_checkout_root", lambda: tmp_path)
    from bird_interact_agents.eval.regrade import regrade_run

    run_dir = tmp_path / "results" / "cloud" / "r4"
    _write_attempt(run_dir, "alien_1", consumed={
        "db": "alien", "instance_id": "SOMEONE_ELSE", "store_fp": "fpWRONG",
    })
    (run_dir / "manifest.json").write_text(json.dumps(
        {"run_id": "r4", "framework": "claude_sdk"}
    ))

    regrade_run(
        run_id="r4", benchmark="mini-interact", run_dir=run_dir,
        instance_ids=None, grader=_StubGrader(), repo_root=tmp_path,
    )
    dest = tmp_path / "runs" / "mini-interact" / "alien" / "alien_1" / "r4.json"
    assert json.loads(dest.read_text())["consumed_edited_models"] is None
