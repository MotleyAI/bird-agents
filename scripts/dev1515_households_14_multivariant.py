"""DEV-1515: rewrite households_14 as the worked example of the
multi-variant-on-source-conflict pattern.

KB 29 (Comfortable Living Household) description says
``Living Condition Score > 3 AND Bathroom Ratio > 0.5``.
The masked sql_snippet for ``"good quality of life"`` (in
``user_query_ambiguity.critical_ambiguity``) says
``lcs > 2 AND bath_ratio > 0.5``.

Two authoritative sources disagree on the same parameter — the LCS
threshold. The audit cannot pick a side without ratifying half the
sources; the right answer is to emit BOTH readings as audit rows and
let the cascade match either.

What this script does:

1. In ``audited_gold/mini_interact_audited.jsonl``:
   * Marks the existing row `variant_id="kb_definition_reading"`,
     keeps `primary=true`, threshold stays `lcs > 3` (KB 29's
     description).
   * Adds a second row `variant_id="snippet_reading"`,
     `primary=false`, threshold flipped to `lcs > 2` (matching the
     masked snippet). Everything else (data-aligned LCS formula,
     dwelling-class synonyms, infra IN-sets, bath_ratio formula,
     ORDER BY) stays identical.
   * Both rows carry an additional `changes[]` entry of
     `clause_kind="source_conflict"` pointing at the other source's
     contrary value.

2. In ``annotations/mini-interact/households/households_14.task.json``:
   * `verdict = "ambiguous"` (was `sufficient` — wrong; sources
     don't converge).
   * `internal_inconsistency` populated:
     - `sources_in_conflict = ["KB#29.description: 'LCS > 3'",
        "critical_ambiguity for 'good quality of life'.sql_snippet:
        'lcs > 2'"]`
     - `description`: 2 sentences naming the disagreement.
     - `audit_resolution = "multi_variant"`.
   * `gold_variants` carries TWO entries, one per audit row.
   * `rationale` foregrounds the conflict.

3. In
   ``annotations/mini-interact/households/households_14.submission.<run>.json``:
   * Re-grade using the now multi-variant audit; cascade picks the
     matching variant via existing N3 logic.
   * Failure_classification adjusts to whatever the agent's real
     error is (likely `agent_miss` — agent missed bath_ratio > 0.5
     entirely).
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_io import (
    read_task_annotation,
    submission_annotation_path,
    task_annotation_path,
    write_submission_annotation,
    write_task_annotation,
)
from bird_interact_agents.eval.annotation_schema import (
    AuditedGoldRef,
    GoldVariantRef,
    InternalInconsistency,
    MetadataSufficiency,
)
from bird_interact_agents.eval.annotate import _user_sim_interaction_from_trajectory
from bird_interact_agents.eval.grade_in_place import _build_submission_annotation
from bird_interact_agents.eval.tolerant_grader import default_executor, grade_submission


INSTANCE_ID = "households_14"
DB = "households"
BENCHMARK = "mini-interact"
RUN_ID = "20260531t1008-claudes-slayer-890419"
AUDIT_FILE_REL = "audited_gold/mini_interact_audited.jsonl"


# ---------------------------------------------------------------------------
# Step 1 — rewrite audited_gold/mini_interact_audited.jsonl
# ---------------------------------------------------------------------------


def _patch_audit_jsonl() -> None:
    audit_path = paths.audited_gold_root() / "mini_interact_audited.jsonl"
    rows: list[dict] = []
    found = False
    for line in audit_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["instance_id"] != INSTANCE_ID:
            rows.append(d)
            continue
        # The existing row becomes the kb_definition_reading variant
        # (already `lcs > 3`).
        found = True
        kb_row = dict(d)
        kb_row["variant_id"] = "kb_definition_reading"
        kb_row["primary"] = True
        kb_row.setdefault("changes", []).append({
            "clause_kind": "source_conflict",
            "original": "lcs > 2 (per critical_ambiguity for 'good quality of life'.sql_snippet)",
            "replacement": "lcs > 3",
            "why_unjustified": (
                "KB#29.description authoritatively states 'Living "
                "Condition Score is greater than 3 AND Bathroom Ratio "
                "is greater than 0.5'. The masked sql_snippet for "
                "'good quality of life' contradicts it with 'lcs > 2'. "
                "This variant follows the KB; the snippet_reading "
                "variant follows the snippet."
            ),
            "justified_by": ["kb:29"],
        })
        kb_row["reasoning_summary"] = (
            "[DEV-1515 source-conflict] KB#29.description: LCS > 3 AND "
            "Bath_Ratio > 0.5. The masked sql_snippet contradicts with "
            "'lcs > 2'. This is the KB-anchored reading; see the "
            "snippet_reading variant for the snippet-anchored alternate."
        )
        rows.append(kb_row)

        # Build the snippet_reading variant: identical to kb_row except
        # `lcs > 3` → `lcs > 2` in audited_sol_sql.
        snippet_row = json.loads(json.dumps(kb_row))  # deep copy
        snippet_row["variant_id"] = "snippet_reading"
        snippet_row["primary"] = False
        snippet_row["audited_sol_sql"] = [
            s.replace("lcs > 3", "lcs > 2") for s in kb_row["audited_sol_sql"]
        ]
        # Adjust the source_conflict entry to point the other way.
        snippet_row["changes"] = [
            c for c in snippet_row["changes"] if c["clause_kind"] != "source_conflict"
        ]
        snippet_row["changes"].append({
            "clause_kind": "source_conflict",
            "original": "lcs > 3 (per KB#29.description)",
            "replacement": "lcs > 2",
            "why_unjustified": (
                "The masked sql_snippet for 'good quality of life' in "
                "user_query_ambiguity.critical_ambiguity authoritatively "
                "uses 'lcs > 2'. KB#29.description contradicts with "
                "'LCS > 3'. This variant follows the snippet (what a "
                "cooperative user_sim would disclose); see the "
                "kb_definition_reading variant for the KB alternate."
            ),
            "justified_by": ["labeled_ambiguity:good quality of life"],
        })
        snippet_row["reasoning_summary"] = (
            "[DEV-1515 source-conflict] Masked sql_snippet for 'good "
            "quality of life': lcs > 2 AND bath_ratio > 0.5. KB#29 "
            "description contradicts with 'LCS > 3'. This is the "
            "snippet-anchored reading; see the kb_definition_reading "
            "variant for the KB-anchored alternate."
        )
        # Recompute audited_sample_row for the snippet_reading
        # variant against the live DB.
        db_path = paths.mini_interact_root() / DB / f"{DB}.sqlite"
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        try:
            cur = conn.cursor()
            cur.execute(snippet_row["audited_sol_sql"][-1])
            rows_out = cur.fetchall()
            if rows_out:
                snippet_row["audited_sample_row"] = list(rows_out[0])
                snippet_row["audited_sample_row_status"] = "ok"
                snippet_row.pop("audited_sample_row_error", None)
            else:
                snippet_row["audited_sample_row"] = []
                snippet_row["audited_sample_row_status"] = "empty"
                snippet_row.pop("audited_sample_row_error", None)
        except Exception as exc:
            snippet_row["audited_sample_row"] = []
            snippet_row["audited_sample_row_status"] = "error"
            snippet_row["audited_sample_row_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            conn.close()
        rows.append(snippet_row)

    if not found:
        raise SystemExit(f"{INSTANCE_ID} not found in {audit_path}")

    audit_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"Patched {audit_path}: {INSTANCE_ID} now has 2 rows "
          f"(kb_definition_reading [primary] + snippet_reading)")


# ---------------------------------------------------------------------------
# Step 2 — rewrite task annotation
# ---------------------------------------------------------------------------


def _patch_task_annotation() -> None:
    p = task_annotation_path(
        benchmark=BENCHMARK,
        selected_database=DB,
        instance_id=INSTANCE_ID,
        repo_root=paths.main_checkout_root(),
    )
    ann = read_task_annotation(p)
    ann.metadata_sufficiency = MetadataSufficiency(
        verdict="ambiguous",
        rationale=(
            "Sources contradict on the LCS threshold for 'Comfortable "
            "Living Household' (KB 29). KB#29.description authoritatively "
            "says 'LCS > 3 AND Bath_Ratio > 0.5'; the masked sql_snippet "
            "for the user-query term 'good quality of life' (in "
            "user_query_ambiguity.critical_ambiguity) authoritatively says "
            "'lcs > 2 AND bath_ratio > 0.5'. An agent cannot satisfy both "
            "thresholds at once. The audit records both readings as gold "
            "variants (kb_definition_reading primary; snippet_reading "
            "alternate); the cascade passes the agent on either match."
        ),
        evidence_sources_consulted=[
            "households_kb.jsonl#29",
            "mini_interact.jsonl: households_14.user_query_ambiguity.critical_ambiguity['good quality of life'].sql_snippet",
        ],
    )
    ann.original_gold_is_correct = False
    ann.gold_variants = [
        GoldVariantRef(
            variant_id="kb_definition_reading",
            interpretation=(
                "Comfortable = LCS > 3 AND Bath_Ratio > 0.5, per KB 29's "
                "description field. Data-aligned LCS formula: dwelling "
                "(brickwork/brick=4, apartment/apt=3, else 1) weighted "
                "0.5; infra avg over data-aligned water/road/park "
                "values weighted 0.5."
            ),
            primary=True,
            anchored_in=["households_kb.jsonl#29.description"],
            audited_gold_ref=AuditedGoldRef(
                file=AUDIT_FILE_REL,
                instance_id=INSTANCE_ID,
                variant_id="kb_definition_reading",
            ),
            notes=(
                "Contradicts the masked sql_snippet for 'good quality "
                "of life' (which says lcs > 2). See the snippet_reading "
                "variant. Both readings are recorded; the cascade matches "
                "either."
            ),
        ),
        GoldVariantRef(
            variant_id="snippet_reading",
            interpretation=(
                "Comfortable = lcs > 2 AND bath_ratio > 0.5, per the "
                "masked sql_snippet for 'good quality of life' in "
                "user_query_ambiguity.critical_ambiguity (what a "
                "cooperative user_sim would disclose). Same data-aligned "
                "LCS formula as kb_definition_reading; only the threshold "
                "differs."
            ),
            primary=False,
            anchored_in=[
                "mini_interact.jsonl: households_14.user_query_ambiguity.critical_ambiguity['good quality of life'].sql_snippet",
            ],
            audited_gold_ref=AuditedGoldRef(
                file=AUDIT_FILE_REL,
                instance_id=INSTANCE_ID,
                variant_id="snippet_reading",
            ),
            notes=(
                "Contradicts KB#29.description (which says LCS > 3). "
                "See the kb_definition_reading variant. Both readings "
                "are recorded; the cascade matches either."
            ),
        ),
    ]
    ann.evaluator_prompt = None
    ann.internal_inconsistency = InternalInconsistency(
        sources_in_conflict=[
            "households_kb.jsonl#29.description: 'Living Condition Score is greater than 3 AND Bathroom Ratio is greater than 0.5'",
            "mini_interact.jsonl: households_14.user_query_ambiguity.critical_ambiguity['good quality of life'].sql_snippet: 'lcs > 2 AND bath_ratio > 0.5'",
        ],
        description=(
            "KB 29's public description and the task's own masked "
            "sql_snippet disagree on the LCS threshold: KB says > 3, "
            "snippet says > 2. An agent cannot satisfy both at once — "
            "the threshold is a single value that must be picked."
        ),
        audit_resolution="multi_variant",
    )
    ann.annotated_by = "dev1515-households_14-multivariant"
    ann.annotated_at = (
        _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
    )
    write_task_annotation(ann, p)
    print(f"Patched {p}: verdict=ambiguous, internal_inconsistency set, "
          f"2 gold_variants")


# ---------------------------------------------------------------------------
# Step 3 — re-grade the submission against the now multi-variant audit
# ---------------------------------------------------------------------------


def _regrade_submission() -> None:
    db_path = paths.mini_interact_root() / DB / f"{DB}.sqlite"
    attempt_path = (
        paths.results_root() / "cloud" / RUN_ID / "rows"
        / INSTANCE_ID / "attempt-1.json"
    )
    attempt = json.loads(attempt_path.read_text())
    submitted_sql = attempt.get("submitted_sql", "")
    usage = attempt.get("usage", {}) or {}

    audit_path = paths.audited_gold_root() / "mini_interact_audited.jsonl"
    audit_rows: list[dict] = []
    for line in audit_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["instance_id"] == INSTANCE_ID:
            audit_rows.append(d)

    # Task row.
    mi = paths.mini_interact_root() / "mini_interact.jsonl"
    task_row = None
    for line in mi.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["instance_id"] == INSTANCE_ID:
            task_row = d
            break
    assert task_row is not None

    task_ann_path = task_annotation_path(
        benchmark=BENCHMARK, selected_database=DB,
        instance_id=INSTANCE_ID, repo_root=paths.main_checkout_root(),
    )
    task_ann = read_task_annotation(task_ann_path)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    try:
        cascade = grade_submission(
            task_annotation=task_ann,
            audited_gold_rows=audit_rows,
            original_sol_sql=list(task_row.get("sol_sql") or []),
            submitted_sql=submitted_sql,
            db_path=db_path,
            conn=conn,
            executor=default_executor,
            llm_judge=None,
        )
    finally:
        conn.close()

    traj = list(attempt.get("trajectory") or [])
    user_sim = _user_sim_interaction_from_trajectory(traj)
    ann = _build_submission_annotation(
        task_annotation=task_ann,
        cascade=cascade,
        benchmark=BENCHMARK,
        run_id=RUN_ID,
        trajectory_path=str(attempt_path),
        predicted_row_count=attempt.get("predicted_row_count"),
        duration_s=attempt.get("duration_s"),
        cost_usd_agent=usage.get("cost_usd_agent"),
        cost_usd_user_sim=usage.get("cost_usd_user_sim"),
        n_agent_turns=usage.get("n_agent_turns"),
        n_ask_user_calls=usage.get("n_ask_user_calls"),
        user_sim_interaction=user_sim,
    )
    sub_dest = submission_annotation_path(
        benchmark=BENCHMARK, selected_database=DB,
        instance_id=INSTANCE_ID, run_id=RUN_ID,
        repo_root=paths.main_checkout_root(),
    )
    write_submission_annotation(ann, sub_dest)
    print(f"Re-graded submission: cascade={cascade.model_dump()}")
    print(f"  primary={ann.failure_classification.primary}  "
          f"matched_variant_id={cascade.matched_variant_id}")


def main() -> None:
    _patch_audit_jsonl()
    _patch_task_annotation()
    _regrade_submission()


if __name__ == "__main__":
    main()
