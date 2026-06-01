"""DEV-1515: tighten the ``metadata_sufficiency.verdict`` definition.

Sharper rule: ``insufficient`` ⇒ even with maximally cooperative
user_sim disclosure of the masked sql_snippet, the answer remains
underspecified (e.g. arbitrary policy thresholds with no semantic
anchor). Cases where the masked snippet contains exact constraints
the user_sim could disclose drop to ``ambiguous`` — the failure on
the submission side moves to ``user_sim_under_disclosure``.

Applied to the 2 instances on this branch:

* ``households_12``: predicates ``total_vehicles>1`` and
  ``domestichelp='no domestic workers'`` are exact constraints in the
  masked snippet. User_sim refused to disclose. Reclassified
  ``insufficient`` → ``ambiguous``; submission's
  ``failure_classification.primary`` set to
  ``user_sim_under_disclosure``.
* ``alien_2``: gold uses population-stddev, agent used sample-stddev.
  No user_sim (one-shot). KB/sources don't pin the convention.
  Reclassified ``insufficient`` → ``ambiguous``; submission's failure
  stays ``metadata_ambiguity`` (which it already was).

``households_15`` stays ``insufficient`` (audit-status=unrecoverable;
coefficients have no semantic anchor).
"""
from __future__ import annotations

from bird_interact_agents import paths
from bird_interact_agents.eval.annotation_io import (
    read_submission_annotation,
    read_task_annotation,
    submission_annotation_path,
    task_annotation_path,
    write_submission_annotation,
    write_task_annotation,
)


_TASK_PATCHES = (
    (
        "households_12",
        "households",
        "ambiguous",
        (
            "Audit-resolvable via user_sim disclosure: the masked "
            "sql_snippet specifies exact predicates "
            "`total_vehicles>1` and `domestichelp='no domestic workers'` "
            "as part of the household-need definition. KB 42 + column "
            "meanings don't name those predicates, but a cooperative "
            "user_sim could disclose them on a well-phrased ask. "
            "Verdict is `ambiguous` (not `insufficient`) because the "
            "task IS solvable with user_sim help; in the actual run "
            "the sim refused row-count disclosure and gave a vague "
            "'use the service criteria' reply (see submission "
            "failure_classification = user_sim_under_disclosure)."
        ),
        ["households_kb.jsonl#42", "amb_user_query (masked sql_snippet)"],
    ),
    (
        "alien_2",
        "alien",
        "ambiguous",
        (
            "Audit added SQRT around variance; sample-stddev (Bessel "
            "n-1) vs population-stddev (n) is a known statistical-"
            "convention ambiguity. Neither KB 52 nor the labeled "
            "sql_snippet nor sampled values disambiguate. Verdict is "
            "`ambiguous` (not `insufficient`) because a cooperative "
            "user_sim could clarify which formula is meant. In the "
            "actual one-shot run there is no user_sim — failure stays "
            "metadata_ambiguity."
        ),
        ["alien_kb.jsonl#52", "amb_user_query (masked sql_snippet)"],
    ),
)

_SUBMISSION_FAILURE_PATCHES = (
    (
        "households_12",
        "households",
        "20260531t1008-claudes-slayer-890419",
        "user_sim_under_disclosure",
        ["metadata_ambiguity"],
    ),
)


def main() -> None:
    repo_root = paths.main_checkout_root()
    for iid, db, new_verdict, new_rationale, new_evidence in _TASK_PATCHES:
        p = task_annotation_path(
            benchmark="mini-interact", selected_database=db,
            instance_id=iid, repo_root=repo_root,
        )
        ann = read_task_annotation(p)
        old = ann.metadata_sufficiency.verdict
        ann.metadata_sufficiency.verdict = new_verdict  # type: ignore[assignment]
        ann.metadata_sufficiency.rationale = new_rationale
        ann.metadata_sufficiency.evidence_sources_consulted = list(
            new_evidence
        )
        # Reset original_gold_is_correct + drop evaluator_prompt for
        # the new `ambiguous` verdict.
        ann.original_gold_is_correct = False
        if new_verdict == "ambiguous":
            ann.evaluator_prompt = None
        write_task_annotation(ann, p)
        print(f"  task {iid}: {old} → {new_verdict}")

    for iid, db, run, new_primary, new_secondary in _SUBMISSION_FAILURE_PATCHES:
        p = submission_annotation_path(
            benchmark="mini-interact", selected_database=db,
            instance_id=iid, run_id=run, repo_root=repo_root,
        )
        ann = read_submission_annotation(p)
        old = ann.failure_classification.primary
        ann.failure_classification.primary = new_primary  # type: ignore[assignment]
        ann.failure_classification.secondary = list(new_secondary)
        if not ann.failure_classification.remediation_text:
            ann.failure_classification.remediation_text = (
                "Tighten user_sim disclosure policy on row-count + "
                "predicate-naming asks when masked sql_snippet contains "
                "exact constraints."
            )
        ann.failure_classification.remediation_target = "user_sim"  # type: ignore[assignment]
        write_submission_annotation(ann, p)
        print(f"  submission {iid}: primary {old} → {new_primary}")


if __name__ == "__main__":
    main()
