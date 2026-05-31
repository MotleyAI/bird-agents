"""Evaluation tooling for annotated benchmark runs (DEV-1515).

Two artifact kinds:

* **Task annotation** — per-instance, run-independent. Captures the
  metadata sufficiency verdict, the set of acceptable gold variants
  (1..N, with one tagged ``primary`` for interaction-time feedback),
  and an LLM-judge ``evaluator_prompt`` for tasks the metadata can't
  uniquely pin. Path:
  ``annotations/<benchmark>/<db>/<instance_id>.task.json``.

* **Submission annotation** — per-(instance, run). Carries the cascading
  evaluation block (original gold → primary variant → any variant → tie
  tolerance → LLM judge), the failure classification, decision-point
  reference, and the user-sim interaction summary. Path:
  ``annotations/<benchmark>/<db>/<instance_id>.submission.<run_id>.json``.

The on-disk shape is governed by the Pydantic models in
``annotation_schema`` and the read/write helpers in ``annotation_io``.
"""
from bird_interact_agents.eval.annotation_schema import (
    AuditedGoldRef,
    FailureClassification,
    GoldVariantRef,
    MaskedTerm,
    MetadataSufficiency,
    SubmissionAnnotation,
    SubmissionEvaluation,
    SubmissionMetadata,
    TaskAnnotation,
    TrajectoryDecisionPoint,
    UserSimInteraction,
    VariantMatch,
)
from bird_interact_agents.eval.annotation_io import (
    read_submission_annotation,
    read_task_annotation,
    submission_annotation_path,
    task_annotation_path,
    write_submission_annotation,
    write_task_annotation,
)

__all__ = [
    "AuditedGoldRef",
    "FailureClassification",
    "GoldVariantRef",
    "MaskedTerm",
    "MetadataSufficiency",
    "SubmissionAnnotation",
    "SubmissionEvaluation",
    "SubmissionMetadata",
    "TaskAnnotation",
    "TrajectoryDecisionPoint",
    "UserSimInteraction",
    "VariantMatch",
    "read_submission_annotation",
    "read_task_annotation",
    "submission_annotation_path",
    "task_annotation_path",
    "write_submission_annotation",
    "write_task_annotation",
]
