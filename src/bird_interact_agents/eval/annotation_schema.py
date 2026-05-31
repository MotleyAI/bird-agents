"""Pydantic schemas for task and submission annotations (DEV-1515).

Two top-level models — ``TaskAnnotation`` (per-instance, run-independent)
and ``SubmissionAnnotation`` (per-(instance, run)) — plus the leaf
helper models the agreed JSON shape uses.

The schemas point at on-disk SQL via ``GoldVariantRef.audited_gold_ref``
(``(file, key)`` into the consolidated ``<benchmark>_audited.jsonl``);
SQL is never duplicated inside annotation JSON.

Per the project Python convention, every container field uses a typed
``BaseModel`` rather than ``Dict[...]``.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Top-level enums kept as Literal strings so the JSON encoding is stable
# and human-greppable.
MetadataSufficiencyVerdict = Literal["sufficient", "ambiguous", "insufficient"]
AuditStatus = Literal["clean", "edited", "unrecoverable", "original_correct"]
RowsetRelation = Literal[
    "equal_rowset",
    "strict_subset_of",
    "strict_superset_of",
    "overlapping",
    "disjoint",
    "unevaluated",
]
PhaseVerdict = Literal["pass", "fail", "skip"]
FailurePrimary = Literal[
    "agent_miss",
    "metadata_ambiguity",
    "gold_audit_quality",
    "user_sim_under_disclosure",
    "grader_stability",
    "other",
]
RemediationTarget = Literal["agent", "prompt", "kb", "audit", "grader", "user_sim", "gold_sidecar", "other"]
SubmissionVerdict = Literal[
    "correct",
    "valid_interpretation",
    "invalid",
    "reject_quarantined",
    "pending",
]


class MaskedTerm(BaseModel):
    """One masked term from the task's ``critical_ambiguity`` block.

    Mirrors the BIRD-Interact mini-interact convention: each entry names
    the user-facing term, its ambiguity type, and the metadata sources
    that pin its resolution (or that the annotator believes pin it).
    """
    model_config = ConfigDict(extra="forbid")

    term: str
    type: str
    """E.g. ``knowledge_linking_ambiguity``, ``schema_linking_ambiguity``,
    ``intent_ambiguity``, ``semantic_ambiguity``."""
    is_mask: bool = True
    metadata_evidence: List[str] = Field(default_factory=list)
    """Pointers to the supporting source — e.g.
    ``households_kb.jsonl#10`` or
    ``slayer_models_otf/.../amenities.yaml:cablestatus.sampled_values``."""


class MetadataSufficiency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: MetadataSufficiencyVerdict
    rationale: str
    evidence_sources_consulted: List[str] = Field(default_factory=list)


class AuditedGoldRef(BaseModel):
    """Pointer into the consolidated audited-gold JSONL.

    Lookup key is ``(instance_id, variant_id)``; the file path is
    redundant (callers can resolve from the benchmark) but stored so the
    JSON is self-documenting.
    """
    model_config = ConfigDict(extra="forbid")

    file: str
    """E.g. ``audited_gold/mini_interact_audited.jsonl``."""
    instance_id: str
    variant_id: str = "primary"


class GoldVariantRef(BaseModel):
    """One acceptable interpretation of the task.

    Multi-variant tasks have N entries; one carries ``primary=True``
    (used at interaction time for the user-sim grader and for the
    cascading evaluator's first match). SQL lives in the consolidated
    audited-gold JSONL, NOT inline here.
    """
    model_config = ConfigDict(extra="forbid")

    variant_id: str
    """Slug, e.g. ``canonical_only``, ``synonym_expanded``, ``primary``."""
    interpretation: str
    """1-2 sentences describing what reading this variant embodies."""
    primary: bool = False
    """Exactly one variant per task should have primary=True."""
    anchored_in: List[str] = Field(default_factory=list)
    """Sources in the metadata that license this specific reading."""
    audited_gold_ref: AuditedGoldRef
    notes: Optional[str] = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_jsonl_path: str
    """Where to find the original task row; e.g. ``mini_interact.jsonl``."""
    task_jsonl_instance_id: str
    audited_gold_legacy_path: Optional[str] = None
    """Pre-DEV-1515 callers may still reference per-DB JSONL paths; this
    is a back-pointer for forensic traceability and is allowed to be
    None for benchmarks that never had a per-DB layout."""


class TaskAnnotation(BaseModel):
    """Run-independent annotation of a single benchmark task.

    Path: ``annotations/<benchmark>/<db>/<instance_id>.task.json``.

    For sufficient-and-correct tasks (the common case), ``gold_variants``
    is empty and ``original_gold_is_correct=True``; the evaluator can
    short-circuit against the task's original ``sol_sql``.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    kind: Literal["task_annotation"] = "task_annotation"

    instance_id: str
    selected_database: str
    annotated_by: str
    annotated_at: str
    """ISO-8601 date or datetime."""
    supersedes: Optional[str] = None
    """Path or stable ID of an earlier annotation this replaces."""

    amb_user_query: str
    external_knowledge: List[int] = Field(default_factory=list)
    masked_terms: List[MaskedTerm] = Field(default_factory=list)

    metadata_sufficiency: MetadataSufficiency

    original_gold_is_correct: bool = False
    """True iff metadata_sufficiency.verdict == 'sufficient' AND the
    task's original ``sol_sql`` is the correct answer. When True,
    ``gold_variants`` SHOULD be empty (the original gold is authoritative)."""
    gold_variants: List[GoldVariantRef] = Field(default_factory=list)

    evaluator_prompt: Optional[str] = None
    """Self-contained natural-language LLM-judge prompt. Invoked only
    when ``metadata_sufficiency.verdict == 'insufficient'``. For
    sufficient / ambiguous tasks this MAY be populated as a future-proof
    fallback for novel readings but is not used by the deterministic
    grader."""

    provenance: Provenance


class VariantMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_id: str
    match: RowsetRelation


class SubmissionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cloud_run_id: str
    trajectory_path: str
    submitted_sql_path: Optional[str] = None
    """Optional pointer to a saved .sql sidecar. The submitted SQL is
    also in the trajectory JSON's ``submitted_sql`` field."""
    predicted_row_count: Optional[int] = None
    duration_s: Optional[float] = None
    cost_usd_agent: Optional[float] = None
    cost_usd_user_sim: Optional[float] = None
    n_agent_turns: Optional[int] = None
    n_ask_user_calls: Optional[int] = None


class SubmissionEvaluation(BaseModel):
    """Cascading evaluation, most stringent → most lenient.

    Each field corresponds to one row in the after-run report layout.
    """
    model_config = ConfigDict(extra="forbid")

    phase1_against_original_gold: PhaseVerdict
    phase1_against_audited_primary: PhaseVerdict
    phase1_against_any_audited_variant: PhaseVerdict
    phase1_against_variants: List[VariantMatch] = Field(default_factory=list)
    correct_up_to_tie_order: bool = False
    novel_reading_judgment: Optional[PhaseVerdict] = None
    """Populated only when ``metadata_sufficiency.verdict ==
    'insufficient'`` AND the deterministic check returned no match."""
    verdict: SubmissionVerdict
    matched_variant_id: Optional[str] = None
    rationale: str = ""


class FailureClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: FailurePrimary
    secondary: List[FailurePrimary] = Field(default_factory=list)
    agent_at_fault: bool
    remediation_target: RemediationTarget
    remediation_text: str = ""
    details: str = ""
    """Free-form clarification beyond the enum buckets."""


class TrajectoryDecisionPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_item_index: int
    description: str


class UserSimResponseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_idx: int
    summary: str


class UserSimInteraction(BaseModel):
    """Holds the a-interact summary: how often the agent asked, what the
    sim disclosed vs withheld. For one-shot benchmarks (livesqlbench)
    everything is empty / zero."""
    model_config = ConfigDict(extra="forbid")

    n_asks: int = 0
    key_responses: List[UserSimResponseSummary] = Field(default_factory=list)
    disclosed_resolutions: List[str] = Field(default_factory=list)
    undisclosed_resolutions: List[str] = Field(default_factory=list)


class SubmissionAnnotation(BaseModel):
    """Per-(instance, run) annotation.

    Path:
    ``annotations/<benchmark>/<db>/<instance_id>.submission.<run_id>.json``.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    kind: Literal["submission_annotation"] = "submission_annotation"

    instance_id: str
    selected_database: str
    task_annotation_ref: str
    """Relative path to the task annotation, e.g.
    ``annotations/mini-interact/households/households_10.task.json``."""
    annotated_by: str
    annotated_at: str

    submission: SubmissionMetadata
    evaluation: SubmissionEvaluation
    failure_classification: FailureClassification
    decision_point: Optional[TrajectoryDecisionPoint] = None
    user_sim_interaction: UserSimInteraction = Field(default_factory=UserSimInteraction)
