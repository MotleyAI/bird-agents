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

import datetime as _dt
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator  # noqa: F401

# DEV-1541: caps for AutopsyError excerpt fields.
_AUTOPSY_ERROR_MESSAGE_CAP = 500
_AUTOPSY_ERROR_TRACEBACK_CAP = 2000
_AUTOPSY_ERROR_TRUNCATION_SUFFIX = "...[truncated]"


def _cap_excerpt(value: str, cap: int) -> str:
    """Hard-cap an excerpt string at ``cap`` chars, appending a marker
    when truncation occurred. Used by AutopsyError field validators."""
    if not isinstance(value, str) or len(value) <= cap:
        return value
    keep = cap - len(_AUTOPSY_ERROR_TRUNCATION_SUFFIX)
    if keep <= 0:
        return _AUTOPSY_ERROR_TRUNCATION_SUFFIX[:cap]
    return value[:keep] + _AUTOPSY_ERROR_TRUNCATION_SUFFIX

# Valid prefixes for evidence_sources_consulted entries.
EVIDENCE_SOURCE_PREFIXES: tuple[str, ...] = (
    "kb:",                        # get_knowledge_definition — KB entry N
    "column:",                    # get_column_meaning — <table>.<col>
    "schema:",                    # get_schema — <table>
    "sample_values:",             # get_column_sample_values — <table>.<col>
    "sql:",                       # execute_sql — ad-hoc query description
    "critical_ambiguity:",        # get_ambiguity_resolutions — sql_snippet for masked term
    "critical_ambiguity_evidence:",  # get_ambiguity_resolutions — metadata_evidence for masked term
    "knowledge_ambiguity:",       # get_ambiguity_resolutions — knowledge_ambiguity definition
)

# Top-level enums kept as Literal strings so the JSON encoding is stable
# and human-greppable.
MetadataSufficiencyVerdict = Literal["sufficient", "ambiguous", "insufficient"]
AuditStatus = Literal["clean", "edited", "unrecoverable", "original_correct"]


class AuditedGoldVariant(BaseModel):
    """One audited SQL variant for a task.

    Normalises legacy status values at ingestion: ``clean`` and
    ``original_correct`` → ``"original"``; ``unrecoverable`` is rejected
    (those tasks are represented by ``verdict="insufficient"`` in the task
    annotation instead).

    ``primary=True`` designates the single preferred variant the harness
    and grader use when only one answer is needed.  Exactly one variant
    per task must carry this flag.
    """

    model_config = ConfigDict(extra="ignore")

    variant_id: str
    audit_status: Literal["original", "edited"]
    audited_sol_sql: list[str]
    primary: bool = False
    notes: str | None = None

    @field_validator("audit_status", mode="before")
    @classmethod
    def _normalise_status(cls, v: object) -> str:
        if v in ("clean", "original_correct"):
            return "original"
        if v == "unrecoverable":
            raise ValueError(
                "unrecoverable variants must not be stored in the audited gold JSONL; "
                "use verdict='insufficient' + evaluator_prompt in the task annotation instead"
            )
        return str(v)

    @model_validator(mode="after")
    def _non_empty_sql(self) -> "AuditedGoldVariant":
        if not self.audited_sol_sql:
            raise ValueError("audited_sol_sql must contain at least one SQL string")
        return self


class AuditedGoldRow(BaseModel):
    """One task's complete audited gold — all variants grouped.

    One line per ``instance_id`` in the consolidated JSONL.  Replaces the
    legacy flat-row-per-variant layout.
    """

    model_config = ConfigDict(extra="ignore")

    instance_id: str
    selected_database: str
    benchmark: str
    variants: list[AuditedGoldVariant]

    @model_validator(mode="after")
    def _check_variants(self) -> "AuditedGoldRow":
        if not self.variants:
            raise ValueError(f"{self.instance_id}: variants must be non-empty")
        n_primary = sum(1 for v in self.variants if v.primary)
        if n_primary != 1:
            raise ValueError(
                f"{self.instance_id}: exactly one variant must have primary=True; "
                f"got {n_primary}"
            )
        return self

    @property
    def primary_variant(self) -> "AuditedGoldVariant":
        return next(v for v in self.variants if v.primary)

    @classmethod
    def from_flat_rows(cls, rows: list[dict]) -> "AuditedGoldRow":
        """Build from legacy flat rows (one dict per variant). Skips unrecoverable rows.

        If no variant has ``primary=True``, defaults the first surviving
        variant to primary.  ``variant_id`` is auto-generated as ``"v{i}"``
        when absent (legacy files pre-date the field).
        """
        variants: list[AuditedGoldVariant] = []
        for i, r in enumerate(rows):
            d = dict(r)
            if d.get("audit_status") == "unrecoverable":
                continue
            if "variant_id" not in d:
                d["variant_id"] = f"v{i}"
            if "variant_description" in d and "notes" not in d:
                d["notes"] = d.pop("variant_description")
            else:
                d.pop("variant_description", None)
            d.pop("is_gold", None)
            try:
                variants.append(AuditedGoldVariant.model_validate(d))
            except (ValidationError, Exception):
                continue
        if not variants:
            raise ValueError("no valid variants after filtering unrecoverable rows")
        if not any(v.primary for v in variants):
            variants[0] = variants[0].model_copy(update={"primary": True})
        first = next(
            (r for r in rows if r.get("instance_id") and r.get("selected_database") and r.get("benchmark")),
            None,
        )
        if first is None:
            raise ValueError(
                "legacy rows missing required task-level fields: "
                "instance_id, selected_database, benchmark"
            )
        return cls(
            instance_id=first["instance_id"],
            selected_database=first["selected_database"],
            benchmark=first["benchmark"],
            variants=variants,
        )


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
    "no_fail",
    "agent_miss",
    "metadata_ambiguity",
    "gold_audit_quality",
    "user_sim_under_disclosure",
    "novel_reading_accepted",
    "numerical_precision",
    "row_order",
    "trailing_whitespace",
    "column_order",
    "case_sensitivity",
    "other",
]
"""``no_fail`` ⇒ agent passed N3-strict; nothing to attribute.
``other`` ⇒ failure happened but doesn't fit any listed category.

Cascade-tier relaxation buckets (``row_order``, ``numerical_precision``,
``trailing_whitespace``, ``column_order``, ``case_sensitivity``) each
flag exactly which grader-tolerance dimension flipped the verdict.

``novel_reading_accepted`` ⇒ strict cascade (N1..N4, N6..N8) failed AND
``task.metadata_sufficiency.verdict == "insufficient"`` AND the LLM
judge accepted the agent's reading as a valid novel interpretation
(N5). The underlying issue is the task's metadata — the judge merely
ratified the agent's plausible reading."""
RemediationTarget = Literal["agent", "prompt", "kb", "audit", "grader", "user_sim", "gold_sidecar", "other"]
SubmissionVerdict = Literal[
    "correct",
    "valid_interpretation",
    "agent_miss",          # all cascade tiers failed; task is evaluable
    "eval_failed",         # infrastructure failure / no gold / grader raised
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
    """Verdict semantics (sharpened DEV-1515):

    * ``sufficient``  — published metadata (KB + column meanings +
      sampled values) alone pins the answer; no user_sim help needed.
    * ``ambiguous``   — published metadata licenses multiple readings,
      BUT a maximally cooperative user_sim disclosing the masked
      sql_snippet on a well-phrased ask would resolve them. If the
      run-side sim refused / under-disclosed, the submission's
      ``failure_classification.primary`` should be
      ``user_sim_under_disclosure`` (not ``metadata_ambiguity``).
    * ``insufficient`` — even with maximally cooperative user_sim
      disclosure of the masked sql_snippet, the answer remains
      underspecified (e.g. arbitrary policy thresholds / coefficients
      with no semantic anchor). The companion ``evaluator_prompt`` then
      describes what counts as a valid LLM-judge-accepted reading.
    """
    model_config = ConfigDict(extra="forbid")

    verdict: MetadataSufficiencyVerdict
    rationale: str
    evidence_sources_consulted: List[str] = Field(default_factory=list)

    @field_validator("evidence_sources_consulted")
    @classmethod
    def _validate_evidence_prefixes(cls, v: List[str]) -> List[str]:
        bad = [s for s in v if not any(s.startswith(p) for p in EVIDENCE_SOURCE_PREFIXES)]
        if bad:
            raise ValueError(
                f"evidence_sources_consulted entries have unrecognised prefixes: {bad}. "
                f"Valid prefixes: {', '.join(EVIDENCE_SOURCE_PREFIXES)}"
            )
        return v


class AuditedGoldRef(BaseModel):
    """Pointer into the consolidated audited-gold JSONL.

    Lookup key is ``(instance_id, variant_id)``; the file path is
    redundant (callers can resolve from the benchmark) but stored so the
    JSON is self-documenting.
    """
    model_config = ConfigDict(extra="forbid")

    file: str
    """E.g. ``audited_gold/mini-interact/mini-interact_audited.jsonl``."""
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


InconsistencyResolution = Literal[
    "multi_variant",
    "picked_one_variant",
    "unresolved",
]


class InternalInconsistency(BaseModel):
    """Flag set when two or more authoritative sources within the task
    disagree on the SAME parameter — an agent cannot satisfy all sources
    simultaneously.

    Examples
    --------
    * KB description says ``LCS > 3``; masked sql_snippet says
      ``lcs > 2`` for the same defined concept.
    * KB definition field contradicts the KB description field.
    * The user_query phrasing implies one aggregation; the masked
      snippet specifies another.

    This is task-level: it describes the source landscape, not the
    agent's submission. Most tasks have ``internal_inconsistency =
    None`` (sources converge).
    """
    model_config = ConfigDict(extra="forbid")

    sources_in_conflict: List[str]
    """Citations naming each conflicting source. E.g.
    ``['KB#29.description: "LCS > 3"', 'critical_ambiguity for "good
    quality of life".sql_snippet: "lcs > 2"']``."""

    description: str
    """1-3 sentences explaining what each source says and why an agent
    cannot satisfy them all at once. Quote the disagreeing values."""

    audit_resolution: InconsistencyResolution
    """How the audit handled the inconsistency:

    * ``multi_variant`` — audited_gold JSONL has N rows, one per
      reading; the task's ``gold_variants`` mirrors them.
    * ``picked_one_variant`` — audit chose one reading; the alternate
      is documented here but is NOT in the audited_gold JSONL.
    * ``unresolved`` — audit declared the task unanswerable;
      ``audit_status`` is typically ``unrecoverable``.
    """


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
    external_knowledge: List[Union[int, dict]] = Field(default_factory=list)
    """KB references the task anchors against. Each entry is either an
    integer KB id (the common mini-interact / livesqlbench shape) OR a
    dict carrying the KB body inline (some fixture rows and forward-
    looking benchmark variants). Codex r13: the prior ``List[int]`` was
    overly restrictive — dict-shaped entries would be rejected by
    pydantic at conversion time."""
    masked_terms: List[MaskedTerm] = Field(default_factory=list)

    metadata_sufficiency: MetadataSufficiency

    original_gold_is_correct: Optional[bool] = None
    """True iff metadata_sufficiency.verdict == 'sufficient' AND the
    task's original ``sol_sql`` is the correct answer. When True,
    ``gold_variants`` SHOULD be empty (the original gold is authoritative).

    ``None`` means the field was absent when the annotation was written
    (legacy / pre-DEV-1533 annotation). The grader treats ``None`` as
    ``True`` — preserving the old monotone behaviour for unannotated tasks
    so that N1=True still implies N2+ when the gold provenance is unknown.
    Explicit ``False`` means the annotator confirmed the original is wrong."""
    gold_variants: List[GoldVariantRef] = Field(default_factory=list)

    evaluator_prompt: Optional[str] = None
    """Self-contained natural-language LLM-judge prompt. Invoked only
    when ``metadata_sufficiency.verdict == 'insufficient'``. For
    sufficient / ambiguous tasks this MAY be populated as a future-proof
    fallback for novel readings but is not used by the deterministic
    grader."""

    # Set when authoritative sources disagree on the same parameter (KB vs
    # sql_snippet, KB.description vs KB.definition, etc.).  When set,
    # audit_resolution=multi_variant should be reflected in gold_variants.
    internal_inconsistency: Optional[InternalInconsistency] = None

    @model_validator(mode="after")
    def _check_gold_invariants(self) -> "TaskAnnotation":
        primary_count = sum(1 for v in self.gold_variants if v.primary)
        if primary_count > 1:
            raise ValueError(
                f"gold_variants must contain at most one primary variant; got {primary_count}"
            )
        if self.original_gold_is_correct and self.gold_variants:
            raise ValueError(
                "gold_variants must be empty when original_gold_is_correct is True"
            )
        if self.gold_variants and not self.original_gold_is_correct and primary_count == 0:
            raise ValueError(
                "gold_variants must contain exactly one primary variant; got 0. "
                "Mark one variant with primary=True."
            )
        return self

    provenance: Provenance


class VariantInformational(BaseModel):
    """Tier-2 diagnostic block populated per (submission, variant).

    Carries deterministic, non-verdict-flipping signal: rowset relation,
    column-structure comparison, and the first divergent cell (if any).
    Consumers reading this can decide which variant to fix vs. which is
    closest to the agent's intent without re-executing.
    """
    model_config = ConfigDict(extra="forbid")

    rowset_relation: RowsetRelation
    column_count_match: bool
    column_order_match: bool
    first_divergent_row_index: Optional[int] = None
    first_divergent_cell_diff: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_naming_field(cls, data):
        # DEV-1534 Fix A: column NAMING signals are no longer recorded
        # (autopsy agents misattributed cascade-fails to "column naming
        # mismatch" when the real cause was value differences). Strip
        # the named legacy key on load so pre-Fix-A on-disk annotations
        # still parse. Other unknowns still hit `extra="forbid"`.
        if isinstance(data, dict) and "column_name_match_case_insensitive" in data:
            data = {k: v for k, v in data.items()
                    if k != "column_name_match_case_insensitive"}
        return data


class VariantMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_id: str
    match: RowsetRelation
    informational: Optional[VariantInformational] = None


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


MissPattern = Literal[
    "sql_execution_error",
    "sql_parse_error",
    "empty_agent_result",
    "wrong_table_set",
    "aggregation_shape_mismatch",
    "column_count_mismatch",
    "column_order_mismatch",
    "predicate_count_mismatch",
    "having_presence_mismatch",
    "limit_presence_mismatch",
    "disjoint_rowset",
    "partial_match_overlap",
    "agent_undercount",
    "agent_overcount",
    "never_asked_user",
]
# Column-shape flag design (intentionally split, not lumped):
#   * ``column_count_mismatch`` — agent and gold have different column
#     counts. Cell tuples have different arity → bag equality CANNOT
#     hold (same for BIRD-Interact's ``ex_base``). Load-bearing cause
#     of cascade fail.
#   * ``column_order_mismatch`` — counts match AND the column-name
#     LISTS, after normalisation (lowercase + strip longest dot-prefix,
#     so 'households.housenum' → 'housenum'), are equal AS SETS but
#     differ AS LISTS. Surfaces a "near miss" where the agent's
#     projection content is correct but positionally misaligned —
#     cascade N3 fails because row reprs are positional; N8 column-
#     order tolerance might have rescued it but didn't because
#     slayer's namespacing trips its column-name set check.
#   * Pure column-NAME divergence with matching counts AND non-matching
#     normalised sets is INTENTIONALLY NOT FLAGGED — that's stylistic
#     (agent projected meaningfully different columns) and would fire
#     spuriously on every slayer-namespaced submission. Only
#     ``column_count_match`` and ``column_order_match`` are surfaced
#     post-DEV-1534 (Fix A removed the NAMING signal — autopsy agents
#     misattributed cascade-fails to "column naming mismatch" when the
#     real cause was value differences; column ORDER stays as a real
#     positional cause).


class MissDiagnostics(BaseModel):
    """Diagnostic signals collected at grading time for cascade-fail
    submissions. Populated when the cascade has no passing tier
    (every N1-N9 returned False).

    Comparison reference is the BEST-OVERLAP audited variant — the
    variant with the largest multiset (bag) intersection with the
    agent's rowset. Tie-break: prefer the primary variant; then
    alphabetical variant_id. Single comparison target keeps the schema
    flat; the variant identifier is recorded so downstream consumers
    know which variant the signals are measured against.

    Works uniformly across interactive (a-interact) and one-shot
    benchmarks. Interactive-only fields stay None when the benchmark
    has no user-sim.

    SQL-derived fields are Optional[T] = None when sqlglot can't
    parse the corresponding SQL. The ``*_sql_parse_ok`` booleans
    + ``*_sql_parse_error`` excerpts make the parse state explicit
    so downstream consumers don't read a False/0 as a real signal.
    """
    model_config = ConfigDict(extra="forbid")

    # Comparison reference identifier.
    best_variant_id: str

    # Row counts (multiset cardinality matches grader bag semantics).
    agent_row_count: int
    best_variant_row_count: int
    original_gold_row_count: Optional[int] = None
    overlap_with_best: int
    rowset_relation_to_best: RowsetRelation

    # Column shape. (DEV-1534 Fix A removed
    # column_name_match_case_insensitive, agent_columns, and
    # best_variant_columns — column NAMING is irrelevant to grading;
    # column ORDER stays as a real positional cause.)
    agent_column_count: int
    best_variant_column_count: int
    column_count_match: bool
    column_order_match: bool
    first_divergent_cell_diff: Optional[str] = None

    # SQL parse status gates the SQL-derived fields below.
    agent_sql_parse_ok: bool
    best_variant_sql_parse_ok: bool
    agent_sql_parse_error: Optional[str] = None
    best_variant_sql_parse_error: Optional[str] = None

    # SQL-derived signals — Optional[T]=None on the parse-failed side.
    # Tables: base tables only, CTE / derived aliases excluded;
    # alphabetically sorted for stable JSON diffs.
    agent_tables_referenced: Optional[List[str]] = None
    best_variant_tables_referenced: Optional[List[str]] = None
    table_set_match: Optional[bool] = None
    agent_has_group_by: Optional[bool] = None
    best_variant_has_group_by: Optional[bool] = None
    agent_has_aggregate: Optional[bool] = None
    best_variant_has_aggregate: Optional[bool] = None
    agent_join_count: Optional[int] = None
    best_variant_join_count: Optional[int] = None
    agent_where_conjunct_count: Optional[int] = None
    best_variant_where_conjunct_count: Optional[int] = None
    agent_has_having: Optional[bool] = None
    best_variant_has_having: Optional[bool] = None
    agent_has_limit: Optional[bool] = None
    best_variant_has_limit: Optional[bool] = None

    # Execution status of the agent's SQL.
    agent_sql_executed_ok: bool
    agent_sql_error_excerpt: Optional[str] = None

    # Interactive-only signal — None on one-shot benchmarks.
    user_sim_n_asks: Optional[int] = None

    # Independent flag list — every applicable rule appends. Sorted
    # alphabetically before persist for stable JSON diffs.
    miss_patterns: List[MissPattern] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_naming_fields(cls, data):
        # DEV-1534 Fix A: strip the named legacy column-NAMING fields so
        # pre-Fix-A on-disk annotations still parse. The allowlist is
        # scoped to keys that WERE fields on this model — other unknown
        # keys still hit `extra="forbid"`.
        if not isinstance(data, dict):
            return data
        legacy = (
            "column_name_match_case_insensitive",
            "agent_columns",
            "best_variant_columns",
        )
        if any(k in data for k in legacy):
            data = {k: v for k, v in data.items() if k not in legacy}
        return data


class SubmissionEvaluation(BaseModel):
    """Cascading evaluation, most stringent → most lenient.

    Each field corresponds to one row in the after-run report layout
    (N1 through N8 of the cascade).
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
    correct_under_numeric_epsilon: bool = False
    correct_under_trailing_whitespace: bool = False
    correct_under_column_order: bool = False
    correct_under_case_fold: bool = False
    numeric_epsilon: float = 1e-6
    """Records the epsilon threshold actually used for N6, so consumers
    of the annotation know how lenient the relaxation was."""
    verdict: SubmissionVerdict
    matched_variant_id: Optional[str] = None
    rationale: str = ""
    miss_diagnostics: Optional[MissDiagnostics] = None
    """DEV-1515 session-4 — populated only when the cascade has no
    passing tier (strict miss). Captures rowset-shape, column-shape,
    SQL-derived signals, and execution status to support downstream
    failure-mode analysis without re-running queries."""


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


AutopsyPattern = Literal[
    "never_asked_key_question",
    "asked_but_ignored_answer",
    "user_sim_misleading",
    "late_mutation_corrupted_result",
    "wrong_join_path",
    "output_schema_misread",
    "slayer_generation_artifact",
    "exhausted_budget_guessing",
    "other",
]
"""LLM-produced failure taxonomy for genuine cascade misses on
``claude_sdk_otf*`` frameworks (DEV-1521).

A-interact (claude_sdk_otf_ainteract) tasks use this full 9-pattern set."""

AutopsyPatternOneShot = Literal[
    "late_mutation_corrupted_result",
    "wrong_join_path",
    "output_schema_misread",
    "slayer_generation_artifact",
    "exhausted_budget_guessing",
    "other",
]
"""DEV-1541: one-shot benchmarks (livesqlbench) have no user-sim and
must NOT be tagged with ``never_asked_key_question`` /
``asked_but_ignored_answer`` / ``user_sim_misleading`` — those patterns
are unactionable on one-shot.

* ``never_asked_key_question`` — agent never surfaced a critical
  clarification; the answer was recoverable via ``ask_user`` but skipped.
* ``asked_but_ignored_answer`` — agent asked the right question but
  then disregarded the answer.
* ``user_sim_misleading`` — user-sim gave an incorrect, incomplete, or
  actively misleading answer that steered the agent wrong.
* ``late_mutation_corrupted_result`` — correct intermediate result; a
  late transformation (LOWER/TRIM/ROUND/CAST/schema change) corrupted
  the final output.
* ``wrong_join_path`` — agent joined via a wrong/missing join path, or
  placed encoding on the wrong host model.
* ``output_schema_misread`` — agent projected wrong columns, wrong
  aggregation shape, or misread the expected row structure.
* ``slayer_generation_artifact`` — SQL emitted by SLayer had a bug
  (integer division, broken namespace) not from the agent's encoding
  choices.
* ``exhausted_budget_guessing`` — agent consumed all turns on
  exploratory/repeated attempts without converging.
* ``other`` — doesn't fit the above; details in ``AutopsyAnalysis.other_details``.
"""


class AutopsyAnalysis(BaseModel):
    """LLM-produced post-mortem analysis for an a-interact cascade miss.

    Populated only for ``claude_sdk_otf_ainteract`` runs.
    ``kind="a_interact"`` is the discriminator tag for the
    ``AutopsyResult.analysis`` discriminated union (DEV-1541).
    """
    model_config = ConfigDict(extra="forbid")

    kind: Literal["a_interact"] = "a_interact"
    pattern: AutopsyPattern
    other_details: Optional[str] = None
    """Non-None when ``pattern == "other"``; freeform description of
    the failure mode."""
    narrative: str
    """2–5 sentence explanation of what went wrong, referencing specific
    trajectory steps by index."""
    remediation: str
    """1–2 sentence actionable suggestion: what should change
    (prompt / KB / grader / user_sim)."""


class AutopsyAnalysisOneShot(BaseModel):
    """LLM-produced post-mortem analysis for a one-shot cascade miss.

    Populated only for ``claude_sdk_otf`` runs on one-shot benchmarks
    (livesqlbench). ``kind="one_shot"`` is the discriminator tag for the
    ``AutopsyResult.analysis`` discriminated union (DEV-1541).
    """
    model_config = ConfigDict(extra="forbid")

    kind: Literal["one_shot"] = "one_shot"
    pattern: AutopsyPatternOneShot
    other_details: Optional[str] = None
    narrative: str
    remediation: str


_AutopsyErrorKind = Literal[
    "validation_error",
    "context_overflow",
    "api_error",
    "network_error",
    "missing_tool_use",
    "unknown",
]


class AutopsyError(BaseModel):
    """DEV-1541: persisted record of an autopsy failure.

    Before DEV-1541, autopsy failures were swallowed (``except Exception:
    return None``) and the resulting ``autopsy=None`` annotation was
    indistinguishable from "autopsy didn't run". This type captures the
    failure shape so backfill tooling and reviewers can tell
    silent-fail from skipped from succeeded.

    ``message_excerpt`` and ``traceback_excerpt`` are auto-capped (hard
    truncate with ``...[truncated]`` suffix) at construction time so the
    contract holds even when a caller forgets to pre-truncate.
    """
    model_config = ConfigDict(extra="forbid")

    kind: _AutopsyErrorKind
    exception_class: str
    """Fully-qualified class name, e.g. ``pydantic_core._pydantic_core.ValidationError``."""

    message_excerpt: str
    """Exception message; capped to 500 chars."""

    traceback_excerpt: str
    """``traceback.format_exc()``; capped to 2000 chars."""

    prompt_chars: int
    kb_chars: int
    trajectory_items: int
    model: str
    timestamp: _dt.datetime

    @field_validator("message_excerpt", mode="before")
    @classmethod
    def _cap_message(cls, v: Any) -> Any:
        return _cap_excerpt(v, _AUTOPSY_ERROR_MESSAGE_CAP) if isinstance(v, str) else v

    @field_validator("traceback_excerpt", mode="before")
    @classmethod
    def _cap_traceback(cls, v: Any) -> Any:
        return _cap_excerpt(v, _AUTOPSY_ERROR_TRACEBACK_CAP) if isinstance(v, str) else v


_AnalysisUnion = Annotated[
    Union[AutopsyAnalysis, AutopsyAnalysisOneShot],
    Field(discriminator="kind"),
]


class AutopsyResult(BaseModel):
    """Container for autopsy outputs.

    Either ``analysis`` (successful autopsy) OR ``error`` (DEV-1541
    failure capture) is set, never both, never neither.

    Lives in the schema module (pure data, no client deps) so
    ``grade_in_place`` can import it without pulling in the Anthropic SDK.
    """
    model_config = ConfigDict(extra="forbid")

    analysis: Optional[_AnalysisUnion] = None
    error: Optional[AutopsyError] = None
    decision_point: Optional[TrajectoryDecisionPoint] = None
    user_sim_interaction: Optional[UserSimInteraction] = None

    @model_validator(mode="before")
    @classmethod
    def _inject_legacy_kind(cls, data: Any) -> Any:
        """DEV-1541 back-compat: pre-DEV-1541 on-disk annotations have
        an ``analysis`` dict without ``kind``. Treat them as ``a_interact``
        (the only shape the old code emitted)."""
        if isinstance(data, dict):
            analysis = data.get("analysis")
            if isinstance(analysis, dict) and "kind" not in analysis:
                analysis["kind"] = "a_interact"
        return data

    @model_validator(mode="after")
    def _exactly_one(self) -> "AutopsyResult":
        if (self.analysis is None) == (self.error is None):
            raise ValueError(
                "AutopsyResult must have exactly one of `analysis` or `error` set"
            )
        return self


class SubmissionAnnotation(BaseModel):
    """Per-(instance, run) annotation.

    Path (DEV-1533):
    ``runs/<benchmark>/<db>/<instance_id>/<run_id>.json``.
    """
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _migrate_invalid_verdict(cls, data: Any) -> Any:
        """Silently upgrade legacy verdict="invalid" annotations on read.

        Old on-disk annotations use "invalid" for two different cases that
        are now distinguished: genuine agent misses ("agent_miss") and
        grader/infrastructure failures ("eval_failed").
        failure_classification.primary="other" → infra failure → "eval_failed";
        anything else → "agent_miss".
        """
        if not isinstance(data, dict):
            return data
        ev = data.get("evaluation")
        fc = data.get("failure_classification")
        if isinstance(ev, dict) and ev.get("verdict") == "invalid":
            if isinstance(fc, dict) and fc.get("primary") == "other":
                ev["verdict"] = "eval_failed"
            else:
                ev["verdict"] = "agent_miss"
        return data

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
    user_sim_interaction: Optional[UserSimInteraction] = Field(default_factory=UserSimInteraction)
    """DEV-1541: type widened to ``Optional`` so one-shot runs can persist
    ``null``. The ``default_factory`` is retained so missing-from-JSON
    legacy annotations still parse to ``UserSimInteraction(n_asks=0)``
    rather than ``None`` — preserving the pre-DEV-1541 zero-asks meaning
    of an omitted field."""
    autopsy: Optional[AutopsyResult] = None
    """LLM-produced post-mortem. Populated only for genuine cascade misses
    (all N1–N9 fail) on claude_sdk_otf* frameworks. None for passing
    submissions and for all other frameworks."""

    # DEV-1533: run-level data stored alongside the grading result.
    submitted_sql: Optional[str] = None
    """The agent's final submitted SQL string."""
    predicted_result: Optional[List[Any]] = None
    """The rows returned by executing the agent's SQL against the DB."""
    gold_result: Optional[List[Any]] = None
    """The reference gold rows (from executing the best-matching gold SQL)."""
    original_gold_annotated_correct: Optional[bool] = None
    """Whether the original sol_sql was annotated as correct in the task
    annotation (``task_annotation.original_gold_is_correct``). Used by the
    partition cascade to distinguish L1 (correct original matched) from L2
    (wrong original coincidentally matched). None for migrated/legacy
    annotations that pre-date DEV-1533."""
