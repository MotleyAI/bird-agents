---
name: annotate-task-submission
description: Fill the human-judgment fields on a bird-interact `TaskAnnotation` + `SubmissionAnnotation` (DEV-1515 schema). Use when reviewing a benchmark task's metadata sufficiency, classifying a submission's failure mode, or filling skeletons produced by `scripts/dev1515_convert_runs.py` / `bird_interact_agents.eval.annotate`. Encodes the sharpened `metadata_sufficiency.verdict` rule that accounts for user-sim disclosure of the masked sql_snippet, AND the rule that buggy audits get fixed in place — annotations never ratify a broken audited gold.
---

# Annotate a bird-interact task + submission

This skill is the per-task annotation recipe for DEV-1515. It picks the
right `metadata_sufficiency.verdict` and `failure_classification` for a
given (task, submission) pair, given the published metadata + the masked
sql_snippet + the run-side trajectory.

The Pydantic schemas live in
`src/bird_interact_agents/eval/annotation_schema.py`; they validate with
`extra="forbid"` — never invent fields.

## Two phases — task annotation is self-contained

The recipe has two independent phases, and you can stop after the first:

1. **Task annotation** (`<inst>.task.json`) — needs only task-side
   inputs (KB, column meanings, schema, gold sidecar, audited gold).
   Fills `metadata_sufficiency.verdict` + `gold_variants` +
   `evaluator_prompt` + `internal_inconsistency`. Produce this whenever
   you author a new benchmark DB; no cloud run required.
2. **Submission annotation** (`<inst>.submission.<run-id>.json`) — adds
   per-(instance, run) `failure_classification` + cascade tier on top
   of the task annotation. Skip this phase entirely when no run exists;
   the task annotation stands on its own and downstream tooling
   (cascade summary, U/C buckets) gracefully handles missing
   submissions.

When in doubt, the rule of thumb: if you're not looking at a
`results/cloud/<run-id>/rows/<inst>/attempt-1.json`, you're in phase 1
only. Read just the phase-1 inputs below + the verdict decision flow;
ignore everything tagged "submission" or "failure_classification".

## NON-NEGOTIABLE: fix buggy audits — never annotate around them

Before classifying ANY task where `audited_sol_sql != original_sol_sql`,
verify the audited gold itself is correct against the published
metadata + the masked sql_snippet + the live DB rowset.

* If the audit is correct → write the annotation describing what was
  wrong with the ORIGINAL gold and (with citations) why the audited
  version is canonical. Be generous with rationale text.
* If the audit is wrong → FIX it. Rewrite
  `audited_sol_sql` / `changes` / `reasoning_summary` / `audit_status`
  in `audited_gold/<benchmark>_audited.jsonl` directly. THEN write the
  annotation as above.
* Never carry an `ambiguous` / `insufficient` verdict that silently
  ratifies a buggy audit — that conflates "gold is underspecified" with
  "the audit didn't finish its job".

Annotations are the read-only documentation layer. The audited gold is
the eval pipeline's authoritative reference. Bugs land in the gold,
not in the annotation's rationale.

## When sources contradict: multi-variant golds + internal_inconsistency

A separate case from "audit is wrong" is "audit had to pick a side
because the task's authoritative sources disagree among themselves".
Examples:

* KB description says `LCS > 3`, but the masked sql_snippet for the
  same defined concept says `lcs > 2`.
* KB.description and KB.definition disagree.
* The user_query phrasing implies one aggregation; the masked snippet
  specifies a different one.

The conditions are **internally inconsistent** — an agent cannot
satisfy all sources simultaneously. Don't bury this in the audit's
rationale paragraph. Instead:

1. **Emit BOTH readings as gold variants in the audit JSONL.** Two
   rows sharing `instance_id`, distinct `variant_id`s, exactly one
   with `primary=true`. The KB-anchored reading typically takes
   primary; the snippet-anchored reading is the alternate. Each row's
   `reasoning_summary` cites its source explicitly and names the
   conflict with the other.
2. **Set `TaskAnnotation.internal_inconsistency`** to an
   `InternalInconsistency` record with:
   * `sources_in_conflict`: list of citations, one per source,
     quoting the disagreeing value (e.g.
     `"KB#29.description: 'LCS > 3'"`,
     `"critical_ambiguity for 'good quality of life'.sql_snippet:
     'lcs > 2'"`).
   * `description`: 1–3 sentences explaining what each source says and
     why an agent can't satisfy both at once.
   * `audit_resolution`: typically `"multi_variant"` (see step 1).
     Use `"picked_one_variant"` when emitting two variants would be
     redundant for grading (rare), or `"unresolved"` when the audit
     declared the task unanswerable.
3. **Set `verdict = "ambiguous"`** — NOT `sufficient` (sources DON'T
   converge), NOT `insufficient` (each source independently pins a
   reading). The published metadata licenses multiple readings
   precisely because the sources contradict.
4. **`gold_variants`** carries one entry per audit row, each with an
   `interpretation` field naming which source it's faithful to. The
   `notes` field on each variant can echo the conflict for redundancy.
5. **Submission cascade behaviour falls out**: existing N3 ("any
   audited variant matches") already iterates variants, so an agent
   that picked EITHER reading passes. The remaining failure-class
   work happens on the agent's actual error, not on the source
   conflict.

The `audit-gold-sql` skill carries the audit-side rule (two rows when
sources disagree, `clause_kind="source_conflict"` on the diff entry,
audit_status stays `edited`). This skill carries the annotation-side
rule (the `internal_inconsistency` block + multi-variant
`gold_variants`).

## Inputs

### Phase 1 — task annotation (always required)

Read-only — **mini-interact**:

- `mini-interact/mini_interact.jsonl` — task row. Fields: `amb_user_query`
  (the user-facing question), `external_knowledge`,
  `user_query_ambiguity.critical_ambiguity` (the masked sql_snippets +
  ambiguity-type tags), `knowledge_ambiguity` (KB-id-tagged snippets),
  `sol_sql` (original gold), `selected_database`.
- `mini-interact/<db>/<db>_kb.jsonl` — KB entries referenced by
  `external_knowledge`.
- `mini-interact/<db>/<db>_column_meaning_base.json` — column
  descriptions + sampled-value summaries.
- `mini-interact/<db>/<db>.sqlite` — live DB, for sanity-execute checks.

Read-only — **livesqlbench**:

- `livesqlbench-base-lite-sqlite/livesqlbench_data_sqlite.jsonl` — task
  row. Fields: `query` (natural-language only; NO masked snippet),
  `selected_database`, `category`, `difficulty_tier`, `high_level`,
  `conditions`. NO `sol_sql` or `external_knowledge` here.
- `livesqlbench-base-lite-sqlite/livesqlbench_sqlite_gt_kg_testcases_0528.jsonl`
  — gold sidecar. Fields per row: `instance_id`, `sol_sql`,
  `external_knowledge`, `test_cases`. Join by `instance_id`.
- `livesqlbench-base-lite-sqlite/<db>/<db>_kb.jsonl`
- `livesqlbench-base-lite-sqlite/<db>/<db>_column_meaning_base.json`
- `livesqlbench-base-lite-sqlite/<db>/<db>_schema.txt` — schema dump;
  the agent sees this at task time.
- `livesqlbench-base-lite-sqlite/<db>/<db>.sqlite`

Read+write (both benchmarks):

- `audited_gold/<benchmark>_audited.jsonl` — audit sidecar.
  **Read first to assess correctness; write only when fixing a buggy
  audit.** Filename pattern: `mini_interact_audited.jsonl` /
  `livesqlbench_audited.jsonl`.
- `annotations/<benchmark>/<db>/<inst>.task.json` — task annotation.
  Benchmark slug: `mini-interact` / `livesqlbench`.

### Phase 2 — submission annotation (only when a run exists)

Read-only:

- `results/cloud/<run-id>/rows/<inst>/attempt-1.json` — `submitted_sql`,
  `trajectory`, `usage`, `phase1_observation_*`. (livesqlbench
  trajectories are short — one-shot.)

Read+write:

- `annotations/<benchmark>/<db>/<inst>.submission.<run-id>.json` —
  per-(instance, run) submission annotation. Skip this file entirely
  when phase 1 stands alone.

## What "masked sql_snippet" means

Tasks in BIRD-Interact-style benchmarks ship `amb_user_query` containing
both natural-language framing AND a SQL skeleton with identifying
terms redacted. That skeleton — the **masked sql_snippet** — encodes
predicates / formulas / literal lists from the gold SQL, minus the
specific terms the masking process removed. In a-interact mode the
user_sim has access to the un-masked snippet and is expected to
disclose specifics on a well-phrased ask.

**This is the key rule for `metadata_sufficiency.verdict`.** The verdict
answers: "given perfect user_sim cooperation, can the agent uniquely
derive the gold's reading?"

## `metadata_sufficiency.verdict` decision flow

The flow forks by benchmark mode. `mini-interact` carries masked
sql_snippets in `user_query_ambiguity.critical_ambiguity[].sql_snippet`
that a cooperative user_sim discloses in a-interact mode.
`livesqlbench` is one-shot only — no user_sim, no masked snippets —
so the agent's only sources are the published metadata.

### mini-interact (a-interact mode)

```
1. Audit-correctness gate — is the audited gold itself correct?
     NO  → FIX the audit JSONL row first; come back here.
     YES → proceed.

2. Does the published metadata alone (KB + column meanings + sampled values)
   pin a unique correct answer, with no user_sim help needed?
     YES → verdict = "sufficient"
     NO  → go to step 3.

3. Does the masked sql_snippet (from
   `user_query_ambiguity.critical_ambiguity[].sql_snippet` /
   `knowledge_ambiguity[].sql_snippet`), when fully disclosed by a
   maximally cooperative user_sim, uniquely pin the answer?
     YES → verdict = "ambiguous"
     NO  → verdict = "insufficient"
```

Corollaries:

- **`sufficient` ≠ "agent passed"**. It only says the metadata anchored
  the answer. An agent can still miss a `sufficient` task (then the
  submission-side `primary` is `agent_miss`).
- **`ambiguous` is the "would-have-worked-with-a-good-sim" bucket.** If
  the actual run-side sim withheld snippet content the agent needed,
  the submission-side `primary` should be `user_sim_under_disclosure`,
  not `metadata_ambiguity`. If the agent had `n_asks=0`, it's still
  `agent_miss` — the agent didn't even try.
- **`insufficient` is rare.** Reserved for tasks where even the masked
  snippet encodes arbitrary policy (coefficients, thresholds) with no
  semantic anchor in any source. Audit-status `unrecoverable` is a
  strong signal. Common false positive: the snippet's literal looks
  arbitrary but is actually a translation of a KB-anchored band into
  the data's native units (verify before claiming insufficient).

### livesqlbench (one-shot mode)

```
1. Audit-correctness gate — is the audited gold itself correct?
     NO  → FIX the audit JSONL row first; come back here.
     YES → proceed.

2. Does the published metadata alone (KB + column meanings + sampled
   values + schema.txt + external_knowledge anchors) pin a unique
   correct answer?
     YES → verdict = "sufficient"
     NO  → go to step 3.

3. Does the KB / external_knowledge license a defensible non-gold
   reading?
     YES → verdict = "ambiguous"
     NO  → verdict = "insufficient"
```

Corollaries for livesqlbench:

- There is NO masked sql_snippet and NO user_sim. The agent's only
  resort for resolving ambiguity is published metadata. So `ambiguous`
  here means "the agent has to pick and they may pick a defensible
  non-gold reading"; there is no `user_sim_under_disclosure` failure
  class for these.
- `failure_classification.primary = user_sim_under_disclosure` is
  **never** used for livesqlbench. `n_asks` is always 0.
- `evaluator_prompt` is the only safety net for `insufficient`
  livesqlbench tasks — populate it generously.

## `failure_classification.primary` enum + when to pick each

Available values in the schema (auto-classified from the cascade for the
mechanical buckets; the others require human review):

| Value | When to use |
| -- | -- |
| `no_fail` | Cascade N3 strict-pass — agent matched an audited variant. Auto-assigned. |
| `agent_miss` | Metadata was `sufficient`; agent didn't derive it. Or metadata was `ambiguous` and the sim disclosed correctly, but the agent still picked wrong. |
| `metadata_ambiguity` | Metadata was `ambiguous` or `insufficient`; agent picked a defensible non-gold reading. Use when no user_sim was available (one-shot) or sim disclosed honestly and there's still genuine ambiguity. |
| `gold_audit_quality` | Used as `secondary` on a `no_fail` to record "agent passed but the original gold was buggy and the audit fixed it". Should be rare as `primary`: only when audit itself is incomplete after you've already considered fixing it (see audit-correctness gate above). |
| `user_sim_under_disclosure` | a-interact run; sim withheld snippet content the agent needed to pin the answer. |
| `novel_reading_accepted` | Cascade N5 fired — `metadata_sufficiency == "insufficient"` AND the LLM judge accepted the agent's reading as a valid novel interpretation. Auto-assigned. |
| `numerical_precision` | Cascade N6 fired (or sub-epsilon float noise the deterministic cascade misses). Auto-assigned for N6; can be picked manually for sub-epsilon cases. |
| `row_order` | Cascade N4 fired — agent's rowset matches modulo ORDER BY ties. Auto-assigned. |
| `trailing_whitespace` | Cascade N7 fired. Auto-assigned. |
| `column_order` | Cascade N8 fired. Auto-assigned. |
| `case_sensitivity` | Reserved for future tier; not auto-assigned yet. |
| `other` | Strict miss across all cascade tiers that doesn't fit any of the above. Requires human reasoning in `details`. |

Set `agent_at_fault = true` only for `agent_miss` (and rarely `other`).
For everything else, the agent was either correct or constrained by an
upstream issue (metadata, audit, sim, grader).

`secondary` carries co-existing flags — e.g. a `no_fail` with
`secondary=["gold_audit_quality"]` records "agent passed; original gold
was buggy; audit fixed it".

`remediation_target` enum: `agent`, `prompt`, `kb`, `audit`, `grader`,
`user_sim`, `gold_sidecar`, `other`. Pick the surface that needs the
change.

## Cascade tiers (used by the inline grader + summary)

The cascade is computed by `tolerant_grader.grade_submission` and lives
in the submission annotation's `evaluation` block. Most-stringent →
most-lenient:

| Tier | Field | Failure-class on a "this-tier-flipped-the-verdict" event |
| -- | -- | -- |
| N1 | `phase1_against_original_gold` | n/a (records strict pass against original `sol_sql`) |
| N2 | `phase1_against_audited_primary` | n/a (strict pass against `primary=true` variant) |
| N3 | `phase1_against_any_audited_variant` | `no_fail` |
| N4 | `correct_up_to_tie_order` | `row_order` |
| N5 | `novel_reading_judgment` | `novel_reading_accepted` (fires only when verdict=`insufficient`) |
| N6 | `correct_under_numeric_epsilon` | `numerical_precision` |
| N7 | `correct_under_trailing_whitespace` | `trailing_whitespace` |
| N8 | `correct_under_column_order` | `column_order` |

When the cascade collapses (no audit variants in the gold sidecar),
N2 == N3 == N1.

## U / C vocabulary

A task is **U** (audit-unchanged) when its
`audited_gold/<bench>_audited.jsonl` row has
`audited_sol_sql == original_sol_sql` modulo whitespace, else **C**
(audit-changed). At eval-time the four headline buckets are:

- **U-pass at N1** = "gold was right, agent right"
- **C-cosmetic-pass at N1** = "audit was cosmetic, agent matched both"
- **C-fix-pass at N2/N3** = "audit fixed buggy gold, agent matched fix"
- **U-fail / C-fail** = strict misses; cascade tier and failure-class
  pinpoint why

## Workflow

### Phase 1 — task annotation (always run)

1. Generate skeletons via `scripts/dev1515_convert_runs.py`
   (mini-interact, also touches submissions) or
   `scripts/dev1515_convert_livesqlbench.py --db <db>` (livesqlbench,
   task-only when `--run-id` is omitted). The skeleton pre-fills
   mechanical fields; human-judgment fields land at
   `PENDING_HUMAN_REVIEW`.
2. For each task: check `audit_unchanged` first. If unchanged AND
   published metadata is unambiguous, the task is `sufficient`.
3. For tasks where the audit changed gold (C-instances), START with the
   audit-correctness gate. If the audit is buggy, fix it in
   `audited_gold/<benchmark>_audited.jsonl` before writing the
   annotation. Then walk the decision flow.
4. Read `external_knowledge` + the relevant KB entries +
   `column_meaning_base.json` + the masked snippet inside
   `amb_user_query` before assigning a verdict.
5. Validate every edit with `TaskAnnotation.model_validate(...)`.

Stop here when no run exists. Phase 2 is a strict superset, not a
prerequisite — downstream tooling reads the `.task.json` files
independently of any `.submission.<run-id>.json`.

### Phase 2 — submission annotation (only with a run)

6. Re-run the skeleton generator with the run-id wired in (e.g.
   `dev1515_convert_runs.py` against the desired run, or
   `dev1515_convert_livesqlbench.py --db <db> --run-id <run-id>`). The
   skeleton auto-picks `no_fail` / cascade-tier `failure_class` for
   mechanical passes; only genuine `other` strict misses get
   `PENDING_HUMAN_REVIEW`.
7. If unchanged AND agent failed, the failure is `agent_miss` 90% of
   the time (verify by reading submitted_sql vs gold).
8. For ambiguous + a-interact: scan the trajectory for `ask_user` calls
   and the sim's replies. If the sim withheld snippet content the agent
   needed, the submission's `failure_classification.primary` is
   `user_sim_under_disclosure`.
9. Validate every edit with `SubmissionAnnotation.model_validate(...)`.

## Annotation rationale — write generously

The task annotation's `metadata_sufficiency.rationale` and
`gold_variants[].interpretation` + `notes` are the canonical
documentation of why the audited gold reads the way it does. Future
readers — humans, the auditor revisiting after a KB update, downstream
analysis subagents — must be able to reconstruct the reasoning without
re-doing the audit work. Cite:

- The specific KB entries (`<db>_kb.jsonl#N`) that anchor the audited
  reading.
- The column-meaning entries (`<db>.<table>.<column>`) that pin
  literals / sampled values.
- The masked-snippet structure that confirms predicate shape.
- The audit's `changes[]` entries and `reasoning_summary`.

For ambiguous and insufficient verdicts, also fill `evaluator_prompt`
with a self-contained LLM-judge prompt (~150–300 words) describing
what counts as a COMPATIBLE-if / INCOMPATIBLE-if reading. That prompt
fires only when verdict=`insufficient` AND no strict variant matches,
but you populate it speculatively for `ambiguous` cases too — it's
cheap insurance against future schema changes.

## Re-grading without re-running the cloud

`scripts/dev1515_convert_runs.py` re-executes the grader locally
(`grade_submission` against `<benchmark>/<db>/<db>.sqlite`) using the
submitted SQL captured in `results/cloud/<run-id>/rows/<inst>/attempt-1.json`.
After a grader change OR an audit fix, re-run that script and then
`scripts/dev1515_remap_failure_classes.py` to keep the cascade-tier
auto-classifications in sync. The remap is idempotent.

## Outputs the aggregator wants

`scripts/dev1515_cascade_summary.py` aggregates N1..N8 across all
per-(instance, run) submission annotations and emits the U/C-split
phase1 cascade plus a `failure_classification.primary` tally. Anything
auto-classified into `no_fail` / cascade-tier buckets contributes
without further human work; only `other` strict misses gate the
summary's "needs human attention" residual.
