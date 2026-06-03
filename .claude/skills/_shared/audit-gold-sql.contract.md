# Audit-gold-sql contract (shared across benchmarks)

This file is the SINGLE SOURCE OF TRUTH for the sidecar JSON schema,
audit-status semantics, citation tokens, and the per-clause classification
recipe. Both `audit-gold-sql` (mini-interact) and
`audit-gold-sql-livesqlbench` link to it; per-benchmark differences live
in the SKILL.md of each, not here.

## Sidecar JSON schema (per audited task)

```json
{
  "instance_id": "<task instance_id>",
  "selected_database": "<task db>",
  "benchmark": "<mini_interact | livesqlbench>",
  "audit_status": "<clean | edited | unrecoverable>",
  "variant_id": "<short kebab-case slug, e.g. 'primary', 'labeled_snippet', 'kb_strict'>",
  "primary": true,
  "original_sol_sql": ["<canonical gold SQL string(s)>"],
  "audited_sol_sql":  ["<audited SQL string(s)>"],
  "audited_sample_row": ["<first row of audited_sol_sql[0] against the db, or []>"],
  "changes": [
    {
      "clause_kind": "outer_select | where_predicate | join | cte | …",
      "original": "<exact substring being replaced>",
      "replacement": "<exact replacement, or '(unchanged — kept verbatim)'>",
      "why_unjustified": "<one sentence>",
      "justified_by": ["<citation tokens, see below>"]
    }
  ],
  "reasoning_summary": "<freeform paragraph; for clean rows MUST be substantive: ≥200 chars and cite ≥1 source>",
  "skill_version": "<audit-gold-sql/1.0 | audit-gold-sql-livesqlbench/1.0>",
  "audited_at": "<ISO-8601 UTC>"
}
```

`variant_id` and `primary` are always required:

- `variant_id`: a short kebab-case slug identifying this reading. For single-variant tasks use
  `"primary"`. For multi-variant rows name each variant after its authoritative source
  (e.g. `"labeled_snippet"`, `"kb_strict"`).
- `primary`: boolean, exactly ONE row per `instance_id` must carry `primary: true`. For
  single-variant tasks the single row is always `primary: true`.

### Field rules

- `audit_status` ∈ {`clean`, `edited`, `unrecoverable`}:
  - `clean` → `audited_sol_sql == original_sol_sql` AND `changes == []`. The
    `reasoning_summary` MUST still walk every clause + cite the supporting
    source — that's the only audit-trail artifact for clean rows.
  - `edited` → gold has unjustified parts that CAN be rewritten using
    authorised sources to preserve the user's intent. `audited_sol_sql`
    differs; `changes` non-empty.
  - `unrecoverable` → gold's intent itself is unauthorised. Fall back to
    the natural reading of the user query; one `changes` entry documents
    the gap.
- `original_sol_sql` / `audited_sol_sql` are both `list[str]`, mirroring
  the upstream `sol_sql`'s shape. For most tasks both have length 1.
- `audited_sample_row` is the first row of running `audited_sol_sql[0]`
  against the db; `[]` if the query returns no rows.

## Citation tokens (`changes[].justified_by[]`)

| Token | Meaning | Used by |
| --- | --- | --- |
| `kb:<id>` | KB entry whose `definition`/`description` entails the clause | both benchmarks |
| `column_meaning:<Table>\|<Column>` (or `\|<SubField>` for JSONB) | Column-meaning text authorises the column reference + categorical legal value set | both benchmarks |
| `primitive` | Standard SQL primitive (SELECT, JOIN, +, COUNT, CASE WHEN, …). Never the sole justification for a literal/threshold | both benchmarks |
| `dialect:<engine>:<feature>` | Mechanical SQL-dialect-correctness fix not traceable to KB/column-meaning (e.g. `dialect:sqlite:integer_division`) | both benchmarks |
| `labeled_ambiguity:<term>` | Exact `term` from a `critical_ambiguity` / `non_critical_ambiguity` entry — its `sql_snippet` operationalises a clause | **mini-interact only** |
| `knowledge_ambiguity:<term>` | Exact `term` from a `knowledge_ambiguity` entry — `sql_snippet` operationalises a clause | **mini-interact only** |
| `external_knowledge:<id>` | KB id that is also in the task's `external_knowledge` anchor list (strictly stronger than `kb:<id>` alone) | **livesqlbench only** |

## `clause_kind` (informal)

Use one of: `outer_select`, `projection`, `from`, `join`, `where_predicate`,
`group_by`, `having`, `order_by`, `limit`, `cte`, `subquery`, `case_branch`,
`literal_threshold`, `value_bucket`, `set_arithmetic`.

## Per-clause classification recipe

1. **Decompose** the gold SQL into clauses (projections, FROM/JOIN,
   WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, every CTE/subquery
   recursively).
2. **Cite** every column, operator, literal, threshold, value-set, and
   derived expression against one of the citation tokens above. A
   primitive (`COUNT(*)`, `JOIN`) shape needs no citation; the operands
   still do.
3. **Identify unjustified clauses** — anything no citation covers.
4. **Classify**:
   - All justified → `audit_status = clean`. Write a `reasoning_summary`
     that walks the clauses and lists the citation tokens.
   - Some unjustified, intent recoverable → `audit_status = edited`.
     Build the rewrite by minimal string-substitution, preserving every
     justified clause. One `changes` entry per replaced clause.
   - Intent itself unauthorised → `audit_status = unrecoverable`.
     `audited_sol_sql` becomes the natural reading of the user query;
     one `changes` entry documents the gap.
5. **Sanity-execute** `audited_sol_sql[0]` against the live sqlite and
   record the first row in `audited_sample_row`. Fix any SQL errors
   before persisting.
6. **Persist** the JSONL line per the benchmark's on-disk layout (see
   each SKILL.md).

## Multi-variant audit on source contradiction (MANDATORY)

If two of the task's authoritative sources DIRECTLY contradict on the
SAME parameter, the audit MUST emit **multi-variant** — one row per
defensible reading. Collapsing to a single variant with a note is
**forbidden**. The agent under test cannot satisfy mutually-exclusive
sources; forcing one reading tags genuine ambiguity as `agent_miss`.

What counts as a contradiction:

| Same parameter | Source A | Source B | Verdict |
| --- | --- | --- | --- |
| Boundary operator on a threshold | KB definition: "greater than 7" | labeled sql_snippet: `score >= 7` | **contradiction** (strict vs inclusive) |
| Numeric threshold value | KB: "> 3" | labeled sql_snippet: `> 2` | **contradiction** |
| Bucket inclusion | column_meaning enum: `{A,B,C}` | KB cites bucket `D` | **contradiction** |
| Aggregation choice | KB: "average of …" | labeled sql_snippet: `SUM(…)` | **contradiction** |
| KB-internal | KB.description: one cutoff | KB.definition: another | **contradiction** |

Not a contradiction: one source silent, the other specifies; one source
operationalises in SQL what the other describes in prose; different
parameters that happen to live near each other in the metadata.

**The recipe addendum.** Before classifying the audit as single-variant
in step 4 of the per-clause recipe, scan ALL loaded authoritative
sources (KB items, column meanings, every labeled/anchor citation
shape your benchmark exposes — see the per-benchmark SKILL.md for the
full source list) for pairwise direct contradictions. ANY contradicting
pair triggers multi-variant. No "labeled wins" tie-breaker. No "KB-
anchored is primary" tie-breaker. If you find yourself writing a
regretful note in `reasoning_summary` that names two sources and
explains why you went with one, STOP — that's the multi-variant
trigger and you skipped it. Go back and emit two rows.

**Mechanics when emitting multi-variant.**

- Each row shares `instance_id`; rows are distinguished by `variant_id`
  (a short kebab-case slug naming the reading, e.g. `labeled_snippet`,
  `kb_strict`).
- Exactly one row carries `primary: true`. The choice is arbitrary —
  it only controls which row the grader's strict-N2 path targets by
  default; `n3_any_audited_variant` accepts either. Document the
  choice in `reasoning_summary` as bookkeeping, not as authority.
- Every row keeps `audit_status: "edited"`.
- Each row's `changes[]` carries AT LEAST one entry with
  `clause_kind: "source_conflict"` pointing at the other variant:
  - `original`: what THIS variant rejected (paraphrase the other
    variant's reading, in parens cite the source).
  - `replacement`: what THIS variant chose (paraphrase, in parens
    cite the source).
  - `why_unjustified`: quote the rejected source's contrary value.
  - `justified_by`: the citation tokens backing THIS variant's choice.
- The downstream TaskAnnotation carries
  `internal_inconsistency.audit_resolution = "multi_variant"` with
  `sources_in_conflict[]` quoting both sides verbatim, and one
  `gold_variants[]` entry per audit row.

**Minimal worked example (synthetic).** A task asks "list high-score
items"; KB X.definition says "score must be greater than 7" (strict);
the labeled sql_snippet for "high score" says `score >= 7` (inclusive).
The audit emits two rows:

```jsonl
{"instance_id": "demo_5", "variant_id": "labeled_snippet", "primary": true,
 "audit_status": "edited",
 "audited_sol_sql": ["… WHERE foo.score >= 7 …"],
 "changes": [{"clause_kind": "source_conflict",
   "original": "foo.score > 7  (KB X strict reading)",
   "replacement": "foo.score >= 7  (labeled-snippet inclusive reading)",
   "why_unjustified": "KB X.definition: 'score must be greater than 7' is strict; this variant rejects KB X in favor of the labeled sql_snippet's inclusive boundary.",
   "justified_by": ["labeled_ambiguity:high score"]}],
 "reasoning_summary": "Task is internally inconsistent on the score threshold. This variant follows the labeled sql_snippet."}
{"instance_id": "demo_5", "variant_id": "kb_strict", "primary": false,
 "audit_status": "edited",
 "audited_sol_sql": ["… WHERE foo.score > 7 …"],
 "changes": [{"clause_kind": "source_conflict",
   "original": "foo.score >= 7  (labeled-snippet inclusive reading)",
   "replacement": "foo.score > 7  (KB X strict reading)",
   "why_unjustified": "labeled_ambiguity:high score uses inclusive >=; this variant rejects the labeled snippet in favor of KB X.definition's explicit 'greater than' wording.",
   "justified_by": ["kb:foo_kb#X"]}],
 "reasoning_summary": "Task is internally inconsistent on the score threshold. This variant follows KB X's strict reading."}
```

(For LiveSQLBench tasks the citation token for the snippet-anchored
variant would be `external_knowledge:<id>` or `column_meaning:…`
instead of `labeled_ambiguity:` — LiveSQLBench has no labeled-ambiguity
blocks. See the per-benchmark SKILL.md for which sources can take part
in a contradiction.)

## Anti-patterns (call out in the audit)

- **Set arithmetic on counts** with no natural-language analogue in the
  user query.
- **Hidden bucketing** (CASE labels not in any source).
- **Hidden ranking** (ORDER BY a derived expression not in the question
  or any KB).
- **Implicit deduplication** (`SELECT DISTINCT` when the user didn't
  ask for unique values).
- **JOINs to tables that don't appear in any KB/column-meaning reference
  for the question's terms** — usually a sign the gold is pulling in
  unrelated columns.

## Edge cases (defer cleanly)

- **Management category tasks** — eval depends on side-effects /
  preprocess+cleanup. Out of scope for the row-count audit. Mark
  `audit_status=unrecoverable`, copy gold verbatim into `audited_sol_sql`,
  `changes[].clause_kind="management_category"`.
- **Tasks with non-empty `test_cases`** — eval uses custom Python
  comparators that may embed their own gold logic. Same posture as
  management.
- **Multi-statement `sol_sql`** (length > 1) — each statement needs its
  own audit; rare for SELECT tasks.
