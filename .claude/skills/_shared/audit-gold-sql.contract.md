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
