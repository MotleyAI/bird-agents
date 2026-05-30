---
name: audit-gold-sql-livesqlbench
description: Audit a LiveSQLBench (`livesqlbench-base-lite-sqlite`) task's gold SQL (`sol_sql` from the gated gold sidecar) against the dataset's own justification sources (`<db>_kb.jsonl` + `<db>_column_meaning_base.json` + `<db>_schema.txt` + the task's `external_knowledge` anchor list). For each clause that is not derivable from those sources, propose a minimal rewrite — as close to the original as possible — that IS derivable, and write the audit (original + audited SQL + per-clause changes + citations) to `bird-interact-agents/audited_gold/livesqlbench_audited.jsonl` (single consolidated file across all DBs). The auditor is Claude executing this recipe; there is no script.
---

# Audit a LiveSQLBench gold SQL against authorised sources

This skill is the LiveSQLBench-shaped sibling of `audit-gold-sql` (the
mini-interact version). Both share `_shared/audit-gold-sql.contract.md`
for the sidecar JSON schema, citation tokens, audit-status semantics,
per-clause classification recipe, and anti-patterns. **Read that file
FIRST.** This file only describes what's different for LiveSQLBench.

## What's different vs mini-interact

| Aspect | mini-interact (`audit-gold-sql`) | livesqlbench (this skill) |
| --- | --- | --- |
| Per-task fields | `amb_user_query`, `sol_sql`, `user_query_ambiguity`, `knowledge_ambiguity`, `external_knowledge`, `category`, … all inline in `mini_interact.jsonl` | Public file (`livesqlbench_data_sqlite.jsonl`) carries `query`, `category`, `conditions`, `difficulty_tier`; gated sidecar (`livesqlbench_sqlite_gt_kg_testcases_0528.jsonl`) carries `sol_sql`, `test_cases`, `external_knowledge` |
| Labeled-ambiguity sources | Yes (`critical_ambiguity`, `non_critical_ambiguity`, `knowledge_ambiguity`) | **No** — LiveSQLBench has no labeled-ambiguity blocks. Citations CANNOT use `labeled_ambiguity:` / `knowledge_ambiguity:` |
| KB anchor signal | None — KB is a flat reference set | The task's `external_knowledge` is a CURATED LIST of KB ids anchoring the task. A KB id in that list gets the stronger `external_knowledge:<id>` citation (always paired with `kb:<id>`) |
| Schema FK info | Implicit in column meanings only | EXPLICIT in `<db>_schema.txt` — the auditor MUST consult schema FKs when the KB underspecifies a join host (see DEV-1510 museum_9 worked example below) |
| Audit sidecar layout | `audited_gold/<db>/<db>_audited.jsonl` (one file per DB) | `audited_gold/livesqlbench_audited.jsonl` (ONE file, all DBs, `selected_database` field discriminates) |
| Skill version stamp | `audit-gold-sql/1.0` | `audit-gold-sql-livesqlbench/1.0` |
| `benchmark` field on rows | omitted (mini-interact is implicit) | required: `"livesqlbench"` |

## Inputs (per task)

Read-only:

- `<lsb_root>/livesqlbench_data_sqlite.jsonl` — public task data.
  Filter by `instance_id` → get `query` (the user-facing question),
  `selected_database`, `category`, `high_level`, `conditions`,
  `difficulty_tier`. `sol_sql` / `external_knowledge` / `test_cases` are
  EMPTY here — they live in the sidecar.
- `<lsb_root>/livesqlbench_sqlite_gt_kg_testcases_0528.jsonl` — gated
  gold sidecar. Filter by `instance_id` → get the canonical `sol_sql`,
  `external_knowledge` (anchor list of KB ids), `test_cases`. **Use the
  `_sqlite_gt_` file**, NOT the `_gt_` file (Postgres-only; see the
  `reference_livesqlbench_sqlite_gold` memory).
- `<lsb_root>/<db>/<db>_kb.jsonl` — KB entries (id, knowledge,
  description, definition, type, children_knowledge).
- `<lsb_root>/<db>/<db>_column_meaning_base.json` — column meanings.
  Keys are `<db>|<table>|<column>` (case-sensitive); JSONB columns
  have nested `fields_meaning`.
- `<lsb_root>/<db>/<db>_schema.txt` — full CREATE TABLE statements
  INCLUDING declared FOREIGN KEYs. The KB rarely names host tables for
  derived metrics; the schema's FK declarations are the canonical
  disambiguator.
- `<lsb_root>/<db>/<db>.sqlite` (or `<db>_template.sqlite`) — live DB,
  read-only. Used for the sanity-execute step.

Where `<lsb_root>` resolves to `paths.livesqlbench_root()`. For the
test/dev environment that's
`/home/james/Dropbox/SLayer/livesqlbench-base-lite-sqlite/`.

## Output

ONE consolidated JSONL at
`bird-interact-agents/audited_gold/livesqlbench_audited.jsonl`. Each
line is one task's full audit. `selected_database` discriminates per
DB; `instance_id` is the dedup key (latest-wins on append).

When auditing an instance whose row already exists, **rewrite the whole
file** with that row replaced (don't dup-append; the cloud overlay
reads latest-wins but the contract tests pin "no duplicates").

## Citation rules (livesqlbench-specific)

See the shared contract for the full citation-token table. Specific to
livesqlbench:

- **Do NOT use** `labeled_ambiguity:` or `knowledge_ambiguity:` — no
  source for them in livesqlbench. Resolvability tests will fail.
- **DO use** `external_knowledge:<id>` whenever the cited KB id is also
  in the task's `external_knowledge` anchor list. ALWAYS paired with
  `kb:<id>`. Example: `["kb:16", "external_knowledge:16"]`.
- **Schema FKs are first-class evidence.** When a KB defines a derived
  metric (e.g. KB 8 `LER = LightLux × LightSensWeight × VisibleExpLxh`)
  but does NOT pin the host table that links the columns to a per-
  artifact context, the schema's declared FOREIGN KEY is the
  disambiguator. Cite via `column_meaning:<HostTable>|<FKColumn>` — the
  column-meaning text typically restates the FK's semantic purpose.

## Procedure (per task)

Follow the 6-step recipe in `_shared/audit-gold-sql.contract.md`,
substituting livesqlbench's inputs / citations / output path:

1. **Load**: read the task row from BOTH the public data file AND the
   gated gold sidecar. Read `<db>_kb.jsonl`, `<db>_column_meaning_base.json`,
   `<db>_schema.txt`. Print the user `query` and the canonical `sol_sql`
   side by side. Write the natural reading of the question.
2. **Cite each clause**: walk the decomposition (CTE → joins → WHERE → …)
   and cite every column, operator, literal, threshold, value-set, and
   derived expression. For each KB id you cite, check whether it's in the
   task's `external_knowledge` and append the paired
   `external_knowledge:<id>` token if so.
3. **Inspect the schema for FK clues** whenever a clause's host table or
   join chain isn't obviously pinned by KB+column-meaning text. The
   `<db>_schema.txt` is small (one CREATE TABLE per table) — grep the
   relevant table names + `FOREIGN KEY` lines.
4. **Classify** per shared contract (clean / edited / unrecoverable).
5. **Sanity-execute** `audited_sol_sql[0]` against `<db>.sqlite` (or
   `<db>_template.sqlite`). Record first row in `audited_sample_row`.
6. **Persist**: append/overwrite the row in
   `audited_gold/livesqlbench_audited.jsonl`. Each row carries
   `benchmark: "livesqlbench"` and `skill_version: "audit-gold-sql-livesqlbench/1.0"`.

## DEV-1510 worked examples

These are the two locked decisions for the museum audit (issue
DEV-1510). The recipe was first validated against them.

### museum_7 — `audit_status="edited"` (NULL-safe ≥3-of-flags rewrite)

KB 16 (Showcase Failure Risk):

> "Occurs when SESR < 4 OR at least three of \[4 condition flags\]
> (sealcondition='Poor', maintstatus='Overdue', filterstatus='Replace Now',
> silicagelstatus='Replace Now')."

Gold treats each flag as INDEPENDENTLY sufficient (`OR sealcondition='Poor' OR maintstatus='Overdue' OR ...`).
Direct contradiction of KB 16. Task `external_knowledge=[16]` confirms
KB 16 is the anchored authority.

Audited rewrite uses a NULL-safe `CASE WHEN ... THEN 1 ELSE 0 END` sum:

```sql
WHERE (10 - ((er.tempvar24h + er.humvar24h / 5.0 + s.leakrate) / 3.0)) < 4
   OR (CASE WHEN s.sealcondition   = 'Poor'        THEN 1 ELSE 0 END
     + CASE WHEN s.maintstatus     = 'Overdue'     THEN 1 ELSE 0 END
     + CASE WHEN s.filterstatus    = 'Replace Now' THEN 1 ELSE 0 END
     + CASE WHEN s.silicagelstatus = 'Replace Now' THEN 1 ELSE 0 END) >= 3
```

**NULL-safety note:** SQLite boolean arithmetic `(col='lit')+(col='lit')+...`
returns NULL whenever ANY flag column is NULL (since `NULL + 1 = NULL`).
That would mask three positive flags whenever one is null. The CASE form
returns 0 for null columns (3-valued logic: `col='lit'` on NULL is NULL;
neither WHEN nor ELSE? Actually ELSE fires on non-match including NULL,
so CASE returns 0). NULL-safe.

Citations: `["kb:16", "external_knowledge:16"]`.

### museum_9 — `audit_status="clean"` (column-meaning resolves KB underspec)

KB 8 (LER): `LER = LightLux × LightSensWeight × VisibleExpLxh / 1000`.
Names the columns but NOT the host table linking light readings to
per-artifact context.

Two structurally-valid join chains:

1. `ConditionAssessments → LightAndRadiationReadings` via
   `ConditionAssessments.LightReadRefObserved → LightAndRadiationReadings.LightRadRegistry`
   (single-hop declared FK).
2. `UsageRecords → Showcases → EnvironmentalReadingsCore → LightAndRadiationReadings`
   (3 hops, one-to-many at every step).

Gold uses #1. Agent (in the motivating cloud run) used #2 because KB
alone didn't disambiguate. **Audit decision: `clean`**, because
`column_meaning:ConditionAssessments|LightReadRefObserved` says
"Associates the assessment with relevant light data" — that's the
single-hop declared FK that resolves the KB-alone underspec. The agent's
reading is KB-faithful but column-meaning-blind.

`audited_sol_sql == original_sol_sql`. `reasoning_summary` MUST:

- Name BOTH chains explicitly (ConditionAssessments + UsageRecords).
- Name an endpoint of the agent's 3-hop chain
  (Showcases / EnvironmentalReadingsCore / ShowcaseRef).
- Cite `column_meaning:ConditionAssessments|LightReadRefObserved`
  (exact token form, not just substring `LightReadRefObserved`).
- Mention `kb:8` as the under-specifying KB.

This is the worked example for the follow-up Linear issue
(claude_sdk_otf should consult column meanings + schema FKs when KB
underspecifies the join graph).

## Anti-patterns specific to livesqlbench

The shared contract lists the cross-benchmark anti-patterns. In
addition:

- **KB underspec via missing host table**: a KB defines a derived
  metric whose formula names columns but no host. The schema's
  declared FKs almost always disambiguate; never declare a clause
  `unrecoverable` without first checking
  `<db>_schema.txt` for `FOREIGN KEY ... REFERENCES ...` rows on the
  candidate join columns.
- **Gold uses a 3-hop indirect join when a 1-hop declared FK exists**:
  this is a smell that the gold's author missed the FK. Usually the
  audit picks the 1-hop FK as canonical (with the supporting
  column-meaning citation) and marks `clean`; the agent's
  alternative-chain reading then becomes a follow-up agent-thoroughness
  signal, not a gold-side rewrite.
- **Postgres-only SQL in the SQLite gold**: the `_sqlite_gt_` file is
  cleaned, but a stray `:: cast` or `NULLS FIRST` etc. can slip
  through. Mark with `dialect:sqlite:<feature>` and rewrite to the
  SQLite equivalent.

## After auditing all instances in a DB

1. Run the contract tests: `pytest tests/test_livesqlbench_audited_gold.py`.
   These pin schema + coverage + status-claim consistency + citation
   resolvability (skip-if-livesqlbench-data-absent) + the museum_7 /
   museum_9 decisions.
2. Run the integration tests:
   `pytest -m integration tests/integration/test_livesqlbench_audited_gold_execute.py`
   to validate every `audited_sol_sql` actually runs against the live
   `<db>.sqlite`.
3. (Cloud) The `audited_gold/livesqlbench_audited.jsonl` ships via the
   image's BuildKit `audited-gold=` context. A change to the file
   changes `image.data_hash`, forcing a clean image rebuild on the next
   submit so cloud actors see the audit.
