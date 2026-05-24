---
name: audit-gold-sql
description: Audit a BIRD-Interact mini-interact task's gold SQL (`sol_sql`) against the dataset's own justification sources (`<db>_kb.jsonl` + `<db>_column_meaning_base.json` + the task's labeled ambiguities). For each clause that is not derivable from these sources, propose a minimal rewrite — as close to the original as possible — that IS derivable, and write the audit (original + audited SQL + per-clause changes + citations) to `bird-interact-agents/audited_gold/<db>/<db>_audited.jsonl`. The auditor is Claude executing this recipe; there is no script.
---

# Audit a BIRD-Interact gold SQL against authorised justification sources

## When to use

Run once per `(db, instance_id)` pair to produce or refresh an entry in
the audit sidecar. Skill is domain-agnostic — invoked with a DB name and
one (or many) instance_ids. The worked example throughout this file uses
a small **synthetic** `shop` database (instance `shop_3`); it mirrors the
shape of a real motivating case without reproducing any benchmark data.

The audited gold SQL is consumed at eval time when the bird-interact
runner is called with `--use-audited-gold-sql`; the harness overlay
swaps `sample_status.original_data["sol_sql"]` to the audited version
before `execute_submit_action` runs.

## Inputs (per task)

Read-only:

- `../mini-interact/mini_interact.jsonl` — one line per task. Filter by
  `instance_id` to get the record. The record carries every field the
  audit needs: `amb_user_query`, `sol_sql`, `category`, `test_cases`,
  `user_query_ambiguity`, `knowledge_ambiguity`, `external_knowledge`,
  `selected_database`.
- `../mini-interact/<db>/<db>_kb.jsonl` — KB entries (id, knowledge,
  description, definition, type, children_knowledge).
- `../mini-interact/<db>/<db>_column_meaning_base.json` — column
  meanings. Top-level keys are `<db>|<table>|<column>`; JSONB columns
  have nested `fields_meaning` blobs.
- `../mini-interact/<db>/<db>.sqlite` — live DB; only used for the
  sanity-execute step at the end.

## Outputs

One JSONL line appended (or updated) at
`bird-interact-agents/audited_gold/<db>/<db>_audited.jsonl`. If a line
already exists for this `instance_id`, **overwrite** it in place
(latest-wins; the verifier dedups on read).

### Sidecar schema (verbatim)

```json
{
  "instance_id": "shop_3",
  "selected_database": "shop",
  "audit_status": "edited",
  "original_sol_sql": ["WITH SupplierTier AS (..."],
  "audited_sol_sql":  ["SELECT COUNT(*) ..."],
  "audited_sample_row": [0],
  "changes": [
    {
      "clause_kind": "outer_select",
      "original":  "COUNT(CASE WHEN supplier_class='Premium' THEN 1 END) - COUNT(CASE WHEN supplier_class='Standard' THEN 1 END)",
      "replacement": "COUNT(*)",
      "why_unjustified": "Subtraction of Premium vs Standard counts is not in KB / column-meanings / labeled-ambiguity. The user asked for 'a count of heavy orders from premium suppliers', not for a difference.",
      "justified_by": ["primitive"]
    }
  ],
  "reasoning_summary": "Removed the unauthorised premium-minus-standard subtraction; kept the heavy-and-premium predicate as is, since KB 22 + KB 25 + the labeled ambiguities for 'heavy orders' and 'premium suppliers' authorise it directly.",
  "skill_version": "audit-gold-sql/1.0",
  "audited_at": "<ISO-8601 timestamp at time of audit>"
}
```

Field rules:

- `audit_status` is one of three:
  - `"clean"` — gold passes audit unchanged. `audited_sol_sql ==
    original_sol_sql`; `changes == []`.
  - `"edited"` — gold has unjustified parts that CAN be rewritten using
    authorised sources to preserve the user's intent. `audited_sol_sql`
    differs; `changes` is non-empty.
  - `"unrecoverable"` — gold has unjustified parts that cannot be
    rewritten to preserve the gold's intent (the intent itself is
    unauthorised). Fall back to the natural reading of `amb_user_query`
    using only authorised sources. `audited_sol_sql` is the natural
    reading; one `changes` entry documents the gap.
- `original_sol_sql` and `audited_sol_sql` are both list[str], mirroring
  upstream `sol_sql`'s shape. For most tasks both have length 1.
- `audited_sample_row` is the first row of running `audited_sol_sql[0]`
  against `<db>.sqlite`. If the query returns no rows, store `[]`.
- Every entry in `changes[].justified_by` is one of:
  - `"kb:<id>"` (numeric id from `<db>_kb.jsonl`)
  - `"column_meaning:<table>|<column>"` or `"column_meaning:<table>|<column>|<sub_field>"`
  - `"labeled_ambiguity:<term>"` (the exact `term` field from the
    matching `critical_ambiguity` entry)
  - `"knowledge_ambiguity:<term>"` (the exact `term` field from the
    matching `knowledge_ambiguity` entry — distinct citation prefix
    from `labeled_ambiguity:` so the verifier can resolve the term
    against the right source list)
  - `"primitive"` (standard SQL primitives — never the only justification
    for a value or threshold, but fine for shape operators like
    `COUNT(*)` or `JOIN`)
  - `"dialect:<engine>:<feature>"` (mechanical SQL-dialect correctness
    fix that isn't traceable to a KB / column-meaning source — the
    dialect itself is the source). Use for post-hoc bug fixes to
    audited SQL: e.g. `dialect:sqlite:integer_division`,
    `dialect:sqlite:no_regex_replace`. Free-form `<feature>` —
    every dialect bug has its own shape.
- `clause_kind` is informal but should be one of: `outer_select`,
  `projection`, `from`, `join`, `where_predicate`, `group_by`,
  `having`, `order_by`, `limit`, `cte`, `subquery`, `case_branch`,
  `literal_threshold`, `value_bucket`, `set_arithmetic`.

## The justification model

Every column, operator, literal, threshold, value-set, and derived
expression in the gold must cite one of:

1. **`kb:<id>`** — a KB entry whose `definition` or `description`
   entails the clause. Example: `KB 11` ("Order Density =
   units / pack size") authorises the `unit_count / Pack_Size`
   division. `KB 22` ("Premium Supplier = top-tier rating + fast
   ship class") authorises the premium-defining predicate shape.
2. **`column_meaning:<table>|<column>[|<sub_field>]`** — the column
   meaning JSON authorises every column reference and the legal
   value-set of categorical columns. JSON sub-fields are reached via
   `fields_meaning` (e.g. `pkg_specs|Pack_Size`).
3. **`labeled_ambiguity:<term>`** — the dataset's own resolution of an
   ambiguous term in this question. The `sql_snippet` field of a
   matching entry in `user_query_ambiguity.critical_ambiguity[]` or
   `knowledge_ambiguity[]` operationalises the gold's literal /
   bucket / predicate. The canonical user-sim
   (`../BIRD-Interact/usersim-guard/user_simulator/prompts.py:38-71`)
   is bound to answer within these snippets when the agent calls
   `ask_user`; that's the recoverability channel.
4. **`primitive`** — standard SQL primitives (SELECT, JOIN types, WHERE,
   GROUP BY, ORDER BY, LIMIT, +, -, *, /, COUNT, SUM, AVG, MIN, MAX,
   CAST, NULLIF, COALESCE, CASE WHEN, JSON_EXTRACT, LOWER, TRIM, IN,
   LIKE, IS NULL, comparison operators). These never need citation —
   they're SQL itself. BUT: the *operands* of a primitive must still
   cite one of the other three classes. `COUNT(*)` is fine; `COUNT(...)
   - COUNT(...)` is fine *as a shape*, but the *meaning* (subtracting
   two specific buckets) needs class-1/2/3 support.

A clause is **unjustified** when nothing in classes 1–4 covers it. Examples:

- Outer subtraction `COUNT(Premium) - COUNT(Standard)` with no KB /
  column-meaning / labeled-ambiguity citing "the answer is a
  difference of two bucket counts". *Structural* logic invented by the
  gold author.
- An arbitrary numeric threshold `density > 3` when no KB and no
  labeled ambiguity says the threshold is `3`. Note: if the labeled
  ambiguity's `sql_snippet` DOES contain `> 3`, the literal IS
  justified by class 3.
- A `CASE WHEN ... THEN 'Premium' ELSE 'Standard'` bucketing where the
  bucket labels appear nowhere in KB / column-meaning / ambiguity.

### What about the user-sim's `unlabeled(segment)` action?

The canonical user-sim has three actions: `labeled(term)`,
`unlabeled(segment)`, and `unanswerable()`. In principle the user-sim
COULD reveal a gold-SQL clause that's not in any labeled ambiguity, by
choosing `unlabeled(segment)`. **The audit does not treat this as a
justification source**, because:

- the `unlabeled` choice is a non-deterministic LLM judgment baked into
  the user-sim's reasoning — not a deterministic property of the
  dataset.
- in practice the gold-SQL parts that fall outside labeled ambiguities
  are *structural* (subtractions, custom bucketing, hidden ordering),
  not literal-extractable. Asking the user-sim about a structural
  invention triggers `unanswerable()` because the structure isn't a
  term-clarification but an answer-shape question. We don't want the
  audit's notion of "what's recoverable" to depend on
  reverse-engineering the user-sim's LLM heuristics.

So: every gold-SQL literal that matters for the answer should appear
in a labeled-ambiguity `sql_snippet`. If it doesn't, that's the bug
the audit is here to find.

## Procedure (per task)

### Step 1 — Load the inputs

```bash
# Inside the agent's working dir (bird-interact-agents/):
DB="shop"
INST="shop_3"

# Read the task line
uv run python -c "
import json
with open('../mini-interact/mini_interact.jsonl') as f:
    for line in f:
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get('instance_id') == '${INST}':
            print(json.dumps(d, indent=2))
            break
" > /tmp/claude/task.json

# Read the KB and column meanings
cp ../mini-interact/${DB}/${DB}_kb.jsonl /tmp/claude/kb.jsonl
cp ../mini-interact/${DB}/${DB}_column_meaning_base.json /tmp/claude/cm.json
```

Or use the `Read` tool on the same files; either is fine.

### Step 2 — Read the user query and the gold SQL

Print these two side-by-side. Write down the *natural reading* of the
question — what would a domain analyst compute, ignoring the gold? The
natural reading is the fallback for `unrecoverable` cases.

For `shop_3`:
- User query: *"Could you give me a count of all the heavy orders from
  our premium suppliers?"*
- Natural reading: a single integer = count of orders satisfying
  *heavy AND from-a-premium-supplier*.

### Step 3 — Inventory the labeled ambiguities

Read every entry in:

- `task.user_query_ambiguity.critical_ambiguity[]`
- `task.user_query_ambiguity.non_critical_ambiguity[]`
- `task.knowledge_ambiguity[]`

For each, note the `term` and the `sql_snippet`. These are the
dataset's sanctioned literal/bucket/predicate definitions for this
question.

For `shop_3`:

| Source | Term | Snippet covers |
|---|---|---|
| critical_ambiguity[0] | "heavy orders" | `(unit_count / NULLIF(...$.Pack_Size..., 0)) > 2` — defines order-density predicate INCLUDING the `> 2` threshold |
| critical_ambiguity[1] | "premium suppliers" | `LOWER(tier) IN ('gold','platinum') AND LOWER(ship_class) IN ('express','priority')` — defines the premium predicate |
| knowledge_ambiguity[0] | "Premium Supplier" (deleted_knowledge=22) | Same `LOWER+IN` premium predicate |

### Step 4 — Decompose the gold SQL

Parse the gold SQL into clauses manually. List every:

- SELECT projection (column refs, aggregates, derived expressions).
- FROM / JOIN clauses (tables touched, join keys).
- WHERE predicates.
- GROUP BY columns.
- HAVING predicates.
- ORDER BY clauses.
- LIMIT / OFFSET.
- CTEs and subqueries — recurse into each.

For `shop_3`:

```sql
WITH SupplierTier AS (
  SELECT CASE WHEN <premium-predicate> THEN 'Premium' ELSE 'Standard' END AS supplier_class
  FROM orders o
  JOIN packages p ON o.order_no = p.order_link
  JOIN suppliers s ON p.supplier_link = s.supplier_ref
  WHERE <Pack_Size > 0>
    AND <density > 2>
)
SELECT COUNT(CASE WHEN supplier_class='Premium' THEN 1 END)
     - COUNT(CASE WHEN supplier_class='Standard' THEN 1 END)
FROM SupplierTier;
```

Clauses:
- `J1` join orders/packages on order_no/order_link — primitive
  (column_meaning resolves both columns).
- `J2` join packages/suppliers on supplier_link/supplier_ref — primitive.
- `W1` WHERE Pack_Size > 0 — guards against div/zero; column_meaning
  authorises `Pack_Size`, primitive authorises the `> 0` guard (it's
  a NULLIF-equivalent).
- `W2` WHERE density > 2 — `labeled_ambiguity:heavy orders` provides
  the FULL predicate including the threshold.
- `C` CASE WHEN premium-predicate THEN 'Premium' ELSE 'Standard' — `kb:22`
  authorises the premium-predicate; `labeled_ambiguity:premium suppliers`
  also covers it. Bucketing the rows as 'Premium' vs 'Standard' is
  primitive (a CASE) and the predicate itself is justified, but the
  *purpose* of bucketing (so we can subtract counts) is in service of
  the outer SELECT.
- `O` outer SELECT `COUNT(Premium) - COUNT(Standard)` — **this is the
  unjustified clause**. The subtraction shape is not in KB, column-
  meanings, or any labeled-ambiguity sql_snippet. The user asked for a
  count, not a difference.

### Step 5 — Classify each clause

For each clause from Step 4, write `<justified>` or `<unjustified>`
with the citation. Build a per-clause table. If ALL clauses are
justified, set `audit_status = "clean"` and skip to Step 7 with no
changes.

### Step 6 — Propose the minimal rewrite

For each unjustified clause:

1. Identify the user-query intent the clause is trying to express. If
   the intent itself is unauthorised (i.e. the user didn't ask for
   what the gold computes), this is `unrecoverable` — the audited SQL
   becomes the natural reading of the user query.
2. If the intent is recoverable (the user asked for X but the gold
   computes X with an unjustified twist), strip the unjustified twist
   and keep the rest.
3. Build the audited SQL by string-substitution on the original.
   Preserve every justified clause verbatim.
4. Each replacement gets one entry in `changes[]` with:
   - `clause_kind` — see allowed values above
   - `original` — exact substring being replaced
   - `replacement` — exact replacement (or `"(unchanged — kept
     verbatim)"` if you want to record a "we considered this and kept
     it" decision)
   - `why_unjustified` — one sentence
   - `justified_by` — citations supporting the replacement

For `shop_3`:

- Strip the outer `COUNT(Premium) - COUNT(Standard)` and replace with
  `COUNT(*)`.
- Since the bucketing CASE was only there to feed the subtraction,
  drop the CASE-and-CTE and inline the WHERE predicate directly.
- Audited SQL:

```sql
SELECT COUNT(*)
FROM orders o
JOIN packages p ON o.order_no = p.order_link
JOIN suppliers s ON p.supplier_link = s.supplier_ref
WHERE CAST(json_extract(p.pkg_specs, '$.Pack_Size') AS REAL) > 0
  AND (o.unit_count / CAST(json_extract(p.pkg_specs, '$.Pack_Size') AS REAL)) > 2
  AND LOWER(s.tier) IN ('gold', 'platinum')
  AND LOWER(s.ship_class) IN ('express', 'priority');
```

`audit_status = "edited"`.

### Step 7 — Sanity-execute the audited SQL

Run `audited_sol_sql[0]` against `<db>.sqlite`. Capture the first row
(or `[]` if no rows). The verifier will re-execute later; this is
just to record a snapshot for human spot-check and to fail-fast on
typos.

```python
import sqlite3
con = sqlite3.connect("../mini-interact/<db>/<db>.sqlite")
cur = con.execute(audited_sol_sql[0])
sample_row = list(cur.fetchone() or [])
```

If the SQL throws, fix it before writing the JSONL line. The audit
is incomplete until execution succeeds.

### Step 8 — Append/overwrite the sidecar line

Path: `bird-interact-agents/audited_gold/<db>/<db>_audited.jsonl`.

If the file doesn't exist, create the directory and the file. If a
line already exists for this `instance_id`, rewrite the whole file
with the line replaced (latest-wins).

Build the JSON object per the schema above, serialise with
`json.dumps(obj, ensure_ascii=False)`, append a newline. Each JSONL
line is one task's full audit.

### Step 9 — Per-DB summary (optional, after auditing all tasks)

When done with a whole DB, print a small summary:

```text
shop: 12 tasks audited
  clean:          N
  edited:         N
  unrecoverable:  N

For each non-clean task, the most-cited unjustified pattern:
  shop_X: <pattern>
```

This is informational; the verifier script handles the rigorous checks.

## Worked example: shop_3

End-to-end, the produced sidecar line is (synthetic — no benchmark data):

```json
{
  "instance_id": "shop_3",
  "selected_database": "shop",
  "audit_status": "edited",
  "original_sol_sql": [
    "WITH SupplierTier AS (SELECT CASE WHEN LOWER(s.tier) IN ('gold', 'platinum') AND LOWER(s.ship_class) IN ('express', 'priority') THEN 'Premium' ELSE 'Standard' END AS supplier_class FROM orders o JOIN packages p ON o.order_no = p.order_link JOIN suppliers s ON p.supplier_link = s.supplier_ref WHERE CAST(json_extract(p.pkg_specs, '$.Pack_Size') AS REAL) > 0 AND (o.unit_count / CAST(json_extract(p.pkg_specs, '$.Pack_Size') AS REAL)) > 2) SELECT COUNT(CASE WHEN supplier_class = 'Premium' THEN 1 END) - COUNT(CASE WHEN supplier_class = 'Standard' THEN 1 END) FROM SupplierTier;"
  ],
  "audited_sol_sql": [
    "SELECT COUNT(*) FROM orders o JOIN packages p ON o.order_no = p.order_link JOIN suppliers s ON p.supplier_link = s.supplier_ref WHERE CAST(json_extract(p.pkg_specs, '$.Pack_Size') AS REAL) > 0 AND (o.unit_count / CAST(json_extract(p.pkg_specs, '$.Pack_Size') AS REAL)) > 2 AND LOWER(s.tier) IN ('gold', 'platinum') AND LOWER(s.ship_class) IN ('express', 'priority');"
  ],
  "audited_sample_row": [0],
  "changes": [
    {
      "clause_kind": "outer_select",
      "original": "COUNT(CASE WHEN supplier_class = 'Premium' THEN 1 END) - COUNT(CASE WHEN supplier_class = 'Standard' THEN 1 END)",
      "replacement": "COUNT(*)",
      "why_unjustified": "The subtraction of premium minus standard counts is not authorised by any KB entry, column-meaning, or labeled-ambiguity sql_snippet. The user query asked for 'a count of heavy orders from our premium suppliers', which is a single non-negative count, not a difference.",
      "justified_by": ["primitive"]
    },
    {
      "clause_kind": "cte",
      "original": "WITH SupplierTier AS (SELECT CASE WHEN <premium-predicate> THEN 'Premium' ELSE 'Standard' END AS supplier_class FROM <joins> WHERE <density-guard> AND <density>2>) ... FROM SupplierTier",
      "replacement": "FROM <joins> WHERE <density-guard> AND <density>2> AND <premium-predicate>",
      "why_unjustified": "The CTE existed only to feed the unjustified outer subtraction. With the subtraction removed, the CTE collapses into a direct WHERE+JOIN. No semantics are lost — the premium-predicate moves verbatim from the CASE's WHEN into the WHERE.",
      "justified_by": ["kb:22", "labeled_ambiguity:premium suppliers", "primitive"]
    }
  ],
  "reasoning_summary": "Removed the unauthorised premium-minus-standard subtraction. The heavy predicate (unit_count/Pack_Size > 2) is fully sanctioned by labeled_ambiguity 'heavy orders'; the premium predicate is fully sanctioned by KB 22 and labeled_ambiguity 'premium suppliers' (also knowledge_ambiguity 'Premium Supplier'). The audited SQL keeps every justified clause and replaces the outer SELECT with a plain COUNT(*), which matches the natural reading of the user query.",
  "skill_version": "audit-gold-sql/1.0",
  "audited_at": "<ISO-8601 timestamp>"
}
```

Running the audited SQL against `shop.sqlite` returns a single
non-negative integer — the count of heavy orders that are also from
premium suppliers. That is the natural reading of the user query.

(For comparison: the original gold returns a *negative* number, because
the unjustified `premium_count - standard_count` subtraction drops below
zero whenever Standard rows outnumber Premium ones; a plain count never
can. That sign flip is the smell that first motivates the audit.)

## Edge cases

### "Query" vs "Management" tasks

This skill is currently scoped to `category="Query"` tasks (the default
row-count comparator in `test_case_default`). For `category="Management"`
tasks, the eval also runs `preprocess_sql` and `clean_up_sqls` around
the submission, and may rely on side-effects rather than rows. If you
encounter a Management task, **flag and stop** — auditing those needs
a separate pass that's out of the current skill's scope. Set
`audit_status="unrecoverable"`, copy the gold verbatim into
`audited_sol_sql`, and add a `changes` entry of
`clause_kind="management_category"` documenting the deferral.

### Custom `test_cases`

If `task.test_cases` is non-empty, the eval doesn't use the default
row-count comparator — it executes the task's own Python `test_case`
functions. Those snippets may embed their own copies of the gold's
logic. Auditing them is out of scope for this skill (the worked
example has empty `test_cases`, so this won't come up here). Flag with
`clause_kind="custom_test_cases"`, copy gold to audited verbatim, set
status `"unrecoverable"`.

### Multiple `sol_sql` entries

Most tasks have a single-element `sol_sql`. Tasks with multiple
elements (e.g. multi-statement Management tasks) need every entry
audited. Out of scope here — same `unrecoverable` deferral as above.

### LIKE / pattern matching not in column meaning

If the gold uses `LIKE '%Apartment%'` and the column meaning for
`Dwelling_Class` lists `"Apartment"` as a sample value, that's
class-2 justified (the value's in the legal set; LIKE is primitive).
If the gold uses a value NOT in any source — e.g. `WHERE
status = 'foo'` where `'foo'` appears in no column-meaning sample-value
list — that's unjustified. Strip or replace.

### Numeric thresholds

`> 2`, `>= 5`, etc. The literal is justified iff:
- It appears in a labeled-ambiguity `sql_snippet`, OR
- A KB entry's `definition` names the threshold explicitly.

If neither: replace with the threshold from the natural reading of the
user query, or remove the predicate.

### CAST / NULLIF / COALESCE

Always primitive — never need citation. The operand columns still
need column_meaning citations.

### COUNT(DISTINCT x)

The DISTINCT modifier is primitive but is semantically loaded. If the
user asked for "the number of unique X", DISTINCT is justified by the
natural reading. If the user just asked for a count, COUNT(DISTINCT)
narrowing to one bucket needs a citation explaining why.

## Anti-patterns to call out

If you find any of these in a gold SQL, the audit should at minimum
mention it:

- **Set arithmetic on counts** that doesn't match a natural-language
  difference/ratio in the user query (the worked example's
  `premium_count - standard_count`).
- **Hidden bucketing** — CASE WHEN that introduces categorical labels
  not present in any source.
- **Hidden ranking / ordering** — ORDER BY a derived expression that's
  not in the user's question or any KB.
- **Implicit deduplication** — `SELECT DISTINCT` when the user didn't
  ask for unique values.
- **JOINs to tables that don't appear in any KB / column-meaning
  reference for the question's terms** — usually a sign the gold is
  pulling in unrelated columns.

## After the per-DB pass

1. Run the verifier: `uv run python scripts/verify_audited_gold.py
   --db <db>`. Fix any failures (typos, missing citations, SQL that
   doesn't execute).
2. Spot-check a random `audit_status="edited"` line by hand —
   re-derive the audited SQL from the same KB / column-meaning /
   ambiguity inputs and check it matches.
3. If the DB has any `unrecoverable` rows, that's a finding worth
   surfacing: it means the upstream gold is mis-spec'd against its
   own KB. Consider filing a Linear issue describing the pattern.
