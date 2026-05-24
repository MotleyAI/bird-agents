---
name: translate-mini-interact-kb
description: Translate one BIRD-Interact mini-interact database's `*_kb.jsonl` knowledge base into SLayer model enrichments via the SLayer MCP, then export the result to YAML at `bird-interact-agents/slayer_models/<db>/`. Wraps the domain-agnostic `kb-to-slayer-models` skill with mini-interact-specific bookkeeping (KB ids, deferred-KB SLayer memories, verifier gate).
---

# Translate one mini-interact KB into a SLayer model directory

This skill orchestrates translation of a single BIRD-Interact
mini-interact database's KB into SLayer models. The actual recipes for
encoding KB entries live in the **`kb-to-slayer-models` skill** —
defer to it for any "how do I encode this entry" question. This file
covers only the mini-interact-specific bookkeeping wrapped around the
generic recipes.

## When to use

Apply once per mini-interact database, after auto-ingesting it through
the SLayer MCP. The user typically invokes by naming the DB
("translate the credit DB", "run on households", etc.).

## Inputs (per DB)

- `mini-interact/<db>/<db>_kb.jsonl` — KB entries (id, knowledge,
  description, definition, type, children_knowledge).
- `mini-interact/<db>/<db>_column_meaning_base.json` — column meanings;
  use to resolve prose-only formulas onto column names.
- `mini-interact/<db>/<db>_schema.txt` — schema; use only when the KB
  references a cross-table relationship that auto-FK ingestion may have
  missed (rare).

## Outputs (per DB)

- `bird-interact-agents/slayer_models/<db>/*.yaml` — exported SLayer
  models for the DB.
- **One SLayer memory per KB entry left unencoded.** Saved via
  `mcp__slayer__save_memory(data_source=<db>, learning=…, linked_entities=…)`.
  The memory's `learning` body's first non-blank line is
  **load-bearing**: it must match `^KB (\d+) —` (em-dash, U+2014).
  `linked_entities` must be a non-empty list of canonical refs of
  the form `<db>.<model>[.<leaf>]`; the verifier requires at least
  one entry starting with `<db>.`. See the **"Deferred-KB memory
  recording"** section in `kb-to-slayer-models` for the full body
  template and a fully-worked example using KB 16 (households).
- **The legacy `_notes/<db>.md` file is no longer written or read.**
  Pre-existing files for unmigrated DBs stay in place (orphaned)
  until each is re-encoded under the new skill.

## The mandatory `meta.kb_id` annotation

When invoking any recipe from `kb-to-slayer-models` to encode a KB
entry, **always pass `meta={"kb_id": <id>}`** where `<id>` is the
source entry's numeric id from `<db>_kb.jsonl`. One entity, one KB id.

The plural `meta.kb_ids` is deprecated. If a single entity feels
like it should carry multiple ids (e.g. one JSON column standing in
for several `value_illustration` sub-fields, or one model bundling a
calc with its threshold), that's a Bucket A/B/C/D/E/F split case —
see "Splitting multi-KB entities" in `kb-to-slayer-models`. Apply
the matching `R-SPLIT-*` recipe; produce N single-KB entities.

This is non-optional:

- The W1c verifier matches encoded entities to KB ids by walking
  `meta.kb_id` across every column / measure / aggregation / model.
  Missing → false-negative "unaccounted KB id".
- The W5 HARD-8 preprocessor drops entities whose `kb_id` is in the
  task's `deleted_knowledge` list. Missing → entity stays visible to
  the agent in tasks where it should be hidden. The preprocessor
  also accepts the legacy `meta.kb_ids` plural for entities that
  haven't been split yet, so live storage stays consistent through a
  partial split rollout.

For query-backed models (R-MULTISTAGE / R-WINDOW / R-EXISTS / R-VAR),
attach `meta={"kb_id": <id>}` on the *model* itself via a follow-up
`mcp__slayer__edit_model(model_name=…, meta={"kb_id": <id>})` call,
since `create_model(query=…)` doesn't expose `meta` directly.

When one KB entry produces multiple entities (a Column + a
ModelMeasure for an R-FILTER ratio, say), every one of them carries
the same `kb_id`. The verifier deduplicates.

## Workflow

### W4c refresh override (use when re-running on a partially-encoded DB)

When you're refreshing an already-translated DB — the storage already
contains entities with `meta.kb_id` set from a prior pass and may
already have deferred-KB memories saved — follow the generic skill's
two-pass "Working from a partially-encoded model" discipline:

1. **Surface prior deferral memories** before re-encoding. Call
   `mcp__slayer__recall_memories(data_source=<db>)` (or
   `mcp__slayer__search` if you want a question-scoped recall) and
   list each `KB <n> — …` learning. These are the load-bearing
   context for any KB id that previously deferred; the encoder may
   either preserve the memory or update it via
   `forget_memory(identifier=<id>, data_source=<db>) + save_memory(…)`.
2. **Verify-then-fill**: walk the KB jsonl. For each id, look up the
   matching entity (any `meta.kb_id`, or — on legacy entities not
   yet split — any `meta.kb_ids` list covering that id) via
   `models_summary` + `inspect_model`. Faithful encodings stay;
   wrong ones get edited; misclassified ones get `delete_model`'d
   / replaced. Only after this verify pass do you try to encode the
   still-unencoded.
3. **No gold-SQL grounding.** The encoding decisions come from KB
   text (`definition`, `type`, `children_knowledge`), the schema,
   the column meanings, the recipes in the generic skill, and the
   search-first preamble output. Do not consult
   `mini_interact.jsonl` task `sol_sql` fields.

### W4d refresh override (multi-KB-entity splitting pass)

When the planner gives you a per-DB instruction file at
`_w4d_instructions/<db>.md` listing entities with `meta.kb_ids`
(plural) and a target split shape per entity:

1. Read `_w4d_instructions/<db>.md`. It lists every multi-KB entity
   in this DB plus the bucket label (A/B/C/D/E/F) and the target
   split (per kb_id: target entity name, host model, recipe).
2. For each listed entity, apply the matching `R-SPLIT-*` recipe
   from `kb-to-slayer-models` ("Splitting multi-KB entities"). End
   state per entity: N single-KB entities each carrying its own
   `meta.kb_id`. Existing single-KB entities on the same model are
   load-bearing — don't disturb them.
3. **Don't re-run the W4c verify-then-fill.** W4d is purely a
   splitting pass; coverage is already complete. For Bucket-F
   secondary ids, save a new memory with
   `Status: not-applicable — duplicate of KB <primary>` per the
   generic skill's "Deferred-KB memory recording" template.
4. After splitting: re-export YAML (step 5 below), then run BOTH
   gates:
   - `scripts/verify_kb_coverage.py --db <db>` (exit 0 — coverage
     unchanged).
   - `scripts/multi_kb_audit.py --db <db>` (exit 0 — no multi-KB
     entities remain).

Otherwise (first-time translation of a fresh DB, or a W4c refresh
without a `_w4d_instructions/<db>.md` file), proceed with the
standard steps below.

### 1. Read the inputs

Load `<db>_kb.jsonl`, `<db>_column_meaning_base.json`, and
`<db>_schema.txt`. Build a working list of KB entries to process.

### 1b. Regenerate the SLayer base model (deterministic + LLM date typing)

Before encoding any KB entries, prepare the live SLayer storage with
the schema-derived models, deterministic typing, JSONB-leaf expansion
(DEV-1379), and LLM TEXT-as-date detection (DEV-1381):

```bash
python bird-interact-agents/scripts/regenerate_slayer_model.py --db <db>
```

This subsumes the old `ingest_slayer_models.py` path for a single DB.
It runs four phases:

1. `slayer datasources create` + `slayer ingest` (schema → one Column
   per source column).
2. `column_meaning` overlay — descriptions, leading-type-token typing,
   and DEV-1381 date-format annotations (`"Date stored as TEXT in
   '<strftime>'. Cast at encode time to TIMESTAMP."`).
3. JSONB-leaf expansion — for every JSONB column with a
   `fields_meaning` sidecar, append one Column per terminal leaf with
   full-path `__` naming, `JSON_EXTRACT` sql (no CAST), copied
   description, and `meta.derived_from` for idempotency.
4. LLM TEXT-as-date detection — sample up to 20 distinct values from
   each remaining `string`-typed column, classify via the Anthropic
   API, and retype confident matches to TIMESTAMP with a SQLite-native
   parse expression (cached in `meta.date_source_format`).

Review the warning files before continuing to step 2:

- `bird-interact-agents/results/<db>/column_typing_warnings.txt` —
  leaves whose `fields_meaning` text had no leading type token
  (defaulted to TEXT). Consider enriching `column_meaning_base.json`.
- `bird-interact-agents/results/<db>/jsonb_drift_warnings.txt` —
  top-level keys present in JSON data but absent from
  `fields_meaning`, and vice versa.
- `bird-interact-agents/results/<db>/date_detection_warnings.txt` —
  LLM-uncertain TEXT columns left untyped.

Sandboxing note: the script writes to `$SLAYER_STORAGE` (default
`~/.local/share/slayer`); ensure your sandbox allows writes there
or run the script unsandboxed. SQLite reads of `mini-interact/<db>/`
use `immutable=1` URI so the data dir doesn't need write access.

Global LLM model override (applies to phase 4 and any future LLM
step in the pipeline): `BIRD_AGENTS_LLM_MODEL=claude-sonnet-4-6`.
Default: `claude-haiku-4-5`.

### 2. Encode each entry

Invoke `kb-to-slayer-models` recipes per entry. Pick the recipe whose
trigger matches the entry's `definition` and `type`. Pass
`meta={"kb_id": <id>}` on every entity created. Follow the 6-pass
order from the generic skill so dependencies (descriptions → joins →
columns → measures → multistage → host placement) resolve cleanly.

If an entry can't be encoded for any reason, skip it and queue it for
step 4.

### 3. Sanity-check the result

Pick one or two representative tasks for this DB from
`mini_interact.jsonl` (filter by `selected_database == "<db>"`). For
each, write the natural agent SLayer query that would answer the
task's `amb_user_query`. The query should be short — measure list +
dimensions + filters — with no per-task formula derivation. If a task
needs derivation, that's a sign the relevant KB entry didn't get
encoded; loop back to step 2.

### 4. Record a memory per deferred KB entry

For every KB entry queued in step 2, call
`mcp__slayer__save_memory(data_source=<db>, learning=…, linked_entities=…)`
following the **"Deferred-KB memory recording"** section in
`kb-to-slayer-models`. Load-bearing contract:

- First non-blank line of `learning` is exactly `KB <id> — <KB.knowledge>`
  (em-dash, U+2014).
- `linked_entities` is non-empty; at least one entry starts with
  `<db>.`.
- The body covers: verbatim KB item, reason, status (one of the
  canonical values), clarifying questions (REQUIRED for AMBIGUOUS-
  PROSE / SCHEMA-GAP), related KB ids, relevant entities.

If no entries were skipped, no memories need to be written — the
verifier reads `meta.kb_id` directly off the encoded entities.

After every `save_memory` call, sanity-check with
`mcp__slayer__search(question="KB <id>", data_source=<db>)` (or the
wrapper at `scripts/slayer_search_for_db.py`) and confirm the memory
comes back at rank 1.

### 5. Export to YAML

```
python bird-interact-agents/scripts/export_slayer_models.py --db <db>
```

This reads the live SLayer storage (the one the MCP writes to —
`~/.local/share/slayer` by default, override via `$SLAYER_STORAGE`)
and writes a YAMLStorage-shaped tree at
`bird-interact-agents/slayer_models/<db>/` containing exactly the
models with `data_source == "<db>"` plus the datasource itself.
Layout:

```
bird-interact-agents/slayer_models/<db>/
├── datasources/
│   └── <db>.yaml
├── memories.yaml
└── models/
    ├── <model_a>.yaml
    └── <model_b>.yaml
```

Idempotent: re-runs overwrite the destination directory, so this also
serves as the canonical refresh after fixing a recipe and re-applying.

### 6. Run the verifier — definition of done

```
python bird-interact-agents/scripts/verify_kb_coverage.py --db <db>
```

The DB is "done" when this exits 0. Anything else means at least one
KB id is unaccounted for or one KB id is in both the encoded set and
the notes file. Fix the offender and re-run until green.

## Verifier contract (so step 6 is unambiguous)

The verifier reads:

- The KB JSONL → set of all `id` values.
- The exported YAML directory → every `meta.kb_id` value across
  Columns, ModelMeasures, Aggregations, and SlayerModels.
- The per-DB SLayer memories at `slayer_models/<db>/memories.yaml`
  → for each KB id not already in the encoded set, search for
  `KB <id> — <knowledge>` via the per-DB SearchService and accept
  any hit whose first non-blank line matches `KB <id> —` AND
  whose linked entities contain at least one ref starting with
  `<db>.`.

Pass condition: every KB id is in **exactly one** of {encoded,
documented}. The verifier exits 1 (with a list of offending ids) if
any id is in zero or both sets.

`--all` mode iterates every `slayer_models/<db>/` directory. After
the kb-notes-to-slayer-memories migration, only DBs that have been
re-encoded under the new skill will pass; the legacy notes files for
unmigrated DBs are ignored. Use `--db <db>` for the day-to-day gate.

## Constraints & gotchas

- **Don't recreate auto-ingested models.** Use `edit_model` to enrich
  what's already there. Only `create_model` for query-backed
  multistage models that have no host table.
- **Peer-join handling.** Two child tables that both FK to the same
  parent (e.g. `thermalsolarwindandgrid` and `waterandwaste` both
  joining to `equipment` in polar) cannot reach each other through
  bare `Column.sql` references. **Encode any cross-peer composite
  as R-MULTISTAGE** per the generic skill's "Peer-join via shared
  parent" section. Do *not* defer; the multistage DAG is the
  default encoding, not a workaround. Cardinality is safe regardless
  of m:N (each peer is collapsed to one row per parent in its own
  stage).
- **KB ambiguity surfaces during sanity-queries, not in the
  verifier.** The verifier checks KB-coverage, not semantic
  correctness. If a sanity-query produces nonsense values
  (negative scores, exploded ranges), the encoding has a
  KB-interpretation bug — fix the formula, but the verifier will
  still pass either way. If you can't immediately resolve, save an
  additional memory with the encoded entity in `linked_entities`,
  body status `Status: encoded but value range is suspect; see …`,
  so the agent surfaces the caveat at query time.
- **Datasource name = DB name.** `mcp__slayer__create_datasource(name="<db>", …)`.
  The W4 fan-out depends on this — parallel agents pick distinct names
  this way.
- **`data_source=<db>` on every MCP call.** See the "Important rules"
  banner in `kb-to-slayer-models`; the long-lived MCP server sees
  every datasource it has registered and silently picks the wrong
  one otherwise.
- **Re-runnable.** `edit_model` upserts by name; re-running the skill
  on a DB updates the YAML in place. Deferred-KB memory refreshes
  go through `forget_memory(identifier=<old>) + save_memory(…)` to
  avoid duplicates.
