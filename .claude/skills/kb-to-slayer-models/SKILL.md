---
name: kb-to-slayer-models
description: Translate a structured knowledge base (calculation entries, descriptions, enum bands, multistage formulas) into edits on a SLayer datasource via the v0.6.1+ MCP server. A recipe book — a caller picks the recipe whose trigger matches each KB entry and applies it. Domain-agnostic; downstream wrappers add domain bookkeeping. Deferred KBs are recorded as SLayer memories carrying the KB content, reason, clarifying questions, and linked entities.
---

# Translate a KB into SLayer model enrichments via MCP

## Premise

By the time this skill runs:

- The target database has been registered as a SLayer datasource and
  **auto-ingested** — `mcp__slayer__create_datasource` with
  `auto_ingest=True`, or `mcp__slayer__ingest_datasource_models`.
- One SLayer model per table already exists, with **FK joins**, **column
  types**, and one **Column** per source column populated.
- **SLayer 0.6.1+ MCP** is the runtime. `edit_model` and `create_model`
  expose the v3 unified `columns` list and the v3 named-formula
  `measures` list with `formula` . The rank-family
  transforms (`rank`, `percent_rank`, `dense_rank`, `row_number`,
  `ntile`) and `covar_pop` / `covar_samp` / `corr` aggregations are
  available as ModelMeasure formulas.

This skill **edits the existing models in place** via the SLayer MCP
server. **Do not recreate** auto-ingested models. Reserve `create_model`
for query-backed (multistage) models that have no host table.

**All edits go through the MCP** (`mcp__slayer__edit_model` /
`create_model` / `delete_model`). Do not edit YAML files directly.
The MCP writes to the live storage; downstream wrappers re-export YAML
(e.g. via `scripts/export_slayer_models.py`) once the in-storage state
is correct.

This skill is **domain-agnostic**. It says nothing about JSONL files,
KB ids, notes files, or verifiers. Wrappers (e.g.
`translate-mini-interact-kb`) layer that bookkeeping on top by passing
their own `meta=…` payloads through the recipes.

## Important rules (read first)

- **`data_source=<db>` on every MCP call.** `models_summary`,
  `inspect_model`, `edit_model`, `create_model`, `delete_model`,
  `query`, `save_memory`, `recall_memories` all accept an explicit
  `data_source` (or equivalent named arg) — pass it every time. This
  is the no-restart-required substitute for per-DB MCP isolation; the
  long-lived MCP server sees every datasource it has registered and
  silently picks the wrong one otherwise.
- **`mcp__slayer__search` is the only tool whose `question` channel
  cannot be DB-scoped via an arg.** Use the wrapper script
  `python scripts/slayer_search_for_db.py --db <db> --question "<text>"`
  whenever the encoder needs free-text search; it boots a short-lived
  `SearchService` against `slayer_models/<db>/` so cross-DB pollution
  is impossible. Direct `mcp__slayer__search` is fine when the agent
  is using the `entities=[...]` channel only (canonical refs already
  carry the DB prefix).
- **Verifier scope.** After this migration, `verify_kb_coverage --all`
  fails for every DB that has not been re-encoded under the new
  skill: it expects deferred KBs to live as SLayer memories, and DBs
  whose previous `_notes/<db>.md` file is the only documenting source
  will report all those ids as unaccounted. Use `--db <db>` per
  re-encoded DB.

## MCP tools (in order of preference)

1. `mcp__slayer__models_summary(datasource_name=<db>)` — list existing
   models for the datasource.
2. `mcp__slayer__inspect_model(model_name=<m>)` — see one model's
   columns, named-formula measures, custom aggregations, joins, sample
   data.
3. **`mcp__slayer__edit_model(...)`** — primary write path. Upserts
   `columns`, `measures` (named formulas), `aggregations`, `joins`;
   appends/removes model `filters`; updates `description`. Re-runnable
   safely (upsert by name on each list).
4. `mcp__slayer__create_model(name=…, query=[…])` — **only** for
   query-backed (multistage) models that have no host table.
5. `mcp__slayer__query(...)` — verification step at the end.

## v3 vocabulary cheat-sheet

| Concept | Where it lives | Shape |
|---|---|---|
| Row-level expression on a model (used as group-by dim or aggregation source) | `model.columns` | `{name, sql, type, primary_key, allowed_aggregations, filter, label, description, hidden, meta}` |
| Named formula on a model (a saved metric) | `model.measures` | `{name, formula, label, description, meta}` — `formula` is e.g. `"revenue:sum / *:count"` or `"duration:sum"` |
| Custom (parametrized) aggregation | `model.aggregations` | `{name, formula, params: [{name, sql}], description, meta}` — `formula` uses `{value}` and named params |
| Joined-table reference in raw SQL | inside `Column.sql` / `Column.filter` / model `filters` | `target_alias.col` (single dot) or `path__alias.col` (multi-hop) |
| Joined-table reference in queries | `dimensions` / `measures` strings | `model.col` or `model.subpath.col` (dots) |

`meta` on every entity is an open `Dict[str, Any]`. Recipes here pass
through whatever `meta` the caller supplies; the caller decides
semantics (KB ids, owner tags, source links, etc.).

## Recipe selection

For each KB entry, scan `definition` (the formula / description) and
`type`; pick the **first** recipe whose trigger matches; apply it; move
on.

### Search-first preamble (run before recipe selection)

For every KB entry being processed, the FIRST action is to call the
per-DB search wrapper with the KB content as the question:

```bash
python scripts/slayer_search_for_db.py --db <db> \
    --question "KB <id> — <KB.knowledge>: <KB.definition>"
```

The wrapper returns a `SearchResponse` JSON with three channels —
`memories`, `entities`, `example_queries`. Read them in this order:

- **`memories`** — prior deferral memories from this DB. For each hit,
  the first non-blank line of `text` matches `KB <n> —`:
  - If `<n>` equals the **current** KB id, this is a refresh.
    Re-read the existing reason / clarifying questions before
    deciding. If you replace the memory, call
    `mcp__slayer__forget_memory(identifier=<hit.id>, data_source=<db>)`
    first; otherwise the corpus accumulates duplicates.
  - If `<n>` is a **different** id, this is a related KB. Cite it
    in the "Related KB ids" section of any new memory you save, and
    reuse its operationalisation when picking a recipe (often you
    can avoid creating a new entity by stamping `meta.kb_id=<current>`
    on the existing one — see R-RESOLVE).
- **`entities`** — existing columns / measures / aggregations that
  already implement this KB or an adjacent one. Strongly prefer
  `edit_model(data_source=<db>, ...)` to stamp `meta.kb_id` on those
  rather than creating new entities. Use the entity's `id`
  (`<db>.<model>[.<leaf>]`) directly.
- **`example_queries`** — prior-saved `SlayerQuery` examples a future
  agent could fire to answer this KB. If one already covers the KB
  intent, the current encoder doesn't need to create a new query-
  backed model; just stamp `meta.kb_id` on the underlying
  contributing entities.

Only after the search-first preamble do you proceed to recipe
selection below.

| Recipe | Trigger |
|---|---|
| **R-DESCRIBE** | Prose narrative with no algebraic formula. |
| **R-JOIN** | Cross-table reference whose join chain isn't reachable through auto-FK joins (rare — check the schema first). |
| **R-COL** | Row-level arithmetic over columns of one host table; no Σ/AVG-over-rows; no anchor date; no quartile. |
| **R-CASE** | Definition maps an enum / numeric band to a label or scalar. |
| **R-FILTER** | "Count of rows matching X" / "sum where X" / a ratio whose numerator is a filtered count of the same table. |
| **R-MEASURE** | Named metric is a single-column aggregate (Σ, AVG, MIN, MAX, COUNT, COUNT DISTINCT) over rows of one table; not parametrized. |
| **R-AGG** | Named metric is an aggregation that needs more than one column or has tunable parameters (MKT, weighted_avg with a non-default weight column, trimmed_mean with bounds, …). |
| **R-RESOLVE** | Definition contains "where X is the …" or any unqualified reference to another already-encoded metric by name. |
| **R-MULTISTAGE** | A row-level expression references an aggregate of child rows; one stage of arithmetic must run after a stage of aggregation. |
| **R-PEER-JOIN** | A composite formula references measures or columns on two (or more) tables that share a common ancestor in the FK graph. **Default encoding: R-MULTISTAGE** (per-peer aggregation, then joined DAG). See "Peer-join via shared parent" below. |
| **R-WINDOW** | "Top quartile / above median / rank ≤ N / percentile ≥ p / NTILE bucket / argmin-by-time" used as a row-level dimension or filter. Available transforms: `rank`, `percent_rank`, `dense_rank`, `ntile`, `first` (FIRST_VALUE asc / argmin), `last` (FIRST_VALUE desc / argmax), `lag`, `lead`. Pass `partition_by=<dim>` (or a list) to scope the window; cross-model dotted paths work (e.g. `partition_by=actuation_data.robot_details.ctrltypeval`). For non-standard window expressions, fall back to a raw window inside `Column.sql` — SLayer auto-promotes Column-level windows so they survive the SQL generator. |
| **R-EXISTS** | "Exists ≥1 child meeting X", "all children meet X", "% of children meeting X" as a Boolean attribute on the parent. |
| **R-VAR** | Definition depends on a "report date" / "as of" / "past N months" — anything that's a query parameter, not data. |
| **R-HOST** | Boolean predicate composes facts from multiple tables; no obvious host model — pick by predicate grain. |
| **R-PROSE** | Algebraic intent expressed only in prose — resolve column refs from the surrounding column-meaning metadata, then emit via R-COL / R-MEASURE / R-AGG. |

## KB self-annotation discipline

Every encoded entity carries three mandatory fields:

- `label` = `KB.knowledge` (verbatim).
- `description` containing a canonical block:

  ```
  [kb=<id>]
  <KB.definition> — <KB.description>
  [/kb=<id>]
  ```

  Optional free-text caveats (e.g. "encoded with proxy because schema
  lacks X") go ABOVE the block, separated by a blank line. The block
  is regenerable — re-runs strip and re-emit; caveats survive.

- `meta.kb_id` (singular int). One entity, one KB id. The plural
  `meta.kb_ids` is deprecated; if a single entity feels like it
  should carry multiple ids, use a splitting recipe (see
  "Splitting multi-KB entities" below). Multiple entities may share
  the same `meta.kb_id` — the verifier dedupes.

**kb_id placement rule.** Stamp `meta.kb_id` on the entity that
*operationalizes* the KB (the helper column that extracts and
computes), NOT on the data carrier (a JSON blob column that holds
the raw values). If both exist, the helper carries the kb_id and
the JSON blob stays untagged.

## NULL handling — be explicit, never implicit

Whenever you author a Column or ModelMeasure that references a column
that may carry NULLs (i.e. its `NOT NULL` constraint is absent in the
schema, or `fields_meaning` text says "Contains NULL when..."), make
a deliberate choice rather than letting defensive habits diverge from
the gold encoding.

1. **Numeric columns used in aggregations** (`SUM`, `AVG`, `COUNT
   DISTINCT`, ratios):
   - If "missing" semantically equals zero (a count of events, a fee,
     a tax-paid), wrap `COALESCE(<col>, 0)` in `Column.sql` or in the
     measure formula. State the reason in the description: `"NULL
     means 'no events'; coalesced to 0 so SUM is well-defined."`
   - If "missing" semantically means "unknown / not measured" (an
     optional rating, a sometimes-omitted score), DO NOT coalesce —
     let NULL propagate so `AVG` correctly excludes unknowns. Note the
     choice in the description.

2. **Categorical / enum-like text columns** where `''`, `'NA'`,
   `'N/A'`, `'-'`, `'Unknown'` appears as a sentinel for "no value":
   - The `Column.sql` SHOULD do `NULLIF(<col>, '<sentinel>')` so
     downstream `IS NULL` filters work uniformly. Document the
     sentinel and its source.
   - If multiple sentinels coexist (e.g. `''` and `'NA'`), chain
     `NULLIF`s.

3. **Row-level filters** on a column whose NULL-handling matters
   (e.g. "exclude tasks that have not yet completed" → completion
   date `IS NOT NULL`):
   - Express the filter at the Filter level via `IS NOT NULL`, not
     via a coalesce-then-compare pattern, so the filter's intent is
     visible to downstream readers and the SLayer filter-tree.

4. **Ratios / divisions:**
   - Always `NULLIF(<denominator>, 0)` to make division-by-zero
     yield NULL rather than throwing. Document the policy in the
     measure description.

**Default rule.** If the KB / `column_meaning` / schema do not state
the NULL semantics for a column, pick the conservative reading (let
NULL propagate in numeric aggregations; do not assume sentinels in
text columns). Add a `description` line stating the assumption
explicitly so reviewers can correct it if wrong.

This convention exists because the KB-vs-data audit on BIRD-Interact
mini found 36.3% of NULL-handling operators were KB-silent — gold SQL
diverged from encoded models on coalesce/NULLIF choices that nobody
had documented. Codified here (DEV-1380) so future encoding runs make
the choice deliberately and visibly.

## LLM TEXT-as-date detection (pipeline recipe)

Not a per-KB recipe — a pipeline step run **before** KB encoding by
`scripts/regenerate_slayer_model.py` (DEV-1381). Documented here so
non-BIRD callers can reuse the same contract.

**Trigger.** A `Column.type == TEXT` whose schema-derived name and
sampled values suggest it holds dates or timestamps.

**Mechanism.** For every TEXT column that lacks both
`meta.date_source_format` (already inferred) and `meta.derived_from`
(JSONB leaf — sampling via `JSON_EXTRACT` is deferred), sample up to
20 distinct non-NULL values from the live datasource and ask the LLM
(default `claude-haiku-4-5`, override via `$BIRD_AGENTS_LLM_MODEL`)
to classify with this contract:

Return JSON only:

```json
{"is_date": true|false, "source_format": "<strftime>", "confidence": 0.0-1.0, "notes": "..."}
```

- `is_date=false` → leave the column alone.
- `is_date=true` with `confidence >= 0.8` AND `source_format` valid
  under `datetime.strptime` for EVERY sampled value → retype to
  `TIMESTAMP`; rewrite `Column.sql` to the SQLite-native parse
  expression for the source format (passthrough for ISO; `SUBSTR`
  concatenation for `%d/%m/%Y`, `%m/%d/%Y`, `%d-%m-%Y`; `REPLACE`
  for `%Y/%m/%d`); cache the inference in
  `meta.date_source_format=<strftime>` plus `meta.detected_by='llm'`
  for idempotency on re-run.
- `is_date=true` with `confidence < 0.8` OR a format that fails the
  strptime gate on at least one sample → leave the column TEXT and
  log to `results/<db>/date_detection_warnings.txt`.

**Why deterministic-first.** The pipeline phase 2 already retypes
columns whose `column_meaning` description starts with a recognised
SQL-type token (`DATE`, `TIMESTAMP`, `JSONB`, etc.) or carries the
DEV-1381 annotation grammar (`"Date stored as TEXT in '<strftime>'.
Cast at encode time to TIMESTAMP."`). LLM detection runs only on
what's left, so the typing is cheap and replayable.

**Why dialect-agnostic in spirit only.** The committed `Column.sql`
is dialect-native (SQLite for BIRD). `meta.date_source_format`
preserves the format string so a future Postgres re-emit can
regenerate the SQL — the SLayer model carries enough information to
round-trip without an LLM call.

## Peer-join via shared parent

A common shape: child A and child B both FK→same parent P. Auto-FK
ingestion gives `A.joins=[→P]` and `B.joins=[→P]` only — there's no
edge from P to its children, and no sibling edge between A and B. A
composite formula that pulls measures from both children can't be
expressed as a single dotted reference from either side.

**Default encoding: R-MULTISTAGE.** Build a query-backed model with
three named stages forming a DAG:

- **Stage 1** aggregates A by the FK column → one row per parent.
- **Stage 2** aggregates B by its FK column → one row per parent.
- **Stage 3** anchors at the dim-bearing model (typically the parent),
  declares an explicit `joins` clause referencing prior named stages
  on the parent key, and applies the composite formula.

Why multistage instead of "manual reverse joins on P + a single
formula":

- **Cardinality-safe regardless of m:N** — each peer is collapsed to
  one row per parent in its own stage; the final join is one-to-one.
  The auto-emitted SQL never produces an m×n cross-product.
- **No manual reverse joins required** on the parent (auto reverse
  joins are tracked by DEV-1338; until they ship, multistage avoids
  needing them entirely).
- **GROUP BY dimensions can live anywhere.** If the dim is on a peer
  rather than the parent, anchor stage 3 at that peer (or add a
  fourth stage). Cross-stage references use dot notation
  (`stage_a.measure_x`).
- **Generalises to deeper FK chains.** When the "shared parent" is a
  grandparent two hops up, the inner stages just have longer FK
  chains — same recipe.

For the design analysis behind this default, see
[DEV-1350](https://linear.app/motley-ai/issue/DEV-1350).

The canonical example is in Pass 5 below (R-MULTISTAGE).

## Workflow

Process the KB in this pass order so each pass uses the previous pass's
output. Skip a pass entirely when the KB has no entries of that flavour.

### Step 0 — orient

```
mcp__slayer__models_summary(datasource_name="<db>")
# for each model the KB references:
mcp__slayer__inspect_model(model_name="<m>")
```

Build a mental map: KB entry → host model.

### Pass 1 — descriptions (R-DESCRIBE)

For every prose-only entry, attach the text as `description`. `edit_model`
upserts by name, so updating a column's description doesn't disturb its
SQL or type.

```
mcp__slayer__edit_model(
  model_name="<m>",
  description="…<model-level prose>…",
  columns=[
    {"name": "<existing_col>",
     "description": "<column-level prose>",
     "meta": {…caller bookkeeping…}},
  ],
)
```

### Pass 2 — missing joins (R-JOIN)

Auto-ingest already encoded every FK as a `ModelJoin`. Add joins only
when a definition references a cross-table relationship the schema
doesn't have as an FK. Confirm against the schema first.

```
mcp__slayer__edit_model(
  model_name="<host>",
  joins=[{"target_model": "<other>", "join_pairs": [["<src_col>", "<tgt_col>"]]}],
)
```

### Pass 3 — row-level enrichments (R-COL, R-CASE, R-FILTER)

**R-COL.** Encode the synthetic index as a Column on the host model.

```
# Example: SNQI = SnrRatio − 0.1 × |NoiseFloorDbm|
mcp__slayer__edit_model(
  model_name="signal_observations",
  columns=[{
    "name": "snqi",
    "type": "number",
    "sql": "SnrRatio - 0.1 * ABS(NoiseFloorDbm)",
    "description": "Signal-to-Noise Quality Indicator (SNQI)",
    "meta": {…caller bookkeeping…},
  }],
)
```

**R-CASE.** Same path; `sql` carries a CASE WHEN.

```
# Example: Brickwork=4, Apartment=3, other=1
mcp__slayer__edit_model(
  model_name="households",
  columns=[{
    "name": "dwelling_type_score",
    "type": "number",
    "sql": "CASE dwellingType WHEN 'Brickwork' THEN 4 WHEN 'Apartment' THEN 3 ELSE 1 END",
    "description": "Dwelling Type Score",
    "meta": {…caller bookkeeping…},
  }],
)
```

**R-FILTER.** Use `Column.filter` directly — it wraps the column in a
`CASE WHEN` inside any aggregation at query time. Then a named formula
expresses the ratio.

```
# Example: ECR = applications with attorney / total
mcp__slayer__edit_model(
  model_name="applications",
  columns=[{
    "name": "with_attorney",
    "sql": "id",
    "type": "number",
    "filter": "attorney_id IS NOT NULL",
    "description": "Filtered count helper: rows with an attorney",
    "meta": {…caller bookkeeping…},
  }],
  measures=[{
    "name": "external_counsel_rate",
    "formula": "with_attorney:count / *:count",
    "description": "External Counsel Rate (ECR)",
    "meta": {…caller bookkeeping…},
  }],
)
# query: measures=[{"formula": "external_counsel_rate", "name": "ECR"}]
```

### Pass 4 — named aggregations (R-MEASURE, R-AGG, R-RESOLVE)

**R-MEASURE.** Save the named metric as a ModelMeasure formula. The
agent calls it by **bare name**.

```
# Example: TED = Σ tᵢ across child rows (per parent)
mcp__slayer__edit_model(
  model_name="excursion_events",
  measures=[{
    "name": "ted",
    "formula": "duration:sum",
    "description": "Temperature Excursion Duration (TED)",
    "meta": {…caller bookkeeping…},
  }],
)
# query: measures=[{"formula": "ted", "name": "TED"}]
```

**R-AGG.** Parametrized aggregation — needs more than one column or
tunable parameters.

For the **OLS regression slope** of y on x — the canonical KB pattern
"slope of regression(y, x) across qualifying rows" — express it as a
ModelMeasure formula using built-in `covar_pop` / `var_pop`:

```
measures=[{
  "name": "y_x_regression_slope",
  "formula": "y:covar_pop(other=x) / x:var_pop",
  "description": "OLS slope of y on x across the queried rows",
  "meta": {…},
}]
# query: filter to qualifying rows, group by whatever scope the KB defines.
# Algebra: slope = covar_pop(x, y) / var_pop(x).
```

`corr(x, y) = covar(x,y) / (stddev(x) * stddev(y))` is also available
via `:corr(other=…)` for Pearson correlation.

```
# Example: MKT (uses ln/exp UDFs registered by SLayer)
mcp__slayer__edit_model(
  model_name="excursion_events",
  aggregations=[{
    "name": "mkt",
    "formula": "(-{deltaH}/{R}) / ln(SUM(exp(-{deltaH}/({R} * {value}))) / COUNT(*))",
    "params": [
      {"name": "deltaH", "sql": "83144"},
      {"name": "R",      "sql": "8.314"},
    ],
    "description": "Mean Kinetic Temperature (MKT)",
    "meta": {…caller bookkeeping…},
  }],
)
# query: measures=[{"formula": "temperature_kelvin:mkt", "name": "MKT"}]
```

**R-RESOLVE.** Definition references another already-encoded metric by
name in prose ("where AHI is the Account Health Index"). Apply the
referenced entry's recipe **first** (so it has a name in the model),
then reference its bare name in the new formula. Cycles are rejected by
SLayer at save time.

### Pass 5 — multistage / query-backed models (R-MULTISTAGE, R-WINDOW, R-EXISTS, R-VAR)

The only pass that uses `create_model`. Use it only when the formula
needs aggregation-then-row semantics — there's no host table whose
row-level columns alone can carry the expression.

Each stage in the `query=[…]` list is a SlayerQuery dict. Use `measures`
(not `fields`) for the aggregated SELECTs.

`meta` on a query-backed model goes on the model itself via a separate
`edit_model(meta=…)` call after `create_model`.

**R-MULTISTAGE — composite that crosses an aggregation boundary; default for peer-joins.**

The canonical example is a peer-join via a shared parent. Child A
(`thermalsolarwindandgrid`) carries REC; child B (`waterandwaste`)
carries WRMI; both FK to parent `equipment`. The composite RSSI =
0.6·REC + 0.4·WRMI cannot be expressed as a single dotted formula
because the auto-FK graph has no edge between A and B. Encode as a
three-stage DAG:

```
mcp__slayer__create_model(
  name="resource_self_sufficiency_index",
  query=[
    {
      "name": "rec_per_eq",
      "source_model": "thermalsolarwindandgrid",
      "dimensions": ["thermaleqref"],
      "measures": [{"formula": "rec:avg", "name": "rec"}],
    },
    {
      "name": "wrmi_per_eq",
      "source_model": "waterandwaste",
      "dimensions": ["wasteeqref"],
      "measures": [{"formula": "wrmi:avg", "name": "wrmi"}],
    },
    {
      "source_model": {
        "source_name": "rec_per_eq",
        "joins": [{"target_model": "wrmi_per_eq",
                   "join_pairs": [["thermaleqref", "wasteeqref"]]}],
      },
      "dimensions": ["thermaleqref"],
      "measures": [{"formula": "0.6 * rec + 0.4 * wrmi_per_eq.wrmi",
                    "name": "rssi"}],
    },
  ],
)
mcp__slayer__edit_model(model_name="resource_self_sufficiency_index",
                        meta={…caller bookkeeping…})
```

Stages 1 and 2 each collapse one peer to one row per parent. Stage 3
joins them on the parent key and applies the composite formula. The
emitted SQL is cardinality-safe regardless of how many child rows each
parent has.

When the GROUP BY dim lives on a peer rather than on the parent,
anchor stage 3 at that peer instead (and add a fourth stage if the
peer dim itself comes from a different table). The shape generalises.

**R-MULTISTAGE portfolio — many derived metrics on one per-parent view.**
When several KBs all want their value at parent grain (e.g. EER, TWR,
JDI, OCE all "per robot"), don't build one query-backed model per
metric. Build *one* per-parent model that aggregates each peer child
to per-parent in its own named stage, joins them all in the final
stage, and exposes the derived metrics as **computed columns inside
the final stage's `ModelExtension.columns`**. Example skeleton:

```
mcp__slayer__create_model(
  name="<parent>_per_<parent>",
  query=[
    {"name": "_per_<peer_a>", "source_model": "<peer_a>",
     "dimensions": ["<fk_to_parent>"],
     "measures": [{"formula": "x:sum", "name": "x_total"}, …]},
    {"name": "_per_<peer_b>", "source_model": "<peer_b>",
     "dimensions": ["<fk_to_parent>"],
     "measures": [{"formula": "y:max", "name": "y_max"}, …]},
    # …one stage per peer…
    {"source_model": {
        "source_name": "_per_<peer_a>",
        "joins": [
          {"target_model": "_per_<peer_b>",
           "join_pairs": [["<fk_a>", "<fk_b>"]]},
          # …join every other named stage…
        ],
        "columns": [
          # Derived per-parent metrics composed from joined stages.
          # Each becomes an auto-introspected column on the model.
          {"name": "ratio_a_over_b",
           "sql": "CASE WHEN _per_<peer_b>.y_max > 0 THEN _per_<peer_a>.x_total / _per_<peer_b>.y_max END",
           "type": "number"},
          # Pass-throughs for ergonomic access:
          {"name": "y_max", "sql": "_per_<peer_b>.y_max", "type": "number"},
        ],
     },
     "dimensions": ["<fk_to_parent>", "x_total", "y_max", "ratio_a_over_b", …]},
  ],
)

# Cascade categories ride the same per-parent view as ModelMeasures:
mcp__slayer__edit_model(
  model_name="<parent>_per_<parent>",
  measures=[
    {"name": "is_ratio_high",
     "formula": "ratio_a_over_b:max > 0.01 and y_max:max > 1000",
     "meta": {"kb_id": <cascade_kb>}},
  ],
)
```

This shape was used for `robot_per_robot`: per-child-table rollups
for `operation`, `performance_and_safety`, `joint_condition`,
`actuation_data`, `maintenance_and_fault`, plus a `_robot_meta`
stage carrying the parent attributes (`modelseriesval`,
`ctrltypeval`, …). Derived columns: `eer`, `twr`, `oce`, `jdi`.
Measures: `eer_value`, `twr_value`, `is_energy_inefficient`,
`is_joint_health_risk`, `eer_rank`, `model_avg_position_error` …

A simpler **non-peer-join** R-MULTISTAGE — one parent + one child set:

```
# VRA = VRI × RES, where VRI is an avg over child rows of vendors
mcp__slayer__create_model(
  name="vendor_risk_amplification",
  query=[
    {
      "name": "vri_per_vendor",
      "source_model": "vendor_security_assessments",
      "dimensions": ["vendor_id"],
      "measures": [{"formula": "VendSecRate:avg", "name": "vri"}],
    },
    {
      "source_model": {
        "source_name": "vendors",
        "joins": [{"target_model": "vri_per_vendor",
                   "join_pairs": [["id", "vendor_id"]]}],
      },
      "dimensions": ["id"],
      "measures": [{"formula": "vri_per_vendor.vri:avg * RES:avg",
                    "name": "vra"}],
    },
  ],
)
mcp__slayer__edit_model(model_name="vendor_risk_amplification",
                        meta={…caller bookkeeping…})
```

**R-WINDOW — quartile / rank / percentile as a row dimension or filter.**

Available transforms in SLayer 0.5.0+: `rank`, `percent_rank`,
`dense_rank`, `ntile`, `first` (FIRST_VALUE asc / argmin),
`last` (FIRST_VALUE desc / argmax), `lag`, `lead`. The natural shape
is a ModelMeasure formula:

```
# EER ranked per application type, descending
mcp__slayer__edit_model(
  model_name="robot_per_robot",
  measures=[{
    "name": "eer_rank",
    "formula": "percent_rank(eer:max, partition_by=apptypeval)",
    "description": "EER percentile rank within application type",
    "meta": {…caller bookkeeping…},
  }],
)
```

The transform applies after the underlying aggregate (`eer:max`), so
it lives in the formula layer beside `rank` / `lag` / etc. Sort is
descending by default; pass a list to `partition_by=` for multi-key
partitions; cross-model dotted paths work
(`partition_by=actuation_data.robot_details.ctrltypeval`).

For non-standard window expressions not covered by these transforms,
fall back to a raw window in `Column.sql` inside a query-backed
model:

```
# TEI quartile as a row dimension on accounts
mcp__slayer__create_model(
  name="account_with_tei_quartile",
  query=[
    {
      "name": "tei_per_account",
      "source_model": "accounts",
      "dimensions": ["id"],
      "measures": [{"formula": "tei", "name": "tei_v"}],
    },
    {
      "source_model": {
        "source_name": "accounts",
        "joins": [{"target_model": "tei_per_account",
                   "join_pairs": [["id", "id"]]}],
        "columns": [{
          "name": "tei_quartile",
          "type": "number",
          "sql": "NTILE(4) OVER (ORDER BY tei_per_account.tei_v DESC)",
        }],
      },
      "dimensions": ["id", "tei_quartile"],
    },
  ],
)
```

**Argmin / argmax patterns** ("the value at the row with smallest /
largest X"). Three shapes, depending on grain and ordering column:

1. **Global argmin/argmax — the one row with smallest/largest X
   across the whole table.** Plain query-backed model with `order` +
   `limit=1`, no multistage needed:

   ```
   mcp__slayer__create_model(
     name="latest_volmeter",
     query=[{
       "source_model": "marketstats",
       "dimensions": ["volmeter"],
       "order": [{"column": "marketstatsmark", "direction": "desc"}],
       "limit": 1,
     }],
   )
   ```

2. **Per-group argmin/argmax over a *time* dimension.** Use the
   built-in `first` / `last` transforms — they emit FIRST_VALUE
   windows ordered by the model's time dimension:

   ```
   measures=[{"formula": "score:last", "name": "latest_score"}]
   # ↑ FIRST_VALUE(score) OVER (ORDER BY <time_dim> DESC ...)
   ```

3. **Per-group argmin/argmax over a *non-time* column** (the robot
   KB 5 case: faultpredscore at the row with smallest upkeepduedays
   per robot). The transforms don't accept arbitrary order columns,
   so use a two-stage multistage with the standard SQL argmin idiom
   — stage 1 = `:min` per group, stage 2 = self-join + filter on
   `column = stage1.min_value`:

   ```
   mcp__slayer__create_model(
     name="recent_fault_prediction_score",
     query=[
       {
         "name": "min_due_per_robot",
         "source_model": "maintenance_and_fault",
         "dimensions": ["upkeeprobot"],
         "measures": [{"formula": "upkeepduedays:min", "name": "min_due"}],
       },
       {
         "source_model": {
           "source_name": "maintenance_and_fault",
           "joins": [{"target_model": "min_due_per_robot",
                      "join_pairs": [["upkeeprobot", "upkeeprobot"]]}],
         },
         "dimensions": ["upkeeprobot"],
         "measures": [{"formula": "faultpredscore:max", "name": "rfps"}],
         "filters": ["upkeepduedays = min_due_per_robot.min_due"],
       },
     ],
   )
   ```

   The `:max` in stage 2 isn't really aggregating — the filter
   leaves one row per robot — it's just the colon-syntax handle for
   "extract the value". Ties on `upkeepduedays` resolve to the row
   with the largest faultpredscore (deterministic).

**R-EXISTS — ∃ child meeting predicate.**

```
mcp__slayer__create_model(
  name="sites_with_high_fidelity_mesh",
  query=[
    {
      "name": "hifi_mesh_count_per_site",
      "source_model": "meshes",
      "dimensions": ["site_id"],
      "measures": [{"formula": "*:count", "name": "n_hifi"}],
      "filters": ["fidelity_score >= 0.9"],
    },
    {
      "source_model": {
        "source_name": "sites",
        "joins": [{"target_model": "hifi_mesh_count_per_site",
                   "join_pairs": [["id", "site_id"]]}],
      },
      "dimensions": ["id"],
      "filters": ["hifi_mesh_count_per_site.n_hifi >= 1"],
    },
  ],
)
```

**R-VAR — time-anchor parameter.** For "rolling 30 days" against a
normal model, prefer the built-in window inside a plain query
(`{"formula": "revenue:sum(window='30d')"}`) — no new model needed.
Use this recipe only when the metric needs a parameterised anchor
date.

```
mcp__slayer__create_model(
  name="active_customers_recent",
  query=[
    {
      "source_model": "orders",
      "dimensions": ["customer_id"],
      "measures": [{"formula": "*:count", "name": "order_count"}],
      "filters": [
        "created_at > date('{as_of_date}', '-6 months')",
        "created_at <= date('{as_of_date}')",
      ],
    },
  ],
  variables={"as_of_date": "today"},
)
# at query time:
# mcp__slayer__query(source_model="active_customers_recent", variables={"as_of_date": "2025-04-01"})
```

### Pass 6 — host placement (R-HOST)

Place the encoded column / measure on the model whose PK matches the
predicate's grain. Cross-table refs come via dotted joined paths. Avoid
coarse predicates on fine-grained models — fan-out distorts aggregations.

## Working from a partially-encoded model

When some KB ids are already encoded (entities with `meta.kb_id` set
on a prior pass), don't blindly re-encode. Two passes:

1. **Verify existing encodings first.** Walk the KB; for each id,
   look up matching entities via
   `mcp__slayer__models_summary(datasource_name=<db>)` then
   `mcp__slayer__inspect_model(...)` on each model the KB
   references. For each entity carrying a `meta.kb_id` (or, on
   legacy entities not yet split, a `meta.kb_ids` list — see
   "Splitting multi-KB entities"), check the encoding against the
   KB's `definition` text and `type`:
   - If the formula / sql / case-when faithfully expresses the KB's
     definition: leave it alone.
   - If the encoding is wrong (incorrect formula, wrong host, missing
     filter): edit via `edit_model`. Keep the same `meta.kb_id`.
   - If the recipe was misapplied (e.g. an aggregation encoded as a
     plain column, or a metric that should be query-backed encoded
     as a row-level column): delete the offending entity and
     re-encode under the right recipe.

2. **Encode the rest.** Any KB id not yet covered: pick the recipe,
   apply it, stamp `meta.kb_id` (caller's domain bookkeeping
   dictates the exact key — see wrappers).

The verifier deduplicates ids — multiple entities tagged with the
same `meta.kb_id` is fine. What's not fine is the same id appearing
in **both** the encoded set AND the deferral notes file (see next
subsection).

## Splitting multi-KB entities

When a refresh workflow encounters an entity with `meta.kb_ids`
(plural; the deprecated form), pick the recipe whose pattern matches
and split it into N single-KB entities. The verifier still passes:
the KB ids stay accounted for, just under different homes.

| Recipe | Bucket | Pattern | Split shape |
|---|---|---|---|
| **R-SPLIT-CALC-THRESH** | A | one entity holds the calc *and* its threshold/classification | calc Column with one kb_id + CASE-WHEN sibling Column with the other kb_id |
| **R-SPLIT-ILLUSTRATION** | B | one JSON blob column described by N value_illustration KB entries about distinct sub-fields | one helper column per sub-field, each carrying its own kb_id; the JSON blob itself stays untagged (per kb_id placement rule) |
| **R-SPLIT-TRINITY** | C | illustration + calc + threshold all bundled (the robot pattern) | 3-way split: helper column + calc + classification, each with its own kb_id |
| **R-SPLIT-MULTI-FORMULA** | D | one model lumps several unrelated formulas | one entity per formula, each carrying its own kb_id |
| **R-SPLIT-MONSTER** | E | aggressive over-grouping (e.g. one model carrying 21 kb_ids) | aggressive split, target shape per-entity must come from the caller's instruction file |
| **R-SPLIT-DUP** | F | KB X is a verbatim restatement of KB Y | retain primary kb_id; document secondary in the per-DB notes file with `Status: not-applicable — duplicate of KB <primary>` |

For Buckets A–E the splits all produce single-KB entities (verifier
counts them all as encoded). For Bucket F the secondary id moves
from the entity into the notes file (the notes header counts as
documented; that's also a pass).

**Worked example — Bucket A (R-SPLIT-CALC-THRESH).** One Column
carrying both the EWR calc (KB 10) and the EWR classification (KB
50). After:

```
mcp__slayer__edit_model(
  model_name="equipment",
  columns=[
    {"name": "ewr_score",
     "type": "number",
     "sql": "<calc>",
     "meta": {"kb_id": 10}},
    {"name": "ewr_status",
     "type": "string",
     "sql": "CASE WHEN ewr_score > 0.7 AND ... THEN 'Ready' ELSE 'Not Ready' END",
     "meta": {"kb_id": 50}},
  ],
)
```

**Worked example — Bucket F (R-SPLIT-DUP).** One Column with
`meta.kb_ids=[4, 51]` where KB 4 and KB 51 state the same BFR
formula in different words. Smaller id wins; 4 stays on the entity,
51 moves to notes.

```
mcp__slayer__edit_model(
  model_name="signals",
  columns=[{"name": "bfr", "meta": {"kb_id": 4}}],
)
# slayer_models/_notes/<db>.md gets:
# ## KB 51 — <KB.knowledge>
#
# Reason: Verbatim restatement of KB 4; encoded entity is
# `signals.bfr` with `meta.kb_id = 4`.
#
# Status: not-applicable — duplicate of KB 4
```

For Bucket B / E, the caller's instruction file lists the per-kb_id
target entity (recipe + new name + host model) — the agent should
not invent the split shape from prose alone for these. For Bucket A
/ C / D the bucket label plus the KB texts are usually enough.

## Deferred-KB memory recording

When a KB id cannot be encoded after both passes above — the recipe
trigger needs information the schema does not carry, the prose
underspecifies thresholds, or the KB describes DML rather than
queryable shape — record it as a **SLayer memory** so the agent
recovers it via `search` / `recall_memories` at query time.

```python
mcp__slayer__save_memory(
    data_source=<db>,
    learning=<learning_body>,           # see template below
    linked_entities=[<canonical refs>], # ["<db>.<model>", "<db>.<model>.<leaf>", …]
)
```

`<learning_body>` is Markdown. The **first non-blank line is
load-bearing**: the verifier reads `KB <id> —` off it to attribute
the memory to a KB id. Use exactly an em-dash (`—`, U+2014), not a
hyphen.

```text
KB <id> — <KB.knowledge>

KB item (verbatim from <db>_kb.jsonl):
  id: <id>
  knowledge: "<knowledge>"
  description: "<description>"
  definition: "<definition>"
  type: <type>
  children_knowledge: [<ids>]

Reason: <one-paragraph schema-grounded explanation of why this KB
couldn't be encoded as a normal column/measure/aggregation>

Status: <one of the canonical statuses below>

Clarifying questions (REQUIRED when Status is AMBIGUOUS-PROSE or
SCHEMA-GAP; each is a single sentence the agent could pose to the
user at query time to disambiguate or to point at the missing data):
  Q1: <…>
  Q2: <…>
  …

Related KB ids:
  - KB <n> (<knowledge>) — <one-line note on the relationship>
  …

Relevant entities:
  - <db>.<model>[.<leaf>]
  …
```

`linked_entities` MUST be a non-empty list of canonical refs of the
form `<db>.<model>[.<leaf>]` so the verifier can attribute the
memory to a DB. At least one entry must start with `<db>.`.

Canonical `Status:` values:

| `Status:` value | Meaning |
|---|---|
| `deferred — AMBIGUOUS-PROSE` | KB underspecifies thresholds or labels; the predicate would be a guess. Encode helper columns so the agent can compose ad-hoc at query time. |
| `deferred — TIME-ANCHOR` | KB's predicate depends on a query-time anchor date ("rolling 30 days from `today`") and the model has no anchor. R-VAR may fix this; otherwise defer. |
| `deferred — DML` | KB describes an INSERT / UPDATE / DELETE workflow. Models describe queryable shape, not mutations. Permanently not-applicable. |
| `deferred — SCHEMA-GAP` | The data the KB names is not in the schema — verify by inspecting the schema and column meanings, including JSON subfields, before trusting this status. Many "schema gaps" turn out to be JSON sub-keys or differently-named columns. |
| `deferred — CROSS-DB` | KB requires data from a second datasource in a single query. SLayer doesn't support cross-datasource joins today. |
| `not-applicable` | KB is descriptive metadata (a column-value enumeration, a table-purpose narrative) rather than a metric — captured via `description` on the host column / model rather than as an encoded entity. |
| `not-applicable — duplicate of KB <primary>` | KB is a verbatim restatement of an already-encoded KB. The primary id stays on the entity; the secondary id is documented here. See R-SPLIT-DUP under "Splitting multi-KB entities". |

### Worked example — KB 16 "Service Support Score" (households)

KB 16's definition is "a weighted score that combines a household's
domestic-help service tier and its social-assistance participation
into one ordinal index", with the **weights left unspecified**. The
encoder builds helper columns (`domestic_help_score`,
`soc_support_score`) but cannot pick canonical weights, so KB 16
defers as `AMBIGUOUS-PROSE`:

```python
mcp__slayer__save_memory(
    data_source="households",
    learning=(
        "KB 16 — Service Support Score\n"
        "\n"
        "KB item (verbatim from households_kb.jsonl):\n"
        "  id: 16\n"
        "  knowledge: \"Service Support Score\"\n"
        "  description: \"Weighted ordinal score combining domestic-help "
        "tier and social-assistance participation.\"\n"
        "  definition: \"Score = w1*domestic_help_score + w2*soc_support_score "
        "(weights unspecified).\"\n"
        "  type: metric\n"
        "  children_knowledge: [30]\n"
        "\n"
        "Reason: KB defines the score as a weighted sum of two helper "
        "columns that ARE encoded (households.service_types.domestic_help_score "
        "and households.service_types.soc_support_score), but does not "
        "specify the weights w1 and w2. Picking weights silently would "
        "burn-in an arbitrary policy. Helpers are exposed so the agent "
        "can compose any specific weighting at query time.\n"
        "\n"
        "Status: deferred — AMBIGUOUS-PROSE\n"
        "\n"
        "Clarifying questions:\n"
        "  Q1: What weights should the Service Support Score use? "
        "(e.g. 1.0 and 1.0, or 0.7 and 0.3.)\n"
        "  Q2: Should households with NULL `domestichelp` count as 0, "
        "or be excluded from the score entirely?\n"
        "\n"
        "Related KB ids:\n"
        "  - KB 30 (Self-Sufficient Household) — uses Service Support "
        "Score's helpers under a different weighting scheme.\n"
        "\n"
        "Relevant entities:\n"
        "  - households.service_types.domestic_help_score\n"
        "  - households.service_types.soc_support_score\n"
        "  - households.service_types.is_supported_household\n"
    ),
    linked_entities=[
        "households.service_types.domestic_help_score",
        "households.service_types.soc_support_score",
        "households.service_types.is_supported_household",
    ],
)
```

If both passes leave zero ids deferred, no memories need to be
written — every KB is encoded, the verifier picks them up via
`meta.kb_id`.

## Constraints & gotchas

- **Idempotency**: `edit_model` upserts each list by name. Re-running
  the skill is safe and the recommended way to refine.
- **`create_model` collisions**: same name twice errors. Use a fresh
  name or `delete_model` first.
- **Naming**: `name` is `snake_case` ASCII (no `.`, no `__`). Display
  names go in `description` / `label`.
- **`meta` is opaque to this skill**: pass through whatever the caller
  supplies; this skill takes no position on its content.
- **Query-backed model namespace.** Auto-introspected columns and
  ModelMeasures share a single namespace per model. If the final
  stage exposes a `ratio` column, you can't add a `ratio` measure on
  the same model — name the measure `ratio_value` (or similar).
- **Bare ModelMeasure name on QB model queries.** When querying a
  query-backed model, `{"formula": "<measure_name>"}` errors with
  "use colon syntax" — bare-name resolution to a saved
  ModelMeasure's formula isn't honored on QB models today. Use the
  underlying expression with explicit colon syntax (e.g.
  `{"formula": "ratio:max"}`) or define the measure with the
  inline expression at query time. The saved ModelMeasure still
  carries `meta.kb_id` correctly for the verifier; this only
  affects ergonomic re-use at query time.
- **Verification**: after Pass 4 / 5, hit a representative query and
  confirm the natural agent expression is short — measure list +
  dimensions + filters, no per-call formula re-derivation.
