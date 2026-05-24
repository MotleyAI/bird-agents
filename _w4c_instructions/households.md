# W4c: households

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- The schema is small and most non-composite KBs map to row-level
  helper columns on the `households`, `properties`, `infrastructure`,
  `transportation_assets`, `amenities`, `service_types`, or
  `socioeconomic` (JSON-bearing) tables — many encode via R-CASE or
  R-COL.
- JSON columns (`households.socioeconomic`, `properties.dwelling_specs`,
  `transportation_assets.vehicleinventory`) hold values referenced by
  multiple KBs as `value_illustration`-type entries; encode helper
  columns via `json_extract(...)` and stamp `meta.kb_id` (or
  `meta.kb_ids` when one column covers multiple KBs).
- Some KBs name a *concept* without pinning numbers (e.g. "high
  vehicle ownership", "crowded household"). If the KB's `definition`
  truly underspecifies thresholds, defer as `AMBIGUOUS-PROSE` and
  encode the helper columns the agent will compose ad-hoc at query
  time. **Verify the KB text itself** — the prior notes pass got at
  least one wrong (a KB labelled "DML" was actually a SELECT
  predicate; a KB labelled "weights unspecified" had them in the
  definition). Don't trust the prior notes file; trust the
  `definition` field.
- Several KBs cascade off other KBs (e.g. "high mobility urban
  household" depends on "urban household"). Encode parents first.

### Specific KBs to (re-)encode this round

These were deferred in the prior round but the dataset operationalizes
them with concrete values you can encode against. Treat the
operationalization values below as facts about the dataset (verify
against the schema and column-meanings JSON).

- **KB 19 — Socioeconomic Index (SEI).** KB describes "weighted sum
  of income score, expenditure ratio, tenure score". Dataset uses
  `0.4 · income_score + 0.4 · (1 − expend_coeff) + 0.2 · tenure_score`.
  Income score from `households.socioeconomic.Income_Bracket`:
  `'Has no income'→1`, `'More than R$ 880 and less'→2`,
  `'More than R$ 880 and less than R$ 1,760'→3`,
  `'More than R$ 1,760 and less than R$ 2,640'→4`,
  `'More than R$ 2,640 and less than R$ 4,400'→5`,
  `'More than R$ 4,400'→6`. Tenure score from `Tenure_Type`:
  `OWNED→3, RENTED→1, else 0`. Expenditure ratio from
  `Expend_Coeff` (already a 0-1 fraction; cast `REAL` after
  replacing comma decimal separator). Encode as helper columns
  (R-CASE) on `households` plus a `ModelMeasure` for SEI; or as a
  query-backed model.

- **KB 26 — Modern Dwelling.** KB describes "specific Dwelling Type
  and active Cable TV Status". Dataset operationalizes the dwelling
  type as `LOWER(properties.dwelling_specs->>'Dwelling_Class') IN ('brickwork house', 'apartment')`
  and active cable as `LOWER(amenities.cablestatus) IN ('avail', 'available', 'yes')`.
  Encode as a row-level boolean column on `properties` (joined to
  `amenities` via the existing FK chain).

- **KB 42 — Economically Independent Household.** KB describes
  "high Income Classification and social support status of 'No'".
  "High Income Classification" maps to the top-3 brackets of KB 2's
  ordering (income score ≥ 4, i.e. one of `'More than R$ 1,760 and less than R$ 2,640'`,
  `'More than R$ 2,640 and less than R$ 4,400'`, `'More than R$ 4,400'`).
  "Social support status of 'No'" is `service_types.socsupport = 'No'`.
  Encode as a row-level boolean column on `households` joined to
  `service_types` (or as a query-backed model if the join graph
  needs it).

- **KB 32 stays deferred** as `deferred — DML`. The KB definition
  literally says "Insert a new record". One benchmark task does
  reference KB 32 in `external_knowledge`, but the predicate that
  task evaluates is KB 42's territory, not KB 32's; treat that as
  benchmark mis-tagging, not a SLayer encoding gap.
