# households_12 — failure trace + metadata audit

Run: `20260531t1008-claudes-slayer-890419`. Duration 1238s. submission_status=wrong_result.

audit_status: `edited`.

## User query + masked terms

> "Which dwelling type has the highest average prosperity score among independent households, and how many vehicles do those households own?"

Masked term resolutions:
- `independent households` → `(s.serviceref IS NULL OR (LOWER(s.domestichelp)='no domestic workers' AND s.socsupport='No')) AND total_vehicles > 1`
- `prosperity score` → `AVG(residentcount * income_score)` where `income_score` is the **ordinal rank over the data's actual R$ brackets** (0..11). Original `sol_sql` used `'low income'..'very high income'` labels — which appear **zero times** in the data.
- `dwelling type` (NOT masked) → canonical-class CASE bucket.

## Auditor's view

`audit_status=edited`, `reasoning_summary`:
> "income_score = ordinal rank over the data's actual R$ brackets (the gold's 'low income' labels appear 0 times in the data). Dwelling type grouped by canonical category (synonyms folded; 'other' catch-all excluded as a non-type). Output (dwelling_class, SUM total_vehicles)."

Audit hardcoded a folding map: `('brickwork house','brick house') → 'brickwork house'`, `('apartment','apt') → 'apartment'`, `('condominium')`, `'shack%' → 'shack'`, `('unfinished brickwork','unfinished') → 'unfinished brickwork'`, else `'other'`. Rewrote noise-normalisation to a 4-pass `REPLACE(REPLACE(REPLACE(REPLACE(x, CHAR(9), ' '), ' ', '<>'), '><', ''), '<>', ' '))` (collapses tabs + runs of internal spaces).

## ask_user

6 calls. User-sim disambiguated:
- `[20]` (prosperity): *"residents × income score; brackets mapped to numeric values ('has no income'=0, 'R$ 440 or less'=1, … 'more than R$ 26,400'=11)"* — **exact gold formula given.**
- `[34]` (vehicles + indep): *"total = SUM(Auto+Bike+Motor) across households of that dwelling type … don't have a specific cutoff for high income — just use all households that qualify as independent based on the service criteria."* — User-sim **refused to disclose** the `domestichelp='no domestic workers'` and `total_vehicles>1` parts of the mask.
- `[106]/[147]/[233]`: *"'Brickwork house' and 'Brick house' should be treated as the same category … dwelling_class string should be 'brickwork house' (lowercase) … exclude 'Other'/NULL."*
- `[180]` (asking for the expected number): *"Sorry, I don't have that specific information."*

## Submitted vs audited gold

Final submitted SQL did fold case/whitespace and merged `brickwork`/`brick house` (LIKE `'%brick%'`). Diverged on the independent-household predicate:

- **Agent WHERE**: `CASE WHEN LOWER(TRIM(service_types.socsupport))='no' THEN 1 ELSE 0 END = 1` and `dwelling_class <> 'other'`. **No `domestichelp='no domestic workers'`**, **no `serviceref IS NULL` fallback**, **no `total_vehicles > 1` filter**.
- **Gold WHERE**: all three present.

Predicted `('brickwork house', 2106)` vs gold `('brickwork house', 1373)` — same dwelling class (alias merge worked), but vehicle SUM 53% high because agent included (a) single-vehicle households and (b) households with `domestichelp` other than `'no domestic workers'`.

## Decision-point lock-in

`[34]` — agent explicitly asked whether to use a high-income cutoff; user-sim said "no specific cutoff, just use the service criteria." Agent took "service criteria" to mean `socsupport='No'` only. Never tested adding `domestichelp` or `total_vehicles>1` floor. Asked again at `[180]/[232]` for expected count, got refused, re-submitted same vehicle filter.

## Metadata audit

**KB 42 ("Economically Independent Household")**: *"high Income Classification and social support status of 'No'."* — no `domestichelp` mention, no `total_vehicles>1`, and "high Income" itself is broken (KB 2 labels don't exist in data).

KB 14 (Vehicle Ownership Index = Auto+Bike+Motor) and KB 18 (Mobility Score) covered the vehicle math. KB 32 (Register New Household) irrelevant.

`households_column_meaning_base.json` for `properties.dwelling_specs.Dwelling_Class`: *"Ex. Brickwork house, Apartment"* — no alias list, no mention of `Brick house`/`Shack`/`Condominium`/`Unfinished brickwork`.

**SLayer model `properties.yaml` DID pre-bake `dwelling_type_clean`** with sampled values `Brickwork house, Apartment, BRICKWORK HOUSE, brick house, BRICK HOUSE, "brickwork\thouse", Unfinished brickwork, Shack` enumerated — and a `dwelling_type_score` CASE. The alias problem was visible in the slayer cache.

**Crucially, no KB defines `total_vehicles > 1`** as part of "independent" and **no KB mentions `domestichelp='no domestic workers'`**. These come only from the masked `sql_snippet` in `critical_ambiguity`.

## Verdict

**Gold-spec gap (primary) + user-sim under-disclosure (compounding).**

- `total_vehicles > 1` and `domestichelp = 'no domestic workers'` are part of the mask but absent from KB 42, column metadata, and any KB hint. They survive only inside `critical_ambiguity.sql_snippet` which the agent never sees.
- User-sim, holding that snippet, declined to disclose either when directly asked.
- Agent did the dwelling-alias normalisation correctly (KB-encoded in slayer). Not primarily a noisy-data issue.

Remediation: either expand KB 42 to spell out the full independent-household predicate, or instruct the user-sim to volunteer every clause of `critical_ambiguity.sql_snippet` when asked an open-ended "what exactly counts as X" question.

Provenance: trajectory items 20, 34, 106, 147, 180, 233; `households_kb.jsonl#42`; `slayer_models_otf/households/models/households/properties.yaml` (dwelling_type_clean pre-baked).
