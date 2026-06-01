# households_10 — failure trace + metadata audit + deep dive

Run: `20260531t1008-claudes-slayer-890419`. Duration 926s. submission_status=wrong_result.

audit_status: `edited`.

## User prompt

> "List the home IDs for all highly mobile homes that are also updated residences, with many vehicles."

`external_knowledge: [6, 7, 10, 14, 23, 26]`. Three masked terms in `critical_ambiguity` (held by user-sim):

- **"highly mobile homes"** → `vehicles>2 AND LOWER(Newest_Year) IN ('2005 to 2009', '2010 to 2013', '2012 to 2013', '2014 or newer')`
- **"updated residences"** → `LOWER(Dwelling_Class) IN ('brickwork house', 'apartment') AND LOWER(cablestatus) IN ('avail', 'available', 'yes')`
- **"many vehicles"** → `(Auto+Bike+Motor) > 2`

## The two golds

**Original gold** (`sol_sql`): returns the **single region with the most matching households**:
```sql
WITH TargetHouseholds AS (
  SELECT h.housenum, h.locregion FROM households h JOIN ... WHERE (vehicle_sum)>2
    AND LOWER(Newest_Year) IN ('2005 to 2009','2010 to 2013','2012 to 2013','2014 or newer')
    AND LOWER(Dwelling_Class) IN ('brickwork house','apartment')
    AND LOWER(a.cablestatus) IN ('avail','available','yes'))
SELECT locregion FROM TargetHouseholds GROUP BY locregion ORDER BY COUNT(*) DESC LIMIT 1;
```

Prompt asks for "home IDs" but original gold returns a region. Output-shape mismatch.

**Audited gold** (`audit_status="edited"`): switches to `SELECT DISTINCT h.housenum`, wraps every value comparison in a 4-pass REPLACE chain (`CHAR(9)→space → collapse runs of spaces`) plus CASE folding:

- `Newest_Year`: folds `'2014 or newer'`, `'2014+'`, `'after 2014'`, `'2014 and newer'`; `'2010 to 2013'`, `'2010-2013'`, `'10-13'`, `'2012 to 2013'`; `'2005 to 2009'`, `'2005-2009'`, `'05-09'`.
- `Dwelling_Class`: `IN ('brickwork house','brick house','apartment','apt')`.
- `cablestatus`: `IN ('avail','available','yes','y','have')`.

**Audit's `changes` block records** `clause_kind="manual_data_align_reaudit"`, `source="manual_dev1478"`, **`justified_by: []`**. Per `.claude/skills/_shared/audit-gold-sql.contract.md`, clauses with empty `justified_by` should be DEFERRED, not encoded. The audit violated its own contract.

## What KBs 6/7/10 actually say

```text
KB 6 Dwelling Type:      "Values based on schema include 'Brickwork house', 'Apartment', 'Condominium', etc."
KB 7 Cable TV Status:    "Values indicating availability, based on schema, are 'avail', 'available', and 'yes'."
KB 10 Vehicle Year Range:"Text ranges like '1995 to 1999', '2005 to 2009', or '2010 to 2013'."
```

KBs 6, 10 hedge (`"include … etc."`, `"like"`). KB 7 does not. None mention `'brick house'`, `'apt'`, `'y'`, `'have'`, `'2014+'`, `'after 2014'`, `'2010-2013'`, `'10-13'`, `'05-09'`.

## Slayer model exposes the noise

`slayer_models_otf/households/models/households/amenities.yaml` — `cablestatus.sampled_values` (123 distinct values; structured top-50):
```text
Available, Not available, available, AVAILABLE, yes, Y, not available, Yes, avail, have,
NOT AVAILABLE, unavailable, not avail, N, no, No, dont have, Y, YEs, YES, Avail, y, ...
```

`properties.yaml` — `dwelling_specs__Dwelling_Class.sampled_values` (116 distinct):
```text
Brickwork house, Apartment, BRICKWORK HOUSE, Brickwork House, brickwork house,
brick house, Brickwork house , apartment, Brickwork  house, APARTMENT,
Shack (with floor finish), apt, Apt, Unfinished brickwork, BRICK HOUSE, ...
```

`Y`/`y`/`have`/`brick house`/`apt`/`2014+` are all there.

## Did the agent look?

Yes. Trajectory items 37, 39, 40, 41: `inspect_model('amenities')`, `('transportation_assets')`, `('properties')`, `('households')`. Full response bodies truncated to 500 chars in saved trajectory (`agent.py:~370` clip), but agent saw them live. Agent then did distinct-value queries via `mcp__slayer__query` at items 95 (`dwelling_specs__Dwelling_Class like '%ondo%' or '%apart%'`), 136 (`Newest_Year like '2005%' or '2010%'`), 174 (cablestatus == 'Available' count), 176 (Dwelling_Class in canonical list count). Actively exploring noise space.

## Slayer search rendering — what EntityHit.text shows

Both motley-slayer 0.6.10 and 0.7.0 have identical `render_column_text`. It emits only the truncated `Sample values:` string (NOT the structured `sampled_values: List[str]`).

Empirical check on the truncated `Sample values:` for each filter column:

| column | EntityHit.text gives the agent |
| -- | -- |
| `amenities.cablestatus` | `Y`, `y`, `have`, `dont have`, `avail` — **all visible** in top-20 |
| `properties.dwelling_specs__Dwelling_Class` | `brick house`, `apt`, `Apt`, `BRICK HOUSE`, `Shack` — **all visible** in top-20 |
| `transportation_assets.vehicleinventory__Newest_Year` | `2014+`, `2010-2013`, `10-13`, `05-09`, `2005-2009` — visible; `after 2014`, `2014 and newer` — structured-only (hidden by truncation); `'2012 to 2013'` — **not in data at all**, original gold buggy on that literal |

## Did the agent use `search(entities=[...])`?

Looking at trajectory: **all 7 `mcp__slayer__search` calls used `question=`** (semantic free-text). **Zero `search(entities=[...])` calls.** The agent never used the focused per-column read primitive.

Reason: nothing in the `claude_sdk_otf_ainteract` prompt or any tool description points the agent at `search(entities=[...])` as canonical column-reading. The `_host_discovery_playbook.py:43-67` instruction is scoped to host/join discovery, not filter-value enrichment.

## What the agent submitted

```sql
WHERE (Auto+Bike+Motor) > 2
  AND LOWER(TRIM(Newest_Year))      IN ('2005 to 2009','2010 to 2013','2014 or newer')
  AND LOWER(TRIM(cablestatus))      IN ('available','avail','yes')
  AND LOWER(TRIM(Dwelling_Class))   IN ('brickwork house','apartment')
```

LOWER+TRIM handles case + leading/trailing whitespace but NOT internal whitespace and NOT synonyms.

## User-sim role

- `[35]`: listed only `'2005 to 2009', '2010 to 2013', '2014 or newer'`. Never volunteered synonyms. Dropped `'2012 to 2013'` from gold's mask (string doesn't exist in data anyway).
- `[105]`: refused — *"I don't have access to expected row counts or specific example values"*.
- `[117]`: canonical-only confirmation on case sensitivity question.

Sim not adversarial, but strictly literal. When asked for ground-truth row counts (the one signal that would have surfaced the gap), refused.

## Predicate row-count verification (against live `households.sqlite`)

| query | DISTINCT housenum |
| -- | --: |
| KB-faithful canonical literals (agent-style) | **153** |
| Synonym-expanded predicate (approximation of audit) | **223** |
| Audited gold (exact, including internal-whitespace collapse) | **233** |

Agent returned 153. Audit expected 233. 80-row gap: ~60 from `Y`/`y` cable, ~30 from `2014+`/`after 2014`/`2014 and newer`/`'10-13'`/`'2010-2013'`/`'05-09'`/`'2005-2009'` synonyms, ~30 from `brick house`/`apt` Dwelling_Class synonyms (overlap reduces union to 80).

## Revised verdict

Mixed:

1. **The answer IS derivable in principle.** Truncated `Sample values:` strings expose every variant the audit added (with two exceptions: `after 2014`, `2014 and newer`).
2. **KB hedges legitimately invite expansion** — KB 6 ends in "etc.", KB 10 says "like". KB 7 does not hedge, but `'y'`/`'have'` are still visible in sampled values.
3. **Agent inspected the right tables** and was actively exploring noise. Had the evidence. Settled on canonical-only reading after user-sim confirmed at idx 35.
4. **Audit DID violate its contract** (`justified_by: []` clauses should defer, not encode), picks one specific expansion set among many plausible.
5. **User-sim's literalness blocked recovery.** When agent asked for a sample housenum at idx 105, sim declined.

Closest framing: **agent miss, but only barely** — KB hedges, sim's literal answer, and data noise put four reasonable readings on the table (canonical-only, lower+trim, lower+trim+light-synonym, lower+trim+full-synonym-expansion). Audit picks one; agent picks another. Both internally consistent.

Prompt-level fixes that would close this:
1. Symmetric companion to rule-3 line 114-116: *"if sampled values show variants of KB-named literals, extend the IN-set."*
2. Promote `search(entities=[...])` from host-discovery section to general column-reading guidance.

Provenance: trajectory items 35, 37, 39, 40, 41, 95, 105, 117, 136, 174, 176; `slayer_models_otf/households/models/households/{amenities.yaml, properties.yaml, transportation_assets.yaml}#sampled,sampled_values`; `households_kb.jsonl#6,7,10`; audited gold (`audit_status: edited`, `changes[0].justified_by: []`).
