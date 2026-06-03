# museum_5 — failure trace

Run: `20260531t1013-claudes-slayer-48eb0f`. Duration 387s. submission_status=wrong_result.

## Task

Show artifacts with ID, name, current temperature, current humidity, high sensitivity count, and a Yes/No flag for "Accelerated Deterioration". 6 columns in gold.

KB 7 (MDR): `MDR = (ArtAgeYears × ERF × (RelHumidity−50)² × TempC) / 100000`. KB 7 also requires "latest reading per showcase" → `MAX(EnvReadRegistry) GROUP BY ShowcaseRef`.

## Output diff

Gold returns 6 columns; agent returns 7 (added a raw `mdr` value column). But the dominant failure is values, not column shape:

- Predicted sample: `tempc=20, relhumidity=50, mdr=0.0` for every row.
- Gold sample: `tempc=21, relhumidity=53`.

Agent used an arbitrary env reading. `relhumidity=50` everywhere zeroes `(rh-50)²` → MDR=0 → flag='No' for every row.

## Trajectory trace

ThinkingBlocks all signature-only (empty `thinking=''`). Agent went straight from `get_knowledge_definition` lookups (trajectories[6,11,13,15]) to model encoding via `edit_model` at trajectories[28,30,36,44,65,67,84]. No explicit assistant text enumerating expected output columns.

First `slayer__query` at `trajectory[33]`:
> `dimensions: ['artrefused', 'artifactscore.artname', 'artifactscore.artageyears', 'sensitivitydata.erf', 'sensitivitydata.high_sens_count', 'environmentalreadingscore.tempc', 'environmentalreadingscore.relhumidity']` — 7 entries, pulling `artageyears` and `erf` for verification.

Submit attempts:
- `trajectory[41, 46, 62, 69]` — 4 submissions with **6 columns** (matching prompt count). All failed `ex_base returned 0 but expected 1`.
- `trajectory[89]` — FINAL submit ADDED `"mdr"` first in dimensions list, producing the 7-column predicted_result.

Final assistant text:
> "Despite multiple attempts varying the encoding (ERF over 4 vs 11 sensitivities, high_sens_count scope, filtered vs unfiltered, trimmed artifact IDs, **including MDR as a numeric column**), the submission keeps failing…"

Agent was misreading `ex_base=0` as a hint to add columns. The real problem in the first 4 attempts wasn't column shape (6-col submissions had right shape) — values diverged because the agent used the wrong env reading.

## Slayer's role

Slayer's `submit_query` requires explicit dimensions — does NOT default to returning all measures. The 6-column submissions at items 41/46/62/69 prove the agent had full projection control. `mdr` extra column is the agent's choice, not a slayer side effect.

Slayer DID contribute a separate noise: namespaced column names (`usagerecords.artrefused`, `usagerecords.artifactscore.artname` etc.) vs gold's bare `ArtRegistry`, `ArtName`. If `ex_base` is column-name-aware this contributes; row_count=951 matching for both suggests `ex_base` is value-based and namespace clutter is secondary.

## Verdict

**Agent miss.** The model never decoded KB 7's "latest reading per showcase" (`MAX(EnvReadRegistry) GROUP BY ShowcaseRef` semantics). Never inspected its own zero-everywhere `mdr` output. Reacted to repeated `ex_base=0` by adding a column instead of debugging values. Slayer's namespaced names a contributing irritant, not primary cause.

Failure class: agent miss (KB-coverage discipline + self-check on suspicious values absent).

Remediation: prompt nudge — "always sanity-check that computed columns aren't degenerate (all-zero, all-null) before submitting". Also: KB 7's "latest" language could be more explicit about the join idiom.

Provenance: `results/cloud/20260531t1013-claudes-slayer-48eb0f/rows/museum_5/attempt-1.json`; `museum_kb.jsonl` (KB 7).
