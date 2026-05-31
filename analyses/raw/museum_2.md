# museum_2 — failure trace + metadata audit

Run: `20260531t1013-claudes-slayer-48eb0f` (claude_sdk_otf, livesqlbench, one-shot, Opus, reasoning_effort=high). Duration 499s. submission_status=wrong_result.

## Task

amb_user_query: identify artifacts with high Environmental Risk Factor (ERF) — show ID, name, ERF score, and a JSON summary of all 'High' sensitivity ratings; threshold ERF > 7.

external_knowledge: `[1, 2]` (sensitivity weights, ERF formula).

## Trajectory trace

KB list discovery (`get_all_external_knowledge_names`) returned all 12 KB names at `trajectory[5]`. Agent pulled exactly KB 2 (ERF) and KB 1 (Sensitivity Weights) at `trajectory[6,8]`. CPI (KB 0) chased much later as a distraction at `trajectory[59]`, but didn't feed into submission.

Reasoning oscillation between two readings of ERF cardinality:

- `trajectory[17]` — first encoding uses 4 columns (env/light/temp/humidity), divisor 4.0 (matches KB 1's enumeration).
- `trajectory[27]` — after submit-1 fails, agent correctly diagnoses: *"The test failed. Let me reconsider — the KB says 'EnvSensitivity, LightSensitivity, TempSensitivity, etc' suggesting ALL sensitivities, not just 4."* Re-encodes with 11 sensitivities at `trajectory[32]`. Also fails.
- `trajectory[63]` — flips back: *"ERF uses the 4 named sensitivities (per KB1's explicit weight mapping), but the JSON summary lists ALL 11 sensitivity fields rated 'High'"*.

Final submission (submit-9) lands on 4-sensitivity encoding. Remaining attempts varied JSON whitespace and `TRIM(artref)` cosmetics. Predicted 169 rows; gold 100.

## Decision-point lock-in

`trajectory[63]`. After both 4-sens and 11-sens hypotheses failed `ex_base`, the agent received only `"ex_base returned 0 but expected 1. Please try again."` — no row count, no diff, no column delta. With single-bit feedback the agent concluded both must be partially right and started splitting hairs about JSON formatting and TRIM rather than re-examining the divisor.

## Metadata audit

Did the `museum_column_meaning_base.json` SensitivityData entries hint that ALL 11 columns are equally-weighted ERF inputs?

**No.** Each `museum|SensitivityData|*Sensitivity` entry follows the same template ("<TYPE> describing <axis>, possible values: Low/Medium/High"). The other 7 (Vibra/Pollutant/Pest/Handle/Transport/Display/Storage) are described operationally (packaging, storage, handling) rather than environmentally. Nothing labels any as ERF inputs. Nothing distinguishes the 4 "main" from the other 7.

Did the agent inspect SensitivityData via slayer? **Yes** — at `trajectory[14]` `inspect_model(model_name="sensitivitydata", sections=["columns"])`; and `models_summary(datasource_name="museum")` at `trajectory[10]`. Plus two `search` calls at `trajectory[35, 58]` with `"Environmental Risk Factor ERF sensitivities"` and `"sensitivity attributes list count weight averaging"`. Both surfaced only KBs 1 and 2 (already known).

**Critical finding** — the slayer-encoded model has the wrong answer **pre-baked**:

`slayer_models_otf_livesqlbench/museum/models/museum/sensitivitydata.yaml:311-326`:
```
erf:
  formula: (env_sens_weight + light_sens_weight + temp_sens_weight + humid_sens_weight) / 4.0
  description: "Computed as the arithmetic mean of the four encoded sensitivity weights (env, light, temp, humid)…"
```

The slayer encoder committed to the 4-column reading during ingestion (driven by KB 1's enumeration). The agent inheriting this datasource sees ERF defined on 4 columns; an agent that trusts the slayer model would get the wrong answer.

Any other KB pinning cardinality? **No.** All 10 sensitivity-mentioning KBs in `museum_kb.jsonl` (1, 2, 3, 4, 8, 14, 20, 23, 24, 53) name SUBSETS: KB 1 enumerates 4 weight mappings; KB 4 (DSD) uses 3; KB 8 (LER) uses 1; KB 14 references "SensitivityData values" generically; KB 20 uses 1; KB 23/24 are per-axis value illustrations. KB 2's "EnvSensitivity, LightSensitivity, TempSensitivity, etc" is the only "etc" — deliberately under-specified.

## Verdict

Genuinely underspecified by the metadata as shipped, and *mis-specified toward the 4-column answer*. The agent did consult the right sources. Every signal points to 4:

- KBs 1, 4, 8 use sensitivity subsets; KB 2's "etc." is the only opening, and lists those same 4 by name first.
- column_meaning makes the other 7 look like operational (packaging/storage/handling), not environmental.
- ERF is named "**Environmental** Risk Factor", reinforcing the subset reading.
- slayer model has erf=4-col pre-baked.

Failure class: **gold-spec gap, compounded by slayer encoder pre-baking the wrong answer** AND **single-bit grader feedback preventing disambiguation**.

Provenance: `results/cloud/20260531t1013-claudes-slayer-48eb0f/rows/museum_2/attempt-1.json` (trajectory items 5, 7, 9, 17, 27, 32, 63); `livesqlbench-base-lite-sqlite/museum/museum_kb.jsonl` (KBs 1, 2).
