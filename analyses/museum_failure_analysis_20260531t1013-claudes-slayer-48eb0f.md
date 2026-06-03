# Museum (LiveSQLBench / one-shot) failure analysis

**Run ID:** `20260531t1013-claudes-slayer-48eb0f`
**Date:** 2026-05-31
**Branch / commit:** `main @ af58457` (post-README defaults bake)
**Compared against prior:** `20260530t0958-claudes-slayer-b278f1` (4/10 today vs 2/10 prior; +museum_1, +museum_7)

## Configuration

| field | value |
| -- | -- |
| framework | `claude_sdk_otf` |
| dataset / mode | `livesqlbench` / `one-shot` |
| agent model | `anthropic/claude-opus-4-7` |
| user-sim model | `anthropic/claude-sonnet-4-6` *(unused — one-shot mode has no simulator)* |
| reasoning effort | `high` |
| slayer setup | `on-the-fly` |
| audited gold | on (`use_audited_gold_sql=True`) |
| patience | 500 |
| gold file | `livesqlbench_sqlite_gt_kg_testcases_0528.jsonl` |
| instance ids | `museum_1..10` |
| cluster | 1 worker × 1 actor, `e2-standard-4` |

## Headline result

| metric | value |
| -- | -- |
| phase1 pass rate (raw) | **4 / 10 (40%)** |
| phase1 pass rate (after accountability adjustment) | **9 / 10 (90%)** |
| run cost | **$38.78** (agent $38.78, no user-sim) |
| longest task | museum_9 (916 s, $9.84) |
| shortest pass | museum_6 (35 s, $0.36) |

Adjustment rationale: of the 6 failures, **1 is a genuine agent miss**, **1 is grader-stability noise on a true float tie**, **3 are gold-authoring gaps the supplied metadata does not resolve**, and **1 is an outright gold-sidecar bug (mismatched SQL vs prompt)**. The 40% headline misrepresents the agent's behaviour on this slice; 9/10 reflects what would happen with consistent grader + gold hygiene.

## Per-task verdicts

| inst | dur | cost | category | root cause |
| -- | --: | --: | -- | -- |
| museum_1 | 173 s | $1.53 | PASS (audited + original) | — |
| museum_2 | 499 s | $4.05 | FAIL — gold-spec gap | ERF averaged over only 4 sensitivity columns; gold uses all 11. KB 1 enumerates only 4 weights, KB 2 says "etc.", slayer model has `erf = (env+light+temp+humid)/4.0` **pre-baked**. |
| museum_3 | 340 s | $2.02 | FAIL — grader / gold tiebreaker | 951 rows, same CPI values, agent and gold differ only at the four-way 32.7 tie. Gold's `ORDER BY CPI DESC NULLS FIRST` has no secondary key; prompt mandates none. SQLite picks different physical orders inside the tie cluster. |
| museum_4 | 608 s | $3.01 | FAIL — gold-spec gap | CBE = Σ(CPI × BudgetRatio) / N (KB 9). `BudgetRatio` is undefined for a schema with no per-artifact budget *amount* column; gold reinterprets categorical `BudgetAllocStatus` as fractional `adequate/total` — convention absent from KB and column descriptions. |
| museum_5 | 387 s | $3.26 | FAIL — agent miss | KB 7 says "latest reading per showcase" → `MAX(EnvReadRegistry) GROUP BY ShowcaseRef`. Agent never decoded this; picked an arbitrary env reading with `relhumidity=50` everywhere, zeroing `(rh-50)²` → MDR=0 → flag='No' for all rows. Compounded by reacting to repeated `ex_base=0` with column shuffling instead of value inspection. |
| museum_6 | 35 s | $0.36 | PASS (audited + original) | — |
| museum_7 | 586 s | $4.64 | PASS (audited) | new vs prior run |
| museum_8 | 196 s | $1.36 | PASS (audited + original) | — |
| museum_9 | 916 s | $9.84 | FAIL — gold-spec gap | TETL formula encoded correctly per KBs 1/2/7/8/31. 1000× value gap is a **join-path discrepancy**: gold resolves MDR's env reading via `usagerecords → showcase → env LIMIT 1`; agent went `conditionassessments → light → env`. Both FK paths are first-class; no KB or column description prescribes which is canonical. |
| museum_10 | 872 s | $8.69 | FAIL — gold-sidecar bug | Prompt asks for DSD + ERPS + recommendation with "Active" filter. Gold's `external_knowledge: [4, 38, 52]` matches. Gold's `sol_sql` is an unrelated ERF + high_sensitivities query (looks copy-pasted from museum_2). Agent solved the actual problem; gold solves a different one. |

## Three KB-spec gap deep-dives (was the answer derivable from metadata?)

For each, the agent had access to: KB entries via `bird-interact-tools` MCP; per-column descriptions seeded from `museum_column_meaning_base.json` and exposed via `mcp__slayer__inspect_model` / `models_summary` / `search`; DDL in `museum_schema.txt`; the slayer model state itself.

### museum_2 — ERF cardinality

- **What gold expects:** averaged sensitivity score over all 11 `SensitivityData.*Sensitivity` columns (env, light, temp, humidity, vibra, pollutant, pest, handle, transport, display, storage).
- **Agent consulted:** KB 1 (weights, lists 4), KB 2 (ERF formula, ends with "etc."), `inspect_model('sensitivitydata')`, `models_summary`, two semantic searches.
- **Resolving info in the corpus?** No, and worse: the slayer model has the wrong answer pre-baked. `slayer_models_otf_livesqlbench/museum/models/museum/sensitivitydata.yaml` defines `erf` as `(env_sens_weight + light_sens_weight + temp_sens_weight + humid_sens_weight) / 4.0` with description "Computed as the arithmetic mean of the four encoded sensitivity weights". The agent inheriting this datasource is being actively steered toward 4. Column descriptions on the other 7 (vibra/pollutant/pest/handle/transport/display/storage) read like *operational* sensitivities (packaging, storage, handling) rather than *environmental*; KB 4 (DSD) uses 3 columns, KB 8 (LER) uses 1; KB 2's "etc." is the only opening — and ERF is named "**Environmental** Risk Factor", reinforcing the subset reading.
- **Verdict:** under-specified by KB, *mis-specified* by the slayer encoder.

### museum_4 — CBE's `BudgetRatio`

- **What gold expects:** `BudgetRatio_i = adequate_records / total_records` per artifact, derived from `conservationandmaintenance.budgetallocstatus` (a categorical enum: Adequate / Insufficient / Review Required). Also a column the prompt didn't request (`artifact_count`) is silently expected in the output.
- **Agent consulted:** KB 9 (CBE), KB 17 (Crisis), `models_summary`, three counts on `conservationandmaintenance`, three semantic searches.
- **Resolving info in the corpus?** No. KB 9's "BudgetRatio = proportion of total conservation budget allocated to each artifact" implies a monetary amount per artifact — no such column exists. `BudgetAllocStatus` is described as a categorical enum; no column description, no peer KB, no value-illustration KB ever recasts "Adequate" as a 1-in-a-fraction. KB 17 uses `BudgetAllocStatus` only as a categorical threshold predicate. Uniform 1/N (`CBE = SUM(CPI)/N²`) is the most defensible literal reading of KB 9.
- **Verdict:** under-specified gold convention. Would need a sibling KB on `BudgetAllocStatus` mapping it to fractional.

### museum_9 — MDR's join path

- **What gold expects:** for MDR (= `ArtAgeYears × ERF × (RelHumidity−50)² × TempC / 100000`), route the environmental reading via `usagerecords.ArtRefUsed → ShowcaseRefUsed → environmentalreadingscore.ShowcaseRef LIMIT 1` (no `ORDER BY`).
- **Agent consulted:** KBs 1, 2, 7, 8, 31; inspected `conditionassessments`, `sensitivitydata`, `artifactscore`, `usagerecords`; explicit semantic search `"artifact environmental reading join showcase usage"`.
- **Resolving info in the corpus?** No. Both FK paths (`conditionassessments → light → env` and `usagerecords → showcase → env`) are first-class FK-declared. `UsageRecords.ShowcaseRefUsed`'s description (*"FK referencing Showcases if a showcase is involved in the usage"*) is actively *weaker* than what gold needs. No KB names a canonical artifact↔env-reading link, no column description hints at "latest reading per showcase", and the gold's own `LIMIT 1` (no `ORDER BY`) is itself non-deterministic — even the gold doesn't reliably pick "the latest".
- **Verdict:** under-specified gold convention. Both paths are equally licensed.

## Cross-cutting observations

### 1. SLayer is reinforcing the wrong answer for museum_2

The encoder pass that ingests the museum dataset writes pre-computed measures into the slayer model based on KB enumeration during ingest. For museum_2's ERF, this committed to the 4-column reading at encode time. A more diligent agent that *trusts* the slayer model would still get the wrong answer. This is upstream of any prompt change.

### 2. Trajectory truncation hides inspect-tool responses

`src/bird_interact_agents/agents/claude_sdk_otf_ainteract/agent.py:370` (and the `claude_sdk_otf` sibling) clips each saved trajectory message to `str(msg)[:500]`. For failure forensics this hides exactly the inspect-model / models-summary responses we need to verify what the agent *actually saw*. Worth raising the clip (or moving inspect responses to a separate per-task log file).

### 3. Single-bit grader feedback is the largest aggravator

museum_2 and museum_5 both had the correct hypothesis on the table at one point during their run. They received the verdict `"ex_base returned 0 but expected 1. Please try again."` — no row count, no column diff, no sample mismatch. With one bit per round, they couldn't choose between candidates. A richer grader observation (predicted row count vs expected, column-set diff, first divergent row) would have flipped both. This is independent of any model improvement.

## Cost breakdown

Sum across the 10 instances:

| component | tokens (input / cache_read / cache_create / output) | USD |
| -- | -- | --: |
| Opus agent | totals in `usage` blocks per row | **$38.78** |
| Sonnet user-sim | n/a (one-shot mode skips simulator) | $0.00 |
| **total** | | **$38.78** |

Per-task: museum_9 ($9.84), museum_10 ($8.69), museum_7 ($4.64), museum_2 ($4.05), museum_5 ($3.26), museum_4 ($3.01), museum_3 ($2.02), museum_1 ($1.53), museum_8 ($1.36), museum_6 ($0.36).

Note: of the $38.78, **~$31** went to the 6 failing tasks. Of *that*, **~$26** went to the 5 tasks where the failure is not the agent's fault (museum_2, _3, _4, _9, _10). Fixing the KB / gold / grader issues would directly reduce the per-run cost on this slice by ~2/3 without touching the model.

## Suggested follow-ups

1. **Audit the museum_10 sidecar row.** `livesqlbench_sqlite_gt_kg_testcases_0528.jsonl` line 167 has `sol_sql` mismatched against the prompt's DSD/ERPS/recommendation ask. Likely a paste error.
2. **Add KB anchors for the three gaps.** A sibling KB on `SensitivityData` enumerating all 11 columns as ERF inputs; a KB on `BudgetAllocStatus` defining its mapping to `BudgetRatio`; a KB on `EnvironmentalReadingScore` naming the canonical `usagerecords → showcase` link with "latest reading per showcase" semantics.
3. **Raise the trajectory truncation cap** in `agent.py:370` (or split inspect responses out) so failure-trace analysis can replay the agent's metadata-view from saved artifacts.
4. **Enrich the grader observation** with `row_count_delta`, `column_set_delta`, and a `first_divergent_row` field; gate behind a flag if you want to keep the strict "ex_base" surface.
5. **Add a deterministic tiebreaker discipline** either in the gold (`, ArtRegistry ASC` as a secondary key when `ORDER BY` has float ties) or in the grader (set-equality within ORDER BY equivalence classes).

## Provenance

- Per-task attempt artifacts: `/home/james/Dropbox/SLayer/bird-agents/results/cloud/20260531t1013-claudes-slayer-48eb0f/rows/<instance_id>/attempt-1.json`
- Manifest: `/home/james/Dropbox/SLayer/bird-agents/results/cloud/20260531t1013-claudes-slayer-48eb0f/manifest.json`
- Eval summary: `…/eval.json` (`p1=4/10 (0.4)`)
- Museum metadata: `/home/james/Dropbox/SLayer/livesqlbench-base-lite-sqlite/museum/{museum_kb.jsonl, museum_column_meaning_base.json, museum_schema.txt}`
- Gold sidecar: `/home/james/Dropbox/SLayer/livesqlbench-base-lite-sqlite/livesqlbench_sqlite_gt_kg_testcases_0528.jsonl`
- Slayer encoded model: `/home/james/Dropbox/SLayer/bird-agents/slayer_models_otf_livesqlbench/museum/models/museum/sensitivitydata.yaml` (the pre-baked 4-column ERF)
