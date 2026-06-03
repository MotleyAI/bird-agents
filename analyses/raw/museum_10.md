# museum_10 — failure trace

Run: `20260531t1013-claudes-slayer-48eb0f`. Duration 872s. submission_status=wrong_result.

This is the most concerning failure of the batch — but the verdict is unexpected.

## User query (verbatim from `livesqlbench_data_sqlite.jsonl` line 167)

> "Generate a rotation schedule using the standard ERPS formula. Show me each artifact's ID, name, material type, current display duration, their Display Safety Duration (DSD) limit, the calculated ERPS and a clear recommendation ('Immediate Rotation' or 'Monitor') based on ERPS Decision Threshold. Only include artifacts currently marked as 'Active' in the rotation schedule."

The prompt **explicitly names**: DSD, ERPS, "recommendation", "Immediate Rotation"/"Monitor", "ERPS Decision Threshold", and the `Active` filter.

## Gold's external_knowledge

`external_knowledge: [4, 38, 52]`:
- **KB 4 (DSD)**: `DSD = BaseDuration × (10 − LightSensWeight)(10 − TempSensWeight)(10 − HumidSensWeight) / 1000`
- **KB 38 (ERPS)**: `ERPS = (DSD − DisplayDurMonths) × (LER + 1) × (CPI + 1) ÷ 100`
- **KB 52 (ERPS Decision Threshold)**: `When ERPS < 0, trigger 'Immediate Rotation'; otherwise 'Monitor'`

These match the prompt 1:1.

## Agent's KB retrieval

`get_all_external_knowledge_names` at `trajectory[3]`, then targeted:

- `trajectory[6]` ERPS, `[8]` ERPS Decision Threshold, `[10]` DSD — **exactly the three gold KBs.**
- `[12]` Sensitivity Weight Values, `[14]` LER, `[16]` CPI — the three sub-formulas DSD/ERPS transitively need (KB 1, 8, 0). Required, not scope creep.
- `[21]` `ArtifactsCore.ConserveStatus`, `[23]` `SensitivityData.LightSensitivity`, `[25]` `Exhibition Rotation Candidate` — column/concept clarifications.
- `[194]` bulk `get_all_knowledge_definitions` late while debugging.

**No spurious KB fetched.** No AVS, MDR, TETL, MAP, ECI, etc.

## Agent's encoding sequence

ThinkingBlocks redacted (`thinking=''` at `[1]`). Action sequence is unambiguous:

After pulling ERPS → ERPS Decision Threshold → DSD at `[6,8,10]`, agent wires those weights into SLayer via `edit_model`:
- `[37]` `light_sens_weight`/`temp_sens_weight`/`humid_sens_weight` CASE expressions (matches KB 1)
- `[39]` `conserve_status_num` for CPI's `ConserveStatus` factor
- `[40]` `light_lux_x_visible_exp` for LER

Final submitted SQL projects exactly: `artifact_id, name, mattype, displaydurmonths, DSD, ERPS, CASE WHEN ERPS<0 THEN 'Immediate Rotation' ELSE 'Monitor' END`. Filtered on `displayrotatesched = 'active'`. Verbatim the prompt's column list.

## Gold sidecar contents

`livesqlbench_sqlite_gt_kg_testcases_0528.jsonl` line 167 — `sol_sql`:
- Returns 4 columns: `ArtRegistry, ArtName, ERF, high_sensitivities`
- 323 rows
- Filters: ERF > 7
- Has nothing to do with DSD/ERPS/decision-threshold
- Doesn't mention "Active", "Immediate Rotation", or "Monitor"

**This is the same query shape as museum_2's gold** (ERF + high_sensitivities). Looks copy-pasted.

## Verdict

**Not the agent's fault — gold-file bug.** The agent solved the prompt correctly with the gold's own listed `external_knowledge` (KBs 4/38/52). The `sol_sql` is mismatched against both the prompt AND the `external_knowledge` field. Either the prompt and SQL were swapped at gold-build time, or the row's `sol_sql` was copy-pasted from museum_2 and only `external_knowledge` got updated.

872s wallclock reflects real SLayer model construction for a legitimately hard 7-column compound calculation — not confusion.

Remediation target: gold sidecar row 167 — replace `sol_sql` with a DSD+ERPS+recommendation query, OR replace the prompt to match the ERF query.

Provenance: `results/cloud/20260531t1013-claudes-slayer-48eb0f/rows/museum_10/attempt-1.json`; `livesqlbench_data_sqlite.jsonl` line 167; `livesqlbench_sqlite_gt_kg_testcases_0528.jsonl` line 167 (the buggy row); `museum_kb.jsonl` (KBs 4, 38, 52).
