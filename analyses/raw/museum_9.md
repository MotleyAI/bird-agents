# museum_9 — failure trace + metadata audit

Run: `20260531t1013-claudes-slayer-48eb0f`. Duration 916s. submission_status=wrong_result.

## Task

TETL (Total Environmental Threat Level) top-10. 2 columns: artref, tetl.

TETL formula (KB 31): `TETL = ERF + LER + (MDR × 2)`. With KB 2 (ERF) + KB 8 (LER) + KB 7 (MDR) + KB 1 (sensitivity weights Low=1/Medium=5/High=10).

## Output diff

Gold top values: 215.74, 199.27, 198.93. Agent: 160245, 154577, 151439. **Off by ~1000×.**

## Agent's encoding — correct

`trajectory[32]` created `mdr_value`, `ler_value`, `tetl_value`. Submitted SQL contains `2.0 * (artageyears * (sens_sum/4.0) * (relhumidity-50)*(relhumidity-50) * tempc) / 100000.0` — POWER expanded, `/100000.0` present, ERF as sum/4.0. Sensitivity ratings not double-counted: ERF appears once inside MDR and once standalone per KB 7 + KB 31.

ThinkingBlocks signature-only; encoded SQL is faithful to KB.

## The 1000× gap — NOT a missing divisor

At `trajectory[215]` agent ran `tetl_no_lsw` (LER without `light_sens_w` term) and got 16K — clean 10× drop. Only multiplicative knob is sensitivity weight (max 10), not a missing divisor.

The gap comes from **join-path / row-cardinality semantics**, not formula:

- **Gold** (`livesqlbench_sqlite_gt_kg_testcases_0528.jsonl[175]`): MDR via `usagerecords.artrefused → showcase → environmentalreadingscore` with explicit `LIMIT 1`. TETL assembled in a `LEFT JOIN`ed CTE with `COALESCE(l.ler, 0) + COALESCE(m.mdr, 0)*2`. Most artifacts have null LER/MDR → TETL ≈ ERF (1-10), survivors hit ~200.
- **Agent**: TETL built on `conditionassessments → lightandradiationreadings → environmentalreadingscore` (path discovered around `trajectory[73-87]`). Every assessed row has all three components live; every `light_lux × visibleexplxh` term fires; per-assessment magnitudes blow up.

Agent saw the discrepancy and tried `tetl_lin` (210), `tetl_no_lsw` (215), `tetl_11` with normalized erf (224), `tetl_alt` (237), `tetl_abs` (198) — none changed join topology, all produced same scale.

## Metadata audit

**Was the canonical join path discoverable?**

`museum_column_meaning_base.json:148-150`:
- `ArtRefUsed`: *"A CHAR(10) NOT NULL foreign key referencing ArtifactsCore(ArtRegistry), indicating which artifact is being used."*
- `ShowcaseRefUsed`: *"A CHAR(12) foreign key referencing Showcases(ShowcaseReg) if a showcase is involved in the usage."* (**`if a showcase is involved`** actively WEAKENS the canonical claim)

Plain FKs, no "canonical link" / "current display" / "primary location" language.

`EnvironmentalReadingScore`:
- `EnvReadRegistry`: *"BIGSERIAL PRIMARY KEY..."* (no temporal hint)
- `ShowcaseRef`: *"...linking the reading to a specific showcase being monitored."*
- `ReadTimestamp`: *"TIMESTAMP NOT NULL indicating the date and time the reading was recorded."*

No "use the latest", "LIMIT 1", or "one reading per showcase". Gold's `LIMIT 1` (no ORDER BY) is itself non-deterministic — picks arbitrary row, not even latest by timestamp.

**Schema FKs:**

`museum_schema.txt`:
- `conditionassessments` (lines 88-100): FKs on `artrefexamined`, `showcaserefexamined`, `lightreadrefobserved → lightandradiationreadings(lightradregistry)`.
- `lightandradiationreadings` (lines 181-190): FK `envreadref → environmentalreadingscore(envreadregistry)`.
- `environmentalreadingscore` (lines 141-153): FK `showcaseref → showcases(showcasereg)`.
- `usagerecords` (lines 308-338): FKs `artrefused → artifactscore`, `showcaserefused → showcases`.

**BOTH paths first-class FK-declared.** Schema alone doesn't disambiguate.

**KB sweep for "showcase / current display / latest"**: KB 31 (TETL), 7 (MDR), 2 (ERF), 8 (LER), 1 (SensWeights) define formulas with `RelHumidity`, `TempC` etc. as bare variable names — no table-qualification. No KB names a canonical artifact↔env-reading link. KB 15 mentions "current display duration" but doesn't anchor a join path.

**Agent's exploration**:
- `trajectory[3,7,8,9]`: KB searches for TETL/LER/MDR/SensWeights.
- `trajectory[19]`: `inspect_model('conditionassessments', sections=['columns','joins','samples'])`.
- `trajectory[20]`: `inspect_model('sensitivitydata', …)`.
- `trajectory[76]`: `inspect_model('artifactscore', …)`.
- `trajectory[131]`: `search('artifact environmental reading join showcase usage')` — explicit join-path semantic search.
- `trajectory[145]`: `inspect_model('usagerecords', sections=['columns','joins'], num_rows=3)` — agent DID inspect UsageRecords.
- `trajectory[148-158]`: edits adding env→light join, trimming `showcaseref` on showcases/env/usagerecords, adding showcases→env join.
- `trajectory[159]`: defines `tetl_usage` on `usagerecords` but the formula references `showcases__environmentalreadingscore__lightandradiationreadings.*` — env reached via LIGHT path on the CA side, not `usagerecords.showcaserefused → showcases → env`.

Agent saw `usagerecords.showcaserefused`'s description, materialised it into the source model, but the column description didn't carry a "use me for MDR" signal, and the formula was already wired through `conditionassessments`. Never switched paths.

## Verdict

**KB-spec gap.** The resolving information is genuinely absent from every metadata source:
- DDL: both paths FK-declared; no UNIQUE constraint disambiguates.
- Column descriptions: `ShowcaseRefUsed` is *weaker* than what gold needs ("if a showcase is involved").
- KB: no entry names a canonical artifact↔env join.
- Gold's `LIMIT 1` is itself non-deterministic (no ORDER BY) — even the gold doesn't take "the latest" reliably.

Agent explored UsageRecords, searched explicitly for `"artifact environmental reading join showcase usage"`, saw the same FK graph as the gold author. No metadata cue prescribed usagerecords-via-showcase over conditionassessments-via-light.

Remediation: KB-spec addendum naming the canonical env-reading join for MDR purposes (e.g., "route via `UsageRecords.ShowcaseRefUsed`, take latest by `ReadTimestamp`").

Provenance: trajectory items 32, 88, 210, 215, 218, 240, 243; `museum_kb.jsonl` (ids 1, 2, 7, 8, 31); `museum_schema.txt:88-98, 181-192, 308-336`; gold sidecar line 175 (mdr_calc uses `usagerecords … LIMIT 1`).
