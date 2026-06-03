# museum_3 — failure trace

Run: `20260531t1013-claudes-slayer-48eb0f`. Duration 340s. submission_status=wrong_result.

## Task

Rank all artifacts by Conservation Priority Index (CPI). 4 columns: ArtRegistry, ArtName, ArtDynasty, CPI. Prompt: "rank by CPI" — no tiebreaker, no NULLS placement specified.

external_knowledge: `[]` in task row; KB 0 (CPI formula) lives in the sidecar.

## Output diff

Both queries return 951 rows. Row 1 in both: `ART58660 / Health Painting / Ming / 34.2` (unique max). At the four-way `32.7` tie:

- pred: `ART11605, ART32997, ART36966, ART42439` (alphabetical by `artref` — accident of agent's CTE)
- gold: `ART32997, ART42439, ART36966, ART11605` (no apparent secondary key)

`ex_base` is positional row-comparison, so the tie-cluster reorder flips it to `wrong_result`.

## Trajectory trace — CPI math

`trajectory[14]` `edit_model artifactscore.conserve_status_num`:
- mapping `LOWER(TRIM(conservestatus)) WHEN 'excellent' THEN 1 WHEN 'good' THEN 3 WHEN 'fair' THEN 5 WHEN 'poor' THEN 7 WHEN 'critical' THEN 10 END`
- description `"[kb=0] Numeric mapping … Excellent=1, Good=3, Fair=5, Poor=7, Critical=10"`

`trajectory[15]` `edit_model artifactratings.cpi_score`:
- `((histsignrating + researchvalrating + cultscore) * (10 - artifactscore.conserve_status_num)) / 30.0`
- description `"[kb=0] Conservation Priority Index = (HistSignRating + ResearchValRating + CultScore) * (10 - ConserveStatusNum) / 30"`

KB 0 text: `CPI = (H+R+C) * (10 - ConserveStatus) / 30`. Gold uses `Excellent=9, Good=7, Fair=5, Poor=3, Critical=0` directly; agent uses `1/3/5/7/10` then `(10-x)`. **Arithmetically identical**. Verified by row-level CPI values matching.

Agent later cast to REAL (item 32) and ROUND(2) (item 58) chasing a non-existent numeric mismatch.

## Did the agent think about ordering?

**No — except DESC/ASC direction.** Across 25+ tool calls, every `order` spec uses single-key `cpi_score`. No secondary sort, no NULLS FIRST/LAST, no RANK() tiebreak (one experiment at item 72 used `RANK() OVER (ORDER BY cpi_score DESC)` but still single key, never `(ORDER BY cpi DESC, artref ASC)`).

ThinkingBlocks along this path all have `thinking=''` (encrypted/stripped). Tool-call sequence shows zero ordering-stability deliberation: item 29 sanity-checked direction, item 49 counted ConserveStatus categories (suspecting mapping bug), item 63 queried min/max of ratings (suspecting numeric bug), item 66 counted rows (suspecting JOIN bug). Every failure diagnosed as a math/value problem.

## Does the prompt disambiguate ordering?

**No.** Query: *"Calculate and rank all artifacts by their Conservation Priority Index (CPI) to identify which need urgent attention. The report must include Artifact ID, name, its historical Dynasty and CPI Score."*

No tiebreaker, no NULLS placement, no deterministic secondary order. Gold's own `ORDER BY CPI DESC NULLS FIRST` (no secondary key) is itself plan-dependent at ties.

## Verdict

**Evaluator issue, not agent.** The prompt is ambiguous and the gold's order is plan-dependent at ties. `ex_base` should tolerate row reorderings within equal-CPI clusters, OR the gold needs a deterministic secondary sort. Demanding tiebreaker discipline from the agent here is unreasonable.

Remediation target: grader (tolerate float-tie reorderings) OR gold (`, ArtRegistry` secondary key).

Provenance: `results/cloud/20260531t1013-claudes-slayer-48eb0f/rows/museum_3/attempt-1.json` (items 14, 15, 18, 23, 29, 34, 46, 49, 60, 63, 66, 78); `livesqlbench_data_sqlite.jsonl#museum_3` (`external_knowledge: []`); `museum_kb.jsonl#0`; gold sidecar (`ORDER BY CPI DESC NULLS FIRST`).
