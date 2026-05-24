# W4d: fake

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### account.accindex  (column)

- Current `meta.kb_ids`: `[81, 83]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 81 [calculation_knowledge] "member count"
    def: Calculated using COUNT(DISTINCT account_index) for accounts grouped by the cluster identifier.
  - KB 83 [calculation_knowledge] "member account IDs"
    def: \{ \text{accindex}_a \mid a \in cluster C \}

### moderationaction.coordscore  (column)

- Current `meta.kb_ids`: `[21, 82]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 21 [value_illustration] "moderationaction.coordscore"
    def: Ranges from 0 to 1. Scores above 0.7 strongly indicate coordinated behavior, while scores below 0.2 suggest independent actions.
  - KB 82 [calculation_knowledge] "maximum coordination score"
    def: \max_{a \in cluster C} (\text{coordscore}_a)

### securitydetection.detection_score_profile  (column)

- Current `meta.kb_ids`: `[20, 29]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 20 [value_illustration] "detection_score_profile.overall.confval"
    def: Ranges from 0 to 1. Values above 0.8 indicate high-confidence detections, while values below 0.3 suggest uncertain results requiring manual review.
  - KB 29 [value_illustration] "detection_score_profile.behavior_scores.botlikscore"
    def: Ranges from 0 to 100. Scores above 70 strongly indicate bot behavior, while scores below 20 suggest human-like behavior patterns.

### tei_quartile_per_account.tei_quartile_per_account  (model)

- Current `meta.kb_ids`: `[70, 79]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 70 [calculation_knowledge] "TEI quartile"
    def: Q_{TEI} = egin{cases} 1 & 	ext{if TEI} in [0, P_{25}] \ 2 & 	ext{if TEI} in (P_{25}, P_{50}] \ 3 & 	ext{if TEI} in (P_{50}, P_{75}] \ 4 & 	ext{if TEI} in (P_{75}, P_{100}] \end{cases} where P_n represents the nth percentile of TEI values
  - KB 79 [domain_knowledge] "TEI Risk Category"
    def: A category assigned as 'Low Risk' (Quartile 1), 'Moderate Risk' (Quartile 2), 'High Risk' (Quartile 3), or 'Very High Risk' (Quartile 4) based on the account's calculated TEI Quartile.
