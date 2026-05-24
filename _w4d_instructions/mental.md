# W4d: mental

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### assessmentsymptomsandrisk.phq9_score  (column)

- Current `meta.kb_ids`: `[0, 20, 30]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 0 [calculation_knowledge] "Average PHQ-9 Score by Facility (APSF)"
    def: APSF = \frac{\sum_{i \in assessmentsymptomsandrisk} (mental\_health\_scores_i['depression']['phq9\_score'])} {|assessmentsymptomsandrisk|}
  - KB 20 [value_illustration] "PHQ-9 Score (Depression)"
    def: Ranges from 0 to 27. A score of 0–4 indicates minimal depression, 5–9 mild, 10–14 moderate, 15–19 moderately severe, and 20–27 severe.
  - KB 30 [calculation_knowledge] "Symptom Severity Index (SSI)"
    def: SSI = \frac{APSF + AGSF}{2}, \text{using Average PHQ-9 Score (APSF) and Average GAD-7 Score (AGSF)}

### assessmentsymptomsandrisk.gad7_score  (column)

- Current `meta.kb_ids`: `[1, 21]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 1 [calculation_knowledge] "Average GAD-7 Score by Facility (AGSF)"
    def: AGSF = \frac{\sum_{i \in assessmentsymptomsandrisk} (mental\_health\_scores_i['anxiety']['gad7\_score'])} {|assessmentsymptomsandrisk|}, \text{where } mental\_health\_scores_i['anxiety']['gad7\_score'] \text{ is the GAD-7 score for each assessment in the assessmentsymptomsandrisk table, linked to a …
  - KB 21 [value_illustration] "GAD-7 Score (Anxiety)"
    def: Ranges from 0 to 21. A score of 0–4 indicates minimal anxiety, 5–9 mild, 10–14 moderate, and 15–21 severe.

### encounters.missappt  (column)

- Current `meta.kb_ids`: `[9, 29]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 9 [calculation_knowledge] "Missed Appointment Rate (MAR)"
    def: MAR = \frac{\sum_{i \in encounters} missappt_i} {|patients|}
  - KB 29 [value_illustration] "Missed Appointment Count"
    def: A numeric value indicating the number of missed appointments. A value of 0 indicates perfect attendance, while higher values (e.g., 5) indicate frequent absences.

### treatmentbasics.crisisint  (column)

- Current `meta.kb_ids`: `[7, 27]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 7 [calculation_knowledge] "Crisis Intervention Frequency (CIF)"
    def: CIF = \frac{\sum_{i \in treatmentbasics} crisisint_i} {|patients|}
  - KB 27 [value_illustration] "Crisis Intervention Count"
    def: A numeric value indicating the number of crisis interventions. A value of 0 indicates no interventions, while higher values (e.g., 3) indicate frequent interventions.
