# W4d: organ_transplant

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### clinical.med_urgency  (column)

- Current `meta.kb_ids`: `[25, 45]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 25 [domain_knowledge] "Medical Urgency Status"
    def: A tiered system that reflects how critically ill a patient is. For calculation purposes, statuses are mapped to numerical values: Status 1A is assigned a value of 5, Status 1B is 4, Status 2 is 3, and Status 3 is 2. All other statuses are assigned a value of 1.
  - KB 45 [value_illustration] "Medical Urgency Tiers"
    def: Patients are prioritized based on their risk of dying while on the waitlist. Status 1A is the highest urgency, reserved for critically ill patients in the ICU. Status 1B is a lower, but still urgent, category. Lower statuses (e.g., Status 2) are for more stable patients.

### compatibility_metrics.hla_mis_count  (column)

- Current `meta.kb_ids`: `[4, 44]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 4 [calculation_knowledge] "HLA Mismatch Score"
    def: A count of the mismatched HLA antigens (A, B, DR) between the donor and recipient. Formula: $S_{mismatch} = \sum_{i \in \{A, B, DR\}} (HLA_i^{donor} \neq HLA_i^{recipient})$
  - KB 44 [value_illustration] "HLA Mismatch Levels"
    def: The number of mismatches impacts rejection risk. A '0-mismatch' is a perfect HLA match and is ideal. A '6-mismatch' is a complete mismatch across the A, B, and DR loci, representing the highest immunological barrier outside of ABO or positive crossmatch issues.

### function_and_recovery.don_co_desc  (column)

- Current `meta.kb_ids`: `[32, 47]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 32 [domain_knowledge] "Anoxia"
    def: A donor's cause of death resulting from a complete lack of oxygen. [cite_start]Within the database, this specific cause is recorded as 'Anoxia' in the donor's cause of death description (`don_co_desc` [cite: 15]). This factor is critical for assessing the viability of oxygen-sensitive organs, such a…
  - KB 47 [value_illustration] "Cause of Death Impact"
    def: The mechanism of death influences organ viability. For example, death due to Anoxia (oxygen deprivation) can compromise the function of oxygen-sensitive organs like the heart and kidneys. In contrast, death from head trauma may leave abdominal organs in optimal condition.

### recipients_immunology.pra_score  (column)

- Current `meta.kb_ids`: `[23, 43]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 23 [domain_knowledge] "Panel Reactive Antibody (PRA)"
    def: A measure of a recipient's sensitization to foreign HLA antigens, represented by the `pra_score`[cite: 36]. For matching purposes, a recipient with a `pra_score` of 80 or higher is defined as having a 'high PRA score', indicating a state of high immunological sensitization.
  - KB 43 [value_illustration] "Panel Reactive Antibody (PRA) Score Interpretation"
    def: A PRA score reflects a recipient's sensitization level. A score of 0-10% indicates low sensitization, making it easier to find a compatible donor. A score > 80% indicates high Immunological Sensitization, meaning the patient is likely to be incompatible with over 80% of potential donors.

### recipients_immunology.cross_result  (column)

- Current `meta.kb_ids`: `[24, 46]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 24 [domain_knowledge] "Crossmatch Test"
    def: A laboratory test that directly mixes the recipient's serum with the donor's lymphocytes. A 'positive' Crossmatch Test indicates the presence of donor-specific antibodies and is a contraindication to transplantation, as it predicts immediate organ rejection.
  - KB 46 [value_illustration] "Crossmatch Results"
    def: A 'Negative' result means no pre-formed donor-specific antibodies were detected, and the transplant can proceed. A 'Positive' result indicates the presence of these antibodies, making transplant highly risky or contraindicated due to the high chance of hyperacute rejection.

### risk_evaluation.egs_val  (column)

- Current `meta.kb_ids`: `[13, 28]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 13 [calculation_knowledge] "Expected Graft Survival (EGS) Score"
    def: A predictive score based on factors like the Immunological Compatibility Score and donor age. Formula: $EGS = \frac{1}{1 + e^{-(-0.5 + 1.5 S_{immune} - 0.02 Age_{donor})}}$
  - KB 28 [domain_knowledge] "Graft Survival"
    def: A key outcome measure in transplantation. Successful Graft Survival means the organ is performing its intended biological functions without signs of rejection or failure.

### risk_evaluation.cost_qaly  (column)

- Current `meta.kb_ids`: `[12, 48]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 12 [calculation_knowledge] "Cost-Effectiveness Ratio (CER)"
    def: The ratio of the net cost of an intervention to its net health gain. Formula: $CER = \frac{Cost_{net}}{QALY_{gained}}$
  - KB 48 [value_illustration] "Cost-Effectiveness Thresholds"
    def: A common benchmark for the Cost-Effectiveness Ratio (CER) in the US is $50,000 to $150,000 per Quality-Adjusted Life Year (QALY) gained. Interventions below this threshold are generally considered a good value.
