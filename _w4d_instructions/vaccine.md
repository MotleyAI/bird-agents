# W4d: vaccine

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### container_efficiency_score.container_efficiency_score  (model)

- Current `meta.kb_ids`: `[36, 41, 48]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 36 [calculation_knowledge] "Container Efficiency Score (CES)"
    def: CES = \text{SER} \times \text{CEI} \times (1 - \text{CRI})
  - KB 41 [domain_knowledge] "Severe Container Risk"
    def: A container where CES < 0.3 and CEI < 0.5
  - KB 48 [domain_knowledge] "Container Alert Status"
    def: A status where CES < 0.4 and TRS > 0.7

### datalogger.commproto  (column)

- Current `meta.kb_ids`: `[22, 26]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 22 [value_illustration] "CommProto: 'RF'"
    def: Uses 433MHz ISM band radio frequency transmission with FSK modulation for short-range data exchange below 100m.
  - KB 26 [value_illustration] "CommProto: 'Satellite'"
    def: Indicates the system is using satellite networks for global coverage and reliable data transmission.

### logger_reliability_score.logger_reliability_score  (model)

- Current `meta.kb_ids`: `[34, 45]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 34 [calculation_knowledge] "Logger Reliability Score (LRS)"
    def: LRS = 	ext{LHI} \times (1 - \text{CMR}) \times \text{TSS}
  - KB 45 [domain_knowledge] "Logger Critical State"
    def: A logger where LRS < 0.3 and CMR > 0.8

### quality_maintenance_index.quality_maintenance_index  (model)

- Current `meta.kb_ids`: `[37, 42, 46]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 37 [calculation_knowledge] "Quality Maintenance Index (QMI)"
    def: QMI = \text{MCS} \times \text{SQI} \times (1 - \frac{\text{TBS}}{10})
  - KB 42 [domain_knowledge] "High Maintenance Priority"
    def: Equipment where CMR > 0.7 and QMI < 0.4
  - KB 46 [domain_knowledge] "Quality Alert Status"
    def: A status where QMI < 0.3 and VSI < 0.5

### route_risk_factor.route_risk_factor  (model)

- Current `meta.kb_ids`: `[38, 43, 47]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 38 [calculation_knowledge] "Route Risk Factor (RRF)"
    def: RRF = (1 - \frac{	ext{RCP}}{100}) \times \text{TRS} \times (1 - \text{CEI})
  - KB 43 [domain_knowledge] "Critical Route Status"
    def: A route where RRF > 0.8 and TSR < 0.3
  - KB 47 [domain_knowledge] "Transport Safety Alert"
    def: A condition where TSR < 0.5 and RRF > 0.7

### transport_safety_rating.transport_safety_rating  (model)

- Current `meta.kb_ids`: `[35, 40]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 35 [calculation_knowledge] "Transport Safety Rating (TSR)"
    def: TSR = \frac{\text{RCP}}{100} \times (1 - \text{TRS}) \times \text{HQI}
  - KB 40 [domain_knowledge] "Critical Transport Condition"
    def: A transport condition where TSR < 0.4 and TRS > 0.6

### vaccine_safety_index.vaccine_safety_index  (model)

- Current `meta.kb_ids`: `[39, 44]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 39 [calculation_knowledge] "Vaccine Safety Index (VSI)"
    def: VSI = \frac{	ext{VVP}}{365} \times \text{CEI} \times (1 - \text{TRS})
  - KB 44 [domain_knowledge] "Unsafe Vaccine Condition"
    def: A condition where VSI < 0.4 and CES < 0.5
