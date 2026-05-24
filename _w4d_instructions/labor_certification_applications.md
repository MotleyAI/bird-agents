# W4d: labor_certification_applications

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### case_worksite.wage_differential_rate  (column)

- Current `meta.kb_ids`: `[11, 31]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 11 [calculation_knowledge] "Wage Differential Rate (WDR)"
    def: WDR = ((Offered Wage - Prevailing Wage) / Prevailing Wage) × 100%. Both offered and prevailing wages are converted to the same payment unit (hourly, weekly, or annually) before calculation. Wage information is included in the wage determination data.
  - KB 31 [calculation_knowledge] "Industry Wage Differential (IWD)"
    def: IWD = (Sum of Wage Differential Rates in Industry / Number of Applications in Industry). Wage and industry information are included in the application data.

### employer_scale_indicator.employer_scale_indicator  (model)

- Current `meta.kb_ids`: `[30, 46]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 30 [calculation_knowledge] "Employer Scale Indicator (ESI)"
    def: ESI = (Employer Number of Applications / Average Applications per Employer). Employer and application information are tracked as part of the application data.
  - KB 46 [domain_knowledge] "Employer Size Classification"
    def: Employers are classified as Small-scale users (fewer than 5 applications annually), Medium-scale users (5-25 applications annually), or Large-scale users (more than 25 applications annually). Employer and application information are tracked as part of the application data.

### prevailing_wage.wage_details  (column)

- Current `meta.kb_ids`: `[5, 8]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 5 [value_illustration] "Prevailing Wage Levels"
    def: Prevailing wage levels range from I to IV, representing increasingly higher wages based on skill, experience, education, and responsibility: Level I (entry-level), Level II (qualified), Level III (experienced), and Level IV (fully competent). These levels are determined based on the position's requi…
  - KB 8 [value_illustration] "Wage Payment Units"
    def: Wage payment units indicate how wages are calculated and paid, with common units including: Hour (payment calculated per working hour), Week (payment calculated as a weekly salary), Month (payment calculated as a monthly salary), and Year (payment calculated as an annual salary). Different units may…
