# W4d: credit

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### credit_accounts_and_history.credutil  (column)

- Current `meta.kb_ids`: `[1, 23]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 1 [calculation_knowledge] "Credit Utilization Ratio (CUR)"
    def: CUR = \frac{\text{Total Credit Used}}{\text{Total Credit Limit}} = credutil
  - KB 23 [value_illustration] "Credit Utilization Impact"
    def: Credit Utilization (credutil) ranges from 0-1 (or above). Utilization under 0.30 is optimal for credit scores, 0.30-0.50 has moderate negative impact, 0.50-0.70 has significant negative impact, and above 0.70 severely impacts credit scores.

### employment_and_income.debincratio  (column)

- Current `meta.kb_ids`: `[0, 22]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 0 [calculation_knowledge] "Debt-to-Income Ratio (DTI)"
    def: DTI = \frac{\text{Total Monthly Debt Payments}}{\text{Monthly Income}} = debincratio
  - KB 22 [value_illustration] "Debt-to-Income Ratio Interpretation"
    def: Debt-to-Income ratio (debincratio) ranges from 0-1 (or above). Below 0.36 is typically considered excellent, 0.36-0.43 is good, 0.43-0.50 is concerning, and above 0.50 is risky for new credit approval.

### expenses_and_assets.ltv  (column)

- Current `meta.kb_ids`: `[2, 24]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 2 [calculation_knowledge] "Loan-to-Value Ratio (LTV)"
    def: LTV = \frac{\text{Mortgage Balance}}{\text{Property Value}} = \frac{\text{propfinancialdata.mortgagebits.mortbalance}}{\text{propfinancialdata.propvalue}}
  - KB 24 [value_illustration] "Loan-to-Value Ratio Significance"
    def: Loan-to-Value ratio (LTV) typically ranges from 0-1 (or above). Below 0.80 generally avoids private mortgage insurance requirements, 0.80-0.95 typically requires PMI, and above 0.95 indicates high leverage and increased lending risk.
