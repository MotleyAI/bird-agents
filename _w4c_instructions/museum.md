# W4c: museum

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- The `conservationandmaintenance.BudgetAllocStatus` column carries
  a categorical budget-status label ('Adequate' / 'Insufficient' /
  ...). Even though it's categorical rather than a numeric
  proportion, it IS the budget-related signal the budget-efficiency
  KB needs — encode by counting / proportioning these categorical
  values per dynasty or per artifact. Don't defer this as
  schema-gap.
- Cascading composites (Budget Crisis depends on Budget Efficiency)
  resolve once their parents do.
