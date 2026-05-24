# W4d: exchange_traded_funds

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### funds.alpha3y_for_regression  (column)

- Current `meta.kb_ids`: `[82, 83]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 82 [calculation_knowledge] "Alpha-Turnover Slope"
    def: eta_{\alpha, T} = 	ext{Slope of regression}(	ext{3-Year Alpha}, 	ext{Turnover Ratio})
  - KB 83 [calculation_knowledge] "Fit Quality"
    def: R^2 = 	ext{R-squared of regression}(	ext{3-Year Alpha}, 	ext{Turnover Ratio})
