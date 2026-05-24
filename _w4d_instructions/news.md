# W4d: news

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### news_recommendations.news_recommendations  (model)

- Current `meta.kb_ids`: `[11, 15]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 11 [domain_knowledge] "Personalization Priority (PP)"
    def: Content that strongly matches user preferences and exhibits high interaction rates is prioritized in personalized recommendations.
  - KB 15 [domain_knowledge] "Content Recommendation Strategy (CRS)"
    def: Recommendation strategies are developed by jointly considering content relevance, user behavior analytics, and system performance metrics.

### news_sessions.news_sessions  (model)

- Current `meta.kb_ids`: `[16, 18]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 16 [domain_knowledge] "User Behavior Paradigm (UBP)"
    def: By analyzing metrics such as session duration, bounce rates, and interaction sequences, common behavioral patterns are identified which inform future personalization efforts.
  - KB 18 [domain_knowledge] "Short Session Anomaly Detection (SSAD)"
    def: Sessions with a duration significantly below the average, accompanied by low engagement and minimal page views, are flagged as anomalies.

### news_users.testgrp  (column)

- Current `meta.kb_ids`: `[12, 55]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 12 [domain_knowledge] "AB Testing Cohort Analysis (ABTCA)"
    def: Users are assigned to predetermined groups to facilitate controlled comparisons of experimental feature performance across cohorts.
  - KB 55 [calculation_knowledge] "Cohort Percentage"
    def: (Group registrations / Total monthly registrations) * 100
