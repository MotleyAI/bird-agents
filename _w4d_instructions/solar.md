# W4d: solar

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### panel_performance.efficiency_profile  (column)

- Current `meta.kb_ids`: `[26, 27]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 26 [value_illustration] "AnnDegRate (Annual Degradation Rate)"
    def: Expressed as a percentage, representing how much a panel's output decreases per year. Quality silicon panels typically degrade at 0.5% to 0.7% annually, while lower quality or certain thin-film technologies may degrade at rates above 1% annually.
  - KB 27 [value_illustration] "CumDegPct (Cumulative Degradation Percentage)"
    def: Expressed as a percentage, representing total performance degradation since installation. New panels start at 0%, while panels approaching end-of-life may have values of 15-20% or higher, with manufacturer warranties typically covering degradation up to 20% over 25 years.
