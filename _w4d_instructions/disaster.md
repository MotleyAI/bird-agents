# W4d: disaster

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### disasterevents.impactmetrics  (column)

- Current `meta.kb_ids`: `[5, 7]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 5 [value_illustration] "impactMetrics.communication"
    def: Operational indicates fully functioning communication networks, Limited means partial communication capabilities with restrictions, and Down represents complete failure of communication infrastructure requiring alternative methods
  - KB 7 [value_illustration] "impactMetrics.damage_level"
    def: Minor represents limited structural damage with most systems functional, Moderate indicates significant damage with some systems compromised, Severe shows extensive damage with most systems affected, and Catastrophic represents complete devastation with total system failures
