# W4d: museum

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### artifact_value_categories.artifact_value_categories  (model)

- Current `meta.kb_ids`: `[11, 51]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 11 [domain_knowledge] "High-Value Artifact"
    def: An artifact is considered high-value when its InsValueUSD exceeds $1,000,000 OR when both HistSignRating and CultScore are in the top 10% of all artifacts.
  - KB 51 [domain_knowledge] "High-Value Category"
    def: An artifact falls into 'Monetary High-Value' category when its InsValueUSD exceeds $1,000,000. It qualifies as 'Cultural/Historical High-Value' when both its HistSignRating and CultScore are in the top 10% of all artifacts (percentile rank = 1). Otherwise 'Other'.

### lightandradiationreadings.lightlux  (column)

- Current `meta.kb_ids`: `[29, 53]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 29 [value_illustration] "LightAndRadiationReadings.LightLux"
    def: Values in lux range from near-dark (5-10 lux) to typical indoor lighting (300-500 lux). Conservation standards recommend 50 lux for highly sensitive materials, 150-200 lux for paintings and wood, and up to 300 lux for stone and metal. Daylight can exceed 10,000 lux and should be filtered.
  - KB 53 [domain_knowledge] "Light Exposure Thresholds"
    def: High sensitivity artifacts (textiles, paper) must not exceed 50 lux; Medium sensitivity (paintings, wood) must not exceed 200 lux. Based on conservation research about light-induced deterioration rates.
