# W4d: households

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### households.socioeconomic  (column)

- Current `meta.kb_ids`: `[1, 2]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 1 [value_illustration] "Household Tenure Status"
    def: Values based on schema include 'OWNED', 'RENTED', 'OCCUPIED'. The 'OWNED' status corresponds to owner-occupied properties.
  - KB 2 [value_illustration] "Income Classification"
    def: Ranges from 'Low Income' to 'Very High Income'. Null indicates undisclosed or irregular income.

### transportation_assets.vehicleinventory  (column)

- Current `meta.kb_ids`: `[10, 36]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 10 [value_illustration] "Vehicle Year Range"
    def: Text ranges like '1995 to 1999', '2005 to 2009', or '2010 to 2013'. Null indicates no vehicles or unknown age.
  - KB 36 [value_illustration] "Vehicle Type Distribution"
    def: An array of counts representing vehicle types. Null indicates unknown or unverified ownership.
