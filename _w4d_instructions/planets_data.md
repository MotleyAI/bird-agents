# W4d: planets_data

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### stars.stellarprops  (column)

- Current `meta.kb_ids`: `[38, 41, 42, 43, 44]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 5 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 38 [domain_knowledge] "Blended Measurement"
    def: A quality flag indicating that a measurement's value is affected by light from stellar companions, potentially reducing its accuracy.
  - KB 41 [value_illustration] "Apparent Magnitude Value"
    def: This is a logarithmic scale where smaller numbers are brighter. A magnitude of 1.0 is 100 times brighter than a magnitude of 6.0. The brightest stars have negative magnitudes (e.g., Sirius is -1.46).
  - KB 42 [value_illustration] "Stellar Temperature Value"
    def: Measured in Kelvin (K). Cool red dwarfs can be around 3,000 K, a Sun-like star is about 5,800 K, and very hot blue stars can exceed 30,000 K.
  - KB 43 [value_illustration] "Stellar Mass Value"
    def: Measured in solar masses ($M_{\odot}$), where 1 is the mass of our Sun. Most known host stars range from low-mass red dwarfs (~0.1 $M_{\odot}$) to stars several times more massive than the Sun.
  - KB 44 [value_illustration] "Stellar Radius Value"
    def: Measured in solar radii ($R_{\odot}$), where 1 is the radius of our Sun. Values range from small neutron stars to giant stars like Betelgeuse, which would extend beyond the orbit of Mars if in our solar system.
