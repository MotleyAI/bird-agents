# W4d: cybermarket

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### securitymonitoring.alertsev_numeric  (column)

- Current `meta.kb_ids`: `[9, 30]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 9 [value_illustration] "cybermarket|securitymonitoring|alertsev"
    def: 'Low' alerts indicate minor anomalies requiring minimal attention and posing limited security risk; 'Medium' alerts signal notable deviations from baseline behavior requiring investigation within standard timeframes; 'High' alerts denote significant security concerns demanding prompt attention and i…
  - KB 30 [calculation_knowledge] "Market Vulnerability Index (MVI)"
    def: MVI = (100 - MSI) + (COUNT(CASE WHEN alertsev IS NOT NULL THEN 1 END) / 10) \times (alertsev_numeric \times 2) - (vendcount \times 0.05) + (COUNT(CASE WHEN lawinterest = 'High' THEN 1 END) / 5), \text{where alertsev_numeric maps Low=1, Medium=2, High=3, Critical=4 as defined in the alert severity sy…

### vendors.sizecluster_numeric  (column)

- Current `meta.kb_ids`: `[1, 31]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 1 [value_illustration] "cybermarket|markets|sizecluster"
    def: Size clusters represent market scale and reach: 'Small' markets typically have under 10,000 monthly users and limited vendor presence; 'Medium' markets host 10,000-50,000 monthly users with moderate vendor diversity; 'Large' markets serve 50,000-100,000 users with extensive product catalogs; 'Mega' …
  - KB 31 [calculation_knowledge] "Vendor Network Centrality (VNC)"
    def: VNC = (COUNT(DISTINCT mktref) \times 5) + \frac{vendtxcount}{50} + (VTI \times 0.1) - (1 - sizecluster_numeric) \times 10, \text{where sizecluster_numeric maps Small=1, Medium=2, Large=3, Mega=4 as described in the market size classification, and higher scores indicate more central market positionin…
