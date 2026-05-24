# W4d: cold_chain_pharma_compliance

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### environmentalmonitoring.env_metrics  (column)

- Current `meta.kb_ids`: `[8, 40, 50]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 8 [domain_knowledge] "Temperature Monitoring Gap"
    def: A period of over 15 minutes where no temperature data points were recorded in a shipment that should have continuous monitoring.
  - KB 40 [domain_knowledge] "Temperature Profile Categorization"
    def: Temperature profiles are categorized as: 'Stable' (minimal variations within range), 'Cyclic' (regular patterns of variation within range), 'Trend' (gradual increase or decrease over time), 'Excursion' (periods outside acceptable range), or 'Erratic' (unpredictable variations suggesting monitoring i…
  - KB 50 [domain_knowledge] "Last Mile Delivery Metrics"
    def: Key last mile metrics include: First Attempt Delivery Success Rate, Temperature Deviation Frequency, Receiver Wait Time, Documentation Completion Rate, and Handler Qualification Status.

### monitoringdevices.monitoringdevices  (model)

- Current `meta.kb_ids`: `[35, 52, 33]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 35 [value_illustration] "Qualification Status of Temperature Monitoring Devices"
    def: Monitoring devices can be: 'Fully Qualified' (calibrated with NIST-traceable standards and validated for pharmaceutical use), 'Partially Qualified' (calibrated but not fully validated), 'Unqualified' (not formally qualified), with null values indicating qualification status is unknown.
  - KB 52 [value_illustration] "Electronic Monitoring System Tiers"
    def: Electronic monitoring systems are classified as: Tier 1 (basic data loggers with manual download), Tier 2 (enhanced loggers with USB or Bluetooth download), Tier 3 (network-connected devices with real-time alerts), and Tier 4 (fully integrated IoT systems with predictive capabilities and automated i…
  - KB 33 [domain_knowledge] "Package Integrity Monitoring Systems"
    def: Package integrity monitoring systems include: Shock Indicators (mechanical devices that show when a package has received an impact), Tilt Indicators (show if a package was tilted beyond acceptable angles), Electronic Impact Recorders (provide detailed shock measurements), and Pressure Indicators (mo…

### productbatches.store_cond  (column)

- Current `meta.kb_ids`: `[9, 37, 49]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 9 [value_illustration] "Product Storage Classifications"
    def: Four standard storage classifications: '2-8°C' (refrigerated), '-20°C' (frozen), '-70°C' (ultra-low temperature), and '15-25°C' (controlled room temperature).
  - KB 37 [domain_knowledge] "Storage Temperature Requirements for Biologics"
    def: Most biologics require storage at either '2-8°C' (refrigerated) or '-20°C' (frozen), with certain specialized biologic products requiring '-70°C' (ultra-low temperature) storage. Room temperature (15-25°C) storage is rarely suitable for biologics unless specifically formulated for stability at those…
  - KB 49 [domain_knowledge] "Acceptable Temperature Deviation Limits"
    def: For refrigerated products (2-8°C): brief excursions (< 30 min) to 0-12°C may be acceptable; For frozen products (-20°C): brief excursions to -15°C may be acceptable; For ultra-frozen products (-70°C): brief excursions to -60°C may be acceptable; For controlled room temperature products (15-25°C): br…

### productbatches.tempsense  (column)

- Current `meta.kb_ids`: `[6, 47, 31]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 6 [value_illustration] "Temperature Sensitivity Tiers"
    def: Products are classified into three sensitivity tiers: 'Low' (can tolerate up to 24 hours of minor temperature deviations), 'Medium' (can tolerate up to 8 hours of minor temperature deviations), 'High' (cannot tolerate temperature deviations beyond 2 hours).
  - KB 47 [value_illustration] "Thermodynamic Stability Class"
    def: Products are classified as: Class A (highly stable, can tolerate brief temperature excursions with minimal degradation), Class B (moderately stable), Class C (limited stability, requires strict temperature control), and Class D (highly unstable, no temperature excursions permitted).
  - KB 31 [domain_knowledge] "Pharmaceutical Stability Budget"
    def: The maximum cumulative duration (typically specified in hours or days) that a pharmaceutical product can experience conditions outside its labeled storage requirements without significant impact to its quality, safety, or efficacy. Null values indicate stability budget has not been determined for th…

### productbatches.pack_type  (column)

- Current `meta.kb_ids`: `[29, 33, 43]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 29 [domain_knowledge] "Premium Transport Container Types"
    def: Premium transport containers include: Envirotainer RAP e2, Envirotainer RKN e1, va-Q-tainer USx, CSafe RKN, DoKaSch Opticooler, and Sonoco ThermoSafe PharmaPort 360.
  - KB 33 [domain_knowledge] "Package Integrity Monitoring Systems"
    def: Package integrity monitoring systems include: Shock Indicators (mechanical devices that show when a package has received an impact), Tilt Indicators (show if a package was tilted beyond acceptable angles), Electronic Impact Recorders (provide detailed shock measurements), and Pressure Indicators (mo…
  - KB 43 [value_illustration] "Package Thermal Efficiency Rating"
    def: Thermal efficiency rating is classified as: 'Basic' (<24 hours of protection), 'Standard' (24-48 hours of protection), 'Enhanced' (48-96 hours of protection), 'Extended' (>96 hours of protection).

### qualitycompliance.product_release_status  (column)

- Current `meta.kb_ids`: `[34, 51]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 34 [domain_knowledge] "Product Release Decision Framework"
    def: A decision framework considering: 1) Presence of temperature excursions, 2) Stability data for specific excursion profiles, 3) Package integrity, 4) Product appearance, and 5) Analytical testing results if required. Products can be 'Released' (meets all criteria), 'Released with CAPA' (meets essenti…
  - KB 51 [domain_knowledge] "Batch Release Critical Path Elements"
    def: Critical path elements include: Complete temperature history with no unexplained gaps, Confirmation that any excursions were within stability budgets, Intact security seals or acceptable explanation for compromised seals, Complete chain of custody documentation, and Acceptable visual inspection resu…

### shipments.destination_nation  (column)

- Current `meta.kb_ids`: `[16, 42]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 16 [domain_knowledge] "Major Pharmaceutical Markets"
    def: Major pharmaceutical markets include: United States, European Union (specifically Germany, France, Italy, Spain, and United Kingdom), Japan, China, Brazil, India, and Russia.
  - KB 42 [domain_knowledge] "Primary Cold Chain Monitoring Authorities"
    def: Primary regulatory authorities for pharmaceutical cold chains include: FDA (US Food and Drug Administration), EMA (European Medicines Agency), MHRA (UK Medicines and Healthcare products Regulatory Agency), Health Canada, TGA (Australian Therapeutic Goods Administration), PMDA (Japanese Pharmaceutica…
