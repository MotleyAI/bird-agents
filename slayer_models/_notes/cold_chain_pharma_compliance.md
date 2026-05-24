# cold_chain_pharma_compliance — KB entries not encoded as model entities

## KB 1 — Temperature Excursion Severity Index (TESI)

Reason: Schema gap. TESI = TED × (|T_max − T_allowed| + |T_min − T_allowed|) / 2
needs T_allowed (the midpoint of the product's allowed temperature
range). T_allowed lives on `productbatches.tempmin` / `tempmax`, but
the schema has no FK linking `shipments` (or its child telemetry rows
in `environmentalmonitoring`) to a specific batch — `productbatches`
joins to `products` only. There is no per-shipment batch reference, so
the temperature limits cannot be reached row-wise without a join the
schema does not provide.

Status: deferred — depends on a missing shipment↔batch FK.

## KB 2 — Critical Temperature Exposure

Reason: Schema gap. The KB definition requires identifying an
exposure event where temperature deviates more than 5°C from spec for
**more than 60 consecutive minutes**. The source data
(`environmentalmonitoring.env_metrics.temperature`) only stores
shipment-level totals (`excursion_count`, `excursion_duration_min`,
`avg_c`, `min_c`, `max_c`); per-event timestamps and per-event
durations are not preserved. Consecutive-minute classification is
therefore not derivable from the stored telemetry.

Status: deferred — schema does not store per-event excursion records.

## KB 10 — Cold Chain Monitoring Compliance Score (CCMCS)

Reason: Schema gap. CCMCS = 0.4×GPS% + 0.4×Temp% + 0.2×(100 − ER).
GPS% is available as `environmentalmonitoring.gps_completeness_pct`,
but Temp% (percentage of expected temperature readings actually
received) and ER (error rate of readings) are not stored. The closest
proxies — `temppts`, `recintmin`, alarm_count — do not yield the same
quantities and a partial encoding would produce a score with two of
three weights silently zero.

Status: deferred — Temp% and ER are not in the schema.

## KB 15 — Carrier Performance Index (CPI)

Reason: Depends on undefined sub-metrics. CPI = 0.4×CCCR + 0.3×(100 −
ATNR) + 0.2×(100 − ASDI) + 0.1×DPR. CCCR (KB #5) is encoded as a
measure on `environmentalmonitoring`, but ATNR (Average Temperature
Non-conformance Rate), ASDI (Average Shock and Damage Incidents) and
DPR (Documentation Problem Rate) are not defined as separate KB
entries and have no canonical formula. Encoding CPI would require
inventing three sub-metric formulas, which mini-interact tasks do not
authorise.

Status: deferred — three of four CPI sub-metrics are undefined.

## KB 20 — Shipment Risk Score (SRS)

Reason: Depends on KB #1 (TESI), KB #10 (CCMCS) and KB #19 (PIRF / RRF).
KB #1 and KB #10 are both deferred above (schema gaps), and KB #19's
PIRF needs cross-table reach into `productbatches.store_cond` that
the schema doesn't expose per-shipment. Until those three are
encoded, SRS = 0.3×TESI + 0.25×PIRF + 0.25×(100 − CCMCS) + 0.2×RRF
cannot be assembled.

Status: deferred — depends on KB #1, #10 and #19 (all blocked on the
same shipment↔batch FK / per-event telemetry gaps).

## KB 32 — Stability Budget Consumption Rate (SBCR)

Reason: Schema gap. SBCR = TED / SB_total × 100%. TED is encoded
(`environmentalmonitoring.ted` measure), but SB_total — the total
allowable stability budget per product (KB #31) — is described in the
KB as "typically null in source data" and indeed nothing in
`productbatches`, `products`, or any other table stores a budget
duration. Without SB_total, the ratio reduces to TED alone.

Status: deferred — SB_total is not stored.

## KB 41 — Excursion Impact Assessment (EIA)

Reason: Schema gap. EIA = Σ (|T_i − T_limit| × t_i) over each
excursion event i. Source data only stores aggregate
`excursion_count` and `excursion_duration_min` per shipment plus
overall `min_c`/`max_c`; there is no per-event temperature or
per-event duration series. The summation is not derivable from the
stored telemetry.

Status: deferred — schema does not store per-event excursion records.

## KB 45 — Cold Chain Cost Efficiency Ratio (CCER)

Reason: Schema gap. CCER = (C_monitoring + C_packaging +
C_transport) / V_product × 100%. Product value V_product is
available as `productbatches.valusd_num`, but the three cost
components are not stored anywhere — `insuranceclaims.claimusd` and
`reviewsandimprovements.carbonkg` are the closest schema attributes
and neither maps to monitoring/packaging/transport cost.

Status: deferred — three of four CCER inputs are not stored.

## KB 57 — Temperature Accuracy Impact Factor (TAIF)

Reason: Cross-table reach blocked. TAIF = |T_max − T_upper|/A +
|T_min − T_lower|/A. T_max/T_min come from
`environmentalmonitoring.env_metrics.temperature` (reachable),
A comes from `monitoringdevices.devacc_num` (reachable via the
`devlink` join from environmentalmonitoring), but T_upper/T_lower
require `productbatches.tempmax`/`tempmin` per shipment — the same
missing shipment↔batch FK that blocks KB #1 (TESI). Without the
batch link there is no row-level path to the product's temperature
limits.

Status: deferred — depends on the same shipment↔batch FK that blocks
KB #1.
