# Solar KB — unencoded entries

23 of 53 KB ids are encoded in `slayer_models/solar/` with `meta.kb_id`. The
30 entries below are deferred for the reasons listed.

## KB 1 — Panel Efficiency Loss Rate (PELR)

Reason: PELR = (CurrentEfficiencyPercent − InitialEfficiencyPercent) /
Years_Since_Installation × 100%. `InitialEfficiencyPercent` is not a column
in any solar table (efficiency_profile only stores the current snapshot),
and `Years_Since_Installation` requires an "as of" report date that is a
query parameter, not data.

Status: deferred — SCHEMA-GAP

## KB 2 — Temperature Performance Coefficient Impact (TPCI)

Reason: TPCI = PowerRatedW × TempCoef × (CellTempC − 25). Mixes columns
from `panel` and `environment`, which share `plant` as their common
ancestor in the FK graph. Default encoding is R-MULTISTAGE per the
peer-join discipline — out of scope for this first cut.

Status: deferred — AMBIGUOUS-PROSE

## KB 3 — Energy Production Efficiency (EPE)

Reason: EPE = PPR × (1 − SoilingLossPct/100) × (1 − CumDegPct/100). PPR
lives on `performance`, SoilingLossPct on `environment`, CumDegPct on
`performance.efficiency_profile__degradation__cumdegpct`. Cross-table
composite via `plant` — R-MULTISTAGE.

Status: deferred — AMBIGUOUS-PROSE

## KB 4 — Inverter Efficiency Loss (IEL)

Reason: IEL = MeasuredPowerW × (1 − InverterEfficiencyPercent/100).
`performance.measpoww` × `inverter.power_metrics__inverteffpct` — both
tables peer-join to `plant`. R-MULTISTAGE.

Status: deferred — AMBIGUOUS-PROSE

## KB 5 — Fill Factor Degradation Rate (FFDR)

Reason: FFDR = (FillFactorInitial − FillFactorCurrent) /
Years_Since_Installation × 100%. Years_Since_Installation needs an "as of"
report date that is a query parameter, not data.

Status: deferred — SCHEMA-GAP

## KB 9 — Irradiance Utilization Ratio (IUR)

Reason: IUR = (MeasuredPowerW / PanelAreaM2) / POAIrradianceWM2.
`PanelAreaM2` is not a column on `panel` — the schema captures rated power
and efficiency but not surface area.

Status: deferred — SCHEMA-GAP

## KB 10 — Critical Performance Threshold

Reason: "PPR below 80% of the expected value for its age, accounting for
normal degradation." The "expected for age" curve is not pinned by the KB;
it would require a manufacturer warranty curve or an explicit
linear-degradation assumption.

Status: deferred — AMBIGUOUS-PROSE

## KB 11 — Hot Spot Risk

Reason: "Voc or Isc deviations >5% compared to other panels in the same
string" requires a cross-row stddev/mean over panels grouped by string,
plus a `panel-to-string` mapping not modelled in the schema.

Status: deferred — SCHEMA-GAP

## KB 14 — End-of-Life Indicator

Reason: "Cumulative degradation > 20% OR maintenance costs in a 12-month
period > 30% of replacement value." The maintenance arm requires a rolling
12-month window (R-WINDOW with an "as of" parameter) and a replacement-cost
reference not in the schema.

Status: deferred — SCHEMA-GAP

## KB 16 — Panel String Mismatch

Reason: "Standard deviation of current measurements across panels in the
same string exceeds 3% of the mean under the same irradiance conditions."
Requires a panel-to-string grouping not in the schema.

Status: deferred — SCHEMA-GAP

## KB 17 — Warranty Claim Threshold

Reason: "EPE > 10% below the manufacturer's warranty curve for its age."
Manufacturer warranty curve is not available as data; the KB does not
specify a substitute parametric curve.

Status: deferred — SCHEMA-GAP

## KB 30 — Temperature Adjusted Performance Ratio (TAPR)

Reason: TAPR = PPR + (TPCI / PowerRatedW). Depends on KB 2 (TPCI),
deferred R-MULTISTAGE.

Status: deferred — AMBIGUOUS-PROSE

## KB 31 — Total System Loss (TSL)

Reason: TSL = (PowerRatedW × CumDegPct/100) + (MeasuredPowerW ×
SoilingLossPct/100) + IEL. Aggregates losses across `panel`,
`performance`, `environment`, and `inverter` — multi-peer R-MULTISTAGE.

Status: deferred — AMBIGUOUS-PROSE

## KB 33 — Effective Performance Index (EPI)

Reason: EPI = PPR × (1 − TSL/PowerRatedW) × (IUR/PaneEffPct). Depends on
KB 31 (TSL, deferred) and KB 9 (IUR, SCHEMA-GAP).

Status: deferred — SCHEMA-GAP

## KB 34 — Normalized Degradation Index (NDI)

Reason: NDI = PELR / AnnDegRate. PELR (KB 1) is deferred for needing an
"as of" date / initial efficiency snapshot.

Status: deferred — SCHEMA-GAP

## KB 35 — Weather Corrected Efficiency (WCE)

Reason: WCE = CurrentEfficiencyPercent × (1 + TempCoef × (25 − CellTempC)
/ 100) × (1000 / POAIrradianceWM2). Mixes `performance`, `panel`, and
`environment` — R-MULTISTAGE.

Status: deferred — AMBIGUOUS-PROSE

## KB 36 — Expected Energy Yield (EEY)

Reason: EEY = PowerRatedW × EPE × SIF × POAIrradianceWM2 / 1000. Depends
on EPE (KB 3, deferred) and reaches across `panel`, `environment`, and
`performance`.

Status: deferred — AMBIGUOUS-PROSE

## KB 39 — Financial Impact of Degradation (FID)

Reason: FID = GenCapMW × 1000 × NDI × PELR × ElectricityPricePerKWh × 24
× 365. Depends on NDI/PELR (deferred) plus `ElectricityPricePerKWh` which
is a query parameter, not data.

Status: deferred — SCHEMA-GAP

## KB 40 — Advanced Performance Degradation Alert

Reason: Predicate over NDI > 1.5 AND TAPR < Critical Performance
Threshold. All inputs (KB 34, KB 30, KB 10) are deferred.

Status: deferred — AMBIGUOUS-PROSE

## KB 41 — Premium Maintenance Candidate

Reason: Predicate over MROI > 2.0 AND EPE between 75% and 90%. EPE (KB 3)
is deferred R-MULTISTAGE.

Status: deferred — AMBIGUOUS-PROSE

## KB 42 — Optimal Cleaning Schedule

Reason: "Schedule when SIF × DustDensity exceeds the Soiling Cleaning
Threshold OR when EEY is reduced by more than 3% due to soiling." Second
arm depends on KB 36 (EEY, multistage).

Status: deferred — AMBIGUOUS-PROSE

## KB 44 — End-of-Warranty Optimization

Reason: "Evaluate for warranty claims when approaching warranty expiration
if NDI > 0.9 or if the Warranty Claim Threshold is missed." Both NDI (KB
34) and Warranty Claim Threshold (KB 17) are deferred.

Status: deferred — SCHEMA-GAP

## KB 46 — System Upgrade Candidate

Reason: "FID > 10% of replacement cost AND EPI < 0.85 for three
consecutive months." Depends on FID (KB 39, SCHEMA-GAP), EPI (KB 33,
deferred), plus a 3-month windowed aggregation and a replacement-cost
reference.

Status: deferred — SCHEMA-GAP

## KB 47 — Inverter-Panel Compatibility Index

Reason: "WCE within 5% of manufacturer specs AND InverterEfficiencyPct
> 97%." Manufacturer specs aren't in the schema; WCE itself is deferred.

Status: deferred — SCHEMA-GAP

## KB 48 — Environmental Stress Classification

Reason: "Combining Weather Severity Index and exposure time outside the
Optimal Performance Window." Requires a time-windowed aggregation of
out-of-window minutes per panel.

Status: deferred — AMBIGUOUS-PROSE

## KB 49 — Total Economic Performance

Reason: "Comprehensive economic evaluation combining MROI, FID, and
revenue generation adjusted by EPI." Several inputs deferred; the KB
doesn't pin the combining formula.

Status: deferred — AMBIGUOUS-PROSE

## KB 50 — Maintenance Urgency Classification

Reason: Classification depends on "critical alerts" (cross-table to
`alerts` filtered by `alertstat = 'Critical'`) AND MROI > 2.0. Requires a
peer-join / multistage encoding between `alerts` and `maintenance`.

Status: deferred — AMBIGUOUS-PROSE

## KB 51 — Cleaning Triggers

Reason: "Meet Soiling Cleaning Threshold (KB 13) OR >30 days since last
cleaning." First arm is encoded as `environment.needs_cleaning_threshold`
(KB 13). The "30 days since" arm requires a non-deterministic `date('now')`
reference whose result varies per run.

Status: deferred — AMBIGUOUS-PROSE

## KB 52 — Degradation Severity Classification

Reason: Classes are bands of NDI (KB 34, deferred).

Status: deferred — SCHEMA-GAP

## KB 53 — Alert Specification Protocol

Reason: Specifies a write operation (insert into `alerts` with a specific
shape and id prefix) plus an update-within-30-days policy. The SLayer
semantic layer is read-only by design.

Status: deferred — DML
