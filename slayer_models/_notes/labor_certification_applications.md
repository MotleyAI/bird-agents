## KB 22 — Cost of Living Adjusted Wage (CLAW)

Reason: The formula CLAW = (Offered Wage / Cost of Living Index) * 100 needs a
per-location cost-of-living index. The schema and the column-meaning JSON
expose no such index — neither as a column nor as a sub-key on any JSONB
column (employer_contact_info, attorney_profile, prevailing_wage.wage_details,
poc_contact_info). Worksite location columns (city, state, ZIP, county) are
present but not joined to any COL series. The offered wage exists
(prevailing_wage.wage_details.offered_wage), so the *numerator* is fully
encoded as `prevailing_wage.offered_wage_annual`; the denominator is the
schema gap.

Status: deferred — SCHEMA-GAP

## KB 38 — Worksite Cost Index (WkCI)

Reason: Same schema gap as KB 22. WkCI = (Worksite Cost of Living / National
Average Cost of Living) * 100 needs both a per-worksite cost-of-living value
and a national-average reference. Neither is in the schema (worksite,
case_worksite, or any JSON sub-key). Worksite city/state/zip are available
but no COL data is joined to them.

Status: deferred — SCHEMA-GAP

## KB 55 — Position Scarcity Index

Reason: The KB defines three categories — Abundant, Moderate Scarcity, High
Scarcity — described as "often correlated with wage premiums and processing
priorities" but with no thresholds, no formula, and no explicit signal column.
Encoding would require inventing the bands (e.g. mapping ODI quantiles or WDR
ranges) which is a guess rather than a faithful encoding of the KB. The
helper signals the agent would compose with are already encoded:
`occupational_demand_index.odi` (KB #21) for relative SOC demand, and
`prevailing_wage.wage_premium_rate` / `wage_competitiveness_tier` (KB #17 /
#48) for wage-premium signals.

Status: deferred — AMBIGUOUS-PROSE
