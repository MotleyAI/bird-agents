# organ_transplant — KB entries not encoded as model entities

Encoded coverage handles 47 of the 55 KB entries directly on model
columns / measures (descriptions, parsed numeric helpers, KB
formulas using SLayer's `pow`/`exp` UDFs for BMI, eGFR, age
compatibility, EGS sigmoid, and the composite allocation /
comprehensive match scores). The remaining KB ids below are
deferred because their formula or definition requires a column
the source database does not expose, or because they are pure
domain prose without a corresponding row-level / aggregate
expression.

## KB 11 — Quality-Adjusted Life Year (QALY)

Reason: KB defines `QALY = Years_life × Quality_score`. The
`risk_evaluation.qol_val` column gives a quality-of-life score,
but no `Years_life` (life-expectancy) column exists in any of
the 14 source tables. Cannot compute the product.

Status: deferred — missing data column (Years_life). Cascades
into KB 52 (Net Health Benefit Score) which multiplies EGS by
QALY-gain.

## KB 18 — Resource Utilization Index

Reason: KB definition is `S_surgical × Days_stay`. The
`risk_evaluation.resource_consumption` field captures
units/day, and `staff` captures hrs/case, but neither expresses
the recipient's hospital length of stay. Without Days_stay the
index can't be computed.

Status: deferred — missing data column (Days_stay /
length-of-hospital-stay).

## KB 37 — Multi-Organ Transplant

Reason: A single `transplant_matching` row carries one
`org_spec` (Heart / Kidney / Liver / Lung / Pancreas), so the
"two or more organs from a single donor" predicate would
require detecting multiple match rows for the same
(donor_ref_reg, recip_ref_reg) pair — a query-time aggregation,
not a single column. The schema also lacks a flag indicating
multi-organ batches.

Status: deferred — needs an R-EXISTS / R-MULTISTAGE encoding
(GROUP BY donor+recipient with COUNT(DISTINCT org_spec) > 1)
that the agent can express directly at query time.

## KB 38 — Post-Transplant Monitoring

Reason: Pure domain-knowledge prose describing the ongoing
follow-up care plan. No corresponding column or table in the
schema (nothing about lab tests, imaging, immunosuppressive
dosing post-tx).

Status: deferred — no DB attribute; documentation-only entry.

## KB 52 — Net Health Benefit Score

Reason: Formula is `(EGS × QALY_gain) − (S_surgical × 0.2)`.
Cascades from KB 11: without QALY_gain (life-years × quality)
the first term cannot be evaluated.

Status: deferred — cascades from KB 11 (no Years_life column).

## KB 53 — Marginal Donor Acceptance Criteria

Reason: A clinician-facing decision framework, not a formula.
The KB itself says acceptance is a "risk-benefit analysis,
weighing the organ's imperfections against the recipient's
urgent medical need" — no quantitative threshold to encode.
Components it cites (KB 27 High-Risk Donor, KB 9 Renal Function
Score, KB 1 Donor-Recipient Age Difference) are already
encoded; the agent can compose an ad-hoc filter at query time.

Status: deferred — prose framework, no canonical predicate.

## KB 54 — Immunosuppression Protocol Tiers

Reason: Prose mapping of (Immunological Compat Score, AMR Risk
Stratification) to therapeutic strategy ("Maintenance" vs
"Induction Therapy"). No therapy/regimen field exists in the
schema, so there's no encoded column to map to. Components
(KB 10, KB 51) are already encoded; the protocol assignment
is prescriptive, not data.

Status: deferred — no therapy/regimen column in the source
DB.
