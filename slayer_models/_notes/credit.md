# credit — KB entries not encoded as model entities

KB ids in the headers below are deliberately omitted from
`slayer_models/credit/`. The verifier
(`scripts/verify_kb_coverage.py`) reads the `## KB <id> — …` headers to
distinguish "skipped on purpose" from "missed".

The credit DB is a single-snapshot dataset: each customer has exactly
one row in each of the six chained tables, with no historical
versioning. KB entries that depend on a temporal trend (delta over
3 / 6 months, "increasing", "decreasing") therefore have nothing to
compute against and are deferred.

## KB 41 — Credit Utilization Alert

Reason: Definition requires "an increasing trend in utilization" and
limited available credit. The schema has only a single snapshot of
`credutil` and `totcredlimit` per customer — no history table — so the
"increasing" predicate cannot be evaluated. The non-trend portion
(`credutil > 0.8` AND `totcredlimit < mthincome * 2`) is fully
expressible at query time using already-encoded `credit_accounts_and_history.credutil`
and joined `mthincome`.
Status: deferred (schema gap — no time series).

## KB 47 — Declining Credit Health

Reason: Composite of three trends — negative CHM, CRI growth > 10% in
6 months, DTI rising > 5% in 6 months. CHM, CRI, and DTI are encoded
as point-in-time row-level columns on `credit_accounts_and_history`,
but their derivatives over time can't be computed against single-row
snapshot data.
Status: deferred (schema gap — no time series).

## KB 48 — Relationship Attrition Risk

Reason: Requires "decreasing produsescore" and "hardinq > 2 in past
3 months". The static-CRR component (CRR > 0.7) is already computed
via the `crr` column on `credit_accounts_and_history`, but the
"decreasing produsescore" delta and the 3-month windowing of `hardinq`
need historical observations the schema doesn't have.
Status: deferred (schema gap — no time series).
