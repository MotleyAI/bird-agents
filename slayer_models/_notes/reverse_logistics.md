# reverse_logistics — KB entries not encoded as model entities

KB ids in the headers below are deliberately omitted from
`slayer_models/reverse_logistics/`. The verifier
(`scripts/verify_kb_coverage.py`) reads the `## KB <id> — …` headers to
distinguish "skipped on purpose" from "missed".

## KB 3 — Customer Return Frequency Index (CRFI)

Reason: CRFI = T_r / T_y (Total Returns / Customer Tenure in years).
Total Returns is encoded (`customers.total_returns`, KB #17), but
Customer Tenure (KB #18) cannot be computed from the supplied data —
`orders.txndate` is `None` for every row in the sqlite (1488/1488). A
ratio whose denominator is undefined for every customer is undefined
itself; encoding CRFI as a column would emit a divide-by-NULL on every
row, masking the upstream gap.

Status: deferred — SCHEMA-GAP. Once `orders.txndate` is populated,
encode CRFI as an R-MULTISTAGE model: stage 1 aggregates per-customer
`min(txndate)` -> tenure_years; stage 2 joins customers + stage 1 and
computes `total_returns / tenure_years`.

## KB 18 — Customer Tenure (Years)

Reason: Defined as "(current date - first recorded transaction) /
365". The only candidate source field is `orders.txndate`, but every
row in the supplied sqlite has `txndate = None` (1488/1488). With no
first-transaction date, tenure cannot be computed.

Status: deferred — SCHEMA-GAP. Once `orders.txndate` is populated,
encode as a column on `customers` derived via the `rl_orders` join, or
as an R-MULTISTAGE model aggregating per-customer `min(txndate)` and
joining back to customers.
