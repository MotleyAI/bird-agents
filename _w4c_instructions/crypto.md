# W4c: crypto

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- The schema has a `users` table that the prior pass's notes file
  may have missed. `orders.userlink → users.userstamp` and
  `accountbalances.usertag → users.userstamp` both exist, so KBs
  that compose per-order data (via `riskandmargin → orders`) with
  per-account data (`accountbalances`) **are** peer-join via
  `users`; encode as R-MULTISTAGE through that shared parent.
- KBs that need a "latest market state" scalar (whatever
  `marketstats` row has the maximum `marketstatsmark`) are not a
  schema gap — encode as a multistage with a 1-row scalar stage
  cross-joined to the per-row data. Pattern: stage 1 has
  `order=[{column: marketstatsmark, direction: desc}], limit=1` over
  `marketstats`; stage 2 cross-joins it to the row-level model.
- KBs that depend on future-time price movement (e.g. "smart-money
  accuracy" comparing current vs N-period-ahead price) are encodable
  via the built-in `lead` transform on a per-(market_pair) ordering.
  Not a schema gap.
