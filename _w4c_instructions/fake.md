# W4c: fake

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- Some peer-join composites span `account` / `profile` /
  `sessionbehavior` / `contentbehavior` / `messaginganalysis` /
  `moderationaction` / `securitydetection` / `networkmetrics` /
  `technicalinfo`. The canonical shared parent depends on the KB's
  grain (account, profile, or session); pick the model whose PK
  matches the KB's predicate grain and route via R-MULTISTAGE.
- Cluster-grain predicates (e.g. "cluster of size > N where average
  X > Y") need an aggregation over a cluster identifier (e.g.
  `clustsize` on `moderationaction`); encode as R-MULTISTAGE with
  the cluster id as the stage 1 dimension.
- KBs framed as "value at the row with the latest detection time"
  per group are the per-group argmin/argmax-by-time pattern — use
  the built-in `last(<col>)` transform with the model's time
  dimension set appropriately. No window function needed.
- KBs framed as "quartile of X" without specifying that X is
  pre-aggregated over a partition: read the KB carefully — the
  intended computation may be a count-based bucket arithmetic
  (rank-position divided by total distinct values) rather than a
  true NTILE. Encode as R-MULTISTAGE with the rank divided by the
  count.
