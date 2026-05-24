# W4c: virtual

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- KBs of the shape "fans (or any actor) who span ≥ N distinct
  related entities, each with some per-relationship measure above
  threshold" encode as a two-stage R-MULTISTAGE: stage 1 aggregates
  the per-(actor, related-entity) data with a HAVING-style filter
  on the per-relationship measure; stage 2 anchors at the actor,
  joins stage 1, and applies a `count_distinct` plus a HAVING-like
  filter via `*:count` ≥ N (the colon-syntax `count_distinct`
  aggregation).
