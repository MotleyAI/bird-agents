# W4c: news

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- The session-level KBs that combine a per-session aggregation over
  `interactions` (e.g. `AVG(seqval)` per session) with a
  row-level `sessions` column (e.g. `bncrate * (1 - ctrval/100)`)
  are the standard "aggregate-child-then-divide-by-parent-row" shape
  — encode as a two-stage R-MULTISTAGE: stage 1 aggregates
  `interactions` by session, stage 2 anchors at `sessions` and joins
  in the aggregate.
