# W4c: vaccine

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- Many KBs compose per-container metrics across peer children of
  `container` (e.g. `vaccinedetails`, `sensordata`, `transportinfo`).
  These are the canonical R-PEER-JOIN cases — encode as R-MULTISTAGE
  via `container`.
- Some KBs scope their aggregation to a rolling time window
  (e.g. "average TSS over the past 1 year"). Encode the time
  predicate as a stage-level filter using the dialect's date math
  (e.g. `alerttime >= date('now', '-1 year')`); if the agent prefers
  to surface the anchor as a query variable, encode under R-VAR
  with `{anchor_date}` and default it to "now".
- Cascading composites resolve once their parents do.
