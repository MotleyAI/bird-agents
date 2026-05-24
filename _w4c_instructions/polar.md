# W4c: polar

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- Many KBs compose per-equipment metrics across peer children of
  `equipment` (e.g. `thermalsolarwindandgrid`, `waterandwaste`,
  `cabinenvironment`, `lightingandsafety`, `operationmaintenance`,
  `weatherandstructure`, `powerbattery`, `communication`). These
  are the canonical R-PEER-JOIN cases — encode as R-MULTISTAGE via
  `equipment`.
- All peer children of `equipment` happen to be 1:1 in this DB, so
  the multistage aggregations (`x:avg`, `x:max`, etc.) collapse
  trivially per equipment row, but use the multistage pattern
  anyway — it's the canonical encoding and matches the skill's
  default.
- Cascading composites and "equivalent to KB #N" duplicates resolve
  once their parents do.
