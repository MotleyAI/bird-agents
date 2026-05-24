# W4c: reverse_logistics

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- KBs of the shape "row's value relative to the average over its
  peer-group" (e.g. "shipping fee divided by the channel-average
  shipping fee") encode as a two-stage R-MULTISTAGE: stage 1 = avg
  per group (e.g. `return_channel`); stage 2 = source rows joined
  to stage 1, formula = `row_value / stage1.group_avg`.
