# robot — KB entries not encoded as model entities

All 60 KB entries for the robot DB are now encoded as model entities.
There are no deferred KBs remaining; this notes file is intentionally
empty of `## KB <id>` sections so the verifier sees an empty
documented-set and the multi-KB audit reports zero plural entities.

The peer-table fan-out and per-robot-aggregate-divisor cases that
were previously deferred to W4b are now realized via two query-backed
models built on SLayer 0.5.0 multistage queries:

- `robot_per_robot` — one row per robot, joins per-child-table rollups
  (operation, performance_and_safety, joint_condition,
  joint_performance, actuation_data, maintenance_and_fault). Carries
  EER, TWR, OCE, JDI, MJT, APE, ATCS, MRUL plus the cascade
  classifications (is_energy_inefficient, is_joint_health_risk,
  is_tool_replace_*) and KB #50 (JDI-TOH regression slope), KB #52
  (EER Rank), KB #57/#58/#59 (model-series rollups).
- `program_efficiency` — one row per program, carrying avg_program_oce
  and the KB #54 dense_rank, plus KB #55 components
  (`is_most_efficient_program`, `avg_program_efficiency`).

KB #5 (RFPS) is encoded directly on `maintenance_and_fault` via a
correlated-subquery column; KB #53 (APE Rank) is a `percent_rank`
ModelMeasure on `actuation_data`. See those models' descriptions for
operational details.
