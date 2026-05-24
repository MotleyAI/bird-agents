# museum — KB entries not encoded as model entities

KB ids in the headers below are deliberately omitted from
`slayer_models/museum/`. The verifier
(`scripts/verify_kb_coverage.py`) reads the `## KB <id> — ...` headers
to distinguish "skipped on purpose" from "missed".

A note on dataset reality: the `artifactsecurityaccess.insvalueusd`
column has a max value of ~$997,619 in the museum dataset, so the
"InsValueUSD > $1,000,000" leg of KB #11 / #51 / #55 / `is_arv`
never matches any row. Encoded anyway because the predicate is
well-defined.

## KB 39 — Environmental Compliance Index (ECI)

Reason: ECI = `10 - (|TempC - IdealTemp| + |RelHumidity - IdealHumidity|/5 + ERF/2)`.
KB defines neither IdealTemp nor IdealHumidity. KB #27 suggests ideal
temperature is 18-22 deg C (a range, not a point) and KB #28 suggests
45-55% RH (also a range). Picking a single value inside either range
would be a guess that biases the score. Helper columns `tempc`,
`relhumidity`, and `erf` are already encoded so the agent at query
time can pick midpoints (20 deg C / 50% RH) and compute ECI directly
via dimensions.

Status: deferred — AMBIGUOUS-PROSE

## KB 47 — Environmental Control Failure

Reason: ECI < 4 AND Environmental Instability Event. Depends on KB
#39 (ECI, deferred above). The Environmental Instability Event leg is
encoded as `environmentalreadingscore.is_env_instability_event`, but
without ECI the composite predicate cannot be expressed.

Status: deferred — AMBIGUOUS-PROSE
