# polar — KB entries not encoded as model entities

All but one KB entry is encoded as model entities under
`slayer_models/polar/`. The one deferred entry is below.

## KB 41 — Emergency Response Readiness Status (ERRS)

Reason: KB requires `emergencycommunicationstatus = 'Operational'`
as one of the conjuncts. No column with that name (or close
synonym for an emergency-channel/comm operational flag) exists in
the polar schema; the schema's emergency-related columns are
`emergencybeaconstatus` (cabinenvironment), `emergencystopstatus`
and `emergencylightstatus` (lightingandsafety) — none of which
captures the same semantic. Without that conjunct, the predicate
cannot be encoded faithfully. The other conjuncts (OSPI > 0.75,
LSSR > 0.8, backuppowerstatus = 'Active', batterystatus.level_percent > 85)
are all reachable via existing encoded entities + JSON extracts.

Status: deferred — SCHEMA-GAP

## KB 50 — Extreme Weather Readiness Status (EWRS)

Reason: Verbatim restatement of KB 10; encoded entity is `extreme_weather_ready.extreme_weather_ready` with `meta.kb_id = 10`.

Status: not-applicable — duplicate of KB 10
