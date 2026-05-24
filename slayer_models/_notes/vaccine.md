# vaccine — KB entries not encoded as model entities

KB ids in the headers below are deliberately omitted from
`slayer_models/vaccine/`. The verifier
(`scripts/verify_kb_coverage.py`) reads the `## KB <id> — …` headers to
distinguish "skipped on purpose" from "missed".

The remaining unencoded entries fall into two groups:

1. **Underspecified weights** — KBs whose definitions reference free
   parameters (`alpha`, `beta`, `gamma`) or normalisation constants
   (`CDR_max`) the KB never pins. Encoding any one choice would be a
   guess that misleads downstream agents; these are deferred until the
   KB itself fixes the parameter or the agent passes it as a query
   variable.

2. **Sequence / window predicates** — KBs whose definitions say
   "increases over N consecutive readings" or "decreases over N
   consecutive readings". SLayer's window transforms cover ranks /
   percentiles / LAG / FIRST_VALUE, but a "monotone over N points"
   predicate is a multi-LAG inequality chain that no single transform
   currently expresses. These cascade-defer through any composite that
   names them.

## KB 50 — Thermal Stability Coefficient (TSC)

Reason: TSC = TSS * exp(-|TempNowC - StoreTempC|/5) * (1 - alpha *
(TempNowC - TempPrevC)/ReadingInterval). The factor `\alpha` has no
default in the KB — picking 1.0 would silently bake an arbitrary
weight into every downstream metric. TempPrevC also requires a
LAG(tempnowc) over sensordata ordered by alerttime; ReadingInterval
is fixed at 15 minutes per KB #60. Once alpha is pinned, the encoding
is a one-stage R-MULTISTAGE / R-WINDOW combination.

Status: deferred — AMBIGUOUS-PROSE

## KB 51 — Multi-Parameter Risk Assessment (MPRA)

Reason: MPRA = sqrt(CRI^2 + TBS^2 + (1 - HQI)^2) * (1 + CDR/CDR_max).
CRI/TBS/HQI/CDR are all encoded, but `CDR_max` is a normalisation
constant the KB never defines — could be the dataset-wide max, the
per-container max, or a fixed scalar (e.g. "the worst observed CDR in
the test fleet"). Each choice changes the metric's scale and ranking
behaviour materially. Once the scope is pinned, the encoding is a
two-stage R-MULTISTAGE (max(CDR) at the chosen grain, then row
arithmetic).

Status: deferred — AMBIGUOUS-PROSE

## KB 52 — Time-Weighted Quality Decay (TWQD)

Reason: TWQD = -d/dt(VVP) * (1 + beta*TBS) * (1 + gamma*(1-TSS)). VVP
(KB #3) is encoded, TBS / TSS are encoded, but `\beta` and `\gamma`
are unspecified weights and `-d/dt(VVP)` is a finite-difference rate
across consecutive sensordata readings. With both weights and the
sampling interval pinned, the encoding is a multistage with a LAG
plus row arithmetic.

Status: deferred — AMBIGUOUS-PROSE

## KB 55 — Critical Cascade Condition

Reason: predicate over MPRA (KB #51), TSC (KB #50), TWQD (KB #52),
ESF (KB #53). MPRA / TSC / TWQD are all deferred (KBs 50–52); the
composite resolves automatically once those parameters are pinned.

Status: deferred — AMBIGUOUS-PROSE (cascades from KBs 50/51/52)

## KB 56 — Compound Quality Risk

Reason: "VSI decreases over three consecutive readings AND LPM < 0.5
AND ESF > 0.6". VSI (KB #39), LPM (KB #54), and ESF (KB #53) are
encoded, but "decreases over three consecutive readings" needs a
multi-LAG monotonicity predicate over sensordata ordered by alerttime
that no single SLayer transform currently expresses. Encodable as a
multistage with two LAG calls and explicit inequality chain once
that pattern is added to the recipe book.

Status: deferred — AMBIGUOUS-PROSE

## KB 57 — Dynamic Stability Threshold

Reason: "average TSC over last 5 readings < 0.7 AND MPRA > 0.6".
Depends on KB #50 (TSC) and KB #51 (MPRA), both deferred.

Status: deferred — AMBIGUOUS-PROSE (cascades from KBs 50/51)

## KB 58 — Multi-System Failure Risk

Reason: "(CRI + TBS + (1 - HQI))/3 > 0.7 AND all of (TSC, LPM, ESF) <
0.3". Depends on KB #50 (TSC), which is deferred. CRI/TBS/HQI/LPM/ESF
are all encoded, so once TSC has a fixed alpha the predicate becomes
a single CASE WHEN over a per-container join.

Status: deferred — AMBIGUOUS-PROSE (cascades from KB 50)

## KB 59 — Predictive Degradation Alert

Reason: "TWQD increases over 3 consecutive readings AND ESF > 0.5 AND
TempDevCount > 3". Depends on KB #52 (TWQD, deferred) plus the same
multi-LAG monotonicity pattern flagged for KB 56.

Status: deferred — AMBIGUOUS-PROSE (cascades from KB 52)
