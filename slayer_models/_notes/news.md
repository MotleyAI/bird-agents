# news — KB entries not encoded as model entities

KB ids in the headers below are deliberately omitted from
`slayer_models/news/`. The verifier (`scripts/verify_kb_coverage.py`)
reads the `## KB <id> — …` headers to distinguish "skipped on purpose"
from "missed".

The news DB encodes most calculation_knowledge entries as row-level
columns or measures on the host table (sessions, articles, users,
systemperformance, interactions). The deferrals below cluster around
two shapes:

- **Composites that bottom out on RRS (KB #2)**, whose definition
  averages per-recommendation scores (`recscore`, `confval`) with a
  per-session score (`recutil`). The KB is silent on which grain the
  average is taken at, so the formula isn't well-defined as a row-level
  column nor as a single named measure.
- **Domain-knowledge prose** (`type=domain_knowledge`) that names a
  category or principle without giving a discriminating predicate
  (no thresholds, no operationalised classifier).

## KB 2 — Recommendation Relevance Score (RRS)

Reason: Definition is `RRS = (recscore + confval + recutil) / 3`.
`recscore` and `confval` live on `news_recommendations` (one row per
shown recommendation); `recutil` lives on `news_sessions` (one row per
session). The KB does not specify which grain the average operates at
(per-recommendation with `recutil` broadcast from the parent session,
or per-session with `recscore`/`confval` first averaged across the
session's recommendations). Either choice is a guess; agents can
compose the per-grain version inline at query time once they pick
one.
Status: deferred — AMBIGUOUS-PROSE

## KB 32 — Dynamic Content Value (DCV)

Reason: `DCV = (AQI + RRS + (100 - ARS)) / 3`. AQI and ARS are encoded
as columns on `news_articles`, but RRS (KB #2) is deferred — its grain
isn't pinned down — so the composite inherits the ambiguity. Once the
RRS grain is decided, DCV becomes a straightforward
arithmetic-on-articles column.
Status: deferred — AMBIGUOUS-PROSE

## KB 33 — Optimized Recommendation Score (ORS)

Reason: `ORS = RRS × SPI / k` with `k` declared a "normalization
constant" but never quantified in the KB. Both the RRS grain (KB #2)
and the value of `k` are unknowns, so the formula cannot be encoded
faithfully.
Status: deferred — AMBIGUOUS-PROSE

## KB 34 — Adjusted Read Time Estimator (ARTE)

Reason: `ARTE = readsec × ARS / UER`. `readsec` and ARS are
article-row attributes; UER is a session-row attribute. The natural
articles↔sessions bridge runs through `news_interactions` (article id
is non-FK on interactions, sessions joined via `seshlink2`), and the
KB is silent on how to aggregate UER across the sessions that touched
each article (or, conversely, how to broadcast article-level readsec
to sessions). Either direction is a guess.
Status: deferred — AMBIGUOUS-PROSE

## KB 35 — Personalization Accuracy Metric (PAM)

Reason: `PAM = (RRS + PP) / 2`. RRS (KB #2) is deferred for grain
ambiguity. PP (KB #11) is `domain_knowledge` prose with no
predicate or numeric form, so it cannot be combined arithmetically.
Status: deferred — AMBIGUOUS-PROSE

## KB 36 — Conversion Impact Factor (CIF)

Reason: `CIF = UER / ITI`. UER (KB #0) is encoded as a row-level
column on `news_sessions`. ITI (KB #19) is `domain_knowledge` prose
("a minimal delay between content exposure and user interaction…")
with no quantified definition — there is no `iti` column to divide
by.
Status: deferred — AMBIGUOUS-PROSE

## KB 38 — Composite System Stability (CSS)

Reason: `CSS = sqrt(SPI × DPM)`. SPI (KB #4) is encoded on
`news_systemperformance`. DPM (Device Performance Metric) is referenced
but not defined anywhere in the KB and there is no `dpm`-shaped
column or formula on `news_devices` or `news_systemperformance`.
Status: deferred — SCHEMA-GAP

## KB 39 — Interactive Content Amplifier (ICA)

Reason: `ICA = CIE + α × RRS`. RRS (KB #2) is deferred for grain
ambiguity, and α is left as an unspecified weighting constant in the
KB. Both unknowns sink the formula.
Status: deferred — AMBIGUOUS-PROSE

## KB 40 — High Engagement Indicator (HEI)

Reason: Definition is "marked as high engagement when UER is above
the threshold and CIE is high". The KB names no threshold for either
factor, so any encoded predicate would be a guess. UER and CIE are
both available; agents can compose the predicate inline once they
pick a threshold.
Status: deferred — AMBIGUOUS-PROSE

## KB 41 — Premium Article Distinction (PAD)

Reason: "Article qualifies as premium when it surpasses the AQI
threshold and complies with the Premium Content Rule". AQI (KB #1)
is encoded as a column on `news_articles`, but the AQI threshold
is unspecified and the Premium Content Rule (KB #10) is itself prose
with no operational predicate.
Status: deferred — AMBIGUOUS-PROSE

## KB 42 — Targeted Personalization Benchmark (TPB)

Reason: Combines PP (KB #11, prose) with PAM (KB #35, deferred). No
quantified threshold given for either component.
Status: deferred — AMBIGUOUS-PROSE

## KB 43 — User Churn Predictor (UCP)

Reason: "Inconsistent engagement per ECP combines with anomalies
detected by SSAD". ECP (KB #13) and SSAD (KB #18) are both
`domain_knowledge` prose with no operational thresholds — no
inconsistency band, no anomaly cutoff is defined.
Status: deferred — AMBIGUOUS-PROSE

## KB 44 — Content Virality Threshold (CVT)

Reason: "Content is viral when DCV and ICA both exceed virality
thresholds". DCV (KB #32) and ICA (KB #39) are themselves deferred,
and the thresholds are unspecified.
Status: deferred — AMBIGUOUS-PROSE

## KB 45 — System Resilience Factor (SRF)

Reason: "System demonstrates resilience when CSS and SPI both
indicate strong performance". CSS (KB #38) is deferred for a missing
DPM column; "strong performance" thresholds for either factor are
not specified.
Status: deferred — AMBIGUOUS-PROSE

## KB 46 — Session Drop-off Risk (SDR)

Reason: "High drop-off risk when low RTSE coincides with elevated
ABR". RTSE (KB #31) and ABR (KB #37) are both encoded as
query-backed multistage models, but the KB names no thresholds for
"low" or "elevated" — agents must compose the predicate inline once
they pick cutoffs.
Status: deferred — AMBIGUOUS-PROSE

## KB 47 — Conversion Potential Indicator (CPI)

Reason: "Both CIF and ICA are optimized" — neither component has a
defined threshold; CIF (KB #36) and ICA (KB #39) are themselves
deferred.
Status: deferred — AMBIGUOUS-PROSE

## KB 48 — Content Consumption Consistency (CCC)

Reason: ARS, UER, and PP "together display steady patterns". No
operational definition of "steady" (variance band, coefficient of
variation, n-session window).
Status: deferred — AMBIGUOUS-PROSE

## KB 49 — Subscription Valuation Rule (SVR)

Reason: "Comprehensive user valuation is achieved by weighting
subscription duration, engagement performance, and demographic
indicators". The KB lists ingredients (USV, UDS, plus an unspecified
engagement factor) but gives no formula or weights — purely
descriptive prose.
Status: deferred — AMBIGUOUS-PROSE
