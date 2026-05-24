# sports_events KB notes

The mini-interact `sports_events` SQLite contains the F1 schema reduced to:
`circuits`, `constructors`, `constructor_results`, `constructor_standings`,
`driver_standings`, `drivers`, `lap_times`, `pit_stops`, `qualifying`,
`races`, `sprint_results`. **There is no `results` table** — the canonical
upstream Ergast/F1 dataset has one (per-race per-driver finishing position,
classified status, total race time, fastest lap flag, etc.), but this
mini-interact slice does not.

That gap blocks every KB entry whose definition depends on a driver's
finishing position in the main Grand Prix race or on per-driver total
race time. Such entries are listed below as `Status: deferred — no
per-race driver finishing-position data in this DB`. They could be
encoded if the underlying data source were extended with a `results`
table.

## KB 8 — Championship Points System (Race)

Reason: maps race finishing position (1..10) to championship points
(25/18/15/12/10/8/6/4/2/1). The mini-interact `sports_events` schema
has no per-race per-driver finishing-position table (the upstream
`results` table is absent), so there is no row for the points lookup
to attach to as either an `R-CASE` column or a filtered measure.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 11 — Podium Finish

Reason: predicate on a driver's final race rank (1, 2, or 3). No
per-race driver finishing-position table.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 12 — Points Finish

Reason: depends on KB 8 (Championship Points System Race) — the top-N
finishing positions that earn points. Same blocker as KB 8.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 13 — Fastest Lap Award

Reason: definition requires both the per-driver fastest lap of the
race AND that the driver achieved a Points Finish (KB 12). The
fastest-lap component is recoverable from `lap_times`, but the
Points Finish gate is not because of the missing race-results data.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 16 — Race Winner

Reason: requires the driver classified 1st in the main Grand Prix
race. `constructor_standings.trophy_w` and `driver_standings.topmark`
are season-cumulative wins; race-level winners are not directly
queryable without the missing `results` table.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 18 — Hat Trick

Reason: composite of Pole Position (KB 10, encoded), Race Winner
(KB 16, deferred), and Fastest Lap Award (KB 13, deferred). Cannot
be computed without those two race-result-dependent components.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 19 — Constructor's Double Podium

Reason: requires both drivers from a constructor to achieve a Podium
Finish (KB 11) in the same race. Same blocker as KB 11.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 29 — Race Time Delta to Winner

Reason: requires per-driver total race time and the winner's total
race time. Neither is stored — `lap_times` carries individual lap
times, but the per-driver total / classified race time and the race
winner are not present.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 30 — Race Performance Index (RPI)

Reason: defined as `(21 - P_finish) + Position Gain / Loss`, both of
which depend on the missing per-race finishing-position data.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 33 — Adjusted Race Time Delta

Reason: builds on KB 29 (Race Time Delta to Winner). Pit-stop data
is available, but the underlying delta is not.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 38 — Team's Combined Race Result

Reason: sum of two teammates' race finishing positions. Per-race
driver-finishing-position data is missing.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 40 — Clutch Performer

Reason: requires Position Gain / Loss > 5 (KB 24, only available for
sprint sessions in this DB) AND a Podium Finish (KB 11, deferred) in
the same main Grand Prix race.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 42 — Dominant Victory

Reason: requires Race Time Delta to Winner > 5 s for the runner-up
(KB 29, deferred).

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 43 — Strategic Masterclass

Reason: requires Race Winner (KB 16) plus Efficient Pit Stop (KB 17,
encoded). Race Winner is deferred.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 44 — Grand Chelem

Reason: composite of Hat Trick (KB 18, deferred) and lap leadership
across every lap of the race. The lap leadership component could be
derived from `lap_times.pp = 1` for every lap, but the Hat Trick
gate is unreachable.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 46 — High-Altitude Ace

Reason: requires Race Performance Index (KB 30, deferred) at a
High-Altitude Circuit divided by the driver's seasonal-average RPI.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 47 — Underdog Win

Reason: requires Race Winner (KB 16, deferred) with a low PPR before
the event. PPR is encoded; Race Winner is not.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 48 — Flawless Team Weekend

Reason: composite of Pole Position (KB 10, encoded), Race Winner
(KB 16, deferred), and every pit stop of the car being an Efficient
Pit Stop (KB 17, encoded). Race Winner is the blocker.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 49 — Veteran's Podium

Reason: requires Driver Age >= 35 (KB 20, encoded) AND a Podium
Finish (KB 11, deferred).

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 51 — Pole-Based Race Win Probability

Reason: assigns a probability based on whether a Pole Position
starter wins the race (KB 16, deferred). Without Race Winner the
conditional cannot be evaluated against historical events.

Status: deferred — no per-race driver finishing-position data in this DB.

## KB 52 — Pole-Based Fastest Lap Probability

Reason: assigns a probability based on whether a Pole Position
starter sets the Fastest Lap (KB 13, deferred). Same blocker as KB 51.

Status: deferred — no per-race driver finishing-position data in this DB.
