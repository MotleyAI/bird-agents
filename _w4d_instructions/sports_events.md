# W4d: sports_events

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### average_stops_per_car.average_stops_per_car  (model)

- Current `meta.kb_ids`: `[54, 55]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 54 [calculation_knowledge] "Average Stops Per Car"
    def: For a given race, this is calculated as: (Total Number of Pit Stops) / (Number of Unique Cars that made a Pit Stop).
  - KB 55 [domain_knowledge] "Pit Strategy Cluster"
    def: Three categories: 'Single-Stop Race' (<1.5 stops), 'Standard Two-Stop' (1.5-2.5 stops), 'High-Strategy Event' (≥2.5 stops)

### circuits.location_metadata  (column)

- Current `meta.kb_ids`: `[3, 4]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 3 [value_illustration] "Data Unavailability for Circuit Location"
    def: When a circuit's city, latitude, or longitude are not provided, it indicates that this information was not supplied or is unknown in the source data feed.
  - KB 4 [value_illustration] "Data Unavailability for Circuit Elevation"
    def: When a circuit's elevation in meters is not provided, it signifies that this specific data point was not recorded for the circuit.

### qualifying_with_pole_deficit.qualifying_with_pole_deficit  (model)

- Current `meta.kb_ids`: `[28, 41, 53]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 28 [calculation_knowledge] "Qualifying Time Deficit to Pole"
    def: \Delta_{qualifying} = T_{driver} - T_{pole}, \text{where } T_{driver} \text{ is the driver's best Lap Time in Seconds during qualifying, and } T_{pole} \text{ is the time achieved by the driver in Pole Position.}
  - KB 41 [domain_knowledge] "Qualifying Specialist"
    def: A driver is a Qualifying Specialist if their Qualifying Time Deficit to Pole is less than 0.2 seconds.
  - KB 53 [domain_knowledge] "Qualifying Performance Cluster"
    def: Three tiers: 'Pole Threat' (<0.15s), 'Mid Gap' (0.15s-0.4s), 'Backmarker' (≥0.4s)

### races.event_schedule  (column)

- Current `meta.kb_ids`: `[0, 1, 2, 7]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 4 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 0 [value_illustration] "Race Weekend Structure"
    def: A typical race weekend consists of several sessions: up to three Free Practice sessions for teams to tune their cars, a Qualifying session to determine the starting order for the main race, and the Grand Prix Race itself. Some weekends also include a Sprint session.
  - KB 1 [value_illustration] "Qualifying Format Explained"
    def: Qualifying is divided into three parts: Q1, Q2, and Q3. In Q1, all drivers compete, and the slowest are eliminated. The remaining drivers proceed to Q2, where more are eliminated. The final top drivers advance to Q3 to compete for Pole Position.
  - KB 2 [value_illustration] "Sprint Session Explained"
    def: A Sprint is a shorter race held on some race weekends. It has its own abbreviated qualifying and awards fewer championship points than the main Grand Prix. Its result determines the starting grid for the main race. The inclusion of a Sprint session modifies the standard Race Weekend Structure.
  - KB 7 [value_illustration] "Indeterminate Event Timings"
    def: When the date or time for any session (practice, qualifying, sprint, or race) is not provided, it signifies that the schedule for that session is To Be Determined (TBD), not applicable for the event, or not yet published.
