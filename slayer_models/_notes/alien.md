# alien — KB entries not encoded as model entities

Most cross-peer composites (e.g. ECI, Technosignature, CCS) are
encoded on the query-backed `signal_full` model, which joins every
per-signal child table plus the signals → telescopes → observatories
chain so a single Column.sql can reference any of those columns.
That model's `meta.kb_ids` lists every KB id it covers.

The single-host calculations and value illustrations are encoded in
place on the relevant model:

- `signals` (host of SNQI, BFR, SSM, MCS, NTM, NTM classification,
  is_analyzable, plus the SignalClass=Narrowband / PolarMode=Circular
  value illustrations).
- `observatories` (host of AOI, DISF, LIF, OOW, SDS, high-LIF, plus
  the WeathProfile=Clear / SeeingProfile=Excellent / GeomagStatus=Storm
  value illustrations).
- `telescopes` (host of OQF, OCL, has_equipment_problem; reaches
  observatories via the existing FK).
- `signalprobabilities` (host of TOLS, RPI, TOLS Category, is_too,
  is_potential_biosignature, plus the FalsePosProb<0.01 illustration).
- `signalclassification` (host of SCR and PRC, plus the SigClassType
  =Broadband Transient illustration).
- `signaldecoding` (host of the EncodeType=Frequency Hopping
  illustration).
- `signaladvancedphenomena` (host of the EncryptEvid=Strong Pattern
  illustration).
- `sourceproperties` (host of CLSF).

Only one KB id is left unencoded:

## KB 44 — Research Critical Signal

Reason: definition references **IMDF < 0.5** in addition to
TOO + PRC > 0.8. There is no `imdf` column anywhere in the alien
schema (no obvious source column for an Information / Modulation
Distortion Factor either — the closest candidates `interflvl`,
`atmointerf`, and `sigdisp` are all categorical strings, not the
numeric distortion measure the formula expects). The TOO + PRC
clauses are encodable, but without IMDF the predicate is incomplete
and would silently always pass the third clause; encoding it would
be misleading rather than helpful.

Status: deferred — missing source column (IMDF). If a downstream
task supplies a mapping (e.g. IMDF := some derived combination of
sigdisp + interflvl bands), this can be added on `signal_full`
following the same multi-peer pattern as the other cascades.

## KB 51 — Bandwidth-to-Frequency Ratio (BFR)

Reason: Verbatim restatement of KB 4; encoded entity is `signals.bfr` with `meta.kb_id = 4`.

Status: not-applicable — duplicate of KB 4
