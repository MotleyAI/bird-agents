# W4c: archeology

Workflow: see `_w4c_instructions/_README.md`.

## Hints

- Many KB definitions compose values from peer children of `scans`
  (e.g. `scanpointcloud`, `scanmesh`, `scanregistration`,
  `scanprocessing`, `scanspatial`, `scanfeatures`). These are the
  canonical R-PEER-JOIN cases — encode as R-MULTISTAGE via the shared
  parent (`scans`, sometimes via `sites` for site-grain metrics).
- Cascading composites (a KB whose definition references another
  KB's metric by name) resolve naturally once their parent KB does;
  encode the parent first, then reference the named measure in the
  child's formula.
