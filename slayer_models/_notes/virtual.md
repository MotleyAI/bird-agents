# virtual — KB entries not encoded as model entities

KB ids in the headers below are deliberately omitted from
`slayer_models/virtual/`. The verifier
(`scripts/verify_kb_coverage.py`) reads the `## KB <id> — …` headers to
distinguish "skipped on purpose" from "missed".

## KB 43 — Quality Inconsistent Creator

Reason: predicate is "has at least one content piece with
contqualrate > 8.5 BUT overall CQC < 5". The schema stores a single
aggregate `contqualrate` per fan in
`socialcommunity.community_engagement.content_creation.contqualrate`
— there is no per-content-piece grain table. The "at least one
piece" predicate cannot be expressed against the available schema.

Status: deferred — SCHEMA-GAP
