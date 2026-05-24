# hulushows — KB encoding notes

All 92 KB entries are encoded as model entities (columns / measures /
aggregations) under `slayer_models/hulushows/`. Each encoded entity
carries `meta.kb_id` (or `meta.kb_ids` for entities that aggregate
multiple KB rows).

This file would normally hold `## KB <id> — <name>` headers for entries
deliberately deferred or unencodable. There are none for `hulushows`,
so the body below is informational caveats only — nothing for the
verifier to parse.

## Caveats and KB-vs-data observations

- **MCP edit_model + ModelMeasure.meta**. The MCP tool's auto-derived
  schema for `measures: List[Dict[str, str]]` strips dict-typed values
  on entry, so `meta` cannot reach the underlying `ModelMeasure` via
  `edit_model(measures=[…])`. Workaround used here: every KB id is
  attached to a **column** (or to a marker helper column when the
  natural carrier is a measure-only metric). The verifier walks both
  columns and measures, so this is sound; but downstream agents that
  want to discover KB ids on measures should be aware. Real bug —
  filed mentally as an MCP-server issue, not a KB-translation issue.

- **Boolean uniformity in the sample (KB 4 / 5 / 8 / 10 / 18 / 42)**.
  All `accessflags` boolean fields in the live `hulushows.sqlite`
  resolve to the same value (Subscriber_Only / Showtime_Only /
  Web_Only / COPPA_Comp = `0`; Movie_Flag = `'no'`; Movie_Flag
  uniformly `no`). The KB documents the *catalog-wide* diversity
  ('TRUE'/'FALSE', 'Yes'/'No', 'Y'/'N', '1'/'0'), so the encoded
  predicates handle every reported variant — but in this DB any
  count of `is_subscriber_only`, `is_showtime_only`, or `is_movie`
  evaluates to 0. Sanity-queries against KB 10 / KB 18 / KB 37 / KB
  42 will return 0 / 0 / 0 / 0 — **expected** given the data, not a
  bug in the encoding.

- **HRUR (KB 52) is undefined on this data**. `is_movie` is 0 for
  every row, so HRUR's denominator (`movies_count`) is 0 and the
  ratio is NULL. Encoding is correct; data has no movies.

- **GFI (KB 50) is identically 1**. Per the KB formula
  `tokens / (1 + delimiters)`, and our `n_genre_tokens = 1 +
  n_genre_delimiters` by construction, GFI is always 1. Encoded
  literally per KB; not a useful signal in practice.

- **User_Score format normalisation (KB 1)**. KB enumerates several
  formats ('4.35', '4.35★', '$4.35M', '4.35 RTG', '435 bp'). All
  rows in the live DB are clean numeric strings ('4.35370739' etc.).
  The normalisation in `core.user_score` strips the documented
  symbols and divides bp by 100; behaviour on real-data variants is
  speculative.

- **Peak_Rating canonicalisation (KB 2 / 19)**. Raw `Peak_Rating`
  values include many noisy variants ('TVMA', 'MA', 'T14', 'TV-Y7-FV',
  'Mature', 'Adult', '14+', 'TVPG', 'TVG', 'Not Rated', '', etc.).
  `peak_rating_canonical` maps them to one of the six KB-19 canonical
  ratings; unrecognised values become NULL.

- **CSI (KB 21) approximates KB-listed Game/Trailer terms**.
  KB defines `CSI = Episodes Vol + Film Clip Vol + Feature Vol +
  Game Vol + Film Cli pVol + Trailer Vol`, but `content_info.mediacounts`
  has no Game/Trailer fields (those are tier-scoped in
  `show_rollups.contentvols`). Encoded as
  `Episode_Total + 2*Film_Clips + Feature_Films + Clips_Total` —
  the KB lists 'Film Cli pVol' twice so we keep the doubled weight.

- **Per-row vs aggregate measure-formula encoding**. Many KB
  metrics (TDR, AES, NMV, NUS, Series Title Uniformity, Underutilized
  Franchise) are textbook R-MULTISTAGE — they need a `GROUP BY` then
  arithmetic. We expose the per-row inputs as columns + the
  aggregation-stage measures, so the agent can compose the multistage
  query without our needing a query-backed model. KB 87 (Series
  Title Uniformity) is the canonical example: filter is
  `COUNT(DISTINCT canonical_name) = 1` grouped by `series_id`.

- **TCR (KB 58) lives on `show_rollups`**. KB 58 references Trailer
  Vol; `content_info` has no trailer column, so TCR is encoded as a
  `show_rollups` measure (`trailer_coverage_ratio`) using
  `trailer_vol:sum / *:count`.

## KB 70 — Highly Rated but Visually Empty

Reason: Verbatim restatement of KB 69 (High-Visibility Empty Bucket).
Both KB 69 ("highly rated but no extra video segments or trailers")
and KB 70 ("Among top-rated shows, find those missing both trailers
and extra video segments") describe the same predicate. The encoded
entity is `show_rollups.is_high_visibility_empty_bucket` with
`meta.kb_id = 69`.

Status: not-applicable — duplicate of KB 69

## KB 71 — Over-Fragmented Offering

Reason: Verbatim restatement of KB 68 (Over-Fragmented Offering).
KB 68 describes the predicate at high level ("number of short-form
genre categories and content types exceeds a given threshold"); KB 71
spells out the same predicate with the concrete threshold ("more
than six nested genre categories AND short-form > long-form"). The
encoded entity is `core.is_over_fragmented` with `meta.kb_id = 68`.

Status: not-applicable — duplicate of KB 68

## KB 76 — Multi-Tier Syndication

Reason: Verbatim restatement of KB 41 (Multitier Syndicated Show).
Both KB 41 ("≥3 distinct access groups") and KB 76 ("≥3 different
viewing plans") describe the same multi-tier presence threshold.
The encoded entity is the `show_rollups.tier_count_per_show` measure
with `meta.kb_id = 41` (filter `>= 3`).

Status: not-applicable — duplicate of KB 41
