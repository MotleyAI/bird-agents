# W4d: hulushows

Workflow: see `_w4d_instructions/_README.md`.
Skill order: read `kb-to-slayer-models` (recipes incl. R-SPLIT-* in "Splitting multi-KB entities"), then `translate-mini-interact-kb` (W4d refresh override).

## Multi-KB entities to split

### availabilitys.accessflags  (column)

- Current `meta.kb_ids`: `[4, 5, 8]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 4 [value_illustration] "High-Resolution Logo Flags"
    def: Typical values include 'TRUE', 'FALSE', 'Yes', 'No', '1', '0' and all indicate presence or absence of a high-resolution logo.
  - KB 5 [value_illustration] "Movie Identifier Formats"
    def: Can appear as 'TRUE', 'FALSE', 'Y', 'N', 'Movie', 'Series'. These values signal whether content is a movie.
  - KB 8 [value_illustration] "Boolean Value Variants"
    def: Can appear as strings ('Yes', 'No'), booleans ('TRUE', 'FALSE'), or numerics ('1', '0').

### availabilitys.all_boolean_values_concat  (column)

- Current `meta.kb_ids`: `[54, 61]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 54 [calculation_knowledge] "Boolean Value Redundancy Rate (BVRR)"
    def: BVRR = \frac{\text{Unique Representations}}{\text{Total Boolean Fields}}, by comparing the variety of true/false indicators used
  - KB 61 [domain_knowledge] "Redundant Boolean Format"
    def: Identified by BVRR > threshold, by checking for repeated ways of marking true or false values

### availabilitys.boolean_unique_values_per_field_helper  (column)

- Current `meta.kb_ids`: `[54, 61]`
- Bucket: **A** — R-SPLIT-CALC-THRESH (calc + threshold/classification)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 54 [calculation_knowledge] "Boolean Value Redundancy Rate (BVRR)"
    def: BVRR = \frac{\text{Unique Representations}}{\text{Total Boolean Fields}}, by comparing the variety of true/false indicators used
  - KB 61 [domain_knowledge] "Redundant Boolean Format"
    def: Identified by BVRR > threshold, by checking for repeated ways of marking true or false values

### core.series_id  (column)

- Current `meta.kb_ids`: `[13, 49, 66, 72, 73, 85, 86, 87]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 8 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 13 [domain_knowledge] "Franchise Group"
    def: Content entries recognized as part of the same story world or universe are grouped together as a franchise.
  - KB 49 [domain_knowledge] "Underutilized Franchise"
    def: Franchises with ≥ 3 content items and aggregate Media Total < 10 are underutilized.
  - KB 66 [domain_knowledge] "Rating Inconsistency in Franchise"
    def: Franchise Groups where the range of rating types is very wide are labeled inconsistent
  - KB 72 [domain_knowledge] "Franchise Engagement Summary"
    def: For each franchise group, compute the number of shows and the total episode count by summing up values across all entries that share the same franchise.
  - KB 73 [domain_knowledge] "Syndicated Franchise Engagement"
    def: Franchises that have at least 3 shows and are available in 3 or more unique subscription tiers.
  - KB 85 [calculation_knowledge] "Series Entry Count"
    def: For each series, we count how many unique shows or episodes it includes. This helps show how big or ongoing the series is.
  - KB 86 [calculation_knowledge] "Series Size"
    def: This tells us how many titles are part of a single series, giving a sense of how large or deep a series is.
  - KB 87 [calculation_knowledge] "Series Title Uniformity Flag"
    def: If every title in a series uses the exact same name, it’s marked as 'True'. If they don’t match, it’s marked as 'False'.

### core.studiolink  (column)

- Current `meta.kb_ids`: `[84, 88, 89]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 84 [calculation_knowledge] "Studio Activity Index"
    def: This measures how many different shows a studio is involved in, giving a sense of how active or prolific each studio is.
  - KB 88 [calculation_knowledge] "Studio Catalog Size"
    def: This counts how many different shows are linked to each studio, showing how big their catalog is.
  - KB 89 [calculation_knowledge] "Title Count per Studio"
    def: We total up how many shows are tied to each studio, to find out which ones have produced the most content.

### core.genreclass  (column)

- Current `meta.kb_ids`: `[0, 1, 6, 74]`
- Bucket: **B** — R-SPLIT-ILLUSTRATION (one helper column per illustrated sub-field; JSON blob itself stays untagged per kb_id placement rule)
- Target: produce 4 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 0 [value_illustration] "Content Type Labels"
    def: Includes 'show', 'movie', and 'special' to distinguish the type of content displayed.
  - KB 1 [value_illustration] "User Score Formats"
    def: Examples include raw numbers like '4.35', symbols like '4.35★', monetized ratings like '$4.35M', and normalized formats like '4.35 RTG'.
  - KB 6 [value_illustration] "Genre Hierarchy Format"
    def: Uses '~' for sub genres and '|' for alternatives. Example: 'Comedy~Sitcom|Teen'.
  - KB 74 [value_illustration] "Primary Genre Classification"
    def: Examples include 'Drama', 'Comedy', 'Documentary', 'Reality', and 'Animation'. These genres can appear in the genre metadata of content records.

### core.user_score  (column)

- Current `meta.kb_ids`: `[28, 30, 34, 53, 78, 83]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 6 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 28 [calculation_knowledge] "Normalized User Score"
    def: NUS = \frac{score - min}{max - min}
  - KB 30 [calculation_knowledge] "Average Episode Rating (AER)"
    def: AER = The average rating across all episodes of a show is calculated by taking all their ratings and finding the mean.
  - KB 34 [calculation_knowledge] "User Score Dispersion (USD)"
    def: USD = User score dispersion is the variance in episode ratings for a show.
  - KB 53 [calculation_knowledge] "Rating Diversity Score (RDS)"
    def: RDS = stddev(Rating) across all content within a Franchise Group, by checking the different rating categories assigned within the same franchise
  - KB 78 [calculation_knowledge] "Episode Rating Band"
    def: We calculate the average user rating across all episodes in a show, and sort it into one of three categories: Low (under 3.5), Medium (3.5 to 4.2), or High (above 4.2).
  - KB 83 [calculation_knowledge] "TieredUserScoreCoverage"
    def: Each show gets a user rating, which is then matched to a category: Low (0–2), Medium (2–4), or High (4–5). We count how many shows fall into each of those rating bands.

### core.title_length  (column)

- Current `meta.kb_ids`: `[36, 90]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 36 [calculation_knowledge] "Title-to-Episode Ratio (TER)"
    def: TER = The ratio of the length of a show’s title to its total number of episodes, plus one.
  - KB 90 [calculation_knowledge] "Average Title Length per Studio"
    def: We look at how many characters are in the titles of a studio’s shows, then average those lengths to see what naming style they tend to use.

### core.is_over_fragmented  (column)

- Current `meta.kb_ids`: `[68, 71]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 68 [domain_knowledge] "Over-Fragmented Offering"
    def: A show is over-fragmented if the number of short-form genre categories and content types exceeds a given threshold
  - KB 71 [domain_knowledge] "Over-Fragmented Offering"
    def: A show is considered over-fragmented if it is associated with more than six nested genre categories and the quantity of its short-form video assets exceeds that of its long-form feature content.

### promo_info.promotional_message_count  (column)

- Current `meta.kb_ids`: `[25, 80]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 25 [calculation_knowledge] "Promotional Message Count"
    def: PMC = The count of promotional messages is found by adding up all the filled-in promotional and availability notes for a piece of content.
  - KB 80 [calculation_knowledge] "Promotional Intensity Summary"
    def: We check each show or movie for various promo message types—like availability, promotions, alerts, or expirations—and count how many of them are actually filled in.

### show_rollups.srlinks  (column)

- Current `meta.kb_ids`: `[16, 41, 76]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 16 [domain_knowledge] "Multi-Tier Presence"
    def: If content is available in two or more distinct subscription tiers, it qualifies as multi-tier.
  - KB 41 [domain_knowledge] "Multitier Syndicated Show"
    def: If a show is made available in three or more distinct access groups, it's considered multi-tier.
  - KB 76 [domain_knowledge] "Multi-Tier Syndication"
    def: A show is considered 'multi-tier' if it appears in at least three different viewing plans, like free, basic, and premium.

### show_rollups.launchmoment  (column)

- Current `meta.kb_ids`: `[9, 48]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 9 [value_illustration] "Cache Time Formats"
    def: Includes examples like '2024-12-08', 'Dec 8, 2024', '08/12/24', and ISO-8601 format.
  - KB 48 [domain_knowledge] "Missing Launch Moment"
    def: If Launch Moment is NULL while Media Total > 0, the show is considered legacy-uploaded.

### show_rollups.ratinginfo  (column)

- Current `meta.kb_ids`: `[2, 12, 19]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 2 [value_illustration] "TV Rating Types"
    def: Includes 'TV-Y', 'TV-Y7', 'TV-G', 'TV-PG', 'TV-14', and 'TV-MA' to represent different maturity levels.
  - KB 12 [domain_knowledge] "Unrated Media"
    def: Media is considered unrated when there is no official rating information available.
  - KB 19 [domain_knowledge] "Canonical Rating Enumeration"
    def: The six recognized TV content ratings are TV-Y, TV-Y7, TV-G, TV-PG, TV-14, TV-MA.

### show_rollups.media_total  (column)

- Current `meta.kb_ids`: `[24, 26, 51, 55]`
- Bucket: **D** — R-SPLIT-MULTI-FORMULA (one entity per formula)
- Target: produce 4 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 24 [calculation_knowledge] "Unrated Share"
    def: Unrated content share is calculated as the amount of unrated material divided by all available content.
  - KB 26 [calculation_knowledge] "Average Media per Tier"
    def: AMT = \frac{\sum Media Total tier}{|Tiers|}
  - KB 51 [calculation_knowledge] "Tier-Normalized Media Load (TNML)"
    def: TNML = \frac{\text{Total Media Volume}}{\text{Number of Availability Tiers}}, where tiers are determined by content access grouping.
  - KB 55 [calculation_knowledge] "Unrated Proportion per Tier (UPT)"
    def: UPT = \frac{\text{Unrated Vol}_{tier}}{\text{Media Total}_{tier}}, using the standard set of content ratings and grouping by access level

### show_rollups.peak_rating_canonical  (column)

- Current `meta.kb_ids`: `[2, 19, 81]`
- Bucket: **C** — R-SPLIT-TRINITY (helper + calc + classification, 3-way split)
- Target: produce 3 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 2 [value_illustration] "TV Rating Types"
    def: Includes 'TV-Y', 'TV-Y7', 'TV-G', 'TV-PG', 'TV-14', and 'TV-MA' to represent different maturity levels.
  - KB 19 [domain_knowledge] "Canonical Rating Enumeration"
    def: The six recognized TV content ratings are TV-Y, TV-Y7, TV-G, TV-PG, TV-14, TV-MA.
  - KB 81 [calculation_knowledge] "Most Common Peak TV Rating"
    def: This identifies which TV rating (like TV-MA or TV-PG) appears the most often across all shows, showing the most typical maturity level in the catalog.

### show_rollups.is_high_visibility_empty_bucket  (column)

- Current `meta.kb_ids`: `[69, 70]`
- Bucket: **?** — Bucket unclear — agent reads KB texts and picks the best split shape
- Target: produce 2 single-KB entities, each carrying its own `meta.kb_id`.

KB excerpts:
  - KB 69 [domain_knowledge] "High-Visibility Empty Bucket"
    def: A show is highly rated but has no extra video segments or trailers
  - KB 70 [domain_knowledge] "Highly Rated but Visually Empty"
    def: Among top-rated shows, find those missing both trailers and extra video segments
