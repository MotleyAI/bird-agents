# households_2 — failure trace + metadata audit

Run: `20260531t1008-claudes-slayer-890419` (claude_sdk_otf_ainteract, mini-interact, a-interact, Opus + Sonnet sim, reasoning_effort=high). Duration 908s. submission_status=wrong_result.

audit_status: `clean` (audited == original).

## Task

amb_user_query: **"Where is the highest concentration of high-risk families?"**

`critical_ambiguity` (held by user-sim, not visible to agent):
- `"highest concentration"` → `ORDER BY (high_risk_count / total_count) DESC LIMIT 1` (ratio per region, top 1).
- `"high-risk families"` → `s.socsupport='Yes' AND (Auto+Bike+Motor)>2 AND Newest_Year IN ('2010 to 2013', '2012 To 2013', '2014 or newer')`.
- `"Where"` (non-mask) → `SELECT locregion`.

## ask_user

5 mentions; 4 actual `mcp__bird-interact-tools__ask_user` calls at `trajectory[18, 99, 133, 176]`.

- `[19]`: high-risk = `socsupport='Yes' + vehicles>2 + "2010 to 2013" or newer categories` (verbatim — no mention of typo'd `'2012 To 2013'`).
- `[100]`: **"Please use strict exact matches without normalizing case or trimming for the filtering con[ditions]"** — gold-aligned guidance.
- `[134]`: paraphrase of [19].
- `[177]`: declined formatting questions; reiterated answer is a region name.

## Submitted vs gold

Two differences:

(a) **Locregion normalization.** Gold groups by raw `households.locregion`. Agent submitted a `locregion_canonical` derived dim that `LOWER+TRIM`s, collapses runs of spaces (`'  '→' '`), and CASE-maps to canonical capitalization. This collapses dirty `'Riacho  Fundo I'` (two spaces, 2 households, 1 high-risk → 0.5) INTO `'Riacho Fundo I'` (132 households, 1 high-risk → 0.0076). After normalization, top region is `Riacho Fundo II` (1/100=0.01).

(b) **Newest_Year IN-list.** Gold uses `IN ('2010 to 2013', '2012 To 2013', '2014 or newer')`. Agent uses `IN ('2010 to 2013', '2014 or newer')` after `LOWER+TRIM`. Non-load-bearing here — even without `'2012 To 2013'`, raw-locregion grouping still yields `Riacho  Fundo I = 0.5` as the gold answer (verified by direct SQLite query).

## Decision-point lock-in

**Trajectory `[125] → [127]`.** At `[125]` agent ran a query grouped by raw `households.locregion` and got back exactly the gold result: `Riacho  Fundo I | total=2 | hrc=1 | share=0.500` (top of list). ThinkingBlock signature hidden, but at `[127]` agent immediately ran a NEW query grouped by `locregion_clean` and got `riacho fundo i | 132 | 1 | 0.008`. At `[130]` submitted the normalized version. Final submitted query at `[202]` grouped by `locregion_canonical` and returned `Riacho Fundo II`.

Final text at `[205]` explicitly:
> "Top region: **Riacho Fundo I** (1 out of 2 = 50% with strict matching, or 0.76% out of 132)"

Agent saw both candidates and chose to submit the wrong one anyway.

## Metadata sufficiency

**Resolving information was present** in both metadata AND user-sim:

- `slayer_models_otf/households/models/households/transportation_assets.yaml:254-255` Newest_Year description: *"EX. 2012 To 2013, Not applicable, 2014 or newer"* — literally names the missing `'2012 To 2013'` value. `sampled_values` lines 268-318 list `'2010 to 2013'`, `'2010 To 2013'`, `'2014 or newer'`, `'2014 Or Newer'` as distinct sampled rows.
- `slayer_models_otf/households/models/households/households.yaml:76` locregion `sampled_values` contains `Riacho  Fundo I` (two spaces) as a distinct value alongside `Riacho Fundo I` (line 43) — dirty-data signal.
- KBs 23, 24, 9, 10, 14 describe Mobile and Supported Household conceptually but don't pin the `>2` threshold or year-list. The user-sim filled the threshold gap at `[19]` but didn't mention typo'd `'2012 To 2013'`.

User-sim at `[100]` gave unambiguous "strict exact matches, no normalization" instruction.

## Verdict

**`agent_miss`** — the only one in the households batch. User-sim explicitly said not to normalize. Slayer model exposed `Riacho  Fundo I` as distinct sampled value. Agent's idx 125 raw-locregion query returned the gold answer. Agent then overruled both signals.

Failure class: data-hygiene prior overrode an explicit user-sim instruction and the sampled-values caveat.

Remediation: agent-side prompt nudge — "If the user explicitly says 'use strict matches / do not normalize', that overrides your dirty-data priors and the slayer model's normalised columns. Quote the user instruction in your final reasoning."

Provenance: `results/cloud/20260531t1008-claudes-slayer-890419/rows/households_2/attempt-1.json` (trajectory items 19, 100, 125, 127, 130, 202, 205); `slayer_models_otf/households/models/households/transportation_assets.yaml:254-318`; `slayer_models_otf/households/models/households/households.yaml:26-88`; `households_kb.jsonl` (KB 9, 10, 14, 23, 24); audited gold (`audit_status: clean`).
