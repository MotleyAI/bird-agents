# crypto — KB entries not encoded as model entities

## KB 14 — Market Maker Activity

Reason: defined as "exectune is **predominantly** 'Maker' AND makermotion
is 'High'". The "predominantly" threshold is unquantified — >50%? >70%?
within what window (per-snapshot? per-pair? rolling)? Encoding any
specific predicate would fix one of those choices arbitrarily, so the
column-meaning enums (Maker/Taker on KB 29 and makermotion levels on
KB 28) are encoded as descriptions and the agent must compose the
predicate ad-hoc at query time.

Status: deferred — AMBIGUOUS-PROSE.

## KB 17 — Momentum Divergence

Reason: "price makes new highs/lows while momentum indicators (buyforce,
sellforce) move in the opposite direction". Needs an explicit comparison
window (what's the lookback for "new highs/lows"? what's the slope
direction window for buyforce/sellforce?) which the KB does not provide.

Status: deferred — TIME-ANCHOR.

## KB 19 — Technical Breakout

Reason: "price exceeds highspotday or falls below lowspotday with volume
(volday) at least 50% above the **30-day average**". The schema's
marketstats holds one row per snapshot with a 24h-window figure (volday);
there is no 30-day rolling volume series in the schema, and no anchor
date parameter is provided to compute one.

Status: deferred — SCHEMA-GAP.

## KB 42 — Technical Reversal Signal

Reason: defined as "Technical Signal Strength > 8 AND Momentum Divergence
present". Technical Signal Strength (KB 39) is encoded on
analyticsindicators, but Momentum Divergence (KB 17) is itself deferred
above for lack of a window definition, so this composite predicate
cannot be encoded faithfully without first resolving KB 17.

Status: deferred — TIME-ANCHOR.

## KB 44 — Optimal Trading Window

Reason: defined as "Volatility-Adjusted Spread < 1.0 AND Market Maker
Activity = 'High'". VAS (KB 37) is encoded on marketstats, but Market
Maker Activity (KB 14) is itself deferred above as AMBIGUOUS-PROSE
(unquantified "predominantly Maker"), so this composite predicate
cannot be encoded faithfully without first resolving KB 14.

Status: deferred — AMBIGUOUS-PROSE.

## KB 48 — Perfect Technical Setup

Reason: defined as "Technical Signal Strength > 7 AND techmeter direction
matches mktfeel sentiment direction AND no Momentum Divergence".
Technical Signal Strength (KB 39) and the techmeter / mktfeel enums
(KB 25, 26) are all encoded, but the "no Momentum Divergence" leg
depends on KB 17, which is itself deferred above as TIME-ANCHOR (no
comparison window). The composite cannot be encoded faithfully without
first resolving KB 17.

Status: deferred — TIME-ANCHOR.
