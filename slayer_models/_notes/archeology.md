# archeology — KB entries not encoded as model entities

## KB 16 — Digital Conservation Priority

The KB defines a scoring system whose predicate references
`TypeSite = 'Rare'` or `'Unique'`, but the `sites.typesite` enum in
this schema is `Burial, Industrial, Military, Settlement, Religious`
(per the column-meaning catalog). The OR clause that depends on those
values is unsatisfiable, and there is no derived rarity attribute on
sites that maps to "Rare" or "Unique". Encoding the partial predicate
(only the GuessDate-older-than-1000-BCE branch combined with
Degradation Risk Zone) would silently drop a key disjunct of the KB
definition.

The narrative "highest priority for digital preservation through
Premium Quality Scans" is also a recommendation rather than a
computable score — the KB names the scoring system but doesn't
specify the score expression.

Status: deferred — SCHEMA-GAP

## KB 41 — Texture-Critical Artifact

The KB requires `TextureStudy` to contain `'Detailed'` or `'Critical'`,
but `scanfeatures.texturestudy` only takes the values
`Partial, Completed, Not Required` per the column-meaning catalog
(and matches the data sample). The first conjunct of the predicate is
unsatisfiable, so no row will ever flag as Texture-Critical regardless
of TDI. TDI itself is encoded on `scanmesh.tdi` (KB #5); the
predicate's other inputs (`structkind`, `matkind`, etc.) are
documented but cannot reproduce the KB's intent because the texture
status enum is misaligned.

Status: deferred — SCHEMA-GAP
