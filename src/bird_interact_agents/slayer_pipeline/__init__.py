"""Deterministic + LLM phases that prepare a SLayer datasource for
KB encoding. Each module here is one phase of
`scripts/regenerate_slayer_model.py`.

- `types`   — parse leading-type-token from prose descriptions; map to
              SLayer `DataType` enum.
- `overlay` — apply `<db>_column_meaning_base.json` to live SLayer
              models (descriptions + deterministic typing + DEV-1381
              date-format annotations).
- `jsonb`   — walk `fields_meaning` and emit one Column per leaf
              (full-path naming, JSON_EXTRACT sql, copied description,
              meta.derived_from for idempotency); plus drift warnings.
- `dates`   — LLM fallback that types TEXT columns still flagged as
              dates after the deterministic pass.
"""
