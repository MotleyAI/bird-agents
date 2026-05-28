"""DEV-1462 — CodeRabbit (third /process-reviews round): otf_encode's
post-encode verification (`_entity_exists`) and setup-ref resolution
(`_resolve_ref`) looked up models with an UNSCOPED `storage.get_model(name)`
then asserted `model.data_source == db`.

SLayer resolves a bare name via datasource PRIORITY
(`_resolve_target_or_none(data_source=None)` → `resolve_model_identity`), so
when two datasources in one storage carry the same model name, the bare lookup
can return the WRONG datasource's model — the `== db` guard then fails and a
perfectly-good entity is reported missing (spurious `status="error"` / lost
setup-ref reuse). The precondition — same model name under two datasources — is
exactly this PR's LiveSQLBench-vs-mini-interact shared-name situation. The
sibling lookups at factories.py L774/L908 already pass `data_source=db`; these
four were inconsistent.

These tests use a storage stub that faithfully reproduces SLayer's
priority-based bare-name resolution (bare lookup returns the *other* db's
model) so the scoping fix is provable without depending on SLayer's default
priority ordering.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bird_interact_agents.agents.pydantic_ai_otf_encode.deps import EncodedEntity
from bird_interact_agents.agents.pydantic_ai_otf_encode.factories import (
    _entity_exists,
    _resolve_ref,
)


class _PriorityCollisionStorage:
    """Two datasources (`alien`, `polar`) both own a model named `signals`.

    Mirrors SLayer: ``get_model(name)`` with no ``data_source`` resolves by
    priority — here the higher-priority `polar` always wins, reproducing the
    cross-datasource collision. ``get_model(name, data_source=X)`` returns X's
    own model.
    """

    def __init__(self):
        self._alien = SimpleNamespace(
            data_source="alien",
            columns=[SimpleNamespace(name="flux")],
            measures=[], aggregations=[],
        )
        self._polar = SimpleNamespace(
            data_source="polar",
            columns=[SimpleNamespace(name="temp")],
            measures=[], aggregations=[],
        )

    async def get_model(self, name, data_source=None):
        if name != "signals":
            return None
        if data_source is None:
            # Bare-name resolution: priority picks `polar` (the bug trigger).
            return self._polar
        if data_source == "alien":
            return self._alien
        if data_source == "polar":
            return self._polar
        return None


@pytest.mark.asyncio
async def test_entity_exists_model_scopes_by_datasource():
    """`_entity_exists` for an `alien.signals` model must resolve `alien`'s
    own model — not the higher-priority `polar.signals` (which would fail the
    `data_source == db` guard and be reported missing)."""
    storage = _PriorityCollisionStorage()
    ent = EncodedEntity(
        kind="model", host_model=None, name="signals", entity_ref="alien.signals",
    )
    assert await _entity_exists(ent, storage, "alien") is True


@pytest.mark.asyncio
async def test_entity_exists_column_scopes_by_datasource():
    """A column on the host model must also resolve against `alien`'s model
    (whose columns contain `flux`), not `polar`'s (whose columns don't)."""
    storage = _PriorityCollisionStorage()
    ent = EncodedEntity(
        kind="column", host_model="signals", name="flux",
        entity_ref="alien.signals.flux",
    )
    assert await _entity_exists(ent, storage, "alien") is True


@pytest.mark.asyncio
async def test_resolve_ref_model_scopes_by_datasource():
    """`_resolve_ref('alien.signals')` must resolve to `alien`'s model, not
    return None because the bare lookup found `polar`'s."""
    storage = _PriorityCollisionStorage()
    ent = await _resolve_ref(storage, "alien", "alien.signals")
    assert ent is not None
    assert ent.kind == "model"
    assert ent.name == "signals"
    assert ent.entity_ref == "alien.signals"


@pytest.mark.asyncio
async def test_resolve_ref_column_scopes_by_datasource():
    """`_resolve_ref('alien.signals.flux')` must resolve the column on
    `alien`'s model (it has `flux`), not be lost to `polar`'s model."""
    storage = _PriorityCollisionStorage()
    ent = await _resolve_ref(storage, "alien", "alien.signals.flux")
    assert ent is not None
    assert ent.kind == "column"
    assert ent.host_model == "signals"
    assert ent.name == "flux"
