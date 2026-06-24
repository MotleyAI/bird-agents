"""DEV-1589: the encoder output types move to a NEUTRAL module.

`EncodedEntity` / `EncoderResult` / `EncoderCaptureDeps` move out of the
(obsolete) `agents.pydantic_ai_otf_encode.deps` into
`slayer_otf.encoder_types`, so the framework-neutral scheduler
(`slayer_otf.reference_build`) and the NEW claude_sdk encoder no longer import
from the obsolete agent module. The pydantic `deps` module RE-EXPORTS them for
back-compat, so every existing import keeps working AND resolves to the SAME
class object (identity), which is load-bearing for `EncoderResult.model_validate`
on reference reuse.
"""

from __future__ import annotations

import importlib


def test_neutral_module_defines_the_three_types():
    mod = importlib.import_module("bird_interact_agents.slayer_otf.encoder_types")
    assert hasattr(mod, "EncodedEntity")
    assert hasattr(mod, "EncoderResult")
    assert hasattr(mod, "EncoderCaptureDeps")


def test_pydantic_deps_reexports_same_objects():
    from bird_interact_agents.slayer_otf import encoder_types as neutral
    from bird_interact_agents.agents.pydantic_ai_otf_encode import deps

    # Identity, not just structural equality — model_validate round-trips and
    # isinstance checks across the codebase rely on ONE class object.
    assert deps.EncoderResult is neutral.EncoderResult
    assert deps.EncodedEntity is neutral.EncodedEntity
    assert deps.EncoderCaptureDeps is neutral.EncoderCaptureDeps


def test_reference_build_imports_encoder_result_from_neutral():
    """The 2 lazy imports inside reference_build must resolve to the neutral
    module's class (so the live harness no longer reaches into the obsolete
    agent package)."""
    from bird_interact_agents.slayer_otf import encoder_types as neutral
    import bird_interact_agents.slayer_otf.reference_build as rb

    # _encode_all + _load_reference_entry each `from ... import EncoderResult`.
    # Drive the cheap one (the deferred-cycle path in _encode_all) and assert
    # the constructed object is the neutral type.
    import asyncio

    async def _run():
        # A single self-referential KB → it cycles → _encode_all returns a
        # deferred EncoderResult constructed from the imported symbol.
        rows = [{"id": 1, "knowledge": "k", "children_knowledge": 1}]
        edges = {1: [1]}

        async def _never_called(*a, **k):  # cycle members are never scheduled
            raise AssertionError("cycle member must not be encoded")

        out = await rb._encode_all(kb_rows=rows, edges=edges, run_one=_never_called)
        return out

    out = asyncio.run(_run())
    assert len(out) == 1
    assert isinstance(out[0], neutral.EncoderResult)
    assert out[0].status == "deferred"


def test_reference_build_source_no_longer_imports_obsolete_deps():
    """Codex test #9: because pydantic deps RE-EXPORTS the neutral class, an
    identity test alone can't prove reference_build stopped reaching into the
    obsolete module. Assert it at the source level: reference_build imports the
    neutral module and NOT `pydantic_ai_otf_encode.deps import EncoderResult`."""
    import inspect

    import bird_interact_agents.slayer_otf.reference_build as rb

    src = inspect.getsource(rb)
    assert "slayer_otf.encoder_types import" in src or \
        "from bird_interact_agents.slayer_otf.encoder_types" in src
    assert "pydantic_ai_otf_encode.deps import" not in src, (
        "reference_build must not import EncoderResult from the obsolete "
        "pydantic_ai_otf_encode.deps module"
    )


def test_encoder_result_shape_unchanged():
    from bird_interact_agents.slayer_otf.encoder_types import (
        EncodedEntity,
        EncoderResult,
    )

    r = EncoderResult(
        kb_id=3,
        status="encoded",
        entities=[EncodedEntity(
            kind="column", host_model="m", name="c", entity_ref="db.m.c",
        )],
        notes="n",
        clarifying_questions=["q?"],
    )
    assert r.kb_id == 3 and r.status == "encoded"
    assert r.entities[0].entity_ref == "db.m.c"
    # round-trips (the reuse path does EncoderResult.model_validate)
    assert EncoderResult.model_validate(r.model_dump()) == r
