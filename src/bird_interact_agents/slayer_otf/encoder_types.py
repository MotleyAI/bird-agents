"""Framework-neutral encoder output types (DEV-1589).

`EncodedEntity` / `EncoderResult` / `EncoderCaptureDeps` are the structured
payloads the build-time OTF *reference* encoders produce. They moved here (out
of the now-obsolete ``agents.pydantic_ai_otf_encode.deps``) so the
framework-neutral scheduler (:mod:`slayer_otf.reference_build`) and the new
``claude_sdk`` encoder can import them without reaching into an agent package.

``agents.pydantic_ai_otf_encode.deps`` re-exports these for back-compat, so
every existing import keeps resolving to the SAME class object — load-bearing
for ``EncoderResult.model_validate`` on reference reuse and for ``isinstance``
checks across the codebase.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EncodedEntity(BaseModel):
    """One SLayer entity the encoder wrote for a KB.

    ``host_model`` is None for kind='model' (query-backed models live at the
    datasource root, not on a host).
    """

    kind: Literal["column", "measure", "aggregation", "model"]
    host_model: str | None = None
    name: str
    entity_ref: str


class EncoderResult(BaseModel):
    """The encoder's structured output AND (for the task-time encoder) the
    per-task dedup registry's per-kb value — store the full result, not just
    ``.entities``, so failure reuse + notes survival work correctly.
    """

    kb_id: int
    status: Literal["encoded", "deferred", "error"]
    entities: list[EncodedEntity] = Field(default_factory=list)
    notes: str = ""
    error: str | None = None
    # The setup encoder (no ask_user) defers ambiguous KB items, recording the
    # questions a later per-task agent must ask. Empty for the encoded/error
    # paths. List of str (not a Dict — global LLM-output rule).
    clarifying_questions: list[str] = Field(default_factory=list)


class EncoderMetaSettings(BaseModel):
    """DEV-1605: the encoder settings recorded in ``_encoder_meta.json``.

    Stored generously (provenance, not a contract) — every field is optional
    so the reference build can record whatever it knows.
    """

    reasoning_effort: str | None = None
    model_settings_json: str | None = None
    version_was_explicit: bool = False


class EncoderMeta(BaseModel):
    """DEV-1605: self-describing provenance written next to ``_reference_fp.txt``
    so an encoded OTF reference records WHO built it (model + framework +
    version) and WHEN, independent of the incidental ``_setup_usage.json``
    breakdown name.
    """

    schema_version: int = 1
    version: str
    encoder_model: str
    encoder_framework: str
    benchmark: str
    db: str
    reference_fp: str
    built_at: str
    settings: EncoderMetaSettings = Field(default_factory=EncoderMetaSettings)


class ConsumedReference(BaseModel):
    """DEV-1605: which encoded OTF reference a consumer run actually used for a
    given db. Stamped into the run manifest (a per-db list) and the per-task
    ``SubmissionAnnotation`` so ``runs/`` answers "which version did this run
    consume" without reaching into GCS.
    """

    db: str
    version: str
    encoder_model: str
    reference_fp: str


class EncoderCaptureDeps(BaseModel):
    """Per-run deps for the build-time setup encoder, which has no task context
    (no ``SharedTaskState`` / user-sim). Carries the ``submit_encoding`` result
    so the encoder can reason in text and deliver its ``EncoderResult`` via a
    tool rather than structured output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoder_submission: EncoderResult | None = None
