"""Dependency models for the on-the-fly KB-encode adapter.

Mirrors `pydantic_ai_recursive/deps.py` with these additions for
DEV-1454:

* `EncodedEntity` + `EncoderResult` — the encoder agent's
  `output_type=` structured payload; also the cache value in the
  per-task dedup registry.
* `AgentRecord.role` Literal gains `"kb_encoder"`; new optional
  `kb_id` field for trajectory grouping.
* `SharedTaskState.kb_encoded: list[EncoderResult]` — the dedup
  registry.
* `SharedTaskState._kb_locks: dict[int, asyncio.Lock]` — per-kb
  defensive concurrency shim (Codex finding 2; spawns are already
  `sequential=True` so contention is theoretical, but the shim
  covers any future change).
* `SharedTaskState._kb_rows_by_id: dict[int, dict] | None` — the
  lazy-loaded KB row map populated by `_ensure_kb_rows_loaded` on
  the first `kb_to_slayer` call. `None` means "not loaded yet".
* `SharedTaskState._kb_load_failures: dict[int, str]` — per-kb
  reasons for memories that failed to parse so `kb_to_slayer` can
  surface them as per-id errors (Codex test-review finding 6).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from bird_interact_agents.harness import SampleStatus
from bird_interact_agents.usage import TokenUsage


# ---------------------------------------------------------------------------
# Encoder output types (also serve as the registry payload)
# ---------------------------------------------------------------------------


class EncodedEntity(BaseModel):
    """One SLayer entity the encoder wrote for a KB.

    `host_model` is None for kind='model' (query-backed models live at
    the datasource root, not on a host).
    """

    kind: Literal["column", "measure", "aggregation", "model"]
    host_model: str | None = None
    name: str
    entity_ref: str


class EncoderResult(BaseModel):
    """The encoder sub-agent's structured output AND the per-task
    dedup registry's per-kb value (Codex finding 8 — store the full
    result, not just `.entities`, so failure reuse + notes survival
    work correctly).
    """

    kb_id: int
    status: Literal["encoded", "deferred", "error"]
    entities: list[EncodedEntity] = Field(default_factory=list)
    notes: str = ""
    error: str | None = None
    # DEV-1454: the setup encoder (no ask_user) defers ambiguous KB items,
    # recording the questions a later per-task agent must ask. Empty for the
    # encoded/error paths. List of str (not a Dict — global LLM-output rule).
    clarifying_questions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AgentRecord — extends the recursive adapter's roles with "kb_encoder"
# and adds an optional kb_id field for trajectory grouping.
# ---------------------------------------------------------------------------


class AgentRecord(BaseModel):
    """One agent.run() in the spawn tree. See
    `pydantic_ai_recursive.deps.AgentRecord` for the baseline."""

    role: Literal[
        "root_clarifier", "sub_clarifier",
        "projection_resolver", "query_constructor",
        "kb_encoder",
    ]
    depth: int
    parent_idx: int | None = None
    focus: str | None = None
    instruction: str
    output: str = ""
    user_sim_transcript: list[dict] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    tool_call_stats: dict | None = None
    n_agent_turns: int | None = None
    error: str | None = None
    started_at: float = 0.0
    ended_at: float = 0.0
    kb_id: int | None = None  # populated for role="kb_encoder" runs


# ---------------------------------------------------------------------------
# SharedTaskState — adds kb_encoded registry + lazy KB row cache +
# per-kb lock map + per-kb load-failure map.
# ---------------------------------------------------------------------------


class SharedTaskState(BaseModel):
    """Singletons shared across every agent in one task. See the
    recursive adapter's deps module for the baseline shape; the
    additions below are DEV-1454-specific."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: SampleStatus
    data_path_base: str
    db_name: str
    amb_user_query: str
    slayer_storage_dir: str = ""
    user_sim_model: str
    user_sim_prompt_version: str
    agent_records: list[AgentRecord] = Field(default_factory=list)
    submitter_result: dict | None = None
    kb_encoded: list[EncoderResult] = Field(default_factory=list)
    _slayer_client: Any = None
    _slayer_storage: Any = None
    _slayer_server: Any = None
    _kb_locks: dict[int, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _kb_rows_by_id: dict[int, dict] | None = PrivateAttr(default=None)
    _kb_load_failures: dict[int, str] = PrivateAttr(default_factory=dict)


# ---------------------------------------------------------------------------
# TaskDeps — identical shape to the recursive adapter.
# ---------------------------------------------------------------------------


class TaskDeps(BaseModel):
    """Per-agent dependencies. One per agent.run() invocation. See
    `pydantic_ai_recursive.deps.TaskDeps` for the baseline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shared: SharedTaskState
    depth: int = 0
    max_depth: int = 3
    self_record_idx: int | None = None
    user_sim_transcript: list[dict] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    # DEV-1454: per-run capture slots for the submit_* tools. The encoders and
    # the projection resolver deliver their result via a tool (not pydantic-ai
    # structured output) so the model can reason in text between tool calls —
    # structured output forces tool_choice='any', which forbids reasoning. One
    # TaskDeps per agent.run() invocation → these are per-run.
    encoder_submission: EncoderResult | None = None
    projection_submission: list[str] | None = None


class EncoderCaptureDeps(BaseModel):
    """Per-run deps for the build-time setup encoder, which has no task context
    (no ``SharedTaskState`` / user-sim). Carries the ``submit_encoding`` result
    so the encoder can reason in text and deliver its ``EncoderResult`` via a
    tool rather than pydantic-ai structured output (DEV-1454)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoder_submission: EncoderResult | None = None


# ---------------------------------------------------------------------------
# _LegacyAdapter — plain Python bridge for _submit.* duck-typing.
# ---------------------------------------------------------------------------


class _LegacyAdapter:
    """Plain Python bridge from TaskDeps to the flat-attribute
    interface that ``bird_interact_agents.agents._submit`` duck-types
    against (`status`, `data_path_base`, `user_sim_model`, `result`,
    `_slayer_client`, etc.). See the recursive adapter's deps module
    for the rationale on why it's not a Pydantic model."""

    def __init__(self, deps: TaskDeps) -> None:
        self._deps = deps

    @property
    def status(self) -> SampleStatus:
        return self._deps.shared.status

    @property
    def data_path_base(self) -> str:
        return self._deps.shared.data_path_base

    @property
    def user_sim_model(self) -> str:
        return self._deps.shared.user_sim_model

    @property
    def user_sim_prompt_version(self) -> str:
        return self._deps.shared.user_sim_prompt_version

    @property
    def slayer_storage_dir(self) -> str:
        return self._deps.shared.slayer_storage_dir

    @property
    def user_sim_transcript(self) -> list[dict]:
        return self._deps.user_sim_transcript

    @property
    def usage(self) -> TokenUsage:
        return self._deps.usage

    @property
    def result(self) -> dict | None:
        return self._deps.shared.submitter_result

    @result.setter
    def result(self, value: dict | None) -> None:
        self._deps.shared.submitter_result = value

    @property
    def _slayer_client(self) -> Any:
        return self._deps.shared._slayer_client

    @_slayer_client.setter
    def _slayer_client(self, value: Any) -> None:
        self._deps.shared._slayer_client = value

    @property
    def _slayer_storage(self) -> Any:
        return self._deps.shared._slayer_storage

    @_slayer_storage.setter
    def _slayer_storage(self, value: Any) -> None:
        self._deps.shared._slayer_storage = value
