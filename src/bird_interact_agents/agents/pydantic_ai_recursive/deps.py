"""Dependency models for the recursive pydantic-ai adapter.

Three pieces:

* :class:`AgentRecord` — one entry per agent.run() in the spawn tree.
  ``parent_idx`` lets a reader rebuild the tree from a flat
  completion-order list; ``error`` survives failed sub-agents.
* :class:`SharedTaskState` — singletons shared across every agent in
  one task: the budget pool, the SLayer storage/client cache, the
  collector of agent records, the final submitter result.
* :class:`TaskDeps` — per-agent dependencies (depth, max_depth,
  ``self_record_idx``, user_sim_transcript, usage). One ``TaskDeps``
  per agent.run() invocation.

:class:`_LegacyAdapter` is a plain Python bridge that exposes the flat
attribute interface that :mod:`bird_interact_agents.agents._submit`
duck-types against (``status``, ``data_path_base``, ``user_sim_model``,
``result``, ``_slayer_client``, etc.), routing each read/write into the
right slot of ``SharedTaskState`` or ``TaskDeps``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from bird_interact_agents.harness import SampleStatus
from bird_interact_agents.usage import TokenUsage


class AgentRecord(BaseModel):
    """One agent.run() in the spawn tree.

    Populated incrementally: pre-reserved with role/depth/parent_idx/
    instruction/started_at when the spawn fires; output/messages/usage/
    user_sim_transcript/ended_at filled in when the run returns.
    ``error`` is set when the run raised; the failed agent still gets
    a record so readers can locate the failure site in the tree.
    """

    role: Literal[
        "root_clarifier", "sub_clarifier",
        "projection_resolver", "query_constructor",
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


class SharedTaskState(BaseModel):
    """Singletons shared across every agent in one task.

    The whole spawn tree holds ONE Python ref to this — that's how
    budget bookkeeping, the SLayer storage cache, and the
    agent_records collector remain consistent across sub-agents.
    """

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
    _slayer_client: Any = None
    _slayer_storage: Any = None
    _slayer_server: Any = None


class TaskDeps(BaseModel):
    """Per-agent dependencies. One per agent.run() invocation.

    ``self_record_idx`` is the slot where THIS agent's record lives in
    ``shared.agent_records``. When this agent calls ``spawn_subagent``,
    the wrapper reads ``self_record_idx`` and uses it as the spawned
    child's ``parent_idx``. The child's own ``self_record_idx`` is the
    newly-appended slot, so grandchildren correctly point at the child
    instead of at a sibling that happened to complete first.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shared: SharedTaskState
    depth: int = 0
    max_depth: int = 3
    self_record_idx: int | None = None
    user_sim_transcript: list[dict] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)


class _LegacyAdapter:
    """Plain Python bridge from :class:`TaskDeps` to the flat-attribute
    interface that :mod:`bird_interact_agents.agents._submit` (and the
    upstream BIRD-Interact harness) duck-types against.

    NOT a Pydantic model on purpose: ``_submit.*`` mutates ``.result``
    and ``._slayer_client`` freely, and Pydantic's validation would
    either reject the writes or silently coerce them. The shape is
    intentionally minimal — only the attributes the helpers actually
    touch are routed.
    """

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
