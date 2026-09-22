"""Pydantic models for the submission JSONL row + per-step prompt_flow entry.

The serialization shape is pinned by Section II of the BIRD-INTERACT-1.0
submission guidelines (a-Interact custom-agent variant):

* Per-instance: ``instance_id``, ``subtask_1_predicted_sql`` (list[str]),
  ``subtask_2_predicted_sql`` (list[str]), ``prompt_flow``.
* Per ``prompt_flow`` entry: ``model``, ``user_simulator``, ``prompt``,
  ``response``, ``action``, ``remaining_budget``, ``action_input_tokens``,
  ``action_output_tokens``, ``action_cost``.
"""

from __future__ import annotations

from pydantic import BaseModel


class PromptFlowEntry(BaseModel):
    model: str
    user_simulator: str
    prompt: str
    response: str
    action: str
    remaining_budget: float
    action_input_tokens: int
    action_output_tokens: int
    action_cost: float

    model_config = {"extra": "forbid"}


class SubmissionRow(BaseModel):
    instance_id: str
    subtask_1_predicted_sql: list[str]
    subtask_2_predicted_sql: list[str]
    prompt_flow: list[PromptFlowEntry]

    model_config = {"extra": "forbid"}
