"""Base protocol that all agent implementations must satisfy."""

from typing import Protocol


class SystemAgent(Protocol):
    """Interface for a BIRD-Interact system agent.

    Each agent framework (Claude SDK, PydanticAI, smolagents, etc.)
    provides a concrete implementation of this protocol.
    """

    async def run_task(
        self,
        task_data: dict,
        data_path_base: str,
        budget: float,
        query_mode: str,
    ) -> dict:
        """Run a single BIRD-Interact task.

        Args:
            task_data: Parsed task dict from mini_interact.jsonl.
            data_path_base: Path to directory containing SQLite DBs and metadata.
            budget: Total bird-coin budget for this task.
            query_mode: "slayer" or "raw".

        Returns:
            Dict with keys:
                phase1_passed: bool
                phase2_passed: bool
                total_reward: float
                trajectory: list[dict]  — turn-by-turn log
                error: str | None
        """
        ...
