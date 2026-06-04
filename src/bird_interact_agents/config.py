"""Centralized configuration via environment variables and .env file.

Filesystem paths (mini-interact data dir, audited_gold, slayer_models,
results) are resolved by `bird_interact_agents.paths`. `db_path` and
`data_path` are exposed here as string-typed read-through properties so
existing callers keep working.
"""

from pydantic_settings import BaseSettings

from . import paths


class Settings(BaseSettings):
    # User simulator LLM model (LiteLLM format)
    user_sim_model: str = "anthropic/claude-haiku-4-5-20251001"

    # User simulator prompt version: "v1" or "v2"
    user_sim_prompt_version: str = "v2"

    # Model used by deterministic-pipeline LLM steps (e.g. text-as-date
    # detection in scripts/regenerate_slayer_model.py). Bare Anthropic
    # model id, not LiteLLM-prefixed. Env var: BIRD_AGENTS_LLM_MODEL.
    agents_llm_model: str = "claude-haiku-4-5"

    model_config = {"env_prefix": "BIRD_", "env_file": ".env", "extra": "ignore"}

    @property
    def db_path(self) -> str:
        """mini-interact/ data dir as a string — delegates to paths.py so
        worktree resolution kicks in automatically."""
        return str(paths.benchmark_data_root("mini-interact"))

    @property
    def data_path(self) -> str:
        """mini_interact.jsonl as a string — delegates to paths.py."""
        return str(paths.benchmark_data_file("mini-interact"))


settings = Settings()


def get_default_llm_model() -> str:
    return settings.agents_llm_model
