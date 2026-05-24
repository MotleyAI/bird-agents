"""Helpers for enforcing a uniform `strict` value on tool definitions.

Cerebras's OpenAI-compatible API rejects requests whose tool array has
inconsistent `strict` values across entries. We force a single value
across all tools — default `False` (matches the existing behaviour of
smolagents/agno/mcp_agent and avoids OpenAI strict-mode constrained
decoding), opt in to `True` via the `--strict` CLI flag.

Each framework adapter wires this into its own tool-serialisation path
and, if it cannot honour `strict=True`, calls `warn_unsupported` to fail
fast at startup rather than silently produce a non-strict request.
"""

from __future__ import annotations


STRICT_DEFAULT: bool = False


def warn_unsupported(framework: str) -> None:
    """Fail fast when --strict can't be honoured by the chosen framework.

    Raises SystemExit with a clear message — caller is the framework
    adapter's __init__ or run_task entry point.
    """
    raise SystemExit(
        f"--strict True is not supported for framework {framework!r}. "
        "Either omit --strict or switch to a framework that exposes a "
        "uniform strict knob (pydantic_ai, smolagents, agno)."
    )
