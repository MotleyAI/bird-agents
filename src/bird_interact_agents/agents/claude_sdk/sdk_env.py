"""Env injected into the Claude Agent SDK CLI subprocess (DEV-1561).

The Agent SDK launches a bundled `claude` Node binary, which by default
makes outbound calls for telemetry, error reporting, and auto-updates as
part of its initialize handshake. In a benchmark / batch run those side
channels add zero value AND can stall the run for minutes — the symptom
DEV-1561 chases:

* every CLI startup ate 5-10 minutes of silence between process start and
  the first agent log line on local runs;
* the SDK's own 60s ``CLAUDE_CODE_STREAM_CLOSE_TIMEOUT`` default would have
  fired sooner, except interactive users tend to bump it to ~10 min, which
  is exactly what the observed wallclock matches.

A live diagnose (``DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1
DISABLE_AUTOUPDATER=1 bird-interact …``) collapses the silence: SDK
``__aenter__`` returns in ~5s, first AssistantMessage in another ~2s.

We unconditionally disable the side channels for every SDK-spawned CLI by
passing the disable knobs through ``ClaudeAgentOptions.env`` — the SDK
transport layers ``options.env`` on top of the inherited environment, so
a user can still re-enable a specific channel by exporting an explicit
override before launching the runner.
"""

from __future__ import annotations


# Env names recognised by the bundled CLI to opt OUT of outbound side
# channels at initialize time. All "set-to-1 ⇒ disabled" — empty / unset
# keeps the CLI default (= enabled).
#
# - ``DISABLE_TELEMETRY``: analytics ping.
# - ``DISABLE_ERROR_REPORTING``: Sentry crash reporter.
# - ``DISABLE_AUTOUPDATER``: registry version check + background download.
# - ``DISABLE_BUG_COMMAND``: `/bug` slash-command harness (loads at init).
# - ``DISABLE_NON_ESSENTIAL_MODEL_CALLS``: non-task model calls (haiku
#   "thinking" prepass for the bug reporter, etc.).
# - ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC``: umbrella switch added
#   in later CLI versions; gates everything above plus a few smaller
#   probes. Safe to set alongside the per-channel switches.
_DISABLE_OUTBOUND_TELEMETRY_ENV: dict[str, str] = {
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_BUG_COMMAND": "1",
    "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}


def disable_cli_telemetry_env() -> dict[str, str]:
    """Return a fresh copy of the env mapping that disables outbound CLI
    side channels. Merge into ``ClaudeAgentOptions.env`` before passing
    options to ``ClaudeSDKClient``.

    The returned dict is fresh on every call so callers can ``.update(...)``
    additional env vars (e.g. registry-provider auth) without mutating
    the module-level constant.
    """
    return dict(_DISABLE_OUTBOUND_TELEMETRY_ENV)
