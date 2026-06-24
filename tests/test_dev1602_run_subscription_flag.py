"""DEV-1602: the local ``run.py`` ``--subscription-auth`` flag.

The cloud submitter threads the subscription choice through the manifest +
driver. For LOCAL runs the agents execute in-process, so the only thing the
flag must do is translate the operator's choice into the
``BIRD_INTERACT_SUBSCRIPTION_AUTH`` signal env var that ``sdk_env`` reads —
and, crucially, CLEAR an ambient signal when the operator did NOT opt in (else
a stray exported var would silently flip an API-key run onto the subscription).

These pin ``run._apply_subscription_auth_env`` directly (it is the unit the
CLI ``main`` calls right after ``parse_args``); the flag is Anthropic-only and
claude_sdk-only, mirroring the cloud CLI policy.
"""

from __future__ import annotations

import pytest

from bird_interact_agents import run


_GOOD_OAUTH = "sk-ant-oat01-good-token"


class _CalledError(RuntimeError):
    """Stand-in for argparse's ``parser.error`` (which exits)."""


def _err(msg: str):
    raise _CalledError(msg)


@pytest.fixture(autouse=True)
def _clear_signal(monkeypatch):
    monkeypatch.delenv("BIRD_INTERACT_SUBSCRIPTION_AUTH", raising=False)


def test_subscription_on_sets_signal(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_OAUTH)
    run._apply_subscription_auth_env(
        subscription_auth=True,
        framework="claude_sdk_otf",
        agent_model="anthropic/claude-opus-4-7",
        error=_err,
    )
    import os
    assert os.environ["BIRD_INTERACT_SUBSCRIPTION_AUTH"] == "1"


def test_subscription_off_clears_ambient_signal(monkeypatch):
    """Default/`--no-subscription-auth` must actively clear an ambient signal so
    an exported BIRD_INTERACT_SUBSCRIPTION_AUTH cannot hijack the run."""
    monkeypatch.setenv("BIRD_INTERACT_SUBSCRIPTION_AUTH", "1")
    run._apply_subscription_auth_env(
        subscription_auth=False,
        framework="claude_sdk_otf",
        agent_model="anthropic/claude-opus-4-7",
        error=_err,
    )
    import os
    assert "BIRD_INTERACT_SUBSCRIPTION_AUTH" not in os.environ


def test_subscription_on_missing_token_errors(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    with pytest.raises(_CalledError):
        run._apply_subscription_auth_env(
            subscription_auth=True,
            framework="claude_sdk_otf",
            agent_model="anthropic/claude-opus-4-7",
            error=_err,
        )
    import os
    assert "BIRD_INTERACT_SUBSCRIPTION_AUTH" not in os.environ


def test_subscription_on_malformed_token_errors(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-api-not-oauth")
    with pytest.raises(_CalledError, match="sk-ant-oat01-"):
        run._apply_subscription_auth_env(
            subscription_auth=True,
            framework="claude_sdk_otf",
            agent_model="anthropic/claude-opus-4-7",
            error=_err,
        )
    import os
    # A failed opt-in must not leave the signal set.
    assert "BIRD_INTERACT_SUBSCRIPTION_AUTH" not in os.environ


def test_subscription_on_registry_model_errors(monkeypatch):
    """--subscription-auth is Anthropic-only (Codex finding #3): a registry
    model with the flag must error on the Anthropic-only rule EVEN with a valid
    OAuth token present — proving the registry check fires before (and
    independent of) token validation."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_OAUTH)
    with pytest.raises(_CalledError, match="Anthropic-only"):
        run._apply_subscription_auth_env(
            subscription_auth=True,
            framework="claude_sdk_otf",
            agent_model="moonshot/kimi-k2.7-code",
            error=_err,
        )
    import os
    assert "BIRD_INTERACT_SUBSCRIPTION_AUTH" not in os.environ


def _find_add_argument(flag: str):
    """Return the ast.Call node for ``parser.add_argument("<flag>", ...)`` in
    ``run.main`` (mirrors the wiring-test pattern in the otf run-wiring tests)."""
    import ast
    import inspect

    src = inspect.getsource(run.main)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == flag
        ):
            return node
    return None


def test_main_registers_subscription_auth_flag_default_off():
    """main() must register --subscription-auth as a BooleanOptionalAction with
    default False (local default-off, preserving existing invocations)."""
    import ast

    node = _find_add_argument("--subscription-auth")
    assert node is not None, "--subscription-auth not registered in run.main"
    kw = {k.arg: k.value for k in node.keywords}
    # BooleanOptionalAction (argparse.BooleanOptionalAction attribute access).
    assert "action" in kw
    assert getattr(kw["action"], "attr", None) == "BooleanOptionalAction"
    # default False.
    assert isinstance(kw.get("default"), ast.Constant) and kw["default"].value is False


def test_main_invokes_subscription_auth_helper():
    """main() must actually CALL the helper (wiring), not merely mention it — a
    comment / string literal / dead alias must not satisfy this."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(run.main))
    assert any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_apply_subscription_auth_env"
        for node in ast.walk(tree)
    ), "run.main does not call _apply_subscription_auth_env"


def test_subscription_on_non_claude_sdk_framework_errors(monkeypatch):
    """The flag only applies to claude_sdk* frameworks (pydantic_ai etc. use
    their own auth) — opting in on another framework is a misuse."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_OAUTH)
    with pytest.raises(_CalledError):
        run._apply_subscription_auth_env(
            subscription_auth=True,
            framework="pydantic_ai",
            agent_model="anthropic/claude-opus-4-7",
            error=_err,
        )
    import os
    assert "BIRD_INTERACT_SUBSCRIPTION_AUTH" not in os.environ
