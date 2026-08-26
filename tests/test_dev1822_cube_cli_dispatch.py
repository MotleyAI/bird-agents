"""DEV-1822: `cube` query-mode CLI acceptance + dispatch + validation matrix.

- local `bird-interact` parser accepts `--query-mode cube`;
- cloud `bird-interact-cloud` REJECTS it (local-only v1);
- the `claude_sdk` aggregator routes (one_shot × cube) to ClaudeSDKOtfCubeAgent;
- `_validate_cube_mode` rejects every unsupported combo.
"""

from __future__ import annotations

import ast
import inspect
import os
from types import SimpleNamespace
from unittest import mock

import pytest

import bird_interact_agents.run as run_mod


@pytest.fixture(autouse=True)
def _legacy_inprocess(monkeypatch):
    monkeypatch.setenv("BIRD_INTERACT_LOCAL_INPROCESS", "1")


def _arg_choices(argname: str) -> set[str]:
    src = inspect.getsource(run_mod.build_arg_parser)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == argname):
            for kw in node.keywords:
                if kw.arg == "choices" and isinstance(kw.value, ast.List):
                    return {e.value for e in kw.value.elts if isinstance(e, ast.Constant)}
    raise AssertionError(f"no choices for {argname}")


# --- CLI acceptance ---------------------------------------------------------

def test_local_parser_accepts_cube():
    assert "cube" in _arg_choices("--query-mode")


def test_cloud_parser_rejects_cube(capsys):
    from bird_interact_agents.cloud import cli as cli_mod
    argv = [
        "submit", "--framework", "claude_sdk", "--query-mode", "cube",
        "--agent-model", "anthropic/claude-haiku-4-5-20251001",
        "--instance-ids", "x_1", "--mode", "one-shot",
        "--dataset", "livesqlbench-base-lite", "--no-require-annotation",
        "--no-subscription-auth",
    ]
    with pytest.raises(SystemExit):
        cli_mod.parse_args(argv)
    err = capsys.readouterr().err
    assert "query-mode" in err and "cube" in err  # error identifies the bad value


# --- dispatch ---------------------------------------------------------------

def _runner_kwargs(**over):
    kw = dict(
        framework="claude_sdk", dataset="livesqlbench-base-lite",
        query_mode="cube", mode="one-shot",
        agent_model="anthropic/claude-haiku-4-5-20251001",
        strict=False, prompt_cache=True, max_depth=3,
        slayer_storage_root=None, slayer_setup=None, reasoning_effort=None,
    )
    kw.update(over)
    return kw


def test_aggregator_routes_cube_to_cube_agent():
    spy = mock.MagicMock()
    spy.return_value.run_task = mock.AsyncMock()
    with mock.patch(
        "bird_interact_agents.agents.claude_sdk_otf_cube.ClaudeSDKOtfCubeAgent", spy
    ):
        run_mod.make_runner(**_runner_kwargs())
    assert spy.called


def test_direct_token_routes_to_cube_agent():
    spy = mock.MagicMock()
    spy.return_value.run_task = mock.AsyncMock()
    with mock.patch(
        "bird_interact_agents.agents.claude_sdk_otf_cube.ClaudeSDKOtfCubeAgent", spy
    ):
        run_mod.make_runner(**_runner_kwargs(framework="claude_sdk_otf_cube"))
    assert spy.called


# --- validation matrix ------------------------------------------------------

def test_validate_cube_mode_accepts_valid():
    run_mod._validate_cube_mode(
        framework="claude_sdk", query_mode="cube", mode="one-shot",
        dataset="livesqlbench-base-lite",
    )


def test_validate_cube_mode_noop_for_non_cube():
    # raw/slayer are unaffected by the cube gate
    run_mod._validate_cube_mode(
        framework="pydantic_ai", query_mode="raw", mode="a-interact",
        dataset="mini-interact",
    )


@pytest.mark.parametrize("framework,mode,dataset", [
    ("claude_sdk_v1", "one-shot", "livesqlbench-base-lite"),       # v1 family
    ("pydantic_ai", "one-shot", "livesqlbench-base-lite"),          # non-claude_sdk
    ("claude_sdk", "a-interact", "livesqlbench-base-lite"),         # non one-shot mode
    ("claude_sdk", "one-shot", "livesqlbench-base-lite-sqlite"),    # sqlite backend
    ("claude_sdk", "one-shot", "bird-interact-lite-exp"),           # non one-shot benchmark
])
def test_validate_cube_mode_rejects(framework, mode, dataset):
    with pytest.raises(ValueError):
        run_mod._validate_cube_mode(
            framework=framework, query_mode="cube", mode=mode, dataset=dataset,
        )


# --- local cube bootstrap (_maybe_bootstrap_local_cube) --------------------

def _args(**over):
    kw = dict(dataset="livesqlbench-base-lite", query_mode="cube", pg_port=None,
              instance_ids=["solar_panel_6"])
    kw.update(over)
    return SimpleNamespace(**kw)


def _patch_cube_bootstrap(monkeypatch):
    """Stub the cube_local side + db resolver; record calls."""
    import bird_interact_agents.cube_local.model_gen as mg_mod
    import bird_interact_agents.cube_local.deploy as deploy_mod
    from bird_interact_agents import local_postgres

    calls = {}
    monkeypatch.setattr(mg_mod, "ensure_models",
                        lambda bench, dbs, pg_env: calls.__setitem__("models", (bench, tuple(dbs))))
    info = SimpleNamespace(base_url="http://127.0.0.1:4008/cubejs-api/v1", api_secret="S")
    monkeypatch.setattr(deploy_mod, "ensure_cube_running",
                        lambda bench, pg_env: calls.__setitem__("run", bench) or info)
    monkeypatch.setattr(deploy_mod, "poll_models_ready",
                        lambda info, dbs, **kw: calls.__setitem__("poll", tuple(dbs)))
    monkeypatch.setattr(local_postgres, "resolve_dbs_for",
                        lambda benchmark, instance_ids=None: ["solar_panel"])
    return calls


def test_bootstrap_cube_exports_env(monkeypatch):
    from types import SimpleNamespace as NS  # noqa: F401
    for v in ("BIRD_CUBE_URL", "BIRD_CUBE_API_SECRET"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("BIRD_PG_HOST", "127.0.0.1")
    monkeypatch.setenv("BIRD_PG_PORT", "5544")
    calls = _patch_cube_bootstrap(monkeypatch)

    run_mod._maybe_bootstrap_local_cube(_args(), ["solar_panel_6"])

    assert calls.get("run") == "livesqlbench-base-lite"
    assert os.environ["BIRD_CUBE_URL"] == "http://127.0.0.1:4008/cubejs-api/v1"
    assert os.environ["BIRD_CUBE_API_SECRET"] == "S"


def test_bootstrap_noop_for_non_cube(monkeypatch):
    monkeypatch.delenv("BIRD_CUBE_URL", raising=False)
    calls = _patch_cube_bootstrap(monkeypatch)
    run_mod._maybe_bootstrap_local_cube(_args(query_mode="raw"), ["solar_panel_6"])
    assert calls == {}
    assert "BIRD_CUBE_URL" not in os.environ


def test_bootstrap_noop_when_url_preset(monkeypatch):
    monkeypatch.setenv("BIRD_CUBE_URL", "http://preset/cubejs-api/v1")
    calls = _patch_cube_bootstrap(monkeypatch)
    run_mod._maybe_bootstrap_local_cube(_args(), ["solar_panel_6"])
    assert calls == {}  # bring-your-own cube → no provisioning


def test_bootstrap_runs_after_postgres_bootstrap():
    """Ordering: cube provisioning needs BIRD_PG_* already exported."""
    src = inspect.getsource(run_mod)
    assert src.index("_maybe_bootstrap_local_postgres(") < \
        src.rindex("_maybe_bootstrap_local_cube(")
