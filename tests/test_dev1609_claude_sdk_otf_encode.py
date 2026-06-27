"""DEV-1609: make `claude_sdk_otf_encode` a first-class framework so the
claude_sdk reference encoder (DEV-1589) is reachable from the cloud submit
path, and the default OTF encoder.

These tests pin the NON-cloud-keying contract (the cloud artifact-keying
parity lives in `tests/cloud/test_dev1609_encode_keying.py`):

* `frameworks.is_otf_encode_framework` is the single source of truth and
  recognises BOTH encode frameworks.
* `claude_sdk_otf_encode` is in the `--framework` choices of both the local
  `run.py` parser and the cloud `cli.py` submit parser.
* `run._make_runner` builds a `ClaudeSDKOtfEncodeAgent` for the new framework.
* `ClaudeSDKOtfEncodeAgent` is build-only (DEV-1609 D1): it constructs the
  claude_sdk build-encoder and calls `ensure_db_reference` with the
  benchmark-scoped roots, then returns a finalized row — NO per-task masking,
  NO agentic eval loop.
* The agent enforces the same guardrails as the legacy encode
  (`slayer_setup='on-the-fly'`, `--query-mode slayer`, `--mode a-interact`).
* `--otf-rebuild` force-wipes both OTF layers for the new framework.
* `--subscription-auth` gating treats it as a claude_sdk-family framework.
"""

from __future__ import annotations

import argparse
import ast
import inspect
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Single source of truth: is_otf_encode_framework
# ---------------------------------------------------------------------------


def test_is_otf_encode_framework_recognises_both_encoders():
    from bird_interact_agents.frameworks import (
        OTF_ENCODE_FRAMEWORKS,
        is_otf_encode_framework,
    )

    assert is_otf_encode_framework("claude_sdk_otf_encode")
    # legacy string stays recognised for back-compat (D2)
    assert is_otf_encode_framework("pydantic_ai_otf_encode")
    assert OTF_ENCODE_FRAMEWORKS == frozenset(
        {"pydantic_ai_otf_encode", "claude_sdk_otf_encode"}
    )


@pytest.mark.parametrize(
    "fw",
    ["claude_sdk", "claude_sdk_v1", "pydantic_ai", "pydantic_ai_recursive",
     "mcp_agent", "agno", "smolagents", "", "claude_sdk_otf"],
)
def test_is_otf_encode_framework_rejects_non_encoders(fw):
    from bird_interact_agents.frameworks import is_otf_encode_framework

    assert not is_otf_encode_framework(fw)


# ---------------------------------------------------------------------------
# --framework choices include the new framework (local + cloud)
# ---------------------------------------------------------------------------


def _framework_choices_from_func(func) -> set[str]:
    """Read the inline `choices=[...]` of the `--framework` add_argument in
    `func`'s source (AST), without invoking argparse."""
    src = inspect.getsource(func)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--framework"
        ):
            for kw in node.keywords:
                # tolerate both list and tuple literals (robustness, not
                # asserting on incidental argparse impl details).
                if kw.arg == "choices" and isinstance(
                    kw.value, (ast.List, ast.Tuple)
                ):
                    return {
                        elt.value for elt in kw.value.elts
                        if isinstance(elt, ast.Constant)
                    }
    raise AssertionError("could not find --framework choices")


def test_run_py_framework_choices_include_encode():
    from bird_interact_agents import run as run_mod

    choices = _framework_choices_from_func(run_mod.main)
    assert "claude_sdk_otf_encode" in choices
    # legacy choice preserved
    assert "pydantic_ai_otf_encode" in choices


def test_cloud_cli_framework_choices_include_encode():
    from bird_interact_agents.cloud import cli as cli_mod

    # the submit parser's --framework add_argument lives in `parse_args`.
    choices = _framework_choices_from_func(cli_mod.parse_args)
    assert "claude_sdk_otf_encode" in choices
    assert "pydantic_ai_otf_encode" in choices


# ---------------------------------------------------------------------------
# _make_runner dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_runner_builds_claude_sdk_encode_agent(monkeypatch):
    """`framework='claude_sdk_otf_encode'` constructs `ClaudeSDKOtfEncodeAgent`
    (intercept the constructor and short-circuit)."""
    from bird_interact_agents import run as run_mod

    constructed: list[dict] = []

    class _Sentinel(Exception):
        pass

    class _FakeAgent:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            raise _Sentinel("stop here")

    monkeypatch.setattr(
        "bird_interact_agents.agents.claude_sdk_otf_encode."
        "ClaudeSDKOtfEncodeAgent",
        _FakeAgent,
        raising=False,
    )

    with pytest.raises(_Sentinel):
        run_mod._make_runner(
            framework="claude_sdk_otf_encode",
            dataset="mini-interact",
            query_mode="slayer",
            mode="a-interact",
            agent_model="anthropic/claude-sonnet-4-5",
            strict=False,
            prompt_cache=True,
            max_depth=3,
            slayer_storage_root=None,
            slayer_setup="on-the-fly",
            reasoning_effort=None,
        )
    assert len(constructed) == 1
    assert constructed[0].get("slayer_setup") == "on-the-fly"
    assert constructed[0].get("model") == "anthropic/claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# ClaudeSDKOtfEncodeAgent — constructor + run_task guardrails
# ---------------------------------------------------------------------------


def test_agent_rejects_non_on_the_fly_setup():
    from bird_interact_agents.agents.claude_sdk_otf_encode import (
        ClaudeSDKOtfEncodeAgent,
    )

    with pytest.raises(ValueError):
        ClaudeSDKOtfEncodeAgent(
            model="anthropic/claude-sonnet-4-5", slayer_setup="pre-encoded",
        )


@pytest.mark.asyncio
async def test_agent_rejects_non_slayer_query_mode():
    from bird_interact_agents.agents.claude_sdk_otf_encode import (
        ClaudeSDKOtfEncodeAgent,
    )

    agent = ClaudeSDKOtfEncodeAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError):
        await agent.run_task(
            {"selected_database": "alien", "instance_id": "alien_1",
             "dataset": "mini-interact", "amb_user_query": "q"},
            "/tmp", 1.0, "raw", eval_mode="a-interact",
        )


@pytest.mark.asyncio
async def test_agent_rejects_non_a_interact_mode():
    from bird_interact_agents.agents.claude_sdk_otf_encode import (
        ClaudeSDKOtfEncodeAgent,
    )

    agent = ClaudeSDKOtfEncodeAgent(model="anthropic/claude-sonnet-4-5")
    with pytest.raises(ValueError):
        await agent.run_task(
            {"selected_database": "alien", "instance_id": "alien_1",
             "dataset": "mini-interact", "amb_user_query": "q"},
            "/tmp", 1.0, "slayer", eval_mode="c-interact",
        )


# ---------------------------------------------------------------------------
# ClaudeSDKOtfEncodeAgent.run_task — build-only wiring (DEV-1609 D1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_task_builds_reference_via_claude_sdk_encoder(
    monkeypatch, tmp_path
):
    """run_task must:
      * build the encoder with `make_claude_sdk_build_encoder` (raw model +
        native self_model_id),
      * call `ensure_db_reference` with the benchmark-scoped reference/cache
        roots and that build_encoder,
      * NOT do any per-task masking / variant storage,
      * return a finalized row with the standard key set.
    """
    import bird_interact_agents.agents.claude_sdk_otf_encode.agent as agent_mod
    from bird_interact_agents.agents.claude_sdk_otf_encode import (
        ClaudeSDKOtfEncodeAgent,
    )

    calls: dict = {}

    def _fake_make_encoder(*, model, reasoning_effort, self_model_id):
        calls["make_encoder"] = dict(
            model=model, reasoning_effort=reasoning_effort,
            self_model_id=self_model_id,
        )
        return "BUILD_ENCODER_SENTINEL"

    class _FakeEntry:
        reference_dir = tmp_path / "ref" / "alien"
        setup_results = []

    async def _fake_ensure(db, **kwargs):
        calls["ensure"] = dict(db=db, **kwargs)
        return _FakeEntry()

    monkeypatch.setattr(agent_mod, "make_claude_sdk_build_encoder",
                        _fake_make_encoder)
    monkeypatch.setattr(agent_mod, "ensure_db_reference", _fake_ensure)
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed",
                        lambda *a, **k: None)
    monkeypatch.setattr(agent_mod, "materialize_task_db",
                        lambda *a, **k: None)
    # benchmark-scoped roots → deterministic sentinels we can assert on.
    monkeypatch.setattr(
        agent_mod.paths, "slayer_models_otf_root",
        lambda *, benchmark: Path(f"/ref/{benchmark}"),
    )
    monkeypatch.setattr(
        agent_mod.paths, "slayer_otf_cache_root",
        lambda *, benchmark: Path(f"/cache/{benchmark}"),
    )
    # build_task_variant_storage MUST NOT be called (build-only). If the
    # module imported it, blow up on use.
    if hasattr(agent_mod, "build_task_variant_storage"):
        def _boom(*a, **k):
            raise AssertionError("per-task masking must not run for encode")
        monkeypatch.setattr(agent_mod, "build_task_variant_storage", _boom)

    agent = ClaudeSDKOtfEncodeAgent(model="anthropic/claude-opus-4-7")
    row = await agent.run_task(
        {"selected_database": "alien", "instance_id": "alien_1",
         "dataset": "mini-interact", "amb_user_query": "q"},
        str(tmp_path / "data"), 5.0, "slayer", eval_mode="a-interact",
    )

    # encoder built from the RAW model string
    assert calls["make_encoder"]["model"] == "anthropic/claude-opus-4-7"
    assert calls["make_encoder"]["self_model_id"]  # native id resolved

    # ensure_db_reference got the build_encoder + benchmark-scoped roots
    assert calls["ensure"]["db"] == "alien"
    assert calls["ensure"]["build_encoder"] == "BUILD_ENCODER_SENTINEL"
    assert calls["ensure"]["reference_root"] == Path("/ref/mini-interact")
    assert calls["ensure"]["cache_root"] == Path("/cache/mini-interact")

    # finalized row shape (parity with the legacy encode's minimal row)
    assert row["instance_id"] == "alien_1"
    assert row["database"] == "alien"
    assert row["total_reward"] == 0.0
    assert row["submission_status"] == "never_submitted"
    assert row["error"] is None
    # reuse / empty-results path: no setup results, no _setup_usage.json on
    # disk → kb_encoded empty and usage is zero (no LLM ran on reuse).
    assert row["kb_encoded"] == []
    assert row["usage"]["prompt_tokens"] == 0
    assert row["usage"]["completion_tokens"] == 0


def test_agent_module_does_not_import_eval_or_masking_machinery():
    """DEV-1609 D1: the encode agent is BUILD-ONLY. It must NOT pull in the
    per-task HARD-8 masking or the agentic clarifier/resolver/constructor
    eval machinery — a strong tripwire against a future eval-loop creeping in.
    """
    import bird_interact_agents.agents.claude_sdk_otf_encode.agent as agent_mod

    forbidden = [
        "build_task_variant_storage",   # per-task HARD-8 masking
        "extract_deleted_kb_ids",       # per-task deletion set
        "_build_root_clarifier",        # agentic eval loop
        "_build_projection_resolver",
        "_build_query_constructor",
        "_run_projection_resolver",
    ]
    present = [name for name in forbidden if hasattr(agent_mod, name)]
    assert not present, (
        f"encode agent must stay build-only; leaked eval/masking symbols: "
        f"{present}"
    )


@pytest.mark.asyncio
async def test_run_task_success_telemetry_from_reference_entry(
    monkeypatch, tmp_path
):
    """Success-path telemetry comes from the returned `ReferenceEntry`:
    `kb_encoded` from `setup_results` (model_dump'd) and `usage` from the
    best-effort `_setup_usage.json` written in the reference dir."""
    import bird_interact_agents.agents.claude_sdk_otf_encode.agent as agent_mod
    from bird_interact_agents.agents.claude_sdk_otf_encode import (
        ClaudeSDKOtfEncodeAgent,
    )
    from bird_interact_agents.slayer_otf.reference_build import _SETUP_USAGE
    from bird_interact_agents.usage import TokenUsage

    ref_dir = tmp_path / "ref" / "alien"
    ref_dir.mkdir(parents=True)
    # The encoder writes a usage sidecar on a fresh build.
    built_usage = TokenUsage(prompt_tokens=11, completion_tokens=22)
    (ref_dir / _SETUP_USAGE).write_text(built_usage.model_dump_json())

    class _Result:
        def model_dump(self):
            return {"kb_id": 7, "status": "encoded"}

    class _FakeEntry:
        reference_dir = ref_dir
        setup_results = [_Result()]

    async def _fake_ensure(db, **kwargs):
        return _FakeEntry()

    monkeypatch.setattr(agent_mod, "make_claude_sdk_build_encoder",
                        lambda **k: "ENC")
    monkeypatch.setattr(agent_mod, "ensure_db_reference", _fake_ensure)
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed",
                        lambda *a, **k: None)
    monkeypatch.setattr(agent_mod, "materialize_task_db",
                        lambda *a, **k: None)

    agent = ClaudeSDKOtfEncodeAgent(model="anthropic/claude-opus-4-7")
    row = await agent.run_task(
        {"selected_database": "alien", "instance_id": "alien_1",
         "dataset": "mini-interact", "amb_user_query": "q"},
        str(tmp_path / "data"), 5.0, "slayer", eval_mode="a-interact",
    )
    assert row["kb_encoded"] == [{"kb_id": 7, "status": "encoded"}]
    assert row["usage"]["prompt_tokens"] == 11
    assert row["usage"]["completion_tokens"] == 22


@pytest.mark.asyncio
async def test_run_task_returns_error_row_on_build_failure(
    monkeypatch, tmp_path
):
    """A failure inside the build returns a finalized error row (keeps the
    batch alive), not a raised exception."""
    import bird_interact_agents.agents.claude_sdk_otf_encode.agent as agent_mod
    from bird_interact_agents.agents.claude_sdk_otf_encode import (
        ClaudeSDKOtfEncodeAgent,
    )

    monkeypatch.setattr(agent_mod, "make_claude_sdk_build_encoder",
                        lambda **k: "ENC")
    monkeypatch.setattr(agent_mod, "load_db_data_if_needed",
                        lambda *a, **k: None)
    monkeypatch.setattr(agent_mod, "materialize_task_db",
                        lambda *a, **k: None)

    async def _boom(*a, **k):
        raise RuntimeError("encode blew up")

    monkeypatch.setattr(agent_mod, "ensure_db_reference", _boom)

    agent = ClaudeSDKOtfEncodeAgent(model="anthropic/claude-opus-4-7")
    row = await agent.run_task(
        {"selected_database": "alien", "instance_id": "alien_1",
         "dataset": "mini-interact", "amb_user_query": "q"},
        str(tmp_path / "data"), 5.0, "slayer", eval_mode="a-interact",
    )
    assert row["instance_id"] == "alien_1"
    assert row["total_reward"] == 0.0
    assert "encode blew up" in (row["error"] or "")


# ---------------------------------------------------------------------------
# --otf-rebuild force-wipe gate covers the new framework
# ---------------------------------------------------------------------------


@pytest.fixture
def spy_purges(monkeypatch, tmp_path: Path):
    from bird_interact_agents import run as run_mod
    from bird_interact_agents.slayer_otf import reference_build

    monkeypatch.setattr(
        run_mod.paths, "slayer_otf_cache_root",
        lambda *, benchmark=None: tmp_path / "cache",
    )
    monkeypatch.setattr(
        run_mod.paths, "slayer_models_otf_root",
        lambda *, benchmark=None: tmp_path / "ref",
    )
    cache_calls: list = []
    ref_calls: list = []
    monkeypatch.setattr(
        reference_build, "purge_caches",
        lambda root, dbs: (cache_calls.append((Path(root), set(dbs))) or []),
    )
    monkeypatch.setattr(
        reference_build, "purge_references",
        lambda root, dbs: (ref_calls.append((Path(root), set(dbs))) or []),
    )
    return cache_calls, ref_calls, tmp_path


def test_otf_rebuild_wipes_both_layers_for_claude_sdk_encode(spy_purges):
    from bird_interact_agents import run as run_mod

    cache_calls, ref_calls, tmp_path = spy_purges
    run_mod._maybe_force_wipe_otf(
        otf_rebuild=True, framework="claude_sdk_otf_encode",
        dbs={"db_a", "db_b"}, benchmark="mini-interact",
    )
    assert cache_calls == [(tmp_path / "cache", {"db_a", "db_b"})]
    assert ref_calls == [(tmp_path / "ref", {"db_a", "db_b"})]


def test_otf_rebuild_off_is_noop_for_claude_sdk_encode(spy_purges):
    from bird_interact_agents import run as run_mod

    cache_calls, ref_calls, _ = spy_purges
    run_mod._maybe_force_wipe_otf(
        otf_rebuild=False, framework="claude_sdk_otf_encode",
        dbs={"db_a"}, benchmark="mini-interact",
    )
    assert cache_calls == []
    assert ref_calls == []


# ---------------------------------------------------------------------------
# subscription-auth gating treats it as claude_sdk-family
# ---------------------------------------------------------------------------


def test_prereqs_is_claude_sdk_framework_true_for_encode():
    from bird_interact_agents.cloud.prereqs import _is_claude_sdk_framework

    assert _is_claude_sdk_framework("claude_sdk_otf_encode")


def test_local_subscription_auth_requires_explicit_choice_for_encode():
    """`run._apply_subscription_auth_env` is claude_sdk-family gated; for the
    new encode framework on an Anthropic model, a None choice is an error
    (no silent default)."""
    from bird_interact_agents import run as run_mod

    errors: list[str] = []

    run_mod._apply_subscription_auth_env(
        subscription_auth=None,
        framework="claude_sdk_otf_encode",
        agent_model="anthropic/claude-opus-4-7",
        error=lambda msg: errors.append(msg),
    )
    assert errors, "claude_sdk_otf_encode + Anthropic must require an explicit choice"
