"""Shared test fixtures and configuration.

Pre-seeds `BIRD_DB_PATH` from the central `paths` helper so tests that use
``portable_connection`` work from any checkout without per-developer env setup.
"""

import os

# Force litellm to use its bundled model-cost map instead of fetching the
# latest over the network at import time — faster import and fully offline
# (no real network during tests). Must be set before litellm is imported.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import pytest

from bird_interact_agents import paths

# Set BIRD_DB_PATH for portable_connection.py SQLite re-anchoring in tests.
os.environ.setdefault("BIRD_DB_PATH", str(paths.benchmark_data_root("mini-interact")))


# DEV-1508 perf: session-scoped cache for ``YAMLStorage.get_model`` /
# ``get_datasource``. Profiling the suite (118s, 1613 tests) found:
#
#   * 1564 calls to ``yaml.safe_load`` accumulating to ~23s of CPU,
#   * 286 of those originating in ``_prompt_builders.build_slayer_c_interact_
#     prompt`` (5.7s cumulative), which loads the same committed
#     ``slayer_models/<db>/`` tree from disk five times per test (once per
#     framework adapter — claude_sdk / pydantic_ai / agno / mcp_agent /
#     smolagents).
#
# The cache key is ``(abs_path, st_mtime_ns)`` so any test that writes a
# YAML file (e.g. ``save_model`` / ``edit_model``) automatically invalidates
# its entry on the next read (mtime advances on every write). Read-only
# committed dirs hit the cache 4× in 5 across the prompt-parity test alone.
#
# Cached values are returned BY REFERENCE — a test that mutates the
# returned Pydantic model would leak that mutation to subsequent cache
# hits. None of the current tests mutate the returned model in place
# (they pass it to a renderer or read attributes), so this is safe today.
# If a future test needs to mutate, it can ``.model_copy()`` defensively.
_YAML_STORAGE_GET_MODEL_CACHE: dict = {}
_YAML_STORAGE_GET_DATASOURCE_CACHE: dict = {}


@pytest.fixture(scope="session", autouse=True)
def _cache_yaml_storage_reads():
    """Memoize ``YAMLStorage.get_model`` / ``get_datasource`` by file
    mtime for the whole session. See block-comment above for rationale."""
    import os as _os

    from slayer.storage import yaml_storage as _yaml_storage

    real_get_model = _yaml_storage.YAMLStorage.get_model
    real_get_datasource = _yaml_storage.YAMLStorage.get_datasource

    async def _cached_get_model(self, name, data_source=None):
        # Compute the resolved path the same way the real method does. If
        # any step misses (no datasource, file gone), fall through to the
        # real method so error semantics are preserved.
        target = await self._resolve_target_or_none(
            name, data_source=data_source,
        )
        if target is None:
            return None
        ds, mname = target
        path = self._model_path(ds, mname)
        try:
            mtime = _os.stat(path).st_mtime_ns
        except OSError:
            return await real_get_model(self, name, data_source=data_source)
        key = (path, mtime)
        if key in _YAML_STORAGE_GET_MODEL_CACHE:
            return _YAML_STORAGE_GET_MODEL_CACHE[key]
        result = await real_get_model(self, name, data_source=data_source)
        if result is not None:
            _YAML_STORAGE_GET_MODEL_CACHE[key] = result
        return result

    async def _cached_get_datasource(self, name):
        # Datasource path is stable: ``<base_dir>/datasources/<name>.yaml``.
        ds_path = _os.path.join(self.base_dir, "datasources", f"{name}.yaml")
        try:
            mtime = _os.stat(ds_path).st_mtime_ns
        except OSError:
            return await real_get_datasource(self, name)
        key = (ds_path, mtime)
        if key in _YAML_STORAGE_GET_DATASOURCE_CACHE:
            return _YAML_STORAGE_GET_DATASOURCE_CACHE[key]
        result = await real_get_datasource(self, name)
        if result is not None:
            _YAML_STORAGE_GET_DATASOURCE_CACHE[key] = result
        return result

    _yaml_storage.YAMLStorage.get_model = _cached_get_model
    _yaml_storage.YAMLStorage.get_datasource = _cached_get_datasource
    yield
    _yaml_storage.YAMLStorage.get_model = real_get_model
    _yaml_storage.YAMLStorage.get_datasource = real_get_datasource
    _YAML_STORAGE_GET_MODEL_CACHE.clear()
    _YAML_STORAGE_GET_DATASOURCE_CACHE.clear()


# DEV-1508 perf: session-scoped consolidation of the three import-isolation
# subprocess tests:
#
#   * ``test_pydantic_ai_agent_imports_without_claude_sdk`` (test_harness_imports)
#   * ``test_import_does_not_pull_pydantic_ai_adapter_packages`` (claude_sdk_otf)
#   * ``test_import_does_not_pull_pydantic_ai_adapter_packages``
#     (claude_sdk_otf_ainteract)
#
# Each previously spawned its own fresh Python interpreter (~5-7s wall) just
# to inspect ``sys.modules`` after a clean import. Profile showed ~18s of
# combined wall time. They all need a fresh sys.modules state — but they
# can share a single subprocess that runs all three checks in sequence with
# ``sys.modules`` cleared between each. That collapses ~18s → ~6s.
#
# Each test below still asserts independently (so failures still identify
# WHICH boundary leaked); they just consume from this shared fixture.


@pytest.fixture(scope="session")
def import_isolation_results() -> dict:
    """Run all three adapter import-boundary checks in ONE subprocess and
    return their results keyed by check name.

    Returns a dict with three entries; each value is a sub-dict carrying
    ``ok: bool`` and, on failure, ``error`` (string) and/or ``leaked``
    (list of module names that should not have been imported).
    """
    import json as _json
    import subprocess as _subprocess
    import sys as _sys
    import textwrap as _textwrap

    program = _textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import json
        import sys

        results = {}

        def _clear_state() -> None:
            for k in list(sys.modules):
                if (
                    k == "bird_interact_agents"
                    or k.startswith("bird_interact_agents.")
                    or k == "pydantic_ai"
                    or k.startswith("pydantic_ai.")
                    or k == "claude_agent_sdk"
                    or k.startswith("claude_agent_sdk.")
                ):
                    del sys.modules[k]

        # --- check 1: pydantic_ai adapter imports without claude_agent_sdk ---
        class _BlockClaudeSDK(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
                    raise ImportError(
                        f"claude_agent_sdk is masked for this isolation test "
                        f"(attempted to import {name})"
                    )
                return None

        _blocker = _BlockClaudeSDK()
        sys.meta_path.insert(0, _blocker)
        try:
            importlib.import_module("bird_interact_agents.agents.pydantic_ai.agent")
            results["pydantic_ai_without_claude_sdk"] = {"ok": True}
        except Exception as e:
            results["pydantic_ai_without_claude_sdk"] = {
                "ok": False, "error": repr(e),
            }
        sys.meta_path.remove(_blocker)
        _clear_state()

        # --- check 2: claude_sdk_otf must not pull pydantic_ai ADAPTER ---
        try:
            importlib.import_module(
                "bird_interact_agents.agents.claude_sdk_otf.agent"
            )
            leaked = sorted(
                n for n in sys.modules
                if n.startswith("bird_interact_agents.agents.pydantic_ai")
            )
            results["claude_sdk_otf_no_pydantic_ai_adapter"] = {
                "ok": not leaked, "leaked": leaked,
            }
        except Exception as e:
            results["claude_sdk_otf_no_pydantic_ai_adapter"] = {
                "ok": False, "error": repr(e), "leaked": [],
            }
        _clear_state()

        # --- check 3: claude_sdk_otf_ainteract must not pull pydantic_ai ADAPTER ---
        try:
            importlib.import_module(
                "bird_interact_agents.agents.claude_sdk_otf_ainteract.agent"
            )
            leaked = sorted(
                n for n in sys.modules
                if n.startswith("bird_interact_agents.agents.pydantic_ai")
            )
            results["claude_sdk_otf_ainteract_no_pydantic_ai_adapter"] = {
                "ok": not leaked, "leaked": leaked,
            }
        except Exception as e:
            results["claude_sdk_otf_ainteract_no_pydantic_ai_adapter"] = {
                "ok": False, "error": repr(e), "leaked": [],
            }

        sys.stdout.write("RESULT:" + json.dumps(results))
        """
    )
    proc = _subprocess.run(
        [_sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "import_isolation_results subprocess crashed: "
            f"rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
    marker = "RESULT:"
    idx = proc.stdout.rfind(marker)
    if idx < 0:
        raise RuntimeError(
            f"import_isolation_results: marker missing from stdout={proc.stdout!r}"
        )
    return _json.loads(proc.stdout[idx + len(marker):])


@pytest.fixture(autouse=True)
def _disable_real_embeddings(monkeypatch):
    """Globally force SLayer's embedding channel OFF for the whole suite.

    SLayer auto-embeds on every write (``save_memory`` / ``edit_model`` /
    ``ingest``) and runs dense search whenever ``embeddings.client.is_available()``
    is True — which it is on any machine with an ``OPENAI_API_KEY`` in the env.
    Left unmocked, tests that touch a real SLayer storage would make real
    (paid, non-deterministic) OpenAI embedding calls AND write ``embeddings.db``
    sidecars into committed dirs. Forcing ``is_available`` False short-circuits
    every refresh hook + the dense search channel (BM25 + tantivy still work),
    keeping the suite offline and deterministic (the no-real-APIs rule).

    Tests that specifically exercise the embedding hook re-enable it locally
    (monkeypatch ``is_available`` True) AND stub ``SearchService.upsert_memory``
    (SLayer 0.7.3+; the legacy ``EmbeddingService.refresh_memory`` path was
    removed) so they still never hit the wire.
    """
    try:
        from slayer.embeddings import client as _emb_client
    except Exception:  # noqa: BLE001 — embeddings extra not installed → nothing to do
        return
    monkeypatch.setattr(_emb_client, "is_available", lambda: False)


@pytest.fixture(autouse=True)
def _forbid_real_llm_completions(request, monkeypatch):
    """Fail fast if a NON-integration test reaches a real LLM completion.

    A unit/non-integration test must never make a paid, network-dependent
    completion call — those belong behind ``@pytest.mark.integration`` (the
    no-real-APIs rule, same spirit as ``_disable_real_embeddings`` above).

    Patches the completion seams to raise: the user-sim / tracked entry
    (``usage._acompletion`` — an indirection seam bound at import to
    ``litellm.acompletion``) and the raw ``litellm.acompletion`` /
    ``litellm.completion`` used by other adapters. Tests that legitimately
    exercise these wrappers stub the same seam themselves — their ``setattr``
    runs AFTER this autouse fixture and wins; anything that reaches a live call
    trips the guard with an actionable message.

    Motivating regression (DEV-1649 postmortem): a trajectory-shape test ran in
    a-interact mode with ``model="test"``, whose TestModel auto-invoked the
    ``ask_user`` tool — firing a REAL user-sim (haiku) call on every run. It
    looked green for months and only failed once Anthropic credits ran out.

    Does NOT cover the ``claude_sdk`` subprocess path (it spawns the bundled
    ``claude`` Node CLI, not litellm); those tests stub the SDK client or are
    integration-marked.
    """
    if request.node.get_closest_marker("integration"):
        return
    msg = (
        "Real LLM completion in a non-integration test. Stub the seam "
        "(usage._acompletion / litellm.acompletion / the ask_user_impl / the "
        "agent model), or mark the test @pytest.mark.integration."
    )

    async def _aboom(*_a, **_k):
        raise RuntimeError(msg)

    def _boom(*_a, **_k):
        raise RuntimeError(msg)

    try:
        from bird_interact_agents import usage as _usage
        monkeypatch.setattr(_usage, "_acompletion", _aboom, raising=False)
    except Exception:  # noqa: BLE001 — never block collection on a defensive guard
        pass
    try:
        import litellm as _litellm
        monkeypatch.setattr(_litellm, "acompletion", _aboom, raising=False)
        monkeypatch.setattr(_litellm, "completion", _boom, raising=False)
    except Exception:  # noqa: BLE001 — litellm always present; defensive
        pass
