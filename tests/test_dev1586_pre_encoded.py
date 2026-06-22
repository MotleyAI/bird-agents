"""DEV-1586 — pre-encoded mode for the SLayer claude_sdk agents.

Mechanical contract tests only (no prompt-content assertions, per the
project rule): the shared helper, write-tool filtering across all four
agents' surfaces, the benchmark-aware storage resolver + fail-clear guards,
cloud artifact routing, provenance roundtrip, and the batch-encoder script.
Behavioural validation is left to real cloud smokes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bird_interact_agents.agents import _pre_encoded as pe


# ---------------------------------------------------------------------------
# Shared helper: sources, derivation, source-root, tool filtering
# ---------------------------------------------------------------------------


def test_sources_and_derivation():
    assert pe.PRE_ENCODED_SOURCES == ("otf", "custom")
    assert pe.derive_slayer_setup(None) == "on-the-fly"
    assert pe.derive_slayer_setup("otf") == "pre-encoded"
    assert pe.derive_slayer_setup("custom") == "pre-encoded"


def test_validate_pre_encoded_source():
    pe.validate_pre_encoded_source(None)
    pe.validate_pre_encoded_source("otf")
    pe.validate_pre_encoded_source("custom")
    with pytest.raises(ValueError):
        pe.validate_pre_encoded_source("bogus")


def test_source_root_selection():
    from bird_interact_agents import paths

    otf = pe.pre_encoded_source_root("otf", benchmark="mini-interact")
    assert otf == paths.slayer_models_otf_root(benchmark="mini-interact")
    custom = pe.pre_encoded_source_root("custom", benchmark="mini-interact")
    assert custom == paths.slayer_models_root()


def test_otf_read_root_matches_cloud_download_root(monkeypatch, tmp_path):
    """Codex r2 #4: under the cloud env override, the root the AGENT reads
    (pre_encoded_source_root) must equal the root the worker DOWNLOADS into
    (paths.slayer_models_otf_root, used by ray_app._slayer_artifacts_for)."""
    from bird_interact_agents import paths
    from bird_interact_agents.cloud import ray_app

    monkeypatch.setenv("BIRD_SLAYER_MODELS_OTF_ROOT", str(tmp_path))
    agent_root = pe.pre_encoded_source_root("otf", benchmark="mini-interact")
    cfg = {
        "framework": "claude_sdk", "slayer_setup": "pre-encoded",
        "pre_encoded_source": "otf", "dataset": "mini-interact",
    }
    (artifact, download_root, required) = ray_app._slayer_artifacts_for(cfg)[0]
    assert artifact == "slayer_models_otf"
    assert download_root == agent_root == paths.slayer_models_otf_root(
        benchmark="mini-interact"
    )
    # honoured the env override
    assert str(tmp_path) in str(agent_root)


def test_write_tool_name_sets():
    assert pe.WRITE_SLAYER_TOOLS == frozenset(
        {"create_model", "edit_model", "save_memory", "validate_models"}
    )
    assert pe.WRITE_SLAYER_TOOL_NAMES == frozenset(
        f"mcp__slayer__{t}" for t in pe.WRITE_SLAYER_TOOLS
    )


def test_strip_write_slayer_tools_bare():
    out = pe.strip_write_slayer_tools(
        ["help", "inspect_model", "create_model", "search", "save_memory",
         "models_summary", "edit_model", "validate_models"]
    )
    assert out == ["help", "inspect_model", "search", "models_summary"]


def test_strip_write_tool_names_prefixed():
    out = pe.strip_write_tool_names([
        "Task",
        "mcp__slayer__inspect_model",
        "mcp__slayer__create_model",
        "mcp__slayer__edit_model",
        "mcp__bird-interact-tools__query",
        "mcp__slayer__save_memory",
        "mcp__slayer__validate_models",
    ])
    assert out == [
        "Task",
        "mcp__slayer__inspect_model",
        "mcp__bird-interact-tools__query",
    ]


# ---------------------------------------------------------------------------
# Write tools absent from ALL four agents' pre-encoded surfaces
# ---------------------------------------------------------------------------


def test_v0_slayer_whitelist_drops_write_tools():
    from bird_interact_agents.agents.claude_sdk_otf.agent import SLAYER_MCP_TOOLS

    filtered = pe.strip_write_slayer_tools(SLAYER_MCP_TOOLS)
    assert set(filtered) & pe.WRITE_SLAYER_TOOLS == set()
    # introspection survives
    for keep in ("help", "list_datasources", "models_summary",
                 "inspect_model", "search"):
        assert keep in filtered


@pytest.mark.parametrize(
    "module",
    [
        "bird_interact_agents.agents.claude_sdk_otf_v1.agent",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent",
    ],
)
def test_v1_main_tools_drop_write_tools(module):
    import importlib

    mod = importlib.import_module(module)
    filtered = pe.strip_write_tool_names(mod.MAIN_TOOLS)
    assert set(filtered) & pe.WRITE_SLAYER_TOOL_NAMES == set()
    # Task (subagent spawner) + query + submit survive.
    assert "Task" in filtered
    assert "mcp__bird-interact-tools__query" in filtered
    assert "mcp__bird-interact-tools__submit_query" in filtered
    # DISCOVERY_TOOLS were always read-only.
    assert set(mod.DISCOVERY_TOOLS) & pe.WRITE_SLAYER_TOOL_NAMES == set()


# ---------------------------------------------------------------------------
# Agent construction + prompt selection (all 4 flavors)
# ---------------------------------------------------------------------------

_AGENTS = [
    ("bird_interact_agents.agents.claude_sdk_otf.agent", "ClaudeSDKOtfAgent", "one-shot"),
    ("bird_interact_agents.agents.claude_sdk_otf_ainteract.agent", "ClaudeSDKOtfAInteractAgent", "a-interact"),
    ("bird_interact_agents.agents.claude_sdk_otf_v1.agent", "ClaudeSDKOtfAgent", "one-shot"),
    ("bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent", "ClaudeSDKOtfAInteractAgent", "a-interact"),
]


@pytest.mark.parametrize("module,cls_name,eval_mode", _AGENTS)
def test_agent_accepts_pre_encoded_and_stores_source(module, cls_name, eval_mode):
    import importlib

    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    a = cls(slayer_setup="pre-encoded", pre_encoded_source="otf")
    assert a.pre_encoded_source == "otf"
    assert a.slayer_setup == "pre-encoded"
    # default on-the-fly unchanged
    b = cls()
    assert b.pre_encoded_source is None
    assert b.slayer_setup == "on-the-fly"


@pytest.mark.parametrize("module,cls_name,eval_mode", _AGENTS)
def test_agent_rejects_contradictory_setup(module, cls_name, eval_mode):
    import importlib

    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    # source set but slayer_setup on-the-fly → contradiction
    with pytest.raises(ValueError):
        cls(slayer_setup="on-the-fly", pre_encoded_source="otf")
    # pre-encoded slayer_setup with no source → contradiction
    with pytest.raises(ValueError):
        cls(slayer_setup="pre-encoded", pre_encoded_source=None)
    # bad source value
    with pytest.raises(ValueError):
        cls(slayer_setup="pre-encoded", pre_encoded_source="bogus")


@pytest.mark.parametrize("module,cls_name,eval_mode", _AGENTS)
def test_build_prompt_switches_on_source(module, cls_name, eval_mode):
    import importlib

    mod = importlib.import_module(module)
    td = {"amb_user_query": "q?", "selected_database": "demo"}
    pre = mod._build_prompt(eval_mode, td, 100.0, "otf")
    otf = mod._build_prompt(eval_mode, td, 100.0, None)
    assert "ALREADY ENCODED" in pre
    assert "create_model" not in pre and "edit_model" not in pre
    # on-the-fly prompt is the encoding one (mentions create_model/encode)
    assert pre != otf


# ---------------------------------------------------------------------------
# Benchmark-aware storage resolver (Codex DEV-1586 High#1) + fail-clear
# ---------------------------------------------------------------------------


def _make_ref_dir(root: Path, db: str, *, marker: bool, embeddings: bool) -> Path:
    db_dir = root / db
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "models.yaml").write_text("models: []\n")
    if marker:
        (db_dir / "_reference_fp.txt").write_text("fp\n")
    if embeddings:
        (db_dir / "embeddings.db").write_bytes(b"\x00sqlite-ish")
    return db_dir


async def test_resolver_fail_clear_missing_otf_marker(tmp_path, monkeypatch):
    root = tmp_path / "otf"
    _make_ref_dir(root, "demo", marker=False, embeddings=True)
    monkeypatch.setattr(pe, "pre_encoded_source_root", lambda s, *, benchmark: root)
    with pytest.raises(pe.PreEncodedSetupError):
        await pe.resolve_pre_encoded_storage_dir(
            db_name="demo",
            task_data={"instance_id": "i1"},
            data_path_base=str(tmp_path),
            benchmark="mini-interact",
            source="otf",
        )


async def test_resolver_fail_clear_empty_custom(tmp_path, monkeypatch):
    root = tmp_path / "custom"
    (root / "demo").mkdir(parents=True)  # empty dir
    monkeypatch.setattr(pe, "pre_encoded_source_root", lambda s, *, benchmark: root)
    with pytest.raises(pe.PreEncodedSetupError):
        await pe.resolve_pre_encoded_storage_dir(
            db_name="demo",
            task_data={"instance_id": "i1"},
            data_path_base=str(tmp_path),
            benchmark="mini-interact",
            source="custom",
        )


async def test_resolver_fail_clear_missing_embeddings(tmp_path, monkeypatch):
    root = tmp_path / "otf"
    _make_ref_dir(root, "demo", marker=True, embeddings=False)
    monkeypatch.setattr(pe, "pre_encoded_source_root", lambda s, *, benchmark: root)
    with pytest.raises(pe.PreEncodedSetupError):
        await pe.resolve_pre_encoded_storage_dir(
            db_name="demo",
            task_data={"instance_id": "i1"},
            data_path_base=str(tmp_path),
            benchmark="mini-interact",
            source="otf",
        )


async def test_resolver_threads_benchmark_aware_roots(tmp_path, monkeypatch):
    """Codex High#1: the resolver must call build_task_variant_storage with the
    chosen reference root AND a benchmark/data-path-derived mini_interact_root /
    db_root (NOT the sibling-dir fallback)."""
    root = tmp_path / "otf"
    _make_ref_dir(root, "demo", marker=True, embeddings=True)
    monkeypatch.setattr(pe, "pre_encoded_source_root", lambda s, *, benchmark: root)

    captured: dict = {}

    async def fake_build(**kwargs):
        captured.update(kwargs)
        return tmp_path / "variant" / "demo"

    monkeypatch.setattr(pe, "build_task_variant_storage", fake_build)

    data_base = tmp_path / "data"
    data_base.mkdir()
    out, deleted = await pe.resolve_pre_encoded_storage_dir(
        db_name="demo",
        task_data={"instance_id": "i1"},
        data_path_base=str(data_base),
        benchmark="mini-interact",
        source="otf",
    )
    assert captured["canonical_storage_root"] == root
    assert captured["db_name"] == "demo"
    # benchmark-aware: re-anchored at the resolved data path, not the
    # canonical_storage_root.parent.parent/"mini-interact" fallback.
    assert captured["mini_interact_root"] == data_base.resolve()
    assert captured["db_root"] == data_base.resolve()
    assert deleted == []


# ---------------------------------------------------------------------------
# Validation: omitted-both derives on-the-fly; raw rejects the flag
# ---------------------------------------------------------------------------


def test_make_runner_omitted_both_derives_on_the_fly():
    """Codex r2 #1: a programmatic slayer caller that omits BOTH slayer_setup
    and pre_encoded_source must NOT raise — slayer_setup derives to on-the-fly."""
    from bird_interact_agents import run as run_mod

    run_mod.make_runner(
        framework="claude_sdk",
        dataset="mini-interact",
        query_mode="slayer",
        mode="a-interact",
        agent_model="anthropic/claude-sonnet-4-5",
        strict=False,
        prompt_cache=True,
        max_depth=3,
        slayer_storage_root=None,
    )


def test_validate_rejects_pre_encoded_in_raw_mode():
    """Codex r2 #2: --pre-encoded-models is slayer-only; raw must reject it
    (the framework gate is no longer bypassed by the raw early-return)."""
    from bird_interact_agents import run as run_mod

    with pytest.raises(ValueError):
        run_mod._validate_slayer_setup(
            slayer_setup="pre-encoded", framework="claude_sdk",
            query_mode="raw", mode="a-interact", pre_encoded_source="otf",
        )


# ---------------------------------------------------------------------------
# Cloud artifact routing + driver job-args + resubmit back-compat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [("otf", "slayer_models_otf"), ("custom", "slayer_models"), (None, "slayer_models")],
)
def test_ray_app_artifacts_for_pre_encoded_source(source, expected):
    from bird_interact_agents.cloud import ray_app

    cfg = {
        "framework": "claude_sdk", "slayer_setup": "pre-encoded",
        "dataset": "mini-interact",
    }
    if source is not None:
        cfg["pre_encoded_source"] = source
    artifacts = ray_app._slayer_artifacts_for(cfg)
    assert [a for a, _, _ in artifacts] == [expected]


@pytest.mark.parametrize(
    "source,expected",
    [("otf", "slayer_models_otf"), ("custom", "slayer_models")],
)
def test_driver_uploads_for_pre_encoded_source(source, expected):
    from types import SimpleNamespace
    from bird_interact_agents.cloud import driver

    args = SimpleNamespace(
        framework="claude_sdk", slayer_setup="pre-encoded",
        pre_encoded_source=source, dataset="mini-interact",
        instance_ids=("alien_1",),
    )
    uploads = driver._slayer_uploads_for(args)
    assert [art for _, art, _ in uploads] == [expected]


def test_driver_job_args_emit_pre_encoded_models():
    from types import SimpleNamespace
    from bird_interact_agents.cloud import driver

    args = SimpleNamespace(
        framework="claude_sdk", query_mode="slayer", mode="a-interact",
        dataset="mini-interact", agent_model="anthropic/claude-sonnet-4-6",
        user_sim_model="anthropic/claude-haiku-4-5-20251001", patience=3,
        max_depth=3, workers=1, actors_per_worker=1, strict=False,
        use_audited_gold_sql=False, prompt_cache=True, reasoning_effort=None,
        user_sim_prompt_version=None, slayer_setup="pre-encoded",
        slayer_storage_root="/data/slayer_models", pre_encoded_source="otf",
        instance_ids=["alien_1"],
    )
    job_args = driver._build_job_args(args, "run-1", attempt=1)
    assert "--pre-encoded-models" in job_args
    assert job_args[job_args.index("--pre-encoded-models") + 1] == "otf"


def test_driver_job_args_omit_pre_encoded_when_on_the_fly():
    from types import SimpleNamespace
    from bird_interact_agents.cloud import driver

    args = SimpleNamespace(
        framework="claude_sdk", query_mode="slayer", mode="a-interact",
        dataset="mini-interact", agent_model="anthropic/claude-sonnet-4-6",
        user_sim_model="anthropic/claude-haiku-4-5-20251001", patience=3,
        max_depth=3, workers=1, actors_per_worker=1, strict=False,
        use_audited_gold_sql=False, prompt_cache=True, reasoning_effort=None,
        user_sim_prompt_version=None, slayer_setup="on-the-fly",
        slayer_storage_root="/data/slayer_models", pre_encoded_source=None,
        instance_ids=["alien_1"],
    )
    job_args = driver._build_job_args(args, "run-1", attempt=1)
    assert "--pre-encoded-models" not in job_args


def test_resubmit_back_compat_legacy_pre_encoded_defaults_custom():
    """A pre-DEV-1586 manifest with slayer_setup=pre-encoded and no
    pre_encoded_source resubmits as 'custom' (the legacy meaning)."""
    from bird_interact_agents.cloud import driver

    manifest = {
        "run_id": "r", "framework": "claude_sdk", "query_mode": "slayer",
        "mode": "a-interact", "dataset": "mini-interact",
        "agent_model": "anthropic/claude-sonnet-4-6",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "patience": 3, "max_depth": 3, "prompt_cache": True,
        "slayer_setup": "pre-encoded", "instance_ids": ["alien_1"],
        "render_inputs": {"workers": 1, "actors_per_worker": 1},
    }
    job_args = driver._build_resubmit_args(manifest, "r", ["alien_1"], 2)
    assert job_args[job_args.index("--pre-encoded-models") + 1] == "custom"


# ---------------------------------------------------------------------------
# Provenance roundtrip
# ---------------------------------------------------------------------------


def test_run_metadata_records_pre_encoded_source(tmp_path):
    from bird_interact_agents import results_db

    db = results_db.open_db(tmp_path / "results.db")
    results_db.insert_run_metadata(
        db, run_id="r1", agent_model="m", user_sim_model="u",
        framework="claude_sdk", mode="a-interact", query_mode="slayer",
        slayer_setup="pre-encoded", pre_encoded_source="otf",
    )
    row = db.execute(
        "SELECT pre_encoded_source FROM run_metadata WHERE run_id='r1'"
    ).fetchone()
    assert row[0] == "otf"


def test_submission_config_has_pre_encoded_source():
    from bird_interact_agents.eval.annotation_schema import SubmissionConfig

    cfg = SubmissionConfig(slayer_setup="pre-encoded", pre_encoded_source="custom")
    assert cfg.pre_encoded_source == "custom"


# ---------------------------------------------------------------------------
# upload_back skips merge for pre-encoded (read-only) runs
# ---------------------------------------------------------------------------


def test_upload_back_skips_merge_for_pre_encoded():
    from bird_interact_agents.cloud import upload_back

    # The merge-back is gated to otf_encode + on-the-fly; a pre-encoded
    # (read-only) run must early-return without touching GCS. We pass
    # client=None: if the guard failed, the body would dereference it.
    cfg = {
        "query_mode": "slayer", "framework": "claude_sdk",
        "slayer_setup": "pre-encoded", "pre_encoded_source": "otf",
        "dataset": "mini-interact",
    }
    assert upload_back.upload_otf_reference_delta(
        run_id="r", cfg=cfg, shard="demo", uploaded_dbs=set(),
        initial_seed_fp_by_db={}, client=None,
    ) is None


# ---------------------------------------------------------------------------
# Batch encoder script
# ---------------------------------------------------------------------------


def test_batch_script_enumerates_and_calls_ensure_reference(monkeypatch, tmp_path):
    import scripts.build_otf_references as bs

    monkeypatch.setattr(bs, "_dbs_for_benchmark", lambda b: ["db_a", "db_b"])
    monkeypatch.setattr(bs, "_build_model", lambda m: ("MODEL", None))
    monkeypatch.setattr(bs, "make_setup_build_encoder", lambda **kw: "ENC")

    calls: list[str] = []

    async def fake_ensure(db, **kwargs):
        calls.append(db)

        class _E:
            reference_dir = tmp_path / db
        return _E()

    monkeypatch.setattr(bs, "ensure_db_reference", fake_ensure)
    rc = bs.main(["mini-interact", "--agent-model", "anthropic/claude-opus-4-7"])
    assert rc == 0
    assert calls == ["db_a", "db_b"]


def test_batch_script_only_subset(monkeypatch, tmp_path):
    import scripts.build_otf_references as bs

    monkeypatch.setattr(bs, "_dbs_for_benchmark", lambda b: ["db_a", "db_b", "db_c"])
    monkeypatch.setattr(bs, "_build_model", lambda m: ("MODEL", None))
    monkeypatch.setattr(bs, "make_setup_build_encoder", lambda **kw: "ENC")

    calls: list[str] = []

    async def fake_ensure(db, **kwargs):
        calls.append(db)

        class _E:
            reference_dir = tmp_path / db
        return _E()

    monkeypatch.setattr(bs, "ensure_db_reference", fake_ensure)
    rc = bs.main([
        "mini-interact", "--agent-model", "anthropic/claude-opus-4-7",
        "--only", "db_b",
    ])
    assert rc == 0
    assert calls == ["db_b"]
