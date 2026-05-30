"""Verify slayer_mcp_stdio_config builds a valid stdio MCP server config."""

import subprocess

import pytest

from bird_interact_agents.config import settings


def test_slayer_mcp_config_resolves_binary():
    """The helper finds the slayer binary and builds a runnable config."""
    from bird_interact_agents.harness import slayer_mcp_stdio_config

    cfg = slayer_mcp_stdio_config(f"{settings.db_path}/_unused")
    assert cfg["command"].endswith("slayer")
    assert cfg["args"] == ["mcp", "--ingest-on-startup"]
    assert "SLAYER_STORAGE" in cfg["env"]
    # The path is normalised (resolve()) — we don't pin its exact form, just
    # that it ends with the unused subdir we passed in.
    assert cfg["env"]["SLAYER_STORAGE"].endswith("_unused")


def test_slayer_mcp_command_is_executable():
    """The slayer binary the config points at actually launches and shows --help."""
    from bird_interact_agents.harness import slayer_mcp_stdio_config

    cfg = slayer_mcp_stdio_config(f"{settings.db_path}/_unused")
    # Spawn `slayer --help` (not `slayer mcp` — that opens a stdio server).
    # We just want to confirm the binary is real.
    result = subprocess.run(
        [cfg["command"], "--help"],
        env=cfg["env"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "slayer" in result.stdout.lower()


def test_slayer_mcp_config_rejects_empty_storage_dir():
    """Empty storage_dir must raise — Path('').resolve() silently aliases to
    CWD, so any unset slayer_storage_root would otherwise corrupt SLAYER_STORAGE."""
    from bird_interact_agents.harness import slayer_mcp_stdio_config

    with pytest.raises(ValueError, match="non-empty storage_dir"):
        slayer_mcp_stdio_config("")


def test_slayer_mcp_storage_per_task(tmp_path):
    """Different storage dirs produce different SLAYER_STORAGE values."""
    from bird_interact_agents.harness import slayer_mcp_stdio_config

    a = slayer_mcp_stdio_config(str(tmp_path / "alien"))
    b = slayer_mcp_stdio_config(str(tmp_path / "robot"))
    assert a["env"]["SLAYER_STORAGE"] != b["env"]["SLAYER_STORAGE"]
    assert a["env"]["SLAYER_STORAGE"].endswith("alien")
    assert b["env"]["SLAYER_STORAGE"].endswith("robot")


# DEV-1508: opt-out for --ingest-on-startup so an OTF-warm storage doesn't
# pay 30-50s of useless schema re-reflection / sample-value refresh /
# embedding hash-walks at MCP boot — which the Claude Agent SDK has no
# timeout knob to wait through, leaving slayer status='pending' for the
# whole session on big-schema LiveSQLBench DBs.


def test_slayer_mcp_config_default_includes_ingest_flag(tmp_path):
    """Default (back-compat) call still ships ``--ingest-on-startup``. The
    committed-reference adapters depend on this to refresh entity embeddings
    against the active embedding model."""
    from bird_interact_agents.harness import slayer_mcp_stdio_config

    cfg = slayer_mcp_stdio_config(str(tmp_path / "store"))
    assert cfg["args"] == ["mcp", "--ingest-on-startup"]


def test_slayer_mcp_config_ingest_on_startup_false_omits_flag(tmp_path):
    """OTF callers pass ``ingest_on_startup=False`` to skip the startup
    re-ingest: the OTF cache by construction IS the post-ingestion state."""
    from bird_interact_agents.harness import slayer_mcp_stdio_config

    cfg = slayer_mcp_stdio_config(
        str(tmp_path / "store"), ingest_on_startup=False,
    )
    assert cfg["args"] == ["mcp"]


def test_slayer_mcp_config_env_unaffected_by_kwarg(tmp_path):
    """Pin: the new kwarg only affects ``args``. ``env["SLAYER_STORAGE"]``
    must still be the resolved abs path so a regression doesn't silently
    point SLayer at the wrong dir."""
    from bird_interact_agents.harness import slayer_mcp_stdio_config

    a = slayer_mcp_stdio_config(str(tmp_path / "store"))
    b = slayer_mcp_stdio_config(str(tmp_path / "store"), ingest_on_startup=False)
    assert a["env"]["SLAYER_STORAGE"] == b["env"]["SLAYER_STORAGE"]
    assert a["command"] == b["command"]


def test_slayer_mcp_config_kwarg_is_keyword_only(tmp_path):
    """The flag MUST be keyword-only — a positional ``True`` shouldn't be
    misread as ``ingest_on_startup=True`` by a careless caller that thought
    they were passing a different positional arg."""
    from bird_interact_agents.harness import slayer_mcp_stdio_config

    with pytest.raises(TypeError):
        slayer_mcp_stdio_config(str(tmp_path / "store"), False)  # type: ignore[misc]
