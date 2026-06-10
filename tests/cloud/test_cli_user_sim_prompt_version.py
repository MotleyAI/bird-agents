"""DEV-1545: CLI + manifest + resubmit round-trip for
`--user-sim-prompt-version`.

Mirrors the existing `--reasoning-effort` plumbing pattern. Mechanical
contracts only:

  * CLI parses `--user-sim-prompt-version <v2|v3>`, default `None`.
  * Invalid values rejected by argparse choices.
  * Manifest builder serialises the field as snake_case
    (`user_sim_prompt_version`).
  * `_build_job_args` appends `--user-sim-prompt-version <v>` when the
    arg is set and OMITS it otherwise (no `--user-sim-prompt-version None`).
  * `_build_resubmit_args` round-trips the manifest value.
  * Old manifests without the key resubmit WITHOUT emitting the flag
    (Codex #11: backwards-compat).
"""

from __future__ import annotations

import pytest

from bird_interact_agents.cloud import cli, driver  # noqa: E402


def _parse(argv: list[str]):
    """Mirror the local helper in tests/cloud/test_cli.py — auto-add
    `--no-subscription-auth` so the DEV-1535 explicit-auth guard
    doesn't reject the bulk of CLI tests."""
    has_auth = any(
        a in ("--subscription-auth", "--no-subscription-auth") for a in argv
    )
    needs_auth = bool(argv) and argv[0] in ("submit", "annotate")
    if needs_auth and not has_auth:
        argv = [*argv, "--no-subscription-auth"]
    return cli.parse_args(argv)


def _base_submit_argv(extra: list[str]) -> list[str]:
    """Minimal `submit` argv that passes existing guards. `mini-interact`
    + `claude_sdk` + raw query-mode are the cheap-to-validate combo
    used elsewhere in this test file."""
    return [
        "submit",
        "--dataset", "mini-interact",
        "--framework", "claude_sdk",
        "--query-mode", "raw",
        "--agent-model", "anthropic/claude-haiku-4-5-20251001",
        "--instance-ids", "alien_1",
        "--mode", "a-interact",
        "--no-require-annotation",
        *extra,
    ]


# ---------------------------------------------------------------------------
# CLI parse tests
# ---------------------------------------------------------------------------


def test_user_sim_prompt_version_default_is_none() -> None:
    """Default behavior: flag absent → namespace carries `None`. Lets the
    closure-level normalisation in run.py apply "v2" as the effective
    default (Codex #10 + user decision)."""
    ns = _parse(_base_submit_argv([]))
    assert getattr(ns, "user_sim_prompt_version", "MISSING") is None


def test_user_sim_prompt_version_v2_explicit() -> None:
    ns = _parse(_base_submit_argv(["--user-sim-prompt-version", "v2"]))
    assert ns.user_sim_prompt_version == "v2"


def test_user_sim_prompt_version_v3_accepted() -> None:
    ns = _parse(_base_submit_argv(["--user-sim-prompt-version", "v3"]))
    assert ns.user_sim_prompt_version == "v3"


def test_user_sim_prompt_version_invalid_rejected() -> None:
    """argparse `choices=("v2","v3")` → SystemExit(2) on unknown value."""
    with pytest.raises(SystemExit):
        _parse(_base_submit_argv(["--user-sim-prompt-version", "v9"]))


# ---------------------------------------------------------------------------
# Manifest field shape (snake_case, mirrors `reasoning_effort`)
# ---------------------------------------------------------------------------


def test_manifest_includes_user_sim_prompt_version_snake_case() -> None:
    """`driver.build_manifest` must serialise the field as
    `user_sim_prompt_version` (snake_case, mirroring `reasoning_effort`).
    A subtle bug shape is exposing a kebab-case key that ray_app /
    resubmit can't read back."""
    ns = _parse(
        _base_submit_argv(
            ["--user-sim-prompt-version", "v3"]
        )
    )
    manifest = driver.build_manifest(
        ns,
        run_id="rid-test",
        image_uri="us-central1-docker.pkg.dev/p/r/i:t",
        benchmark_data_prefix=None,
    )
    assert "user_sim_prompt_version" in manifest
    assert manifest["user_sim_prompt_version"] == "v3"


def test_manifest_default_none_when_flag_absent() -> None:
    """The manifest key exists but holds `None` — this is what lets
    `_build_*_args` skip the flag when unset (no
    `--user-sim-prompt-version None` ever emitted)."""
    ns = _parse(_base_submit_argv([]))
    manifest = driver.build_manifest(
        ns,
        run_id="rid-test",
        image_uri="us-central1-docker.pkg.dev/p/r/i:t",
        benchmark_data_prefix=None,
    )
    assert manifest.get("user_sim_prompt_version") is None


# ---------------------------------------------------------------------------
# _build_job_args (submit path) appends conditionally
# ---------------------------------------------------------------------------


def test_build_job_args_includes_flag_when_set() -> None:
    ns = _parse(
        _base_submit_argv(["--user-sim-prompt-version", "v3"])
    )
    job_args = driver._build_job_args(ns, "rid-test", attempt=0)
    assert "--user-sim-prompt-version" in job_args
    flag_idx = job_args.index("--user-sim-prompt-version")
    assert job_args[flag_idx + 1] == "v3"


def test_build_job_args_omits_flag_when_unset() -> None:
    """No `--user-sim-prompt-version` and no `None` value ever appears
    in job args when the CLI flag is absent. Codex #10 risk: emitting
    `--user-sim-prompt-version None` (a string `"None"`) would be
    silently accepted by argparse's `default=None` on the receiving
    side and shadow the intended v2 default."""
    ns = _parse(_base_submit_argv([]))
    job_args = driver._build_job_args(ns, "rid-test", attempt=0)
    assert "--user-sim-prompt-version" not in job_args
    assert "None" not in job_args


# ---------------------------------------------------------------------------
# _build_resubmit_args (resubmit path) round-trips and old-manifest compat
# ---------------------------------------------------------------------------


def _minimal_manifest(extra: dict | None = None) -> dict:
    base = {
        "dataset": "mini_interact",
        "framework": "claude_sdk",
        "query_mode": "raw",
        "mode": "a-interact",
        "agent_model": "anthropic/claude-haiku-4-5-20251001",
        "user_sim_model": "anthropic/claude-haiku-4-5-20251001",
        "patience": 250,
        "max_depth": 3,
        "strict": False,
        "use_audited_gold_sql": True,
        "prompt_cache": True,
        "slayer_setup": "pre-encoded",
        "slayer_storage_root": "/data/slayer_models",
        "benchmark_data_prefix": "benchmark-data/mini_interact/x/",
        "render_inputs": {
            "workers": 1,
            "actors_per_worker": 1,
            "worker_type": "e2-standard-4",
            "zone": "us-central1-a",
            "max_runtime_hours": 1,
            "image_uri": "us-central1-docker.pkg.dev/p/r/i:t",
            "project": "p",
            "region": "us-central1",
        },
    }
    if extra:
        base.update(extra)
    return base


def test_resubmit_args_round_trip_v3() -> None:
    manifest = _minimal_manifest({"user_sim_prompt_version": "v3"})
    job_args = driver._build_resubmit_args(
        manifest, run_id="rid-test", missing=["alien_1"], attempt=2
    )
    assert "--user-sim-prompt-version" in job_args
    flag_idx = job_args.index("--user-sim-prompt-version")
    assert job_args[flag_idx + 1] == "v3"


def test_resubmit_args_omits_flag_for_old_manifest_without_key() -> None:
    """Codex #11: a manifest written before this PR will not have the
    key. `_build_resubmit_args` must omit the flag entirely — NOT
    emit `--user-sim-prompt-version None` and NOT crash on `KeyError`."""
    manifest = _minimal_manifest(None)  # NO user_sim_prompt_version key
    assert "user_sim_prompt_version" not in manifest

    job_args = driver._build_resubmit_args(
        manifest, run_id="rid-test", missing=["alien_1"], attempt=2
    )
    assert "--user-sim-prompt-version" not in job_args
    assert "None" not in job_args


def test_resubmit_args_omits_flag_when_manifest_value_is_none() -> None:
    """Newer manifest with the key explicitly None (CLI flag was absent
    at submit time). Same shape — omit, don't serialise `None`."""
    manifest = _minimal_manifest({"user_sim_prompt_version": None})
    job_args = driver._build_resubmit_args(
        manifest, run_id="rid-test", missing=["alien_1"], attempt=2
    )
    assert "--user-sim-prompt-version" not in job_args
    assert "None" not in job_args
