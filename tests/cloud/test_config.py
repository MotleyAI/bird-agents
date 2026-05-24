"""Deployment-identifier env-var overrides for `bird_interact_agents.cloud.config`."""

from __future__ import annotations

import importlib
import os

import pytest

from bird_interact_agents.cloud import config


# Every env var that overrides a module-level default. `config` reads these at
# import time, so observing the shipped defaults requires all of them unset.
_OVERRIDE_VARS = (
    "BIRD_INTERACT_CLOUD_PROJECT",
    "BIRD_INTERACT_CLOUD_REGION",
    "BIRD_INTERACT_CLOUD_ZONE",
    "BIRD_INTERACT_CLOUD_BUCKET",
    "BIRD_INTERACT_CLOUD_AR_REPO",
    "BIRD_INTERACT_CLOUD_IMAGE_NAME",
    "BIRD_INTERACT_CLOUD_WORKER_SA",
)


# ---------------------------------------------------------------------------
# Defaults reflect the shipped (motley-team-475011) deployment.
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    # The module-level constants are computed at import time, so a developer
    # (or CI) with an ambient BIRD_INTERACT_CLOUD_* override set would
    # otherwise make this assert the override, not the shipped default. Clear
    # the overrides and reload to observe the true defaults; restore + reload
    # in `finally` so the module isn't left mutated for other tests.
    saved = {v: os.environ.pop(v, None) for v in _OVERRIDE_VARS}
    try:
        fresh = importlib.reload(config)
        assert fresh.PROJECT == "motley-team-475011"
        assert fresh.REGION == "us-central1"
        assert fresh.ZONE == "us-central1-a"
        assert fresh.BUCKET_NAME == "motley-team-birdbench"
        assert fresh.AR_REPO == "bird-interact-runner"
        assert fresh.WORKER_SA == (
            "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"
        )
        assert fresh.image_uri_prefix() == (
            "us-central1-docker.pkg.dev/motley-team-475011/"
            "bird-interact-runner/runner"
        )
    finally:
        for var, val in saved.items():
            if val is not None:
                os.environ[var] = val
        importlib.reload(config)


# ---------------------------------------------------------------------------
# Env-var overrides take effect for the call-time helper.
# ---------------------------------------------------------------------------


def test_image_uri_prefix_honors_env_var_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BIRD_INTERACT_CLOUD_PROJECT", "staging-proj-42")
    monkeypatch.setenv("BIRD_INTERACT_CLOUD_REGION", "europe-west1")
    monkeypatch.setenv("BIRD_INTERACT_CLOUD_AR_REPO", "staging-runner")
    monkeypatch.setenv("BIRD_INTERACT_CLOUD_IMAGE_NAME", "runner-staging")

    uri = config.image_uri_prefix()
    assert uri == (
        "europe-west1-docker.pkg.dev/staging-proj-42/staging-runner/runner-staging"
    )


def test_image_uri_prefix_partial_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override only the project; everything else falls back to defaults."""
    monkeypatch.setenv("BIRD_INTERACT_CLOUD_PROJECT", "my-sandbox")
    uri = config.image_uri_prefix()
    assert uri == (
        "us-central1-docker.pkg.dev/my-sandbox/bird-interact-runner/runner"
    )
