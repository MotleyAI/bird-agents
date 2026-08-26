"""DEV-1822: Cube container deploy helpers (Codex C6 fingerprint, C8 adopt/port).

Pure helpers are unit-tested here (docker CLI monkeypatched / not invoked); a
real `docker run` round-trip lives in the integration test.
"""

from __future__ import annotations

import os
import stat

import pytest

from bird_interact_agents.cube_local import deploy


PG_ENV = {
    "BIRD_PG_HOST": "127.0.0.1", "BIRD_PG_PORT": "5544",
    "BIRD_PG_USER": "bird_interact", "BIRD_PG_PASSWORD": "pw",
}


# --- port resolution --------------------------------------------------------

def test_resolve_port_prefers_free():
    assert deploy.resolve_port(4008, is_free=lambda p: True) == 4008


def test_resolve_port_bumps_when_busy():
    busy = {4008, 4009}
    assert deploy.resolve_port(4008, is_free=lambda p: p not in busy) == 4010


# --- runtime fingerprint (C6) ----------------------------------------------

def _fp(**over):
    kw = dict(image="cubejs/cube:v1.7.26", conf_hash="abc",
              pg_host="127.0.0.1", pg_port="5544", pg_user="bird_interact",
              secret="sekret")
    kw.update(over)
    return deploy.runtime_fingerprint(**kw)


def test_fingerprint_stable():
    assert _fp() == _fp()


@pytest.mark.parametrize("field,val", [
    ("image", "cubejs/cube:v9.9.9"),
    ("conf_hash", "different"),
    ("pg_host", "10.0.0.1"),
    ("pg_port", "5432"),
    ("pg_user", "other"),
    ("secret", "rotated"),
])
def test_fingerprint_changes_per_input(field, val):
    assert _fp(**{field: val}) != _fp()


def test_fingerprint_has_no_model_input():
    """C6: model content must NOT feed the container fingerprint — dev-mode
    hot-reload picks up regenerated models without a restart, so a `models`
    parameter would force needless restarts."""
    import inspect
    params = set(inspect.signature(deploy.runtime_fingerprint).parameters)
    assert params == {"image", "conf_hash", "pg_host", "pg_port", "pg_user", "secret"}


# --- adopt / restart / start decision (C8) ---------------------------------

def test_decide_action_start_when_absent():
    assert deploy.decide_action(None, want_fingerprint="fp") == "start"


def test_decide_action_adopt_when_running_and_matching():
    state = {"running": True, "fingerprint": "fp"}
    assert deploy.decide_action(state, want_fingerprint="fp") == "adopt"


def test_decide_action_restart_on_fingerprint_drift():
    state = {"running": True, "fingerprint": "OLD"}
    assert deploy.decide_action(state, want_fingerprint="fp") == "restart"


def test_decide_action_restart_when_stopped():
    state = {"running": False, "fingerprint": "fp"}
    assert deploy.decide_action(state, want_fingerprint="fp") == "restart"


# --- container env from BIRD_PG_* ------------------------------------------

def test_container_env_maps_pg_and_sets_dev_mode():
    env = deploy.container_env(PG_ENV, port=4008, secret="sekret")
    assert env["CUBEJS_DB_TYPE"] == "postgres"
    assert env["CUBEJS_DB_HOST"] == "127.0.0.1"
    assert env["CUBEJS_DB_PORT"] == "5544"
    assert env["CUBEJS_DB_USER"] == "bird_interact"
    assert env["CUBEJS_DB_PASS"] == "pw"
    assert env["CUBEJS_API_SECRET"] == "sekret"
    assert env["PORT"] == "4008"  # Cube's API port env is PORT, not CUBEJS_PORT
    assert env["CUBEJS_DEV_MODE"] == "true"
    assert env["CUBEJS_ALLOW_UNGROUPED_WITHOUT_PRIMARY_KEY"] == "true"
    # per-tenant DB is chosen by driverFactory; no global CUBEJS_DB_NAME
    assert "CUBEJS_DB_NAME" not in env


# --- api secret file --------------------------------------------------------

def test_api_secret_created_0600_and_stable(tmp_path):
    s1 = deploy.api_secret(tmp_path)
    secret_file = tmp_path / ".api_secret"
    assert secret_file.exists()
    assert stat.S_IMODE(os.stat(secret_file).st_mode) == 0o600
    assert deploy.api_secret(tmp_path) == s1  # stable across calls


# --- lock -------------------------------------------------------------------

def test_deploy_lock_serializes(tmp_path):
    # The cm must acquire+release cleanly (flock on a lock file under root).
    with deploy.deploy_lock(tmp_path):
        pass
    with deploy.deploy_lock(tmp_path):
        pass
