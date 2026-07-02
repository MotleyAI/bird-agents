"""Pure-helper tests for the local-postgres run tooling.

Covers only the deterministic, side-effect-free logic:
* ``setup_local_postgres.env_exports`` / ``cluster_dir`` — the BIRD_PG_* contract
  and the worktree-safe cluster location.
* ``run_local_postgres.load_env_file`` — the dotenv parser that loads the auth
  token into the environment.

The subprocess-driven cluster lifecycle (initdb/pg_ctl/createdb/psql) and the
GCS annotation sync are integration behaviour, exercised by real local runs, not
unit-tested here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(mod_name: str):
    # scripts/ modules import each other as top-level siblings, so the dir must
    # be importable.
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPTS / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


setup_local_postgres = _load("setup_local_postgres")
run_local_postgres = _load("run_local_postgres")


def test_env_exports_shape():
    exp = setup_local_postgres.env_exports(5544)
    assert exp == {
        "BIRD_PG_HOST": "127.0.0.1",
        "BIRD_PG_PORT": "5544",
        "BIRD_PG_USER": "bird_interact",
        "BIRD_PG_PASSWORD": "bird_interact",
    }
    # Port flows through verbatim.
    assert setup_local_postgres.env_exports(6000)["BIRD_PG_PORT"] == "6000"


def test_cluster_dir_is_under_main_checkout():
    # Worktree-safe: cluster lives in the main checkout, name is stable.
    assert setup_local_postgres.cluster_dir().name == ".local_pg"


def test_required_roles_include_owner_and_login():
    # Dumps do `OWNER TO root`; harness connects as bird_interact. Both needed.
    assert set(setup_local_postgres._REQUIRED_ROLES) == {"bird_interact", "root"}


def test_load_env_file_parses_export_and_quotes(tmp_path, monkeypatch):
    env = tmp_path / ".env.test"
    env.write_text(
        "# a comment\n"
        "\n"
        "export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-abc\n"
        'ANTHROPIC_API_KEY="sk-ant-key"\n'
        "PLAIN=value\n"
        "  export SPACED = spaced_val \n"  # note surrounding spaces
    )
    for k in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "PLAIN", "SPACED"):
        monkeypatch.delenv(k, raising=False)
    n = run_local_postgres.load_env_file(env)
    import os
    assert n == 4
    assert os.environ["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-abc"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-key"  # quotes stripped
    assert os.environ["PLAIN"] == "value"
    assert os.environ["SPACED"] == "spaced_val"  # key/val trimmed


def test_load_env_file_absent_is_noop(tmp_path):
    assert run_local_postgres.load_env_file(tmp_path / "nope.env") == 0
