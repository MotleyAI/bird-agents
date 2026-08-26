"""DEV-1822 integration: real Cube container round-trip against a scratch
Postgres — model generation, meta/load/sql, tenant isolation (C10), and the
core promise that a persisted submission regrades WITHOUT Cube running (C11).

Excluded from the default suite (`-m 'not integration'`); run with
`pytest -m integration -k cube`. Requires docker + network for the pinned image.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time

import pytest

pytestmark = pytest.mark.integration

psycopg2 = pytest.importorskip("psycopg2")

from bird_interact_agents.cube_local import deploy, model_gen
from bird_interact_agents.cube_local.client import CubeClient, CubeApiError, mint_cube_jwt


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _docker(*args, **kw):
    return subprocess.run(["docker", *args], check=kw.pop("check", True),
                          capture_output=True, text=True, **kw)


_INIT_SQL = """
CREATE DATABASE db_a;
CREATE DATABASE db_b;
\\connect db_a
CREATE TABLE customers (id int PRIMARY KEY, region text, attrs jsonb);
INSERT INTO customers VALUES (1,'US','{"tier":"gold"}'),(2,'EU','{"tier":"silver"}');
CREATE TABLE orders (id int PRIMARY KEY, customer_id int REFERENCES customers(id), amount numeric);
INSERT INTO orders VALUES (1,1,100),(2,2,25);
\\connect db_b
CREATE TABLE widgets (id int PRIMARY KEY, kind text);
INSERT INTO widgets VALUES (1,'bolt');
"""


@pytest.fixture(scope="module")
def pg_env(tmp_path_factory):
    if not shutil.which("docker"):
        pytest.skip("docker not available")
    port = _free_port()
    name = "bird-cube-itest-pg"
    _docker("rm", "-f", name, check=False)
    init = tmp_path_factory.mktemp("pg") / "init.sql"
    init.write_text(_INIT_SQL)
    _docker("run", "-d", "--name", name, "-p", f"{port}:5432",
            "-e", "POSTGRES_USER=bird_interact", "-e", "POSTGRES_PASSWORD=pw",
            "-v", f"{init}:/docker-entrypoint-initdb.d/init.sql:ro", "postgres:16")
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                psycopg2.connect(host="127.0.0.1", port=port, dbname="db_a",
                                 user="bird_interact", password="pw",
                                 connect_timeout=3).close()
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        else:
            pytest.fail("scratch postgres did not become ready")
        yield {"BIRD_PG_HOST": "127.0.0.1", "BIRD_PG_PORT": str(port),
               "BIRD_PG_USER": "bird_interact", "BIRD_PG_PASSWORD": "pw"}
    finally:
        _docker("rm", "-f", name, check=False)


@pytest.fixture(scope="module")
def cube(pg_env, tmp_path_factory, monkeypatch_module):
    root = tmp_path_factory.mktemp("cube_local")
    monkeypatch_module.setenv("BIRD_CUBE_LOCAL_ROOT", str(root.parent))
    monkeypatch_module.setattr(
        "bird_interact_agents.paths.cube_local_root",
        lambda *, benchmark: root, raising=False,
    )
    model_gen.ensure_models("livesqlbench-base-lite", ["db_a", "db_b"], pg_env)
    info = deploy.ensure_cube_running("livesqlbench-base-lite", pg_env)
    try:
        yield info
    finally:
        deploy.stop_cube("livesqlbench-base-lite")


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _client(info, db):
    return CubeClient(info.base_url, info.api_secret, db)


def test_meta_and_tenant_isolation(cube):
    a = _client(cube, "db_a").meta()
    b = _client(cube, "db_b").meta()
    assert {c["name"] for c in a["cubes"]} >= {"customers", "orders"}
    assert {c["name"] for c in b["cubes"]} == {"widgets"}


def test_load_and_sql_roundtrip(cube):
    c = _client(cube, "db_a")
    data = c.load({"measures": ["orders.amount_sum"], "dimensions": ["customers.region"]})
    by_region = {r["customers.region"]: r["orders.amount_sum"] for r in data}
    assert by_region["US"] == "100"
    text, params = c.sql({"measures": ["orders.amount_sum"], "dimensions": ["customers.region"]})
    assert "sum(" in text.lower()


def test_cross_tenant_query_blocked(cube):
    with pytest.raises(CubeApiError):
        _client(cube, "db_a").load({"measures": ["widgets.count"]})


def test_submission_regrades_without_cube(cube, pg_env):
    """C11: the materialized SQL is standalone — regrading it needs only the
    persisted SQL + Postgres (BIRD_PG_*), never Cube. Proven by executing it
    against Postgres directly, consulting no BIRD_CUBE_* state at all."""
    from bird_interact_agents.cube_local.submission import cube_query_to_sql
    sql = cube_query_to_sql(
        {"measures": ["orders.amount_sum"], "dimensions": ["customers.region"]},
        _client(cube, "db_a"),
    )
    assert "BIRD_CUBE" not in sql and "$" not in sql  # no placeholders, no cube coupling
    conn = psycopg2.connect(host=pg_env["BIRD_PG_HOST"], port=int(pg_env["BIRD_PG_PORT"]),
                            dbname="db_a", user="bird_interact", password="pw")
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
    finally:
        conn.close()
    assert any(str(r[-1]) in ("100", "100.0", "100.00") for r in rows)
