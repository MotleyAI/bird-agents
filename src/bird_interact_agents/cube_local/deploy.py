"""DEV-1822: lifecycle for the ONE local Cube container (dev mode, multitenant).

Shaped on `local_postgres` / `cloud.bridge_proxy`: flock-serialized, adopt an
already-running container whose label fingerprint matches, else (re)start. The
container fingerprint deliberately excludes model content — dev-mode hot-reload
picks up regenerated models without a restart (Codex C6; spike-verified).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import secrets
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from bird_interact_agents import paths
from bird_interact_agents.cube_local import conf

DEFAULT_PORT = 4008
DEFAULT_IMAGE = "cubejs/cube:v1.7.26"
READY_TIMEOUT_S = 120
_FP_LABEL = "bird_cube_fp"
_PORT_LABEL = "bird_cube_port"


class CubeRuntimeInfo(BaseModel):
    base_url: str
    api_secret: str
    container_name: str
    port: int


def container_name(benchmark: str) -> str:
    return f"bird-cube-local-{benchmark}"


# --- pure helpers (unit-tested) --------------------------------------------

def resolve_port(preferred: int, *, is_free: Callable[[int], bool]) -> int:
    port = preferred
    while not is_free(port):
        port += 1
    return port


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def runtime_fingerprint(*, image: str, conf_hash: str, pg_host: str,
                        pg_port: str, pg_user: str, secret: str) -> str:
    blob = json.dumps({
        "image": image, "conf_hash": conf_hash, "pg_host": pg_host,
        "pg_port": str(pg_port), "pg_user": pg_user,
        "secret": hashlib.sha256(secret.encode()).hexdigest(),
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def decide_action(state: Optional[dict], *, want_fingerprint: str) -> str:
    if state is None:
        return "start"
    if state.get("running") and state.get("fingerprint") == want_fingerprint:
        return "adopt"
    return "restart"


def container_env(pg_env: dict, *, port: int, secret: str) -> dict[str, str]:
    return {
        "CUBEJS_DEV_MODE": "true",
        "CUBEJS_API_SECRET": secret,
        "CUBEJS_DB_TYPE": "postgres",
        "CUBEJS_DB_HOST": pg_env["BIRD_PG_HOST"],
        "CUBEJS_DB_PORT": str(pg_env.get("BIRD_PG_PORT", 5432)),
        "CUBEJS_DB_USER": pg_env.get("BIRD_PG_USER", "bird_interact"),
        "CUBEJS_DB_PASS": pg_env.get("BIRD_PG_PASSWORD", ""),
        # Cube's API server port env is `PORT` (Node standard), NOT CUBEJS_PORT.
        "PORT": str(port),
        "CUBEJS_SCHEDULED_REFRESH_TIMER": "false",
        "CUBEJS_ALLOW_UNGROUPED_WITHOUT_PRIMARY_KEY": "true",
    }


def api_secret(root: Path) -> str:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    f = root / ".api_secret"
    if f.exists():
        return f.read_text().strip()
    secret = secrets.token_hex(24)
    f.write_text(secret)
    os.chmod(f, 0o600)
    return secret


@contextlib.contextmanager
def deploy_lock(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".deploy.lock"
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# --- docker orchestration (integration paths) ------------------------------

def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def _inspect(name: str) -> Optional[dict]:
    proc = _docker(
        "inspect", name, "--format",
        '{{.State.Running}}|{{index .Config.Labels "%s"}}|{{index .Config.Labels "%s"}}'
        % (_FP_LABEL, _PORT_LABEL),
        check=False,
    )
    if proc.returncode != 0:
        return None
    running, fp, port = (proc.stdout.strip().split("|") + ["", "", ""])[:3]
    return {"running": running == "true", "fingerprint": fp,
            "port": int(port) if port.isdigit() else None}


def _wait_ready(port: int, timeout_s: int = READY_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/readyz"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    raise RuntimeError(f"Cube did not become ready on :{port} within {timeout_s}s")


def _run_container(name: str, image: str, conf_dir: Path, env: dict,
                   port: int, fingerprint: str) -> None:
    args = ["run", "-d", "--name", name, "--network", "host",
            "--label", f"{_FP_LABEL}={fingerprint}",
            "--label", f"{_PORT_LABEL}={port}",
            "-v", f"{conf_dir}:/cube/conf"]
    for key, val in env.items():
        args += ["-e", f"{key}={val}"]
    args.append(image)
    _docker(*args)


def ensure_cube_running(benchmark: str, pg_env: dict) -> CubeRuntimeInfo:
    """Idempotently ensure the per-benchmark Cube container is up + healthy."""
    root = paths.cube_local_root(benchmark=benchmark)
    with deploy_lock(root):
        conf_dir = conf.render_conf(root)
        secret = api_secret(root)
        image = os.environ.get("BIRD_CUBE_IMAGE", DEFAULT_IMAGE)
        want_fp = runtime_fingerprint(
            image=image, conf_hash=conf.conf_content_hash(root),
            pg_host=pg_env["BIRD_PG_HOST"], pg_port=str(pg_env.get("BIRD_PG_PORT", 5432)),
            pg_user=pg_env.get("BIRD_PG_USER", "bird_interact"), secret=secret,
        )
        name = container_name(benchmark)
        state = _inspect(name)
        action = decide_action(state, want_fingerprint=want_fp)
        if action == "adopt" and state and state.get("port"):
            port = int(state["port"])
            _wait_ready(port)
            return _info(name, secret, port)
        if action == "restart":
            _docker("rm", "-f", name, check=False)
        preferred = int(os.environ.get("BIRD_CUBE_PORT", DEFAULT_PORT))
        port = resolve_port(preferred, is_free=_port_free)
        _run_container(name, image, conf_dir,
                       container_env(pg_env, port=port, secret=secret), port, want_fp)
        _wait_ready(port)
        return _info(name, secret, port)


def _info(name: str, secret: str, port: int) -> CubeRuntimeInfo:
    return CubeRuntimeInfo(
        base_url=f"http://127.0.0.1:{port}/cubejs-api/v1",
        api_secret=secret, container_name=name, port=port,
    )


def poll_models_ready(info: CubeRuntimeInfo, dbs, *, timeout_s: int = 60) -> None:
    """After a model regen, poll `/v1/meta` per DB until cubes appear (closes the
    write→hot-recompile race before the first task query)."""
    from bird_interact_agents.cube_local.client import CubeClient
    deadline = time.monotonic() + timeout_s
    for db in dbs:
        client = CubeClient(info.base_url, info.api_secret, db)
        while time.monotonic() < deadline:
            try:
                if client.meta().get("cubes"):
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)


def stop_cube(benchmark: str) -> None:
    _docker("rm", "-f", container_name(benchmark), check=False)


def container_status(benchmark: str) -> Optional[dict]:
    return _inspect(container_name(benchmark))
