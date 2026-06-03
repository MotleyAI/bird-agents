"""Cluster lifecycle wrappers around `ray up` / `ray down` / `ray job submit`."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


from bird_interact_agents.cloud import config


DEFAULT_ZONE = config.ZONE
DEFAULT_WORKER_SA = config.WORKER_SA


def _template_path() -> Path:
    return Path(__file__).parent / "cluster.yaml.j2"


def _default_cache_dir() -> Path:
    return Path.home() / ".bird-interact-cloud"


def ensure_ssh_keypair(cache_dir: Path | None = None) -> tuple[Path, str]:
    """Generate (once) a dedicated ed25519 keypair for bird-interact-cloud
    clusters and return (private_key_path, public_key_text).

    Pinning our own key — referenced from `auth.ssh_private_key` and
    pushed to the VM via instance metadata — avoids Ray's GCP autoscaler
    auto-generating keys. The auto-generated path is flaky: `ray up` and
    `ray exec` derive keys differently but share the `<name>.pub`
    filename, so an `exec` between two `up`s corrupts the keypair and the
    next `up` fails with `private key contents do not match public`."""
    cache = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    priv = cache / "cluster_ssh_key"
    pub = cache / "cluster_ssh_key.pub"
    if not priv.exists() or not pub.exists():
        # Remove a half-present pair before regenerating.
        priv.unlink(missing_ok=True)
        pub.unlink(missing_ok=True)
        subprocess.run(
            [
                "ssh-keygen", "-t", "ed25519", "-N", "", "-q",
                "-f", str(priv), "-C", "bird-interact-cloud",
            ],
            check=True, capture_output=True,
        )
        priv.chmod(0o600)
    return priv, pub.read_text().strip()


def render(
    *,
    run_id: str,
    image_uri: str,
    workers: int,
    actors_per_worker: int,
    worker_type: str,
    zone: str,
    worker_sa: str,
    max_runtime_hours: float,
    cache_dir: Path | None = None,
    project: str | None = None,
    region: str | None = None,
) -> Path:
    """Render the cluster YAML to disk and return the path. Cached under
    `~/.bird-interact-cloud/<run-id>.yaml` (or the override `cache_dir`).

    `project` / `region` default to `config.PROJECT` / `config.REGION` — pass
    explicitly only when rendering for a different deployment in the same
    process. `region` drives both `provider.region` and the Artifact Registry
    docker-credential host (`<region>-docker.pkg.dev`)."""
    from jinja2 import Template  # type: ignore[import-not-found]

    cache = Path(cache_dir) if cache_dir else _default_cache_dir()
    # The SSH keypair is deployment-global (one key reused across all
    # clusters), so it lives in the default dir — NOT the per-run yaml
    # cache_dir. Otherwise two renders with different cache_dirs would
    # mint different keys and diverge.
    ssh_priv, ssh_pub = ensure_ssh_keypair()

    text = _template_path().read_text()
    tmpl = Template(text, keep_trailing_newline=True)
    rendered = tmpl.render(
        run_id=run_id,
        image_uri=image_uri,
        workers=workers,
        actors_per_worker=actors_per_worker,
        worker_type=worker_type,
        zone=zone,
        worker_sa=worker_sa,
        max_runtime_seconds=int(max_runtime_hours * 3600),
        project=project or config.PROJECT,
        region=region or config.REGION,
        ssh_private_key=str(ssh_priv),
        ssh_public_key=ssh_pub,
    )
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{run_id}.yaml"
    out.write_text(rendered)
    return out


def render_from_manifest(
    manifest: dict[str, Any], *, cache_dir: Path | None = None
) -> Path:
    ri = manifest["render_inputs"]
    return render(
        run_id=manifest["run_id"],
        image_uri=ri["image_uri"],
        workers=ri["workers"],
        actors_per_worker=ri["actors_per_worker"],
        worker_type=ri["worker_type"],
        zone=ri.get("zone", DEFAULT_ZONE),
        worker_sa=ri.get("worker_sa", DEFAULT_WORKER_SA),
        max_runtime_hours=ri["max_runtime_hours"],
        project=ri.get("project", config.PROJECT),
        region=ri.get("region", config.REGION),
        cache_dir=cache_dir,
    )


def up(yaml_path: Path) -> None:
    subprocess.run(
        ["ray", "up", "--yes", str(yaml_path)],
        check=True,
    )


def down(yaml_path: Path) -> None:
    # `ray down` takes the cluster yaml as a positional argument — there
    # is no `-f` flag (`-f` is `ray up`'s for `--no-restart`, but on
    # `ray down` it means nothing and trips the argparser).
    subprocess.run(
        ["ray", "down", "--yes", str(yaml_path)],
        check=True,
    )


def head_address(yaml_path: Path) -> str:
    """Resolve the head node's address. Production: shell out to `ray
    get-head-ip`. The format the Jobs API wants is `ray://<ip>:10001`."""
    res = subprocess.run(
        ["ray", "get-head-ip", str(yaml_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    ip = res.stdout.strip().splitlines()[-1].strip()
    return f"ray://{ip}:10001"


# Path (inside the head container) the secrets file is rsync'd to. The
# in-cluster driver reads it, applies the env vars as a per-actor
# runtime_env, then deletes it. Keeping secrets OUT of the job's
# runtime_env is deliberate: `ray job list`/the dashboard echo the job
# runtime_env back, so API keys placed there leak into logs.
SECRETS_REMOTE_PATH = "/tmp/bird_interact_secrets.json"


def _rsync_secrets(
    yaml_path: Path, env_vars: dict[str, str], remote_path: str
) -> None:
    """Deliver `env_vars` to the head container as a JSON file via
    `ray rsync-up`. The secret values travel in the file's CONTENT (which
    rsync streams) — never on a command line — so they don't appear in any
    logged argv. The local temp file is deleted immediately afterwards."""
    import os
    import tempfile

    # mkstemp already creates the file 0o600 (owner-only).
    fd, tmp = tempfile.mkstemp(prefix="bird_secrets_", suffix=".json")
    try:
        # Use a buffered file object so the whole payload is written — a raw
        # os.write isn't guaranteed to flush every byte in one call, which
        # could truncate the JSON and break secrets loading nondeterministically.
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(env_vars, f)
        res = subprocess.run(
            # The CLI command is `ray rsync-up` (hyphen), NOT `rsync_up`.
            ["ray", "rsync-up", str(yaml_path), tmp, remote_path],
            capture_output=True, text=True, check=False,
        )
        if res.returncode != 0:
            # stderr is safe to surface — the secret is in the file body,
            # not in argv/stderr.
            raise RuntimeError(
                f"`ray rsync-up` of the secrets file exited "
                f"{res.returncode}.\n--- stderr ---\n{res.stderr}\n"
            )
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


_DEFAULT_RAY_APP_PATH = (
    "/app/bird-interact-agents/src/bird_interact_agents/cloud/ray_app.py"
)


def submit_job(
    *,
    head_address: str,
    args: list[str],
    env_vars: dict[str, str],
    yaml_path: Path | None = None,
    ray_app_path: str | None = None,
) -> str:
    """Submit a Ray Jobs API job by SSH-execing on the head node via
    `ray exec`. Runs `ray job submit` *inside* the head's container,
    where it can reach the dashboard on localhost:8265 without any
    public-port exposure — neither 10001 (Ray Client gRPC) nor 8265
    (dashboard) need to be in the GCE firewall.

    `head_address` is kept in the signature for backward compatibility
    with mocked tests; `yaml_path` is what production needs for `ray exec`.

    Secrets handling: in the production (`yaml_path`) path, `env_vars` are
    delivered out-of-band via `ray rsync_up` (see `_rsync_secrets`) and the
    job is submitted with an EMPTY runtime_env, so API keys never land in
    `ray job list`/dashboard metadata. The in-cluster `ray_app.py` reads the
    rsync'd file (via `--secrets-file`) and applies them per-actor.

    Returns the ray_job_id from stdout.
    """
    app_path = ray_app_path or _DEFAULT_RAY_APP_PATH
    if yaml_path is not None:
        full_args = list(args)
        runtime_env: dict = {}
        if env_vars:
            _rsync_secrets(yaml_path, env_vars, SECRETS_REMOTE_PATH)
            full_args += ["--secrets-file", SECRETS_REMOTE_PATH]
        # Compose the in-cluster `ray job submit` invocation — NO secrets in
        # the runtime-env-json (it would surface in `ray job list`). Quote
        # each value-bearing token exactly once with shlex.quote; the static
        # flags need no quoting. (Avoids a fragile "skip re-quoting tokens
        # that already start with a single quote" heuristic.)
        inner_cmd = " ".join([
            "ray", "job", "submit",
            "--no-wait",
            "--address", "http://localhost:8265",
            "--runtime-env-json", shlex.quote(json.dumps(runtime_env)),
            "--",
            "python", app_path,
            *[shlex.quote(a) for a in full_args],
        ])
        argv = ["ray", "exec", str(yaml_path), inner_cmd]
    else:
        # Legacy direct path (unit-test only): embed env_vars in the job
        # runtime_env. Production always goes through `yaml_path` above, where
        # secrets are delivered out-of-band and never enter the runtime_env.
        runtime_env = {"env_vars": env_vars}
        argv = [
            "ray", "job", "submit", "--no-wait",
            "--address", head_address,
            "--runtime-env-json", json.dumps(runtime_env),
            "--",
            "python", app_path,
            *args,
        ]
    res = subprocess.run(argv, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        # Surface stderr — `check=True` would have raised
        # CalledProcessError but loses the stderr in subprocess's str().
        # With GCP this is where firewall / auth / connectivity issues
        # show up; we want them in the user-visible traceback.
        raise RuntimeError(
            f"`ray job submit` exited {res.returncode}.\n"
            f"--- argv ---\n{argv}\n"
            f"--- stdout ---\n{res.stdout}\n"
            f"--- stderr ---\n{res.stderr}\n"
        )
    # `ray job submit` prints `Submitted Ray Job <id>` (or similar) to stdout.
    # Strip ANSI color escapes — Ray emits SGR sequences (`\x1b[…m`) around
    # the job id when stdout is a TTY (which `ray exec`'s SSH session is),
    # and they bleed into the token if not stripped.
    import re as _re
    ansi_re = _re.compile(r"\x1b\[[0-9;]*m")
    out = ansi_re.sub("", (res.stdout or "")).strip()
    for tok in reversed(out.split()):
        if "raysubmit" in tok:
            return tok.strip(".'\"")
    raise RuntimeError(
        f"could not parse a ray_job_id from `ray job submit` stdout: {out!r}"
    )


def head_is_alive(run_id: str) -> bool:
    """True iff there's at least one GCE instance with the run-id label
    whose status is RUNNING."""
    res = subprocess.run(
        [
            "gcloud", "compute", "instances", "list",
            f"--filter=labels.bird-interact-run-id={run_id}",
            "--format=value(status)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return False
    return any(
        line.strip() == "RUNNING" for line in res.stdout.splitlines()
    )


def fallback_delete_by_label(run_id: str) -> None:
    """Last-ditch cleanup if `ray down` fails. Deletes every instance with
    the run-id label AND its boot disk.

    `gcloud compute instances delete` does NOT accept `--filter`; we have
    to list+delete in two steps. Each delete needs the instance name AND
    its zone (positional + `--zone=`).
    """
    listing = subprocess.run(
        [
            "gcloud", "compute", "instances", "list",
            f"--filter=labels.bird-interact-run-id={run_id}",
            "--format=value(name,zone)",
        ],
        capture_output=True, text=True, check=False,
    )
    if listing.returncode != 0:
        return
    for line in listing.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        name, zone = parts[0], parts[1]
        # `zone` from the list is a URL; we want the basename.
        zone = zone.rsplit("/", 1)[-1]
        subprocess.run(
            [
                "gcloud", "compute", "instances", "delete", name,
                "--quiet", "--delete-disks=all", f"--zone={zone}",
            ],
            check=False,
        )
