"""T6–T8 + T19-fallback: cluster YAML rendering, Ray Jobs API submission,
and the `kill` fallback's argv (delete-disks=all)."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from bird_interact_agents.cloud import cluster, config  # noqa: E402

yaml = pytest.importorskip("yaml")


RUN_ID = "20260521T1422-pydanticai-raw-a1b2c3"
IMAGE_URI = (
    "us-central1-docker.pkg.dev/motley-team-475011/bird-interact-runner/runner:"
    "deadbeef1234-cafebabe5678"
)
WORKER_SA = "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"


@pytest.fixture(autouse=True)
def _isolate_global_key_dir(tmp_path_factory, monkeypatch):
    """Point the deployment-global SSH key dir at a tmp location so
    rendering tests generate their pinned keypair there instead of
    writing to the real ~/.bird-interact-cloud."""
    key_dir = tmp_path_factory.mktemp("bic_keys")
    monkeypatch.setattr(cluster, "_default_cache_dir", lambda: key_dir)


# ---------------------------------------------------------------------------
# T6 — render(...) produces a YAML with the expected shape.
# ---------------------------------------------------------------------------


def test_render_yaml_shape(tmp_path: Path) -> None:
    yaml_path = cluster.render(
        run_id=RUN_ID,
        image_uri=IMAGE_URI,
        workers=4,
        actors_per_worker=4,
        worker_type="e2-standard-4",
        zone="us-central1-a",
        worker_sa=WORKER_SA,
        max_runtime_hours=8,
        cache_dir=tmp_path,
    )
    rendered = yaml.safe_load(Path(yaml_path).read_text())

    # Image URI threads through to the docker config.
    assert IMAGE_URI in yaml.safe_dump(rendered)
    # Worker count matches.
    workers = rendered.get("available_node_types", {}).get("ray-worker-default", {})
    assert workers.get("max_workers") == 4 or workers.get("min_workers") == 4

    # Run-id label present.
    flat = yaml.safe_dump(rendered)
    assert f"bird-interact-run-id: {RUN_ID}" in flat or RUN_ID in flat

    # Worker SA used.
    assert WORKER_SA in flat


def test_render_sets_slayer_env_roots(tmp_path: Path) -> None:
    """DEV-1468: the container must expose the three slayer artifact roots so
    paths.*_root() resolve to the downloaded /data/... dirs in-cluster (this
    is what makes the actor's download land where the local readers look)."""
    yaml_path = cluster.render(
        run_id=RUN_ID,
        image_uri=IMAGE_URI,
        workers=1,
        actors_per_worker=1,
        worker_type="e2-standard-4",
        zone="us-central1-a",
        worker_sa=WORKER_SA,
        max_runtime_hours=1,
        cache_dir=tmp_path,
    )
    rendered = yaml.safe_load(Path(yaml_path).read_text())
    run_opts = rendered["docker"]["run_options"]
    joined = "\n".join(run_opts)
    assert "BIRD_SLAYER_MODELS_ROOT=/data/slayer_models" in joined
    assert "BIRD_OTF_CACHE_ROOT=/data/slayer_otf_cache" in joined
    assert "BIRD_SLAYER_MODELS_OTF_ROOT=/data/slayer_models_otf" in joined
    # De-bake: BIRD_DB_PATH/BIRD_DATA_PATH are NO LONGER pinned to
    # /data/mini-interact — the actor downloads the dataset from GCS per node
    # and sets the benchmark's data-root/data-file env vars at runtime, so one
    # image+cluster serves any benchmark.
    assert "BIRD_DB_PATH=" not in joined
    assert "BIRD_DATA_PATH=" not in joined


def test_render_pins_ssh_key_and_pushes_pubkey(tmp_path: Path) -> None:
    """Pin our own SSH keypair (so Ray's autoscaler doesn't auto-generate
    and corrupt it across `ray up`/`ray exec`), and push the matching
    public key onto every node via instance metadata `ssh-keys` plus
    `enable-oslogin=FALSE`. Verified against Ray's GCP provider: setting
    `auth.ssh_private_key` makes `_configure_key_pair` early-return, and
    the provider passes `node_config.metadata.items` straight to the GCE
    API without touching `ssh-keys`."""
    yaml_path = cluster.render(
        run_id=RUN_ID, image_uri=IMAGE_URI,
        workers=1, actors_per_worker=1, worker_type="e2-standard-4",
        zone="us-central1-a", worker_sa=WORKER_SA, max_runtime_hours=1,
        cache_dir=tmp_path,
    )
    rendered = yaml.safe_load(Path(yaml_path).read_text())

    # 1. ssh_private_key pinned to a real file.
    priv = rendered["auth"]["ssh_private_key"]
    assert priv and Path(priv).exists()
    pub_text = Path(priv + ".pub").read_text().strip()

    # 2. Each node type pushes the matching pubkey + disables OS Login,
    #    in GCE-API metadata.items form.
    for nt in ("ray-head-default", "ray-worker-default"):
        items = rendered["available_node_types"][nt]["node_config"]["metadata"]["items"]
        by_key = {it["key"]: it["value"] for it in items}
        assert by_key["enable-oslogin"] == "FALSE"
        assert by_key["ssh-keys"].startswith("ubuntu:")
        # The pushed public key must be the pair of the pinned private key.
        assert pub_text in by_key["ssh-keys"]


def test_render_node_type_names_are_gce_label_safe(tmp_path: Path) -> None:
    """Ray's GCP node provider applies the node-type key as a GCE label
    value. GCE rejects dots in label values — node types like
    `ray.head.default` make `ray up` fail with:
        Label value 'ray.head.default' violates format constraints.
    """
    yaml_path = cluster.render(
        run_id=RUN_ID,
        image_uri=IMAGE_URI,
        workers=1, actors_per_worker=1,
        worker_type="e2-standard-4",
        zone="us-central1-a",
        worker_sa=WORKER_SA,
        max_runtime_hours=1,
        cache_dir=tmp_path,
    )
    rendered = yaml.safe_load(Path(yaml_path).read_text())
    for key in rendered["available_node_types"]:
        assert "." not in key, (
            f"node-type key {key!r} has a dot — Ray applies this as a GCE "
            "label value, which rejects dots"
        )
    assert "." not in rendered["head_node_type"]


def test_render_uses_self_delete_not_shutdown(tmp_path: Path) -> None:
    """Codex BLOCKER #11 — must `gcloud compute instances delete`, not `shutdown -h`."""
    yaml_path = cluster.render(
        run_id=RUN_ID,
        image_uri=IMAGE_URI,
        workers=1,
        actors_per_worker=1,
        worker_type="e2-standard-4",
        zone="us-central1-a",
        worker_sa=WORKER_SA,
        max_runtime_hours=2,
        cache_dir=tmp_path,
    )
    text = Path(yaml_path).read_text()
    assert "gcloud" in text
    assert "compute instances delete" in text
    assert "--delete-disks=all" in text
    assert "shutdown -h" not in text


def test_render_parametrizes_region(tmp_path: Path) -> None:
    """A3 — `region` drives BOTH `provider.region` and the Artifact Registry
    docker-credential host (`<region>-docker.pkg.dev`). It must not be
    hardwired to us-central1."""
    # Use an image URI whose own host doesn't mention us-central1, so the
    # only us-central1 reference would be a hardwired one.
    image_uri = (
        "europe-west1-docker.pkg.dev/motley-team-475011/"
        "bird-interact-runner/runner:deadbeef1234-cafebabe5678"
    )
    yaml_path = cluster.render(
        run_id=RUN_ID, image_uri=image_uri,
        workers=1, actors_per_worker=1, worker_type="e2-standard-4",
        zone="europe-west1-b", worker_sa=WORKER_SA, max_runtime_hours=1,
        cache_dir=tmp_path, region="europe-west1",
    )
    text = Path(yaml_path).read_text()
    rendered = yaml.safe_load(text)
    assert rendered["provider"]["region"] == "europe-west1"
    assert "europe-west1-docker.pkg.dev" in text
    # No leftover hardwired us-central1 region anywhere.
    assert "us-central1" not in text


def test_render_region_defaults_to_config(tmp_path: Path) -> None:
    """When `region` is omitted, render falls back to config.REGION
    (us-central1 by default)."""
    yaml_path = cluster.render(
        run_id=RUN_ID, image_uri=IMAGE_URI,
        workers=1, actors_per_worker=1, worker_type="e2-standard-4",
        zone="us-central1-a", worker_sa=WORKER_SA, max_runtime_hours=1,
        cache_dir=tmp_path,
    )
    rendered = yaml.safe_load(Path(yaml_path).read_text())
    assert rendered["provider"]["region"] == config.REGION


def test_render_docker_sock_chown_not_world_writable(tmp_path: Path) -> None:
    """B1 — hand docker.sock to `ubuntu` via `chown`, never `chmod 666`,
    which would let any local user drive the Docker daemon as root."""
    yaml_path = cluster.render(
        run_id=RUN_ID, image_uri=IMAGE_URI,
        workers=1, actors_per_worker=1, worker_type="e2-standard-4",
        zone="us-central1-a", worker_sa=WORKER_SA, max_runtime_hours=1,
        cache_dir=tmp_path,
    )
    text = Path(yaml_path).read_text()
    assert "sudo chown ubuntu /var/run/docker.sock" in text
    assert "chmod 666 /var/run/docker.sock" not in text


# ---------------------------------------------------------------------------
# T7 — render_from_manifest reproduces render(...) for the same inputs.
# ---------------------------------------------------------------------------


def test_render_from_manifest_round_trip(tmp_path: Path) -> None:
    render_inputs = {
        "workers": 2,
        "actors_per_worker": 4,
        "worker_type": "e2-standard-4",
        "zone": "us-central1-a",
        "worker_sa": WORKER_SA,
        "max_runtime_hours": 4,
        "image_uri": IMAGE_URI,
    }
    manifest = {"run_id": RUN_ID, "render_inputs": render_inputs}

    direct = cluster.render(
        run_id=RUN_ID, cache_dir=tmp_path / "a", **render_inputs
    )
    from_manifest = cluster.render_from_manifest(
        manifest, cache_dir=tmp_path / "b"
    )
    a = yaml.safe_load(Path(direct).read_text())
    b = yaml.safe_load(Path(from_manifest).read_text())
    assert a == b


# ---------------------------------------------------------------------------
# T8 — submit_job uses Ray Jobs API (not `ray submit`).
# ---------------------------------------------------------------------------


def test_submit_job_uses_ray_jobs_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex BLOCKER #1 — `ray job submit --no-wait --address ray://...
    --runtime-env-json '{"env_vars": {...}}' -- python .../ray_app.py ...`.

    Parses the runtime-env JSON out of argv and asserts the exact shape.
    """
    captured: list[list[str]] = []

    def fake_run(argv, *_args, **_kwargs):
        captured.append(list(argv))

        class _CP:
            returncode = 0
            stdout = "raysubmit_abc123\n"
            stderr = ""

        return _CP()

    monkeypatch.setattr("subprocess.run", fake_run)

    job_id = cluster.submit_job(
        head_address="ray://10.0.0.1:10001",
        args=["--run-id", RUN_ID, "--gcs-bucket", "b", "--attempt", "1"],
        env_vars={"ANTHROPIC_API_KEY": "sk-test", "CEREBRAS_API_KEY": "sk-cb"},
    )

    assert job_id == "raysubmit_abc123"
    assert captured, "submit_job did not shell out"
    argv = captured[-1]
    assert argv[0] == "ray"
    assert argv[1] == "job"
    assert argv[2] == "submit"

    # Required flags by name.
    assert "--no-wait" in argv
    assert "--runtime-env-json" in argv
    assert "--address" in argv or any(
        a.startswith("--address=") for a in argv
    )

    # `--address` value points to the head.
    addr_idx = argv.index("--address") if "--address" in argv else None
    addr = argv[addr_idx + 1] if addr_idx is not None else next(
        a.split("=", 1)[1] for a in argv if a.startswith("--address=")
    )
    assert addr.startswith("ray://10.0.0.1")

    # `--runtime-env-json` value is valid JSON with exactly {"env_vars": {...}}.
    rej_idx = argv.index("--runtime-env-json")
    runtime_env = json.loads(argv[rej_idx + 1])
    assert set(runtime_env) == {"env_vars"}
    assert runtime_env["env_vars"] == {
        "ANTHROPIC_API_KEY": "sk-test",
        "CEREBRAS_API_KEY": "sk-cb",
    }

    # `--` separator + the in-cluster invocation are on the right of it.
    assert "--" in argv
    dash_idx = argv.index("--")
    rest = argv[dash_idx + 1:]
    # The in-cluster command is `python ...ray_app.py <args>`.
    assert rest[0] == "python"
    assert any("ray_app" in r for r in rest)
    # Caller args are forwarded after the in-cluster command.
    flat = " ".join(shlex.quote(a) for a in rest)
    assert RUN_ID in flat
    assert "--attempt 1" in flat


def test_submit_job_keeps_secrets_out_of_job_runtime_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Production (`yaml_path`) path: API keys must be delivered out-of-band
    via `ray rsync-up` (secret in file CONTENT, not argv) and the job
    submitted with NO secrets in its runtime_env — otherwise `ray job list`
    /the dashboard echo them. `ray_app` is told where to read them via
    `--secrets-file`."""
    captured: list[list[str]] = []
    rsynced_bodies: list[str] = []

    def fake_run(argv, *_args, **_kwargs):
        argv = list(argv)
        captured.append(argv)
        if argv[:2] == ["ray", "rsync-up"]:
            # argv = [ray, rsync-up, <yaml>, <local-tmp>, <remote>]
            rsynced_bodies.append(Path(argv[3]).read_text())

        class _CP:
            returncode = 0
            stdout = "raysubmit_xyz\n"
            stderr = ""

        return _CP()

    monkeypatch.setattr("subprocess.run", fake_run)
    yaml_path = tmp_path / "cluster.yaml"
    yaml_path.write_text("x\n")
    secret = "sk-ant-SUPERSECRETVALUE"

    job_id = cluster.submit_job(
        head_address="ray://10.0.0.1:10001",
        args=["--run-id", RUN_ID, "--attempt", "1"],
        env_vars={"ANTHROPIC_API_KEY": secret},
        yaml_path=yaml_path,
    )

    assert job_id == "raysubmit_xyz"
    # rsync-up came first, then the `ray exec` job submit.
    assert captured[0][:2] == ["ray", "rsync-up"]
    assert captured[1][:2] == ["ray", "exec"]
    # THE key assertion: the secret value appears in NO argv anywhere.
    flat = " ".join(" ".join(c) for c in captured)
    assert secret not in flat, "secret leaked into a command line"
    # It DID travel via the rsync'd file's content (the delivery channel).
    assert any(secret in body for body in rsynced_bodies)
    # The in-cluster command names the secrets file and carries no key.
    inner = captured[1][3]
    assert "--secrets-file" in inner
    assert cluster.SECRETS_REMOTE_PATH in inner
    assert "ANTHROPIC_API_KEY" not in inner
    # The job runtime-env-json is empty (no env_vars).
    assert "'{}'" in inner or '"{}"' in inner


def test_submit_job_no_rsync_when_no_env_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no env vars there's nothing to deliver — skip rsync and don't
    pass `--secrets-file`."""
    captured: list[list[str]] = []

    def fake_run(argv, *_args, **_kwargs):
        captured.append(list(argv))

        class _CP:
            returncode = 0
            stdout = "raysubmit_none\n"
            stderr = ""

        return _CP()

    monkeypatch.setattr("subprocess.run", fake_run)
    yaml_path = tmp_path / "cluster.yaml"
    yaml_path.write_text("x\n")

    cluster.submit_job(
        head_address="ray://10.0.0.1:10001",
        args=["--run-id", RUN_ID],
        env_vars={},
        yaml_path=yaml_path,
    )

    assert all(c[:2] != ["ray", "rsync-up"] for c in captured)
    assert captured[0][:2] == ["ray", "exec"]
    assert "--secrets-file" not in captured[0][3]


# ---------------------------------------------------------------------------
# CR#3 — submit_job raises when it can't parse a ray_job_id from stdout.
# ---------------------------------------------------------------------------


def test_submit_job_raises_on_unparseable_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(argv, *_args, **_kwargs):
        class _CP:
            returncode = 0
            stdout = "some unrelated\nmultiline\noutput\n"  # no `raysubmit_` token
            stderr = ""

        return _CP()

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="ray_job_id"):
        cluster.submit_job(
            head_address="ray://10.0.0.1:10001",
            args=["--run-id", RUN_ID],
            env_vars={"ANTHROPIC_API_KEY": "sk"},
        )


# ---------------------------------------------------------------------------
# T19 split: cluster.fallback_delete_by_label asserts --delete-disks=all.
# ---------------------------------------------------------------------------


def test_ray_down_no_dash_f(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ray down` takes the cluster yaml as a positional argument — `-f`
    is not a valid flag and the argparser rejects it outright. Regression
    for the smoke-time crash."""
    captured: list[list[str]] = []

    def fake_run(argv, *_args, **_kwargs):
        captured.append(list(argv))

        class _CP:
            returncode = 0
            stdout = ""
            stderr = ""

        return _CP()

    monkeypatch.setattr("subprocess.run", fake_run)
    cluster.down(Path("/tmp/cluster.yaml"))
    assert captured
    argv = captured[-1]
    assert argv[0:2] == ["ray", "down"]
    assert "-f" not in argv
    # Yaml path is positional.
    assert str(Path("/tmp/cluster.yaml")) in argv


def test_fallback_delete_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex MAJOR — `kill` fallback must include `--delete-disks=all` so
    we don't leak boot disks. AND `gcloud compute instances delete`
    doesn't accept `--filter`, so the fallback has to list+loop+delete."""
    captured: list[list[str]] = []

    def fake_run(argv, *_args, **_kwargs):
        captured.append(list(argv))

        class _CP:
            returncode = 0
            # Two instances tagged with this run-id, in us-central1-a.
            stdout = (
                "vm-head us-central1-a\n"
                "vm-worker-0 us-central1-a\n"
            )
            stderr = ""

        return _CP()

    monkeypatch.setattr("subprocess.run", fake_run)
    cluster.fallback_delete_by_label(RUN_ID)

    # Step 1: list with --filter.
    list_calls = [c for c in captured if "list" in c]
    assert list_calls, "no `gcloud compute instances list` invocation"
    list_flat = " ".join(list_calls[-1])
    assert "compute instances list" in list_flat
    assert f"labels.bird-interact-run-id={RUN_ID}" in list_flat

    # Step 2: one delete per listed instance, with --delete-disks=all + --zone.
    delete_calls = [c for c in captured if "delete" in c]
    assert len(delete_calls) == 2, f"expected 2 delete calls, got {len(delete_calls)}"
    for argv in delete_calls:
        assert argv[0] == "gcloud"
        flat = " ".join(argv)
        assert "compute instances delete" in flat
        assert "--delete-disks=all" in argv
        assert any(a.startswith("--zone=") for a in argv)
        # The instance name is positional and must NOT be a URL — basename only.
        assert "us-central1-a" in flat


def test_fallback_delete_handles_url_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`gcloud compute instances list --format=value(name,zone)` may emit
    the zone as a full self-link URL. The fallback must extract the basename
    before passing it to `--zone=`."""
    captured: list[list[str]] = []

    def fake_run(argv, *_args, **_kwargs):
        captured.append(list(argv))

        class _CP:
            returncode = 0
            stdout = (
                "vm-head https://www.googleapis.com/compute/v1/"
                "projects/p/zones/us-central1-a\n"
            )
            stderr = ""

        return _CP()

    monkeypatch.setattr("subprocess.run", fake_run)
    cluster.fallback_delete_by_label(RUN_ID)
    delete_calls = [c for c in captured if "delete" in c]
    assert delete_calls
    flat = " ".join(delete_calls[-1])
    assert "--zone=us-central1-a" in flat
    # The full URL must NOT be passed as --zone.
    assert "googleapis.com" not in flat
