"""Detect-and-fail-fast prereq checks. Split across the *submitter*
(the local user / ADC) and the *worker SA* (the service account VMs
run as)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any

from bird_interact_agents import provider_registry
from bird_interact_agents.cloud import config, gcs


# Module-level aliases for backwards compatibility — these now route
# through `config` so env-var overrides apply uniformly. Code that reads
# these gets the deployment-default; tests that monkeypatch them still
# work without rewriting.
PROJECT = config.PROJECT
REGION = config.REGION
WORKER_SA = config.WORKER_SA
AR_REPO_LOCATION = config.REGION
AR_REPO = config.AR_REPO

SUBMITTER_REQUIRED_ROLES = (
    "roles/storage.admin",
    "roles/artifactregistry.writer",
    "roles/iam.serviceAccountUser",
    "roles/compute.instanceAdmin.v1",
)

WORKER_SA_REQUIRED_ROLES = (
    "roles/storage.objectUser",
    "roles/artifactregistry.reader",
    # The head's autoscaler runs as this SA and creates/deletes worker
    # VMs (and the self-delete timer deletes instances+disks), so the SA
    # needs compute admin. Without it, preflight passes but workers never
    # launch.
    "roles/compute.instanceAdmin.v1",
)


class PrereqError(RuntimeError):
    """Raised when an environmental precondition isn't satisfied. Carries
    a `.remediation` string with the exact command to fix the gap."""

    def __init__(self, message: str, remediation: str):
        super().__init__(message)
        self.remediation = remediation


# ---------------------------------------------------------------------------
# Helpers (overridable in tests)
# ---------------------------------------------------------------------------


def _python_version_info() -> tuple[int, int, int]:
    return sys.version_info[:3]


def _which(name: str) -> str | None:
    return shutil.which(name)


def _gcloud_active_project() -> str:
    res = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True, text=True, check=False,
    )
    return res.stdout.strip()


def _gcloud_default_region() -> str:
    res = subprocess.run(
        ["gcloud", "config", "get-value", "compute/region"],
        capture_output=True, text=True, check=False,
    )
    return res.stdout.strip()


def _adc_token() -> str:
    res = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True, text=True, check=False,
    )
    return res.stdout.strip()


def _active_principal() -> str:
    """`gcloud config get-value account` → `user:<email>` for the active
    `gcloud auth` principal, or empty string if unresolved."""
    res = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True, text=True, check=False,
    )
    account = (res.stdout or "").strip()
    return account


def _list_submitter_roles() -> set[str]:
    """Roles bound to the active gcloud principal on the project (NOT
    every member's roles)."""
    account = _active_principal()
    if not account:
        return set()
    # Project-level roles for this principal.
    res = subprocess.run(
        [
            "gcloud", "projects", "get-iam-policy", PROJECT,
            "--flatten=bindings[].members",
            f"--filter=bindings.members:user:{account}",
            "--format=value(bindings.role)",
        ],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        return set()
    return {line.strip() for line in res.stdout.splitlines() if line.strip()}


def _roles_for_member(argv: list[str], member: str) -> set[str]:
    """Roles bound to `member` in the IAM policy returned (as JSON) by `argv`.
    Returns an empty set on any gcloud/parse failure."""
    import json as _json
    res = subprocess.run(argv, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return set()
    try:
        policy = _json.loads(res.stdout)
    except _json.JSONDecodeError:
        return set()
    return {
        b["role"] for b in policy.get("bindings", [])
        if member in b.get("members", [])
    }


# Project-level roles that confer `iam.serviceAccounts.actAs` on EVERY service
# account in the project — so holding one lets the worker SA actAs itself even
# without a resource-level self-binding.
_PROJECT_ACTAS_ROLES = frozenset({"roles/iam.serviceAccountUser", "roles/owner"})


def _list_worker_sa_roles(sa_email: str) -> set[str]:
    """Roles bound to `sa_email` across all relevant scopes — project,
    bucket, AND the Artifact Registry repo. The prereq check doesn't care
    *where* a role is bound from, only that the effective grant is
    present.

    Why all three: `compute.instanceAdmin.v1` is typically project-scoped,
    `storage.objectUser` is bucket-scoped (set by `ensure_bucket`), and
    `artifactregistry.reader` is repo-scoped (bucket-level lookups don't
    surface it).

    Each scope is fetched as JSON and filtered in Python — `--filter` is
    inconsistently supported across gcloud subcommands (e.g. `gcloud
    storage buckets get-iam-policy` rejects it outright), so relying on
    it silently returned empty sets when the syntax differed.
    """
    member = f"serviceAccount:{sa_email}"
    project_roles = _roles_for_member([
        "gcloud", "projects", "get-iam-policy", PROJECT, "--format=json",
    ], member)
    bucket_roles = _roles_for_member([
        "gcloud", "storage", "buckets", "get-iam-policy",
        f"gs://{gcs.BUCKET_NAME}", "--format=json",
    ], member)
    ar_roles = _roles_for_member([
        "gcloud", "artifacts", "repositories", "get-iam-policy",
        AR_REPO, f"--location={AR_REPO_LOCATION}",
        "--project", PROJECT, "--format=json",
    ], member)
    return project_roles | bucket_roles | ar_roles


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_python_version() -> None:
    major, minor, _ = _python_version_info()
    if (major, minor) < (3, 11):
        raise PrereqError(
            f"Python >= 3.11 required (have {major}.{minor})",
            remediation=(
                "Install/activate a Python 3.11+ env "
                "(e.g. `conda activate motley3.11`)."
            ),
        )


def check_local_tools() -> None:
    """Required: gcloud, docker, ray. NOT required: gsutil (SDK-only)."""
    for name in ("gcloud", "docker", "ray"):
        if _which(name) is None:
            raise PrereqError(
                f"`{name}` not on PATH",
                remediation=f"Install {name} and ensure it's on PATH.",
            )


def check_gcloud_config() -> None:
    proj = _gcloud_active_project()
    if proj != PROJECT:
        raise PrereqError(
            f"gcloud project is {proj!r}, want {PROJECT!r}",
            remediation=f"gcloud config set project {PROJECT}",
        )
    region = _gcloud_default_region()
    if region != REGION:
        raise PrereqError(
            f"gcloud compute region is {region!r}, want {REGION!r}",
            remediation=f"gcloud config set compute/region {REGION}",
        )


def check_adc() -> None:
    if not _adc_token():
        raise PrereqError(
            "gcloud Application Default Credentials missing",
            remediation="gcloud auth application-default login",
        )


def _required_api_keys(model: str) -> tuple[str, ...]:
    # DEV-1555: registry open-weight providers (moonshot, ...) declare
    # their own auth env var — the registry is the single source.
    registry_keys = provider_registry.required_env_for(model)
    if registry_keys:
        return registry_keys
    if model.startswith("anthropic/"):
        return ("ANTHROPIC_API_KEY",)
    if model.startswith("cerebras/"):
        return ("CEREBRAS_API_KEY",)
    if model.startswith("openai/") or model.startswith("gpt"):
        return ("OPENAI_API_KEY",)
    if model.startswith("gemini/") or model.startswith("google/"):
        return ("GEMINI_API_KEY",)
    return ()


def _is_claude_sdk_framework(framework: str) -> bool:
    return framework.startswith("claude_sdk") or framework == "annotator"


def check_api_keys(
    *, agent_model: str, user_sim_model: str, query_mode: str = "raw",
    framework: str = "", no_subscription_auth: bool = False,
) -> None:
    # DEV-1604: the annotator is now provider-aware — a registry open-weight
    # model is allowed (it runs through the bridge). A non-Anthropic,
    # non-registry model (openai/*, gemini/*, …) still has no claude_sdk session
    # and is rejected; registry models take the provider-key path below.
    if (
        framework == "annotator"
        and not agent_model.startswith("anthropic/")
        and provider_registry.get_provider(agent_model) is None
    ):
        raise PrereqError(
            f"the annotator agent model must be anthropic/* or a registry "
            f"open-weight provider; got {agent_model!r}.",
            remediation="pass an anthropic/* or registry --agent-model for annotate.",
        )
    # DEV-1602 (CodeRabbit): --subscription-auth is Anthropic-ONLY. Reject a
    # claude_sdk* subscription run on a non-Anthropic, non-registry model
    # (openai/*, gemini/*) — otherwise the OAuth branch below (gated only on
    # "not a registry model") would wrongly require CLAUDE_CODE_OAUTH_TOKEN for it.
    if (
        _is_claude_sdk_framework(framework)
        and not no_subscription_auth
        and provider_registry.get_provider(agent_model) is None
        and not agent_model.startswith("anthropic/")
    ):
        raise PrereqError(
            f"--subscription-auth is Anthropic-only; got non-Anthropic agent "
            f"model {agent_model!r}.",
            remediation=(
                "pass an anthropic/* --agent-model for subscription auth, use a "
                "registry model (provider-key path), or pass --no-subscription-auth."
            ),
        )
    # claude_sdk* + subscription auth opted-in → OAuth path. Missing or
    # malformed token = hard failure (no silent fall-back to API key —
    # see `driver.read_api_keys_from_local_env` for the analogous guard).
    # DEV-1602 (registry-first): a registry open-weight agent model takes the
    # provider-key path even with no_subscription_auth=False, so gate the OAuth
    # branch on the agent model not being a registry model. (Non-Anthropic
    # non-registry models are rejected by the Anthropic-only guard above; the
    # annotator by the guard at the top.)
    if (
        _is_claude_sdk_framework(framework)
        and not no_subscription_auth
        and provider_registry.get_provider(agent_model) is None
    ):
        token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        if not token:
            raise PrereqError(
                "--subscription-auth was selected but CLAUDE_CODE_OAUTH_TOKEN "
                "is not set in the submitter's env.",
                remediation=(
                    "source the env file that exports it "
                    "(e.g. `set -a; source .env.ubuntu; set +a`), run "
                    "`claude setup-token`, or pass `--no-subscription-auth` "
                    "to use the ANTHROPIC_API_KEY path."
                ),
            )
        if not token.startswith("sk-ant-oat01-"):
            raise PrereqError(
                "CLAUDE_CODE_OAUTH_TOKEN does not look like a Claude.ai OAuth token "
                "(expected sk-ant-oat01- prefix).",
                remediation="claude setup-token",
            )
        # Agent uses OAuth; user-sim uses a dedicated env var whose value is
        # ANTHROPIC_API_KEY. Require it locally so the driver can ship it.
        needed: set[str] = set()
        if user_sim_model.startswith("anthropic/"):
            needed.add("ANTHROPIC_API_KEY")
        # DEV-1468: slayer embeddings still need OPENAI_API_KEY.
        if query_mode == "slayer":
            needed.add("OPENAI_API_KEY")
        # Non-anthropic keys for user-sim (CEREBRAS, OPENAI, GEMINI).
        for k in _required_api_keys(user_sim_model):
            if k != "ANTHROPIC_API_KEY":
                needed.add(k)
    else:
        needed = set()
        needed.update(_required_api_keys(agent_model))
        needed.update(_required_api_keys(user_sim_model))
        # DEV-1468: slayer mode requires channel-3 embeddings (default
        # openai/text-embedding-3-small), so OPENAI_API_KEY must be present and
        # delivered to the actors regardless of the agent/user-sim providers.
        if query_mode == "slayer":
            needed.add("OPENAI_API_KEY")

    missing = [k for k in sorted(needed) if not os.environ.get(k)]
    if missing:
        cmds = "\n".join(f"export {k}=<your-key>" for k in missing)
        raise PrereqError(
            f"missing API key env vars: {missing}",
            remediation=cmds,
        )


def _missing_roles(have: set[str], want: tuple[str, ...]) -> list[str]:
    """Returns the set of `want` roles not satisfied by `have`. `roles/owner`
    is treated as universal — it carries every other role's permissions
    transitively, so the explicit check would mis-flag an owner as
    'under-privileged'."""
    if "roles/owner" in have:
        return []
    return [r for r in want if r not in have]


def check_submitter_iam() -> None:
    have = _list_submitter_roles()
    missing = _missing_roles(have, SUBMITTER_REQUIRED_ROLES)
    if missing:
        rem_lines = [
            f"gcloud projects add-iam-policy-binding {PROJECT} "
            f"--member=user:$(gcloud config get-value account) --role={r}"
            for r in missing
        ]
        raise PrereqError(
            f"submitter is missing IAM roles: {missing}",
            remediation="\n".join(rem_lines),
        )


def _sa_can_act_as_self(sa_email: str) -> bool:
    """True iff `sa_email` can `iam.serviceAccounts.actAs` ITSELF.

    The head node's autoscaler runs AS the worker SA and launches worker
    VMs that also run as the worker SA — so the SA must be able to
    `actAs` itself, or worker launches fail with
    `SERVICE_ACCOUNT_ACCESS_DENIED` (silent: the head comes up, but
    workers never do and actors hang PENDING).

    The grant can be conferred two ways, both accepted, so a valid setup
    isn't flagged with a redundant-binding remediation:
      1. resource-level: `roles/iam.serviceAccountUser` bound to the SA on
         its OWN IAM policy;
      2. project-level: `roles/iam.serviceAccountUser` / `roles/owner` bound
         to the SA at the project — these confer actAs on every project SA,
         including itself.
    (We can't use `gcloud ... test-iam-permissions`: it tests the CALLING
    identity's permission on the SA, not the worker SA's permission.)"""
    member = f"serviceAccount:{sa_email}"
    # 1. Resource-level self-binding.
    sa_roles = _roles_for_member(
        [
            "gcloud", "iam", "service-accounts", "get-iam-policy", sa_email,
            "--project", PROJECT, "--format=json",
        ],
        member,
    )
    if "roles/iam.serviceAccountUser" in sa_roles:
        return True
    # 2. Project-level grant covering all SAs.
    project_roles = _roles_for_member(
        ["gcloud", "projects", "get-iam-policy", PROJECT, "--format=json"],
        member,
    )
    return bool(project_roles & _PROJECT_ACTAS_ROLES)


def check_worker_sa_iam(sa_email: str = WORKER_SA) -> None:
    have = _list_worker_sa_roles(sa_email)
    missing = _missing_roles(have, WORKER_SA_REQUIRED_ROLES)
    if missing:
        rem_lines = []
        for r in missing:
            if r == "roles/storage.objectUser":
                rem_lines.append(
                    f"gcloud storage buckets add-iam-policy-binding "
                    f"gs://{gcs.BUCKET_NAME} "
                    f"--member=serviceAccount:{sa_email} --role={r}"
                )
            elif r == "roles/artifactregistry.reader":
                rem_lines.append(
                    f"gcloud artifacts repositories add-iam-policy-binding "
                    f"{AR_REPO} --location={AR_REPO_LOCATION} "
                    f"--member=serviceAccount:{sa_email} --role={r}"
                )
            else:
                # Project-scoped roles (e.g. compute.instanceAdmin.v1).
                rem_lines.append(
                    f"gcloud projects add-iam-policy-binding {PROJECT} "
                    f"--member=serviceAccount:{sa_email} --role={r}"
                )
        raise PrereqError(
            f"worker SA is missing IAM roles: {missing}",
            remediation="\n".join(rem_lines),
        )
    # The SA must be able to actAs itself so the head autoscaler can
    # launch worker VMs running as the SA.
    if not _sa_can_act_as_self(sa_email):
        raise PrereqError(
            f"worker SA {sa_email} cannot actAs itself — worker VM launches "
            "will fail with SERVICE_ACCOUNT_ACCESS_DENIED",
            remediation=(
                f"gcloud iam service-accounts add-iam-policy-binding {sa_email} "
                f"--member=serviceAccount:{sa_email} "
                f"--role=roles/iam.serviceAccountUser --project={PROJECT}"
            ),
        )


def ensure_bucket_and_artifact_repo() -> None:
    """Best-effort: create the bucket if missing; verify AR repo exists."""
    try:
        gcs.ensure_bucket(worker_sa_email=WORKER_SA)
    except Exception as e:  # noqa: BLE001
        raise PrereqError(
            f"failed to ensure GCS bucket: {e}",
            remediation=(
                f"gcloud storage buckets create gs://{gcs.BUCKET_NAME} "
                f"--location={REGION} --uniform-bucket-level-access"
            ),
        ) from e
    res = subprocess.run(
        [
            "gcloud", "artifacts", "repositories", "describe", AR_REPO,
            f"--location={AR_REPO_LOCATION}",
        ],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        raise PrereqError(
            f"Artifact Registry repo {AR_REPO!r} not found",
            remediation=(
                f"gcloud artifacts repositories create {AR_REPO} "
                f"--repository-format=docker --location={AR_REPO_LOCATION}"
            ),
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def check(args: Any) -> None:
    """Run every prereq check. Raises on the first failure.

    `ensure_bucket_and_artifact_repo` runs BEFORE `check_worker_sa_iam`
    because `ensure_bucket` is what binds `roles/storage.objectUser` to
    the worker SA on first submit — if we checked first we'd have a
    chicken-and-egg failure on a freshly provisioned project.
    """
    check_python_version()
    check_local_tools()
    check_gcloud_config()
    check_adc()
    check_api_keys(
        agent_model=args.agent_model,
        user_sim_model=args.user_sim_model,
        query_mode=getattr(args, "query_mode", "raw"),
        framework=getattr(args, "framework", ""),
        no_subscription_auth=getattr(args, "no_subscription_auth", False),
    )
    check_submitter_iam()
    ensure_bucket_and_artifact_repo()
    check_worker_sa_iam(WORKER_SA)
