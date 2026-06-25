"""T31: prereq checks, split submitter vs worker SA."""

from __future__ import annotations

import pytest

from bird_interact_agents.cloud import prereqs  # noqa: E402


# ---------------------------------------------------------------------------
# Each individual check raises PrereqError with `.remediation` on failure.
# ---------------------------------------------------------------------------


def test_missing_python_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereqs, "_python_version_info", lambda: (3, 9, 7))
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_python_version()
    assert hasattr(exc.value, "remediation")
    assert exc.value.remediation


def test_missing_local_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereqs, "_which", lambda name: None if name == "docker" else "/x")
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_local_tools()
    assert "docker" in str(exc.value).lower()
    assert exc.value.remediation


def test_gcloud_config_wrong_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereqs, "_gcloud_active_project", lambda: "wrong-proj")
    monkeypatch.setattr(prereqs, "_gcloud_default_region", lambda: "us-central1")
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_gcloud_config()
    assert "gcloud config set project motley-team-475011" in exc.value.remediation


def test_gcloud_config_wrong_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereqs, "_gcloud_active_project", lambda: "motley-team-475011")
    monkeypatch.setattr(prereqs, "_gcloud_default_region", lambda: "us-east1")
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_gcloud_config()
    assert "gcloud config set compute/region us-central1" in exc.value.remediation


def test_missing_api_key_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_api_keys(
            agent_model="anthropic/claude-sonnet-4-5",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
        )
    # Exact remediation: `export ANTHROPIC_API_KEY=...`
    assert "export ANTHROPIC_API_KEY=" in exc.value.remediation


def test_missing_api_key_cerebras(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_api_keys(
            agent_model="cerebras/zai-glm-4.7",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
        )
    assert "export CEREBRAS_API_KEY=" in exc.value.remediation


# ---------------------------------------------------------------------------
# DEV-1468 — slayer mode requires OPENAI_API_KEY (channel-3 embeddings are
# mandatory in cloud; default model openai/text-embedding-3-small).
# ---------------------------------------------------------------------------


def test_slayer_requires_openai_key_even_with_anthropic_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_api_keys(
            agent_model="anthropic/claude-sonnet-4-5",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            query_mode="slayer",
        )
    assert "OPENAI_API_KEY" in str(exc.value)
    assert "export OPENAI_API_KEY=" in exc.value.remediation


def test_slayer_passes_when_openai_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    # Must not raise.
    prereqs.check_api_keys(
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        query_mode="slayer",
    )


def test_raw_mode_does_not_require_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # raw mode with anthropic models: OPENAI not needed → no raise.
    prereqs.check_api_keys(
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        query_mode="raw",
    )


def test_submitter_owner_satisfies_all_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A submitter with `roles/owner` should pass — owner carries all the
    explicit role permissions transitively. Without this, real org owners
    (the common case) would be wrongly flagged as under-privileged."""
    monkeypatch.setattr(prereqs, "_list_submitter_roles", lambda: {"roles/owner"})
    # Must not raise.
    prereqs.check_submitter_iam()


def test_submitter_iam_missing_storage_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prereqs,
        "_list_submitter_roles",
        lambda: {"roles/storage.objectViewer"},
    )
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_submitter_iam()
    # Exact gcloud command for the missing role.
    rem = exc.value.remediation
    assert "gcloud projects add-iam-policy-binding motley-team-475011" in rem
    assert "--role=roles/storage.admin" in rem


def test_submitter_iam_missing_artifact_registry_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prereqs,
        "_list_submitter_roles",
        lambda: {"roles/storage.admin", "roles/iam.serviceAccountUser",
                 "roles/compute.instanceAdmin.v1"},
    )
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_submitter_iam()
    rem = exc.value.remediation
    assert "--role=roles/artifactregistry.writer" in rem


def test_worker_sa_missing_object_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prereqs,
        "_list_worker_sa_roles",
        lambda _sa: {"roles/storage.objectViewer"},
    )
    sa = "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_worker_sa_iam(sa)
    rem = exc.value.remediation
    assert "gcloud storage buckets add-iam-policy-binding" in rem
    assert "--role=roles/storage.objectUser" in rem
    assert sa in rem


def test_worker_sa_missing_artifact_registry_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prereqs,
        "_list_worker_sa_roles",
        lambda _sa: {"roles/storage.objectUser"},
    )
    sa = "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"
    # actAs-self present so we reach the AR-reader check, not the actAs one.
    monkeypatch.setattr(prereqs, "_sa_can_act_as_self", lambda _sa: True)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_worker_sa_iam(sa)
    assert "--role=roles/artifactregistry.reader" in exc.value.remediation


def test_worker_sa_requires_compute_instance_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker SA needs compute.instanceAdmin.v1 so the head autoscaler
    can create/delete worker VMs (A2). It's project-scoped, so its
    remediation must be a `projects add-iam-policy-binding`, NOT an AR-repo
    or bucket binding."""
    assert "roles/compute.instanceAdmin.v1" in prereqs.WORKER_SA_REQUIRED_ROLES
    sa = "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"
    # Everything present EXCEPT compute.instanceAdmin.v1.
    monkeypatch.setattr(
        prereqs, "_list_worker_sa_roles",
        lambda _sa: {"roles/storage.objectUser",
                     "roles/artifactregistry.reader"},
    )
    monkeypatch.setattr(prereqs, "_sa_can_act_as_self", lambda _sa: True)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_worker_sa_iam(sa)
    rem = exc.value.remediation
    # The single remediation line for this role must be the project-scoped
    # binding — not mis-routed to an AR-repo or bucket binding.
    [line] = [
        ln for ln in rem.splitlines()
        if "roles/compute.instanceAdmin.v1" in ln
    ]
    assert line.startswith(
        "gcloud projects add-iam-policy-binding motley-team-475011"
    )
    assert f"--member=serviceAccount:{sa}" in line
    assert "artifacts repositories" not in line
    assert "storage buckets" not in line


def test_worker_sa_cannot_act_as_self(monkeypatch: pytest.MonkeyPatch) -> None:
    """Roles all present, but the SA can't actAs itself → the head
    autoscaler can't launch workers. Remediation must be the
    add-iam-policy-binding on the SA itself."""
    sa = "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"
    monkeypatch.setattr(
        prereqs, "_list_worker_sa_roles",
        lambda _sa: {"roles/storage.objectUser", "roles/artifactregistry.reader",
                     "roles/compute.instanceAdmin.v1"},
    )
    monkeypatch.setattr(prereqs, "_sa_can_act_as_self", lambda _sa: False)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_worker_sa_iam(sa)
    rem = exc.value.remediation
    assert "service-accounts add-iam-policy-binding" in rem
    assert f"--member=serviceAccount:{sa}" in rem
    assert "--role=roles/iam.serviceAccountUser" in rem


def test_worker_sa_act_as_self_parses_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_sa_can_act_as_self` returns True only when the SA is a member with
    serviceAccountUser on its OWN policy."""
    import subprocess as _sp
    import json as _json
    sa = "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"

    def make_run(policy):
        def _run(argv, *_a, **_kw):
            return _sp.CompletedProcess(argv, 0, stdout=_json.dumps(policy), stderr="")
        return _run

    # SA bound on itself → True
    monkeypatch.setattr(_sp, "run", make_run({
        "bindings": [{"role": "roles/iam.serviceAccountUser",
                      "members": [f"serviceAccount:{sa}", "user:egor@motley.ai"]}]
    }))
    assert prereqs._sa_can_act_as_self(sa) is True

    # Only a user bound (not the SA itself) → False
    monkeypatch.setattr(_sp, "run", make_run({
        "bindings": [{"role": "roles/iam.serviceAccountUser",
                      "members": ["user:egor@motley.ai"]}]
    }))
    assert prereqs._sa_can_act_as_self(sa) is False


def test_sa_can_act_as_self_via_project_level_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A project-level serviceAccountUser/owner grant on the SA confers actAs
    on every project SA — including itself — so preflight must accept it even
    with NO resource-level self-binding (Codex: avoid a false-positive
    redundant-binding remediation)."""
    import subprocess as _sp
    import json as _json
    sa = "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"
    member = f"serviceAccount:{sa}"

    def fake_run(argv, *_a, **_kw):
        if "service-accounts" in argv:
            # SA's OWN policy: no self-binding.
            policy = {"bindings": [{"role": "roles/iam.serviceAccountUser",
                                    "members": ["user:egor@motley.ai"]}]}
        else:
            # Project policy: the SA holds serviceAccountUser project-wide.
            policy = {"bindings": [{"role": "roles/iam.serviceAccountUser",
                                    "members": [member]}]}
        return _sp.CompletedProcess(argv, 0, stdout=_json.dumps(policy),
                                    stderr="")

    monkeypatch.setattr(_sp, "run", fake_run)
    assert prereqs._sa_can_act_as_self(sa) is True


def test_sa_cannot_act_as_self_when_no_grant_anywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No resource-level self-binding AND no project-level actAs role → False."""
    import subprocess as _sp
    import json as _json
    sa = "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"

    def fake_run(argv, *_a, **_kw):
        # Neither policy grants the SA an actAs-conferring role.
        policy = {"bindings": [{"role": "roles/storage.objectUser",
                                "members": [f"serviceAccount:{sa}"]}]}
        return _sp.CompletedProcess(argv, 0, stdout=_json.dumps(policy),
                                    stderr="")

    monkeypatch.setattr(_sp, "run", fake_run)
    assert prereqs._sa_can_act_as_self(sa) is False


def test_check_orchestrator_calls_each_subcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    """`prereqs.check(args)` must invoke every individual check at least once."""
    invoked: set[str] = set()
    for name in (
        "check_python_version",
        "check_local_tools",
        "check_gcloud_config",
        "check_adc",
        "check_api_keys",
        "check_submitter_iam",
        "check_worker_sa_iam",
    ):
        def _stub(*_a, _name=name, **_kw):
            invoked.add(_name)

        monkeypatch.setattr(prereqs, name, _stub)
    # Also stub bucket + AR existence calls (they're not parameterless).
    monkeypatch.setattr(
        prereqs, "ensure_bucket_and_artifact_repo", lambda *_a, **_kw: invoked.add(
            "ensure_bucket_and_artifact_repo"
        )
    )

    class _Args:
        agent_model = "anthropic/claude-sonnet-4-5"
        user_sim_model = "anthropic/claude-haiku-4-5-20251001"

    prereqs.check(_Args())
    expected = {
        "check_python_version",
        "check_local_tools",
        "check_gcloud_config",
        "check_adc",
        "check_api_keys",
        "check_submitter_iam",
        "check_worker_sa_iam",
        "ensure_bucket_and_artifact_repo",
    }
    assert expected <= invoked


def test_no_gsutil_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex MINOR #17 — gsutil is NOT a prereq (SDK-only)."""
    monkeypatch.setattr(prereqs, "_which", lambda name: None if name == "gsutil" else "/x")
    # check_local_tools should succeed without gsutil on PATH.
    prereqs.check_local_tools()


# ---------------------------------------------------------------------------
# CR#9 — _list_submitter_roles is scoped to the active principal.
# ---------------------------------------------------------------------------


def test_list_submitter_roles_scopes_to_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lookup must filter by `user:$(gcloud config get-value account)`
    — not collect all bindings on the project."""
    import subprocess as _sp
    captured: list[list[str]] = []

    def fake_run(argv, *_args, **_kw):
        captured.append(list(argv))
        if argv[:3] == ["gcloud", "config", "get-value"]:
            return _sp.CompletedProcess(argv, 0, stdout="alice@motley.ai\n", stderr="")
        # The projects.get-iam-policy call must carry the filter.
        return _sp.CompletedProcess(
            argv, 0,
            stdout="roles/storage.admin\nroles/iam.serviceAccountUser\n",
            stderr="",
        )

    monkeypatch.setattr(_sp, "run", fake_run)
    roles = prereqs._list_submitter_roles()
    assert roles == {"roles/storage.admin", "roles/iam.serviceAccountUser"}
    # The gcloud projects call must carry the `user:alice@motley.ai` filter.
    iam_calls = [c for c in captured if "projects" in c and "get-iam-policy" in c]
    assert iam_calls, "no `gcloud projects get-iam-policy` invocation"
    flat = " ".join(iam_calls[-1])
    assert "user:alice@motley.ai" in flat


def test_list_submitter_roles_empty_without_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereqs, "_active_principal", lambda: "")
    assert prereqs._list_submitter_roles() == set()


# ---------------------------------------------------------------------------
# CR#10 — _list_worker_sa_roles is implemented (not a stub).
# ---------------------------------------------------------------------------


def test_list_worker_sa_roles_union_project_bucket_and_ar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_list_worker_sa_roles` must union role bindings across project,
    bucket, AND Artifact Registry repo scopes — each returned as JSON
    (not `--filter`'d, since that flag is inconsistently supported
    across gcloud subcommands)."""
    import subprocess as _sp
    import json as _json
    sa = "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"
    member = f"serviceAccount:{sa}"

    def _policy_json(role: str) -> str:
        return _json.dumps({
            "bindings": [{"role": role, "members": [member, "user:noise@x.io"]}]
        })

    def fake_run(argv, *_args, **_kw):
        if argv[:3] == ["gcloud", "projects", "get-iam-policy"]:
            return _sp.CompletedProcess(
                argv, 0, stdout=_policy_json("roles/compute.instanceAdmin.v1"),
                stderr="",
            )
        if argv[:4] == ["gcloud", "storage", "buckets", "get-iam-policy"]:
            return _sp.CompletedProcess(
                argv, 0, stdout=_policy_json("roles/storage.objectUser"),
                stderr="",
            )
        if argv[:5] == ["gcloud", "artifacts", "repositories", "get-iam-policy",
                        prereqs.AR_REPO]:
            return _sp.CompletedProcess(
                argv, 0, stdout=_policy_json("roles/artifactregistry.reader"),
                stderr="",
            )
        return _sp.CompletedProcess(argv, 1, stdout="", stderr="unexpected call")

    monkeypatch.setattr(_sp, "run", fake_run)
    roles = prereqs._list_worker_sa_roles(sa)
    assert roles == {
        "roles/compute.instanceAdmin.v1",
        "roles/storage.objectUser",
        "roles/artifactregistry.reader",
    }


def test_list_worker_sa_roles_ignores_other_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Role bindings that don't list the SA must not bleed into the
    result. Regression for the gcloud-side `--filter` mismatch."""
    import subprocess as _sp
    import json as _json
    sa = "bird-interact-runner@motley-team-475011.iam.gserviceaccount.com"

    def fake_run(argv, *_args, **_kw):
        # Policy contains an irrelevant role bound to a different member.
        policy = _json.dumps({
            "bindings": [
                {"role": "roles/owner", "members": ["user:someone@else.com"]},
                {
                    "role": "roles/storage.objectUser",
                    "members": [f"serviceAccount:{sa}"],
                },
            ]
        })
        return _sp.CompletedProcess(argv, 0, stdout=policy, stderr="")

    monkeypatch.setattr(_sp, "run", fake_run)
    roles = prereqs._list_worker_sa_roles(sa)
    assert "roles/owner" not in roles
    assert "roles/storage.objectUser" in roles


def test_list_worker_sa_roles_real_impl_no_longer_stub() -> None:
    """Regression guard for CR#10: the function must actually do
    real lookups, not return a hardcoded empty set. (The gcloud call now
    lives in the shared `_roles_for_member` helper, which itself runs
    subprocess.run.)"""
    import inspect
    src = inspect.getsource(prereqs._list_worker_sa_roles)
    assert "_roles_for_member" in src or "subprocess.run" in src
    assert "subprocess.run" in inspect.getsource(prereqs._roles_for_member)
    assert "return set()" not in src.split("\n", 1)[1].strip().split("\n")[0]


# ---------------------------------------------------------------------------
# DEV-1517 — OAuth token path for claude_sdk* frameworks.
# ---------------------------------------------------------------------------

_GOOD_TOKEN = "sk-ant-oat01-good-token"
_BAD_TOKEN = "sk-bad-prefix-token"
_ANTHROPIC_KEY = "sk-ant-api-key"


def test_claude_sdk_oauth_anthropic_usersim_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """claude_sdk + OAuth present + anthropic user-sim but no ANTHROPIC_API_KEY
    → fail: user-sim still needs the API key to be shipped as
    BIRD_INTERACT_LITELLM_ANTHROPIC_API_KEY."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_api_keys(
            agent_model="anthropic/claude-sonnet-4-5",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
        )
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_claude_sdk_oauth_plus_api_key_anthropic_usersim_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """claude_sdk + valid OAuth + ANTHROPIC_API_KEY + anthropic user-sim → pass."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    # Must not raise.
    prereqs.check_api_keys(
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
    )


def test_claude_sdk_oauth_plus_api_key_openai_usersim_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """claude_sdk + OAuth + openai user-sim → pass; ANTHROPIC_API_KEY not
    required because neither agent nor user-sim is anthropic on this path."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Must not raise.
    prereqs.check_api_keys(
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="openai/gpt-4o",
        framework="claude_sdk",
    )


def test_claude_sdk_registry_agent_skips_oauth_when_subscription_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEV-1602 (Codex finding #1): a registry open-weight agent model must take
    the provider-key path even with no_subscription_auth=False — the OAuth
    branch is gated on the agent model NOT being a registry model. No
    CLAUDE_CODE_OAUTH_TOKEN is required here."""
    # A valid OAuth token is ALSO present — the registry-first gate must ignore
    # it (the provider key path wins) rather than enter the OAuth branch.
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)  # for the user-sim
    # Must NOT raise about CLAUDE_CODE_OAUTH_TOKEN.
    prereqs.check_api_keys(
        agent_model="moonshot/kimi-k2.7-code",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
    )


def test_claude_sdk_registry_agent_still_requires_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEV-1602: the registry-first gate must not skip BOTH OAuth and
    provider-key validation. A registry agent missing its provider key still
    fails (so the implementation can't pass by merely short-circuiting OAuth)."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    with pytest.raises(prereqs.PrereqError, match="MOONSHOT_API_KEY"):
        prereqs.check_api_keys(
            agent_model="moonshot/kimi-k2.7-code",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
        )


def test_non_anthropic_non_registry_subscription_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CodeRabbit: --subscription-auth is Anthropic-only. A claude_sdk*
    subscription run on a non-Anthropic, non-registry model (openai/*) is
    rejected rather than requiring an OAuth token."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    with pytest.raises(prereqs.PrereqError, match="Anthropic-only"):
        prereqs.check_api_keys(
            agent_model="openai/gpt-4o",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
            no_subscription_auth=False,
        )


@pytest.mark.parametrize("no_sub", [False, True])
@pytest.mark.parametrize("bad_model", ["moonshot/kimi-k2.7-code", "openai/gpt-4o"])
def test_annotator_rejects_non_anthropic_model(
    monkeypatch: pytest.MonkeyPatch, no_sub: bool, bad_model: str,
) -> None:
    """DEV-1602 (Codex): the Anthropic-only `annotator` rejects ANY non-Anthropic
    agent model EARLY — registry (moonshot/*) or other (openai/*), for both
    --subscription-auth and --no-subscription-auth — rather than reaching the
    OAuth or provider-key branch and failing late."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("MOONSHOT_API_KEY", "ms-key-1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    with pytest.raises(prereqs.PrereqError, match="Anthropic-only"):
        prereqs.check_api_keys(
            agent_model=bad_model,
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            framework="annotator",
            no_subscription_auth=no_sub,
        )


def test_claude_sdk_oauth_slayer_still_requires_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """claude_sdk + OAuth + slayer mode: OPENAI_API_KEY is still required for
    channel-3 embeddings even though ANTHROPIC_API_KEY is not shipped."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_api_keys(
            agent_model="anthropic/claude-sonnet-4-5",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
            query_mode="slayer",
        )
    assert "OPENAI_API_KEY" in str(exc.value)


def test_claude_sdk_no_oauth_raises_when_subscription_auth_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEV-1535: claude_sdk with the default subscription auth path
    (no_subscription_auth=False) but no OAuth token → PrereqError. The
    silent-fall-through-to-legacy gap is gone; explicit opt-out
    (no_subscription_auth=True) is required to use the API-key path."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    with pytest.raises(prereqs.PrereqError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        prereqs.check_api_keys(
            agent_model="anthropic/claude-sonnet-4-5",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
        )


def test_claude_sdk_no_oauth_legacy_path_when_opted_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """claude_sdk + no_subscription_auth=True → legacy API-key path,
    no OAuth token required."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    # Must not raise.
    prereqs.check_api_keys(
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
        no_subscription_auth=True,
    )


def test_pydantic_ai_ignores_oauth_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pydantic_ai framework with OAuth token set locally → legacy path;
    ANTHROPIC_API_KEY is still required."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    # Must not raise (legacy path, OAuth silently ignored).
    prereqs.check_api_keys(
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        framework="pydantic_ai",
    )


def test_pydantic_ai_without_oauth_still_requires_anthropic_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pydantic_ai + OAuth set + no ANTHROPIC_API_KEY → fail (legacy path
    requires the key; OAuth is not honoured for non-claude_sdk frameworks)."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_api_keys(
            agent_model="anthropic/claude-sonnet-4-5",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            framework="pydantic_ai",
        )
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_claude_sdk_oauth_bad_prefix_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OAuth token without sk-ant-oat01- prefix → PrereqError with
    `claude setup-token` remediation."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _BAD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_api_keys(
            agent_model="anthropic/claude-sonnet-4-5",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
        )
    assert "claude setup-token" in exc.value.remediation


def test_check_orchestrator_passes_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    """`prereqs.check(args)` must forward `args.framework` to `check_api_keys`."""
    captured: dict = {}

    def _stub_check_api_keys(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(prereqs, "check_api_keys", _stub_check_api_keys)
    # Stub out everything else check() calls.
    for name in (
        "check_python_version", "check_local_tools", "check_gcloud_config",
        "check_adc", "check_submitter_iam", "check_worker_sa_iam",
        "ensure_bucket_and_artifact_repo",
    ):
        monkeypatch.setattr(prereqs, name, lambda *_a, **_kw: None)

    class _Args:
        agent_model = "anthropic/claude-sonnet-4-5"
        user_sim_model = "anthropic/claude-haiku-4-5-20251001"
        framework = "claude_sdk_otf"
        query_mode = "raw"

    prereqs.check(_Args())
    assert captured.get("framework") == "claude_sdk_otf"


# ---------------------------------------------------------------------------
# DEV-1530 — --no-subscription-auth flag forces legacy API-key path.
# ---------------------------------------------------------------------------


def test_claude_sdk_no_subscription_auth_flag_falls_through_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """claude_sdk + valid OAuth + no_subscription_auth=True + no ANTHROPIC_API_KEY
    → falls through to legacy path and fails missing ANTHROPIC_API_KEY."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_api_keys(
            agent_model="anthropic/claude-sonnet-4-5",
            user_sim_model="anthropic/claude-haiku-4-5-20251001",
            framework="claude_sdk",
            no_subscription_auth=True,
        )
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_claude_sdk_no_subscription_auth_flag_with_api_key_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """claude_sdk + valid OAuth + no_subscription_auth=True + ANTHROPIC_API_KEY
    → legacy path passes."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    prereqs.check_api_keys(
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
        no_subscription_auth=True,
    )


def test_annotator_no_subscription_auth_flag_falls_through_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """annotator + valid OAuth + no_subscription_auth=True + no ANTHROPIC_API_KEY
    → falls through to legacy path and fails missing ANTHROPIC_API_KEY.
    (On the OAuth path this would pass because annotator has no user-sim;
    the legacy path requires ANTHROPIC_API_KEY for the agent itself.)"""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(prereqs.PrereqError) as exc:
        prereqs.check_api_keys(
            agent_model="anthropic/claude-opus-4-7",
            user_sim_model="",
            framework="annotator",
            no_subscription_auth=True,
        )
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_no_subscription_auth_flag_noop_on_pydantic_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pydantic_ai + no_subscription_auth=True → already legacy; flag is a no-op."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _GOOD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    prereqs.check_api_keys(
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        framework="pydantic_ai",
        no_subscription_auth=True,
    )


def test_no_subscription_auth_flag_skips_bad_token_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """claude_sdk + bad OAuth token prefix + no_subscription_auth=True
    → no PrereqError from token validation (token is ignored on legacy path)."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", _BAD_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _ANTHROPIC_KEY)
    prereqs.check_api_keys(
        agent_model="anthropic/claude-sonnet-4-5",
        user_sim_model="anthropic/claude-haiku-4-5-20251001",
        framework="claude_sdk",
        no_subscription_auth=True,
    )


def test_check_orchestrator_forwards_no_subscription_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`prereqs.check(args)` must forward `args.no_subscription_auth` to
    `check_api_keys`."""
    captured: dict = {}

    def _stub_check_api_keys(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(prereqs, "check_api_keys", _stub_check_api_keys)
    for name in (
        "check_python_version", "check_local_tools", "check_gcloud_config",
        "check_adc", "check_submitter_iam", "check_worker_sa_iam",
        "ensure_bucket_and_artifact_repo",
    ):
        monkeypatch.setattr(prereqs, name, lambda *_a, **_kw: None)

    class _Args:
        agent_model = "anthropic/claude-sonnet-4-5"
        user_sim_model = "anthropic/claude-haiku-4-5-20251001"
        framework = "claude_sdk"
        query_mode = "raw"
        no_subscription_auth = True

    prereqs.check(_Args())
    assert captured.get("no_subscription_auth") is True


def test_check_orchestrator_no_subscription_auth_defaults_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`prereqs.check(args)` uses False when args has no no_subscription_auth."""
    captured: dict = {}

    def _stub_check_api_keys(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(prereqs, "check_api_keys", _stub_check_api_keys)
    for name in (
        "check_python_version", "check_local_tools", "check_gcloud_config",
        "check_adc", "check_submitter_iam", "check_worker_sa_iam",
        "ensure_bucket_and_artifact_repo",
    ):
        monkeypatch.setattr(prereqs, name, lambda *_a, **_kw: None)

    class _Args:
        agent_model = "anthropic/claude-sonnet-4-5"
        user_sim_model = "anthropic/claude-haiku-4-5-20251001"
        framework = "claude_sdk"
        query_mode = "raw"
        # no no_subscription_auth attribute

    prereqs.check(_Args())
    assert captured.get("no_subscription_auth") is False
