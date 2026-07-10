"""DEV-1666: slayer-only ``lean_introspection`` + ``readonly_mode`` flags.

Mechanical contracts only (no prompt-substring / anchor-phrase asserts, per the
project's no-prompt-content-tests rule). The flags drop redundant read/write
SLayer tools from the v0 + v1 SLayer QUERY agents and are recorded per-run.

Covered:
* the shared drop-set helper (`_slayer_tool_surface`) — pure functions;
* False/False identity pin against the real current tool-surface constants;
* lean strips ``inspect_model`` + 3 KB natives; readonly strips the write tools;
* ``query`` / ``submit_query`` survive the KB drop; per-client (v1) scoping;
* run-record + manifest stamping + the resolve-recorded helper (bools only for a
  flag-consuming slayer framework, None for raw AND exempt slayer frameworks);
* CLI parsing (`--no-lean` / `--readonly-mode`) on run.py and cloud/cli.py;
* the in-scope agent constructors accept + store the two flags;
* prompt builders: structural identity pin (False/False == legacy constant),
  the flag has an effect, and no stray str.format placeholders are introduced.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from bird_interact_agents.agents import _slayer_tool_surface as tsurf
from bird_interact_agents.agents._pre_encoded import (
    WRITE_SLAYER_TOOLS,
    strip_write_slayer_tools,
)

# Bare names the flags target.
_INSPECT_MODEL = "inspect_model"
_KB_NATIVES = frozenset({
    "get_all_external_knowledge_names",
    "get_knowledge_definition",
    "get_all_knowledge_definitions",
})
_WRITE_TOOLS = frozenset({"create_model", "edit_model", "save_memory", "validate_models"})


# --------------------------------------------------------------------------- #
# 1. shared drop-set helper (pure)
# --------------------------------------------------------------------------- #
class TestDropSet:
    def test_write_set_matches_pre_encoded(self):
        # Single source of truth: readonly reuses the existing write set.
        assert tsurf.READONLY_DROP == WRITE_SLAYER_TOOLS == _WRITE_TOOLS

    def test_lean_constants(self):
        # list_datasources folded into lean (dead weight; v1 lacks it → no-op there).
        assert tsurf.LEAN_DROP_SLAYER_MCP == frozenset({_INSPECT_MODEL, "list_datasources"})
        assert tsurf.LEAN_DROP_NATIVE_KB == _KB_NATIVES

    def test_drops_false_false_empty(self):
        assert tsurf.slayer_flag_drops(
            lean_introspection=False, readonly_mode=False
        ) == frozenset()

    def test_drops_lean_only(self):
        assert tsurf.slayer_flag_drops(
            lean_introspection=True, readonly_mode=False
        ) == (frozenset({_INSPECT_MODEL, "list_datasources"}) | _KB_NATIVES)

    def test_drops_readonly_only(self):
        assert tsurf.slayer_flag_drops(
            lean_introspection=False, readonly_mode=True
        ) == _WRITE_TOOLS

    def test_drops_both(self):
        assert tsurf.slayer_flag_drops(
            lean_introspection=True, readonly_mode=True
        ) == (frozenset({_INSPECT_MODEL, "list_datasources"}) | _KB_NATIVES | _WRITE_TOOLS)


class TestFilterFlagDrops:
    def test_false_false_is_identity(self):
        names = ["search", "inspect", "inspect_model", "create_model", "query"]
        assert tsurf.filter_flag_drops(
            names, lean_introspection=False, readonly_mode=False
        ) == names

    def test_order_preserving_and_idempotent(self):
        names = ["search", "inspect_model", "inspect", "get_knowledge_definition", "query"]
        out = tsurf.filter_flag_drops(names, lean_introspection=True, readonly_mode=False)
        assert out == ["search", "inspect", "query"]
        # idempotent
        assert tsurf.filter_flag_drops(
            out, lean_introspection=True, readonly_mode=False
        ) == out

    def test_prefix_agnostic_bare_suffix_match(self):
        # v1 uses full mcp__bird-interact-tools__ names; the filter must match
        # on the bare suffix (mirrors _pre_encoded.strip_write_tool_names).
        full = [
            "mcp__bird-interact-tools__search",
            "mcp__bird-interact-tools__inspect_model",
            "mcp__bird-interact-tools__create_model",
            "mcp__bird-interact-tools__query",
        ]
        out = tsurf.filter_flag_drops(full, lean_introspection=True, readonly_mode=True)
        assert out == [
            "mcp__bird-interact-tools__search",
            "mcp__bird-interact-tools__query",
        ]

    def test_query_and_submit_query_survive_kb_drop(self):
        # Codex #4: query/submit_query live alongside the KB natives; the KB
        # drop must never take them.
        names = [*_KB_NATIVES, "query", "submit_query"]
        out = tsurf.filter_flag_drops(names, lean_introspection=True, readonly_mode=False)
        assert out == ["query", "submit_query"]

    def test_composes_with_strip_write(self):
        names = ["create_model", "edit_model", "search", "query"]
        stripped = strip_write_slayer_tools(names)  # existing pre-encoded strip
        out = tsurf.filter_flag_drops(
            stripped, lean_introspection=False, readonly_mode=True
        )
        assert out == ["search", "query"]


# --------------------------------------------------------------------------- #
# 2. identity pin against the REAL current tool-surface constants
# --------------------------------------------------------------------------- #
class TestV0ToolSurfaceIdentity:
    def _slayer_allow(self):
        from bird_interact_agents.agents.claude_sdk_otf.agent import SLAYER_MCP_TOOLS
        return SLAYER_MCP_TOOLS

    def test_false_false_slayer_allow_unchanged(self):
        allow = self._slayer_allow()
        assert tsurf.filter_flag_drops(
            allow, lean_introspection=False, readonly_mode=False
        ) == list(allow)

    def test_lean_removes_inspect_model_and_list_datasources_from_allow(self):
        allow = self._slayer_allow()
        out = tsurf.filter_flag_drops(allow, lean_introspection=True, readonly_mode=False)
        assert _INSPECT_MODEL not in out
        assert "list_datasources" not in out
        for keep in ("search", "inspect", "recommend_root_model", "help"):
            assert keep in out

    def test_readonly_removes_write_tools_from_allow(self):
        allow = self._slayer_allow()
        out = tsurf.filter_flag_drops(allow, lean_introspection=False, readonly_mode=True)
        assert not (_WRITE_TOOLS & set(out))
        assert "inspect_model" in out  # readonly does not touch reads

    def test_lean_removes_kb_natives_from_select_tools(self):
        from bird_interact_agents.agents.claude_sdk_otf.agent import _select_tools
        native_names = [t.name for t in _select_tools("one-shot")]
        out = tsurf.filter_flag_drops(
            native_names, lean_introspection=True, readonly_mode=False
        )
        assert not (_KB_NATIVES & set(out))
        assert "query" in out and "submit_query" in out


class TestV1ToolSurfaceIdentity:
    def _bare(self):
        from bird_interact_agents.agents.claude_sdk_otf_v1.agent import (
            MAIN_NATIVE_TOOL_NAMES,
            DISCOVERY_NATIVE_TOOL_NAMES,
        )
        return MAIN_NATIVE_TOOL_NAMES, DISCOVERY_NATIVE_TOOL_NAMES

    def test_false_false_main_discovery_unchanged(self):
        main, disc = self._bare()
        assert tsurf.filter_flag_drops(
            main, lean_introspection=False, readonly_mode=False
        ) == list(main)
        assert tsurf.filter_flag_drops(
            disc, lean_introspection=False, readonly_mode=False
        ) == list(disc)

    def test_lean_strips_inspect_model_and_kb_from_both_clients(self):
        main, disc = self._bare()
        m = tsurf.filter_flag_drops(main, lean_introspection=True, readonly_mode=False)
        d = tsurf.filter_flag_drops(disc, lean_introspection=True, readonly_mode=False)
        for surface in (m, d):
            bare = {n.split("__")[-1] for n in surface}
            assert _INSPECT_MODEL not in bare
            assert not (_KB_NATIVES & bare)
        # per-client scoping (Codex #5): MAIN keeps recommend_root_model/help;
        # DISCOVERY keeps models_summary. search/inspect on both.
        m_bare = {n.split("__")[-1] for n in m}
        d_bare = {n.split("__")[-1] for n in d}
        assert {"search", "inspect", "recommend_root_model", "help"} <= m_bare
        assert {"search", "inspect", "models_summary"} <= d_bare

    def test_readonly_strips_writes_from_main_only(self):
        main, disc = self._bare()
        m = tsurf.filter_flag_drops(main, lean_introspection=False, readonly_mode=True)
        assert not (_WRITE_TOOLS & {n.split("__")[-1] for n in m})
        # discovery holds no write tools; readonly is a no-op there
        assert tsurf.filter_flag_drops(
            disc, lean_introspection=False, readonly_mode=True
        ) == list(disc)


# --------------------------------------------------------------------------- #
# 3. flag-consuming set + record resolution
# --------------------------------------------------------------------------- #
class TestFlagsApplyAndResolve:
    def test_consuming_set(self):
        assert tsurf.FLAG_CONSUMING_FRAMEWORKS == frozenset({
            "claude_sdk", "claude_sdk_v1",
            "claude_sdk_otf", "claude_sdk_otf_ainteract",
            "claude_sdk_otf_v1", "claude_sdk_otf_ainteract_v1",
        })

    @pytest.mark.parametrize("fw", sorted({
        "claude_sdk", "claude_sdk_v1", "claude_sdk_otf",
        "claude_sdk_otf_ainteract", "claude_sdk_otf_v1", "claude_sdk_otf_ainteract_v1",
    }))
    def test_applies_for_consuming_slayer(self, fw):
        assert tsurf.flags_apply(framework=fw, query_mode="slayer") is True

    def test_not_applies_for_raw(self):
        assert tsurf.flags_apply(framework="claude_sdk", query_mode="raw") is False

    @pytest.mark.parametrize("fw", ["claude_sdk_otf_encode", "pydantic_ai", "oracle"])
    def test_not_applies_for_exempt_slayer(self, fw):
        assert tsurf.flags_apply(framework=fw, query_mode="slayer") is False

    def test_resolve_records_bools_for_consuming_slayer(self):
        assert tsurf.resolve_recorded_flags(
            framework="claude_sdk_otf", query_mode="slayer",
            lean_introspection=True, readonly_mode=False,
        ) == (True, False)
        assert tsurf.resolve_recorded_flags(
            framework="claude_sdk_otf_v1", query_mode="slayer",
            lean_introspection=False, readonly_mode=True,
        ) == (False, True)

    def test_resolve_none_on_raw(self):
        assert tsurf.resolve_recorded_flags(
            framework="claude_sdk", query_mode="raw",
            lean_introspection=True, readonly_mode=False,
        ) == (None, None)

    def test_resolve_none_on_exempt_slayer(self):
        # Codex #6 (narrowed): encode is slayer but exempt -> None/None.
        assert tsurf.resolve_recorded_flags(
            framework="claude_sdk_otf_encode", query_mode="slayer",
            lean_introspection=True, readonly_mode=False,
        ) == (None, None)


# --------------------------------------------------------------------------- #
# 4. run-record schema + manifest stamping
# --------------------------------------------------------------------------- #
class TestSubmissionConfig:
    def test_fields_default_none(self):
        from bird_interact_agents.eval.annotation_schema import SubmissionConfig
        cfg = SubmissionConfig()
        assert cfg.lean_introspection is None
        assert cfg.readonly_mode is None

    def test_roundtrip(self):
        from bird_interact_agents.eval.annotation_schema import SubmissionConfig
        cfg = SubmissionConfig(lean_introspection=True, readonly_mode=False)
        again = SubmissionConfig.model_validate(cfg.model_dump())
        assert again.lean_introspection is True
        assert again.readonly_mode is False


class TestManifest:
    def _args(self, **over):
        base = dict(
            instance_ids=["households_1"],
            framework="claude_sdk_otf", mode="one-shot", dataset="mini-interact",
            query_mode="slayer", agent_model="anthropic/claude-opus-4-8",
            user_sim_model="anthropic/claude-sonnet-4-6", patience=3, strict=False,
            use_audited_gold_sql=True, max_depth=3, prompt_cache=True,
            workers=1, actors_per_worker=1, worker_type="e2-standard-4",
            max_runtime_hours=1, lean_introspection=True, readonly_mode=False,
        )
        base.update(over)
        return SimpleNamespace(**base)

    def test_manifest_records_flags(self, monkeypatch):
        from bird_interact_agents.cloud import driver
        monkeypatch.setattr(driver, "_submit_benchmark", lambda a: "mini-interact")
        man = driver.build_manifest(
            self._args(), image_uri="img", run_id="rid",
        )
        assert man["lean_introspection"] is True
        assert man["readonly_mode"] is False

    def test_manifest_none_on_raw(self, monkeypatch):
        from bird_interact_agents.cloud import driver
        monkeypatch.setattr(driver, "_submit_benchmark", lambda a: "mini-interact")
        man = driver.build_manifest(
            self._args(query_mode="raw", framework="claude_sdk_otf_raw"),
            image_uri="img", run_id="rid",
        )
        assert man["lean_introspection"] is None
        assert man["readonly_mode"] is None

    def test_manifest_none_on_exempt_slayer(self, monkeypatch):
        # Codex #6/#7: encode is slayer but exempt -> recorded None/None.
        from bird_interact_agents.cloud import driver
        monkeypatch.setattr(driver, "_submit_benchmark", lambda a: "mini-interact")
        man = driver.build_manifest(
            self._args(framework="claude_sdk_otf_encode"),
            image_uri="img", run_id="rid",
        )
        assert man["lean_introspection"] is None
        assert man["readonly_mode"] is None


# --------------------------------------------------------------------------- #
# 5. CLI parsing (run.py + cloud/cli.py)
# --------------------------------------------------------------------------- #
class TestCliRun:
    # run.py has several required=True args unrelated to this feature, so we
    # introspect the parser's actions instead of parsing a full arg vector.
    def _actions(self):
        from bird_interact_agents import run as run_mod
        parser = run_mod.build_arg_parser()  # extracted from main()
        return {a.dest: a for a in parser._actions}

    def test_lean_action_store_false_default_true(self):
        a = self._actions()["lean_introspection"]
        assert a.default is True
        assert "--no-lean" in a.option_strings
        # Egor: no `--no-lean-introspection` / `--lean-introspection` pair.
        assert "--no-lean-introspection" not in a.option_strings
        assert "--lean-introspection" not in a.option_strings

    def test_readonly_action_store_true_default_false(self):
        a = self._actions()["readonly_mode"]
        assert a.default is False
        assert "--readonly-mode" in a.option_strings
        assert "--no-readonly-mode" not in a.option_strings


class TestCliCloud:
    def _submit(self, extra, monkeypatch):
        from bird_interact_agents.cloud import cli as cloud_cli
        # The submit parser requires an OAuth token in the env for --subscription-auth.
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test-token")
        base = [
            "submit", "--framework", "claude_sdk", "--query-mode", "slayer",
            "--agent-model", "anthropic/claude-opus-4-8", "--subscription-auth",
            "--dataset", "mini-interact",
            "--instance-ids", "households_1",
        ]
        return cloud_cli.parse_args(base + extra)

    def test_defaults(self, monkeypatch):
        ns = self._submit([], monkeypatch)
        assert ns.lean_introspection is True
        assert ns.readonly_mode is False

    def test_flags(self, monkeypatch):
        ns = self._submit(["--no-lean", "--readonly-mode"], monkeypatch)
        assert ns.lean_introspection is False
        assert ns.readonly_mode is True

    def test_no_disallowed_alias_pairs(self, monkeypatch):
        # Mirror TestCliRun: no `--no-lean-introspection` / `--no-readonly-mode`.
        for bad in ("--no-lean-introspection", "--no-readonly-mode", "--lean-introspection"):
            with pytest.raises(SystemExit):
                self._submit([bad], monkeypatch)


# --------------------------------------------------------------------------- #
# 6. in-scope agent constructors accept + store the flags
# --------------------------------------------------------------------------- #
class TestAgentConstructors:
    @pytest.mark.parametrize("modpath", [
        "bird_interact_agents.agents.claude_sdk_otf.agent",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract.agent",
        "bird_interact_agents.agents.claude_sdk_otf_v1.agent",
        "bird_interact_agents.agents.claude_sdk_otf_ainteract_v1.agent",
    ])
    def test_accepts_and_stores(self, modpath):
        import importlib
        mod = importlib.import_module(modpath)
        Agent = mod.ClaudeSDKOtfAgent if hasattr(mod, "ClaudeSDKOtfAgent") else None
        if Agent is None:
            # a-interact variants use their own class name
            Agent = next(
                getattr(mod, n) for n in dir(mod)
                if n.startswith("ClaudeSDKOtf") and n.endswith("Agent")
            )
        a = Agent(lean_introspection=False, readonly_mode=True)
        assert a.lean_introspection is False
        assert a.readonly_mode is True

    def test_defaults_lean_on_readonly_off(self):
        from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent
        a = ClaudeSDKOtfAgent()
        assert a.lean_introspection is True
        assert a.readonly_mode is False


# --------------------------------------------------------------------------- #
# 7. prompt builders: structural identity pin + flag-has-effect + no stray fields
# --------------------------------------------------------------------------- #
class TestPromptBuilders:
    def test_main_workflow_note_false_is_legacy(self):
        from bird_interact_agents.agents.claude_sdk import partition
        assert partition.build_main_workflow_note(
            query_mode="slayer", lean_introspection=False
        ) == partition.MAIN_WORKFLOW_NOTE

    def test_main_workflow_note_lean_has_effect(self):
        from bird_interact_agents.agents.claude_sdk import partition
        off = partition.build_main_workflow_note(query_mode="slayer", lean_introspection=False)
        on = partition.build_main_workflow_note(query_mode="slayer", lean_introspection=True)
        assert on != off

    def test_main_workflow_note_raw_ignores_lean(self):
        # raw discovery_section has no slayer tools -> lean is inert for raw.
        from bird_interact_agents.agents.claude_sdk import partition
        assert partition.build_main_workflow_note(
            query_mode="raw", lean_introspection=True
        ) == partition.build_main_workflow_note(
            query_mode="raw", lean_introspection=False
        )

    def test_discovery_prompt_false_is_current(self):
        from bird_interact_agents.agents.claude_sdk import partition
        base = partition.build_discovery_prompt(
            with_ask_user=True, query_mode="slayer", lean_introspection=False
        )
        on = partition.build_discovery_prompt(
            with_ask_user=True, query_mode="slayer", lean_introspection=True
        )
        assert on != base

    def test_static_template_constants_unchanged_under_false_false(self):
        # The exported legacy constants MUST equal builder(False, False) so the
        # existing SHA-256 snapshot tests (test_shared_otf_prompts) keep pinning
        # the byte-for-byte False/False text.
        from bird_interact_agents.agents import _shared_otf_prompts as sp
        assert sp.build_slayer_otf_one_shot_v0(
            lean_introspection=False, readonly_mode=False
        ) == sp.SLAYER_OTF_ONE_SHOT_V0

    def test_lean_builder_has_effect_on_static_template(self):
        from bird_interact_agents.agents import _shared_otf_prompts as sp
        off = sp.build_slayer_otf_one_shot_v0(lean_introspection=False, readonly_mode=False)
        on = sp.build_slayer_otf_one_shot_v0(lean_introspection=True, readonly_mode=False)
        assert on != off

    def test_readonly_builder_has_effect_on_static_template(self):
        from bird_interact_agents.agents import _shared_otf_prompts as sp
        off = sp.build_slayer_otf_one_shot_v0(lean_introspection=False, readonly_mode=False)
        ro = sp.build_slayer_otf_one_shot_v0(lean_introspection=False, readonly_mode=True)
        assert ro != off

    def test_builders_introduce_no_stray_format_fields(self):
        # Mechanical contract (allowed): the gated variants must carry the same
        # str.format placeholder set as the False/False text — a lean/readonly
        # swap must not add or drop a runtime {field}.
        import string
        from bird_interact_agents.agents import _shared_otf_prompts as sp

        def fields(s):
            return {f for _, f, _, _ in string.Formatter().parse(s) if f}

        base = fields(sp.build_slayer_otf_one_shot_v0(
            lean_introspection=False, readonly_mode=False))
        for lean in (True, False):
            for ro in (True, False):
                assert fields(sp.build_slayer_otf_one_shot_v0(
                    lean_introspection=lean, readonly_mode=ro)) == base


# --------------------------------------------------------------------------- #
# 7b. remaining gated prompt builders (Codex #6 coverage) — mechanical only
# --------------------------------------------------------------------------- #
def _fmt_fields(s):
    import string
    return {f for _, f, _, _ in string.Formatter().parse(s) if f}


class TestMorePromptBuilders:
    """Every gated template: builder(False,False) == exported legacy constant,
    the flag has an effect, and no stray str.format fields are introduced."""

    # (module attr with the builder, legacy-constant attr, gates that matter)
    # gates = which flags currently change the builder's output. The pre-encoded
    # builders are pass-through today (prompt-text gating tracked as follow-up),
    # so only their False/False == legacy identity is pinned.
    _CASES = [
        ("_shared_otf_prompts", "build_slayer_otf_ainteract_v0",
         "SLAYER_OTF_AINTERACT_V0", ("lean", "readonly")),
        ("_pre_encoded_prompts", "build_slayer_pre_encoded_one_shot",
         "SLAYER_PRE_ENCODED_ONE_SHOT", ()),
        ("_pre_encoded_prompts", "build_slayer_pre_encoded_one_shot_v1",
         "SLAYER_PRE_ENCODED_ONE_SHOT_V1", ()),
    ]

    @pytest.mark.parametrize("mod,builder,legacy,gates", _CASES)
    def test_false_false_is_legacy(self, mod, builder, legacy, gates):
        import importlib
        m = importlib.import_module(f"bird_interact_agents.agents.{mod}")
        out = getattr(m, builder)(lean_introspection=False, readonly_mode=False)
        assert out == getattr(m, legacy)

    @pytest.mark.parametrize("mod,builder,legacy,gates", _CASES)
    def test_flag_has_effect(self, mod, builder, legacy, gates):
        import importlib
        m = importlib.import_module(f"bird_interact_agents.agents.{mod}")
        base = getattr(m, builder)(lean_introspection=False, readonly_mode=False)
        if "lean" in gates:
            assert getattr(m, builder)(
                lean_introspection=True, readonly_mode=False) != base
        if "readonly" in gates:
            assert getattr(m, builder)(
                lean_introspection=False, readonly_mode=True) != base

    @pytest.mark.parametrize("mod,builder,legacy,gates", _CASES)
    def test_no_stray_format_fields(self, mod, builder, legacy, gates):
        import importlib
        m = importlib.import_module(f"bird_interact_agents.agents.{mod}")
        base = _fmt_fields(getattr(m, builder)(
            lean_introspection=False, readonly_mode=False))
        for lean in (True, False):
            for ro in (True, False):
                assert _fmt_fields(getattr(m, builder)(
                    lean_introspection=lean, readonly_mode=ro)) == base


class TestLeanReadonlyAbsence:
    """Egor's contract: the dropped tool names simply DO NOT OCCUR in the gated
    variants (the whole point of authoring them statically)."""

    def test_lean_tools_inventory_tail_has_no_inspect_model(self):
        # The gated SLAYER TOOLS inventory tail drops inspect_model under lean.
        # (Other shared disciplines embedded in the full one-shot template still
        # mention inspect_model — gating those is tracked as a follow-up; the
        # tool is dropped from the surface regardless.)
        from bird_interact_agents.agents import _shared_otf_prompts as sp
        for ro in (True, False):
            tail = sp._slayer_tools_tail(lean_introspection=True, readonly_mode=ro)
            assert "inspect_model" not in tail

    def test_readonly_tools_tail_has_no_write_tools(self):
        # The gated SLAYER TOOLS inventory tail under readonly drops the
        # create_model / edit_model build mention (block-level contract).
        from bird_interact_agents.agents import _shared_otf_prompts as sp
        tail = sp._slayer_tools_tail(lean_introspection=False, readonly_mode=True)
        assert "create_model" not in tail
        assert "edit_model" not in tail

    def test_lean_main_workflow_note_has_no_dropped_read_tools(self):
        from bird_interact_agents.agents.claude_sdk import partition
        lean = partition.build_main_workflow_note(
            query_mode="slayer", lean_introspection=True)
        assert "inspect_model" not in lean
        for kb in ("get_knowledge_definition", "get_all_external_knowledge_names"):
            assert kb not in lean

    def test_lean_discovery_prompt_has_no_inspect_model(self):
        from bird_interact_agents.agents.claude_sdk import partition
        lean = partition.build_discovery_prompt(
            with_ask_user=False, query_mode="slayer", lean_introspection=True)
        assert "inspect_model" not in lean


class TestDiscoveryCompactNote:
    def test_gated_via_build_discovery_prompt(self):
        # _DISCOVERY_COMPACT_NOTE is threaded through build_discovery_prompt;
        # lean must change the slayer discovery prompt, raw must be inert.
        from bird_interact_agents.agents.claude_sdk import partition
        off = partition.build_discovery_prompt(
            with_ask_user=False, query_mode="slayer", lean_introspection=False)
        on = partition.build_discovery_prompt(
            with_ask_user=False, query_mode="slayer", lean_introspection=True)
        assert on != off
        # raw discovery has no search/inspect -> lean inert
        assert partition.build_discovery_prompt(
            with_ask_user=False, query_mode="raw", lean_introspection=True
        ) == partition.build_discovery_prompt(
            with_ask_user=False, query_mode="raw", lean_introspection=False)


# --------------------------------------------------------------------------- #
# 8. tool-surface at the ACTUAL agent build sites (Codex #3 seam)
# --------------------------------------------------------------------------- #
class TestV0BuildSiteSurface:
    def test_effective_slayer_allow_applies_flags(self):
        from bird_interact_agents.agents.claude_sdk_otf.agent import (
            SLAYER_MCP_TOOLS, effective_slayer_allow,
        )
        # False/False identity
        assert effective_slayer_allow(
            lean_introspection=False, readonly_mode=False, pre_encoded_source=None
        ) == list(SLAYER_MCP_TOOLS)
        # lean drops inspect_model + list_datasources
        lean = effective_slayer_allow(
            lean_introspection=True, readonly_mode=False, pre_encoded_source=None)
        assert "inspect_model" not in lean and "list_datasources" not in lean
        # readonly drops writes
        ro = effective_slayer_allow(
            lean_introspection=False, readonly_mode=True, pre_encoded_source=None)
        assert not (_WRITE_TOOLS & set(ro))

    def test_effective_native_tools_drops_kb_under_lean(self):
        from bird_interact_agents.agents.claude_sdk_otf.agent import (
            effective_native_tools,
        )
        names = {t.name for t in effective_native_tools(
            "one-shot", lean_introspection=True)}
        assert not (_KB_NATIVES & names)
        assert "query" in names and "submit_query" in names


class TestV1BuildSiteSurface:
    def test_effective_main_and_discovery_apply_flags(self):
        from bird_interact_agents.agents.claude_sdk_otf_v1.agent import (
            MAIN_NATIVE_TOOL_NAMES, DISCOVERY_NATIVE_TOOL_NAMES,
            effective_main_tools, effective_discovery_tools,
        )
        assert effective_main_tools(
            lean_introspection=False, readonly_mode=False, pre_encoded_source=None
        ) == list(MAIN_NATIVE_TOOL_NAMES)
        assert effective_discovery_tools(lean_introspection=False) == list(
            DISCOVERY_NATIVE_TOOL_NAMES)
        m = {n.split("__")[-1] for n in effective_main_tools(
            lean_introspection=True, readonly_mode=True, pre_encoded_source=None)}
        assert "inspect_model" not in m
        assert not (_KB_NATIVES & m) and not (_WRITE_TOOLS & m)
        d = {n.split("__")[-1] for n in effective_discovery_tools(
            lean_introspection=True)}
        assert "inspect_model" not in d and not (_KB_NATIVES & d)


# --------------------------------------------------------------------------- #
# 9. local runner wiring + warnings (Codex #2 / #4 / #5)
# --------------------------------------------------------------------------- #
class TestMainForwardsFlags:
    """Regression (Codex): main() must forward args.lean_introspection /
    args.readonly_mode into the run_evaluation call — else local runs silently
    ignore --no-lean / --readonly-mode."""

    def test_run_evaluation_call_passes_flags(self):
        import ast
        import inspect
        from bird_interact_agents import run as run_mod

        tree = ast.parse(inspect.getsource(run_mod.main))
        forwarded = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "run_evaluation"):
                forwarded = {kw.arg for kw in node.keywords}
        assert "lean_introspection" in forwarded
        assert "readonly_mode" in forwarded


class TestRunnerWiring:
    """make_runner must pass the flags to in-scope query agents and NOT to
    raw/exempt agents. We patch the agent classes to capture kwargs without
    running a task."""

    def _capture(self, monkeypatch, modpath, clsname):
        import importlib
        mod = importlib.import_module(modpath)
        captured = {}

        class _Stub:
            def __init__(self, *a, **kw):
                captured.update(kw)

            async def run_task(self, *a, **kw):
                return {}
        monkeypatch.setattr(mod, clsname, _Stub)
        return captured

    def test_v0_slayer_receives_flags(self, monkeypatch):
        from bird_interact_agents import run as run_mod
        cap = self._capture(
            monkeypatch,
            "bird_interact_agents.agents.claude_sdk_otf", "ClaudeSDKOtfAgent")
        run_mod.make_runner(
            framework="claude_sdk_otf", dataset="mini-interact",
            query_mode="slayer", mode="one-shot",
            agent_model="anthropic/claude-opus-4-8",
            strict=False, prompt_cache=True, max_depth=3, slayer_storage_root=None,
            lean_introspection=False, readonly_mode=True,
        )
        assert cap.get("lean_introspection") is False
        assert cap.get("readonly_mode") is True

    def test_raw_agent_does_not_receive_flags(self, monkeypatch):
        from bird_interact_agents import run as run_mod
        cap = self._capture(
            monkeypatch,
            "bird_interact_agents.agents.claude_sdk_otf_raw", "ClaudeSDKOtfRawAgent")
        run_mod.make_runner(
            framework="claude_sdk_otf_raw", dataset="mini-interact",
            query_mode="raw", mode="one-shot",
            agent_model="anthropic/claude-opus-4-8",
            strict=False, prompt_cache=True, max_depth=3, slayer_storage_root=None,
            lean_introspection=False, readonly_mode=True,
        )
        assert "lean_introspection" not in cap
        assert "readonly_mode" not in cap


class TestIgnoredFlagWarning:
    def test_warns_on_raw_deviation(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            tsurf.maybe_warn_ignored_flags(
                framework="claude_sdk", query_mode="raw",
                lean_introspection=False, readonly_mode=False)
        assert caplog.records, "expected a warning for --no-lean under raw"

    def test_warns_on_exempt_slayer_deviation(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            tsurf.maybe_warn_ignored_flags(
                framework="claude_sdk_otf_encode", query_mode="slayer",
                lean_introspection=True, readonly_mode=True)
        assert caplog.records

    def test_silent_on_consuming_slayer(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            tsurf.maybe_warn_ignored_flags(
                framework="claude_sdk_otf", query_mode="slayer",
                lean_introspection=False, readonly_mode=True)
        assert not caplog.records

    def test_silent_on_defaults_under_raw(self, caplog):
        # defaults (lean True, readonly False) are NOT an explicit deviation.
        import logging
        with caplog.at_level(logging.WARNING):
            tsurf.maybe_warn_ignored_flags(
                framework="claude_sdk", query_mode="raw",
                lean_introspection=True, readonly_mode=False)
        assert not caplog.records


class TestReadonlyOnTheFlyWarning:
    def test_warns_when_readonly_and_on_the_fly(self, caplog):
        import logging
        from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent
        with caplog.at_level(logging.WARNING):
            ClaudeSDKOtfAgent(readonly_mode=True)  # on-the-fly (no pre_encoded_source)
        assert any("readonly" in r.message.lower() for r in caplog.records)

    def test_no_warn_when_readonly_and_pre_encoded(self, caplog):
        import logging
        from bird_interact_agents.agents.claude_sdk_otf.agent import ClaudeSDKOtfAgent
        with caplog.at_level(logging.WARNING):
            ClaudeSDKOtfAgent(
                readonly_mode=True, pre_encoded_source="otf",
                slayer_setup="pre-encoded")
        assert not any("readonly" in r.message.lower() for r in caplog.records)


# --------------------------------------------------------------------------- #
# 10. cloud plumbing (Codex #1/#2)
# --------------------------------------------------------------------------- #
class TestCloudJobArgs:
    def _args(self, **over):
        base = dict(
            query_mode="slayer", framework="claude_sdk", mode="one-shot",
            agent_model="m", user_sim_model="u", instance_ids=["households_1"],
            benchmark="mini-interact", patience=3, strict=False,
            use_audited_gold_sql=True, prompt_cache=True, max_depth=3,
            reasoning_effort=None, user_sim_prompt_version=None,
            slayer_setup="on-the-fly", pre_encoded_source=None,
            subscription_auth=None, cache_ttl="5m", workers=1, actors_per_worker=1,
            lean_introspection=True, readonly_mode=False,
        )
        base.update(over)
        return SimpleNamespace(**base)

    @staticmethod
    def _patch(monkeypatch):
        from bird_interact_agents.cloud import driver
        monkeypatch.setattr(driver, "_submit_benchmark", lambda a: "mini-interact")
        monkeypatch.setattr(
            driver, "_instance_ids_sorted_by_db", lambda ids, bench: list(ids))
        return driver

    def test_emits_on_deviation(self, monkeypatch):
        driver = self._patch(monkeypatch)
        job = driver._build_job_args(
            self._args(lean_introspection=False, readonly_mode=True),
            "rid", attempt=1, benchmark_data_prefix=None,
        )
        assert "--no-lean" in job
        assert "--readonly-mode" in job

    def test_omits_on_default(self, monkeypatch):
        driver = self._patch(monkeypatch)
        job = driver._build_job_args(
            self._args(), "rid", attempt=1, benchmark_data_prefix=None,
        )
        assert "--no-lean" not in job
        assert "--readonly-mode" not in job


class TestCloudResubmitArgs:
    def test_reemits_from_manifest(self, monkeypatch):
        from bird_interact_agents.cloud import driver
        monkeypatch.setattr(
            driver, "_instance_ids_sorted_by_db", lambda ids, bench: list(ids))
        manifest = {
            "query_mode": "slayer", "mode": "one-shot", "framework": "claude_sdk",
            "agent_model": "m", "user_sim_model": "u",
            "lean_introspection": False, "readonly_mode": True,
            "render_inputs": {"workers": 1, "actors_per_worker": 1},
        }
        args = driver._build_resubmit_args(manifest, "rid", ["households_1"], 2)
        assert "--no-lean" in args
        assert "--readonly-mode" in args


class TestRayAppArgparse:
    def test_main_accepts_flags_into_run_pool(self, monkeypatch):
        from bird_interact_agents.cloud import ray_app
        captured = {}
        monkeypatch.setattr(ray_app, "run_pool", lambda **kw: captured.update(kw) or 0)
        monkeypatch.setattr(ray_app, "_load_task_data", lambda *a, **k: {})
        monkeypatch.setattr(ray_app, "download_benchmark_data", lambda *a, **k: None)
        monkeypatch.setattr(ray_app, "_load_secrets_file", lambda *a, **k: {})
        ray_app.main([
            "--run-id", "r", "--attempt", "1", "--framework", "claude_sdk",
            "--query-mode", "slayer", "--mode", "one-shot",
            "--agent-model", "m", "--user-sim-model", "u",
            "--dataset", "mini-interact", "--instance-ids", "households_1",
            "--no-lean", "--readonly-mode",
        ])
        assert captured.get("lean_introspection") is False
        assert captured.get("readonly_mode") is True
