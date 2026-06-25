"""DEV-1605: ``scripts/build_otf_references.py`` gains ``--version`` (default =
``encoder_version_slug(--agent-model)``) and builds into the versioned root
``slayer_models_otf/<benchmark>/<version>/<db>/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_otf_references.py"
_spec = importlib.util.spec_from_file_location("build_otf_references", SCRIPT)
build_otf_references = importlib.util.module_from_spec(_spec)
sys.modules["build_otf_references"] = build_otf_references
_spec.loader.exec_module(build_otf_references)


def test_version_flag_parses():
    parser = build_otf_references.build_arg_parser()
    args = parser.parse_args(
        ["mini-interact", "--agent-model", "anthropic/claude-opus-4-7",
         "--version", "my-label"]
    )
    assert args.version == "my-label"


def test_version_defaults_to_none_on_cli():
    parser = build_otf_references.build_arg_parser()
    args = parser.parse_args(
        ["mini-interact", "--agent-model", "anthropic/claude-opus-4-7"]
    )
    assert args.version is None


def test_resolve_build_version_defaults_to_slug():
    version, explicit = build_otf_references.resolve_build_version(
        agent_model="anthropic/claude-opus-4-7", version_arg=None,
    )
    assert version == "opus-4-7"
    assert explicit is False


def test_resolve_build_version_explicit_label():
    version, explicit = build_otf_references.resolve_build_version(
        agent_model="anthropic/claude-opus-4-7", version_arg="custom",
    )
    assert version == "custom"
    assert explicit is True


def test_resolve_build_version_empty_string_is_explicit():
    """An explicit `--version ""` (e.g. unset env expansion) is preserved as
    operator-supplied — NOT silently treated as omitted (CodeRabbit)."""
    version, explicit = build_otf_references.resolve_build_version(
        agent_model="anthropic/claude-opus-4-7", version_arg="",
    )
    assert version == ""
    assert explicit is True
