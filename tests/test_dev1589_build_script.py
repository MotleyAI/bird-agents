"""DEV-1589: `scripts/build_otf_references.py` gains
`--encoder-framework {claude_sdk, pydantic_ai}` defaulting to `claude_sdk`.

For `claude_sdk` it must pass the RAW registry model string (e.g. `zai/glm-5.2`)
to `make_claude_sdk_build_encoder` — NOT `build_pydantic_ai_model` (which can't
reach z.ai). For `pydantic_ai` it keeps the existing pydantic path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_otf_references.py"
_spec = importlib.util.spec_from_file_location("build_otf_references", SCRIPT)
build_otf_references = importlib.util.module_from_spec(_spec)
sys.modules["build_otf_references"] = build_otf_references
_spec.loader.exec_module(build_otf_references)


def test_encoder_framework_defaults_to_claude_sdk():
    parser = build_otf_references.build_arg_parser()
    args = parser.parse_args(["mini-interact", "--agent-model", "zai/glm-5.2"])
    assert args.encoder_framework == "claude_sdk"


def test_encoder_framework_choices():
    parser = build_otf_references.build_arg_parser()
    args = parser.parse_args(
        ["mini-interact", "--agent-model", "anthropic/claude-opus-4-7",
         "--encoder-framework", "pydantic_ai"],
    )
    assert args.encoder_framework == "pydantic_ai"
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["mini-interact", "--agent-model", "x",
             "--encoder-framework", "bogus"],
        )


def test_claude_sdk_path_passes_raw_model_string(monkeypatch):
    captured = {}

    def _fake_claude(*, model, self_model_id, **kw):
        captured["model"] = model
        captured["self_model_id"] = self_model_id
        return "CLAUDE_ENCODER"

    def _fail_pydantic(**kw):
        raise AssertionError("must NOT use the pydantic encoder for claude_sdk")

    monkeypatch.setattr(
        build_otf_references, "make_claude_sdk_build_encoder", _fake_claude,
    )
    monkeypatch.setattr(
        build_otf_references, "make_setup_build_encoder", _fail_pydantic,
    )

    enc = build_otf_references.make_build_encoder("claude_sdk", "zai/glm-5.2")
    assert enc == "CLAUDE_ENCODER"
    # RAW registry string, NOT a built pydantic-ai model object.
    assert captured["model"] == "zai/glm-5.2"


def test_pydantic_path_uses_setup_encoder(monkeypatch):
    captured = {}

    def _fake_pydantic(**kw):
        captured["called"] = True
        return "PYDANTIC_ENCODER"

    def _fail_claude(**kw):
        raise AssertionError("must NOT use the claude_sdk encoder for pydantic_ai")

    monkeypatch.setattr(
        build_otf_references, "make_setup_build_encoder", _fake_pydantic,
    )
    monkeypatch.setattr(
        build_otf_references, "make_claude_sdk_build_encoder", _fail_claude,
    )

    enc = build_otf_references.make_build_encoder(
        "pydantic_ai", "anthropic/claude-opus-4-7",
    )
    assert enc == "PYDANTIC_ENCODER"
    assert captured.get("called")
