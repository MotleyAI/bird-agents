"""Tests for the portable SQLite ``connection_string`` helpers.

Round-trip: absolute paths under the mini-interact root convert to
the relative ``sqlite:///<rel>`` form, and that form resolves back
to an absolute path anchored at the supplied root (overridable via
``$BIRD_DB_PATH``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bird_interact_agents.slayer_pipeline.portable_connection import (
    resolve_committed_connection_string,
    to_portable_connection_string,
)


def test_to_portable_strips_mini_interact_prefix(tmp_path):
    """Standard 4-slash absolute path rooted in mini-interact → relative."""
    root = tmp_path / "mini-interact"
    root.mkdir()
    (root / "households").mkdir()
    (root / "households" / "households.sqlite").touch()

    abs_uri = f"sqlite:////{(root / 'households' / 'households.sqlite').as_posix().lstrip('/')}"
    portable = to_portable_connection_string(abs_uri, root)
    assert portable == "sqlite:///households/households.sqlite"


def test_to_portable_normalises_five_slash_form(tmp_path):
    """The pipeline historically emitted ``sqlite://///abs`` (5 slashes);
    the helper should normalise that to the standard 4-slash absolute
    form before deciding whether the path is rooted in mini-interact."""
    root = tmp_path / "mini-interact"
    root.mkdir()
    (root / "households").mkdir()
    (root / "households" / "households.sqlite").touch()

    abs_uri = f"sqlite://///{(root / 'households' / 'households.sqlite').as_posix().lstrip('/')}"
    portable = to_portable_connection_string(abs_uri, root)
    assert portable == "sqlite:///households/households.sqlite"


def test_to_portable_leaves_outside_paths_alone(tmp_path):
    """Absolute paths NOT under mini-interact return unchanged — we don't
    want to silently corrupt connection strings pointing elsewhere."""
    root = tmp_path / "mini-interact"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "x.sqlite"

    abs_uri = f"sqlite:////{outside.as_posix().lstrip('/')}"
    assert to_portable_connection_string(abs_uri, root) == abs_uri


def test_to_portable_passthrough_for_non_sqlite():
    """Postgres / other backends are returned unchanged."""
    assert (
        to_portable_connection_string("postgresql://user@host/db", Path("/tmp"))
        == "postgresql://user@host/db"
    )
    assert to_portable_connection_string("", Path("/tmp")) == ""


def test_resolve_committed_uses_supplied_root(tmp_path, monkeypatch):
    """Relative committed form resolves against the supplied root when
    ``$BIRD_DB_PATH`` is not set."""
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    root = tmp_path / "mini-interact"
    root.mkdir()

    resolved = resolve_committed_connection_string(
        "sqlite:///households/households.sqlite", root
    )
    expected = (root / "households" / "households.sqlite").resolve()
    assert resolved == f"sqlite:////{expected.as_posix().lstrip('/')}"


def test_resolve_committed_honors_env(tmp_path, monkeypatch):
    """``$BIRD_DB_PATH`` overrides the supplied root."""
    env_root = tmp_path / "env-root"
    env_root.mkdir()
    monkeypatch.setenv("BIRD_DB_PATH", str(env_root))

    resolved = resolve_committed_connection_string(
        "sqlite:///solar/solar.sqlite", Path("/should/not/be/used")
    )
    expected = (env_root / "solar" / "solar.sqlite").resolve()
    assert resolved == f"sqlite:////{expected.as_posix().lstrip('/')}"


def test_resolve_committed_passes_absolute_through(tmp_path):
    """Already-absolute connection strings (legacy yamls that haven't
    been re-exported under the portable form) pass through unchanged."""
    abs_uri = "sqlite:////tmp/some/abs/path.sqlite"
    assert resolve_committed_connection_string(abs_uri, tmp_path) == abs_uri


@pytest.mark.parametrize("noise", ["", "postgresql://x", "yaml:///x"])
def test_resolve_committed_passthrough_for_non_sqlite(noise):
    assert resolve_committed_connection_string(noise, Path("/tmp")) == noise


def test_roundtrip(tmp_path, monkeypatch):
    """to_portable → resolve_committed should reproduce the absolute path
    when the supplied roots match."""
    root = tmp_path / "mini-interact"
    root.mkdir()
    (root / "households").mkdir()
    sqlite_path = root / "households" / "households.sqlite"
    sqlite_path.touch()
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)

    abs_uri = f"sqlite:////{sqlite_path.as_posix().lstrip('/')}"
    portable = to_portable_connection_string(abs_uri, root)
    resolved = resolve_committed_connection_string(portable, root)
    assert resolved == abs_uri
