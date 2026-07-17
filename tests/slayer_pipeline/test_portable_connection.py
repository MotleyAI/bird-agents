"""Tests for the portable SQLite ``connection_string`` helpers.

Round-trip: absolute paths under the mini-interact root convert to
the relative ``sqlite:///<rel>`` form, and that form resolves back
to an absolute path anchored at the supplied root (overridable via
``$BIRD_DB_PATH``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import sqlite3

from bird_interact_agents.slayer_pipeline.portable_connection import (
    absolute_sqlite_url,
    expected_connection_string,
    portabilise_postgres_connection_string,
    reanchor_connection_string,
    reanchor_postgres_connection_string,
    resolve_committed_connection_string,
    to_portable_connection_string,
)


# ---------------------------------------------------------------------------
# absolute_sqlite_url: env-independent canonical absolute URL from a path.
# This is the chokepoint the orchestrator uses so a RELATIVE --db-path no
# longer yields a broken `sqlite:////../...` URL.
# ---------------------------------------------------------------------------


def test_absolute_sqlite_url_from_absolute_path(tmp_path):
    p = tmp_path / "alien" / "alien.sqlite"
    want = f"sqlite:////{p.resolve().as_posix().lstrip('/')}"
    assert absolute_sqlite_url(p) == want


def test_absolute_sqlite_url_from_relative_path_resolves(tmp_path, monkeypatch):
    """A RELATIVE path must resolve to a canonical 4-slash ABSOLUTE URL —
    not `sqlite:////../...` which SQLAlchemy reads at filesystem root."""
    db_dir = tmp_path / "lsb" / "alien"
    db_dir.mkdir(parents=True)
    (db_dir / "alien.sqlite").touch()
    monkeypatch.chdir(tmp_path)
    rel = Path("lsb") / "alien" / "alien.sqlite"
    url = absolute_sqlite_url(rel)
    abs_p = (tmp_path / rel).resolve()
    assert url == f"sqlite:////{abs_p.as_posix().lstrip('/')}"
    assert "/../" not in url and not url.startswith("sqlite:////..")


def test_absolute_sqlite_url_opens_a_real_sqlite(tmp_path, monkeypatch):
    """The produced URL must actually open the DB through SQLAlchemy."""
    sa = pytest.importorskip("sqlalchemy")
    db_dir = tmp_path / "lsb" / "alien"
    db_dir.mkdir(parents=True)
    real = db_dir / "alien.sqlite"
    sqlite3.connect(str(real)).close()  # materialise a real sqlite file
    monkeypatch.chdir(tmp_path)
    url = absolute_sqlite_url(Path("lsb") / "alien" / "alien.sqlite")
    eng = sa.create_engine(url)
    with eng.connect() as conn:
        conn.exec_driver_sql("select 1")


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


def test_to_portable_rewrites_postgres_to_canonical():
    """DEV-1685 B1b: a persisted postgres datasource must be portabilised to
    canonical defaults so a committed OTF reference (and its fingerprint) is
    machine-independent — the runtime reanchor supplies the real connection.
    Non-postgres, non-sqlite noise (empty) is still returned unchanged."""
    assert (
        to_portable_connection_string(
            "postgresql://someuser@somehost:5544/alien", Path("/tmp")
        )
        == "postgresql://bird_interact@localhost:5432/alien"
    )
    assert to_portable_connection_string("", Path("/tmp")) == ""


# ---------------------------------------------------------------------------
# DEV-1685: postgres connection is RUNTIME-supplied, never cached. The
# persisted datasource is portabilised to canonical defaults; at task-prep it
# is reanchored to the live cluster from BIRD_PG_* (or defaults when unset —
# the cloud path, which does not forward BIRD_PG_* for postgres benchmarks).
# ---------------------------------------------------------------------------


def test_reanchor_postgres_rewrites_host_port_user_from_env(monkeypatch):
    monkeypatch.setenv("BIRD_PG_HOST", "127.0.0.1")
    monkeypatch.setenv("BIRD_PG_PORT", "5433")
    monkeypatch.setenv("BIRD_PG_USER", "bird_interact")
    out = reanchor_postgres_connection_string(
        "postgresql://staleuser@stalehost:5544/alien"
    )
    assert out == "postgresql://bird_interact@127.0.0.1:5433/alien"


def test_reanchor_postgres_defaults_when_env_unset(monkeypatch):
    """Cloud does NOT forward BIRD_PG_* for postgres benchmarks; the worker's
    bundled server lives at localhost:5432. So an unset env must reanchor a
    laptop-baked non-5432 URL to the canonical default, NOT leave it stale."""
    for v in ("BIRD_PG_HOST", "BIRD_PG_PORT", "BIRD_PG_USER"):
        monkeypatch.delenv(v, raising=False)
    out = reanchor_postgres_connection_string(
        "postgresql://someone@127.0.0.1:5544/households"
    )
    assert out == "postgresql://bird_interact@localhost:5432/households"


def test_reanchor_postgres_preserves_db_name_and_query(monkeypatch):
    monkeypatch.setenv("BIRD_PG_HOST", "localhost")
    monkeypatch.setenv("BIRD_PG_PORT", "5544")
    monkeypatch.setenv("BIRD_PG_USER", "bird_interact")
    out = reanchor_postgres_connection_string(
        "postgresql://u@h:1/solar_panel?sslmode=disable"
    )
    assert out == "postgresql://bird_interact@localhost:5544/solar_panel?sslmode=disable"


def test_reanchor_postgres_drops_password(monkeypatch):
    """The persisted URL is passwordless (password rides PGPASSWORD); a stray
    password in the input must not survive the rewrite into a YAML/argv."""
    monkeypatch.setenv("BIRD_PG_HOST", "localhost")
    monkeypatch.setenv("BIRD_PG_PORT", "5432")
    monkeypatch.setenv("BIRD_PG_USER", "bird_interact")
    out = reanchor_postgres_connection_string(
        "postgresql://u:secretpw@h:5544/alien"
    )
    assert "secretpw" not in out
    assert out == "postgresql://bird_interact@localhost:5432/alien"


@pytest.mark.parametrize("noise", ["", "sqlite:///x/x.sqlite", "yaml:///x"])
def test_reanchor_postgres_passthrough_for_non_postgres(noise):
    assert reanchor_postgres_connection_string(noise) == noise


def test_portabilise_postgres_to_canonical_defaults():
    assert (
        portabilise_postgres_connection_string(
            "postgresql://someuser@somehost:5544/alien?x=1"
        )
        == "postgresql://bird_interact@localhost:5432/alien?x=1"
    )


def test_portabilise_postgres_ignores_env(monkeypatch):
    """Portabilise is env-INDEPENDENT (canonical), unlike reanchor — the
    committed form must be identical on every machine."""
    monkeypatch.setenv("BIRD_PG_HOST", "somewhere")
    monkeypatch.setenv("BIRD_PG_PORT", "9999")
    monkeypatch.setenv("BIRD_PG_USER", "other")
    assert (
        portabilise_postgres_connection_string("postgresql://a@b:1/db")
        == "postgresql://bird_interact@localhost:5432/db"
    )


def test_reanchor_connection_string_dispatches_postgres(monkeypatch):
    """The shared choke point routes postgres URLs through the postgres
    reanchor (so prepare_task_storage / edited_models / reference_build all
    get it), while sqlite behaviour is untouched."""
    monkeypatch.setenv("BIRD_PG_HOST", "127.0.0.1")
    monkeypatch.setenv("BIRD_PG_PORT", "5433")
    monkeypatch.setenv("BIRD_PG_USER", "bird_interact")
    out = reanchor_connection_string(
        "postgresql://stale@stale:5544/alien", "alien", Path("/tmp"),
    )
    assert out == "postgresql://bird_interact@127.0.0.1:5433/alien"


def test_short_postgres_scheme_is_also_handled(monkeypatch):
    """The plan requires BOTH postgres:// and postgresql:// to dispatch. The
    scheme is preserved on rewrite."""
    monkeypatch.setenv("BIRD_PG_HOST", "127.0.0.1")
    monkeypatch.setenv("BIRD_PG_PORT", "5433")
    monkeypatch.setenv("BIRD_PG_USER", "bird_interact")
    # reanchor (env-driven) preserves the postgres:// scheme.
    assert reanchor_postgres_connection_string(
        "postgres://stale@stale:5544/alien"
    ) == "postgres://bird_interact@127.0.0.1:5433/alien"
    # portabilise (canonical) preserves the postgres:// scheme.
    assert portabilise_postgres_connection_string(
        "postgres://a@b:1/alien"
    ) == "postgres://bird_interact@localhost:5432/alien"
    # to_portable dispatches the short scheme too.
    assert to_portable_connection_string(
        "postgres://a@b:1/alien", Path("/tmp")
    ) == "postgres://bird_interact@localhost:5432/alien"
    # reanchor_connection_string dispatches the short scheme too.
    assert reanchor_connection_string(
        "postgres://a@b:1/alien", "alien", Path("/tmp"),
    ) == "postgres://bird_interact@127.0.0.1:5433/alien"


@pytest.mark.parametrize("scheme", ["postgresql", "postgres"])
def test_resolve_committed_leaves_postgres_untouched(scheme):
    """resolve_committed_connection_string is the sqlite RELATIVE-form resolver
    only; postgres (no relative form) still passes through unchanged after
    DEV-1685 — only reanchor_connection_string / to_portable dispatch it."""
    url = f"{scheme}://u@h:5544/alien"
    assert resolve_committed_connection_string(url, Path("/tmp")) == url


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


# ---------------------------------------------------------------------------
# DEV-1478 cloud bug: a deterministic cache built on one machine bakes in an
# ABSOLUTE local sqlite path. `resolve_committed_connection_string` passes
# absolute paths through UNCHANGED, so the foreign path survived into the
# cloud container and every agent query failed "unable to open database
# file". `expected_connection_string` / `reanchor_connection_string` FORCE
# the path to the current root, fixing the transported-cache case.
# ---------------------------------------------------------------------------


def test_expected_connection_string_uses_supplied_root(tmp_path, monkeypatch):
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    root = tmp_path / "mini-interact"
    expected = expected_connection_string("households", root)
    want = f"sqlite:////{(root / 'households' / 'households.sqlite').resolve().as_posix().lstrip('/')}"
    assert expected == want


def test_expected_connection_string_honors_env(tmp_path, monkeypatch):
    """``$BIRD_DB_PATH`` overrides the supplied root — this is what makes
    the cloud container (``BIRD_DB_PATH=/data/mini-interact``) resolve to
    the container path even if the supplied root were stale."""
    env_root = tmp_path / "data" / "mini-interact"
    monkeypatch.setenv("BIRD_DB_PATH", str(env_root))
    expected = expected_connection_string("credit", tmp_path / "elsewhere")
    want = f"sqlite:////{(env_root / 'credit' / 'credit.sqlite').resolve().as_posix().lstrip('/')}"
    assert expected == want


def test_reanchor_rewrites_stale_foreign_absolute_path(tmp_path, monkeypatch):
    """THE regression: a cache built under /home/<user>/... is transported
    to a container where the DB lives under the current root. The foreign
    absolute path must be force-rewritten, not passed through."""
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    container_root = tmp_path / "data" / "mini-interact"
    foreign = "sqlite:////home/someone/Dropbox/SLayer/mini-interact/households/households.sqlite"
    out = reanchor_connection_string(foreign, "households", container_root)
    want = f"sqlite:////{(container_root / 'households' / 'households.sqlite').resolve().as_posix().lstrip('/')}"
    assert out == want
    assert "/home/someone/" not in out, (
        "stale foreign absolute path must NOT survive re-anchoring"
    )


def test_reanchor_resolves_relative_form(tmp_path, monkeypatch):
    """The committed/pre-encoded relative form re-anchors to the same
    place — so the fix is a no-op for the pre-encoded path."""
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    root = tmp_path / "mini-interact"
    out = reanchor_connection_string(
        "sqlite:///households/households.sqlite", "households", root,
    )
    want = f"sqlite:////{(root / 'households' / 'households.sqlite').resolve().as_posix().lstrip('/')}"
    assert out == want


def test_reanchor_honors_env_over_supplied_root(tmp_path, monkeypatch):
    env_root = tmp_path / "data" / "mini-interact"
    monkeypatch.setenv("BIRD_DB_PATH", str(env_root))
    foreign = "sqlite:////home/someone/mini-interact/credit/credit.sqlite"
    out = reanchor_connection_string(foreign, "credit", tmp_path / "stale")
    want = f"sqlite:////{(env_root / 'credit' / 'credit.sqlite').resolve().as_posix().lstrip('/')}"
    assert out == want


# NB: "postgresql://…" is deliberately NOT in this list any more — DEV-1685
# routes postgres URLs through the postgres reanchor (see
# test_reanchor_connection_string_dispatches_postgres). Only genuinely
# unrecognised / empty forms pass through unchanged.
@pytest.mark.parametrize("noise", ["", "yaml:///x"])
def test_reanchor_passthrough_for_non_sqlite(noise, tmp_path):
    assert reanchor_connection_string(noise, "credit", tmp_path) == noise


def test_reanchor_normalises_five_slash_foreign_absolute(tmp_path, monkeypatch):
    """The cache historically emitted the malformed 5-slash form
    (``sqlite://///abs``). Re-anchoring must still produce a clean
    4-slash current-root path."""
    monkeypatch.delenv("BIRD_DB_PATH", raising=False)
    root = tmp_path / "mini-interact"
    foreign5 = "sqlite://///home/someone/mini-interact/households/households.sqlite"
    out = reanchor_connection_string(foreign5, "households", root)
    want = f"sqlite:////{(root / 'households' / 'households.sqlite').resolve().as_posix().lstrip('/')}"
    assert out == want
