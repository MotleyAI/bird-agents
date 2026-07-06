"""DEV-1638: the auth dotenv loader moved into the package.

``run_local_postgres.load_env_file`` is retired; the same parser now lives at
``bird_interact_agents.env_file.load_env_file`` so the installed
``bird-interact`` console script can load an auth dotenv (``--env-file``)
without importing an un-packaged ``scripts/`` sibling. These pin the parser
contract (KEY=VALUE / export KEY=VALUE, surrounding-quote stripping, full-line
comment + blank skipping, missing-file tolerance). Inline comments are NOT
stripped (matches the original parser).
"""

from __future__ import annotations

from pathlib import Path

from bird_interact_agents.env_file import load_env_file


def test_missing_file_returns_zero(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FOO_TOKEN", raising=False)
    n = load_env_file(tmp_path / "does_not_exist.env")
    assert n == 0
    import os
    assert "FOO_TOKEN" not in os.environ


def test_parses_plain_and_export_lines(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text(
        "# a comment\n"
        "\n"
        "A_KEY=plain-value\n"
        "export B_KEY=exported-value\n"
    )
    n = load_env_file(p)
    assert n == 2
    import os
    assert os.environ["A_KEY"] == "plain-value"
    assert os.environ["B_KEY"] == "exported-value"


def test_strips_surrounding_quotes(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("Q1", raising=False)
    monkeypatch.delenv("Q2", raising=False)
    p = tmp_path / ".env"
    p.write_text('Q1="double"\nQ2=\'single\'\n')
    load_env_file(p)
    import os
    assert os.environ["Q1"] == "double"
    assert os.environ["Q2"] == "single"


def test_skips_comment_blank_and_valueless_lines(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("REAL", raising=False)
    p = tmp_path / ".env"
    p.write_text("# comment\n\nNOEQUALS\nREAL=1\n")
    n = load_env_file(p)
    assert n == 1
    import os
    assert os.environ["REAL"] == "1"
    assert "NOEQUALS" not in os.environ
