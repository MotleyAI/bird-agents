"""Load auth credentials from a dotenv-style file into ``os.environ``.

Moved out of ``scripts/run_local_postgres.py`` (DEV-1638) so the installed
``bird-interact`` console script can load an auth dotenv (``--env-file``)
without importing an un-packaged ``scripts/`` sibling. The token loaded here is
the same ``CLAUDE_CODE_OAUTH_TOKEN`` / ``ANTHROPIC_API_KEY`` the cloud
``submit``/``annotate`` reads.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> int:
    """Merge ``KEY=VALUE`` / ``export KEY=VALUE`` lines into ``os.environ``.

    Returns the count of vars set. Silently returns 0 if the file is absent
    (a missing dotenv is not fatal — the shell may already carry the creds).
    Full-line comments (``# ...``) and blank lines are skipped; surrounding
    single/double quotes are stripped. Inline comments are NOT stripped.
    """
    if not path.is_file():
        return 0
    n = 0
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ[key] = val
            n += 1
    return n
