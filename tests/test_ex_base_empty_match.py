"""Verify the harness monkey-patches BIRD-Interact's `ex_base` so that
both sides legitimately returning 0 rows is a pass, not a fail.

The upstream behaviour treats `not predicted_res or not ground_res` as
failure unconditionally, which breaks any audited task whose gold
correctly returns the empty set (e.g. `households_15` — "highly
supported AND financially secure" yields no rows in this dataset).
"""

import sqlite3
import tempfile
from pathlib import Path


def _make_tiny_db() -> Path:
    """Create a one-table sqlite DB that lets us craft empty / matching /
    differing result sets via simple WHERE predicates."""
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    con = sqlite3.connect(f.name)
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
    con.executemany("INSERT INTO t (id, name) VALUES (?, ?)",
                    [(1, "alpha"), (2, "beta"), (3, "gamma")])
    con.commit()
    con.close()
    return Path(f.name)


def test_ex_base_returns_1_when_both_sides_empty():
    # Side-effect: importing harness applies the monkey-patch.
    import bird_interact_agents.harness  # noqa: F401
    from src.envs.bird_interact_env.test_case_utils_sqlite import test_utils as tu

    db_path = _make_tiny_db()
    conn = sqlite3.connect(db_path)
    try:
        # Both predicates filter to zero rows.
        result = tu.ex_base(
            ["SELECT id FROM t WHERE id = 999"],
            ["SELECT id FROM t WHERE name = 'never'"],
            str(db_path), conn,
        )
        assert result == 1, f"expected 1 (empty == empty match), got {result}"
    finally:
        conn.close()
        db_path.unlink(missing_ok=True)


def test_ex_base_returns_0_when_only_predicted_is_empty():
    import bird_interact_agents.harness  # noqa: F401
    from src.envs.bird_interact_env.test_case_utils_sqlite import test_utils as tu

    db_path = _make_tiny_db()
    conn = sqlite3.connect(db_path)
    try:
        # Predicted is empty; gold has rows. Must NOT pass.
        result = tu.ex_base(
            ["SELECT id FROM t WHERE id = 999"],
            ["SELECT id FROM t"],
            str(db_path), conn,
        )
        assert result == 0
    finally:
        conn.close()
        db_path.unlink(missing_ok=True)


def test_ex_base_returns_0_when_only_gold_is_empty():
    import bird_interact_agents.harness  # noqa: F401
    from src.envs.bird_interact_env.test_case_utils_sqlite import test_utils as tu

    db_path = _make_tiny_db()
    conn = sqlite3.connect(db_path)
    try:
        result = tu.ex_base(
            ["SELECT id FROM t"],
            ["SELECT id FROM t WHERE id = 999"],
            str(db_path), conn,
        )
        assert result == 0
    finally:
        conn.close()
        db_path.unlink(missing_ok=True)


def test_ex_base_returns_1_when_both_sides_match_with_rows():
    import bird_interact_agents.harness  # noqa: F401
    from src.envs.bird_interact_env.test_case_utils_sqlite import test_utils as tu

    db_path = _make_tiny_db()
    conn = sqlite3.connect(db_path)
    try:
        result = tu.ex_base(
            ["SELECT id FROM t WHERE id <= 2"],
            ["SELECT id FROM t WHERE name IN ('alpha', 'beta')"],
            str(db_path), conn,
        )
        assert result == 1
    finally:
        conn.close()
        db_path.unlink(missing_ok=True)


def test_ex_base_returns_0_when_row_contents_differ():
    import bird_interact_agents.harness  # noqa: F401
    from src.envs.bird_interact_env.test_case_utils_sqlite import test_utils as tu

    db_path = _make_tiny_db()
    conn = sqlite3.connect(db_path)
    try:
        result = tu.ex_base(
            ["SELECT id FROM t WHERE id = 1"],
            ["SELECT id FROM t WHERE id = 2"],
            str(db_path), conn,
        )
        assert result == 0
    finally:
        conn.close()
        db_path.unlink(missing_ok=True)
