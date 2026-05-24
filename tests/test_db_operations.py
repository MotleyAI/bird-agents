"""Verify SQLite DB operations through the harness."""

from bird_interact_agents.config import settings


def test_load_db_metadata():
    """Loading metadata for the alien DB populates the harness caches."""
    from bird_interact_agents.harness import (
        load_db_data_if_needed,
        _schema_cache,
        _column_meanings_cache,
        _external_knowledge_cache,
    )

    load_db_data_if_needed("alien", settings.db_path)

    assert "alien" in _schema_cache
    assert len(_schema_cache["alien"]) > 100
    assert "alien" in _column_meanings_cache
    assert "alien" in _external_knowledge_cache


def test_get_schema_action():
    """get_schema() returns CREATE TABLE statements."""
    from bird_interact_agents.harness import (
        execute_env_action,
        load_db_data_if_needed,
        SampleStatus,
    )

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    load_db_data_if_needed("alien", settings.db_path)
    status = SampleStatus(idx=0, original_data=task_data)

    obs, success = execute_env_action("get_schema()", status, settings.db_path)
    assert success
    assert "CREATE TABLE" in obs


def test_execute_simple_sql():
    """execute('SELECT 1') runs and returns a result."""
    from bird_interact_agents.harness import (
        execute_env_action,
        load_db_data_if_needed,
        SampleStatus,
    )

    task_data = {
        "selected_database": "alien",
        "knowledge_ambiguity": [],
        "instance_id": "alien_1",
    }
    load_db_data_if_needed("alien", settings.db_path)
    status = SampleStatus(idx=0, original_data=task_data)

    obs, success = execute_env_action("execute('SELECT 1')", status, settings.db_path)
    assert success
    assert "1" in obs
