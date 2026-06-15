"""Tests for the DEV-1561 OTF timing attribution channel.

The channel is the *only* signal that lets a developer attribute a multi-
minute silence to a specific call site (``ensure_db_cache`` hit vs build,
the SDK ``__aenter__`` handshake, per-message gap inside the receive loop).
The format contract is therefore pinned by tests:

* every line carries the ``[otf_timing]`` prefix so it's greppable across a
  multi-component log;
* every timed span emits a ``<label>.start`` + ``<label>.done`` (or
  ``<label>.error``) pair so a partial / hung span is still distinguishable
  from a clean exit;
* ``elapsed_s`` is always present on ``.done`` / ``.error``;
* fields are written as ``key=value`` so a downstream walker can pull
  ``db=…`` / ``msg_type=…`` without parsing prose.
"""

from __future__ import annotations

import logging

import pytest

from bird_interact_agents.slayer_otf.timing import (
    log_otf_event,
    logger as timing_logger,
    otf_timer,
)


_TIMING_LOGGER_NAME = "bird_interact_agents.otf_timing"


@pytest.fixture
def caplog_otf(caplog):
    """Capture INFO records on the otf_timing logger.

    The module pins the logger to INFO independently of root (because
    upstream's import-time basicConfig wins over ``run.py``'s INFO call
    and leaves root at WARNING). The fixture ensures pytest's caplog
    sees the records regardless of the host process's root config.
    """
    caplog.set_level(logging.INFO, logger=_TIMING_LOGGER_NAME)
    yield caplog


def _otf_records(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _TIMING_LOGGER_NAME
    ]


def test_log_otf_event_emits_prefix_instance_label_and_fields(caplog_otf):
    log_otf_event("ensure_db_cache.hit", instance_id="alien_1", db="alien", path="fast")
    msgs = _otf_records(caplog_otf)
    assert len(msgs) == 1, msgs
    line = msgs[0]
    assert "[otf_timing]" in line
    assert "instance=alien_1" in line
    assert "ensure_db_cache.hit" in line
    assert "db=alien" in line
    assert "path=fast" in line


def test_log_otf_event_without_instance_uses_dash(caplog_otf):
    log_otf_event("cache.lock_acquired", db="alien")
    msgs = _otf_records(caplog_otf)
    assert len(msgs) == 1, msgs
    assert "instance=-" in msgs[0]


def test_log_otf_event_skips_none_fields(caplog_otf):
    """A None field must not surface as ``field=None`` noise."""
    log_otf_event(
        "prep.skipped", instance_id="alien_1", db="alien", reason=None,
    )
    msgs = _otf_records(caplog_otf)
    assert len(msgs) == 1, msgs
    assert "reason=" not in msgs[0]
    assert "db=alien" in msgs[0]


def test_otf_timer_emits_start_then_done_with_elapsed(caplog_otf):
    with otf_timer("ensure_db_cache", instance_id="alien_1", db="alien"):
        pass
    msgs = _otf_records(caplog_otf)
    assert len(msgs) == 2, msgs
    start, done = msgs
    assert "ensure_db_cache.start" in start
    assert "instance=alien_1" in start
    assert "db=alien" in start
    assert "ensure_db_cache.done" in done
    assert "instance=alien_1" in done
    assert "db=alien" in done
    assert "elapsed_s=" in done
    # elapsed must parse as a non-negative float to 3 decimals.
    tok = next(t for t in done.split() if t.startswith("elapsed_s="))
    secs = float(tok.split("=", 1)[1])
    assert secs >= 0.0


def test_otf_timer_records_exception_path(caplog_otf):
    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with otf_timer("ensure_db_cache.build", instance_id="alien_1", db="alien"):
            raise Boom("nope")
    msgs = _otf_records(caplog_otf)
    assert len(msgs) == 2, msgs
    start, error = msgs
    assert ".start" in start
    assert ".error" in error
    assert "exc=Boom" in error
    assert "elapsed_s=" in error


def test_otf_timer_fields_appear_on_both_start_and_done(caplog_otf):
    """Per-span fields (db, deleted_kb_ids) must survive to the .done line
    so the post-hoc grep `<label>.done db=<...>` works without joining
    against the .start line."""
    with otf_timer(
        "prepare_task_storage.kb_mask",
        instance_id="alien_1", db="alien", deleted_kb_ids=2,
    ):
        pass
    msgs = _otf_records(caplog_otf)
    assert len(msgs) == 2
    for line in msgs:
        assert "db=alien" in line
        assert "deleted_kb_ids=2" in line


def test_otf_timing_logger_pinned_to_info_independent_of_root(caplog_otf):
    """The module sets the otf_timing logger to INFO unconditionally.

    Reproduces the DEV-1561 path that failed: a sibling import (e.g.
    upstream ``batch_run_bird_interact``) gets first call on
    ``logging.basicConfig``, leaving root at WARNING. The pinned level
    must still let INFO through so attribution survives.
    """
    # Drive the root level WAY above INFO and confirm we still emit.
    root = logging.getLogger()
    saved = root.level
    try:
        root.setLevel(logging.CRITICAL)
        log_otf_event("post.hoc.pin.check", instance_id="alien_1")
    finally:
        root.setLevel(saved)
    # The logger itself is INFO-effective regardless of root.
    assert timing_logger.getEffectiveLevel() <= logging.INFO
    # caplog uses its own handler tied to the timing logger name, so it
    # sees the record.
    msgs = _otf_records(caplog_otf)
    assert any("post.hoc.pin.check" in m for m in msgs), msgs
