"""Lightweight timing logger for the on-the-fly cache + per-task setup path.

DEV-1561 attribution channel. The local `claude_sdk_otf_ainteract` startup
has been sitting silent for 5-11 minutes between process start and the
first agent log line even with the OTF cache fully warm; without a single
greppable timing tag, that gap is unattributable.

The module emits one structured log line per timed span under a stable
`[otf_timing]` prefix:

    [otf_timing] instance=alien_1 ensure_db_cache.done elapsed_s=0.003 db=alien path=fast

Greppable by prefix (`grep '\\[otf_timing\\]'`) and by logger name
(`bird_interact_agents.otf_timing`). All call sites use the same prefix +
field grammar so the gap can be reconstructed from a single tail of the
harness log.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("bird_interact_agents.otf_timing")
# Pin to INFO independently of root: upstream `batch_run_bird_interact`'s
# import-time `logging.basicConfig` call wins the no-op-after-handlers race
# against `run.py`'s INFO config, leaving root at WARNING. We want the
# attribution channel to fire regardless. Records still propagate to the
# root handler (default StreamHandler at NOTSET), so a single setLevel here
# is enough — the parent doesn't re-check level on propagation.
logger.setLevel(logging.INFO)

_PREFIX = "[otf_timing]"


def _fmt_fields(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in fields.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    return " ".join(parts)


def log_otf_event(
    label: str,
    *,
    instance_id: str | None = None,
    **fields: Any,
) -> None:
    """Emit an instantaneous timing event under the `[otf_timing]` prefix.

    Used for atomic events (cache hit, first message arrived) where the
    elapsed of the surrounding span already covers the interval.
    """
    head = f"instance={instance_id}" if instance_id else "instance=-"
    tail = _fmt_fields(fields)
    suffix = f" {tail}" if tail else ""
    logger.info("%s %s %s%s", _PREFIX, head, label, suffix)


@contextmanager
def otf_timer(
    label: str,
    *,
    instance_id: str | None = None,
    **fields: Any,
):
    """Context manager that logs elapsed seconds on exit.

    Logs `<label>.start` on entry and either `<label>.done elapsed_s=…` or
    `<label>.error elapsed_s=… exc=<type>` on exit. Works inside `async
    def` (no event-loop yield) so call sites can wrap a single `await`
    without introducing an async context-manager dance.
    """
    started = time.monotonic()
    log_otf_event(f"{label}.start", instance_id=instance_id, **fields)
    err: str | None = None
    try:
        yield
    except BaseException as e:  # noqa: BLE001 — re-raises below
        err = type(e).__name__
        raise
    finally:
        elapsed = time.monotonic() - started
        if err is None:
            log_otf_event(
                f"{label}.done",
                instance_id=instance_id,
                elapsed_s=f"{elapsed:.3f}",
                **fields,
            )
        else:
            log_otf_event(
                f"{label}.error",
                instance_id=instance_id,
                elapsed_s=f"{elapsed:.3f}",
                exc=err,
                **fields,
            )
