"""Thin wrapper around ``bird_interact_agents.eval.annotate`` — matches
the ``consolidate_mini_interact_audited.py`` / ``verify_audited_gold.py``
discoverability pattern.

Use directly::

    python scripts/generate_annotation_skeletons.py \\
        --run-id <id> --benchmark mini-interact
"""
from __future__ import annotations

from bird_interact_agents.eval.annotate import main

if __name__ == "__main__":
    raise SystemExit(main())
