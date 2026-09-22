"""BIRD-INTERACT-1.0 submission report generator (DEV-1553).

Converts an existing cloud run (or per-instance selection across runs)
into the JSONL + manifest layout that the leaderboard at
``bird.bench25@gmail.com`` expects. Reconstructs ``prompt_flow`` from
existing ``runs/<bench>/<db>/<id>/<run-id>.trajectory.json`` and
``results/<bench>/cloud/<run-id>/results.db`` — no harness-runtime
changes required.

a-Interact custom-agent setting only. The Section VI Universal Cost
Scheme is THE contract: fixed ``ask=2 / submit=3 / execute=1``;
everything else token-aware via the ``in<250 AND out<1000`` rule.
"""
