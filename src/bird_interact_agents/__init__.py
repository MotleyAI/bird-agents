"""bird-interact-agents package.

Side-effect import: registering the DEV-1545 user-sim prompt v3 into
the upstream `USER_SIMULATOR_ENCODER` / `USER_SIMULATOR_DECODER`
dicts. The package top-level is the right hook (vs a single agent
module) because distributed worker processes import via many entry
paths — `bird_interact_agents.run`, `.cloud.ray_app`, individual agent
modules — and we need v3 visible on all of them before any
`build_user_*_prompt(..., user_sim_prompt_version="v3")` call.

See `user_sim_prompts.py` for the prompt body, registration semantics,
and upstream-collision handling.
"""

from bird_interact_agents import user_sim_prompts  # noqa: F401  side-effect
