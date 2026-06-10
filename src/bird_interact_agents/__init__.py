"""bird-interact-agents package.

Side-effect import: registering the DEV-1545 user-sim prompt v3 into
the upstream `USER_SIMULATOR_ENCODER` / `USER_SIMULATOR_DECODER`
dicts. The package top-level is the right hook (vs a single agent
module) because distributed worker processes import via many entry
paths — `bird_interact_agents.run`, `.cloud.ray_app`, individual agent
modules — and we need v3 visible on all of them before any
`build_user_*_prompt(..., user_sim_prompt_version="v3")` call.

The import is wrapped in `try/except ModuleNotFoundError` because the
upstream `src.envs.user_simulator.prompts` lives in the optional
`mini-interact-agent` package (extras: `original` / `all`). A bare
`cloud` install — e.g. `pip install bird-interact-agents[cloud]` —
does NOT pull it in, and an unconditional import would crash any
`from bird_interact_agents import paths` (and similar) on those
installs. When the upstream is missing, v3 simply won't be registered;
any actual user-sim invocation site (`_submit.py`) imports from the
upstream itself and will surface a clearer error there.

See `user_sim_prompts.py` for the prompt body, registration semantics,
and upstream-collision handling.
"""

try:
    from bird_interact_agents import user_sim_prompts  # noqa: F401  side-effect
except ModuleNotFoundError:
    # Upstream `mini-interact-agent` not installed (e.g. bare `cloud` extra
    # without `original` / `all`). v3 prompt is unavailable in that
    # environment; any user-sim call path will hit a clearer error in
    # `_submit.py` on its own upstream import.
    pass
