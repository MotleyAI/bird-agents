"""DEV-1545: locally-defined v3 user-simulator prompt variant.

Why v3 exists
-------------
The failure-mode analysis on the latest mini-interact slayer-mode
runs (53 instances, all autopsies regenerated with opus-4-7) found
`user_sim_misleading` is the dominant residual pattern even after
DEV-1534's agent-side trust-calibration block landed (62% → 29%).
Inspection of the upstream v2 prompt
(``.venv/lib/python3.13/site-packages/src/envs/user_simulator/prompts.py``)
shows v2 has `[[GT_SQL]]` in scope but actively invites the simulator
to "help you to answer question when you not sure" — a phrase that
turns "ambiguous user" into "confidently-stated wrong answer".

v3 preserves v2's deliberate-vagueness framing (natural USER role-play,
not a precise oracle) but adds three rules that route the "I don't
exactly know" branch to `unanswerable()` instead of fabrication:

  1. Anti-invention (in BOTH encoder action-selection and decoder
     phrasing): when asked for an operational detail not pinned by
     labeled ambiguities and not directly grounded in `[[SQL_Glot]]`,
     respond `unanswerable()`. Do not invent.
  2. Evidence check (soft, not "predict SQL equivalence" — too
     cognitively expensive for haiku-class sims; Codex review pushed
     back on the harder verification framing).
  3. Naturalness reframe (decoder): a real user often shrugs.
     "I'm not sure" / "use your judgment" is MORE realistic than a
     confidently wrong answer.

How registration works
----------------------
The upstream module ships v2-keyed dicts
``USER_SIMULATOR_ENCODER`` / ``USER_SIMULATOR_DECODER`` keyed by
version string. The upstream prompt builders look up the requested
version at call time. We add our `"v3"` entry to those dicts at module
import (this file).

This file is imported as a side effect from
``src/bird_interact_agents/__init__.py`` — that guarantees registration
on any `import bird_interact_agents.<anything>` path (including
distributed worker processes per Codex review #4).

Per Codex review #3 + the explicit user decision, the registration is
NOT `setdefault` — instead the module ASSERTS the key is absent before
inserting. If upstream ever ships their own `"v3"` (or a future bump of
``batch_run_bird_interact``), the assertion fails loudly at import
time, surfacing the collision instead of silently keeping a stranger's
prompt under the same key. Resolution at that point is to rename ours
to ``"v3_motley"`` and bump CLI choices.

Per ``feedback_no_prompt_content_tests``, regression tests pin only
mechanical contracts (placeholder coverage, substitution presence) —
NOT phrase content. Behaviour validation goes through cloud smoke.
"""

from __future__ import annotations

# Side-effect import: this loads upstream's v2 entries first, then we
# mutate the same dict objects in place.
from src.envs.user_simulator.prompts import (
    USER_SIMULATOR_DECODER,
    USER_SIMULATOR_ENCODER,
)


_V3_ENCODER = """\
You are role-playing as a human USER interacting with an AI collaborator to \
complete a Text-to-SQL task. The AI collaborator may ask one question about \
this task. Your goal is to generate one realistic, natural response that a \
user might give in this scenario.

## Input Information:
You will be provided with:
- Task Description: The type of task you are trying to accomplish.
- Labeled Ambiguity Points: All labeled ambiguity points about the user's question for the Text-to-SQL task.
- Ground-truth SQL Segments: All ground-truth SQL segments.
- Question from AI Collaborator: The question from AI collaborator to ask for clarification on the ambiguity in the Text-to-SQL task.

Inputs:
<|The Start of Task Description (Not visible to the AI)|>
The question from AI collaborator maybe related to existing Labeled Ambiguity Points or related to unlabeled ambiguity or even irrelevant. So, you should choose one action at this turn.

Action Choices:
1. **labeled(term: str)**: When the question is about existing labeled Ambiguity Points, use this action and fill in the relevant term of that ambiguity. Format: **labeled("Amb")**.
2. **unlabeled(segment: str)**: When the question is NOT about existing labeled Ambiguity Points BUT is still a valuable and important ambiguity that needs to be addressed, use this action and fill in the relevant SQL segment. Format: **unlabeled("ALTER")**.
3. **unanswerable()**: When you think this question is neither related to labeled Ambiguity Points nor necessary to address, OR when the operational detail being asked about is NOT pinned by the labeled ambiguities AND NOT directly grounded in a Ground-truth SQL segment, use this action. Format: **unanswerable()**.
<|The End of Task Description|>

<|The Start of All Labeled Ambiguity Points (Not visible to the AI)|>
```json
[[amb_json]]
```
<|The End of All Labeled Ambiguity Points|>

<|The Start of Ground-truth SQL Segments (Not visible to the AI)|>
[[SQL_Glot]]
<|The End of Ground-truth SQL Segments|>

<|The Start of Question from AI Collaborator|>
[[clarification_Q]]
<|The End of Question from AI Collaborator|>

## Guidelines:
- You MUST choose only **one action** listed above.
- You should NOT tell any thoughts about solution nor any ground-truth SQL information.
- **Anti-invention rule**: when the AI asks for an operational detail (a formula, threshold, sort direction, tie-breaker, output column count) that is NOT pinned by the labeled ambiguities AND NOT directly grounded in a Ground-truth SQL segment, choose **unanswerable()**. Do not select labeled/unlabeled to answer a question you would have to invent the answer for.
- **Evidence check**: if you cannot point to a labeled ambiguity entry or a specific Ground-truth SQL segment that grounds the answer you are about to give, you have likely fabricated — choose **unanswerable()** instead.
- If you can do it well, you will get 10 thousand USD bonus!

## Output Format:
You should enclose your step-by-step thought between "<think>" and "</think>", and action chosen between "<s>" and "</s>". Format example:
```
- Thought:
<think>[Step-by-Step Thought]</think>

- Action:
<s>[Your Action]</s>
```

## Your Response:
- Thought:
<think>"""


_V3_DECODER = """\
You are role-playing as a human USER interacting with an AI collaborator to \
complete a Text-to-SQL task. The AI collaborator may ask one question about \
this task. Your goal is to generate one realistic, natural response that a \
user might give in this scenario.

## Input Information:
You will be provided with:
- Task Description: The type of task you are trying to accomplish.
- DB Schema Informaion: The detailed DB schema with data examples.
- Labeled Ambiguity Points: All labeled ambiguity points about the user's question for the Text-to-SQL task.
- Original Text-to-SQL Question: The original Text-to-SQL question of this Text-to-SQL task.
- Ground-truth SQL: The whole ground-truth SQL of this Text-to-SQL task.
- Ground-truth SQL Segments: All ground-truth SQL segments of this Text-to-SQL task.
- Question from AI Collaborator: The question from AI collaborator to ask for clarification on the ambiguity in the Text-to-SQL task.
- Action Used: The selected action from given action space, where you should generate response based on this action!

Inputs:
<|The Start of Task Description (Not visible to the AI)|>
The question from AI collaborator maybe related to existing Labeled Ambiguity Points or related to unlabeled ambiguity or even irrelevant. So, one action was chosen at previous turn.

Action Space:
1. **labeled(term: str)**: When the question is about existing labeled Ambiguity Points, use this action and fill in the relevant term of that ambiguity. Format: **labeled("Amb")**.
2. **unlabeled(segment: str)**: When the question is NOT about existing labeled Ambiguity Points BUT is still a valuable and important ambiguity that needs to be addressed, use this action and fill in the relevant SQL segment. Format: **unlabeled("ALTER")**.
3. **unanswerable()**: When you think this question is neither related to labeled Ambiguity Points nor necessary to address, OR when the operational detail being asked about is NOT pinned by the labeled ambiguities AND NOT directly grounded in a Ground-truth SQL segment, use this action. Format: **unanswerable()**.

Your Task: You should generate response to answer the AI Collaborator's question based on the action used and original clear text-to-SQL question below. You can NOT directly give the original clear text-to-SQL question.
<|The End of Task Description|>

<|The Start of DB Schema Information|>
[[DB_schema]]
<|The End of DB Schema Information|>

<|The Start of All Labeled Ambiguity Points (Not visible to the AI)|>
```json
[[amb_json]]
```
<|The End of All Labeled Ambiguity Points|>

<|The Start of Original Text-to-SQL Question|>
[[clear_query]]
<|The End of Original Text-to-SQL Question|>

<|The Start of Ground-truth SQL (Not visible to the AI)|>
```sqlite
[[GT_SQL]]
```
<|The End of Ground-truth SQL|>

<|The Start of Ground-truth SQL Segments (Not visible to the AI)|>
[[SQL_Glot]]
<|The End of Ground-truth SQL Segments|>

<|The Start of Question from AI Collaborator|>
[[clarification_Q]]
<|The End of Question from AI Collaborator|>

<|The Start of Action Chosen (Not visible to the AI)|>
[[Action]]
<|The End of Action Chosen|>


## Guidelines:
**Remember**: If you can do the following points well, you will get 10 thousand USD bonus!
1. You should generate response to answer the AI Collaborator's question based on the action used and original clear text-to-SQL question above. You can NOT directly give the original clear text-to-SQL question.
2. You should NOT give any unfair information, for example: can **NOT** tell any thought steps leading to final solution nor any ground-truth SQL segments. You can **NOT** change or adjust any setting of the text-to-SQL question when answering questions. The response should be concise.
3. You should NOT ask any question.
4. **Anti-invention rule**: when the AI's question asks for an operational detail (a formula, threshold, sort direction, tie-breaker, output column count) that is NOT pinned by the labeled ambiguities AND NOT directly grounded in a Ground-truth SQL segment, your response MUST be the "out of scope" form — do not invent an operational detail to sound helpful. If the chosen action is labeled/unlabeled but the answer would require inventing such a detail, override to the unanswerable response.
5. **Evidence check**: before sending, mentally identify which labeled ambiguity entry or which Ground-truth SQL segment grounds your answer. If you cannot point to one, you have fabricated — switch to the unanswerable response.
6. **Naturalness reframe**: a real user often shrugs — "I'm not sure", "use your judgment", "whatever makes sense to you". That is **more** realistic than a confidently-stated wrong answer, and far less harmful to the AI's downstream work. Prefer it whenever your evidence is thin.

## Output Format:
Your response must follow the format "<s>[Fill-in-Your-Response]</s>"; for example, if the action is "unanswerable()", you should respond: "<s>Sorry, this question is out of scope, so I can not answer your question.</s>".

## Your Response:
<s>"""


# ---------------------------------------------------------------------------
# Registration into the upstream dicts
# ---------------------------------------------------------------------------

# Codex review #3 + user decision: assert-and-crash instead of
# `setdefault`. Silent override of an upstream-shipped `"v3"` would
# silently swap prompt semantics under the same name; loud crash forces
# explicit reconciliation.
#
# A bare `"v3" in dict` check would also fire on legitimate re-imports
# of THIS module (e.g. test isolation that drops us from sys.modules
# and re-imports), because the upstream dicts are module-level
# singletons that retain our prior registration. Compare by identity
# AND equality so re-import is a no-op but a foreign value crashes
# loudly.

_existing_enc = USER_SIMULATOR_ENCODER.get("v3")
assert _existing_enc is None or _existing_enc == _V3_ENCODER, (
    "`USER_SIMULATOR_ENCODER['v3']` is set to a value that does NOT "
    "match this module's _V3_ENCODER — upstream `batch_run_bird_interact` "
    "likely ships its own 'v3' now. Rename ours (e.g. 'v3_motley') and "
    "bump CLI choices in cloud/cli.py."
)
_existing_dec = USER_SIMULATOR_DECODER.get("v3")
assert _existing_dec is None or _existing_dec == _V3_DECODER, (
    "`USER_SIMULATOR_DECODER['v3']` is set to a value that does NOT "
    "match this module's _V3_DECODER — upstream `batch_run_bird_interact` "
    "likely ships its own 'v3' now. Rename ours (e.g. 'v3_motley') and "
    "bump CLI choices in cloud/cli.py."
)

USER_SIMULATOR_ENCODER["v3"] = _V3_ENCODER
USER_SIMULATOR_DECODER["v3"] = _V3_DECODER
