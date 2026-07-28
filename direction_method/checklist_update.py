"""checklist_update (SPEC.md §12): the last of the four modules. current checklist + new
(relation, answer) pairs from one candidate image -> updated checklist.

Text-only call, no image. One call per image judgement -- every (relation, answer) pair gathered
for that candidate goes in a single call, not one call per pair.

Prompt text below is used verbatim -- do not paraphrase, shorten, or "clean up" the rule lists,
same policy as the other three modules.

The VLM only ever proposes ADDITIONS; the merge itself -- appending, deduping, and the
append-only invariant -- happens here in code, never inside the prompt.
"""

import json
import re

from llm import LLMClient
from templates import CHECKLIST_KEYS

CHECKLIST_UPDATE_PROMPT = """You maintain a checklist of facts about a scene.

You are given the current checklist and new question-answer pairs from an oracle
describing that scene. Extract NEW assertions from the answers.

Do not output the existing checklist. Output only what should be ADDED.

=== ASSERTION STYLE ===

An assertion is a short phrase, NOT a sentence. It does not restate the region or
the target — those are added later from the region key.

  good:  "open shelving with white dishes"
  good:  "it is navy blue"                       (under Target)
  bad:   "On the left there is open shelving."   (restates the region)
  bad:   "The cabinet is navy blue."             (restates the target)

=== RULES ===

1. ATOMIC. One fact per assertion. Split compounds.
   "a large wooden table with a plant on it"
   -> "a large wooden table", "a plant on the table"

2. KEEP THE ASKED KEY. File each assertion under the key the question asked
   about, even if the answer mentions a different position.

3. NO INVENTION. Record only what the answer states. Do not infer, elaborate, or
   fill in what is typical for such a room.

4. STRIP FRAMING. Remove "I can see", "In this image", "There is/are",
   "It appears that". Keep the content that follows.

5. KEEP HEDGES inside the assertion. "possibly a floor lamp" stays as is.

6. DROP non-visual content: opinions, atmosphere, style judgments, guesses about
   purpose.
   "a cozy reading nook"              -> drop
   "an armchair and a floor lamp"     -> keep

7. EMPTY REGIONS. If the answer says the region holds nothing, output the single
   assertion "nothing visible" for that key.

8. SKIP DUPLICATES. If the current checklist already records the same object for
   the same region, or for a region that overlaps it, do not add it again.
   Different wording for the same object still counts as a duplicate.
   Existing "next to: a nightstand", new answer for "left" says "a nightstand"
   -> skip, already recorded.

=== OUTPUT ===

{"additions": {"left": ["..."], "on": ["..."]}}

Include only keys that have new assertions. If nothing should be added, return
{"additions": {}}."""

# additions' keys are pinned to the same 11-key enum as context_parser's checklist (§10) --
# additionalProperties:False plus an explicit property per key, no per-key "required" since only
# keys with genuinely new assertions should appear.
CHECKLIST_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "additions": {
            "type": "object",
            "properties": {k: {"type": "array", "items": {"type": "string"}} for k in CHECKLIST_KEYS},
            "additionalProperties": False,
        },
    },
    "required": ["additions"],
}


class ChecklistUpdateError(Exception):
    """Raised when the response can't be turned into usable additions -- malformed JSON or a
    missing required field. Also raised if a merge ever violates the append-only invariant
    (should be unreachable given the merge code below; kept as a runtime guard, not a real
    failure mode)."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def build_prompt(checklist: dict, round_answers: list) -> tuple:
    checklist_lines = "\n".join(
        f"{key}: {assertion}" for key, assertions in checklist.items() for assertion in assertions
    )
    answer_lines = "\n".join(f"{relation}: {answer}" for relation, answer in round_answers)
    variable_part = f"current checklist:\n{checklist_lines}\n\nnew answers:\n{answer_lines}"
    prompt = f"{CHECKLIST_UPDATE_PROMPT}\n\n{variable_part}"
    return prompt, CHECKLIST_UPDATE_SCHEMA


def _parse_additions(text: str) -> dict:
    try:
        parsed = json.loads(text)
        additions = parsed["additions"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ChecklistUpdateError(f"checklist_update: malformed response: {text!r}") from e
    # Drop empty-list entries, same normalization as context_parser (§10) -- a server that fills
    # every schema property rather than omitting unused ones must still behave like {}.
    return {k: v for k, v in additions.items() if v}


def merge_checklist(checklist: dict, additions: dict) -> dict:
    """Merge in code, never in the VLM. Returns a NEW dict -- `checklist` itself is never
    mutated, so the caller keeps a clean pre-merge reference for the superset check below.

    - append `additions[key]` to the existing list under `key`
    - never modify, reword, reorder, or delete an existing assertion
    - safety-net exact-match dedup after normalizing case/whitespace (the prompt's rule 8 is the
      primary defence; this only catches the literal repeats it misses) -- checked against both
      the pre-existing assertions AND assertions already added earlier in this same merge
    """
    merged = {key: list(assertions) for key, assertions in checklist.items()}

    for key, new_assertions in additions.items():
        existing = merged.setdefault(key, [])
        seen_normalized = {_normalize(a) for a in existing}
        for assertion in new_assertions:
            normalized = _normalize(assertion)
            if normalized in seen_normalized:
                continue
            existing.append(assertion)
            seen_normalized.add(normalized)

    _assert_superset(checklist, merged)
    return merged


def _assert_superset(pre: dict, post: dict) -> None:
    # Append-only implies every pre-existing assertion list survives as an exact, unreordered
    # prefix of the post-merge list for that key.
    for key, assertions in pre.items():
        post_assertions = post.get(key, [])
        if post_assertions[: len(assertions)] != assertions:
            raise ChecklistUpdateError(f"merge violated the append-only invariant for key {key!r}")


def checklist_update(llm_client: LLMClient, checklist: dict, round_answers: list) -> dict:
    if not round_answers:
        return checklist  # nothing new was asked this round -- no VLM call needed
    prompt, schema = build_prompt(checklist, round_answers)
    result = llm_client.call(prompt, image=None, response_schema=schema)
    additions = _parse_additions(result.text)
    return merge_checklist(checklist, additions)
