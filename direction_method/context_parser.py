"""context_parser (SPEC.md §10): description -> (target_category, target_phrase, checklist).

Text-only call -- no image. Runs once per episode, at episode start.

Prompt text below is used verbatim -- do not paraphrase, shorten, or "clean up" the rule lists,
same policy as self_check.py and zone_gen.py.
"""

import json
from dataclasses import dataclass

from llm import LLMClient
from templates import CHECKLIST_KEYS

CONTEXT_PARSER_PROMPT = """Parse an object description into a target object and a checklist.

The description always describes one main target object, sometimes with other
objects around it.

=== OUTPUT FIELDS ===

target_category — the bare category noun phrase, no attributes.
                  Used to locate the object in other images where attributes differ.

target_phrase   — the target with attributes that belong to the target ITSELF.
                  Do NOT include clauses about other objects.
                  "navy blue cabinet under a white sink" -> "navy blue cabinet"

checklist       — assertions grouped by region key.

=== REGION KEYS ===

left, right, above, below,
left-top, right-top, left-bottom, right-bottom,
on, next to, Target

Relations are from the TARGET's point of view. Reverse the description's wording
when needed:
  "cabinet UNDER a countertop"  -> the countertop is ABOVE the cabinet -> "above"
  "bed BESIDE a nightstand"     -> no side given                       -> "next to"

Use "next to" only when the description gives no side.

=== ASSERTION STYLE ===

An assertion is a short phrase, NOT a sentence. It does not restate the region
or the target — those are added later from the region key.

  good:  "a white farmhouse sink"
  good:  "it is navy blue"                          (under Target)
  bad:   "There is a white sink above the cabinet."  (restates the region)
  bad:   "The cabinet is navy blue."                 (restates the target)

=== RULES ===

1. ATOMIC. One fact per assertion. Split compounds.
   "navy blue with brass handles" -> "it is navy blue", "it has brass handles"

2. PARTS vs OBJECTS. Parts integral to the target (handles, legs, doors, frame)
   go under "Target". Separate objects resting on it (blankets, items, plants)
   go under "on".

3. NO INVENTION. Add nothing the description does not state.

4. If the description names only the category, with no attributes and no other
   objects, the checklist is empty.

=== EXAMPLES ===

"Kitchen lower cabinet"
{"target_category": "kitchen lower cabinet",
 "target_phrase": "kitchen lower cabinet",
 "checklist": {}}

"Navy blue kitchen lower cabinet with brass handles"
{"target_category": "kitchen lower cabinet",
 "target_phrase": "navy blue kitchen lower cabinet",
 "checklist": {"Target": ["it is navy blue", "it has brass handles"]}}

"Kitchen lower cabinet situated beneath a white countertop"
{"target_category": "kitchen lower cabinet",
 "target_phrase": "kitchen lower cabinet",
 "checklist": {"above": ["a white countertop"]}}

"Navy blue kitchen lower cabinet under a white farmhouse sink"
{"target_category": "kitchen lower cabinet",
 "target_phrase": "navy blue kitchen lower cabinet",
 "checklist": {"Target": ["it is navy blue"],
               "above": ["a white farmhouse sink"]}}

"White bed with a blue blanket next to a nightstand"
{"target_category": "bed",
 "target_phrase": "white bed",
 "checklist": {"Target": ["it is white"],
               "on": ["a blue blanket"],
               "next to": ["a nightstand"]}}

"Green display cabinet next to open shelving"
{"target_category": "display cabinet",
 "target_phrase": "green display cabinet",
 "checklist": {"Target": ["it is green"],
               "next to": ["open shelving"]}}"""

# checklist keys are pinned to the 11-key enum (templates.CHECKLIST_KEYS) via additionalProperties:
# False + an explicit property per key -- there is no per-key "required", since only the region
# keys the description actually mentions should appear (rule 4: category-only -> checklist == {}).
#
# maxItems=8 on each key's array: confirmed against a live vllm==0.15.0 server that without it,
# the model can enter a token-repetition loop (the same assertion string appended dozens of
# times) that a strict json_schema grammar does nothing to stop -- an unbounded array is a valid
# completion at every step, so it runs until max_tokens truncates the response mid-string and it
# fails to parse entirely. No real description needs more than a handful of atomic facts per key.
_CHECKLIST_VALUE_SCHEMA = {"type": "array", "items": {"type": "string"}, "maxItems": 8}

CONTEXT_PARSER_SCHEMA = {
    "type": "object",
    "properties": {
        "target_category": {"type": "string", "description": "Bare category noun phrase, no attributes."},
        "target_phrase": {"type": "string", "description": "Target with its own attributes; no clauses about other objects."},
        "checklist": {
            "type": "object",
            "properties": {k: _CHECKLIST_VALUE_SCHEMA for k in CHECKLIST_KEYS},
            "additionalProperties": False,
        },
    },
    "required": ["target_category", "target_phrase", "checklist"],
}


class ContextParserError(Exception):
    """Raised when the response can't be turned into a usable ParsedContext -- malformed JSON or
    a missing required field."""


@dataclass
class ParsedContext:
    target_category: str  # bare noun phrase -- feeds zone_gen.locate (SPEC.md §5-1)
    target_phrase: str  # target + its own attributes -- feeds templates.region_for/question_for
    checklist: dict  # {region_key: [assertion, ...]}; only keys from templates.CHECKLIST_KEYS


def build_prompt(description: str) -> tuple:
    prompt = f"{CONTEXT_PARSER_PROMPT}\n\n{description}"
    return prompt, CONTEXT_PARSER_SCHEMA


def parse_context(llm_client: LLMClient, description: str) -> ParsedContext:
    prompt, schema = build_prompt(description)
    result = llm_client.call(prompt, image=None, response_schema=schema)
    try:
        parsed = json.loads(result.text)
        # Drop any empty-list entries: a real server that fills every schema property rather than
        # omitting unused ones would otherwise leave category-only descriptions with 11 empty
        # keys instead of {} (rule 4) -- normalize both shapes to the same thing here.
        checklist = {k: v for k, v in parsed["checklist"].items() if v}
        return ParsedContext(
            target_category=parsed["target_category"],
            target_phrase=parsed["target_phrase"],
            checklist=checklist,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ContextParserError(f"context_parser: malformed response for description={description!r}: {result.text!r}") from e
